"""Wi-Fi / WebSocket link and the mobile-app side of the mission.

The link behaves like TCP: messages arrive in order, a "lost" packet is not
dropped but delayed by a retransmission, and anything sent or in flight while
the link is down (an outage) is lost.
"""
from __future__ import annotations

from .config import CommCfg, MissionCfg
from .sensors import DelayLine


class Link:
    def __init__(self, cfg: CommCfg, rng):
        self.cfg, self.rng = cfg, rng
        self._dl = DelayLine()
        self._last_ready = 0.0
        self.sent = self.delivered = 0
        self.latencies = []

    def is_up(self, t):
        return not any(t0 <= t < t1 for t0, t1 in self.cfg.outages)

    def send(self, t, msg):
        self.sent += 1
        if not self.is_up(t):
            return False
        c = self.cfg
        lat = c.latency + (self.rng.exponential(c.jitter) if c.jitter > 0 else 0.0)
        if c.loss_prob > 0 and self.rng.random() < c.loss_prob:
            lat += c.retransmit_delay
        t_ready = max(t + lat, self._last_ready)      # in-order delivery
        self._last_ready = t_ready
        self._dl.push(t_ready, (t, msg))
        return True

    def receive(self, t):
        out = []
        for t_sent, msg in self._dl.pop_ready(t):
            if self.is_up(t):
                out.append(msg)
                self.delivered += 1
                self.latencies.append(t - t_sent)
        return out


class App:
    """Plays the mission: sends heartbeats and commands, waits for completion."""

    def __init__(self, mission: MissionCfg, comm: CommCfg):
        self.mission, self.comm = mission, comm
        self.idx = 0
        self.pending = None
        self.next_issue_t = None
        self.last_send_t = -1e9
        self.next_hb_t = 0.0
        self.last_ack = -1
        self.last_done = -1
        self.vehicle_state = "?"
        self.injected = []

    @property
    def finished(self):
        return self.idx >= len(self.mission.commands) and self.pending is None

    def inject(self, msg):
        self.injected.append(msg)

    def step(self, t, telemetry):
        for tm in telemetry:
            self.last_ack = max(self.last_ack, tm["ack_id"])
            self.last_done = max(self.last_done, tm["done_id"])
            self.vehicle_state = tm["state"]
        out = list(self.injected)
        self.injected.clear()
        if t >= self.next_hb_t:
            out.append({"type": "heartbeat", "t": t})
            self.next_hb_t = t + 1.0 / self.comm.heartbeat_hz

        cmds = self.mission.commands
        if self.pending is None and self.idx < len(cmds):
            if self.idx == 0 or self.last_done >= self.idx - 1:
                if self.next_issue_t is None:
                    self.next_issue_t = t + cmds[self.idx].get("wait", 0.0)
                if t >= self.next_issue_t:
                    self.pending = {"type": "command", "id": self.idx, **cmds[self.idx]}
                    out.append(self.pending)
                    self.last_send_t = t
        elif self.pending is not None:
            pid = self.pending["id"]
            if self.last_done >= pid:
                self.pending = None
                self.idx += 1
                self.next_issue_t = None
            elif self.last_ack < pid and t - self.last_send_t > 0.5:
                out.append(self.pending)          # retry until acknowledged
                self.last_send_t = t
        return out
