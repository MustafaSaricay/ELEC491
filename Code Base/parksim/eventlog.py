"""Optional per-run event log (JSON Lines), enabled with `sim.log_dir`
(CLI: --log-dir DIR). Simulator side: it may read ground truth, and it
annotates every onboard event with what was really going on at that moment.

One file per run, one JSON object per line:
  {"type": "run", ...}      name, seed, start time, the full resolved config
  {"type": "event", ...}    every onboard / simulator event: t, kind, msg + truth context
  {"type": "state", ...}    periodic snapshot (sim.log_state_hz): state, true/est pose, errors
  {"type": "task", ...}     each finished task, scored against ground truth
  {"type": "metrics", ...}  end-of-run metrics

Event kinds (from the message text, see KINDS): command, plan, plan_fail,
stop_echo, stop_wall, stop_lidar, stop_map, link_lost, link_back, loc_lost,
loc_inflate, relocalize, correction, task_end, estop, collision, left_testbed,
spawn, kidnap, set, other.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np

KINDS = [
    (r"\[sim\] COLLISION", "collision"),
    (r"\[sim\] LEFT THE TESTBED", "left_testbed"),
    (r"\[sim\] obstacle spawned", "spawn"),
    (r"\[sim\] vehicle kidnapped", "kidnap"),
    (r"^\[sim\] .* = ", "set"),
    (r"^command #", "command"),
    (r"^path ", "plan"),
    (r"^planning failed", "plan_fail"),
    (r"^ultrasonic .*\(wall\) in stop zone", "stop_wall"),
    (r"^ultrasonic .* in stop zone", "stop_echo"),
    (r"^LiDAR points in stop zone", "stop_lidar"),
    (r"^map shows obstacle on path", "stop_map"),
    (r"^heartbeat lost", "link_lost"),
    (r"^heartbeat back", "link_back"),
    (r"^localization lost", "loc_lost"),
    (r"disagree with the map", "loc_inflate"),
    (r"^re-localized", "relocalize"),
    (r"^final error", "correction"),
    (r"^task #", "task_end"),
    (r"E-STOP", "estop"),
]
_RX = [(re.compile(p), k) for p, k in KINDS]


def classify(msg: str) -> str:
    for rx, kind in _RX:
        if rx.search(msg):
            return kind
    return "other"


def _num(v, nd=4):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_num(x, nd) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _num(x, nd) for k, x in v.items()}
    return v


class EventLog:
    def __init__(self, sim, path):
        self.sim = sim
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.f = open(path, "w")
        self.counts: dict[str, int] = {}

    def write(self, rec):
        self.f.write(json.dumps(_num(rec), separators=(",", ":")) + "\n")

    # ---- truth context ---------------------------------------------------
    def free_distance(self, forward=True):
        """True free distance straight ahead of (or behind) the bumper, over the
        car's width, against EVERY obstacle and wall (whatever the sensors see)."""
        sim = self.sim
        veh, vc = sim.vehicle, sim.cfg.vehicle
        x, y, th = veh.pose
        bump = (vc.length - vc.rear_overhang) if forward else -vc.rear_overhang
        ang = th if forward else th + math.pi
        lat = np.linspace(-vc.width / 2, vc.width / 2, 5)
        ox = x + bump * math.cos(th) - lat * math.sin(th)
        oy = y + bump * math.sin(th) + lat * math.cos(th)
        r = sim.world.raycast(ox, oy, np.full(5, ang), -1.0)
        return float(np.min(r))

    def context(self):
        sim = self.sim
        veh, aut = sim.vehicle, sim.autonomy
        est = aut.pose
        return {"state": aut.state, "true": [veh.x, veh.y, veh.yaw], "est": est.tolist(),
                "loc_err": math.hypot(est[0] - veh.x, est[1] - veh.y),
                "sigma": aut.ekf.sigma_xy(), "v": veh.v,
                "clearance": sim.world.footprint_clearance(veh.footprint()),
                "free_front": self.free_distance(True), "free_rear": self.free_distance(False)}

    # ---- records ---------------------------------------------------------
    def header(self, name, cfg_dict):
        self.write({"type": "run", "name": name, "seed": self.sim.cfg.sim.seed,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg_dict})

    def event(self, t, msg):
        kind = classify(msg)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.write({"type": "event", "t": t, "kind": kind, "msg": msg, **self.context()})

    def state(self, t):
        self.write({"type": "state", "t": t, **self.context()})

    def task(self, rec):
        self.write({"type": "task", **rec})

    def close(self, metrics):
        self.write({"type": "metrics", "event_counts": self.counts,
                    **{k: v for k, v in metrics.items() if k != "tasks"}})
        self.f.close()
