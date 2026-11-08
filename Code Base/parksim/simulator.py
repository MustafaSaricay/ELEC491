"""Closed-loop simulation: true world + plant + sensors + Wi-Fi link + app,
driving the onboard Autonomy stack. Ground truth is used only for scoring."""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import geometry as g
from . import layouts
from .autonomy import Autonomy, RefMap
from .comms import App, Link
from .config import Config, get_dotted, set_dotted
from .sensors import BayCamera, DelayLine, Encoder, Imu, Lidar, UltrasonicArray
from .vehicle import Vehicle
from .world import World

STATE_CODES = {"IDLE": 0, "DRIVING": 1, "BLOCKED": 2, "PAUSED": 3, "ESTOP": 4, "FAULT": 5}


class _Clock:
    def __init__(self, rate_hz, t0=0.0):
        self.period = 1.0 / rate_hz
        self.next = t0

    def due(self, t):
        if t + 1e-9 >= self.next:
            self.next += self.period
            if self.next < t:
                self.next = t + self.period
            return True
        return False


@dataclass
class SimResult:
    cfg: Config
    log: dict
    tasks: list
    events: list
    paths: list
    scans: list
    grids: list
    obstacles: list
    metrics: dict
    plan_infos: list = field(default_factory=list)
    pings: list = field(default_factory=list)      # (t_arrival, sensor, range)


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg = cfg.copy()
        ss = np.random.SeedSequence(cfg.sim.seed)
        r = [np.random.default_rng(s) for s in ss.spawn(11)]
        # obstacle layout (fixed list or generated per seed) and when it appears
        obstacles = layouts.generate(cfg, r[10])
        if cfg.world.obstacle_timing not in ("before_park", "after_park"):
            raise ValueError("world.obstacle_timing must be 'before_park' or 'after_park'")
        deferred = obstacles if cfg.world.obstacle_timing == "after_park" else []
        cfg.world.obstacles = [] if deferred else obstacles
        self.world = World(cfg.world)
        off = cfg.vehicle.center_offset
        start = g.center_to_rear(g.deg_pose(cfg.world.start), off)
        self.vehicle = Vehicle(cfg.vehicle, cfg.steering, cfg.drive, cfg.traction, r[0], start)
        self.encoder = Encoder(cfg.encoder, r[1])
        self.imu = Imu(cfg.imu, r[2])
        self.lidar = Lidar(cfg.lidar, r[3])
        self.camera = BayCamera(cfg.camera, r[4], cfg.vehicle)
        self.ultrasonic = UltrasonicArray(cfg.ultrasonic, r[9])
        self.uplink = Link(cfg.comm, r[5])
        self.downlink = Link(cfg.comm, r[6])
        self.rng = r[7]
        self.app = App(cfg.mission, cfg.comm)

        refmap = RefMap(cfg.world.width, cfg.world.height, self.world.walls.copy(),
                        self.world.bay_pose, cfg.world.bay["depth"])
        e = cfg.estimator
        belief = start + r[8].standard_normal(3) * np.array(
            [e.init_sigma_xy, e.init_sigma_xy, math.radians(e.init_sigma_yaw_deg)])
        self.autonomy = Autonomy(cfg, refmap, belief)

        self.sensor_q = DelayLine()
        self.act_q = DelayLine()
        self.pose_hist = deque(maxlen=int(0.5 / cfg.sim.dt) + 5)
        # events with "after_task": k wait until task k finishes, then fire `delay` s later
        evs = [dict(ev) for ev in cfg.mission.events]
        park_id = next((i for i, c in enumerate(cfg.mission.commands) if c.get("cmd") == "park"), 0)
        evs += [{"type": "spawn_obstacle", "after_task": park_id, "delay": cfg.world.after_park_delay,
                 "obstacle": o} for o in deferred]
        self.task_events = [ev for ev in evs if "after_task" in ev]
        self.events = sorted([ev for ev in evs if "after_task" not in ev], key=lambda ev: ev.get("t", 0))
        self.left_testbed = False
        self.collided = False
        self.collision_t = None
        self.min_clearance = math.inf
        self.task_eval = []
        self._task_seen = 0
        self._task_clear = math.inf
        self._task_loc_err = 0.0
        self.t = 0.0
        self.elog = None
        if cfg.sim.log_dir:
            from .eventlog import EventLog
            path = os.path.join(cfg.sim.log_dir, f"{cfg.sim.log_name or 'run'}_s{cfg.sim.seed:04d}.jsonl")
            self.elog = EventLog(self, path)
            self.elog.header(cfg.sim.log_name, cfg.to_dict())
            car_log = self.autonomy.log

            def logged(t, text):           # every onboard event also goes to the file, with truth
                car_log(t, text)
                self.elog.event(t, text)
            self.autonomy.log = logged

    # ------------------------------------------------------------------ #
    def _pose_at(self, times):
        h = np.array(self.pose_hist)
        if len(h) < 2:
            return np.tile(self.vehicle.pose, (len(times), 1))
        yaw = np.unwrap(h[:, 3])
        return np.stack([np.interp(times, h[:, 0], h[:, 1]), np.interp(times, h[:, 0], h[:, 2]),
                         np.interp(times, h[:, 0], yaw)], 1)

    def _process_events(self, t):
        while self.events and self.events[0].get("t", 0) <= t:
            ev = self.events.pop(0)
            kind = ev["type"]
            if kind == "spawn_obstacle":
                ob = dict(ev["obstacle"])
                if ev.get("relative_to_car"):
                    # obstacle x, y, yaw given in the car's rear-axle frame at spawn time
                    x, y, th = self.vehicle.pose
                    c, s_ = math.cos(th), math.sin(th)
                    ob["x"], ob["y"] = x + ob["x"] * c - ob["y"] * s_, y + ob["x"] * s_ + ob["y"] * c
                    ob["yaw_deg"] = ob.get("yaw_deg", 0.0) + math.degrees(th)
                self.world.add_obstacle(ob, t)
                self.autonomy.log(t, "[sim] obstacle spawned")
            elif kind == "kidnap":
                self.vehicle.teleport(ev.get("dx", 0.0), ev.get("dy", 0.0),
                                      math.radians(ev.get("dyaw_deg", 0.0)))
                self.autonomy.log(t, "[sim] vehicle kidnapped (bumped)")
            elif kind == "estop":
                self.app.inject({"type": "estop"})
            elif kind == "set":
                old = get_dotted(self.cfg, ev["key"])
                set_dotted(self.cfg, ev["key"], ev["value"])
                self.autonomy.log(t, f"[sim] {ev['key']} = {ev['value']}")
                if "until" in ev:
                    self.events.append({"t": ev["until"], "type": "set", "key": ev["key"], "value": old})
                    self.events.sort(key=lambda e: e.get("t", 0))
            else:
                raise ValueError(f"unknown event type {kind}")

    def _evaluate_tasks(self, t):
        """Score finished tasks against ground truth."""
        log = self.autonomy.task_log
        while self._task_seen < len(log):
            task = log[self._task_seen]
            self._task_seen += 1
            gc = g.rear_to_center(task["goal"], self.cfg.vehicle.center_offset)
            pc = self.vehicle.center_pose
            c, s = math.cos(gc[2]), math.sin(gc[2])
            d = pc[:2] - gc[:2]
            longi, lat = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
            head = g.wrap(pc[2] - gc[2])
            m = self.cfg.mission
            ok = (task["result"] == "done" and not self.collided and not self.left_testbed
                  and abs(longi) <= m.success_pos_tol
                  and abs(lat) <= m.success_pos_tol and abs(head) <= math.radians(m.success_yaw_tol_deg))
            est = np.array(task["est_pose"])
            rec = {
                "id": task["id"], "kind": task["kind"], "result": task["result"], "success": bool(ok),
                "time": task["t_end"] - task["t_start"], "t_end": task["t_end"],
                "err_long": float(longi), "err_lat": float(lat), "err_heading_deg": math.degrees(head),
                "err_pos": float(math.hypot(longi, lat)),
                "loc_err_end": float(math.hypot(*(est[:2] - self.vehicle.pose[:2]))),
                "loc_err_max": self._task_loc_err,
                "min_clearance": self._task_clear, "collided": self.collided,
                "left_testbed": self.left_testbed,
                "replans": task["replans"], "corrections": task["corrections"],
                "blocked": task["blocked"], "paused": task["paused"],
            }
            self.task_eval.append(rec)
            if self.elog:
                self.elog.task(rec)
            self._task_clear = math.inf
            self._task_loc_err = 0.0
            for ev in [e for e in self.task_events if e["after_task"] == task["id"]]:
                self.task_events.remove(ev)
                ev["t"] = task["t_end"] + ev.get("delay", 0.0)
                self.events.append(ev)
            self.events.sort(key=lambda e: e.get("t", 0))

    # ------------------------------------------------------------------ #
    def run(self, progress=False) -> SimResult:
        cfg = self.cfg
        dt = cfg.sim.dt
        clk = {k: _Clock(r) for k, r in [("enc", cfg.encoder.rate_hz), ("imu", cfg.imu.rate_hz),
                                         ("lidar", cfg.lidar.rate_hz), ("cam", cfg.camera.rate_hz),
                                         ("ctl", cfg.control.rate_hz), ("app", 50.0),
                                         ("tel", cfg.comm.telemetry_hz), ("log", cfg.sim.log_hz),
                                         ("grid", 2.0), ("us", 1.0 / cfg.ultrasonic.ping_interval),
                                         ("elog", max(cfg.sim.log_state_hz, 1e-3))]}
        clk["lidar"].next = 1.0 / cfg.lidar.rate_hz
        L = {k: [] for k in ["t", "x", "y", "yaw", "ex", "ey", "eyaw", "sx", "syaw", "v", "v_ref",
                             "v_est", "delta", "delta_cmd", "duty", "state", "link", "hb_ok",
                             "wl", "wr", "loc_err"]}
        scans, grids, pings = [], [], []
        aut, veh = self.autonomy, self.vehicle
        last_cmd = (0.0, 0.0)
        n_steps = int(cfg.mission.max_time / dt)
        idle_since = None
        for k in range(n_steps):
            t = self.t = k * dt
            self._process_events(t)
            for cmd in self.act_q.pop_ready(t):
                veh.set_command(*cmd)
            veh.step(dt)
            self.pose_hist.append((t + dt, veh.x, veh.y, veh.yaw))
            t = self.t = (k + 1) * dt

            # ---- sensors -> delay lines -> software ------------------
            if clk["imu"].due(t):
                self.sensor_q.push(t + cfg.imu.latency, self.imu.sample(t, veh.omega))
            if clk["enc"].due(t):
                self.sensor_q.push(t + cfg.encoder.latency, self.encoder.sample(t, veh.wheel_dist))
            if cfg.lidar.enabled and clk["lidar"].due(t):
                m = self.lidar.scan(t, self._pose_at, self.world)
                self.sensor_q.push(t + cfg.lidar.latency, m)
            if cfg.ultrasonic.enabled and clk["us"].due(t):
                for delay, m in self.ultrasonic.fire(t, veh.pose, self.world):
                    self.sensor_q.push(t + delay, m)
            if cfg.camera.enabled and clk["cam"].due(t):
                m = self.camera.observe(t, veh.pose, self.world.bay_pose, cfg.world.bay["depth"])
                if m is not None:
                    self.sensor_q.push(t + cfg.camera.latency, m)
            for m in self.sensor_q.pop_ready(t):
                kind = m["kind"]
                if kind == "imu":
                    aut.on_imu(m)
                elif kind == "encoder":
                    aut.on_encoder(m)
                elif kind == "lidar":
                    aut.on_lidar(m)
                    scans.append((t, m["t"], aut.ekf.pose_at(m["t"]), m["angles"], m["ranges"]))
                elif kind == "camera":
                    aut.on_camera(m)
                elif kind == "ultrasonic":
                    aut.on_ultrasonic(m)
                    pings.append((t, m["name"], m["range"]))

            # ---- comms ---------------------------------------------------
            for msg in self.uplink.receive(t):
                aut.on_msg(t, msg)
            if clk["tel"].due(t):
                self.downlink.send(t, aut.telemetry(t))
            if clk["app"].due(t):
                for msg in self.app.step(t, self.downlink.receive(t)):
                    self.uplink.send(t, msg)

            # ---- control ------------------------------------------------
            if clk["ctl"].due(t):
                if cfg.control.skip_prob > 0 and self.rng.random() < cfg.control.skip_prob:
                    pass                      # missed cycle: actuators keep last command
                else:
                    last_cmd = aut.control_step(t, 1.0 / cfg.control.rate_hz)
                    self.act_q.push(t + cfg.control.actuation_latency, last_cmd)
                fp = veh.footprint()
                clear = self.world.footprint_clearance(fp)
                self.min_clearance = min(self.min_clearance, clear)
                self._task_clear = min(self._task_clear, clear)
                self._task_loc_err = max(self._task_loc_err, float(np.hypot(*(aut.pose[:2] - veh.pose[:2]))))
                if clear <= 0.0 and not self.collided:
                    self.collided = True
                    self.collision_t = t
                    aut.log(t, "[sim] COLLISION")
                    if cfg.mission.stop_on_collision:
                        veh.freeze()
                if not self.left_testbed and self.world.outside_testbed(fp):
                    self.left_testbed = True        # crossed an edge that has no wall
                    self.collision_t = t
                    aut.log(t, "[sim] LEFT THE TESTBED")
                    if cfg.mission.stop_on_collision:
                        veh.freeze()
            self._evaluate_tasks(t)

            # ---- log ------------------------------------------------------
            if clk["log"].due(t):
                ep, P = aut.pose, aut.ekf.P
                wl, wr = veh.wheel_angles()
                for key, val in (("t", t), ("x", veh.x), ("y", veh.y), ("yaw", veh.yaw),
                                 ("ex", ep[0]), ("ey", ep[1]), ("eyaw", ep[2]),
                                 ("sx", math.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 0))),
                                 ("syaw", math.sqrt(P[2, 2])), ("v", veh.v), ("v_ref", aut.v_ref),
                                 ("v_est", aut.v_est), ("delta", veh.delta), ("delta_cmd", last_cmd[0]),
                                 ("duty", last_cmd[1]), ("state", STATE_CODES[aut.state]),
                                 ("link", self.uplink.is_up(t)),
                                 ("hb_ok", (t - aut.last_hb) <= cfg.comm.heartbeat_timeout),
                                 ("wl", wl), ("wr", wr),
                                 ("loc_err", math.hypot(ep[0] - veh.x, ep[1] - veh.y))):
                    L[key].append(val)
            if self.elog and clk["elog"].due(t):
                self.elog.state(t)
            if clk["grid"].due(t):
                grids.append((t, aut.grid.probability().astype(np.float32)))

            if progress and k % int(5.0 / dt) == 0:
                print(f"  t={t:6.1f}s state={aut.state}")
            # ---- termination ---------------------------------------------
            if self.app.finished and aut.state in ("IDLE", "FAULT", "ESTOP") and aut.task is None:
                idle_since = t if idle_since is None else idle_since
                if t - idle_since > 1.0:
                    break
            else:
                idle_since = None
            if (self.collided or self.left_testbed) and cfg.mission.stop_on_collision \
                    and t - self.collision_t > 1.0:
                aut._finish(t, "collision" if self.collided else "left_testbed")
                self._evaluate_tasks(t)
                break
        if aut.task is not None:
            aut._finish(self.t, "timeout")
            self._evaluate_tasks(self.t)

        log = {k: np.asarray(v) for k, v in L.items()}
        metrics = self._metrics(log)
        if self.elog:
            self.elog.close(metrics)
        obstacles = [o for o in self.world.obstacles]
        return SimResult(cfg, log, self.task_eval, aut.events, aut.paths, scans, grids,
                         obstacles, metrics, aut.plan_infos, pings)

    def _metrics(self, log):
        cfg = self.cfg
        out = {"sim_time": float(log["t"][-1]) if len(log["t"]) else 0.0,
               "collided": self.collided, "left_testbed": self.left_testbed,
               "min_clearance": float(self.min_clearance),
               "loc_rmse": float(np.sqrt(np.mean(log["loc_err"] ** 2))) if len(log["t"]) else 0.0,
               "loc_err_max": float(log["loc_err"].max()) if len(log["t"]) else 0.0,
               "tasks": self.task_eval,
               "all_success": bool(self.task_eval) and all(t["success"] for t in self.task_eval)
               and len(self.task_eval) >= len(cfg.mission.commands),
               "comm_latency_mean": float(np.mean(self.uplink.latencies)) if self.uplink.latencies else None,
               "comm_latency_max": float(np.max(self.uplink.latencies)) if self.uplink.latencies else None,
               "ekf_stale_updates": self.autonomy.ekf.stale_updates,
               "ekf_rejected": dict(self.autonomy.ekf.rejected),
               "ekf_accepted": dict(self.autonomy.ekf.accepted)}
        # failsafe response for each Wi-Fi outage
        fs = []
        t, v = log["t"], log["v"]
        x, y = log["x"], log["y"]
        for t0, t1 in cfg.comm.outages:
            i0 = int(np.searchsorted(t, t0))
            if i0 >= len(t):
                continue
            moving = abs(v[i0]) > 0.02
            stop_idx = next((i for i in range(i0, len(t)) if abs(v[i]) < 0.01), None)
            rec = {"outage": [t0, t1], "moving_at_loss": bool(moving)}
            if moving and stop_idx is not None:
                rec["stop_time"] = float(t[stop_idx] - t0)
                rec["stop_distance"] = float(np.sum(np.hypot(np.diff(x[i0:stop_idx + 1]), np.diff(y[i0:stop_idx + 1]))))
            fs.append(rec)
        out["failsafe"] = fs
        return out


def run(cfg: Config, progress=False) -> SimResult:
    return Simulation(cfg).run(progress=progress)
