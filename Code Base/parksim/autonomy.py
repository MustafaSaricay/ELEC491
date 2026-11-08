"""Onboard software stack (what will run on the Raspberry Pi 5).

It only ever sees sensor messages, app messages and its reference map - never
the simulator's ground truth - so it can be lifted onto the real car.

States
  IDLE      waiting for a command
  DRIVING   following a planned path (also covers the planning compute delay)
  BLOCKED   stopped because the path/bubble is obstructed; re-plans periodically
  PAUSED    stopped because the app heartbeat timed out (Wi-Fi failsafe)
  ESTOP     latched emergency stop from the app (cleared by a new command)
  FAULT     localization lost / no path - stopped, waiting for a new command
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import geometry as g
from .config import Config
from .control import PathTracker, SpeedController
from .estimation import EKF, ScanMatcher
from .mapping import OccupancyGrid
from .planning.collision import EscapeChecker, FootprintChecker
from .planning.hybrid_astar import HybridAStar


@dataclass
class RefMap:
    """What the car knows in advance: testbed size, wall segments, bay pose."""
    width: float
    height: float
    walls: np.ndarray
    bay_pose: np.ndarray        # bay centre pose (vehicle-centre convention)
    bay_depth: float


class Autonomy:
    def __init__(self, cfg: Config, refmap: RefMap, init_pose, t0=0.0):
        self.cfg = cfg
        self.map = refmap
        self.off = cfg.vehicle.center_offset
        self.ekf = EKF(cfg.estimator, init_pose, t0)
        self.icp = ScanMatcher(refmap.walls, cfg.estimator) if len(refmap.walls) else None
        self.grid = OccupancyGrid(refmap.width, refmap.height, cfg.mapping, refmap.walls)
        self.planner = HybridAStar(cfg.vehicle, cfg.planner)
        self.speed = SpeedController(cfg.control)
        self.tracker: PathTracker | None = None

        self.state = "IDLE"
        self.task = None
        self.ack_id = -1
        self.done_id = -1
        self.last_hb = -math.inf
        self.v_est = 0.0
        self.delta_cmd = 0.0
        self.v_ref = 0.0
        self._gyro_sum = 0.0
        self._gyro_n = 0
        self._last_wz = 0.0
        self._scan_pts = None          # latest scan, vehicle frame at scan time
        self._scan_pose = None
        self._check_path = False
        self.lidar_rejects = 0
        self.relocalizations = 0
        self._bubble_scan_id = None
        self._bubble_streak = 0
        self.us_sensors = {s["name"]: s for s in cfg.ultrasonic.sensors}
        self.us_hist: dict[str, deque] = {}
        self.us_long_rejects: dict[str, int] = {}
        self.inflations = 0
        self._last_path_check = -math.inf
        self.plan_ready_t = 0.0
        self.block_t = 0.0
        self.events: list[tuple[float, str]] = []
        self.task_log: list[dict] = []
        self.paths: list[tuple[float, object]] = []
        self.plan_infos: list[dict] = []

    # ------------------------------------------------------------------ #
    def log(self, t, text):
        self.events.append((t, text))

    @property
    def pose(self):
        return self.ekf.pose

    def _goal_center(self):
        return g.rear_to_center(self.task["goal"], self.off)

    # ---- sensor callbacks -------------------------------------------- #
    def on_imu(self, m):
        self._gyro_sum += m["wz"]
        self._gyro_n += 1
        self._last_wz = m["wz"]

    def on_encoder(self, m):
        dt = m["dt"]
        if dt <= 0:
            return
        wz = self._gyro_sum / self._gyro_n if self._gyro_n else self._last_wz
        self._gyro_sum, self._gyro_n = 0.0, 0
        self.ekf.predict(m["t"], m["ds"], wz * dt, dt)
        a = self.cfg.control.speed_lpf
        self.v_est = (1 - a) * self.v_est + a * (m["ds"] / dt)

    def on_lidar(self, m):
        lc = self.cfg.lidar
        r = m["ranges"]
        ok = np.isfinite(r)
        ang, rng = m["angles"][ok], r[ok]
        pts = np.stack([lc.mount_x + rng * np.cos(ang), rng * np.sin(ang)], 1)
        guess = self.ekf.pose_at(m["t"])
        if self.cfg.estimator.use_lidar and self.icp is not None:    # ICP needs walls
            res = self.icp.match(pts, guess)
            accepted = False
            if res is not None:
                z, cov, quality = res
                accepted = self.ekf.update_pose(m["t"], z, cov)
                ec = self.cfg.estimator
                if not accepted and self.lidar_rejects + 1 >= ec.reloc_after and quality >= ec.reloc_min_quality:
                    self.ekf.relocalize(m["t"], z, cov)
                    self.relocalizations += 1
                    self.log(m["t"], f"re-localized from scan match (inliers {quality*100:.0f}%)")
                    accepted = True
            self.lidar_rejects = 0 if accepted else self.lidar_rejects + 1
        pose = self.ekf.pose_at(m["t"])
        spose = (pose[0] + lc.mount_x * math.cos(pose[2]), pose[1] + lc.mount_x * math.sin(pose[2]), pose[2])
        self.grid.update_scan(spose, m["angles"], m["ranges"])
        self._scan_pts, self._scan_pose = pts, pose
        self._check_path = True

    def on_ultrasonic(self, m):
        s = self.us_sensors.get(m["name"])
        if s is None:
            return
        uc = self.cfg.ultrasonic
        r = m["range"]
        ec = self.cfg.estimator
        if math.isfinite(r) and ec.use_ultrasonic:
            ok = self.ekf.update_range(m["t"], r, s, self.map.walls, uc)
            name = m["name"]
            if ok:
                self.us_long_rejects[name] = 0
            else:
                # Obstacles only shorten echoes. An echo LONGER than the known wall
                # allows means our pose is wrong - don't fail silently. Counted per
                # sensor: other sensors may still agree along a different axis.
                h = self.ekf.expected_range(self.ekf.pose_at(m["t"]), s, self.map.walls, uc)
                if math.isfinite(h) and r > h + ec.us_long_tol:
                    self.us_long_rejects[name] = self.us_long_rejects.get(name, 0) + 1
                    if self.us_long_rejects[name] >= ec.us_diverge_count:
                        self.ekf.inflate(ec.us_inflate_xy, ec.us_inflate_yaw_deg)
                        self.us_long_rejects = {}
                        self.inflations += 1
                        self.log(m["t"], "ultrasonic ranges disagree with the map -> widening uncertainty")
        pose = self.ekf.pose_at(m["t"])
        spose = g.mount_pose(pose, s)
        half = math.radians(uc.half_angle_deg)
        mark_hit = True
        if math.isfinite(r):
            # an echo at the range the known walls predict is the wall, not an obstacle
            r_wall = g.cone_echo_walls(*spose, half, self.map.walls, math.radians(uc.max_incidence_deg))
            mark_hit = not (math.isfinite(r_wall) and r >= r_wall - self.cfg.mapping.us_wall_tol)
        self.grid.update_cone(spose, half, r, mark_hit)
        pts = None
        if math.isfinite(r):
            a = spose[2] + np.array([-half, 0.0, half])
            pts = np.stack([spose[0] + r * np.cos(a), spose[1] + r * np.sin(a)], 1)
        self.us_hist.setdefault(m["name"], deque(maxlen=5)).append((m["t"], r, pose, pts))
        self._check_path = True

    def _ultrasonic_points(self, t):
        """Per sensor: echo points (cone centre and both edges) of its latest
        readings, in the current vehicle frame. Edges are included on purpose -
        the echo may have come from anywhere in the cone (bias to false positives)."""
        sc = self.cfg.safety
        names, chunks = [], []
        for name, hist in self.us_hist.items():
            readings = list(hist)[-sc.us_persist:]
            if len(readings) < sc.us_persist or any(
                    p is None or t - t_r > sc.us_max_age for t_r, _, _, p in readings):
                continue
            names.append(name)
            chunks.extend(p for _, _, _, p in readings)
        if not names:
            return {}
        loc = g.to_local(np.vstack(chunks), self.pose).reshape(len(names), sc.us_persist, 3, 2)
        return {n: list(loc[k]) for k, n in enumerate(names)}

    def on_camera(self, m):
        if not self.cfg.estimator.use_camera or self.task is None or self.task["kind"] != "park":
            return
        cc = self.cfg.camera
        R = np.diag([cc.lateral_sigma ** 2 * 4, math.radians(cc.heading_sigma_deg) ** 2 * 4])
        self.ekf.update_bay(m["t"], m["lateral"], m["heading"], self.map.bay_pose, self.off, R)

    # ---- app messages ------------------------------------------------- #
    def on_msg(self, t, msg):
        kind = msg.get("type")
        if kind == "heartbeat":
            self.last_hb = t
        elif kind == "estop":
            self.log(t, "E-STOP received")
            self._finish(t, "estop")
            self.state = "ESTOP"
        elif kind == "command":
            if msg["id"] <= self.ack_id:
                return
            self.ack_id = msg["id"]
            cmd = msg["cmd"]
            if cmd in ("stop", "estop"):
                self._finish(t, "estop")
                self.state = "ESTOP"
                self.done_id = msg["id"]
                return
            if cmd == "park":
                goal = g.center_to_rear(self.map.bay_pose, self.off)
            elif cmd == "summon":
                goal = g.center_to_rear(g.deg_pose(msg["pose"]), self.off)
            else:
                self.log(t, f"unknown command {cmd}")
                return
            if self.task is not None:
                self._finish(t, "preempted")
            self.task = {"id": msg["id"], "kind": cmd, "goal": goal, "t_start": t,
                         "replans": 0, "corrections": 0, "blocked": 0, "paused": 0}
            self.log(t, f"command #{msg['id']}: {cmd}")
            self._plan(t)

    def telemetry(self, t):
        return {"type": "telemetry", "t": t, "state": self.state, "ack_id": self.ack_id,
                "done_id": self.done_id, "pose": self.pose.tolist()}

    # ---- planning ----------------------------------------------------- #
    def _plan(self, t):
        task = self.task
        path, info = self.planner.plan(self.pose, task["goal"], self.grid)
        info["t"] = t
        self.plan_infos.append(info)
        if path is None:
            if "goal pose is blocked" not in info["reason"]:     # that check is cheap and may clear
                task["plan_failures"] = task.get("plan_failures", 0) + 1
            self.log(t, f"planning failed: {info['reason']}")
            self.tracker = None
            self.state = "BLOCKED"
            self.block_t = t
            return False
        task["plan_failures"] = 0
        self.tracker = PathTracker(self.cfg.vehicle, self.cfg.control, path)
        self.paths.append((t, path))
        pc = self.cfg.planner
        info["sim_compute_time"] = (pc.compute_base + pc.compute_per_expansion * info["expansions"]) * pc.compute_factor
        self.plan_ready_t = t + info["sim_compute_time"]
        self.state = "DRIVING"
        self.log(t, f"path {path.length:.2f} m, {path.n_cusps} cusps, {info['expansions']} expansions"
                    + (" (escaping tight start)" if info.get("escape") else ""))
        return True

    def _finish(self, t, result):
        if self.task is None:
            return
        task = self.task
        task.update({"t_end": t, "result": result, "est_pose": self.pose.tolist()})
        self.task_log.append(task)
        self.done_id = max(self.done_id, task["id"])
        self.task = None
        self.tracker = None
        self.log(t, f"task #{task['id']} {task['kind']} -> {result}")

    def _goal_error(self):
        gc = self._goal_center()
        pc = g.rear_to_center(self.pose, self.off)
        return math.hypot(*(pc[:2] - gc[:2])), abs(g.wrap(pc[2] - gc[2]))

    # ---- safety ------------------------------------------------------- #
    def _stop_zone(self, direction, sensor_period=0.0, persist=1):
        """Stop zone (x_lo, x_hi, half_width) ahead of / behind the bumper. It also
        covers the distance driven while waiting for `persist` more readings."""
        sc, vc = self.cfg.safety, self.cfg.vehicle
        v = abs(self.v_est)
        reach = (sc.bubble_margin + v * v / (2 * sc.decel_assumed) + v * self.cfg.control.actuation_latency
                 + v * sensor_period * persist)
        if self.tracker is not None:
            # never look further than where the car is about to stop anyway
            # (cusp or goal) - otherwise a planned stop near a wall livelocks
            # (padded by 3 sigma of our own position uncertainty)
            reach = min(reach, self.tracker.remaining + sc.bubble_margin + 3 * self.ekf.sigma_xy())
        front, rear = vc.length - vc.rear_overhang, -vc.rear_overhang
        lo, hi = (front, front + reach) if direction > 0 else (rear - reach, rear)
        return lo, hi, vc.width / 2 + sc.bubble_lateral

    def _ultrasonic_hit(self, t, direction):
        if direction == 0 or not self.us_hist:
            return False
        uc = self.cfg.ultrasonic
        cycle = len(uc.fire_groups) * uc.ping_interval      # time between readings of one sensor
        lo, hi, hw = self._stop_zone(direction, cycle, self.cfg.safety.us_persist)
        for name, per in self._ultrasonic_points(t).items():
            # every one of the last `us_persist` readings must reach into the zone
            if all(((p[:, 0] > lo) & (p[:, 0] < hi) & (np.abs(p[:, 1]) < hw)).any() for p in per):
                # Put what stopped us into the map as a definite obstacle (the whole
                # arc at that range - we don't know where in the cone it is), so the
                # re-plan goes around it instead of driving back into it.
                _, r, pose, _ = self.us_hist[name][-1]
                sx, sy, syaw = g.mount_pose(pose, self.us_sensors[name])
                half = math.radians(uc.half_angle_deg)
                r_wall = g.cone_echo_walls(sx, sy, syaw, half, self.map.walls, math.radians(uc.max_incidence_deg))
                if not (math.isfinite(r_wall) and r >= r_wall - self.cfg.mapping.us_wall_tol):
                    # stay one cell inside the cone edges, so later readings that see
                    # through this spot can clear every cell we mark (false alarms)
                    half_m = max(0.0, half - math.atan2(self.grid.res, r))
                    n = max(1, int(2 * half_m * r / (0.5 * self.grid.res)) + 1)
                    a = syaw + np.linspace(-half_m, half_m, n)
                    self.grid.mark_points(sx + r * np.cos(a), sy + r * np.sin(a),
                                          self.cfg.mapping.stop_mark_logodds)
                    return f"{name} @ {r*100:.1f} cm"
                return f"{name} @ {r*100:.1f} cm (wall)"
        return False

    def _bubble_hit(self, direction):
        """LiDAR points inside the stop zone ahead (or behind) the bumper."""
        if self._scan_pts is None or direction == 0:
            return False
        sc, vc = self.cfg.safety, self.cfg.vehicle
        # bring the (possibly old) scan into the current vehicle frame
        world = g.to_world(self._scan_pts, self._scan_pose)
        loc = g.to_local(world, self.pose)
        v = abs(self.v_est)
        reach = sc.bubble_margin + v * v / (2 * sc.decel_assumed) + v * self.cfg.control.actuation_latency
        front = vc.length - vc.rear_overhang
        rear = -vc.rear_overhang
        if direction > 0:
            lo, hi = front, front + reach
        else:
            lo, hi = rear - reach, rear
        m = (loc[:, 0] > lo) & (loc[:, 0] < hi) & (np.abs(loc[:, 1]) < vc.width / 2 + sc.bubble_lateral)
        hit = int(m.sum()) >= sc.min_points
        if self._bubble_scan_id != id(self._scan_pts):       # count once per scan
            self._bubble_scan_id = id(self._scan_pts)
            self._bubble_streak = self._bubble_streak + 1 if hit else 0
        return hit and self._bubble_streak >= sc.persist_scans

    def _path_blocked(self):
        poses = self.tracker.remaining_poses(self.cfg.safety.path_check_horizon)
        if len(poses) < 2:
            return False
        chk = FootprintChecker(self.cfg.vehicle, self.grid, self.cfg.safety.path_margin, use_edt=False)
        if chk.collides(self.pose[None]).any():
            # same rule as the planner: allow creeping away from a close obstacle
            relaxed = FootprintChecker(self.cfg.vehicle, self.grid, -self.grid.res / 2.0, use_edt=False)
            chk = EscapeChecker(chk, relaxed, self.pose, self.cfg.planner.escape_radius)
        hit = chk.collides(poses[1:])
        return bool(hit.any())

    def _stop(self):
        self.v_ref = 0.0

    # ---- main control tick -------------------------------------------- #
    def control_step(self, t, dt):
        cfg = self.cfg
        hb_ok = (t - self.last_hb) <= cfg.comm.heartbeat_timeout
        active = self.state in ("DRIVING", "BLOCKED")

        if active and not hb_ok:
            self.state = "PAUSED"
            self.task["paused"] += 1
            self.log(t, "heartbeat lost -> failsafe stop")
        elif self.state == "PAUSED" and hb_ok:
            if cfg.comm.resume_after_outage:
                self.log(t, "heartbeat back -> re-plan and resume")
                self.state = "BLOCKED"
                self.block_t = t - cfg.safety.blocked_retry_s
            else:
                self._finish(t, "link_lost")
                self.state = "FAULT"

        if self.state in ("DRIVING", "BLOCKED", "PAUSED") and self.task is not None:
            if self.ekf.sigma_xy() > cfg.estimator.lost_sigma or \
                    self.lidar_rejects > cfg.estimator.max_lidar_rejects:
                self.log(t, f"localization lost (sigma={self.ekf.sigma_xy()*100:.1f} cm, "
                            f"rejects={self.lidar_rejects})")
                self._finish(t, "loc_lost")
                self.state = "FAULT"

        self.v_ref = 0.0
        if self.state == "DRIVING" and t >= self.plan_ready_t:
            tr = self.tracker
            moving_dir = tr.direction if tr.phase == "drive" else 0
            reason = None
            if self._bubble_hit(moving_dir):
                reason = "LiDAR points in stop zone"
            else:
                us = self._ultrasonic_hit(t, moving_dir)
                if us:
                    reason = f"ultrasonic {us} in stop zone"
            if reason is None and self._check_path and t - self._last_path_check >= 0.1:
                self._check_path = False
                self._last_path_check = t
                if self._path_blocked():
                    reason = "map shows obstacle on path"
            if reason:
                self.state = "BLOCKED"
                self.block_t = t
                self.task["blocked"] += 1
                self.log(t, f"{reason} -> stop")
            else:
                delta, v_ref, status = tr.step(t, self.pose, self.v_est)
                self.delta_cmd, self.v_ref = delta, v_ref
                if status == "done" and abs(self.v_est) < 0.01:
                    perr, yerr = self._goal_error()
                    c = cfg.control
                    if (perr > c.goal_pos_tol or yerr > math.radians(c.goal_yaw_tol_deg)) and \
                            self.task["corrections"] < c.max_corrections:
                        self.task["corrections"] += 1
                        self.log(t, f"final error {perr*100:.1f} cm / {math.degrees(yerr):.1f} deg -> correction")
                        self._plan(t)
                    else:
                        self._finish(t, "done")
                        self.state = "IDLE"

        elif self.state == "BLOCKED":
            if t - self.block_t >= cfg.safety.blocked_retry_s and abs(self.v_est) < 0.02:
                perr, yerr = self._goal_error()
                if perr <= cfg.control.goal_pos_tol and yerr <= math.radians(cfg.control.goal_yaw_tol_deg):
                    self._finish(t, "done")
                    self.state = "IDLE"
                elif (self.task["replans"] >= cfg.safety.max_replans
                      or self.task.get("plan_failures", 0) >= cfg.safety.max_plan_failures):
                    self._finish(t, "no_path")
                    self.state = "FAULT"
                else:
                    self.task["replans"] += 1
                    self._plan(t)
                    if self.state == "BLOCKED":
                        self.block_t = t

        duty = self.speed.step(self.v_ref, self.v_est, dt)
        return self.delta_cmd, duty
