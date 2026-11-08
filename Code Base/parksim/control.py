"""Low-level control: PID speed loop with dead-zone compensation, and a
segment-by-segment Pure Pursuit path tracker that stops at every cusp."""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .config import ControlCfg, VehicleCfg


class SpeedController:
    def __init__(self, cfg: ControlCfg):
        self.cfg = cfg
        self.integ = 0.0
        self.prev_e = 0.0
        self.prev_sign = 0

    def reset(self):
        self.integ = self.prev_e = 0.0

    def step(self, v_ref, v_est, dt):
        c = self.cfg
        if abs(v_ref) < 1e-6:
            self.reset()
            self.prev_sign = 0
            return 0.0
        sign = 1 if v_ref > 0 else -1
        if sign != self.prev_sign:
            self.reset()
            self.prev_sign = sign
        e = v_ref - v_est
        self.integ = float(np.clip(self.integ + e * dt, -0.3, 0.3))
        de = (e - self.prev_e) / dt
        self.prev_e = e
        duty = (v_ref / c.motor_max_speed + c.kp * e + c.ki * self.integ + c.kd * de
                + sign * c.deadzone_comp)
        # never command the opposite direction while tracking (no "brake by reversing")
        duty = max(duty, 0.0) if sign > 0 else min(duty, 0.0)
        return float(np.clip(duty, -1.0, 1.0))


class PathTracker:
    """Pure Pursuit on the rear axle. kappa = 2 y_r / d^2 holds for forward and
    reverse travel; delta = atan(L * kappa)."""

    def __init__(self, vcfg: VehicleCfg, cfg: ControlCfg, path, final_goal=True):
        self.v, self.c = vcfg, cfg
        self.path = path
        self.segs = path.segments()
        self.i = 0
        self.near = 0
        self.phase = "drive"
        self.t_wait = 0.0
        self.last_delta = 0.0
        self.remaining = math.inf
        self.final_goal = final_goal
        self._prep()

    def _prep(self):
        seg = self.segs[self.i]
        self.s = seg.s
        self.near = 0

    @property
    def segment(self):
        return self.segs[self.i]

    @property
    def direction(self):
        return int(self.segment.dir[0])

    def remaining_poses(self, horizon):
        """Upcoming path poses within `horizon` metres (for the safety checker)."""
        out, acc = [], 0.0
        for k in range(self.i, len(self.segs)):
            seg = self.segs[k]
            start = self.near if k == self.i else 0
            p = seg.poses[start:]
            out.append(p)
            if len(p) > 1:
                acc += float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum())
            if acc > horizon:
                break
        poses = np.vstack(out) if out else np.zeros((0, 3))
        if len(poses) > 1:
            d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
            poses = poses[d <= horizon]
        return poses

    def step(self, t, pose, v_est):
        """Returns (delta_cmd, v_ref, status) with status in drive|cusp|done."""
        c = self.c
        if self.phase == "done":
            return self.last_delta, 0.0, "done"
        seg = self.segment
        d = self.direction
        if self.phase == "cusp":
            if abs(v_est) < 0.01 and t - self.t_wait >= c.cusp_pause:
                self.i += 1
                self._prep()
                self.phase = "drive"
                seg, d = self.segment, self.direction
            else:
                # pre-steer for the next segment while stopped
                return self._pursuit(self.segs[self.i + 1], pose, 0.0, int(self.segs[self.i + 1].dir[0]), 0), 0.0, "cusp"

        # nearest point (monotonic search in a window)
        xy = np.stack([seg.x, seg.y], 1)
        hi = min(len(xy), self.near + 40)
        dd = np.hypot(xy[self.near:hi, 0] - pose[0], xy[self.near:hi, 1] - pose[1])
        self.near += int(dd.argmin())

        end = np.array([seg.x[-1], seg.y[-1]])
        t_end = np.array([math.cos(seg.yaw[-1]), math.sin(seg.yaw[-1])]) * d
        rem_proj = float((end - pose[:2]) @ t_end)
        rem = max(float(self.s[-1] - self.s[self.near]), 0.0)
        rem = min(rem, max(rem_proj, 0.0)) if self.near >= len(xy) - 3 else rem

        self.remaining = rem                      # distance to this segment's stop point
        last = self.i == len(self.segs) - 1
        tol = c.goal_pos_tol if last else c.cusp_tol
        # cut the motor when the remaining distance equals the expected coast-down
        stop_at = max(0.003, abs(v_est) * c.stop_lead_time)
        if self.near >= len(xy) - 3 and rem_proj <= stop_at:
            if last:
                self.phase = "done"
                return self.last_delta, 0.0, "done"
            self.phase = "cusp"
            self.t_wait = t
            return self.last_delta, 0.0, "cusp"

        # speed profile: stop at the end of each segment
        v_max = c.approach_speed if (last and rem < c.approach_dist) else c.cruise_speed
        v_ref = min(v_max, math.sqrt(2 * c.decel * max(rem - 0.5 * tol, 0.0)))
        v_ref = max(v_ref, c.min_speed)
        delta = self._pursuit(seg, pose, abs(v_est), d, self.near)
        return delta, d * v_ref, "drive"

    def _pursuit(self, seg, pose, speed, d, near):
        c = self.c
        Ld = (c.lookahead_fwd if d > 0 else c.lookahead_rev) + c.lookahead_gain * speed
        s = seg.s
        target_s = s[near] + Ld
        if target_s <= s[-1]:
            j = int(np.searchsorted(s, target_s))
            tx, ty = seg.x[j], seg.y[j]
        else:   # extend beyond the segment end along its final tangent
            extra = target_s - s[-1]
            th = seg.yaw[-1]
            tx = seg.x[-1] + d * extra * math.cos(th)
            ty = seg.y[-1] + d * extra * math.sin(th)
        loc = g.to_local(np.array([[tx, ty]]), pose)[0]
        d2 = float(loc @ loc)
        kappa = 0.0 if d2 < 1e-9 else 2.0 * loc[1] / d2
        delta = math.atan(self.v.wheelbase * kappa)
        delta = float(np.clip(delta, -self.v.max_steer, self.v.max_steer))
        self.last_delta = delta
        return delta
