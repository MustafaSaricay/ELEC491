"""Localization: EKF (odometry + gyro prediction, LiDAR scan-match and camera
bay-line updates) with optional out-of-sequence measurement handling, and a
point-to-line ICP scan matcher against the known testbed walls.

State: x = [x, y, yaw, gyro_bias] (rear-axle pose).
"""
from __future__ import annotations

import bisect
import math

import numpy as np

from . import geometry as g
from .config import EstimatorCfg


class EKF:
    def __init__(self, cfg: EstimatorCfg, pose0, t0=0.0):
        self.cfg = cfg
        self.x = np.array([pose0[0], pose0[1], pose0[2], 0.0])
        sxy, syaw = cfg.init_sigma_xy, math.radians(cfg.init_sigma_yaw_deg)
        self.P = np.diag([sxy ** 2, sxy ** 2, syaw ** 2, math.radians(1.0) ** 2])
        # history entries: [t, kind, data, x, P]  (x, P = posterior after entry)
        self.hist = [[t0, "init", None, self.x.copy(), self.P.copy()]]
        self.stale_updates = 0
        self.rejected = {"pose": 0, "camera": 0}
        self.accepted = {"pose": 0, "camera": 0}

    # ------------------------------------------------------------------ #
    @property
    def t(self):
        return self.hist[-1][0]

    @property
    def pose(self):
        return self.x[:3].copy()

    def sigma_xy(self):
        return math.sqrt(max(np.linalg.eigvalsh(self.P[:2, :2]).max(), 0.0))

    def pose_at(self, t):
        if not self.cfg.compensate_latency:
            return self.pose
        times = [h[0] for h in self.hist]
        i = bisect.bisect_right(times, t) - 1
        return self.hist[max(i, 0)][3][:3].copy()

    # ------------------------------------------------------------------ #
    def _predict(self, x, P, data):
        ds, dth_meas, dt = data
        c = self.cfg
        th, b = x[2], x[3]
        dth = dth_meas - b * dt
        thm = th + 0.5 * dth
        cm, sm = math.cos(thm), math.sin(thm)
        xn = np.array([x[0] + ds * cm, x[1] + ds * sm, g.wrap(th + dth), b])
        F = np.eye(4)
        F[0, 2], F[1, 2] = -ds * sm, ds * cm
        F[0, 3], F[1, 3] = ds * sm * dt / 2, -ds * cm * dt / 2
        F[2, 3] = -dt
        # noise on [ds, lateral slip, dtheta]
        G = np.array([[cm, -sm, -ds * sm / 2],
                      [sm, cm, ds * cm / 2],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]])
        q_ds = (c.enc_sigma_frac * abs(ds) + c.enc_sigma_min) ** 2
        q_lat = (c.lateral_sigma_frac * abs(ds)) ** 2
        q_th = (math.radians(c.gyro_sigma_dps) * math.sqrt(max(dt, 1e-6))) ** 2
        Q = G @ np.diag([q_ds, q_lat, q_th]) @ G.T
        Q[3, 3] += (math.radians(c.bias_walk_dps) ** 2) * dt
        return xn, F @ P @ F.T + Q

    def _update(self, x, P, data):
        """data = (z, model, gate, kind). Returns (x, P, accepted)."""
        z, model, gate, kind = data
        out = model(x)
        if out is None:
            return x, P, False
        h, H, R, ang = out
        y = z - h
        for i in ang:
            y[i] = g.wrap(y[i])
        S = H @ P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return x, P, False
        if float(y @ Sinv @ y) > gate:
            return x, P, False
        K = P @ H.T @ Sinv
        xn = x + K @ y
        xn[2] = g.wrap(xn[2])
        IKH = np.eye(4) - K @ H
        Pn = IKH @ P @ IKH.T + K @ R @ K.T
        return xn, Pn, True

    # ------------------------------------------------------------------ #
    def predict(self, t, ds, dth_meas, dt):
        data = (ds, dth_meas, dt)
        self.x, self.P = self._predict(self.x, self.P, data)
        self.hist.append([t, "pred", data, self.x.copy(), self.P.copy()])
        self._trim()

    def _trim(self):
        horizon = self.t - self.cfg.history_s
        k = 0
        while k + 1 < len(self.hist) and self.hist[k + 1][0] < horizon:
            k += 1
        if k:
            del self.hist[:k]

    def update(self, t_meas, z, model, gate, kind):
        data = (np.asarray(z, float), model, gate, kind)
        if not self.cfg.compensate_latency or t_meas >= self.t:
            self.x, self.P, ok = self._update(self.x, self.P, data)
            self.hist.append([max(t_meas, self.t), "upd", data, self.x.copy(), self.P.copy()])
        else:
            times = [h[0] for h in self.hist]
            i = bisect.bisect_right(times, t_meas) - 1
            if i < 0:                       # older than the history window
                self.stale_updates += 1
                self.x, self.P, ok = self._update(self.x, self.P, data)
            else:
                x, P = self.hist[i][3], self.hist[i][4]
                x, P, ok = self._update(x, P, data)
                self.hist.insert(i + 1, [t_meas, "upd", data, x.copy(), P.copy()])
                for e in self.hist[i + 2:]:            # replay later inputs
                    if e[1] == "pred":
                        x, P = self._predict(x, P, e[2])
                    else:
                        x, P, _ = self._update(x, P, e[2])
                    e[3], e[4] = x.copy(), P.copy()
                self.x, self.P = x.copy(), P.copy()
        self.accepted.setdefault(kind, 0)
        self.rejected.setdefault(kind, 0)
        (self.accepted if ok else self.rejected)[kind] += 1
        return ok

    @staticmethod
    def expected_range(pose, sensor, walls, ucfg):
        sx, sy, syaw = g.mount_pose(pose, sensor)
        return g.cone_echo_walls(sx, sy, syaw, math.radians(ucfg.half_angle_deg), walls,
                                 math.radians(ucfg.max_incidence_deg))

    def inflate(self, sxy, syaw_deg):
        """Covariance inflation when measurements persistently disagree."""
        self.P[0, 0] += sxy ** 2
        self.P[1, 1] += sxy ** 2
        self.P[2, 2] += math.radians(syaw_deg) ** 2
        self.hist[-1][4] = self.P.copy()

    def update_range(self, t_meas, r, sensor, walls, ucfg):
        """One ultrasonic range against the known walls. The expected value uses
        the same cone/specular model as the sensor; the Jacobian is numeric.
        Echoes from (unknown) obstacles are short and get gated out."""
        c = self.cfg
        half = math.radians(ucfg.half_angle_deg)
        max_inc = math.radians(ucfg.max_incidence_deg)
        no_circles = np.zeros((0, 3))
        sigma = c.us_sigma + c.us_sigma_prop * r
        R = np.array([[sigma ** 2]])

        def expected(p):
            sx, sy, syaw = g.mount_pose(p, sensor)
            return g.cone_echo_walls(sx, sy, syaw, half, walls, max_inc)

        def model(x):
            h0 = expected(x[:3])
            if not math.isfinite(h0):
                return None
            H = np.zeros((1, 4))
            for i, eps in enumerate((1e-4, 1e-4, 1e-4)):
                p = x[:3].copy()
                p[i] += eps
                hi = expected(p)
                if not math.isfinite(hi) or abs(hi - h0) > 0.05:
                    return None               # at a discontinuity (corner, specular edge)
                H[0, i] = (hi - h0) / eps
            return np.array([h0]), H, R, []

        return self.update(t_meas, [r], model, c.gate_range, "range")

    def relocalize(self, t_meas, z, cov):
        """Hard reset to a trusted scan match: shift the current state by the
        correction found at t_meas and widen the covariance."""
        old = self.pose_at(t_meas)
        d = np.asarray(z) - old
        d[2] = g.wrap(d[2])
        # rotate the correction about the old scan pose, then apply to now
        c, s = math.cos(d[2]), math.sin(d[2])
        rel = self.x[:2] - old[:2]
        self.x[:2] = z[:2] + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        self.x[2] = g.wrap(self.x[2] + d[2])
        self.P[:3, :3] = cov + np.diag([0.01 ** 2, 0.01 ** 2, math.radians(2) ** 2])
        self.P[:3, 3] = self.P[3, :3] = 0.0
        self.hist = [[self.t, "init", None, self.x.copy(), self.P.copy()]]

    # ---- measurement models ------------------------------------------ #
    def update_pose(self, t_meas, z, R):
        H = np.zeros((3, 4))
        H[:, :3] = np.eye(3)
        return self.update(t_meas, z, lambda x: (x[:3].copy(), H, R, [2]),
                           self.cfg.gate_pose, "pose")

    def update_bay(self, t_meas, lateral, heading, bay_pose, center_offset, R):
        bx, by, bth = bay_pose
        nx, ny = -math.sin(bth), math.cos(bth)

        def model(x):
            cx = x[0] + center_offset * math.cos(x[2])
            cy = x[1] + center_offset * math.sin(x[2])
            h = np.array([(cx - bx) * nx + (cy - by) * ny, g.wrap(x[2] - bth)])
            H = np.array([[nx, ny, center_offset * (-math.sin(x[2]) * nx + math.cos(x[2]) * ny), 0.0],
                          [0.0, 0.0, 1.0, 0.0]])
            return h, H, R, [1]

        return self.update(t_meas, [lateral, heading], model, self.cfg.gate_cam, "camera")


class ScanMatcher:
    """Gauss-Newton point-to-line ICP against the known wall segments.
    Points far from every wall (obstacles, interference) are rejected."""

    def __init__(self, walls, cfg: EstimatorCfg):
        self.cfg = cfg
        self.a = walls[:, :2]
        self.e = walls[:, 2:] - walls[:, :2]
        self.ee = (self.e ** 2).sum(1)
        n = np.stack([-self.e[:, 1], self.e[:, 0]], 1)
        self.n = n / np.linalg.norm(n, axis=1, keepdims=True)

    def _associate(self, q):
        t = np.clip(((q[:, None, :] - self.a[None]) * self.e[None]).sum(-1) / self.ee[None], 0, 1)
        foot = self.a[None] + t[..., None] * self.e[None]
        d = np.linalg.norm(q[:, None, :] - foot, axis=-1)
        j = d.argmin(1)
        return j, d[np.arange(len(q)), j]

    def match(self, pts_vehicle, guess):
        c = self.cfg
        if len(pts_vehicle) < c.icp_min_inliers:
            return None
        p = np.array(guess, float)
        thr = c.icp_inlier_dist
        H = None
        for it in range(c.icp_iters):
            q = g.to_world(pts_vehicle, p)
            j, d = self._associate(q)
            inl = d < thr
            if inl.sum() < c.icp_min_inliers:
                return None
            n = self.n[j[inl]]
            qi = q[inl]
            r = ((qi - self.a[j[inl]]) * n).sum(1)
            rel = qi - p[:2]
            J = np.stack([n[:, 0], n[:, 1], -rel[:, 1] * n[:, 0] + rel[:, 0] * n[:, 1]], 1)
            H = J.T @ J
            if np.linalg.eigvalsh(H).min() < 1e-3 * len(r):
                return None                      # degenerate (e.g. only one wall visible)
            delta = -np.linalg.solve(H, J.T @ r)
            p += delta
            p[2] = g.wrap(p[2])
            if it == 2:
                thr = max(0.03, 3 * c.icp_sigma)
            if abs(delta[0]) + abs(delta[1]) < 1e-5 and abs(delta[2]) < 1e-5:
                break
        q = g.to_world(pts_vehicle, p)
        j, d = self._associate(q)
        inl = d < thr
        if inl.sum() < c.icp_min_inliers:
            return None
        r = ((q[inl] - self.a[j[inl]]) * self.n[j[inl]]).sum(1)
        sigma2 = max(float(np.mean(r ** 2)), c.icp_sigma ** 2)
        cov = sigma2 * np.linalg.inv(H) * c.lidar_cov_scale
        cov += np.diag([c.icp_floor_xy ** 2, c.icp_floor_xy ** 2, math.radians(c.icp_floor_yaw_deg) ** 2])
        return p, cov, float(inl.mean())
