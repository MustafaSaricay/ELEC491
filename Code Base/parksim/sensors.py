"""Sensor models. Each produces a measurement dict stamped with the time it was
*taken*; the simulator delivers it to the onboard software after ``latency``.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from . import geometry as g
from .config import CameraCfg, EncoderCfg, ImuCfg, LidarCfg


class DelayLine:
    """Time-ordered queue: push(t_ready, item), pop_ready(now)."""

    def __init__(self):
        self._q = []
        self._seq = itertools.count()

    def push(self, t_ready, item):
        heapq.heappush(self._q, (t_ready, next(self._seq), item))

    def pop_ready(self, now):
        out = []
        while self._q and self._q[0][0] <= now + 1e-12:
            out.append(heapq.heappop(self._q)[2])
        return out

    def __len__(self):
        return len(self._q)


class Encoder:
    def __init__(self, cfg: EncoderCfg, rng):
        self.cfg, self.rng = cfg, rng
        self._last_ticks = 0
        self._missed = 0
        self._last_t = 0.0

    @property
    def m_per_tick(self):
        return 2 * math.pi * self.cfg.wheel_radius / self.cfg.ticks_per_rev

    def sample(self, t, wheel_dist):
        true_m_per_tick = self.m_per_tick * (1.0 + self.cfg.scale_error)
        ticks = math.floor(wheel_dist / true_m_per_tick)
        d = ticks - self._last_ticks
        self._last_ticks = ticks
        if self.cfg.miss_prob > 0 and d != 0:
            d = int(math.copysign(abs(d) - self.rng.binomial(abs(d), self.cfg.miss_prob), d))
        dt = t - self._last_t
        self._last_t = t
        return {"kind": "encoder", "t": t, "ds": d * self.m_per_tick, "dt": dt}


class Imu:
    def __init__(self, cfg: ImuCfg, rng):
        self.cfg, self.rng = cfg, rng
        self.bias = math.radians(cfg.gyro_bias_dps) * rng.standard_normal()
        self._last_t = 0.0

    def sample(self, t, true_rate):
        c = self.cfg
        dt = t - self._last_t
        self._last_t = t
        self.bias += math.radians(c.bias_walk_dps) * math.sqrt(max(dt, 0)) * self.rng.standard_normal()
        w = (1 + c.scale_error) * true_rate + self.bias + math.radians(c.gyro_noise_dps) * self.rng.standard_normal()
        return {"kind": "imu", "t": t, "wz": w}


class Lidar:
    """360 deg 2-D scanner. Obstacles lower than ``mount_height`` are invisible."""

    def __init__(self, cfg: LidarCfg, rng):
        self.cfg, self.rng = cfg, rng

    def beam_angles(self):
        n = self.cfg.n_beams
        return np.arange(n) * (2 * math.pi / n)

    def scan(self, t, pose_at, world):
        """pose_at(times) -> (N, 3) true rear-axle poses (for motion distortion)."""
        c = self.cfg
        a = self.beam_angles()
        n = len(a)
        period = 1.0 / c.rate_hz
        if c.motion_distortion:
            times = t - period + (np.arange(n) + 1) * (period / n)
            stamp = t - period / 2.0
        else:
            times = np.full(n, t)
            stamp = t
        poses = pose_at(times)
        ox = poses[:, 0] + c.mount_x * np.cos(poses[:, 2])
        oy = poses[:, 1] + c.mount_x * np.sin(poses[:, 2])
        world_ang = poses[:, 2] + a + math.radians(c.yaw_offset_deg)
        r = world.raycast(ox, oy, world_ang, c.mount_height)
        rng = self.rng
        if c.range_sigma > 0:
            r = r + c.range_sigma * rng.standard_normal(n)
        if c.spurious_prob > 0:
            sp = rng.random(n) < c.spurious_prob
            hi = np.minimum(r, c.range_max)
            r = np.where(sp, rng.uniform(c.range_min, np.maximum(hi, c.range_min + 1e-3)), r)
        if c.dropout_prob > 0:
            r = np.where(rng.random(n) < c.dropout_prob, np.nan, r)
        r = np.where((r < c.range_min) | (r > c.range_max) | ~np.isfinite(r), np.nan, r)
        return {"kind": "lidar", "t": stamp, "angles": a, "ranges": r}


SPEED_OF_SOUND = 343.0


sensor_pose = g.mount_pose


class UltrasonicArray:
    """HC-SR04 array fired group by group (round robin)."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.slot = 0

    def fire(self, t, rear_pose, world):
        """Fire the next group. Returns [(delay, measurement), ...]."""
        c = self.cfg
        by_name = {s["name"]: s for s in c.sensors}
        group = [n for n in c.fire_groups[self.slot % len(c.fire_groups)] if n in by_name]
        self.slot += 1
        rng = self.rng
        out = []
        for name in group:
            sx, sy, syaw = sensor_pose(rear_pose, by_name[name])
            r_true = world.ultrasonic_echo(sx, sy, syaw, c)
            r = r_true * (1.0 + c.scale_error)
            if math.isfinite(r):
                r += (c.range_sigma + c.range_sigma_prop * r) * rng.standard_normal()
            cap = min(r, c.range_max) if math.isfinite(r) else c.range_max
            for _ in range(len(group) - 1):           # crosstalk from simultaneous pings
                if c.crosstalk_prob > 0 and rng.random() < c.crosstalk_prob:
                    r = rng.uniform(c.range_min, max(cap, c.range_min + 1e-3))
            if c.spurious_prob > 0 and rng.random() < c.spurious_prob:
                r = rng.uniform(c.range_min, max(cap, c.range_min + 1e-3))
            if c.dropout_prob > 0 and rng.random() < c.dropout_prob:
                r = math.nan
            if not math.isfinite(r) or r < c.range_min or r > c.range_max:
                r = math.nan                           # blind zone / timeout / no echo
            elif c.resolution > 0:
                r = round(r / c.resolution) * c.resolution
            tof = 2 * r / SPEED_OF_SOUND if math.isfinite(r) else 2 * c.range_max / SPEED_OF_SOUND
            out.append((tof + c.latency, {"kind": "ultrasonic", "t": t + tof / 2, "name": name, "range": r}))
        return out


class BayCamera:
    """Outputs the vehicle's lateral offset and heading relative to the bay axis
    when the bay end line is inside the camera field of view."""

    def __init__(self, cfg: CameraCfg, rng, vehicle_cfg):
        self.cfg, self.rng, self.vc = cfg, rng, vehicle_cfg

    def observe(self, t, rear_pose, bay_pose, bay_depth):
        c = self.cfg
        if self.rng.random() < c.dropout_prob:
            return None
        x, y, th = rear_pose
        if c.mount == "front":
            mx, cam_yaw = self.vc.length - self.vc.rear_overhang, th
        else:
            mx, cam_yaw = -self.vc.rear_overhang, th + math.pi
        cx, cy = x + mx * math.cos(th), y + mx * math.sin(th)
        bx, by, bth = bay_pose
        # bay end-line centre (the end the car drives towards when parked nose-in)
        ex, ey = bx + bay_depth / 2 * math.cos(bth), by + bay_depth / 2 * math.sin(bth)
        dist = math.hypot(ex - cx, ey - cy)
        bearing = g.wrap(math.atan2(ey - cy, ex - cx) - cam_yaw)
        if dist > c.max_range or abs(bearing) > math.radians(c.fov_deg) / 2:
            return None
        off = self.vc.center_offset
        pcx, pcy = x + off * math.cos(th), y + off * math.sin(th)
        nx, ny = -math.sin(bth), math.cos(bth)
        lateral = (pcx - bx) * nx + (pcy - by) * ny
        heading = g.wrap(th - bth)
        lateral += c.lateral_sigma * self.rng.standard_normal()
        heading += math.radians(c.heading_sigma_deg) * self.rng.standard_normal()
        if c.outlier_prob > 0 and self.rng.random() < c.outlier_prob:
            lateral += self.rng.uniform(-0.05, 0.05)
            heading += math.radians(self.rng.uniform(-10, 10))
        return {"kind": "camera", "t": t, "lateral": lateral, "heading": heading}
