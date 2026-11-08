"""True vehicle plant: kinematic bicycle model + actuator / traction imperfections.

Inputs are what the Raspberry Pi would write to the hardware:
  * ``delta_cmd`` - requested bicycle steering angle [rad] (servo set-point)
  * ``duty``      - motor driver duty cycle in [-1, 1]

Effects modelled (all configurable, see SteeringCfg / DriveCfg / TractionCfg):
  steering: gain error, trim offset, backlash, slew-rate limit, lag, jitter,
            non-ideal Ackermann linkage (tyre scrub reduces curvature)
  drive:    dead zone, motor lag, battery sag, load disturbance
  traction: longitudinal slip (encoder sees wheel, body moves less),
            slip zones, heading random walk, understeer at speed
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .config import DriveCfg, SteeringCfg, TractionCfg, VehicleCfg


class Vehicle:
    def __init__(self, vcfg: VehicleCfg, scfg: SteeringCfg, dcfg: DriveCfg,
                 tcfg: TractionCfg, rng: np.random.Generator, rear_pose):
        self.v_cfg, self.s_cfg, self.d_cfg, self.t_cfg = vcfg, scfg, dcfg, tcfg
        self.rng = rng
        self.x, self.y, self.yaw = (float(v) for v in rear_pose)
        self.delta = 0.0          # actual bicycle steering angle
        self._backlash_out = 0.0
        self.v_wheel = 0.0        # wheel surface speed (what the encoder sees)
        self.v = 0.0              # body speed over ground
        self.omega = 0.0          # yaw rate
        self.wheel_dist = 0.0     # signed integrated wheel travel
        self._slip = 0.0
        self._load = 0.0
        self.t = 0.0
        self.cmd_delta = 0.0
        self.cmd_duty = 0.0
        self.frozen = False

    # ------------------------------------------------------------------ #
    @property
    def pose(self):
        return np.array([self.x, self.y, self.yaw])

    @property
    def center_pose(self):
        return g.rear_to_center(self.pose, self.v_cfg.center_offset)

    def footprint(self):
        c = self.center_pose
        return g.rect_corners(c[0], c[1], c[2], self.v_cfg.length, self.v_cfg.width)

    def set_command(self, delta_cmd: float, duty: float):
        self.cmd_delta, self.cmd_duty = float(delta_cmd), float(duty)

    def teleport(self, dx, dy, dyaw):
        """'Kidnap' disturbance (someone bumps the car)."""
        self.x += dx
        self.y += dy
        self.yaw = g.wrap(self.yaw + dyaw)

    def freeze(self):
        self.frozen = True
        self.v = self.v_wheel = 0.0

    def wheel_angles(self):
        """Actual (left, right) front road-wheel angles from the Ackermann linkage."""
        L, w, d = self.v_cfg.wheelbase, self.v_cfg.track, self.delta
        if abs(d) < 1e-6:
            return 0.0, 0.0
        R = L / math.tan(d)                      # signed, + = left turn
        left = math.atan(L / (R - w / 2.0))
        right = math.atan(L / (R + w / 2.0))
        a = self.s_cfg.ackermann_error           # blend towards parallel steer
        return (1 - a) * left + a * d, (1 - a) * right + a * d

    # ------------------------------------------------------------------ #
    def step(self, dt: float):
        self.t += dt
        if self.frozen:
            return
        sc, dc, tc, vc = self.s_cfg, self.d_cfg, self.t_cfg, self.v_cfg
        rng = self.rng
        dmax = vc.max_steer

        # ---- steering servo + linkage ---------------------------------
        target = sc.gain * self.cmd_delta + math.radians(sc.offset_deg)
        half_play = math.radians(sc.backlash_deg) / 2.0
        if target - self._backlash_out > half_play:
            self._backlash_out = target - half_play
        elif target - self._backlash_out < -half_play:
            self._backlash_out = target + half_play
        desired = min(max(self._backlash_out, -dmax), dmax)
        rate = (desired - self.delta) / max(sc.time_const, dt)
        rl = math.radians(sc.rate_limit_dps)
        self.delta += min(max(rate, -rl), rl) * dt
        delta_eff = self.delta
        if sc.noise_deg > 0:
            delta_eff += math.radians(sc.noise_deg) * rng.standard_normal()

        # ---- drive motor ----------------------------------------------
        u = min(max(self.cmd_duty, -1.0), 1.0)
        dz = dc.dead_zone
        eff = 0.0 if abs(u) <= dz else math.copysign((abs(u) - dz) / (1.0 - dz), u)
        battery = max(0.3, 1.0 - dc.battery_sag_per_min * self.t / 60.0)
        if dc.speed_noise > 0:   # load disturbance, OU process with 0.3 s correlation
            tau = 0.3
            self._load += -self._load * dt / tau + dc.speed_noise * math.sqrt(2 * dt / tau) * rng.standard_normal()
        target_v = eff * dc.max_speed * battery
        self.v_wheel += (target_v - self.v_wheel) * min(1.0, dt / max(dc.time_const, 1e-6))
        v_wheel = self.v_wheel + (self._load if eff != 0.0 else 0.0)

        # ---- traction --------------------------------------------------
        slip = 0.0
        for zone in tc.slip_zones:
            x0, y0, x1, y1 = zone["rect"]
            if x0 <= self.x <= x1 and y0 <= self.y <= y1:
                slip += zone["slip"]
        if tc.slip_std > 0:
            tau = 0.5
            self._slip += -self._slip * dt / tau + tc.slip_std * math.sqrt(2 * dt / tau) * rng.standard_normal()
        slip = min(max(slip + self._slip, -0.2), 0.9)
        v = v_wheel * (1.0 - slip)

        kappa = math.tan(delta_eff) / vc.wheelbase
        if sc.ackermann_error > 0:
            kappa *= 1.0 - 0.3 * sc.ackermann_error * (abs(self.delta) / dmax) ** 2
        if tc.understeer > 0:
            kappa /= 1.0 + tc.understeer * v * v
        omega = v * kappa
        dyaw_noise = tc.yaw_noise * math.sqrt(abs(v) * dt) * rng.standard_normal() if tc.yaw_noise > 0 else 0.0

        # ---- integrate (mid-point) ---------------------------------------
        yaw_mid = self.yaw + 0.5 * omega * dt
        self.x += v * math.cos(yaw_mid) * dt
        self.y += v * math.sin(yaw_mid) * dt
        self.yaw = g.wrap(self.yaw + omega * dt + dyaw_noise)
        self.omega = omega + dyaw_noise / dt
        self.v = v
        self.wheel_dist += v_wheel * dt
