"""Exact rectangle-footprint collision check against occupied grid cells,
with a distance-field quick-accept for poses far from everything."""
from __future__ import annotations

import math

import numpy as np


class FootprintChecker:
    def __init__(self, vehicle_cfg, grid, margin, use_edt=True):
        """use_edt=False skips the distance-field quick-accept (cheaper when only
        a few poses are checked against a map that changes all the time)."""
        self.grid = grid
        self.use_edt = use_edt
        self.occ = grid.occupied_points()
        pad = margin + grid.res / 2.0              # occupied cell = square of side res
        self.hl = vehicle_cfg.length / 2.0 + pad
        self.hw = vehicle_cfg.width / 2.0 + pad
        self.off = vehicle_cfg.center_offset
        self.R = math.hypot(self.hl, self.hw) + grid.res

    def collides(self, poses):
        """poses: (N, 3) rear-axle poses. Returns bool mask (N,)."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        cx = poses[:, 0] + self.off * c
        cy = poses[:, 1] + self.off * s
        hit = np.zeros(len(poses), bool)
        if len(self.occ) == 0:
            return hit
        near = self.grid.distance_at(cx, cy) < self.R if self.use_edt else np.ones(len(poses), bool)
        if not near.any():
            return hit
        idx = np.nonzero(near)[0]
        # only occupied cells inside the bounding box of the poses (+ footprint radius)
        R = self.R
        occ = self.occ
        box = ((occ[:, 0] > cx[idx].min() - R) & (occ[:, 0] < cx[idx].max() + R) &
               (occ[:, 1] > cy[idx].min() - R) & (occ[:, 1] < cy[idx].max() + R))
        if not box.any():
            return hit
        occ = occ[box]
        dx = occ[None, :, 0] - cx[idx, None]
        dy = occ[None, :, 1] - cy[idx, None]
        lx = dx * c[idx, None] + dy * s[idx, None]
        ly = -dx * s[idx, None] + dy * c[idx, None]
        inside = (np.abs(lx) < self.hl) & (np.abs(ly) < self.hw)
        hit[idx] = inside.any(1)
        return hit

    def free(self, poses):
        return not self.collides(poses).any()


class EscapeChecker:
    """Strict checker everywhere, but within `radius` of a start pose that is
    already inside the safety margin, only require the bare footprint to be
    free - so the car can creep away from an obstacle it stopped next to."""

    def __init__(self, strict, relaxed, start, radius):
        self.strict, self.relaxed = strict, relaxed
        self.start = np.asarray(start[:2])
        self.radius = radius

    def collides(self, poses):
        poses = np.atleast_2d(poses)
        hit = self.strict.collides(poses)
        near = np.hypot(poses[:, 0] - self.start[0], poses[:, 1] - self.start[1]) < self.radius
        m = hit & near
        if m.any():
            hit[m] = self.relaxed.collides(poses[m])
        return hit

    def free(self, poses):
        return not self.collides(poses).any()
