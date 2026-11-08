"""Planned path: sampled poses with a driving direction (+1 fwd, -1 rev)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Path:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    dir: np.ndarray            # direction used to *arrive* at each point (dir[0] = dir[1])

    @staticmethod
    def from_array(a):
        a = np.asarray(a, float)
        d = a[:, 3].copy()
        if len(d) > 1:
            d[0] = d[1]
        return Path(a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy(), d)

    def __len__(self):
        return len(self.x)

    @property
    def xy(self):
        return np.stack([self.x, self.y], 1)

    @property
    def poses(self):
        return np.stack([self.x, self.y, self.yaw], 1)

    @property
    def s(self):
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))])

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def n_cusps(self):
        return int(np.count_nonzero(np.diff(self.dir) != 0))

    def segments(self):
        """Split at cusps. Each segment starts at the previous cusp point."""
        segs, start = [], 0
        for i in range(1, len(self.x) - 1):
            if self.dir[i + 1] != self.dir[i]:
                segs.append((start, i, self.dir[i]))
                start = i
        segs.append((start, len(self.x) - 1, self.dir[-1]))
        out = []
        for a, b, d in segs:
            sl = slice(a, b + 1)
            out.append(Path(self.x[sl].copy(), self.y[sl].copy(), self.yaw[sl].copy(),
                            np.full(b - a + 1, d)))
        return out
