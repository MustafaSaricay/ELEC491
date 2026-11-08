"""Log-odds occupancy grid built from LiDAR scans at the *estimated* pose, so
localization error corrupts the map just like on the real car."""
from __future__ import annotations

import math

import numpy as np

from .config import MapCfg
from .geometry import point_segment_distance


class OccupancyGrid:
    def __init__(self, width, height, cfg: MapCfg, walls=None):
        """walls: (M, 4) physical wall segments the car knows about. Default:
        all four testbed edges. The testbed edge is always kept as a
        (virtual) occupied ring so plans stay inside the testbed."""
        self.cfg = cfg
        if walls is None:
            walls = np.array([[0, 0, width, 0], [width, 0, width, height],
                              [width, height, 0, height], [0, height, 0, 0]], float)
        self.walls = np.asarray(walls, float).reshape(-1, 4)
        self._near_mask = None
        self.res = cfg.resolution
        self.width, self.height = width, height
        self.margin = 2 * self.res
        self.ox = self.oy = -self.margin
        self.nx = int(math.ceil((width + 2 * self.margin) / self.res))
        self.ny = int(math.ceil((height + 2 * self.margin) / self.res))
        self.L = np.zeros((self.ny, self.nx))
        xs, ys = self.cell_centers()
        # static ring of cells just outside the known walls
        outside = (xs < 0) | (ys < 0) | (xs > width) | (ys > height)
        ring = outside & ((xs > -self.res) & (ys > -self.res) &
                          (xs < width + self.res) & (ys < height + self.res))
        self.static = ring
        self.L[ring] = cfg.l_max
        self._occ_cache = None
        self._edt_cache = None
        self.version = 0

    def cell_centers(self):
        xs = self.ox + (np.arange(self.nx) + 0.5) * self.res
        ys = self.oy + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def to_idx(self, x, y):
        ix = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        iy = np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        return ix, iy

    def _inside(self, ix, iy):
        return (ix >= 0) & (iy >= 0) & (ix < self.nx) & (iy < self.ny)

    def _near_wall(self, x, y):
        """Points this close to a PHYSICAL wall are the wall, not an obstacle.
        Walls never move, so this is a lookup in a mask computed once per grid
        (cell resolution; points outside the grid count as near a wall)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(self.walls) == 0:
            return np.zeros(x.shape, bool)
        if self._near_mask is None:
            xs, ys = self.cell_centers()
            pts = np.stack([xs.ravel(), ys.ravel()], 1)
            d = point_segment_distance(pts, self.walls).min(1).reshape(xs.shape)
            self._near_mask = d < self.cfg.ignore_near_walls
        ix, iy = self.to_idx(x, y)
        ok = self._inside(ix, iy)
        out = np.ones(x.shape, bool)
        out[ok] = self._near_mask[iy[ok], ix[ok]]
        return out

    # ------------------------------------------------------------------ #
    def update_scan(self, sensor_pose, angles, ranges):
        c = self.cfg
        sx, sy, sth = sensor_pose
        valid = np.isfinite(ranges)
        free_r = np.where(valid, ranges - self.res, c.free_ray_max)
        free_r = np.minimum(free_r, c.free_ray_max)
        wa = sth + angles
        # free space: sample each ray every half cell
        step = self.res * 0.5
        k = np.arange(0, c.free_ray_max, step)
        mask = k[None, :] < free_r[:, None]
        fx = (sx + np.cos(wa)[:, None] * k[None, :])[mask]
        fy = (sy + np.sin(wa)[:, None] * k[None, :])[mask]
        ix, iy = self.to_idx(fx, fy)
        ok = self._inside(ix, iy)
        free = np.unique(iy[ok] * self.nx + ix[ok])
        # hits
        hx = sx + np.cos(wa[valid]) * ranges[valid]
        hy = sy + np.sin(wa[valid]) * ranges[valid]
        keep = ~self._near_wall(hx, hy)
        ix, iy = self.to_idx(hx[keep], hy[keep])
        ok = self._inside(ix, iy)
        hits = np.unique(iy[ok] * self.nx + ix[ok])
        self.L.flat[np.setdiff1d(free, hits, assume_unique=True)] += c.l_miss
        self.L.flat[hits] += c.l_hit
        np.clip(self.L, c.l_min, c.l_max, out=self.L)
        self.L[self.static] = c.l_max
        self._occ_cache = self._edt_cache = None
        self.version += 1

    def update_cone(self, sensor_pose, half_angle, r, mark_hit=True):
        """Ultrasonic inverse sensor model: the cone before the echo is free, and
        the arc at the echo range is (weakly) occupied, since the echo could
        have come from anywhere across the cone. No echo -> no update (the
        surface may simply have reflected the ping away). mark_hit=False when
        the echo is explained by a known wall (only clear free space)."""
        if not np.isfinite(r):
            return
        c, res = self.cfg, self.res
        sx, sy, syaw = sensor_pose
        reach = r + res
        ix0, iy0 = self.to_idx(sx - reach, sy - reach)
        ix1, iy1 = self.to_idx(sx + reach, sy + reach)
        ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
        ix1, iy1 = min(int(ix1), self.nx - 1), min(int(iy1), self.ny - 1)
        if ix1 < ix0 or iy1 < iy0:
            return
        xs = self.ox + (np.arange(ix0, ix1 + 1) + 0.5) * res
        ys = self.oy + (np.arange(iy0, iy1 + 1) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X - sx, Y - sy
        d = np.hypot(dx, dy)
        bearing = (np.arctan2(dy, dx) - syaw + math.pi) % (2 * math.pi) - math.pi
        incone = np.abs(bearing) <= half_angle
        free = incone & (d < r - res)
        hit = incone & (np.abs(d - r) <= 0.75 * res) & ~self._near_wall(X, Y)
        win = self.L[iy0:iy1 + 1, ix0:ix1 + 1]
        win[free] += c.us_l_miss
        if mark_hit:
            win[hit] += c.us_l_hit
        np.clip(win, c.l_min, c.l_max, out=win)
        self.L[self.static] = c.l_max
        self._occ_cache = self._edt_cache = None
        self.version += 1

    def mark_points(self, xs, ys, logodds=None):
        """Raise cells to at least `logodds` (default: certain). Used for
        whatever triggered an emergency stop, so the next plan goes around it.
        Points at a known wall are skipped."""
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        keep = ~self._near_wall(xs, ys)
        ix, iy = self.to_idx(xs[keep], ys[keep])
        ok = self._inside(ix, iy)
        if ok.any():
            v = self.cfg.l_max if logodds is None else logodds
            self.L[iy[ok], ix[ok]] = np.maximum(self.L[iy[ok], ix[ok]], v)
            self._occ_cache = self._edt_cache = None
            self.version += 1

    def mark_obstacles(self, world, min_height=0.0):
        """Rasterise ground-truth obstacles (for planner-only experiments)."""
        xs, ys = self.cell_centers()
        for o in world.obstacles:
            if o.h > min_height:
                m = o.contains(xs, ys)
                if not m.any():   # tiny obstacle: mark its centre cell
                    ix, iy = self.to_idx(o.x, o.y)
                    m = np.zeros_like(xs, bool)
                    m[iy, ix] = True
                self.L[m] = self.cfg.l_max
        self._occ_cache = self._edt_cache = None
        self.version += 1

    # ------------------------------------------------------------------ #
    def occupied_mask(self):
        return self.L > self.cfg.occ_threshold

    def occupied_points(self):
        if self._occ_cache is None:
            xs, ys = self.cell_centers()
            m = self.occupied_mask()
            self._occ_cache = np.stack([xs[m], ys[m]], 1)
        return self._occ_cache

    def distance_field(self):
        """Euclidean distance from every cell centre to the nearest occupied cell."""
        if self._edt_cache is None:
            # exact separable squared EDT (in cells): columns, then rows
            occ = self.occupied_mask()
            big = float(self.nx ** 2 + self.ny ** 2)
            iy = np.arange(self.ny)
            dy2 = (iy[:, None] - iy[None, :]) ** 2                       # (ny, ny)
            col = np.where(occ[None, :, :], dy2[:, :, None], big).min(1)  # (ny, nx)
            ix = np.arange(self.nx)
            dx2 = (ix[:, None] - ix[None, :]) ** 2                       # (nx, nx)
            d2 = (col[:, None, :] + dx2[None, :, :]).min(2)              # (ny, nx)
            self._edt_cache = np.sqrt(d2) * self.res
        return self._edt_cache

    def distance_at(self, x, y):
        edt = self.distance_field()
        ix, iy = self.to_idx(x, y)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        return edt[iy, ix]

    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))
