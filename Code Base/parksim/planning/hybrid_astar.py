"""Hybrid A* [Dolgov et al. 2010] over (x, y, yaw, direction) with Reeds-Shepp
analytic expansion to the exact goal pose.

* Motion primitives: arcs of length ``step`` at ``n_steer`` steering angles,
  forward and reverse, limited to ``steer_margin * max_steer`` so the tracker
  keeps some steering authority.
* Heuristic: obstacle-aware 2-D Dijkstra distance from the goal.
* Cost: length (reverse weighted), direction switches, steering effort/changes.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time

import numpy as np

from .. import geometry as g
from . import reeds_shepp as rs
from .collision import EscapeChecker, FootprintChecker
from .path import Path


class _Node:
    __slots__ = ("x", "y", "yaw", "dir", "steer", "g", "parent", "traj")

    def __init__(self, x, y, yaw, d, steer, cost, parent, traj):
        self.x, self.y, self.yaw, self.dir, self.steer = x, y, yaw, d, steer
        self.g, self.parent, self.traj = cost, parent, traj


class HybridAStar:
    def __init__(self, vehicle_cfg, planner_cfg):
        self.v = vehicle_cfg
        self.c = planner_cfg

    @property
    def max_steer(self):
        return self.v.max_steer * self.c.steer_margin

    @property
    def radius(self):
        return self.v.wheelbase / math.tan(self.max_steer)

    # ------------------------------------------------------------------ #
    def plan(self, start, goal, grid, margin=None):
        t0 = time.perf_counter()
        c = self.c
        margin = c.safety_margin if margin is None else margin
        chk = FootprintChecker(self.v, grid, margin)
        info = {"expansions": 0, "reason": "", "time": 0.0, "escape": False}
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        if chk.collides(start[None]).any():
            relaxed = FootprintChecker(self.v, grid, -grid.res / 2.0)
            chk = EscapeChecker(chk, relaxed, start, c.escape_radius)
            info["escape"] = True
        if chk.collides(goal[None]).any():
            info["reason"] = "goal pose is blocked"
            info["time"] = time.perf_counter() - t0
            return None, info

        targets = self._targets(goal, chk)
        h2d, hgrid = self._holonomic(goal, grid)
        steers = np.linspace(-self.max_steer, self.max_steer, c.n_steer)
        yaw_res = 2 * math.pi / c.yaw_bins
        counter = itertools.count()

        def key(x, y, yaw, d):
            return (int(round(x / c.xy_res)), int(round(y / c.xy_res)),
                    int(round(g.wrap(yaw) / yaw_res)) % c.yaw_bins, d)

        def heur(x, y):
            ix, iy = hgrid(x, y)
            if 0 <= ix < h2d.shape[1] and 0 <= iy < h2d.shape[0] and np.isfinite(h2d[iy, ix]):
                return max(h2d[iy, ix], math.hypot(goal[0] - x, goal[1] - y))
            return 100.0 + math.hypot(goal[0] - x, goal[1] - y)

        # Fail fast: if even a point-sized robot (the coarse 2-D grid) cannot get
        # from anywhere near the start to the goal, no car path exists either.
        ix0, iy0 = hgrid(start[0], start[1])
        rad = int(math.ceil(0.15 / c.xy_res))
        near = h2d[max(iy0 - rad, 0):iy0 + rad + 1, max(ix0 - rad, 0):ix0 + rad + 1]
        if not np.isfinite(near).any():
            info["reason"] = "goal unreachable (no 2-D route)"
            info["time"] = time.perf_counter() - t0
            return None, info

        prims = [(steer, d) for steer in steers for d in (1, -1)]
        prim_k = np.array([math.tan(st) / self.v.wheelbase for st, _ in prims])
        prim_d = np.array([float(d) for _, d in prims])
        root = _Node(start[0], start[1], start[2], 0, 0.0, 0.0, None, None)
        open_ = [(heur(start[0], start[1]), next(counter), root)]
        best = {key(*start, 0): 0.0}
        closed = set()
        while open_ and info["expansions"] < c.max_expansions:
            _, _, n = heapq.heappop(open_)
            k = key(n.x, n.y, n.yaw, n.dir)
            if k in closed:
                continue
            closed.add(k)
            info["expansions"] += 1

            hn = heur(n.x, n.y)
            # analytic shot: often near the goal, rarely far away (it is the costly part)
            interval = c.rs_interval if hn < c.rs_near_dist else 10 * c.rs_interval
            if info["expansions"] % interval == 1 or hn < 0.6:
                shot = self._rs_shot(n, targets, chk)
                if shot is not None:
                    path = self._reconstruct(n, shot, start)
                    info["time"] = time.perf_counter() - t0
                    info["length"] = path.length
                    info["cusps"] = path.n_cusps
                    return path, info

            trajs = self._arcs(n.x, n.y, n.yaw, prim_k, prim_d)                     # (P, m, 4)
            P, m = trajs.shape[:2]
            hits = chk.collides(trajs[:, :, :3].reshape(-1, 3)).reshape(P, m).any(1)  # one batched check
            for (steer, d), traj, hit in zip(prims, trajs, hits):
                if hit:
                    continue
                x, y, yaw = traj[-1, :3]
                k2 = key(x, y, yaw, d)
                if k2 in closed:
                    continue
                cost = n.g + c.step * (c.reverse_cost if d < 0 else 1.0)
                if n.dir != 0 and d != n.dir:
                    cost += c.switch_cost
                cost += c.steer_cost * abs(steer) + c.steer_change_cost * abs(steer - n.steer)
                if cost < best.get(k2, math.inf):
                    best[k2] = cost
                    child = _Node(x, y, yaw, d, steer, cost, n, traj)
                    heapq.heappush(open_, (cost + heur(x, y), next(counter), child))
        info["reason"] = "search exhausted" if not open_ else "expansion limit"
        info["time"] = time.perf_counter() - t0
        return None, info

    # ------------------------------------------------------------------ #
    def _arcs(self, x, y, yaw, k, d):
        """All motion primitives at once: curvatures k (P,), directions d (P,)
        -> (P, m, 4) poses [x, y, yaw, dir] sampled every sample_ds."""
        c = self.c
        m = max(1, int(math.ceil(c.step / c.sample_ds)))
        s = np.linspace(c.step / m, c.step, m)[None, :] * d[:, None]           # (P, m) signed arc length
        kk = k[:, None]
        straight = np.abs(kk) < 1e-9
        ks = np.where(straight, 1.0, kk)
        ths = yaw + s * kk
        xs = np.where(straight, x + s * math.cos(yaw), x + (np.sin(ths) - math.sin(yaw)) / ks)
        ys = np.where(straight, y + s * math.sin(yaw), y - (np.cos(ths) - math.cos(yaw)) / ks)
        return np.stack([xs, ys, g.wrap(ths), np.broadcast_to(d[:, None], s.shape)], 2)

    def _arc(self, x, y, yaw, steer, d):
        c = self.c
        n = max(1, int(math.ceil(c.step / c.sample_ds)))
        s = np.linspace(c.step / n, c.step, n) * d
        k = math.tan(steer) / self.v.wheelbase
        if abs(k) < 1e-9:
            xs, ys, ths = x + s * math.cos(yaw), y + s * math.sin(yaw), np.full(n, yaw)
        else:
            ths = yaw + s * k
            xs = x + (np.sin(ths) - math.sin(yaw)) / k
            ys = y - (np.cos(ths) - math.cos(yaw)) / k
        return np.stack([xs, ys, g.wrap(ths), np.full(n, float(d))], 1)

    def _targets(self, goal, chk):
        """Analytic-expansion targets: a pre-goal pose `final_straight` before the
        goal (arriving forwards or in reverse) plus the straight run into it.
        A straight final approach lets the tracker settle heading before it stops."""
        c = self.c
        out = []
        if c.final_straight > 0:
            n = max(1, int(math.ceil(c.final_straight / c.sample_ds)))
            s = np.linspace(c.final_straight / n, c.final_straight, n)
            cg, sg = math.cos(goal[2]), math.sin(goal[2])
            for d in (1, -1):
                pre = np.array([goal[0] - d * c.final_straight * cg, goal[1] - d * c.final_straight * sg, goal[2]])
                straight = np.stack([pre[0] + d * s * cg, pre[1] + d * s * sg,
                                     np.full(n, goal[2]), np.full(n, float(d))], 1)
                if not chk.collides(np.vstack([pre[None], straight[:, :3]])).any():
                    out.append((pre, straight, d))
        out.append((goal, None, 0))         # direct shot as the last resort
        return out

    def _rs_shot(self, n, targets, chk):
        c = self.c
        best = None
        for target, straight, sdir in targets:
            if straight is None and best is not None:
                break                        # prefer straight-in approaches
            cands = rs.paths((n.x, n.y, n.yaw), target, self.radius)

            def cost(p):
                l = sum(abs(L) * (c.reverse_cost if L < 0 else 1.0) for _, L in p)
                dirs = [1 if L > 0 else -1 for _, L in p]
                if n.dir != 0:
                    dirs = [n.dir] + dirs
                if sdir != 0:
                    dirs = dirs + [sdir]
                    l += c.final_straight * (c.reverse_cost if sdir < 0 else 1.0)
                return l + c.switch_cost * sum(a != b for a, b in zip(dirs, dirs[1:]))

            for segs in sorted(cands[: 3 * c.rs_candidates], key=cost)[: c.rs_candidates]:
                k = cost(segs)
                if best is not None and k >= best[0]:
                    break
                traj = rs.sample((n.x, n.y, n.yaw), segs, self.radius, c.sample_ds)
                if not chk.collides(traj[:, :3]).any():
                    traj[:, 2] = g.wrap(traj[:, 2])
                    if straight is not None:
                        traj = np.vstack([traj, straight])
                    best = (k, traj)
                    break
        return None if best is None else best[1]

    def _reconstruct(self, n, shot, start):
        parts = [shot]
        while n is not None and n.traj is not None:
            parts.append(n.traj)
            n = n.parent
        parts.reverse()
        body = np.vstack(parts)
        first = np.array([[start[0], start[1], start[2], body[0, 3]]])
        return Path.from_array(np.vstack([first, body]))

    def _holonomic(self, goal, grid):
        """8-connected Dijkstra on a coarse grid from the goal (rear-axle position)."""
        res = self.c.xy_res
        nx = int(math.ceil(grid.width / res)) + 1
        ny = int(math.ceil(grid.height / res)) + 1
        xs = (np.arange(nx)) * res
        ys = (np.arange(ny)) * res
        X, Y = np.meshgrid(xs, ys)
        clearance = min(self.v.width / 2.0, self.v.rear_overhang)
        blocked = grid.distance_at(X, Y) < clearance
        dist = np.full((ny, nx), np.inf)

        def cell(x, y):
            return int(round(x / res)), int(round(y / res))

        gx, gy = cell(goal[0], goal[1])
        gx, gy = min(max(gx, 0), nx - 1), min(max(gy, 0), ny - 1)
        dist[gy, gx] = 0.0
        pq = [(0.0, gx, gy)]
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist[y, x]:
                continue
            for mx, my, w in moves:
                x2, y2 = x + mx, y + my
                if 0 <= x2 < nx and 0 <= y2 < ny and not blocked[y2, x2]:
                    nd = d + w * res
                    if nd < dist[y2, x2]:
                        dist[y2, x2] = nd
                        heapq.heappush(pq, (nd, x2, y2))
        return dist, cell
