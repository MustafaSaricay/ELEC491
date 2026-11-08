"""Random obstacle layouts (simulator side - uses ground truth).

world.obstacle_mode:
  fixed   use world.obstacles as written
  circle  posts / cans:   radius in world.circle_radius
  line    thin walls:     length in world.line_length, thickness world.line_thickness,
                          random orientation (flat faces -> ultrasonic specular misses!)
  random  each obstacle is randomly a circle, a box (world.box_size) or a line

Obstacles keep clear of the start pose, the bay (plus its straight approach) and
the summon pickup poses. A layout is accepted only if the planner, on a map of
the true obstacles, can do both legs (start -> bay, bay -> pickup); otherwise it
is re-drawn, up to world.layout_retries times.
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .config import Config

MODES = ("fixed", "circle", "line", "random")


def _random_obstacle(kind, wc, rng):
    h = float(rng.uniform(*wc.obstacle_height))
    x = float(rng.uniform(0.1, wc.width - 0.1))
    y = float(rng.uniform(0.1, wc.height - 0.1))
    if kind == "circle":
        return {"type": "circle", "x": x, "y": y, "r": float(rng.uniform(*wc.circle_radius)), "h": h}
    if kind == "line":
        return {"type": "box", "x": x, "y": y, "l": float(rng.uniform(*wc.line_length)),
                "w": wc.line_thickness, "yaw_deg": float(rng.uniform(0, 180)), "h": h}
    return {"type": "box", "x": x, "y": y, "l": float(rng.uniform(*wc.box_size)),
            "w": float(rng.uniform(*wc.box_size)), "yaw_deg": float(rng.uniform(0, 180)), "h": h}


def _keepout_polys(cfg: Config):
    """Rectangles (car footprints + margin) obstacles must not touch."""
    wc, vc = cfg.world, cfg.vehicle
    m = wc.keepout_margin
    polys = []
    poses = [g.deg_pose(wc.start)]
    for c in cfg.mission.commands:
        if c.get("cmd") == "summon":
            poses.append(g.deg_pose(c["pose"]))
    for p in poses:
        polys.append(g.rect_corners(*p, vc.length + 2 * m, vc.width + 2 * m))
    # the bay itself plus the straight run-in in front of it
    b = wc.bay
    bp = np.array([b["x"], b["y"], math.radians(b["yaw_deg"])])
    run_in = cfg.planner.final_straight + vc.length
    depth = b["depth"] + run_in
    cx = bp[0] - math.cos(bp[2]) * run_in / 2
    cy = bp[1] - math.sin(bp[2]) * run_in / 2
    polys.append(g.rect_corners(cx, cy, bp[2], depth + 2 * m, max(b["width"], vc.width) + 2 * m))
    return polys


def _overlaps(ob, polys, ob_w):
    from .world import Obstacle
    o = Obstacle.from_dict(ob)
    if o.kind == "box":
        c = o.corners()
        if c[:, 0].min() < 0 or c[:, 1].min() < 0 or c[:, 0].max() > ob_w[0] or c[:, 1].max() > ob_w[1]:
            return True
    return any(o.distance_to_polygon(p) <= 0.0 for p in polys)


def _solvable(cfg: Config, obstacles, timing):
    from .mapping import OccupancyGrid
    from .planning.hybrid_astar import HybridAStar
    from .world import World
    wc = cfg.world
    off = cfg.vehicle.center_offset
    planner = HybridAStar(cfg.vehicle, cfg.planner)
    start = g.center_to_rear(g.deg_pose(wc.start), off)
    bay = g.center_to_rear(np.array([wc.bay["x"], wc.bay["y"], math.radians(wc.bay["yaw_deg"])]), off)
    legs = [(start, bay, timing == "before_park")]
    for c in cfg.mission.commands:
        if c.get("cmd") == "summon":
            legs.append((bay, g.center_to_rear(g.deg_pose(c["pose"]), off), True))
    for a, b, with_obstacles in legs:
        tmp = cfg.copy()
        tmp.world.obstacles = obstacles if with_obstacles else []
        tmp.world.obstacle_mode = "fixed"
        grid = OccupancyGrid(wc.width, wc.height, cfg.mapping)
        grid.mark_obstacles(World(tmp.world))
        path, _ = planner.plan(a, b, grid)
        if path is None:
            return False
    return True


def generate(cfg: Config, rng: np.random.Generator):
    """Return the obstacle list for this run (dicts, as in world.obstacles)."""
    wc = cfg.world
    if wc.obstacle_mode not in MODES:
        raise ValueError(f"world.obstacle_mode must be one of {MODES}, got '{wc.obstacle_mode}'")
    if wc.obstacle_mode == "fixed":
        return list(wc.obstacles)
    polys = _keepout_polys(cfg)
    last = []
    for _ in range(max(1, wc.layout_retries)):
        obs = []
        tries = 0
        while len(obs) < wc.n_obstacles and tries < 500:
            tries += 1
            kind = wc.obstacle_mode if wc.obstacle_mode != "random" else \
                ("circle", "box", "line")[int(rng.integers(3))]
            ob = _random_obstacle(kind, wc, rng)
            if not _overlaps(ob, polys, (wc.width, wc.height)):
                obs.append(ob)
        last = obs
        if _solvable(cfg, obs, wc.obstacle_timing):
            return obs
    return last          # no solvable layout found: run it anyway (planner will report it)
