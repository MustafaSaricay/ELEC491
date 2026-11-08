"""Ground-truth testbed: walls, obstacles (with heights), parking bay."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as g
from .config import WorldCfg


SIDES = ("bottom", "right", "top", "left")


def wall_sides(spec):
    """world.walls -> tuple of sides. Accepts True/"all", False/"none"/[], or a list."""
    if spec is True or spec == "all":
        return SIDES
    if spec is False or spec in ("none", None):
        return ()
    if isinstance(spec, str):
        spec = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [x for x in spec if x not in SIDES]
    if bad:
        raise ValueError(f"unknown wall side(s) {bad}; use {SIDES}")
    return tuple(x for x in SIDES if x in spec)


def wall_segments(width, height, sides):
    w, h = width, height
    seg = {"bottom": [0, 0, w, 0], "right": [w, 0, w, h], "top": [w, h, 0, h], "left": [0, h, 0, 0]}
    return np.array([seg[x] for x in sides], float).reshape(-1, 4)


@dataclass
class Obstacle:
    kind: str            # "circle" | "box"
    x: float
    y: float
    h: float             # height [m] - LiDAR only sees it if h > mount height
    r: float = 0.0
    l: float = 0.0
    w: float = 0.0
    yaw: float = 0.0
    spawn_t: float = 0.0

    @staticmethod
    def from_dict(d: dict, t: float = 0.0) -> "Obstacle":
        kind = d.get("type", "circle")
        if kind == "circle":
            return Obstacle("circle", d["x"], d["y"], d.get("h", 0.1), r=d["r"], spawn_t=t)
        if kind == "box":
            return Obstacle("box", d["x"], d["y"], d.get("h", 0.1), l=d["l"], w=d["w"],
                            yaw=math.radians(d.get("yaw_deg", 0.0)), spawn_t=t)
        raise ValueError(f"unknown obstacle type {kind}")

    def corners(self):
        return g.rect_corners(self.x, self.y, self.yaw, self.l, self.w)

    def distance_to_polygon(self, poly) -> float:
        if self.kind == "circle":
            # distance from circle centre to polygon, minus radius
            if g.point_in_convex(poly, (self.x, self.y)):
                return 0.0
            d = g.point_segment_distance(np.array([[self.x, self.y]]),
                                         np.hstack([poly, np.roll(poly, -1, 0)])).min()
            return max(0.0, float(d) - self.r)
        return g.polygon_distance(poly, self.corners())

    def contains(self, px, py):
        """Boolean mask for points inside the obstacle footprint."""
        if self.kind == "circle":
            return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.r ** 2
        loc = g.to_local(np.stack([px.ravel(), py.ravel()], 1), (self.x, self.y, self.yaw))
        m = (np.abs(loc[:, 0]) <= self.l / 2) & (np.abs(loc[:, 1]) <= self.w / 2)
        return m.reshape(np.shape(px))


class World:
    def __init__(self, cfg: WorldCfg):
        self.cfg = cfg
        self.width, self.height = cfg.width, cfg.height
        self.wall_sides = wall_sides(cfg.walls)
        self.walls = wall_segments(self.width, self.height, self.wall_sides)
        self.obstacles: list[Obstacle] = [Obstacle.from_dict(d) for d in cfg.obstacles]
        self._cache_height = None

    # ---- bay ------------------------------------------------------------ #
    @property
    def bay_pose(self):
        b = self.cfg.bay
        return np.array([b["x"], b["y"], math.radians(b["yaw_deg"])])

    def bay_lines(self):
        """The two side lines and the end line of the bay as (3, 4) segments."""
        b = self.cfg.bay
        x, y, th = self.bay_pose
        c = g.rect_corners(x, y, th, b["depth"], b["width"])
        # corners: [front-left, rear-left, rear-right, front-right] in bay frame
        return np.array([[*c[1], *c[0]], [*c[2], *c[3]], [*c[0], *c[3]]])

    # ---- obstacles ------------------------------------------------------ #
    def add_obstacle(self, d: dict, t: float = 0.0):
        self.obstacles.append(Obstacle.from_dict(d, t))
        self._cache_height = None

    def _lidar_geometry(self, height):
        if self._cache_height == height:
            return self._segs, self._circles
        segs = [self.walls] if self.cfg.wall_height > height else []
        circles = []
        for o in self.obstacles:
            if o.h <= height:
                continue                      # below the scan plane: invisible
            if o.kind == "circle":
                circles.append([o.x, o.y, o.r])
            else:
                c = o.corners()
                segs.append(np.hstack([c, np.roll(c, -1, 0)]))
        self._segs = np.vstack(segs) if segs else np.zeros((0, 4))
        self._circles = np.array(circles) if circles else np.zeros((0, 3))
        self._cache_height = height
        return self._segs, self._circles

    def raycast(self, ox, oy, ang, height):
        segs, circles = self._lidar_geometry(height)
        return np.minimum(g.ray_segments(ox, oy, ang, segs), g.ray_circles(ox, oy, ang, circles))

    def _echo_geometry(self):
        """All reflecting objects: segments/circles with an owner-object index and height."""
        key = len(self.obstacles)
        if getattr(self, "_echo_key", None) == key:
            return self._echo
        segs, seg_obj, circles, circ_obj, heights = [self.walls], [0] * len(self.walls), [], [], [self.cfg.wall_height]
        for i, o in enumerate(self.obstacles, start=1):
            heights.append(o.h)
            if o.kind == "circle":
                circles.append([o.x, o.y, o.r])
                circ_obj.append(i)
            else:
                c = o.corners()
                segs.append(np.hstack([c, np.roll(c, -1, 0)]))
                seg_obj += [i] * 4
        segs = np.vstack(segs)
        circles = np.array(circles) if circles else np.zeros((0, 3))
        self._echo = (segs, np.array(seg_obj, int), circles, np.array(circ_obj, int), np.array(heights))
        self._echo_key = key
        return self._echo

    def ultrasonic_echo(self, ox, oy, yaw, ucfg):
        """True first-echo range of an HC-SR04 at (ox, oy) pointing along yaw.
        Low objects are only seen once they are far enough to enter the vertical cone."""
        segs, seg_obj, circles, circ_obj, heights = self._echo_geometry()
        excluded = np.zeros(len(heights), bool)
        tan_v = math.tan(math.radians(ucfg.vertical_half_deg))
        for _ in range(len(heights) + 1):
            sm, cm = ~excluded[seg_obj], ~excluded[circ_obj] if len(circ_obj) else np.zeros(0, bool)
            s, c = segs[sm], circles[cm]
            r, owner = g.cone_echo(ox, oy, yaw, math.radians(ucfg.half_angle_deg), ucfg.n_rays, s, c,
                                   math.radians(ucfg.max_incidence_deg))
            if owner < 0:
                return math.inf
            obj = seg_obj[sm][owner] if owner < len(s) else circ_obj[cm][owner - len(s)]
            if heights[obj] >= ucfg.mount_height - r * tan_v:
                return r
            excluded[obj] = True             # under the beam at this range: look past it
        return math.inf

    # ---- ground truth collision ---------------------------------------- #
    def _side_gaps(self, poly):
        return {"left": poly[:, 0].min(), "bottom": poly[:, 1].min(),
                "right": self.width - poly[:, 0].max(), "top": self.height - poly[:, 1].max()}

    def footprint_clearance(self, poly) -> float:
        """Min distance from the vehicle polygon to the PHYSICAL walls and the
        obstacles (0 = contact). Missing walls are handled by outside_testbed()."""
        gaps = self._side_gaps(poly)
        best = max(0.0, float(min([gaps[s] for s in self.wall_sides], default=math.inf)))
        for o in self.obstacles:
            best = min(best, o.distance_to_polygon(poly))
            if best <= 0.0:
                break
        return best

    def outside_testbed(self, poly) -> bool:
        """True when part of the car has crossed an edge that has no wall."""
        gaps = self._side_gaps(poly)
        return any(gaps[s] < 0 for s in SIDES if s not in self.wall_sides)
