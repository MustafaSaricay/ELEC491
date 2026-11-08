"""Small 2-D geometry helpers (vectorised with numpy where it matters)."""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % TWO_PI - math.pi if isinstance(a, np.ndarray) \
        else (a + math.pi) % TWO_PI - math.pi


def rear_to_center(pose, offset):
    x, y, th = pose
    return np.array([x + offset * math.cos(th), y + offset * math.sin(th), th])


def center_to_rear(pose, offset):
    x, y, th = pose
    return np.array([x - offset * math.cos(th), y - offset * math.sin(th), th])


def mount_pose(rear_pose, s):
    """World pose of a body-mounted sensor {"x", "y", "yaw_deg"} (rear-axle frame)."""
    x, y, th = rear_pose
    c, si = math.cos(th), math.sin(th)
    return (x + s["x"] * c - s["y"] * si, y + s["x"] * si + s["y"] * c, th + math.radians(s["yaw_deg"]))


def deg_pose(p):
    """[x, y, yaw_deg] -> np.array([x, y, yaw_rad])."""
    return np.array([float(p[0]), float(p[1]), math.radians(float(p[2]))])


def rect_corners(cx, cy, yaw, length, width):
    """Corners (4, 2) of a rectangle centred at (cx, cy)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def to_local(points, pose):
    """World points (N, 2) -> frame of pose (x, y, yaw)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    d = np.asarray(points) - np.asarray(pose[:2])
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)


def to_world(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    p = np.asarray(points)
    return np.stack([pose[0] + p[:, 0] * c - p[:, 1] * s,
                     pose[1] + p[:, 0] * s + p[:, 1] * c], axis=1)


# --------------------------------------------------------------------------- #
# Ray casting
# --------------------------------------------------------------------------- #
def ray_segments(ox, oy, ang, segs):
    """Distance along rays to the nearest segment. ox/oy/ang: (N,), segs: (M, 4)."""
    n = len(ang)
    out = np.full(n, np.inf)
    if len(segs) == 0:
        return out
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    px, py = segs[None, :, 0], segs[None, :, 1]
    ex, ey = (segs[:, 2] - segs[:, 0])[None, :], (segs[:, 3] - segs[:, 1])[None, :]
    wx, wy = px - np.asarray(ox)[:, None], py - np.asarray(oy)[:, None]
    denom = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / denom
        u = (wx * dy - wy * dx) / denom
    ok = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
    t = np.where(ok, t, np.inf)
    return np.minimum(out, t.min(axis=1))


def ray_circles(ox, oy, ang, circles):
    """circles: (K, 3) [cx, cy, r]."""
    n = len(ang)
    if len(circles) == 0:
        return np.full(n, np.inf)
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    fx = np.asarray(ox)[:, None] - circles[None, :, 0]
    fy = np.asarray(oy)[:, None] - circles[None, :, 1]
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - circles[None, :, 2] ** 2
    disc = b * b - c
    with np.errstate(invalid="ignore"):
        t = -b - np.sqrt(disc)
    ok = (disc >= 0) & (t > 1e-9)
    return np.where(ok, t, np.inf).min(axis=1)


def ray_hits(ox, oy, ang, segs, circles):
    """Nearest hit per ray against segments (M, 4) and circles (K, 3).
    Returns (t, cos_incidence, owner) where owner < M is a segment index and
    owner >= M is circle index + M. t = inf when nothing is hit."""
    n = len(ang)
    dx, dy = np.cos(ang)[:, None], np.sin(ang)[:, None]
    ox, oy = np.broadcast_to(np.asarray(ox, float), (n,))[:, None], np.broadcast_to(np.asarray(oy, float), (n,))[:, None]
    ts, cs = [], []
    if len(segs):
        ex, ey = (segs[:, 2] - segs[:, 0])[None], (segs[:, 3] - segs[:, 1])[None]
        wx, wy = segs[None, :, 0] - ox, segs[None, :, 1] - oy
        denom = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx * ey - wy * ex) / denom
            u = (wx * dy - wy * dx) / denom
        ok = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0) & (u <= 1)
        ts.append(np.where(ok, t, np.inf))
        ln = np.hypot(ex, ey)
        cs.append(np.abs(dx * (-ey) + dy * ex) / ln)            # |d . n|
    if len(circles):
        fx, fy = ox - circles[None, :, 0], oy - circles[None, :, 1]
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - circles[None, :, 2] ** 2
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        ok = (disc >= 0) & (t > 1e-9)
        t = np.where(ok, t, np.inf)
        ts.append(t)
        with np.errstate(invalid="ignore"):
            hx, hy = fx + np.where(ok, t, 0) * dx, fy + np.where(ok, t, 0) * dy
            cs.append(np.abs(hx * dx + hy * dy) / circles[None, :, 2])
    if not ts:
        return np.full(n, np.inf), np.zeros(n), np.full(n, -1)
    T, C = np.hstack(ts), np.hstack(cs)
    k = T.argmin(1)
    rows = np.arange(n)
    return T[rows, k], np.nan_to_num(C[rows, k]), np.where(np.isfinite(T[rows, k]), k, -1)


def cone_echo(ox, oy, yaw, half_angle, n_rays, segs, circles, max_incidence):
    """First echo of an ultrasonic ping: the nearest hit, over rays spread across
    the cone, whose surface is within `max_incidence` of perpendicular.
    Returns (range, owner) - range inf / owner -1 if no echo comes back."""
    best_r, best_o = math.inf, -1
    # segments (walls, boxes): rays sampled across the cone, specular cut-off
    if len(segs):
        ang = yaw + np.linspace(-half_angle, half_angle, max(n_rays, 1))
        t, cos_inc, owner = ray_hits(ox, oy, ang, segs, np.zeros((0, 3)))
        valid = np.isfinite(t) & (cos_inc >= math.cos(max_incidence))
        if valid.any():
            i = int(np.argmin(np.where(valid, t, np.inf)))
            best_r, best_o = float(t[i]), int(owner[i])
    # circles (posts, cans): exact - a cylinder always has a face pointing back
    # at the sensor, so it echoes whenever its centre bearing is inside the cone
    M = len(segs)
    for k, (cx, cy, rho) in enumerate(circles):
        dx, dy = cx - ox, cy - oy
        d = math.hypot(dx, dy)
        if d <= rho:
            r = 0.0
        else:
            bearing = abs(wrap(math.atan2(dy, dx) - yaw))
            if bearing <= half_angle:
                r = d - rho
            elif bearing - math.asin(min(1.0, rho / d)) <= half_angle:
                # only the rim is inside the cone: echo from the cone edge ray
                edge = yaw + math.copysign(half_angle, wrap(math.atan2(dy, dx) - yaw))
                t, cos_inc, _ = ray_hits(ox, oy, np.array([edge]), np.zeros((0, 4)),
                                         np.array([[cx, cy, rho]]))
                r = float(t[0]) if np.isfinite(t[0]) and cos_inc[0] >= math.cos(max_incidence) else math.inf
            else:
                r = math.inf
        if r < best_r:
            best_r, best_o = r, M + k
    return best_r, best_o


def cone_echo_walls(ox, oy, yaw, half_angle, segs, max_incidence):
    """Exact first-echo range from flat segments (no ray sampling, no occlusion
    between segments - fine for the testbed walls, which is what the onboard
    software uses it for).

    For each segment, with d = perpendicular distance and psi = angle of a ray
    measured from the segment normal, a point on the segment echoes iff
      psi within the cone, |psi| <= max_incidence, and the point is on the segment.
    The nearest such point has the smallest |psi|; its range is d / cos(psi)."""
    if len(segs) == 0:
        return math.inf
    a = segs[:, :2]
    e = segs[:, 2:] - a
    ln = np.hypot(e[:, 0], e[:, 1])
    u = e / ln[:, None]
    n = np.stack([-u[:, 1], u[:, 0]], 1)
    d = (a[:, 0] - ox) * n[:, 0] + (a[:, 1] - oy) * n[:, 1]       # signed distance to the line
    flip = d < 0
    n[flip] *= -1.0                                                  # normal points from sensor to line
    d = np.abs(d)
    t_axis = np.stack([-n[:, 1], n[:, 0]], 1)                        # +psi direction along the line
    # along-line coordinates of the segment ends, relative to the perpendicular foot
    fx, fy = ox + d * n[:, 0], oy + d * n[:, 1]
    s0 = (a[:, 0] - fx) * t_axis[:, 0] + (a[:, 1] - fy) * t_axis[:, 1]
    s1 = (segs[:, 2] - fx) * t_axis[:, 0] + (segs[:, 3] - fy) * t_axis[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        p0, p1 = np.arctan2(np.minimum(s0, s1), d), np.arctan2(np.maximum(s0, s1), d)
    delta = (yaw - np.arctan2(n[:, 1], n[:, 0]) + math.pi) % (2 * math.pi) - math.pi
    lo = np.maximum.reduce([p0, delta - half_angle, np.full_like(d, -max_incidence)])
    hi = np.minimum.reduce([p1, delta + half_angle, np.full_like(d, max_incidence)])
    ok = (lo <= hi) & (d > 1e-9)
    if not ok.any():
        return math.inf
    psi = np.clip(0.0, lo, hi)                                       # closest to perpendicular
    r = np.where(ok, d / np.cos(psi), np.inf)
    return float(r.min())


# --------------------------------------------------------------------------- #
# Distances / intersection for the ground-truth collision checker
# --------------------------------------------------------------------------- #
def point_rect_distance(pts, cx, cy, yaw, length, width):
    """Distance from points (N, 2) to a filled rectangle (0 inside)."""
    loc = to_local(np.atleast_2d(pts), (cx, cy, yaw))
    dx = np.maximum(np.abs(loc[:, 0]) - length / 2.0, 0.0)
    dy = np.maximum(np.abs(loc[:, 1]) - width / 2.0, 0.0)
    return np.hypot(dx, dy)


def _segments_of(corners):
    return np.hstack([corners, np.roll(corners, -1, axis=0)])


def point_segment_distance(pts, segs):
    """(N, 2) points vs (M, 4) segments -> (N, M) distances."""
    a = segs[None, :, :2]
    e = segs[None, :, 2:] - segs[None, :, :2]
    q = pts[:, None, :]
    ee = np.maximum((e * e).sum(-1), 1e-18)
    t = np.clip(((q - a) * e).sum(-1) / ee, 0.0, 1.0)
    foot = a + t[..., None] * e
    return np.linalg.norm(q - foot, axis=-1)


def point_in_convex(poly, p):
    """True if point p lies inside the convex polygon (any winding)."""
    e = np.roll(poly, -1, axis=0) - poly
    d = np.asarray(p) - poly
    cross = e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0]
    return bool(np.all(cross >= 0) or np.all(cross <= 0))


def polygons_intersect(a, b):
    """Separating-axis test for two convex polygons given as (K, 2) corners."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        pa, pb = a @ normals.T, b @ normals.T
        if np.any((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))):
            return False
    return True


def polygon_distance(a, b):
    """Min distance between two convex polygons (0 if they intersect)."""
    if polygons_intersect(a, b):
        return 0.0
    d1 = point_segment_distance(a, _segments_of(b)).min()
    d2 = point_segment_distance(b, _segments_of(a)).min()
    return float(min(d1, d2))
