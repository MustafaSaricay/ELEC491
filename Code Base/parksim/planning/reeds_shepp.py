"""Reeds-Shepp shortest paths for a car that drives forwards and backwards with
a bounded turning radius [Reeds & Shepp 1990]. Formulas follow the corrected
set used by OMPL (CSC, CCC, CCCC, CCSC, CCSCC families with time-flip and
reflection symmetries).

A path is (types, lengths): types like ("L", "S", "R"), lengths in units of the
turning radius, negative = reverse.
"""
from __future__ import annotations

import math

import numpy as np

PI = math.pi
HALF_PI = 0.5 * math.pi
ZERO = 1e-10

TYPES = [
    ("L", "R", "L"), ("R", "L", "R"),
    ("L", "R", "L", "R"), ("R", "L", "R", "L"),
    ("L", "R", "S", "L"), ("R", "L", "S", "R"),
    ("L", "S", "R", "L"), ("R", "S", "L", "R"),
    ("L", "R", "S", "R"), ("R", "L", "S", "L"),
    ("R", "S", "R", "L"), ("L", "S", "L", "R"),
    ("L", "S", "R"), ("R", "S", "L"),
    ("L", "S", "L"), ("R", "S", "R"),
    ("L", "R", "S", "L", "R"), ("R", "L", "S", "R", "L"),
]


def mod2pi(x):
    v = math.fmod(x, 2 * PI)
    if v < -PI:
        v += 2 * PI
    elif v > PI:
        v -= 2 * PI
    return v


def polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def tau_omega(u, v, xi, eta, phi):
    delta = mod2pi(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = mod2pi(t1 + PI) if t2 < 0 else mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)
    return tau, omega


# ---- base formulas (return None or (t, u, v)) ------------------------------ #
def LpSpLp(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -ZERO:
        v = mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def LpSpRp(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def LpRmL(x, y, phi):
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    u1, theta = polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = mod2pi(theta + 0.5 * u + PI)
        v = mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def LpRupLumRm(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def LpRumLumRp(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def LpRmSmLm(x, y, phi):
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    rho, theta = polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = mod2pi(theta + math.atan2(r, -2.0))
        v = mod2pi(phi - HALF_PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def LpRmSmRm(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, theta = polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = mod2pi(t + HALF_PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def LpRmSLmRp(x, y, phi):
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, _ = polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


# ---- families with symmetries -------------------------------------------- #
def _four(fn, x, y, phi, tidx, build, out):
    for xx, yy, pp, flip, refl in ((x, y, phi, 1.0, False), (-x, y, -phi, -1.0, False),
                                   (x, -y, -phi, 1.0, True), (-x, -y, phi, -1.0, True)):
        r = fn(xx, yy, pp)
        if r is not None:
            lengths = tuple(flip * l for l in build(*r))
            out.append((TYPES[tidx ^ 1] if refl else TYPES[tidx], lengths))


def _all_paths(x, y, phi):
    out = []
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    # CSC
    _four(LpSpLp, x, y, phi, 14, lambda t, u, v: (t, u, v), out)
    _four(LpSpRp, x, y, phi, 12, lambda t, u, v: (t, u, v), out)
    # CCC
    _four(LpRmL, x, y, phi, 0, lambda t, u, v: (t, u, v), out)
    _four(LpRmL, xb, yb, phi, 0, lambda t, u, v: (v, u, t), out)
    # CCCC
    _four(LpRupLumRm, x, y, phi, 2, lambda t, u, v: (t, u, -u, v), out)
    _four(LpRumLumRp, x, y, phi, 2, lambda t, u, v: (t, u, u, v), out)
    # CCSC
    _four(LpRmSmLm, x, y, phi, 4, lambda t, u, v: (t, -HALF_PI, u, v), out)
    _four(LpRmSmRm, x, y, phi, 8, lambda t, u, v: (t, -HALF_PI, u, v), out)
    _four(LpRmSmLm, xb, yb, phi, 6, lambda t, u, v: (v, u, -HALF_PI, t), out)
    _four(LpRmSmRm, xb, yb, phi, 10, lambda t, u, v: (v, u, -HALF_PI, t), out)
    # CCSCC
    _four(LpRmSLmRp, x, y, phi, 16, lambda t, u, v: (t, -HALF_PI, u, -HALF_PI, v), out)
    return out


def paths(start, goal, radius):
    """All Reeds-Shepp candidates from start to goal (x, y, yaw), sorted by length.
    Returns list of (types, lengths_in_metres)."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = goal[2] - start[2]
    result = []
    for types, lengths in _all_paths(x, y, phi):
        segs = [(t, l * radius) for t, l in zip(types, lengths) if abs(l) > 1e-9]
        if segs:
            result.append(segs)
    result.sort(key=lambda p: sum(abs(l) for _, l in p))
    return result


def path_length(segs):
    return sum(abs(l) for _, l in segs)


def sample(start, segs, radius, ds):
    """Sample a path -> (N, 4) array of [x, y, yaw, dir]; excludes the start pose."""
    x, y, th = start
    out = []
    for kind, length in segs:
        d = 1.0 if length > 0 else -1.0
        n = max(1, int(math.ceil(abs(length) / ds)))
        s = np.linspace(abs(length) / n, abs(length), n) * d
        if kind == "S":
            xs, ys, ths = x + s * math.cos(th), y + s * math.sin(th), np.full(n, th)
        else:
            k = (1.0 if kind == "L" else -1.0) / radius
            ths = th + s * k
            xs = x + (np.sin(ths) - math.sin(th)) / k
            ys = y - (np.cos(ths) - math.cos(th)) / k
        out.append(np.stack([xs, ys, ths, np.full(n, d)], 1))
        x, y, th = xs[-1], ys[-1], ths[-1]
    return np.vstack(out)
