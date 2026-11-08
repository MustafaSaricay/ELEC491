"""Unit tests for the building blocks (fast)."""
import math

import numpy as np
import pytest

from parksim import geometry as g
from parksim.config import Config
from parksim.estimation import EKF, ScanMatcher
from parksim.mapping import OccupancyGrid
from parksim.planning import reeds_shepp as rs
from parksim.planning.collision import FootprintChecker
from parksim.planning.hybrid_astar import HybridAStar
from parksim.scenarios import build
from parksim.steering_test import circle_test, fit_circle
from parksim.world import World


def test_reeds_shepp_paths_reach_goal():
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-math.pi, math.pi)])
        e = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-math.pi, math.pi)])
        R = rng.uniform(0.2, 1.0)
        cands = rs.paths(s, e, R)
        assert cands, "Reeds-Shepp must always find a path"
        for p in cands:
            tr = rs.sample(s, p, R, 0.01)
            assert np.hypot(*(tr[-1, :2] - e[:2])) < 1e-6
            assert abs(g.wrap(tr[-1, 2] - e[2])) < 1e-6


def test_ideal_turning_radius_matches_bicycle_model():
    cfg = build("ideal")
    for row in circle_test(cfg, steer_degs=(10, 20, 30)):
        assert row["R_measured"] == pytest.approx(row["R_theory"], rel=0.01)
        assert row["cot_do_minus_cot_di"] == pytest.approx(row["w_over_L"], rel=1e-9)


def test_fit_circle():
    a = np.linspace(0, 2 * np.pi, 50)
    assert fit_circle(1 + 0.4 * np.cos(a), 2 + 0.4 * np.sin(a)) == pytest.approx(0.4)


def test_distance_field_exact():
    cfg = Config()
    grid = OccupancyGrid(2, 2, cfg.mapping)
    grid.mark_obstacles(World(cfg.world))
    occ = grid.occupied_points()
    xs, ys = grid.cell_centers()
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    brute = np.sqrt(((pts[:, None] - occ[None]) ** 2).sum(-1).min(1)).reshape(xs.shape)
    assert np.allclose(brute, grid.distance_field())


def test_planner_path_is_collision_free_and_reaches_goal():
    cfg = Config()
    world = World(cfg.world)
    grid = OccupancyGrid(2, 2, cfg.mapping)
    grid.mark_obstacles(world)
    off = cfg.vehicle.center_offset
    start = g.center_to_rear(g.deg_pose(cfg.world.start), off)
    goal = g.center_to_rear(world.bay_pose, off)
    path, info = HybridAStar(cfg.vehicle, cfg.planner).plan(start, goal, grid)
    assert path is not None, info
    assert np.hypot(path.x[-1] - goal[0], path.y[-1] - goal[1]) < 1e-6
    chk = FootprintChecker(cfg.vehicle, grid, 0.0)
    assert not chk.collides(path.poses).any()
    # curvature never exceeds what the car can steer
    step = np.hypot(np.diff(path.x), np.diff(path.y))
    dyaw = np.abs(g.wrap(np.diff(path.yaw)))
    m = step > 1e-6
    assert (dyaw[m] / step[m]).max() <= 1 / cfg.vehicle.min_turn_radius + 1e-6


def test_planner_reports_blocked_goal():
    cfg = Config()
    cfg.world.obstacles = [{"type": "circle", "x": 1.5, "y": 1.82, "r": 0.05, "h": 0.1}]
    world = World(cfg.world)
    grid = OccupancyGrid(2, 2, cfg.mapping)
    grid.mark_obstacles(world)
    off = cfg.vehicle.center_offset
    path, info = HybridAStar(cfg.vehicle, cfg.planner).plan(
        g.center_to_rear(g.deg_pose(cfg.world.start), off), g.center_to_rear(world.bay_pose, off), grid)
    assert path is None and "blocked" in info["reason"]


def test_scan_matcher_recovers_pose_offset():
    cfg = Config()
    world = World(cfg.world)
    true = np.array([0.7, 0.6, 0.3])
    ang = np.arange(360) * np.pi / 180
    lc = cfg.lidar
    ox, oy = true[0] + lc.mount_x * math.cos(true[2]), true[1] + lc.mount_x * math.sin(true[2])
    r = world.raycast(np.full(360, ox), np.full(360, oy), true[2] + ang, lc.mount_height)
    ok = np.isfinite(r)
    pts = np.stack([lc.mount_x + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], 1)
    res = ScanMatcher(world.walls, cfg.estimator).match(pts, true + [0.04, -0.03, math.radians(4)])
    assert res is not None
    p, _, quality = res
    assert np.hypot(*(p[:2] - true[:2])) < 0.003 and abs(g.wrap(p[2] - true[2])) < math.radians(0.3)
    assert quality > 0.8


def test_ekf_out_of_sequence_update_equals_in_order():
    cfg = Config().estimator
    R = np.diag([1e-4, 1e-4, 1e-4])
    a, b = EKF(cfg, [0, 0, 0]), EKF(cfg, [0, 0, 0])
    z = np.array([0.105, 0.002, 0.01])
    for k in range(1, 11):
        a.predict(k * 0.02, 0.01, 0.0, 0.02)
        if k == 5:
            a.update_pose(k * 0.02, z, R)
    for k in range(1, 11):
        b.predict(k * 0.02, 0.01, 0.0, 0.02)
    b.update_pose(5 * 0.02, z, R)         # arrives late
    assert np.allclose(a.x, b.x) and np.allclose(a.P, b.P)


def test_lidar_cannot_see_low_obstacles():
    cfg = Config()
    cfg.world.obstacles = [{"type": "circle", "x": 1.0, "y": 1.0, "r": 0.05, "h": 0.03}]
    world = World(cfg.world)
    r = world.raycast(np.array([0.5]), np.array([1.0]), np.array([0.0]), cfg.lidar.mount_height)
    assert r[0] == pytest.approx(1.5)            # beam passes over it to the wall
    r = world.raycast(np.array([0.5]), np.array([1.0]), np.array([0.0]), 0.02)
    assert r[0] == pytest.approx(0.45)


def _us_world(obstacles):
    cfg = Config()
    cfg.world.obstacles = obstacles
    return cfg, World(cfg.world)


def test_ultrasonic_specular_wall_gives_no_echo():
    cfg, world = _us_world([])
    uc = cfg.ultrasonic
    # facing the wall head-on: echo at the true distance
    assert world.ultrasonic_echo(1.0, 1.0, 0.0, uc) == pytest.approx(1.0, abs=1e-6)
    # big room, sensor 0.3 m above the bottom wall aimed 60 deg off its normal:
    # every ray in the cone hits that wall at >= 45 deg, so the ping reflects away
    cfg.world.width = cfg.world.height = 20.0
    big = World(cfg.world)
    assert big.ultrasonic_echo(10.0, 0.3, math.radians(-30), uc) == math.inf
    assert big.ultrasonic_echo(10.0, 0.3, math.radians(-80), uc) < 0.35   # near-normal: echo


def test_ultrasonic_sees_small_post_anywhere_in_cone():
    cfg, world = _us_world([{"type": "circle", "x": 1.5, "y": 1.0, "r": 0.025, "h": 0.05}])
    uc = cfg.ultrasonic
    for off_deg in (0, 7, 14):
        r = world.ultrasonic_echo(0.5, 1.0, math.radians(off_deg), uc)
        assert r == pytest.approx(0.975, abs=1e-6)
    assert world.ultrasonic_echo(0.5, 1.0, math.radians(25), uc) > 1.0   # outside the cone


def test_ultrasonic_low_obstacle_only_seen_from_afar():
    cfg, world = _us_world([{"type": "circle", "x": 1.0, "y": 1.0, "r": 0.03, "h": 0.01}])
    uc = cfg.ultrasonic               # mount 4 cm, vertical half-angle 8 deg -> visible beyond ~21 cm
    assert world.ultrasonic_echo(0.6, 1.0, 0.0, uc) == pytest.approx(0.37, abs=1e-6)
    assert world.ultrasonic_echo(0.9, 1.0, 0.0, uc) > 0.5                # too close: under the beam


def test_ekf_range_updates_correct_position():
    # Wide-cone ranges to flat walls barely depend on heading (the nearest echo
    # stays the perpendicular one), so heading must come from the gyro: start
    # with gyro-level heading confidence and check that position converges.
    cfg = Config()
    world = World(cfg.world)
    true = np.array([0.8, 0.9, 0.0])
    ekf = EKF(cfg.estimator, true + np.array([0.03, -0.02, 0.0]))
    ekf.P[:2, :2] = np.eye(2) * 0.03 ** 2
    ekf.P[2, 2] = math.radians(0.1) ** 2
    for k in range(40):
        for s in cfg.ultrasonic.sensors:
            r = ekf.expected_range(true, s, world.walls, cfg.ultrasonic)
            if math.isfinite(r):
                ekf.update_range(k * 0.05, r, s, world.walls, cfg.ultrasonic)
    assert np.hypot(*(ekf.x[:2] - true[:2])) < 0.005


def test_wall_options():
    from parksim.world import wall_sides
    assert wall_sides(True) == ("bottom", "right", "top", "left")
    assert wall_sides([]) == () and wall_sides("none") == ()
    assert wall_sides("left,bottom") == ("bottom", "left")
    with pytest.raises(ValueError):
        wall_sides(["front"])
    cfg = Config()
    cfg.world.walls = ["top"]
    cfg.world.obstacles = []
    w = World(cfg.world)
    assert len(w.walls) == 1
    # no wall on the left: driving past x=0 is "outside", not a collision
    poly = g.rect_corners(0.05, 1.0, 0.0, 0.26, 0.16)
    assert w.outside_testbed(poly) and w.footprint_clearance(poly) > 0
    # ultrasonic facing the missing left wall hears nothing
    assert w.ultrasonic_echo(1.0, 1.0, math.pi, cfg.ultrasonic) == math.inf


@pytest.mark.parametrize("mode", ["circle", "line", "random"])
def test_generated_layouts_respect_shape_count_and_keepouts(mode):
    from parksim import layouts
    cfg = Config()
    cfg.world.obstacle_mode = mode
    cfg.world.n_obstacles = 3
    obs = layouts.generate(cfg, np.random.default_rng(1))
    assert len(obs) == 3
    if mode == "circle":
        assert all(o["type"] == "circle" for o in obs)
    if mode == "line":
        assert all(o["type"] == "box" and o["w"] == cfg.world.line_thickness for o in obs)
    polys = layouts._keepout_polys(cfg)
    assert not any(layouts._overlaps(o, polys, (2.0, 2.0)) for o in obs)
    # same seed -> same layout, different seed -> different layout
    assert layouts.generate(cfg, np.random.default_rng(1)) == obs
    assert layouts.generate(cfg, np.random.default_rng(2)) != obs
