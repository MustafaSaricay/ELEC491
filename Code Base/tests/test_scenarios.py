"""Closed-loop regression tests (each runs a full simulation, ~5-15 s)."""
import pytest

from parksim.scenarios import build
from parksim.simulator import run


@pytest.mark.parametrize("name", ["ideal", "nominal"])
def test_park_and_summon_within_requirement(name):
    m = run(build(name)).metrics
    assert m["all_success"], m["tasks"]
    for t in m["tasks"]:
        assert t["err_pos"] < 0.05 and abs(t["err_heading_deg"]) < 5
    assert not m["collided"]


def test_simulation_is_deterministic():
    a = run(build("nominal")).metrics
    b = run(build("nominal")).metrics
    assert a["tasks"] == b["tasks"]


def test_wifi_failsafe_stops_vehicle():
    m = run(build("wifi_dropout")).metrics
    fs = m["failsafe"][0]
    assert fs["moving_at_loss"]
    timeout = build("wifi_dropout").comm.heartbeat_timeout
    assert fs["stop_time"] < timeout + 0.5
    assert m["all_success"]          # resumes after the link returns


def test_rear_blind_spot_between_corner_sensors():
    # 7-sensor layout: a post directly behind is outside both +-140 deg cones
    assert run(build("rear_blind_spot")).metrics["collided"]
    # an 8th sensor pointing straight back sees it and the car stops in time
    assert not run(build("rear_blind_spot_fixed")).metrics["collided"]


def test_lidar_baseline_still_works():
    assert run(build("lidar_baseline")).metrics["all_success"]


def test_surprise_obstacle_triggers_replan_without_collision():
    res = run(build("surprise_obstacle"))
    assert not res.metrics["collided"]
    assert any("obstacle on path" in e for _, e in res.events)
    assert res.metrics["tasks"][0]["success"]


def test_estop_aborts_task():
    m = run(build("estop")).metrics
    assert m["tasks"][0]["result"] == "estop"
    assert m["tasks"][1]["success"]


def test_no_walls_fails_safe_instead_of_crashing():
    m = run(build("no_walls")).metrics
    assert not m["collided"] and not m["left_testbed"]
    assert m["tasks"][0]["result"] == "loc_lost"      # stops itself when it is no longer sure where it is


def test_obstacles_after_park_appear_after_parking():
    res = run(build("obstacles_after_park"))
    park_end = res.tasks[0]["t_end"]
    assert res.tasks[0]["success"]
    assert all(o.spawn_t >= park_end for o in res.obstacles)
    assert res.metrics["all_success"]
