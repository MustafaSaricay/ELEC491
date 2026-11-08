"""Web interface back end: option resolution, preset combination, API round trip."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from parksim.scenarios import combine
from parksim.ui import server as ui


def test_combine_concatenates_events_and_reports_conflicts():
    cfg, conflicts = combine(["wifi_dropout", "surprise_obstacle", "estop", "laggy"])
    assert cfg.comm.outages == [[6.0, 8.0]] and cfg.comm.latency == 0.15
    assert [e["type"] for e in cfg.mission.events] == ["spawn_obstacle", "estop"]
    assert conflicts == []
    cfg, conflicts = combine(["no_walls", "two_walls"])
    assert cfg.world.walls == ["bottom", "left"] and conflicts == ["world.walls"]


def test_build_config_applies_every_option():
    cfg, notes = ui.build_config({
        "presets": ["laggy"], "walls": "bottom+left", "obstacle_mode": "line", "n_obstacles": 2,
        "obstacle_timing": "after_park", "sensors": "both", "firing": "one_by_one", "rear_sensor": True,
        "seed": 7, "overrides": [{"key": "comm.latency", "value": "0.2"}, {"key": "", "value": "x"}]})
    w = cfg.world
    assert w.walls == ["bottom", "left"] and w.obstacle_mode == "line" and w.n_obstacles == 2
    assert w.obstacle_timing == "after_park"
    assert cfg.lidar.enabled and cfg.ultrasonic.enabled
    assert [s["name"] for s in cfg.ultrasonic.sensors][-1] == "B"
    assert cfg.ultrasonic.fire_groups == [[s["name"]] for s in cfg.ultrasonic.sensors]
    assert cfg.comm.latency == 0.2 and cfg.sim.seed == 7
    # "default" keeps whatever the preset says
    cfg, _ = ui.build_config({"presets": ["no_walls"], "walls": "default"})
    assert cfg.world.walls == []
    with pytest.raises(KeyError):
        ui.build_config({"overrides": [{"key": "lidar.nope", "value": "1"}]})


def test_compare_variants_is_cartesian_product():
    vs = ui.compare_variants({"presets": ["laggy"]}, {"walls": ["all", "none"], "rear_sensor": [False, True],
                                                      "obstacle_mode": []})
    assert len(vs) == 4
    assert {(v["walls"], v["rear_sensor"]) for v in vs} == {("all", False), ("all", True), ("none", False), ("none", True)}
    assert all(v["presets"] == ["laggy"] for v in vs)


def test_api_single_run_round_trip():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "window.PRELOAD" in page
        opts = json.load(urllib.request.urlopen(base + "/api/options"))
        assert "ultrasonic.ping_interval" in opts["keys"]
        req = urllib.request.Request(base + "/api/run", json.dumps({"kind": "single", "spec": {"presets": ["ideal"]}}).encode(),
                                     {"Content-Type": "application/json"})
        job = json.load(urllib.request.urlopen(req))["job"]
        for _ in range(240):
            j = json.load(urllib.request.urlopen(f"{base}/api/job/{job}"))
            if j["status"] != "running":
                break
            time.sleep(0.5)
        assert j["status"] == "done", j["error"]
        r = j["result"]
        assert r["metrics"]["all_success"] and r["summary_png"] and len(r["replay"]["t"]) > 100
        assert set(r["pings"]) == {"F", "FL", "FR", "L", "R", "RL", "RR"}
    finally:
        httpd.shutdown()


def test_clamps_and_run_limits():
    cfg, _ = ui.build_config({"overrides": [{"key": "mission.max_time", "value": "100000"},
                                            {"key": "sim.dt", "value": "0.00001"}]})
    assert cfg.mission.max_time == 300.0 and cfg.sim.dt == 0.002
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    old = dict(ui.LIMITS)
    ui.LIMITS.update(max_runs=5, max_sims=8)
    try:
        def post(body):
            req = urllib.request.Request(base + "/api/run", json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req).status, None
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)["error"]
        code, err = post({"kind": "mc", "spec": {}, "runs": 6})
        assert code == 400 and "at most 5" in err
        code, err = post({"kind": "compare", "variants": [{}, {}, {}], "runs": 3})
        assert code == 400 and "9 simulations" in err
    finally:
        ui.LIMITS.clear()
        ui.LIMITS.update(old)
        httpd.shutdown()
