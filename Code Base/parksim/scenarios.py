"""Named scenarios: each is a set of overrides on top of the nominal Config.

Use them as a starting point and stack your own changes with ``--set`` or a
JSON file, e.g.
    python -m parksim run -s laggy --set lidar.latency=0.3
"""
from __future__ import annotations

import copy

from .config import Config, apply_overrides

IDEAL = {
    "steering": {"backlash_deg": 0.0, "time_const": 0.0, "rate_limit_dps": 1e6},
    "drive": {"dead_zone": 0.0, "time_const": 0.0},
    "traction": {"slip_std": 0.0, "yaw_noise": 0.0},
    "imu": {"gyro_noise_dps": 0.0, "gyro_bias_dps": 0.0, "bias_walk_dps": 0.0},
    "lidar": {"range_sigma": 0.0, "dropout_prob": 0.0, "motion_distortion": False, "latency": 0.0},
    "camera": {"lateral_sigma": 0.0, "heading_sigma_deg": 0.0, "dropout_prob": 0.0, "latency": 0.0},
    "comm": {"latency": 0.0, "jitter": 0.0},
    "estimator": {"init_sigma_xy": 1e-4, "init_sigma_yaw_deg": 0.01},
    "control": {"actuation_latency": 0.0, "deadzone_comp": 0.0},
}

NOISY_SENSORS = {
    "ultrasonic": {"range_sigma": 0.01, "range_sigma_prop": 0.02, "dropout_prob": 0.10},
    "imu": {"gyro_noise_dps": 1.0, "gyro_bias_dps": 2.0, "bias_walk_dps": 0.05},
    "camera": {"lateral_sigma": 0.01, "heading_sigma_deg": 3.0, "dropout_prob": 0.3, "outlier_prob": 0.05},
    "encoder": {"ticks_per_rev": 48},
}

LAGGY = {
    "ultrasonic": {"latency": 0.05},
    "camera": {"latency": 0.25},
    "encoder": {"latency": 0.02},
    "control": {"actuation_latency": 0.08},
    "comm": {"latency": 0.15, "jitter": 0.05},
    "planner": {"compute_factor": 5.0},
}

SLIPPERY = {
    "traction": {"slip_std": 0.08, "yaw_noise": 0.01, "understeer": 2.0,
                 "slip_zones": [{"rect": [0.7, 0.8, 1.3, 1.6], "slip": 0.3}]},
}

MISCALIBRATED = {
    "steering": {"offset_deg": 3.0, "gain": 0.9, "backlash_deg": 3.0, "ackermann_error": 0.5},
    "encoder": {"scale_error": 0.05},
    "imu": {"scale_error": 0.03},
    "ultrasonic": {"scale_error": 0.03},     # speed of sound assumed at the wrong temperature (~10 C off)
}

WEAK_MOTOR = {
    "drive": {"dead_zone": 0.2, "battery_sag_per_min": 0.15, "speed_noise": 0.03, "time_const": 0.3},
}

INTERFERENCE = {
    "ultrasonic": {"crosstalk_prob": 0.08, "spurious_prob": 0.03},
    "comm": {"loss_prob": 0.1},
}


def _merge(*parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            out.setdefault(k, {}).update(copy.deepcopy(v))
    return out


SCENARIOS = {
    "ideal": ("No noise, no delay, perfect actuators - sanity baseline", IDEAL),
    "nominal": ("Realistic moderate noise (the Config defaults)", {}),
    "noisy_sensors": ("Noisy ultrasonics (1 cm + 2%/m, 10% dropout), bad gyro, flaky camera, coarse encoder",
                      NOISY_SENSORS),
    "laggy": ("Large sensor/actuation/Wi-Fi latency and a slow planner (Pi under load)", LAGGY),
    "laggy_no_comp": ("Same as laggy but the EKF ignores measurement timestamps",
                      _merge(LAGGY, {"estimator": {"compensate_latency": False}})),
    "slippery": ("Wheel slip, a low-traction patch, heading noise, understeer", SLIPPERY),
    "miscalibrated": ("Steering trim/gain/backlash, wheel-radius, gyro-scale and speed-of-sound errors",
                      MISCALIBRATED),
    "weak_motor": ("Big dead zone, battery sag, load disturbance, sluggish motor", WEAK_MOTOR),
    "interference": ("Ultrasonic crosstalk (8% per neighbour) + 3% random echoes, 10% packet loss", INTERFERENCE),
    "wifi_dropout": ("Wi-Fi outage mid-manoeuvre: heartbeat failsafe must stop the car",
                     {"comm": {"outages": [[6.0, 8.0]]}}),
    "wifi_no_resume": ("Wi-Fi outage, and the task is abandoned instead of resumed",
                       {"comm": {"outages": [[6.0, 8.0]], "resume_after_outage": False}}),
    "low_obstacle": ("1 cm high obstacle (a cable) on the route: only inside the ultrasonic cone beyond ~21 cm",
                     {"world": {"obstacles": [
                         {"type": "circle", "x": 1.0, "y": 1.05, "r": 0.025, "h": 0.05},
                         {"type": "box", "x": 0.55, "y": 1.35, "l": 0.30, "w": 0.10, "yaw_deg": 0.0, "h": 0.10},
                         {"type": "circle", "x": 0.91, "y": 0.78, "r": 0.03, "h": 0.01}]}}),
    "surprise_obstacle": ("Obstacle placed on the route at t=3 s: detect, stop, re-plan",
                          {"mission": {"events": [{"t": 3.0, "type": "spawn_obstacle", "obstacle":
                                                   {"type": "box", "x": 1.27, "y": 1.08, "l": 0.12,
                                                    "w": 0.12, "yaw_deg": 20, "h": 0.08}}]}}),
    "kidnap": ("Car is bumped 8 cm / 10 deg at t=8 s", {"mission": {"events": [
        {"t": 8.0, "type": "kidnap", "dx": 0.08, "dy": 0.0, "dyaw_deg": 10.0}]}}),
    "cpu_overload": ("30% of control cycles missed and a 10x slower planner",
                     {"control": {"skip_prob": 0.3}, "planner": {"compute_factor": 10.0}}),
    "estop": ("User hits emergency stop at t=6 s, then summons the car",
              {"mission": {"events": [{"t": 6.0, "type": "estop"}]}}),
    "rear_blind_spot": ("A 5 cm post appears 0.35 m straight behind the parked car, then it is summoned and "
                        "reverses out (the post sits between the two +-140 deg rear cones)",
                        {"mission": {"events": [{"after_task": 0, "delay": 0.5, "type": "spawn_obstacle",
                                                 "relative_to_car": True, "obstacle":
                                                 {"type": "circle", "x": -0.35, "y": 0.0, "r": 0.025, "h": 0.05}}]}}),
    "rear_blind_spot_fixed": ("Same, with an 8th HC-SR04 pointing straight back",
                              {"mission": {"events": [{"after_task": 0, "delay": 0.5, "type": "spawn_obstacle",
                                                       "relative_to_car": True, "obstacle":
                                                       {"type": "circle", "x": -0.35, "y": 0.0, "r": 0.025, "h": 0.05}}]},
                               "ultrasonic": {"sensors": [*__import__("parksim.config", fromlist=["x"])._default_ultrasonic_layout(),
                                                          {"name": "B", "x": -0.045, "y": 0.0, "yaw_deg": 180.0}],
                                              "fire_groups": [["F", "RL", "RR"], ["FL", "R"], ["FR", "L"], ["B"]]}}),
    # ---- walls ---------------------------------------------------------- #
    "no_walls": ("Open floor: no physical walls - ultrasonics have nothing to localize against",
                 {"world": {"walls": []}}),
    "two_walls": ("Only the bottom and left walls exist",
                  {"world": {"walls": ["bottom", "left"]}}),
    "bay_wall_only": ("Only the wall behind the bay (top) exists",
                      {"world": {"walls": ["top"]}}),
    # ---- obstacle shape / timing ------------------------------------------ #
    "random_circles": ("4 random posts/cans (new layout per seed)",
                       {"world": {"obstacle_mode": "circle", "n_obstacles": 4}}),
    "random_lines": ("3 random thin walls at random angles (flat faces: specular misses)",
                     {"world": {"obstacle_mode": "line", "n_obstacles": 3}}),
    "random_mix": ("4 random obstacles: posts, boxes and thin walls mixed",
                   {"world": {"obstacle_mode": "random", "n_obstacles": 4}}),
    "obstacles_after_park": ("The default obstacles appear only after the car has parked",
                             {"world": {"obstacle_timing": "after_park"}}),
    "random_after_park": ("4 random obstacles placed after parking: the summon must deal with them",
                          {"world": {"obstacle_mode": "random", "n_obstacles": 4,
                                     "obstacle_timing": "after_park"}}),
    "fire_all_at_once": ("All 7 HC-SR04s pinged together: fast updates, heavy crosstalk",
                         {"ultrasonic": {"fire_groups": [["F", "FL", "FR", "L", "R", "RL", "RR"]]}}),
    "fire_one_by_one": ("HC-SR04s pinged one at a time: no crosstalk, 0.42 s between readings",
                        {"ultrasonic": {"fire_groups": [["F"], ["FL"], ["FR"], ["L"], ["R"], ["RL"], ["RR"]],
                                        "crosstalk_prob": 0.0}}),
    "lidar_baseline": ("The original 360 deg LiDAR instead of the ultrasonic array (for comparison)",
                       {"lidar": {"enabled": True}, "ultrasonic": {"enabled": False}}),
    "worst_case": ("Noisy + laggy + slippery + miscalibrated + interference, all at once",
                   _merge(NOISY_SENSORS, LAGGY, SLIPPERY, MISCALIBRATED, INTERFERENCE)),
}


def build(name: str = "nominal", overrides: dict | None = None) -> Config:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. Available: {', '.join(SCENARIOS)}")
    cfg = Config()
    apply_overrides(cfg, SCENARIOS[name][1])
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


# Lists under these keys are concatenated when presets are combined (so e.g.
# two presets' timed events both happen); every other key: the later preset wins.
APPEND_KEYS = {"mission.events", "comm.outages", "traction.slip_zones"}


def _deep_merge(dst, src, conflicts, path=""):
    for k, v in src.items():
        p = f"{path}{k}"
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v, conflicts, p + ".")
        elif p in APPEND_KEYS and k in dst:
            dst[k] = dst[k] + copy.deepcopy(v)
        else:
            if k in dst and dst[k] != v:
                conflicts.append(p)
            dst[k] = copy.deepcopy(v)


def combine(names, overrides: dict | None = None):
    """Build a Config from several presets applied in order.
    Returns (cfg, conflicts) where conflicts lists keys set differently by
    more than one preset (the last one wins)."""
    names = [n for n in names if n] or ["nominal"]
    merged, conflicts = {}, []
    for n in names:
        if n not in SCENARIOS:
            raise KeyError(f"unknown scenario '{n}'. Available: {', '.join(SCENARIOS)}")
        _deep_merge(merged, SCENARIOS[n][1], conflicts)
    cfg = Config()
    apply_overrides(cfg, merged)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg, sorted(set(conflicts))
