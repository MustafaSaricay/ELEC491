"""Simulation configuration.

Every tunable number lives here, grouped by subsystem. Components read their
config object *live* (not cached), so a timed ``set`` event in a scenario can
change any value mid-run (e.g. turn on LiDAR interference at t=20 s).

User-facing poses (start, bay, summon) are **vehicle-centre** poses given as
``[x, y, yaw_deg]`` in the testbed frame (origin = bottom-left corner, metres).
Internally the software uses the **rear-axle** pose, which is the natural
reference point of the kinematic bicycle model.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# Vehicle and actuators (the *true* plant)
# --------------------------------------------------------------------------- #
@dataclass
class VehicleCfg:
    wheelbase: float = 0.17          # L [m]
    track: float = 0.14              # w, front track width [m]
    length: float = 0.26             # body length [m]
    width: float = 0.16              # body width [m]
    rear_overhang: float = 0.045     # rear bumper -> rear axle [m]
    max_steer_deg: float = 30.0      # mechanical steering limit (bicycle angle)

    @property
    def max_steer(self) -> float:
        return math.radians(self.max_steer_deg)

    @property
    def center_offset(self) -> float:
        """Distance rear axle -> body centre along the heading."""
        return self.length / 2.0 - self.rear_overhang

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer)


@dataclass
class SteeringCfg:
    """Steering servo + linkage imperfections."""
    gain: float = 1.0                # actual/commanded angle (servo calibration error)
    offset_deg: float = 0.0          # trim error: wheels not straight at command 0
    backlash_deg: float = 0.5        # total play in the linkage
    rate_limit_dps: float = 400.0    # servo slew rate
    time_const: float = 0.04         # first-order servo lag [s]
    noise_deg: float = 0.0           # servo jitter (white, per physics step)
    ackermann_error: float = 0.0     # 0 = ideal Ackermann, 1 = parallel steer (scrub)


@dataclass
class DriveCfg:
    """DC drive motor + driver (duty in [-1, 1])."""
    max_speed: float = 0.8           # wheel speed at duty 1 with a full battery [m/s]
    dead_zone: float = 0.08          # |duty| below this produces no motion
    time_const: float = 0.12         # motor/inertia lag [s]
    battery_sag_per_min: float = 0.0 # fractional speed loss per minute of run time
    speed_noise: float = 0.0         # random load disturbance on wheel speed [m/s, std]


@dataclass
class TractionCfg:
    """Wheel/floor interaction."""
    slip_std: float = 0.02           # longitudinal slip (fraction, OU process std)
    yaw_noise: float = 0.002         # heading random walk [rad / sqrt(m)]
    understeer: float = 0.0          # K in kappa_eff = kappa / (1 + K v^2) [s^2/m^2]
    # Low-traction patches: [{"rect": [x0, y0, x1, y1], "slip": 0.3}, ...]
    slip_zones: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
@dataclass
class EncoderCfg:
    rate_hz: float = 50.0
    ticks_per_rev: int = 360         # quadrature counts per wheel revolution
    wheel_radius: float = 0.032      # nominal radius the software assumes [m]
    scale_error: float = 0.0         # true radius = nominal * (1 + scale_error)
    miss_prob: float = 0.0           # probability each tick is missed
    latency: float = 0.0


@dataclass
class ImuCfg:
    rate_hz: float = 100.0
    gyro_noise_dps: float = 0.2      # white noise std per sample [deg/s]
    gyro_bias_dps: float = 0.5       # initial bias std [deg/s]
    bias_walk_dps: float = 0.01      # bias random walk [deg/s / sqrt(s)]
    scale_error: float = 0.0
    latency: float = 0.0


def _default_ultrasonic_layout():
    # Positions are in the rear-axle frame [m]: x forward, y left. The front
    # bumper is at x = 0.215, the rear bumper at x = -0.045 and the sides at y = +-0.08.
    return [
        {"name": "F",  "x": 0.215, "y": 0.00,  "yaw_deg": 0.0},
        {"name": "FL", "x": 0.205, "y": 0.07,  "yaw_deg": 40.0},
        {"name": "FR", "x": 0.205, "y": -0.07, "yaw_deg": -40.0},
        {"name": "L",  "x": 0.085, "y": 0.08,  "yaw_deg": 90.0},
        {"name": "R",  "x": 0.085, "y": -0.08, "yaw_deg": -90.0},
        {"name": "RL", "x": -0.035, "y": 0.07, "yaw_deg": 140.0},
        {"name": "RR", "x": -0.035, "y": -0.07, "yaw_deg": -140.0},
    ]


@dataclass
class UltrasonicCfg:
    """HC-SR04 array. Each sensor reports the FIRST echo inside its cone.
    Surfaces hit at more than ``max_incidence_deg`` from their normal reflect the
    ping away (specular), so they produce no echo."""
    enabled: bool = True
    sensors: list = field(default_factory=_default_ultrasonic_layout)
    # Sensors in the same group fire together; groups fire one after another,
    # one every ``ping_interval`` (the datasheet recommends >= 60 ms).
    fire_groups: list = field(default_factory=lambda: [["F", "RL", "RR"], ["FL", "R"], ["FR", "L"]])
    ping_interval: float = 0.06
    half_angle_deg: float = 15.0     # horizontal half-cone (effective)
    vertical_half_deg: float = 8.0   # obstacle visible if h > mount_height - r*tan(this)
    n_rays: int = 31                 # rays sampling the cone for flat surfaces (circles are exact)
    max_incidence_deg: float = 35.0  # specular cut-off
    mount_height: float = 0.04
    range_min: float = 0.02          # blind zone
    range_max: float = 2.0           # practical indoor limit; no echo -> NaN
    range_sigma: float = 0.003       # [m]
    range_sigma_prop: float = 0.005  # extra std per metre of range
    resolution: float = 0.003        # output quantisation
    scale_error: float = 0.0         # speed-of-sound error (temperature), e.g. 0.02
    dropout_prob: float = 0.02       # missed echo
    spurious_prob: float = 0.0       # random short false echo (noise, other robots)
    crosstalk_prob: float = 0.02     # per extra sensor in the group: hears its ping
    latency: float = 0.005           # processing after the echo returns


@dataclass
class LidarCfg:
    enabled: bool = False            # replaced by the ultrasonic array (kept for comparison)
    rate_hz: float = 10.0
    n_beams: int = 360
    range_sigma: float = 0.01        # [m]
    range_min: float = 0.10
    range_max: float = 6.0
    dropout_prob: float = 0.01       # beam returns nothing
    spurious_prob: float = 0.0       # interference: random short false return
    mount_height: float = 0.04       # beam plane height; obstacles lower are INVISIBLE
    mount_x: float = 0.085           # position ahead of rear axle [m]
    yaw_offset_deg: float = 0.0      # unknown mounting yaw error
    motion_distortion: bool = True   # beams sampled along the 1/rate rotation period
    latency: float = 0.03            # driver + transfer delay [s]


@dataclass
class CameraCfg:
    """Bay-line detector (OpenCV pipeline abstracted to its output)."""
    enabled: bool = True
    mount: str = "front"             # "front" or "rear"
    rate_hz: float = 15.0
    fov_deg: float = 120.0           # wide-angle
    max_range: float = 0.6           # lines detectable within this distance [m]
    lateral_sigma: float = 0.004     # [m]
    heading_sigma_deg: float = 1.0
    dropout_prob: float = 0.05
    outlier_prob: float = 0.0        # gross misdetection (+-5 cm, +-10 deg)
    latency: float = 0.08            # capture + inference time [s]


# --------------------------------------------------------------------------- #
# Communication
# --------------------------------------------------------------------------- #
@dataclass
class CommCfg:
    latency: float = 0.03            # one-way base latency [s]
    jitter: float = 0.01             # mean of extra exponential delay [s]
    loss_prob: float = 0.0           # packet loss -> TCP retransmission delay
    retransmit_delay: float = 0.2
    heartbeat_hz: float = 5.0
    heartbeat_timeout: float = 0.5   # vehicle stops if no heartbeat for this long
    telemetry_hz: float = 10.0
    outages: list = field(default_factory=list)   # [[t0, t1], ...] Wi-Fi down
    resume_after_outage: bool = True


# --------------------------------------------------------------------------- #
# Onboard software
# --------------------------------------------------------------------------- #
@dataclass
class EstimatorCfg:
    init_sigma_xy: float = 0.01      # initial belief error std (also P0)
    init_sigma_yaw_deg: float = 2.0
    enc_sigma_frac: float = 0.03     # odometry noise as fraction of distance
    enc_sigma_min: float = 0.0005
    lateral_sigma_frac: float = 0.01 # side-slip noise as fraction of distance
    gyro_sigma_dps: float = 0.3      # gyro rate noise density used by the filter
    bias_walk_dps: float = 0.02
    use_lidar: bool = True
    use_ultrasonic: bool = True      # range-to-known-wall EKF updates
    use_camera: bool = True
    us_sigma: float = 0.01           # range noise the filter assumes [m]
    us_sigma_prop: float = 0.01      # extra std per metre
    gate_range: float = 6.63         # chi2(1) 99 %: echoes from obstacles get rejected
    us_long_tol: float = 0.03        # echo this much longer than the wall predicts = pose error
    us_diverge_count: int = 4        # consecutive such echoes -> inflate covariance
    us_inflate_xy: float = 0.03
    us_inflate_yaw_deg: float = 3.0
    compensate_latency: bool = True  # re-run filter from the measurement timestamp
    history_s: float = 1.0
    gate_pose: float = 16.27         # chi2(3) 99.9 %
    gate_cam: float = 13.82          # chi2(2) 99.9 %
    icp_iters: int = 15
    icp_inlier_dist: float = 0.08
    icp_min_inliers: int = 30
    icp_sigma: float = 0.01
    lidar_cov_scale: float = 4.0     # inflate ICP covariance (correlated errors)
    icp_floor_xy: float = 0.01       # added std: systematic errors (distortion, mounting)
    icp_floor_yaw_deg: float = 1.0
    lost_sigma: float = 0.08         # pose std above this => localization lost
    max_lidar_rejects: int = 15      # consecutive failed/rejected scans => lost
    reloc_after: int = 5             # consecutive gated-out scans before re-localizing
    reloc_min_quality: float = 0.8   # ICP inlier ratio required to trust a re-localization


@dataclass
class MapCfg:
    resolution: float = 0.02
    l_hit: float = 0.85
    l_miss: float = -0.4
    l_min: float = -4.0
    l_max: float = 4.0
    occ_threshold: float = 1.2       # > l_hit: one stray hit is not an obstacle
    ignore_near_walls: float = 0.05  # hits this close to a known wall count as wall
    free_ray_max: float = 3.0
    us_l_hit: float = 0.6            # ultrasonic: the arc at the echo range (position ambiguous)
    us_l_miss: float = -0.3          # ultrasonic: cone interior before the echo
    us_wall_tol: float = 0.04        # echo within this of the predicted wall range = wall
    stop_mark_logodds: float = 2.4   # what triggered an emergency stop is written into the map
                                     # this strongly: a few clear readings can still erase it
                                     # (crosstalk false alarms), real obstacles keep echoing


@dataclass
class PlannerCfg:
    xy_res: float = 0.05
    yaw_bins: int = 72
    step: float = 0.10               # primitive arc length [m]
    sample_ds: float = 0.02          # collision-check / path sample spacing
    n_steer: int = 5
    steer_margin: float = 0.85       # plan with this fraction of max steer
    reverse_cost: float = 1.5
    switch_cost: float = 0.4         # [m-equivalent] per direction change
    steer_cost: float = 0.02
    steer_change_cost: float = 0.05
    safety_margin: float = 0.02      # footprint inflation [m]
    final_straight: float = 0.30     # end every path with this straight run into the goal
    rs_interval: int = 3             # try Reeds-Shepp shot every N expansions (near the goal)
    rs_near_dist: float = 1.0        # ... beyond this heuristic distance only every 10*N
    rs_candidates: int = 8
    max_expansions: int = 6000
    escape_radius: float = 0.15      # relaxed margin near a start pose that is too close
    # Simulated planning time (deterministic, the car waits while it "thinks"):
    #   t = (compute_base + compute_per_expansion * expansions) * compute_factor
    compute_base: float = 0.05
    compute_per_expansion: float = 0.0004
    compute_factor: float = 1.0      # e.g. 5 = Pi heavily loaded


@dataclass
class ControlCfg:
    rate_hz: float = 50.0
    lookahead_fwd: float = 0.10
    lookahead_rev: float = 0.08
    lookahead_gain: float = 0.2      # extra look-ahead per m/s
    cruise_speed: float = 0.25
    approach_speed: float = 0.08
    approach_dist: float = 0.30
    min_speed: float = 0.04
    decel: float = 0.4
    kp: float = 1.0
    ki: float = 3.0
    kd: float = 0.0
    motor_max_speed: float = 0.8     # controller's model of the motor (feed-forward)
    deadzone_comp: float = 0.08      # added to |duty| when moving
    speed_lpf: float = 0.5           # encoder speed low-pass alpha
    stop_lead_time: float = 0.12     # cut motor this long before the end (coast-down)
    goal_pos_tol: float = 0.02       # final error above this triggers a correction
    goal_yaw_tol_deg: float = 4.0
    cusp_tol: float = 0.02
    cusp_pause: float = 0.3
    max_corrections: int = 2         # extra re-plans when final pose is out of tol
    actuation_latency: float = 0.02  # command -> actuator
    skip_prob: float = 0.0           # control cycle missed (CPU overload)


@dataclass
class SafetyCfg:
    bubble_margin: float = 0.02      # emergency-stop zone beyond bumper [m]
    bubble_lateral: float = 0.02
    decel_assumed: float = 0.5
    min_points: int = 2              # LiDAR points needed to trigger (1 = paranoid)
    persist_scans: int = 2           # LiDAR: consecutive scans the bubble must be hit (1 = paranoid)
    us_persist: int = 2              # ultrasonic: consecutive readings per sensor needed
                                     # (the stop zone grows by v * cycle * us_persist to compensate)
    us_max_age: float = 0.4          # ignore ultrasonic readings older than this [s]
    path_check_horizon: float = 0.6
    path_margin: float = 0.0
    max_replans: int = 8             # re-plans after the path got blocked
    max_plan_failures: int = 3       # consecutive planner failures -> give up ("no_path")
    blocked_retry_s: float = 1.0


# --------------------------------------------------------------------------- #
# World + mission
# --------------------------------------------------------------------------- #
@dataclass
class WorldCfg:
    width: float = 2.0
    height: float = 2.0
    wall_height: float = 0.15
    # Which edges have a PHYSICAL wall: any of "bottom", "right", "top", "left";
    # [] / "none" = open floor. Walls are the only localization landmarks for
    # the ultrasonics and the LiDAR. A missing wall is still a (virtual) boundary:
    # the planner stays inside it, and crossing it counts as leaving the testbed.
    walls: list = field(default_factory=lambda: ["bottom", "right", "top", "left"])
    # Obstacles: "fixed" uses the `obstacles` list below; "circle", "line" or
    # "random" generates `n_obstacles` new ones per seed (see layouts.py).
    obstacle_mode: str = "fixed"
    n_obstacles: int = 4
    # "before_park": present from t=0. "after_park": appear `after_park_delay` s
    # after the park task finishes (so they only affect the summon).
    obstacle_timing: str = "before_park"
    after_park_delay: float = 0.5
    # generator ranges [min, max]
    circle_radius: list = field(default_factory=lambda: [0.025, 0.06])
    line_length: list = field(default_factory=lambda: [0.20, 0.50])
    line_thickness: float = 0.02
    box_size: list = field(default_factory=lambda: [0.05, 0.15])
    obstacle_height: list = field(default_factory=lambda: [0.05, 0.15])
    keepout_margin: float = 0.10     # clearance kept around start, bay and pickup poses
    layout_retries: int = 40         # re-draw until both legs are plannable
    start: list = field(default_factory=lambda: [0.35, 0.35, 0.0])
    bay: dict = field(default_factory=lambda: {
        "x": 1.5, "y": 1.82, "yaw_deg": 90.0, "width": 0.26, "depth": 0.36})
    # {"type": "circle", "x", "y", "r", "h"} | {"type": "box", "x", "y", "l", "w", "yaw_deg", "h"}
    obstacles: list = field(default_factory=lambda: [
        {"type": "circle", "x": 1.0, "y": 1.05, "r": 0.025, "h": 0.05},
        {"type": "box", "x": 0.55, "y": 1.35, "l": 0.30, "w": 0.10, "yaw_deg": 0.0, "h": 0.10},
    ])


@dataclass
class MissionCfg:
    commands: list = field(default_factory=lambda: [
        {"cmd": "park", "wait": 0.5},
        {"cmd": "summon", "pose": [0.40, 0.40, 180.0], "wait": 2.0},
    ])
    # Timed events, see README: spawn_obstacle, kidnap, estop, set
    events: list = field(default_factory=list)
    max_time: float = 150.0
    stop_on_collision: bool = True
    success_pos_tol: float = 0.05    # project requirement: +-5 cm
    success_yaw_tol_deg: float = 5.0


@dataclass
class SimCfg:
    dt: float = 0.005
    seed: int = 0
    log_hz: float = 50.0
    # Event logging (off when log_dir is empty): one JSON-lines file per run,
    # <log_dir>/<log_name>_s<seed>.jsonl - see parksim/eventlog.py
    log_dir: str = ""
    log_name: str = "run"
    log_state_hz: float = 1.0


@dataclass
class Config:
    vehicle: VehicleCfg = field(default_factory=VehicleCfg)
    steering: SteeringCfg = field(default_factory=SteeringCfg)
    drive: DriveCfg = field(default_factory=DriveCfg)
    traction: TractionCfg = field(default_factory=TractionCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    imu: ImuCfg = field(default_factory=ImuCfg)
    ultrasonic: UltrasonicCfg = field(default_factory=UltrasonicCfg)
    lidar: LidarCfg = field(default_factory=LidarCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    comm: CommCfg = field(default_factory=CommCfg)
    estimator: EstimatorCfg = field(default_factory=EstimatorCfg)
    mapping: MapCfg = field(default_factory=MapCfg)
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    world: WorldCfg = field(default_factory=WorldCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)
    sim: SimCfg = field(default_factory=SimCfg)

    # ---- helpers -------------------------------------------------------- #
    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def apply(self, overrides: dict) -> "Config":
        apply_overrides(self, overrides)
        return self

    def set(self, dotted_key: str, value: Any) -> "Config":
        set_dotted(self, dotted_key, value)
        return self


def apply_overrides(obj: Any, overrides: dict, _path: str = "") -> None:
    """Recursively merge a nested dict into a dataclass tree."""
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key '{_path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_overrides(current, value, f"{_path}{key}.")
        else:
            setattr(obj, key, copy.deepcopy(value))


def resolve(obj: Any, dotted_key: str):
    """Return (parent_object, attribute_name) for 'lidar.range_sigma'."""
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise KeyError(f"unknown config section '{p}' in '{dotted_key}'")
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown config key '{dotted_key}'")
    return obj, parts[-1]


def set_dotted(cfg: Any, dotted_key: str, value: Any) -> None:
    parent, name = resolve(cfg, dotted_key)
    setattr(parent, name, copy.deepcopy(value))


def get_dotted(cfg: Any, dotted_key: str) -> Any:
    parent, name = resolve(cfg, dotted_key)
    return getattr(parent, name)


def parse_value(text: str) -> Any:
    """CLI value parser: JSON first ('0.02', '[[10,12]]', 'true'), else raw string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def section_names() -> list[str]:
    return [f.name for f in fields(Config)]
