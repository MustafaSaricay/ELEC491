# parksim: architecture and models

This document explains **how the simulator is structured** and **how each component is modelled**. For installing and running it, see [README.md](README.md). All default values quoted here come from [parksim/config.py](parksim/config.py), which is the single place to change them.

---

## 1. The big picture

The code is split into two halves that only talk through messages:

```
 ┌───────────────────── SIMULATED WORLD (ground truth) ─────────────────────┐
 │  world.py      testbed walls, obstacles (with heights), parking bay      │
 │  vehicle.py    the real car: bicycle model + servo/motor/tyre errors     │
 │  sensors.py    encoder, gyro, 7x HC-SR04, (LiDAR), bay camera            │
 │  comms.py      Wi-Fi link + the phone app                                │
 └──────┬──────────────────────────────────────────────────────▲────────────┘
        │ sensor messages (delayed)                            │ steering angle + motor duty
        │ app messages (delayed, lossy)                        │ (delayed)
 ┌──────▼────────────────── ONBOARD SOFTWARE (runs on the Pi) ──┴────────────┐
 │  autonomy.py    state machine, safety supervisor, wires everything below  │
 │  estimation.py  EKF localization (+ ICP scan matcher for the LiDAR)       │
 │  mapping.py     occupancy grid                                            │
 │  planning/      Hybrid A* + Reeds-Shepp, footprint collision checks       │
 │  control.py     Pure Pursuit path tracker + PID speed controller          │
 └───────────────────────────────────────────────────────────────────────────┘
        simulator.py runs the clock, moves messages between the halves,
        and scores the result against ground truth.
```

**The rule that keeps this useful: the onboard software never reads ground truth.** It only sees:
- timestamped sensor messages,
- app messages,
- the `RefMap`: what the car legitimately knows in advance, which is the testbed size, wall positions and bay pose.

That way the bottom half can later move onto the Raspberry Pi, with the simulator swapped for real drivers.

Two coordinate conventions are used throughout:
- **User-facing poses** (start, bay, summon point) are the **vehicle centre**, written `[x, y, yaw_deg]`, in metres in the testbed frame with the origin at the bottom-left corner.
- **Internal poses** are the **rear-axle centre**, `(x, y, yaw_rad)`, the natural reference point of the bicycle model. `geometry.rear_to_center` and `geometry.center_to_rear` convert between the two. The offset is `length/2 - rear_overhang` = 0.085 m.

---

## 2. Repository layout

```
Code Base/
├── README.md               how to install / run, scenario results, findings
├── ARCHITECTURE.md         this file
├── pyproject.toml          package metadata, dependencies (numpy, matplotlib, pillow; pytest for dev)
├── parksim/
│   ├── __main__.py         command-line interface: list | config | run | plan | steer-test | mc | sweep
│   ├── config.py           ALL parameters as nested dataclasses + helpers to override them
│   ├── scenarios.py        named override sets (ideal, nominal, laggy, rear_blind_spot, ...)
│   │
│   │   ── simulated world (ground truth) ──
│   ├── world.py            walls (any subset of sides), obstacles, bay; LiDAR ray casting; HC-SR04 echo
│   │                       model; true collision check; "left the testbed" check
│   ├── layouts.py          obstacle layout per run: fixed list, or random circles / lines / mix
│   ├── vehicle.py          the true car: kinematic bicycle + steering/drive/traction imperfections
│   ├── sensors.py          Encoder, Imu, UltrasonicArray, Lidar, BayCamera, DelayLine
│   ├── comms.py            Link (Wi-Fi), App (heartbeat + mission commands)
│   ├── simulator.py        main loop, scheduling, latency queues, events, scoring, logging
│   │
│   │   ── onboard software (portable to the Pi) ──
│   ├── autonomy.py         Autonomy: sensor callbacks, state machine, safety, planning/tracking glue
│   ├── estimation.py       EKF (with out-of-sequence updates), ultrasonic range model, ICP ScanMatcher
│   ├── mapping.py          OccupancyGrid: LiDAR ray update, ultrasonic cone update, distance field
│   ├── control.py          SpeedController (PID), PathTracker (Pure Pursuit, cusp handling)
│   ├── planning/
│   │   ├── hybrid_astar.py HybridAStar planner
│   │   ├── reeds_shepp.py  Reeds-Shepp shortest forward/reverse paths
│   │   ├── collision.py    FootprintChecker, EscapeChecker
│   │   └── path.py         Path container (poses + direction), split at cusps
│   │
│   │   ── shared ──
│   ├── geometry.py         angles, frames, ray casting, ultrasonic cone_echo, polygon distances
│   │
│   │   ── analysis tools ──
│   ├── steering_test.py    Ackermann / actuator bench tests (circle, servo step, speed step, tracking)
│   ├── montecarlo.py       many-seed runs (parallel), summaries, parameter sweeps
│   ├── viz.py              animation, run summary figure, planner figure
│   └── ui/
│       ├── server.py       web interface back end: options -> Config, background jobs, JSON API
│       └── index.html      the interface (plain HTML/JS, canvas replay, no external libraries)
└── tests/
    ├── test_core.py        unit tests: Reeds-Shepp, kinematics, EDT, planner, ICP, EKF, sensor models
    ├── test_scenarios.py   closed-loop regression tests (full simulations)
    └── test_ui.py          interface back end: option resolution, preset combination, API round trip
```

---

## 3. Simulated world (ground truth)

### 3.1 Testbed: `world.py`

- **Walls** are 0.15 m tall segments along the edges of the 2 × 2 m square. `world.walls` lists which sides physically exist: all four by default, `[]` for open floor, or any subset. Walls are the **only localization landmarks** for the ultrasonics and the LiDAR. A side without a wall is still a boundary: the car's map keeps a virtual keep-out ring around the testbed so plans stay inside, and the simulator records `left_testbed` (a failure) if the car crosses it.
- **Obstacles** are `Obstacle` objects: either a `circle` (x, y, r) or a `box` (x, y, length, width, yaw). **Every obstacle has a height `h`.** Whether a sensor can see an obstacle depends on that height and on the sensor's mount height. Obstacles can also be spawned mid-run.
- **The bay** is a rectangle 0.26 m wide and 0.36 m deep, against the top wall. It is only painted lines, not a physical barrier, and the camera is the only sensor that sees it.
- **Obstacle layout** comes from `layouts.generate()` and is controlled by `world.obstacle_mode`:
  - `fixed`: the `world.obstacles` list.
  - `circle`: posts/cans with radius 2.5–6 cm.
  - `line`: thin walls 20–50 cm long, 2 cm thick, at random angles. Their flat faces often send the ultrasonic ping away.
  - `random`: a mix of circles, boxes (5–15 cm) and lines.

  Generated layouts are **different for every seed**, so Monte Carlo covers many layouts. Obstacles keep clear of the start pose, the bay and its straight run-in, and the pickup pose. A layout is only accepted if the planner can solve both legs on a map of the true obstacles; otherwise it is re-drawn.
- **Obstacle timing** (`world.obstacle_timing`): `before_park` means the obstacles are there from t = 0. `after_park` means they are spawned 0.5 s after the park task finishes, so they only affect the summon.
- **Ground-truth collision:** `footprint_clearance(polygon)` gives the exact minimum distance from the car's rectangle to the walls and every obstacle, using separating-axis and polygon-distance tests. A value of 0 means contact. The simulator checks it at 50 Hz.

### 3.2 Vehicle: `vehicle.py`

The real car receives what the Pi would write to the hardware: a **steering angle command** δ_cmd and a **motor duty** u ∈ [-1, 1]. Each 5 ms physics step does the following.

**Steering servo and linkage** (`steering.*`):
```
target  = gain · δ_cmd + offset                  (calibration errors)
play    = backlash dead band on the target       (linkage slack)
δ       → first-order lag (τ = 40 ms), slew-rate limited (400 °/s), clipped to ±30°
δ_eff   = δ + jitter noise
```

**Drive motor** (`drive.*`):
```
|u| ≤ dead_zone (0.08)  → no torque
eff     = sign(u) · (|u| - dz) / (1 - dz)
v_wheel → first-order lag (τ = 0.12 s) towards eff · max_speed (0.8 m/s) · battery_factor(t)
          + load disturbance (an OU process)
```

**Traction** (`traction.*`):
```
slip    = zone slip (low-grip rectangles) + random OU slip (std 2 %)
v_body  = v_wheel · (1 - slip)       ← the encoder measures v_wheel, the car moves at v_body
κ       = tan(δ_eff) / L
        · (1 - 0.3·ackermann_error·(δ/δmax)²)   (tyre scrub from imperfect Ackermann linkage)
        / (1 + K_understeer · v²)
yaw    += v_body·κ·dt + heading random walk
x, y   += v_body · (cos, sin)(yaw at mid-step) · dt
```

This is the **kinematic bicycle model**. It is reasonable at parking speeds; the tyre effects are heuristic. `wheel_angles()` gives the true left and right front-wheel angles from the Ackermann relation `cot δo − cot δi = w/L`, and these are used for drawing.

**Default geometry:** L = 0.17 m, track 0.14 m, body 0.26 × 0.16 m, δmax = 30°, which gives R_min = L / tan δmax = 0.29 m.

### 3.3 Sensors: `sensors.py`

Each sensor produces a message dict stamped with the time the measurement was **taken**. The simulator delivers it to the software only after the sensor's latency, through a `DelayLine` (a time-ordered queue).

| Sensor | Rate | Model |
|---|---|---|
| `Encoder` | 50 Hz | Converts true **wheel** travel into whole ticks (360 per rev, r = 32 mm). Wheel-radius error (`scale_error`) makes the software's metres-per-tick wrong; missed ticks are optional. Slip is invisible to it. |
| `Imu` (gyro z) | 100 Hz | `ω_meas = (1+scale)·ω_true + bias + white noise`; the bias starts random (0.5 °/s std) and random-walks. |
| `UltrasonicArray` (HC-SR04) | one group every 60 ms | See below. |
| `BayCamera` | 15 Hz | Abstracts the OpenCV pipeline to its output: when the bay's end line is within the FOV (120°) and range (0.6 m), it reports the car's **lateral offset from the bay axis** and its **heading relative to the bay**, with noise, dropouts, gross outliers, and 80 ms capture + inference latency. |
| `Lidar` (off by default) | 10 Hz | 360 beams cast against everything taller than the mount height; noise, dropouts, spurious returns, mount-yaw error, and **motion distortion**: each beam is cast from the pose at its own moment in the rotation. |

#### The HC-SR04 model

The physics lives in `world.ultrasonic_echo` and `geometry.cone_echo`; firing and noise live in `UltrasonicArray`.

1. **Firing schedule.** Sensors listed in the same `fire_groups` entry ping at the same time, and groups take turns every `ping_interval` (60 ms, per the datasheet). The default `[[F,RL,RR],[FL,R],[FR,L]]` gives each sensor a fresh reading every 0.18 s.
2. **First echo in a cone.** A ping covers a **±15° cone**.
   - **Flat surfaces** (walls, box sides) are sampled with 31 rays. A ray only returns an echo if it hits the surface within **35° of perpendicular**; otherwise the sound reflects away (specular). The reported range is the nearest valid echo in the cone.
   - **Cylinders** are handled exactly. A cylinder always has a face pointing back at the sensor, so it echoes whenever any part of it is inside the cone.
3. **Vertical beam.** An object is only seen if `h > mount_height − r·tan(8°)`, so low objects are visible from far away but disappear when the car is close. If the nearest echo comes from an object the beam passes over, the model looks past it.
4. **Errors.** In order:
   - speed-of-sound scale error (temperature),
   - noise of `3 mm + 0.5 %·r`,
   - **crosstalk**: each other sensor in the same group has a 2 % chance of being heard, which gives a random short reading,
   - random echoes and dropouts,
   - the 2 cm blind zone and the 2 m limit, which both give NaN (no reading),
   - 3 mm quantisation.
   The message arrives after the echo's time of flight plus 5 ms of processing.

### 3.4 Wi-Fi link and app: `comms.py`

- **`Link`** behaves like a WebSocket over TCP:
  - one-way latency is 30 ms plus exponential jitter,
  - a "lost" packet is **delayed** by a retransmission (200 ms), not dropped,
  - delivery is always in order,
  - during an **outage window** everything sent or in flight is lost.
  There is one link in each direction.
- **`App`** plays the phone. It sends a **heartbeat** at 5 Hz and issues the mission commands one at a time: each is sent `wait` seconds after the previous task is reported done, and re-sent every 0.5 s until the car acknowledges it. Timed events can inject an e-stop.

---

## 4. Onboard software

### 4.1 State machine and safety: `autonomy.py`

`Autonomy` is the top-level onboard object. The simulator calls its sensor callbacks (`on_encoder`, `on_imu`, `on_ultrasonic`, `on_lidar`, `on_camera`, `on_msg`) as messages arrive, and calls `control_step()` at 50 Hz. `control_step()` returns `(steering angle, motor duty)`.

```
           command (park / summon)
  IDLE ─────────────────────────────► DRIVING ──── path finished & pose within tolerance ──► IDLE
   ▲                                  │   ▲   │
   │                 obstacle on path │   │   │ final error > 2 cm / 4° → re-plan a correction (max 2)
   │                 or in stop zone  ▼   │   │
   │                               BLOCKED ─┘ re-plan every 1 s (max 8 tries → FAULT "no_path")
   │                                  │
   │        heartbeat missing > 0.5 s │ (from DRIVING or BLOCKED)
   │                                  ▼
   │                               PAUSED ── heartbeat back → BLOCKED (re-plan, resume)
   │
   ├── ESTOP  (e-stop from the app; latched until a new command)
   └── FAULT  (localization lost / no path; waits for a new command)
```

**Safety layers**, checked every control tick while driving:
1. **Heartbeat failsafe.** No app heartbeat for 0.5 s → PAUSED, and the motor is cut.
2. **Localization confidence.** EKF position σ above 8 cm, or too many rejected LiDAR scans → FAULT. The car fails safe and stops.
3. **Ultrasonic stop zone** (`_ultrasonic_hit`). This is a box ahead of the bumper when driving forward, or behind it when reversing.
   - **Length:** `margin (2 cm) + braking distance v²/2a + v·actuation latency + v·sensor cycle·persistence`, but **never longer than the distance to where the car is about to stop anyway** (the next cusp or the goal) plus the margin and 3σ of the car's own position uncertainty. Without that cap, a planned stop near a wall makes the car stop early, re-plan the same path and stop again forever.
   - **Test:** for each sensor, the cone centre **and both cone edges** at the echo range are tested against the box, because the echo could have come from anywhere in the cone (bias toward false positives).
   - **Persistence:** it takes **2 consecutive** readings from the same sensor to stop. This filters crosstalk, and the extra `v·cycle` length compensates for the delay.
   - **Memory:** when an echo that is **not** explained by a known wall triggers a stop, the arc at that range is written into the map as occupied (`OccupancyGrid.mark_points`, log-odds 2.4). The re-plan then goes around it instead of driving back into it. Two details matter:
     - The mark is *strong but erasable*: a real obstacle keeps echoing, while a crosstalk false alarm is cleared by a few readings that see through it.
     - The arc stays one cell inside the cone edges, otherwise its ends would sit in cells that no later reading's free-space cone covers, and they could never be cleared.

     The log marks wall-explained stops with `(wall)`.
4. **Map check** (at most 10 Hz). The upcoming 0.6 m of the path is checked against the occupancy grid with the full car footprint.

The log message for each stop names the reason, for example `ultrasonic F @ 8.1 cm in stop zone`.

### 4.2 Localization: `estimation.py`

**State:** `[x, y, yaw, gyro_bias]` (rear axle).

- **Prediction** runs at encoder rate (50 Hz). It combines the encoder distance `ds` with the gyro rate averaged over the interval, minus the estimated bias:
  `x += ds·cos(yaw + Δyaw/2)`, `y += ds·sin(...)`, `yaw += (ω̄ − b)·dt`.
  Process noise covers encoder error (3 % of distance), lateral slip (1 %), gyro noise, and the bias random walk.
- **Ultrasonic update** (`update_range`):
  - **Expected value.** The predicted range comes from `geometry.cone_echo_walls`, an **exact** closed form for the nearest valid echo from flat segments, evaluated against the **known walls only**. For each wall, the ray angle ψ measured from the wall normal must lie in the cone, within ±35° (specular limit) and on the segment; the echo range is `d / cos ψ` for the smallest |ψ|. It matches 2001-ray sampling to about 10⁻⁸ m and is about 10× cheaper. The Jacobian is computed numerically. Readings near discontinuities (corners, specular edges) are skipped.
  - **Gating (χ², 99 %).** Echoes from unknown obstacles are *shorter* than the wall predicts, so they are rejected.
  - **Divergence detection** (in `autonomy.on_ultrasonic`). An echo *longer* than the wall allows cannot come from an obstacle, so it means the pose is wrong. After 4 of these in a row from the same sensor, the filter's uncertainty is inflated so it can re-converge instead of failing silently.
  - **Known limitation.** The nearest echo from a flat wall is the perpendicular one whatever the heading, so heading is mostly observed by the gyro and the camera.
- **Camera update** (`update_bay`), during parking only. Its measurement is `[lateral offset of the car centre from the bay axis, heading − bay heading]`; it pins down the lateral and heading error on the final approach.
- **LiDAR update** (when enabled). `ScanMatcher` runs Gauss-Newton point-to-line ICP of the scan against the known walls. Points far from every wall are treated as obstacles or interference and ignored. The resulting pose and covariance (with a floor for systematic errors) form an EKF update. If scans keep being gated out but ICP is confident, the filter **re-localizes** (`relocalize`).
- **Latency handling.** The filter keeps a 1 s history of every prediction and update. A late measurement is inserted at its true timestamp and everything after it is replayed. Setting `compensate_latency = False` turns this off, which is what the `laggy_no_comp` scenario does.

### 4.3 Mapping: `mapping.py`

`OccupancyGrid` is a 2 cm log-odds grid. Cells just outside the walls are fixed as occupied.

- **`update_cone` (ultrasonic).** Cells inside the cone and closer than the echo are marked free (−0.3). Cells on the arc at the echo range are marked occupied (+0.6), **unless the echo matches the range the known wall predicts**, in which case it is the wall and only the free space is updated. No echo means no update, because the surface may simply have reflected the ping away.
- **`update_scan` (LiDAR).** Free space is marked along each beam and the endpoint is marked occupied. Hits within 5 cm of a known wall count as wall.
- **Occupied threshold** is log-odds > 1.2, so a single stray hit is not an obstacle.
- **`distance_field()`** is an exact Euclidean distance transform, computed as two separable 1-D passes. The planner uses it for quick "far from everything" checks and for its heuristic.

The map is built at the **estimated** pose, so localization error corrupts it just as it would on the real car.

### 4.4 Planning: `planning/`

**`hybrid_astar.HybridAStar.plan(start, goal, grid)`** follows Dolgov et al. 2010:
- **Search space.** Cells of 5 cm × 5 cm × 5°, with a separate forward/reverse flag.
- **Motion primitives.** Arcs of 10 cm at 5 steering angles, forward and reverse. Planning uses 85 % of δmax, which keeps some steering in reserve for the tracker; the planning turning radius is therefore 0.36 m.
- **Cost.** Length (reverse × 1.5), plus a penalty per direction change, steering effort and steering changes.
- **Heuristic.** An obstacle-aware 2-D Dijkstra distance from the goal.
- **Analytic expansion.** Every few expansions the planner tries a **Reeds–Shepp** curve to a pre-goal 0.30 m before the goal, followed by a **straight run into the goal**, arriving either forwards or in reverse. The straight run lets the tracker settle the heading before stopping; without it the car ended about 6° off. A direct shot to the goal is the last resort.
- **Bounded effort.**
  - If the coarse 2-D heuristic grid shows no route at all from near the start, the planner fails immediately with "goal unreachable".
  - Otherwise it gives up after 6,000 expansions (about 15 s of Python in the worst case).
  - The 10 motion primitives of an expansion are generated and collision-checked as one batch.
  - Reeds–Shepp shots, the expensive part, are tried every 3 expansions when within 1 m of the goal (heuristic distance) and only every 30 further away.
  - The car declares `no_path` after 3 consecutive failed searches. "Goal blocked" checks are cheap and don't count; the normal re-plan budget of 8 applies to them.
- **Tight starts.** If the car is already inside the safety margin of an obstacle, the `EscapeChecker` relaxes the margin within 15 cm of the start so the car can creep away.
- **Compute time.** Simulated planning time is `0.05 s + 0.4 ms per expansion` (× `compute_factor`). It is deterministic and never taken from the real clock.

**`reeds_shepp.py`** contains the full set of 18 path families (CSC, CCC, CCCC, CCSC, CCSCC) with time-flip and reflection symmetries. `paths()` returns every candidate sorted by length and `sample()` turns one into poses. A unit test checks that every candidate ends exactly on the goal.

**`collision.FootprintChecker`** tests the exact car rectangle, inflated by the safety margin plus half a cell, against the occupied cell centres. Poses far from everything (according to the distance field) are skipped.

**`path.Path`** holds sampled poses, each with a driving direction, and `segments()` splits the path at cusps.

### 4.5 Control: `control.py`

- **`PathTracker`** is Pure Pursuit on the rear axle, run one segment at a time:
  - It finds the nearest path point, then a look-ahead point (0.10 m forward, 0.08 m reverse, plus 0.2·|v|), extending past the end of the segment along its final heading.
  - Steering: `κ = 2·y_r / d²` and `δ = atan(L·κ)`. This holds for forward and reverse driving.
  - Speed profile: `v = min(cruise 0.25, √(2·a·remaining))`, dropping to 0.08 m/s within 0.3 m of the goal.
  - It cuts the motor when the remaining distance equals the expected coast-down (`|v|·0.12 s`).
  - At each **cusp** it stops, pre-steers for the next segment, waits 0.3 s, then continues.
- **`SpeedController`** is a PID on the encoder speed, with feed-forward `v_ref / motor_max_speed` and **dead-zone compensation**: ±0.08 duty is added whenever the car should move. The integrator resets on stops and direction changes, and the controller never drives the motor in the opposite direction to brake.

---

## 5. The simulation loop: `simulator.py`

`Simulation.run()` advances in 5 ms physics steps. Everything else runs on its own clock:

| Component | Rate |
|---|---|
| Vehicle physics, collision bookkeeping | 200 Hz (dt = 5 ms) |
| IMU | 100 Hz |
| Encoder, control step, app, logging | 50 Hz |
| Ultrasonic group firing | every 60 ms |
| Camera | 15 Hz |
| LiDAR (if enabled) | 10 Hz |
| Telemetry car → app | 10 Hz |
| Heartbeat app → car | 5 Hz |

**Order of one step:**
1. Process timed events.
2. Apply actuator commands whose latency has elapsed.
3. Step the vehicle.
4. Sample the sensors that are due, into the sensor delay queue.
5. Deliver due sensor messages to `Autonomy`.
6. Deliver Wi-Fi messages; send telemetry; run the app.
7. Run the control step. A missed cycle (`control.skip_prob`) leaves the old command in place. Commands go through the actuation delay queue.
8. Check true collision and score any tasks that just finished.
9. Log.

**Timed events** (`mission.events`). Each event has a time `t`, or `"after_task": k` plus a `delay`, which fires the event that long after task k finishes; `obstacle_timing = after_park` is built on this. `spawn_obstacle` also accepts `"relative_to_car": true`, which gives the obstacle's x, y and yaw in the car's rear-axle frame at spawn time; `rear_blind_spot` uses it to drop a post 0.35 m straight behind the parked car. The event types are:
- `spawn_obstacle`: add an obstacle.
- `kidnap`: displace the car, as if someone bumped it.
- `estop`: the app sends an e-stop.
- `set`: change any config key, optionally reverting it at time `until`.

**Scoring.** When the car reports a task finished, the simulator compares the **true** car-centre pose with the goal and reports longitudinal, lateral and heading errors. Success means all three are within ±5 cm / ±5°, the task ended in `done`, and the car neither collided nor left the testbed. It also reports:
- localization error,
- minimum clearance,
- re-plan count,
- Wi-Fi latency,
- for each outage, how long and how far the car took to stop.

**Determinism.** Each component gets its own random stream from `numpy.random.SeedSequence(seed)`, and no wall-clock time enters the dynamics, so a given seed always gives the same run. Tests check this.

---

## 6. Configuration and scenarios

- **`config.py`** holds one dataclass per subsystem (`VehicleCfg`, `SteeringCfg`, ..., `UltrasonicCfg`, `EstimatorCfg`, `PlannerCfg`, ...), all collected in `Config`. Components keep a reference to their section and read it live, which is how a timed `set` event can change behaviour mid-run.
- **Overrides:**
  - `Config().set("ultrasonic.ping_interval", 0.04)`
  - `apply_overrides(cfg, {...nested dict...})`
  - on the command line: `--set key=value` or `--config file.json`
- **`scenarios.py`** contains named override sets layered on the nominal config. `scenarios.build(name)` returns a ready `Config`.

## 7. Analysis tools

| File | What it does |
|---|---|
| `steering_test.py` | **Circle test:** constant steering → fitted turning radius vs `L/tan δ`, plus the ideal inner/outer wheel angles. **Servo step:** command vs actual angle, showing lag, slew, backlash and trim. **Speed step:** the PID speed loop including the dead zone. **Tracking:** Pure Pursuit on a planned path using the *true* pose, which isolates control from localization. |
| `montecarlo.py` | Runs many seeds in parallel processes; reports success rate, mean / p95 / max errors and outcome counts; writes CSV. `sweep()` repeats this over values of one parameter and plots success and p95 error against the ±5 cm line. |
| `viz.py` | `animate()`: the map with the occupancy grid, ultrasonic cones, LiDAR points, planned path (blue forward, orange reverse), true car with Ackermann wheel angles, estimated outline, plus live speed, steering and localization plots. `summary()`: a static overview of a run. `plot_plan()`: a planner-only figure. |
| `__main__.py` | The command-line wrapper for all of the above. |
| `ui/` | `python -m parksim ui` starts a local web interface (stdlib HTTP server on 127.0.0.1:8765). See the next section. |

### The web interface (`ui/`)

- **`server.py`** turns the form into a `Config`:
  - `build_config(spec)` first combines the ticked presets (`scenarios.combine`). Lists under `mission.events`, `comm.outages` and `traction.slip_zones` are concatenated; for any other key the later preset wins and the conflict is reported.
  - It then applies walls, obstacle mode/count/timing, sensor suite, HC-SR04 firing, the optional 8th rear sensor, the free-form `key = value` overrides, and the seed. Any field left at "preset / default" is not touched.
- **Jobs** run in a background thread, one at a time, polled through `/api/job/<id>`:
  - `single`: one run, serialized for the replay with the log at about 25 Hz, plans, pings, map snapshots every second, and the summary PNG.
  - `mc`: Monte Carlo through `montecarlo.monte_carlo(callback=...)` with "spawn" worker processes, so it can be cancelled.
  - `compare`: one Monte Carlo per variant. `compare_variants()` takes the Cartesian product of the ticked values for walls × obstacles × timing × sensors × rear sensor.
- **Inlined results.** `/?job=<id>&t=<seconds>` serves the page with the options and that finished job inlined, so a reload or a link shows the result (and restores the form) right away.
- **`index.html`** draws the replay on a `<canvas>`: the occupancy grid, walls (dashed where missing), bay, obstacles (dashed red when below the sensor height), plan, trails, ultrasonic cones or LiDAR points, the true car with its wheel angles, and the estimated outline. Three time-series charts have a cursor at the current time.

## 8. Tests

- **`tests/test_core.py`** (fast) covers:
  - Reeds–Shepp endpoints,
  - turning radius against the bicycle model and the Ackermann identity,
  - the exact distance transform,
  - planner path validity (collision-free, curvature within limits) and blocked-goal reporting,
  - ICP recovery,
  - out-of-sequence EKF updates matching in-order updates,
  - LiDAR height visibility,
  - ultrasonic specular / cone / vertical-beam behaviour,
  - ultrasonic EKF convergence.
- **`tests/test_scenarios.py`** (full simulations) covers:
  - parking accuracy in `ideal` and `nominal`,
  - determinism,
  - the Wi-Fi failsafe,
  - the rear blind spot (collision with 7 sensors, none with 8),
  - the LiDAR baseline,
  - re-planning around a surprise obstacle,
  - e-stop.

## 9. Common changes

- **Measured the real chassis?** Edit `VehicleCfg` (wheelbase, track, size, δmax), then re-run `steer-test` and the scenarios.
- **Move, add or remove an ultrasonic sensor.** Edit `UltrasonicCfg.sensors` (x, y in the rear-axle frame, yaw in degrees) and `fire_groups`. Nothing else needs to change.
- **Walls / obstacle shape / timing.** Set `world.walls`, `world.obstacle_mode` (+ `n_obstacles`) and `world.obstacle_timing`, or use the CLI flags `--walls`, `--obstacles`, `--n-obstacles` and `--obstacle-timing`.
- **New scenario.** Add an entry to `SCENARIOS` in `scenarios.py`. If you change defaults and the default route moves, re-place the obstacles in `low_obstacle`, `surprise_obstacle` and `rear_blind_spot*`.
- **New sensor type.** Write the truth model in `sensors.py` (and `world.py` if it needs geometry), schedule it in `simulator.run()`, add an `on_<sensor>` callback in `autonomy.py`, then add its EKF and/or map update.
- **Porting to the Pi.** Keep `autonomy.py`, `estimation.py`, `mapping.py`, `control.py`, `planning/` and `geometry.py`. Replace the simulator with drivers that call the same `on_*` callbacks with real timestamps and run `control_step()` in the high-priority control process.
