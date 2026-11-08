# parksim: simulation environment for the autonomous parking & summoning vehicle

A closed-loop 2-D simulator for the ELEC 491 project. It tests **Ackermann steering**, **path planning**, and the **complete park-and-summon stack** under noise, delay, interference, miscalibration and failures.

The default car uses **7 HC-SR04 ultrasonic sensors** (front centre, two front corners, both sides, two rear corners) plus encoders, a gyro and the bay camera. The original 360° LiDAR is still available for comparison (`-s lidar_baseline`).

```
              ground truth (scoring only)                      onboard software (portable to the Pi)
 ┌──────────────────────────────────────────────┐           ┌───────────────────────────────────────────┐
 │ World: 2x2 m walls, obstacles WITH HEIGHT    │  sensors  │ EKF (odom+gyro) ◄─ ultrasonic range vs    │
 │ Vehicle: bicycle model + servo/motor/slip    │ ─latency► │   known walls (+ camera bay-line update)  │
 │ Encoder · IMU · 7x HC-SR04 · bay camera      │           │ Occupancy grid (cone sensor model)        │
 │ Wi-Fi link (latency, jitter, loss, outages)  │ ◄─latency │ Hybrid A* + Reeds-Shepp → Pure Pursuit    │
 │ Mobile app: heartbeat + park/summon/e-stop   │  commands │ PID speed · safety supervisor · failsafe  │
 └──────────────────────────────────────────────┘           └───────────────────────────────────────────┘
```

**How every component is modelled, and what each file does: see [ARCHITECTURE.md](ARCHITECTURE.md).**

The onboard modules (`autonomy.py`, `estimation.py`, `mapping.py`, `control.py`, `planning/`) never read ground truth. They receive only timestamped sensor messages, app messages and the reference map (wall positions and bay pose), so they can be moved onto the Raspberry Pi largely unchanged.

## Setup

```bash
cd "Code Base"
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
source .venv/bin/activate
```

## The interface (easiest way in)

```bash
python -m parksim ui          # opens http://127.0.0.1:8765 in your browser (Ctrl+C to stop)
```

- **Left panel.** Tick any number of **scenario presets** to combine them (e.g. `laggy` + `wifi_dropout` + `surprise_obstacle`). Then choose **walls**, **obstacle shape / count / timing**, **sensor suite**, **HC-SR04 firing** and an optional **8th rear sensor**. You can also override any parameter with autocomplete.
- **Run & replay.** An interactive replay: play, pause, scrub, speed, and layer toggles for the map, sensor cones, estimate, plan and trails. It comes with speed, steering and localization charts, the task results, and a clickable event log. The *Summary figure* tab shows the static overview.
- **Monte Carlo.** Many seeds with the current settings. You get success and collision rates, a per-task error table, and a per-seed table where **replay** opens that exact run.
- **Combinations.** Tick several values for walls × obstacles × timing × sensors × rear sensor. Every combination is run as a Monte Carlo and shown in one table, with **load** (put a variant into the form) and **replay fail** (open the first failing seed).
- **Preview config.** Shows the fully resolved configuration and warns when combined presets set the same value differently.
- **Sharing.** The URL keeps the last result (`?job=...`), so a reload brings it back while the server is running.

## Commands

```bash
python -m parksim list                          # scenarios
python -m parksim run -s nominal --animate      # live animation (add --summary for plots)
python -m parksim run -s rear_blind_spot --save-anim out.gif --save-summary out.png --no-show
python -m parksim plan -s nominal --goal 0.4 0.4 180     # planner only
python -m parksim steer-test -s miscalibrated   # Ackermann/actuator bench tests
python -m parksim mc -s worst_case -n 50 --csv runs.csv  # Monte Carlo over seeds
python -m parksim sweep --param ultrasonic.crosstalk_prob --values 0 0.02 0.05 0.1 -n 10
python -m parksim config -s laggy               # print the resolved config as JSON
python -m parksim run -s nominal --log-dir logs/mine     # JSON-lines event log per run (also mc / sweep)
python tools/scan.py --out logs/scan             # full logged robustness scan (~1,550 runs)
python tools/analyze_scan.py logs/scan           # results.json + charts; collisions re-simulated
python tools/report/make_content.py logs/scan    # report content -> node tools/report/build_report.js
python -m pytest                                # 31 tests (~1.5 min)
```

### World options: walls, obstacle shape, obstacle timing

```bash
python -m parksim run --walls none --animate                 # open floor (also: all, or e.g. bottom,left)
python -m parksim run --obstacles random --n-obstacles 5     # fixed | circle | line | random
python -m parksim run --obstacle-timing after_park           # obstacles appear only after the car has parked
python -m parksim mc --obstacles line -n 30                  # new random layout for every seed
```

| Option | Values | Effect |
|---|---|---|
| `world.walls` / `--walls` | `all`, `none`, any of `bottom,right,top,left` | Walls are the only landmarks the ultrasonics (and LiDAR) can localize against. Missing walls are a virtual boundary; crossing one is a failure (`left_testbed`). |
| `world.obstacle_mode` / `--obstacles` | `fixed`, `circle`, `line`, `random` | `fixed` = the `world.obstacles` list; the others generate `n_obstacles` per seed, always solvable and clear of the start, bay and pickup |
| `world.obstacle_timing` / `--obstacle-timing` | `before_park`, `after_park` | `after_park` places the obstacles 0.5 s after parking finishes, so only the summon has to deal with them |

You can change any parameter with `--set section.key=value` (values are parsed as JSON), or with `--config overrides.json` (a nested dict):

```bash
python -m parksim run --set ultrasonic.half_angle_deg=20 --set ultrasonic.ping_interval=0.04 --animate
```

A run prints each task's final **lateral, longitudinal and heading error against ground truth**, the minimum clearance, localization RMSE, Wi-Fi latency and failsafe stop time/distance. Every emergency stop names the sensor and range that triggered it, for example `ultrasonic F @ 8.1 cm in stop zone`.

## The HC-SR04 model

- **Layout and firing order:** the layout is `ultrasonic.sensors` (name, x, y, yaw in the rear-axle frame). Sensors in the same `fire_groups` entry ping together, and the groups fire one after another every `ping_interval` (60 ms, per the datasheet). The default is 3 groups, so each sensor updates every 0.18 s.
- **Cone and first echo:** a reading is the **first echo inside a ±15° cone**. Flat surfaces hit more than 35° from their normal reflect the ping away (**specular: no echo**). Cylinders always echo.
- **Vertical beam:** a low object is seen only once it is far enough to enter the beam: `h > mount_height - r·tan(vertical_half)`.
- **Errors:** blind zone below 2 cm, 2 m range limit, noise growing with range, 3 mm quantisation, speed-of-sound error (temperature), dropouts, random echoes, and **crosstalk** between sensors fired in the same group.

On the car:
- **Localization:** the EKF compares each echo with the range the known walls predict (same cone model, numeric Jacobian). Shorter echoes come from obstacles and are gated out. Echoes that are persistently *longer* than the wall allows mean the pose estimate is wrong; the car then widens its uncertainty rather than failing silently.
- **Mapping:** the cone before the echo is marked free, and the arc at the echo range is marked occupied unless the known wall explains the echo.
- **Emergency stop:** the cone centre and both edges are tested against a stop zone. It takes 2 consecutive echoes to stop, and the zone is lengthened by speed × cycle time to make up for that delay.

## Effects you can inject

Every value lives in [parksim/config.py](<Code Base/parksim/config.py>) and is documented there.

| Area | Effects |
|---|---|
| Steering (`steering.*`) | gain error, trim offset, backlash, slew-rate limit, servo lag, jitter, non-ideal Ackermann linkage (tyre scrub) |
| Drive (`drive.*`) | dead zone, motor lag, battery sag over time, load disturbance |
| Traction (`traction.*`) | longitudinal slip (encoder sees the wheel, the body moves less), low-grip zones, heading random walk, understeer |
| Encoder / IMU | resolution, wheel-radius error, missed ticks, gyro noise, bias, bias drift, scale error, latency |
| Ultrasonic (`ultrasonic.*`) | layout, firing groups and interval, cone width, specular cut-off, mount height and vertical cone, noise, speed-of-sound error, dropouts, random echoes, **crosstalk**, latency |
| LiDAR (`lidar.*`, off by default) | noise, dropouts, spurious returns, mount height, mount yaw, motion distortion, latency |
| Camera (`camera.*`) | noise, dropouts, gross outliers, processing/**inference latency**, FOV, range |
| Wi-Fi (`comm.*`) | latency, jitter, packet loss → TCP retransmit delay, outage windows, heartbeat timeout, resume vs. abort |
| Compute | missed control cycles (`control.skip_prob`), planner compute time (`planner.compute_*`), actuation latency |

**Timed events** go in `mission.events`: `spawn_obstacle`, `kidnap`, `estop`, and `set` (change any config key mid-run, optionally reverting it with `until`):
```json
{"t": 3.0, "type": "spawn_obstacle", "obstacle": {"type": "circle", "x": 1.2, "y": 1.1, "r": 0.025, "h": 0.05}}
{"t": 5.0, "type": "set", "key": "ultrasonic.crosstalk_prob", "value": 0.3, "until": 9.0}
```

## Scenario results (single run, seed 0, 7× HC-SR04)

| Scenario | Outcome |
|---|---|
| ideal, nominal, noisy_sensors, laggy, laggy_no_comp, slippery, miscalibrated, weak_motor, interference, low_obstacle, surprise_obstacle, kidnap, cpu_overload, fire_all_at_once, fire_one_by_one, lidar_baseline | park + summon succeed |
| wifi_dropout | the car stops about 0.76 s / 12 cm after the link drops, then re-plans and resumes |
| wifi_no_resume / estop | the task is aborted as designed; the next command works |
| **rear_blind_spot** | **crash**: a 5 cm post 0.35 m straight behind the parked car is never seen when it reverses out, because it falls between the two ±140° rear cones |
| rear_blind_spot_fixed | an 8th sensor pointing straight back sees the post, the car maps it and plans around it, and the summon succeeds |
| worst_case | park succeeds; summon ends 5.2° off heading (limit is 5°) |

**Monte Carlo, 20 seeds, ultrasonic vs LiDAR:** both are 100% successful.

| | Park error mean / max | Park heading p95 | Park time | Re-plans per park |
|---|---|---|---|---|
| 7× HC-SR04 | 0.40 / 0.85 cm | 3.9° | 18.9 s | 1.4 |
| LiDAR | 0.19 / 0.42 cm | 0.7° | 10.8 s | 0 |

**Random layouts, Monte Carlo, 20 seeds each (a new layout every seed; 7× HC-SR04):**

| Scenario | Success | Collisions | Main failure |
|---|---|---|---|
| random_circles | 80% | 5% | `no_path` (3): arc-smeared posts make gaps look closed; 1 collision |
| random_lines | 85% | 10% | thin walls hit at an angle give no echo, and the rear blind spot |
| random_mix | 90% | 0% | `no_path` (2) on the summon |
| random_after_park | 85% | 15% | the summon reverses out of the bay into obstacles behind the car (blind spot) |
| no_walls (single run) | stops with `loc_lost` | none | nothing to range against, so the uncertainty grows past 8 cm and the car stops itself |

## Findings about the ultrasonic design

1. **Rear blind spot.** With only ±140° rear corners, nothing covers straight behind (180° ± 25°). This matters because the car reverses into and out of the bay.
2. **Heading comes from the gyro, not the sensors.** A cone sensor reports the *perpendicular* distance to a flat wall whatever the heading, so walls barely constrain heading. Heading relies on the gyro and, at the bay, the camera. A poor gyro hurts more than it did with the LiDAR.
3. **Crosstalk versus update rate.** Firing in groups is faster (0.18 s per cycle) but sensors hear each other's pings. Stopping on a single echo caused spurious stops every few seconds, so 2 echoes are required and the stop zone is lengthened to compensate. Firing one sensor at a time removes crosstalk but means 0.42 s between readings.
4. **Wide cones blur the map.** An echo could come from anywhere on a 30° arc, so obstacles appear as smeared arcs and the planner re-plans more often. That is where the extra parking time comes from.
5. **Low obstacles.** At a 4 cm mount height the vertical cone sees a 1 cm cable from more than 21 cm away, but not when the car is right on top of it.

## Model notes and limits

- The model is kinematic (no tyre dynamics). That is reasonable at parking speeds (≤ 0.3 m/s), but slip and understeer are heuristic.
- The default car is 0.26 × 0.16 m with L = 0.17 m and δmax = 30°, which gives R_min = 0.29 m. Change `vehicle.*` once the real chassis is measured.
- The ultrasonic model is geometric. It does not simulate amplitude, multipath, soft or absorbing materials, or floor echoes.
- User-facing poses are **vehicle-centre** `[x, y, yaw_deg]`; internally the software uses the **rear axle**.
- Only the testbed walls are known in advance; obstacles are discovered by the sensors.
- Python speed: a 35 s mission takes about 10 s to simulate, and `mc` runs in parallel.
