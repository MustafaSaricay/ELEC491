"""Compose the robustness report (content JSON for build_report.js) from a scan's results.json.

    .venv/bin/python tools/report/make_content.py logs/scan_2026-11-08
"""
import json
import os
import sys

scan = sys.argv[1]
A = os.path.join(scan, "analysis")
R = json.load(open(os.path.join(A, "results.json")))
P, SW, CB = R["presets"], R["sweeps"], R["combo_by"]
acc, stops, cs = R["accuracy"], R["stops"], R["collision_summary"]


def f(x, d=1):
    return f"{x:.{d}f}"


def sweep_pt(key, val):
    return next(p for p in SW[key] if abs(p["value"] - val) < 1e-9)


n_coll = cs["n"]
rev = cs["by_direction"].get("reversing", 0)
fwd = cs["by_direction"].get("forward", 0)
hits = cs["by_hit"]
pool = R["n_runs_pool"]
fm = R["failure_modes"]
coll_runs = fm.get("collision", 0)

blocks = []
H1 = lambda t: blocks.append({"type": "h1", "text": t})
H2 = lambda t: blocks.append({"type": "h2", "text": t})
Pp = lambda t: blocks.append({"type": "p", "text": t})
B = lambda items: blocks.append({"type": "bullets", "items": items})
NOTE = lambda t: blocks.append({"type": "note", "text": t})
MAXW = {"presets.png": 500, "sweeps.png": 560, "coverage.png": 560}
IMG = lambda p, w, h, c: blocks.append({"type": "image", "path": p, "width": w, "height": h, "caption": c,
                                        "maxWidth": MAXW.get(p, 640)})


def bold(lead, rest):
    return {"runs": [{"text": lead, "bold": True}, {"text": rest}]}


# ----------------------------------------------------------------- summary
H1("1. Summary")
Pp(f"The park-and-summon stack is accurate but not yet safe. Parks that succeed land a median "
   f"{f(acc['p50'], 2)} cm from the target, far inside the ±5 cm requirement. But {coll_runs} of {pool:,} runs "
   f"({f(100 * coll_runs / pool)}%) ended in a collision, almost always with an obstacle the seven ultrasonic "
   f"sensors could not see.")
B([
    bold("Accuracy is a strength. ", f"Across {acc['n']:,} successful parks: median error {f(acc['p50'], 2)} cm, "
         f"95th percentile {f(acc['p95'], 2)} cm, worst {f(acc['max'], 1)} cm; heading error 95th percentile "
         f"{f(acc['head_p95'], 1)}°."),
    bold("The system tolerates most degradations. ", "It holds 95–100% success under noisy sensors, sensor and Wi-Fi "
         "latency, Wi-Fi drop-outs, 30% packet loss, gyro drift, steering trim and backlash up to 5–6°, a 3× motor "
         "dead zone and being bumped."),
    bold("Blind sectors are the main weakness. ", f"The 7 cones cover only 43% of directions around the car. Of the "
         f"{n_coll} collisions, {rev} happened while reversing and {fwd} while driving forward; the obstacles hit were "
         f"{hits.get('post (cylinder)', 0)} posts, {hits.get('thin line', 0)} thin walls, {hits.get('box', 0)} boxes "
         f"and {hits.get('wall', 0)} testbed walls."),
    bold("Thin, flat obstacles are the hardest. ", f"With random thin walls, {f(CB['obs']['line']['collision'], 0)}% of "
         f"runs collided, against {f(CB['obs']['circle']['collision'])}% with random posts. A flat face at an angle "
         f"reflects the ping away, so the sensor hears nothing."),
    bold("Over-caution is the second weakness. ", f"{fm.get('no path', 0)} runs gave up with “no path”, mostly because "
         f"the goal looked blocked ({R['plan_fail_reasons'].get('goal pose is blocked', 0)} such planner refusals). "
         f"{f(stops['false_share'], 0)}% of emergency stops on an echo were false alarms, and parking takes "
         f"{f(P['nominal']['park_time'])} s against {f(P['lidar_baseline']['park_time'])} s with a LiDAR."),
    bold("Localization needs walls and a known speed of sound. ", "With no walls the car cannot localize and stops "
         f"itself in every run. A 6% speed-of-sound error (roughly a 15 °C temperature mistake) drops success to "
         f"{f(sweep_pt('ultrasonic.scale_error', 0.06)['success'], 0)}%."),
])
NOTE("Top recommendations: add a rear-centre sensor (reversing collisions drop from 85% to 0% in the targeted test), "
     "close the gaps between the corner and side sensors, extend the emergency stop to the car's sides while "
     "turning, and compensate the speed of sound for temperature. Section 8 has the full list.")

# ----------------------------------------------------------------- method
H1("2. What was tested")
Pp("All runs used the parksim simulator with the current design: 7 HC-SR04 sensors, wheel encoders, a gyro and the "
   "bay camera. The car's own software (EKF localization, occupancy map, Hybrid A* with Reeds–Shepp curves, Pure "
   "Pursuit and the safety supervisor) never sees the simulator's ground truth. Each run is a park followed by a "
   "summon to the pickup point in the 2 × 2 m testbed.")
blocks.append({"type": "table", "columns": ["Group", "What varies", "Variants", "Seeds", "Runs"],
               "widths": [1400, 4638, 1300, 1000, 1300], "numeric": [2, 3, 4],
               "rows": [["Presets", "Every named scenario: noise, latency, slip, miscalibration, Wi-Fi faults, "
                         "interference, obstacles, kidnap, CPU overload, sensor firing, LiDAR baseline", "30", "20", "600"],
                        ["Combinations", "Walls (all / bottom+left / none) × obstacles (fixed / circles / thin "
                         "lines / mix) × placed before or after parking × 7 or 8 sensors", "48", "10", "480"],
                        ["Sweeps", "One parameter at a time from nominal: 12 parameters, 3–5 values each", "46", "10", "460"],
                        ["Total", "", "124", "", str(R["n_runs_total"])]],
               "caption": "Random obstacle layouts are drawn per seed and kept only if the planner can solve them on "
                          "a map of the true obstacles, so a failure points to sensing or control, not an impossible layout."})
Pp("A run succeeds when both tasks finish with the true car centre within ±5 cm and ±5° of the target, with no "
   "collision and without leaving the testbed. Two presets abort by design (e-stop and Wi-Fi without resume). The "
   "no_walls preset and the 160 no-wall combinations fail every time for a real reason, covered in section 5.5. All "
   f"of these are kept out of the failure totals so they don't swamp the other statistics; those totals therefore cover {pool:,} runs.")
Pp("Every run wrote a JSON-lines event log (the new --log-dir option). Each onboard event, such as a plan, an "
   "emergency stop or a localization warning, is stored with the ground truth at that moment: true and estimated "
   "pose, clearance, and the true free distance ahead of and behind the bumper. An emergency stop on an echo counts "
   "as a false alarm when the true free space in the direction of travel was more than 30 cm. Every collision was "
   "re-simulated to find what was hit and the direction of travel.")

# ----------------------------------------------------------------- overview
H1("3. Results overview")
Pp("Twenty of the 30 scenarios reach 90–100% success. The failures concentrate in scenarios with obstacles the "
   "sensors miss, and in the combined miscalibration and worst-case stress tests.")
IMG("presets.png", 1224, 1258, "Figure 1. Success per scenario over 20 seeds. Grey bars are below 90%; red text gives "
    "the collision rate. estop, wifi_no_resume and no_walls fail by design.")
IMG("combos.png", 1258, 510, "Figure 2. Combinations with all four walls: obstacle shape × placement × 7 or 8 "
    "sensors, 10 seeds per bar. Thin lines are the weak case; with 10 seeds, differences under about 30 points are noise.")
IMG("failures.png", 1190, 442, f"Figure 3. Outcome of the {R['n_failed']} failed runs among the {pool:,} runs that are "
    "not designed to fail.")

# ----------------------------------------------------------------- strengths
H1("4. Strengths")
H2("4.1 Parking accuracy is far inside the requirement")
Pp(f"When a task completes, the error is a small fraction of the ±5 cm allowance: 95% of parks are within "
   f"{f(acc['p95'], 2)} cm and the worst of {acc['n']:,} was {f(acc['max'], 1)} cm. The straight final approach "
   "into the bay and the camera's bay-line correction keep the heading 95th percentile at "
   f"{f(acc['head_p95'], 1)}°. Even heavy range noise (4 cm) only raises the 95th percentile to "
   f"{f(sweep_pt('ultrasonic.range_sigma', 0.04)['park_err_p95'], 2)} cm.")
IMG("accuracy.png", 1190, 408, "Figure 4. Final position error of every successful park in the scan.")
H2("4.2 It fails safe on Wi-Fi problems")
Pp(f"With a 2 s Wi-Fi outage mid-manoeuvre the heartbeat failsafe stops the car (0.76 s and 12 cm after the link "
   f"drops in the reference run), then re-plans and resumes: {f(P['wifi_dropout']['success'], 0)}% success. When "
   "resuming is disabled, all 20 runs abort without a collision. Up to 30% packet loss had no effect at all.")
H2("4.3 It tolerates actuator and calibration errors individually")
Pp("One at a time, steering trim error up to 5°, steering backlash up to 6°, a motor dead zone three times the "
   "compensated value, gyro bias drift up to 0.2 °/s/√s and 0.2 s of actuation latency all kept 100% success. "
   f"Latency mainly costs accuracy: 0.2 s raises the 95th-percentile park error from "
   f"{f(sweep_pt('control.actuation_latency', 0.0)['park_err_p95'], 2)} to "
   f"{f(sweep_pt('control.actuation_latency', 0.2)['park_err_p95'], 2)} cm.")
H2("4.4 It notices when it is lost")
Pp(f"Without walls the car has nothing to measure its position against, and it stops itself with “localization "
   f"lost” in {CB['walls']['none']['outcomes'].get('localization lost', 0)} of 160 runs instead of driving blind "
   f"({CB['walls']['none']['outcomes'].get('collision', 0)} collisions). The long-echo check widened the filter's "
   f"uncertainty in {R['loc_inflate_runs']} runs and let it re-converge. A bumped (kidnapped) car still succeeds in "
   f"{f(P['kidnap']['success'], 0)}% of runs, and the slippery floor in {f(P['slippery']['success'], 0)}%.")
H2("4.5 Obstacles in front are handled")
Pp(f"Obstacles in the front sensors' view are mapped and planned around: random posts give "
   f"{f(CB['obs']['circle']['collision'])}% collisions, and the 1 cm cable in low_obstacle is seen from 21 cm away "
   "and avoided in all 20 runs. The laggy and interference presets had no collisions in 20 runs each.")

# ----------------------------------------------------------------- weaknesses
H1("5. Weaknesses and failure modes")
Pp("Ranked by impact on safety, then on task success. Each one was traced in the event logs and by re-simulating the runs.")
H2("5.1 Blind sectors between the sensor cones (highest impact)")
Pp(f"Seven cones of ±15° cover only 43% of directions at 45 cm from the car. The largest gap is straight behind "
   f"(about 50°), and there are 20° gaps between each corner sensor and the side sensors. {rev} of the {n_coll} "
   f"collisions happened while reversing. In rear_blind_spot, a post placed behind the parked car is hit in "
   f"{f(P['rear_blind_spot']['collision'], 0)}% of runs; with an 8th sensor pointing back, "
   f"{f(P['rear_blind_spot_fixed']['collision'], 0)}%. surprise_obstacle collides in "
   f"{f(P['surprise_obstacle']['collision'], 0)}% of runs; in 6 of those 7 runs the car reversed into the box during "
   f"parking, and in the 7th it touched the box while turning slowly.")
IMG("coverage.png", 1292, 664, "Figure 5. Directions covered by at least one cone at 45 cm. The 8th sensor closes "
    "the rear gap but not the gaps beside the corner sensors.")
H2("5.2 Thin, flat obstacles reflect the ping away")
Pp(f"A flat face hit more than 35° from perpendicular returns no echo. Thin walls at random angles caused "
   f"{f(CB['obs']['line']['collision'], 0)}% collisions across the combinations, against {f(CB['obs']['circle']['collision'])}% "
   f"for posts. {hits.get('thin line', 0)} of the {n_coll} collisions were thin walls, often struck head-on with no "
   "warning event in the preceding 3 s. This is a property of ultrasound, not of the software.")
H2("5.3 No protection for the car's sides while turning")
Pp("The emergency stop zone only extends ahead of the bumper (or behind it when reversing). While turning, the "
   f"front corners and sides sweep into objects the zone never checks. Of the {fwd} forward collisions, several hit "
   "obstacles 60–95° off the car's axis, for example a post at −58° in the gap between the front-right and right "
   "sensors. The map check covers the whole footprint, but only for obstacles already in the map.")
H2("5.4 Over-caution: phantom obstacles and slow parking")
Pp(f"{fm.get('no path', 0)} runs gave up with “no path”. The planner refused "
   f"{R['plan_fail_reasons'].get('goal pose is blocked', 0)} times because the goal looked blocked, against "
   f"{R['plan_fail_reasons'].get('expansion limit', 0) + R['plan_fail_reasons'].get('search exhausted', 0)} "
   "genuinely failed searches. Wide cones smear every echo into an arc, so obstacles look larger than they are, and "
   f"crosstalk adds false echoes: {stops['false_stops']} of {stops['echo_stops']} echo stops "
   f"({f(stops['false_share'], 0)}%) had more than 30 cm of free space ahead. Parking needs "
   f"{f(P['nominal']['park_time'])} s and 1.3 re-plans on average, against {f(P['lidar_baseline']['park_time'])} s "
   "and none with the LiDAR.")
H2("5.5 Localization depends on walls and the speed of sound")
Pp(f"Ranges are only useful against known walls: no walls gives 0% success, and with one or two walls the heading "
   f"95th percentile grows to {f(P['bay_wall_only']['park_head_p95'])}–{f(P['two_walls']['park_head_p95'])}°. Because a "
   "wide cone reports the perpendicular distance to a wall whatever the heading, heading comes mostly from the gyro. "
   f"A 6% speed-of-sound error raises localization RMSE to {f(sweep_pt('ultrasonic.scale_error', 0.06)['loc_rmse_cm'])} cm "
   f"and drops success to {f(sweep_pt('ultrasonic.scale_error', 0.06)['success'], 0)}%.")
Pp("The no-wall failures have two parts. Without walls nothing corrects the position along the way, so the true "
   "error grows to 7–19 cm. The supervisor also stops the car once its own estimated uncertainty passes 8 cm, at about "
   "8.4 s into every run, even when the true error was still 4.5 cm. In a check with that threshold disabled (6 seeds), "
   "all 6 parks succeeded thanks to the bay camera, 4 of 6 summons succeeded, and one summon drove out of the testbed. "
   "The threshold is therefore a deliberate, conservative safety choice in line with the proposal's preference for "
   "false alarms over collisions. The dependence on walls is the actual limitation.")
H2("5.6 Errors that are harmless alone add up")
Pp(f"Each calibration error was tolerated alone (section 4.3), but together in miscalibrated (3° trim, 3° backlash, "
   f"5% wheel radius, 3% gyro and 3% speed-of-sound error) success falls to {f(P['miscalibrated']['success'], 0)}% "
   f"with {f(P['miscalibrated']['collision'], 0)}% collisions. worst_case, every degradation at once, reaches "
   f"{f(P['worst_case']['success'], 0)}%.")

# ----------------------------------------------------------------- sensitivity
H1("6. Sensitivity")
Pp("Starting from nominal, one parameter at a time. Most curves stay flat, which means a single degradation rarely "
   "breaks the system; the ones that bend show where the real car's limits are.")
IMG("sweeps.png", 1292, 1052, "Figure 6. Success rate against each parameter, 10 seeds per point. The dashed line marks 90%.")
rows = []
for key, label, unit in [("control.cruise_speed", "Cruise speed", "m/s"), ("ultrasonic.crosstalk_prob", "Ultrasonic crosstalk", ""),
                         ("ultrasonic.scale_error", "Speed-of-sound error", ""), ("traction.slip_std", "Wheel slip", "std"),
                         ("ultrasonic.range_sigma", "Range noise", "m"), ("control.actuation_latency", "Actuation latency", "s")]:
    pts = SW[key]
    safe = max((p["value"] for p in pts if p["success"] >= 100), default=None)
    bad = next((p for p in pts if p["success"] < 100), None)
    last = pts[-1]
    effect = (f"{f(bad['success'], 0)}% at {bad['value']:g}" if bad else "none in range") + \
             f"; park p95 {last.get('park_err_p95', '–')} cm, false stops {last['false_stops_per_run']}/run at {last['value']:g}"
    rows.append([label + (f" ({unit})" if unit else ""), f"{pts[0]['value']:g}–{pts[-1]['value']:g}",
                 f"{safe:g}" if safe is not None else "–", effect])
blocks.append({"type": "table", "columns": ["Parameter", "Tested", "100% up to", "First drop, and effect at the largest value"],
               "widths": [2300, 1300, 1300, 4738], "rows": rows,
               "caption": "With 10 seeds per point a single failed run moves the rate by 10 points, so read single dips "
                          "(crosstalk 0.05) as noise and steady trends as real."})

# ----------------------------------------------------------------- model limits
H1("7. Limits of the simulation model")
B([
    bold("Vehicle: ", "a kinematic bicycle model with first-order actuator lag. There are no tyre forces, weight "
         "transfer or inertia beyond that, and slip and understeer are approximations. This is reasonable at parking "
         "speeds of 0.3 m/s and below."),
    bold("Ultrasonic sensors: ", "geometric only: a uniform ±15° cone, a hard 35° reflection limit and range-dependent "
         "noise. Echo amplitude, beam side lobes, multipath, soft or sound-absorbing materials and floor echoes are not "
         "modelled. Real HC-SR04s may be better (wider beam) or worse (multipath) than this."),
    bold("World: ", "static obstacles that are vertical extrusions of circles and rectangles. There are no moving "
         "people or objects, and collisions are exact geometric overlap."),
    bold("Camera: ", "modelled as its output (lateral offset and heading from the bay lines, with noise, dropouts and "
         "latency). Image-processing failures such as glare or worn lines are not simulated."),
    bold("Compute: ", "planning time is modelled from the search effort and missed control cycles are random. The "
         "Pi's real scheduling, and the separation of the control loop into its own process, are not simulated."),
    bold("Statistics: ", "10–20 seeds per variant give roughly ±10–20 percentage points of uncertainty. For example, "
         "the 7- versus 8-sensor bars in the combinations differ by less than that in most cells and should not be "
         "over-read; the targeted rear_blind_spot test is the reliable comparison."),
    bold("Tuning: ", "the car's parameters were tuned on this simulator, so real-vehicle calibration of the steering, "
         "encoders and sensor mounting is still needed before these numbers transfer."),
])

# ----------------------------------------------------------------- recommendations
H1("8. Recommendations")
Pp("Ordered by expected safety benefit per unit of effort.")
blocks.append({"type": "table", "columns": ["#", "Change", "Addresses", "Expected effect"],
               "widths": [500, 3900, 1500, 3738], "rows": [
    ["1", "Add an 8th HC-SR04 pointing straight back", "5.1", f"Reversing into a post behind: {f(P['rear_blind_spot']['collision'], 0)}% → {f(P['rear_blind_spot_fixed']['collision'], 0)}% collisions"],
    ["2", "Close the gaps beside the corner sensors: turn FL/FR to ±50° and add two sensors at ±65°, or add short-range ToF sensors (the proposal's VL53-type option) at the corners", "5.1, 5.3", "Closes the 20° gaps beside the corners; not yet simulated, so re-run the scan with the new layout"],
    ["3", "Software: check the car's swept outline over the next 0.3 m (both sides) against the latest echoes of every sensor, not only the zone ahead", "5.3", "Should catch side swipes during turns; not yet simulated"],
    ["4", "Measure air temperature (or calibrate once at the venue) and correct the speed of sound", "5.5", "Keeps speed-of-sound error under 1%; 6% cost 30 points of success"],
    ["5", "Keep cruise speed at or below 0.35 m/s", "5.4", "At 0.45 m/s: 80% success and 1.5 false stops per run"],
    ["6", "Choose demo obstacles that echo (cylinders, boxes ≥ 5 cm facing the car) and test thin flat objects explicitly", "5.2", "Avoids the 20%-collision case; documents the limitation honestly"],
    ["7", "Map: decay stale arc evidence near the goal faster and prefer cone-centre evidence over arc edges", "5.4", "Fewer “goal blocked” refusals and shorter parking"],
    ["8", "Calibrate steering trim, servo gain, wheel radius and gyro scale before integration tests", "5.6", "Avoids the combined-error drop to 60%"],
    ["9", "Keep at least two non-parallel walls in the testbed", "5.5", "Localization needs them; open floor gives 0%"],
               ]})

# ----------------------------------------------------------------- appendix
H1("9. Appendix: reproducing the study")
B([
    bold("Scan: ", ".venv/bin/python tools/scan.py --out logs/scan_2026-11-08 (1,550 runs, about 70 minutes on 7 cores)."),
    bold("Analysis and charts: ", ".venv/bin/python tools/analyze_scan.py logs/scan_2026-11-08"),
    bold("This report: ", ".venv/bin/python tools/report/make_content.py logs/scan_2026-11-08, then node tools/report/build_report.js on the content file."),
    bold("Logging a single run: ", "python -m parksim run -s surprise_obstacle --log-dir logs/mine (works for mc and sweep too)."),
    bold("Log format: ", "one JSON object per line: run (config), event (t, kind, msg, true/est pose, loc error, clearance, "
         "free distance ahead/behind), state (1 Hz), task (scored result), metrics. Event kinds include plan, plan_fail, "
         "stop_echo, stop_wall, stop_map, link_lost, loc_lost, loc_inflate, correction, collision."),
    bold("Determinism: ", "every run is reproducible from its group, variant and seed (listed in logs/scan_2026-11-08/runs.csv)."),
])

content = {"title": "Autonomous Parking Simulation: Robustness Study",
           "subtitle": f"Strengths and weaknesses of the park-and-summon system, from {R['n_runs_total']:,} logged simulations",
           "meta": "ELEC 491 Design Project · 8 November 2026 · design under test: 7× HC-SR04 ultrasonic configuration",
           "footer": "parksim robustness study", "toc": False, "blocks": blocks}
out = os.path.join(A, "report_content.json")
json.dump(content, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out, len(blocks), "blocks")
