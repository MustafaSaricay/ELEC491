"""Full robustness scan with event logging.

    .venv/bin/python tools/scan.py --out logs/scan            # ~1540 runs, ~30-40 min on 8 cores
    .venv/bin/python tools/scan.py --out logs/quick --quick   # 2 seeds per variant, for testing

Groups
  preset  every named scenario x 20 seeds
  combo   walls x obstacle shape x obstacle timing x 7/8 sensors, 10 seeds each
  sweep   one parameter at a time from nominal, 10 seeds per value

Writes
  <out>/logs/<group>/<variant>/<variant>_s<seed>.jsonl   event log per run (parksim/eventlog.py)
  <out>/runs.csv                                         one row per run (appended as runs finish)
  <out>/progress.txt                                     done / total
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SWEEPS = {
    "ultrasonic.range_sigma": [0.003, 0.01, 0.02, 0.04],
    "ultrasonic.crosstalk_prob": [0.0, 0.02, 0.05, 0.1, 0.2],
    "ultrasonic.dropout_prob": [0.0, 0.05, 0.15, 0.3],
    "ultrasonic.scale_error": [0.0, 0.01, 0.03, 0.06],
    "traction.slip_std": [0.0, 0.04, 0.08, 0.15],
    "imu.bias_walk_dps": [0.01, 0.05, 0.1, 0.2],
    "steering.offset_deg": [0.0, 1.0, 3.0, 5.0],
    "steering.backlash_deg": [0.0, 1.0, 3.0, 6.0],
    "control.actuation_latency": [0.0, 0.05, 0.1, 0.2],
    "comm.loss_prob": [0.0, 0.1, 0.3],
    "control.cruise_speed": [0.15, 0.25, 0.35, 0.45],
    "drive.dead_zone": [0.08, 0.15, 0.25],
}
COMBO = {"walls": ["all", "bottom+left", "none"], "obstacle_mode": ["fixed", "circle", "line", "random"],
         "obstacle_timing": ["before_park", "after_park"], "rear_sensor": [False, True]}
FALSE_ALARM_FREE = 0.30   # an echo stop with more true free space than this ahead = false alarm


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_")[:90]


def jobs(seeds_preset, seeds_other):
    from parksim import scenarios
    out = []
    for name in scenarios.SCENARIOS:
        for s in range(seeds_preset):
            out.append(("preset", name, {"presets": [name]}, "", "", s))
    import itertools
    for combo in itertools.product(*COMBO.values()):
        spec = dict(zip(COMBO.keys(), combo))
        label = f"walls={spec['walls']}|obs={spec['obstacle_mode']}|{spec['obstacle_timing']}|" \
                f"{'8' if spec['rear_sensor'] else '7'}sensors"
        for s in range(seeds_other):
            out.append(("combo", label, spec, "", "", s))
    for key, values in SWEEPS.items():
        for v in values:
            spec = {"overrides": [{"key": key, "value": json.dumps(v)}]}
            for s in range(seeds_other):
                out.append(("sweep", f"{key}={v}", spec, key, v, s))
    return out


def run_one(args):
    group, variant, spec, param, value, seed, out = args
    from parksim.simulator import Simulation
    from parksim.ui.server import build_config
    cfg, _ = build_config({**spec, "seed": seed})
    cfg.sim.log_dir = os.path.join(out, "logs", group, slug(variant))
    cfg.sim.log_name = slug(variant)
    t0 = time.time()
    row = {"group": group, "variant": variant, "param": param, "value": value, "seed": seed}
    try:
        sim = Simulation(cfg)
        res = sim.run()
    except Exception as exc:                     # a crash is a result too
        row.update(error=repr(exc)[:200], wall_s=round(time.time() - t0, 1))
        return row
    m = res.metrics
    row.update(wall_s=round(time.time() - t0, 1), all_success=m["all_success"], collided=m["collided"],
               left_testbed=m["left_testbed"], loc_rmse_cm=round(m["loc_rmse"] * 100, 3),
               loc_max_cm=round(m["loc_err_max"] * 100, 3), min_clear_cm=round(m["min_clearance"] * 100, 2),
               sim_time=round(m["sim_time"], 2), log=os.path.relpath(sim.elog.path, out))
    for tk in res.tasks:
        p = tk["kind"]
        row.update({f"{p}_result": tk["result"], f"{p}_ok": tk["success"],
                    f"{p}_err_cm": round(tk["err_pos"] * 100, 3), f"{p}_lat_cm": round(tk["err_lat"] * 100, 3),
                    f"{p}_long_cm": round(tk["err_long"] * 100, 3), f"{p}_head_deg": round(tk["err_heading_deg"], 3),
                    f"{p}_time_s": round(tk["time"], 2), f"{p}_replans": tk["replans"],
                    f"{p}_corrections": tk["corrections"], f"{p}_clear_cm": round(tk["min_clearance"] * 100, 2)})
    # event counts + false-alarm classification of emergency stops, from the log
    counts, false_stops, true_stops, first_fail = {}, 0, 0, ""
    with open(sim.elog.path) as f:
        for line in f:
            r = json.loads(line)
            if r["type"] != "event":
                continue
            k = r["kind"]
            counts[k] = counts.get(k, 0) + 1
            if k in ("stop_echo", "stop_lidar"):
                free = r["free_front"] if (r.get("v") or 0) >= 0 else r["free_rear"]
                if free is not None and free > FALSE_ALARM_FREE:
                    false_stops += 1
                else:
                    true_stops += 1
            if not first_fail and k in ("collision", "left_testbed", "loc_lost", "plan_fail"):
                first_fail = f"{k}@{r['t']:.1f}s"
    row.update({f"ev_{k}": v for k, v in counts.items()})
    row.update(false_stops=false_stops, true_stops=true_stops, first_problem=first_fail)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/scan")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = jobs(2, 2) if a.quick else jobs(20, 10)
    js = [(*j, a.out) for j in js]
    total = len(js)
    print(f"{total} runs -> {a.out}  ({a.workers} workers)", flush=True)
    rows, done, t0 = [], 0, time.time()
    fields = None
    csv_path = os.path.join(a.out, "runs.csv")
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, j) for j in js]
        for f in as_completed(futs):
            row = f.result()
            rows.append(row)
            done += 1
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                msg = f"{done}/{total} runs, {el/60:.1f} min elapsed, ~{el/done*(total-done)/60:.1f} min left"
                print(msg, flush=True)
                with open(os.path.join(a.out, "progress.txt"), "w") as p:
                    p.write(msg + "\n")
    # one CSV with the union of all columns
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("group", "variant", "param", "value", "seed"), k))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["group"], r["variant"], r["seed"])))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
