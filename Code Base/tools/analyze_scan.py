"""Analyse a scan made by tools/scan.py.

    .venv/bin/python tools/analyze_scan.py logs/scan_2026-11-08

Writes <scan>/analysis/results.json (all numbers the report uses) and
<scan>/analysis/*.png (charts). Collisions are re-simulated (runs are
deterministic) to find what was hit and whether the car was reversing.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

BLUE, ORANGE, AQUA, GREY, LIGHT, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#d9d8d4", "#0b0b0b", "#52514e"
RED = "#e34948"
NICE = {"comm.loss_prob": "Wi-Fi packet loss", "control.actuation_latency": "actuation latency (s)",
        "control.cruise_speed": "cruise speed (m/s)", "drive.dead_zone": "motor dead zone (duty)",
        "imu.bias_walk_dps": "gyro bias drift (deg/s/sqrt s)", "steering.backlash_deg": "steering backlash (deg)",
        "steering.offset_deg": "steering trim error (deg)", "traction.slip_std": "wheel slip (std)",
        "ultrasonic.crosstalk_prob": "ultrasonic crosstalk (prob.)", "ultrasonic.dropout_prob": "ultrasonic dropout (prob.)",
        "ultrasonic.range_sigma": "range noise (m)", "ultrasonic.scale_error": "speed-of-sound error"}


def pct(x):
    return round(100.0 * float(x), 1)


# --------------------------------------------------------------------------- #
def load(scan):
    df = pd.read_csv(os.path.join(scan, "runs.csv"))
    for c in ("all_success", "collided", "left_testbed", "park_ok", "summon_ok"):
        if c in df:
            df[c] = df[c].astype("boolean")
    for c in [c for c in df.columns if c.startswith("ev_")] + ["false_stops", "true_stops"]:
        df[c] = df[c].fillna(0).astype(int)
    return df


def outcome(r):
    """One label per run: the first thing that went wrong."""
    if isinstance(r.get("error"), str) and r["error"]:
        return "crash (exception)"
    if r["collided"]:
        return "collision"
    if r["left_testbed"]:
        return "left testbed"
    for p in ("park", "summon"):
        res = r.get(f"{p}_result")
        if isinstance(res, str) and res != "done":
            return {"no_path": "no path", "loc_lost": "localization lost", "link_lost": "Wi-Fi lost",
                    "estop": "e-stop (by design)", "timeout": "timeout"}.get(res, res)
    for p in ("park", "summon"):
        if r.get(f"{p}_ok") is False:
            return "out of tolerance"
    return "success"


def summarize(g):
    n = len(g)
    d = {"runs": n, "success": pct(g["all_success"].mean()), "collision": pct(g["collided"].mean()),
         "outcomes": dict(Counter(g["outcome"]))}
    for p in ("park", "summon"):
        ok = g[g[f"{p}_ok"] == True] if f"{p}_ok" in g else g.iloc[0:0]  # noqa: E712
        if len(ok):
            d[f"{p}_err_p50"] = round(float(ok[f"{p}_err_cm"].median()), 2)
            d[f"{p}_err_p95"] = round(float(ok[f"{p}_err_cm"].quantile(0.95)), 2)
            d[f"{p}_head_p95"] = round(float(ok[f"{p}_head_deg"].abs().quantile(0.95)), 2)
            d[f"{p}_time"] = round(float(ok[f"{p}_time_s"].mean()), 1)
    d["false_stops_per_run"] = round(float(g["false_stops"].mean()), 2)
    d["loc_rmse_cm"] = round(float(g["loc_rmse_cm"].mean()), 2)
    return d


# --------------------------------------------------------------------------- #
def forensic(args):
    """Re-run one collided run and describe the impact."""
    group, variant, seed = args
    import scan
    from parksim.simulator import Simulation
    from parksim.ui.server import build_config
    spec = next(j[2] for j in scan.jobs(1, 1) if j[0] == group and j[1] == variant)
    cfg, _ = build_config({**spec, "seed": int(seed)})
    sim = Simulation(cfg)
    res = sim.run()
    veh, w = sim.vehicle, sim.world
    fp = veh.footprint()
    hit, best = "wall", min(w._side_gaps(fp)[s] for s in w.wall_sides) if w.wall_sides else math.inf
    for o in w.obstacles:
        d = o.distance_to_polygon(fp)
        if d < best:
            best = d
            if o.kind == "circle":
                hit = "post (cylinder)"
            elif min(o.l, o.w) <= cfg.world.line_thickness + 1e-6:
                hit = "thin line"
            else:
                hit = "box"
            low = o.h <= cfg.ultrasonic.mount_height
            if low:
                hit += " (below sensor height)"
            spawned = o.spawn_t > 0
    ev = next((e for e in res.events if "COLLISION" in e[1]), None)
    # direction of travel just before impact
    log = res.log
    i = int(np.searchsorted(log["t"], sim.collision_t)) - 3
    v = float(log["v"][max(i, 0)])
    task = next((t["kind"] for t in res.tasks if t["result"] == "collision"), "?")
    return {"group": group, "variant": variant, "seed": int(seed), "hit": hit,
            "direction": "reversing" if v < -0.01 else ("forward" if v > 0.01 else "slow/turning"),
            "task": task, "t": round(sim.collision_t or 0, 2),
            "spawned_obstacle": bool(locals().get("spawned", False))}


def plan_fail_reasons(scan, df):
    c = Counter()
    for path in df.loc[df["ev_plan_fail"] > 0, "log"].dropna():
        with open(os.path.join(scan, path)) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "event" and r["kind"] == "plan_fail":
                    c[r["msg"].split(": ", 1)[-1].split(" (")[0]] += 1
    return dict(c)


def stop_stats(scan, df, sample=400):
    """Echo-stop events: true free distance ahead vs the echo range."""
    rows = []
    for path in df["log"].dropna().iloc[:sample]:
        with open(os.path.join(scan, path)) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "event" and r["kind"] in ("stop_echo", "stop_wall"):
                    free = r["free_front"] if (r.get("v") or 0) >= 0 else r["free_rear"]
                    rows.append({"kind": r["kind"], "free": free, "v": r.get("v")})
    return rows


# --------------------------------------------------------------------------- #
def charts(out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": LIGHT,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 1. presets: success vs collision, sorted
    pr = sorted(res["presets"].items(), key=lambda kv: (kv[1]["success"], -kv[1]["collision"]))
    names = [k for k, _ in pr]
    succ = [v["success"] for _, v in pr]
    coll = [v["collision"] for _, v in pr]
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    y = np.arange(len(names))
    ax.barh(y, succ, color=[BLUE if s >= 90 else GREY for s in succ], height=0.62)
    for yi, s, c in zip(y, succ, coll):
        ax.text(s + 1, yi, f"{s:.0f}%" + (f"  ({c:.0f}% collisions)" if c > 0 else ""), va="center",
                fontsize=8, color=RED if c > 0 else INK2)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(90, color=LIGHT, lw=1, ls="--")
    ax.set_xlabel("runs where park AND summon met ±5 cm / ±5° (%)")
    ax.set_title("Success rate per scenario (20 seeds each)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "presets.png"), dpi=170)
    plt.close(fig)

    # 2. combos: obstacle shape x timing, 7 vs 8 sensors (walls = all)
    cb = res["combo_table"]
    modes, timings = ["fixed", "circle", "line", "random"], ["before_park", "after_park"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, tm in zip(axs, timings):
        x = np.arange(len(modes))
        for k, (sens, col) in enumerate((("7", GREY), ("8", BLUE))):
            vals = [cb.get(f"all|{m}|{tm}|{sens}", {}).get("success", np.nan) for m in modes]
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=f"{sens} sensors")
            for xi, v in zip(x, vals):
                ax.text(xi + (k - 0.5) * 0.36, v + 2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x, modes)
        ax.set_ylim(0, 112)
        ax.set_title("obstacles placed " + ("before parking" if tm == "before_park" else "after parking"),
                     fontsize=9, color=INK2, loc="left")
    axs[0].set_ylabel("success (%)")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Adding a rear-centre sensor, all four walls (10 seeds per bar)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "combos.png"), dpi=170)
    plt.close(fig)

    # 3. sweeps: small multiples of success
    sw = res["sweeps"]
    keys = list(sw.keys())
    cols = 4
    rows = math.ceil(len(keys) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(7.6, 2.0 * rows), sharey=True)
    for ax, key in zip(axs.ravel(), keys):
        pts = sw[key]
        xs = [p["value"] for p in pts]
        ys = [p["success"] for p in pts]
        ax.plot(range(len(xs)), ys, color=BLUE, marker="o", ms=4, lw=1.6)
        ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs], fontsize=7)
        ax.set_title(NICE.get(key, key), fontsize=8, color=INK2, loc="left")
        ax.set_ylim(-5, 108)
        ax.axhline(90, color=LIGHT, lw=0.8, ls="--")
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("success %", fontsize=8)
    fig.suptitle("One parameter at a time from nominal (10 seeds per point)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sweeps.png"), dpi=170)
    plt.close(fig)

    # 4. what goes wrong: failure modes over all non-designed-failure runs
    fm = res["failure_modes"]
    items = sorted(((k, v) for k, v in fm.items() if k != "success"), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.barh([k for k, _ in items], [v for _, v in items], color=[RED if k == "collision" else GREY for k, _ in items],
            height=0.6)
    for i, (_, v) in enumerate(items):
        ax.text(v + 2, i, str(v), va="center", fontsize=8, color=INK2)
    ax.set_xlabel("runs")
    ax.set_title(f"How the {res['n_failed']} failed runs failed (of {res['n_runs']})", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "failures.png"), dpi=170)
    plt.close(fig)

    # 5. accuracy of successful parks (nominal-like runs)
    acc = res["accuracy"]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.hist(acc["park_err_cm"], bins=30, color=BLUE, alpha=0.85)
    ax.axvline(5, color=RED, lw=1.2)
    ax.text(5, ax.get_ylim()[1] * 0.85, "  ±5 cm requirement", color=RED, fontsize=8)
    ax.set_xlabel("final parking position error (cm), successful parks across all scenarios")
    ax.set_ylabel("runs")
    ax.set_title(f"Parking accuracy: median {acc['p50']:.2f} cm, 95th percentile {acc['p95']:.2f} cm",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "accuracy.png"), dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(scan):
    out = os.path.join(scan, "analysis")
    os.makedirs(out, exist_ok=True)
    df = load(scan)
    df["outcome"] = df.apply(outcome, axis=1)
    res = {"n_runs_total": int(len(df)), "groups": {g: int(n) for g, n in df["group"].value_counts().items()}}
    res["errors"] = int(df["error"].notna().sum()) if "error" in df else 0
    res["wall_time_mean_s"] = round(float(df["wall_s"].mean()), 1)

    pre = df[df.group == "preset"]
    res["presets"] = {v: summarize(g) for v, g in pre.groupby("variant")}

    cb = df[df.group == "combo"].copy()
    parts = cb["variant"].str.extract(r"walls=(?P<walls>[^|]+)\|obs=(?P<obs>[^|]+)\|(?P<timing>[^|]+)\|(?P<sens>\d)sensors")
    cb = pd.concat([cb, parts], axis=1)
    res["combo_table"] = {f"{r.walls}|{r.obs}|{r.timing}|{r.sens}": summarize(g)
                          for r, g in ((g.iloc[0], g) for _, g in cb.groupby("variant"))}
    res["combo_by"] = {dim: {k: summarize(g) for k, g in cb.groupby(dim)} for dim in ("walls", "obs", "timing", "sens")}
    w_all = cb[cb.walls == "all"]
    res["combo_rear_sensor"] = {k: summarize(g) for k, g in w_all.groupby("sens")}

    sw = df[df.group == "sweep"]
    res["sweeps"] = {}
    for key, g in sw.groupby("param", sort=False):
        pts = []
        for val, gg in g.groupby("value"):
            s = summarize(gg)
            s["value"] = float(val)
            pts.append(s)
        res["sweeps"][key] = sorted(pts, key=lambda p: p["value"])

    # failure modes across the whole scan, excluding scenarios that fail by design
    designed = {"estop", "wifi_no_resume", "no_walls"}
    pool = df[~((df.group == "preset") & df.variant.isin(designed)) & ~((df.group == "combo") & (cb.reindex(df.index)["walls"] == "none"))]
    res["failure_modes"] = dict(Counter(pool["outcome"]))
    res["n_failed"] = int((pool["outcome"] != "success").sum())
    res["n_runs_pool"] = int(len(pool))
    res["n_runs"] = int(len(pool))

    ok = df[df["park_ok"] == True]  # noqa: E712
    res["accuracy"] = {"park_err_cm": [round(float(v), 3) for v in ok["park_err_cm"]],
                       "p50": float(ok["park_err_cm"].median()), "p95": float(ok["park_err_cm"].quantile(0.95)),
                       "p99": float(ok["park_err_cm"].quantile(0.99)), "max": float(ok["park_err_cm"].max()),
                       "head_p95": float(ok["park_head_deg"].abs().quantile(0.95)), "n": int(len(ok))}
    res["stops"] = {"echo_stops": int(df["true_stops"].sum() + df["false_stops"].sum()),
                    "false_stops": int(df["false_stops"].sum()),
                    "false_share": pct(df["false_stops"].sum() / max(1, df["true_stops"].sum() + df["false_stops"].sum())),
                    "wall_stops": int(df.get("ev_stop_wall", pd.Series([0])).sum()),
                    "map_stops": int(df.get("ev_stop_map", pd.Series([0])).sum())}
    res["plan_fail_reasons"] = plan_fail_reasons(scan, df)
    res["loc_inflate_runs"] = int((df.get("ev_loc_inflate", pd.Series([0])) > 0).sum())

    coll = df[df["collided"] == True][["group", "variant", "seed"]]  # noqa: E712
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        res["collisions"] = list(ex.map(forensic, [tuple(r) for r in coll.itertuples(index=False)]))
    c = res["collisions"]
    res["collision_summary"] = {"n": len(c), "by_hit": dict(Counter(x["hit"] for x in c)),
                                "by_direction": dict(Counter(x["direction"] for x in c)),
                                "by_task": dict(Counter(x["task"] for x in c))}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    charts(out, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("accuracy", "collisions", "combo_table", "sweeps", "presets")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
