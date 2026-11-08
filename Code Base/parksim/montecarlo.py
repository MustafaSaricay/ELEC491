"""Monte Carlo runs (many seeds) and one-parameter robustness sweeps."""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import Config, set_dotted
from .simulator import run


def _one(args):
    cfg, seed = args
    cfg = cfg.copy()
    cfg.sim.seed = seed
    try:
        m = run(cfg).metrics
    except Exception as exc:          # a crash in the stack is a result too
        return {"seed": seed, "error": repr(exc), "tasks": [], "all_success": False, "collided": False}
    m["seed"] = seed
    return m


def monte_carlo(cfg: Config, n=20, seed0=0, workers=None, progress=True, callback=None, mp_context=None):
    """callback(done, total, metrics) is called as runs finish; if it returns
    False the remaining runs are cancelled. Results are returned in seed order."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if callback is not None:
        ctx = mp_context and __import__("multiprocessing").get_context(mp_context)
        out = {}
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        try:
            futs = [ex.submit(_one, (cfg, seed0 + i)) for i in range(n)]
            for f in as_completed(futs):
                m = f.result()
                out[m["seed"]] = m
                if callback(len(out), n, m) is False:
                    break
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        return [out[k] for k in sorted(out)]
    jobs = [(cfg, seed0 + i) for i in range(n)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(_one, jobs)):
            results.append(m)
            if progress:
                tag = "ok " if m.get("all_success") else "FAIL"
                extra = m.get("error", "") or ", ".join(
                    f"{t['kind']}:{t['result']}:{t['err_pos']*100:.1f}cm" for t in m["tasks"])
                print(f"  [{i+1:3d}/{n}] seed {m['seed']:4d} {tag} {extra}")
    return results


def summarize(results, cfg: Config):
    n = len(results)
    out = {"runs": n,
           "success_rate": sum(r.get("all_success", False) for r in results) / max(n, 1),
           "collision_rate": sum(bool(r.get("collided")) for r in results) / max(n, 1),
           "errors": sum("error" in r for r in results)}
    per = {}
    for i, cmd in enumerate(cfg.mission.commands):
        ts = [r["tasks"][i] for r in results if len(r["tasks"]) > i]
        if not ts:
            continue
        def stat(key, scale=1.0, absval=True):
            v = np.array([abs(t[key]) if absval else t[key] for t in ts]) * scale
            return {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
        results_hist = {}
        for t in ts:
            results_hist[t["result"]] = results_hist.get(t["result"], 0) + 1
        per[f"{i}:{cmd['cmd']}"] = {
            "success_rate": sum(t["success"] for t in ts) / n,
            "outcomes": results_hist,
            "pos_err_cm": stat("err_pos", 100), "lat_err_cm": stat("err_lat", 100),
            "long_err_cm": stat("err_long", 100), "heading_err_deg": stat("err_heading_deg"),
            "time_s": stat("time", absval=False), "loc_err_max_cm": stat("loc_err_max", 100),
            "min_clearance_cm": {"min": float(min(t["min_clearance"] for t in ts) * 100)},
            "replans_mean": float(np.mean([t["replans"] for t in ts])),
        }
    out["tasks"] = per
    return out


def print_summary(s, title=""):
    print(f"\n=== {title} ===" if title else "")
    print(f"runs {s['runs']}  success {s['success_rate']*100:.0f}%  collisions {s['collision_rate']*100:.0f}%"
          + (f"  crashed-runs {s['errors']}" if s["errors"] else ""))
    for k, v in s["tasks"].items():
        print(f"  task {k:10s} success {v['success_rate']*100:5.1f}%  outcomes {v['outcomes']}")
        print(f"     pos err  mean {v['pos_err_cm']['mean']:.2f}  p95 {v['pos_err_cm']['p95']:.2f}  "
              f"max {v['pos_err_cm']['max']:.2f} cm   (lat p95 {v['lat_err_cm']['p95']:.2f}, "
              f"long p95 {v['long_err_cm']['p95']:.2f})")
        print(f"     heading  mean {v['heading_err_deg']['mean']:.2f}  p95 {v['heading_err_deg']['p95']:.2f} deg"
              f"   time mean {v['time_s']['mean']:.1f} s   loc-err max p95 {v['loc_err_max_cm']['p95']:.2f} cm"
              f"   min clearance {v['min_clearance_cm']['min']:.1f} cm   replans {v['replans_mean']:.1f}")


def save_csv(results, path):
    rows = []
    for r in results:
        for t in r["tasks"]:
            rows.append({"seed": r["seed"], **{k: v for k, v in t.items()}})
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sweep(cfg: Config, key: str, values, n=10, workers=None):
    out = []
    for v in values:
        c = cfg.copy()
        set_dotted(c, key, v)
        print(f"\n--- {key} = {v} ---")
        res = monte_carlo(c, n=n, workers=workers, progress=False)
        s = summarize(res, c)
        print_summary(s, f"{key}={v}")
        out.append((v, s))
    return out


def plot_sweep(key, data, save=None, show=True):
    from . import viz
    plt = viz._plt()
    xs = [d[0] for d in data]
    labels = [str(x) for x in xs]
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs)
    pos = xs if numeric else list(range(len(xs)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=viz.SURFACE)
    for a in axs:
        viz._style(a)
    task_keys = list(data[0][1]["tasks"].keys()) if data and data[0][1]["tasks"] else []
    colors = [viz.C_TRUE, viz.C_REV, viz.C_EST]
    for k, tk in enumerate(task_keys[:3]):
        sr = [d[1]["tasks"].get(tk, {}).get("success_rate", 0) * 100 for d in data]
        p95 = [d[1]["tasks"].get(tk, {}).get("pos_err_cm", {}).get("p95", math.nan) for d in data]
        axs[0].plot(pos, sr, marker="o", ms=5, lw=1.5, color=colors[k], label=tk)
        axs[1].plot(pos, p95, marker="o", ms=5, lw=1.5, color=colors[k], label=tk)
    axs[1].axhline(5.0, color=viz.INK2, lw=1, ls="--")
    axs[1].text(pos[0], 5.1, "±5 cm requirement", fontsize=7, color=viz.INK2)
    axs[0].set_ylabel("task success [%]")
    axs[0].set_ylim(-5, 105)
    axs[1].set_ylabel("final position error p95 [cm]")
    for a in axs:
        a.set_xlabel(key)
        if not numeric:
            a.set_xticks(pos, labels)
        a.legend(fontsize=8)
    fig.suptitle(f"Robustness sweep: {key}", fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, facecolor=viz.SURFACE)
    if show:
        plt.show()
    else:
        plt.close(fig)
