"""Command line interface:  python -m parksim <command> [options]

  ui                         web interface (browser): choose options, combine, run, compare
  list                       list scenarios
  config   -s NAME           print the resolved configuration as JSON
  run      -s NAME           closed-loop run (park + summon), optional animation / plots
  plan     -s NAME           planner only, on a map of the scenario's obstacles
  steer-test -s NAME         Ackermann / actuator bench tests
  mc       -s NAME -n 50     Monte Carlo over random seeds
  sweep    -s NAME --param lidar.range_sigma --values 0 0.01 0.03 0.05

Common options: --set key=value (repeatable, JSON values), --config file.json, --seed N,
  --walls all|none|bottom,left   --obstacles fixed|circle|line|random   --n-obstacles N
  --obstacle-timing before_park|after_park
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from . import scenarios
from .config import parse_value


def _cfg(args):
    over = None
    if getattr(args, "config", None):
        with open(args.config) as f:
            over = json.load(f)
    cfg = scenarios.build(args.scenario, over)
    for item in args.set or []:
        if "=" not in item:
            sys.exit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        cfg.set(k.strip(), parse_value(v.strip()))
    if getattr(args, "seed", None) is not None:
        cfg.sim.seed = args.seed
    if getattr(args, "log_dir", None):
        cfg.sim.log_dir = args.log_dir
        cfg.sim.log_name = getattr(args, "scenario", "run") or "run"
    if getattr(args, "walls", None) is not None:
        w = args.walls.strip().lower()
        cfg.world.walls = {"all": ["bottom", "right", "top", "left"], "none": []}.get(
            w, [x.strip() for x in w.split(",") if x.strip()])
    if getattr(args, "obstacles", None):
        cfg.world.obstacle_mode = args.obstacles
    if getattr(args, "n_obstacles", None) is not None:
        cfg.world.n_obstacles = args.n_obstacles
    if getattr(args, "obstacle_timing", None):
        cfg.world.obstacle_timing = args.obstacle_timing.replace("-", "_")
    return cfg


def _backend(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")


def cmd_list(args):
    w = max(len(k) for k in scenarios.SCENARIOS)
    for k, (desc, _) in scenarios.SCENARIOS.items():
        print(f"  {k:<{w}}  {desc}")


def cmd_config(args):
    print(json.dumps(_cfg(args).to_dict(), indent=2))


def cmd_run(args):
    from .simulator import run
    cfg = _cfg(args)
    show = not args.no_show
    _backend(show and (args.animate or args.summary))
    print(f"scenario '{args.scenario}', seed {cfg.sim.seed} ...")
    res = run(cfg, progress=args.verbose)
    for t, e in res.events:
        print(f"  {t:7.2f}  {e}")
    print()
    for tk in res.tasks:
        print(f"  task #{tk['id']} {tk['kind']:7s} {tk['result']:10s} {'SUCCESS' if tk['success'] else 'FAIL':7s} "
              f"pos {tk['err_pos']*100:5.2f} cm (lat {tk['err_lat']*100:+.2f}, long {tk['err_long']*100:+.2f})  "
              f"heading {tk['err_heading_deg']:+5.2f} deg  time {tk['time']:5.1f} s  "
              f"min clearance {tk['min_clearance']*100:4.1f} cm")
    m = res.metrics
    print(f"\n  localization RMSE {m['loc_rmse']*100:.2f} cm, max {m['loc_err_max']*100:.2f} cm;  "
          f"collision: {m['collided']};  Wi-Fi latency mean "
          f"{(m['comm_latency_mean'] or 0)*1e3:.0f} ms / max {(m['comm_latency_max'] or 0)*1e3:.0f} ms")
    for fs in m["failsafe"]:
        if fs.get("moving_at_loss"):
            print(f"  failsafe: outage {fs['outage']} -> stopped after {fs.get('stop_time', math.nan):.2f} s, "
                  f"{fs.get('stop_distance', math.nan)*100:.1f} cm")
        else:
            print(f"  failsafe: outage {fs['outage']} (vehicle was not moving)")
    print(f"\n  RESULT: {'ALL TASKS SUCCEEDED' if m['all_success'] else 'FAILED'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(m, f, indent=2, default=float)
    from . import viz
    if args.summary or args.save_summary:
        viz.summary(res, save=args.save_summary, show=show and args.summary)
    if args.animate or args.save_anim:
        viz.animate(res, save=args.save_anim, speed=args.speed, fps=args.fps)
    return 0 if m["all_success"] else 1


def cmd_plan(args):
    import numpy as np
    from . import geometry as g
    from . import viz
    from .mapping import OccupancyGrid
    from .planning.hybrid_astar import HybridAStar
    from .world import World
    cfg = _cfg(args)
    show = not args.no_show
    _backend(show)
    world = World(cfg.world)
    grid = OccupancyGrid(cfg.world.width, cfg.world.height, cfg.mapping)
    grid.mark_obstacles(world, 0.0 if args.map == "truth" else cfg.lidar.mount_height)
    off = cfg.vehicle.center_offset
    start = g.center_to_rear(g.deg_pose(args.start or cfg.world.start), off)
    goal = g.center_to_rear(g.deg_pose(args.goal) if args.goal else world.bay_pose, off)
    path, info = HybridAStar(cfg.vehicle, cfg.planner).plan(start, goal, grid)
    if path is None:
        print(f"no path: {info['reason']} ({info['expansions']} expansions, {info['time']*1e3:.0f} ms)")
    else:
        print(f"path {path.length:.2f} m, {path.n_cusps} cusps, {info['expansions']} expansions, "
              f"{info['time']*1e3:.0f} ms;  R_min used {HybridAStar(cfg.vehicle, cfg.planner).radius:.3f} m")
    viz.plot_plan(cfg, world, grid, path, info, start, goal, save=args.save, show=show)
    return 0 if path is not None else 1


def cmd_steer(args):
    from . import steering_test
    cfg = _cfg(args)
    show = not args.no_show
    _backend(show)
    steering_test.report(cfg, save=args.save, show=show)


def cmd_mc(args):
    from . import montecarlo as mc
    cfg = _cfg(args)
    print(f"Monte Carlo: scenario '{args.scenario}', {args.n} runs")
    res = mc.monte_carlo(cfg, n=args.n, seed0=args.seed or 0, workers=args.workers)
    s = mc.summarize(res, cfg)
    mc.print_summary(s, args.scenario)
    if args.csv:
        mc.save_csv(res, args.csv)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(s, f, indent=2)


def cmd_sweep(args):
    from . import montecarlo as mc
    cfg = _cfg(args)
    show = not args.no_show
    _backend(show)
    values = [parse_value(v) for v in args.values]
    data = mc.sweep(cfg, args.param, values, n=args.n, workers=args.workers)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"value": v, "summary": s} for v, s in data], f, indent=2)
    mc.plot_sweep(args.param, data, save=args.save, show=show)


def cmd_ui(args):
    from .ui.server import serve
    serve(port=args.port, open_browser=not args.no_browser, host=args.host, max_runs=args.max_runs,
          max_sims=args.max_sims, workers=args.workers, keep_jobs=args.keep_jobs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="parksim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-s", "--scenario", default="nominal")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE")
        sp.add_argument("--config", help="JSON file with nested overrides")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--log-dir", metavar="DIR",
                        help="write a JSON-lines event log per run into DIR (one file per seed)")
        sp.add_argument("--walls", metavar="all|none|SIDES",
                        help="physical walls: all, none, or e.g. bottom,left")
        sp.add_argument("--obstacles", choices=["fixed", "circle", "line", "random"],
                        help="obstacle shapes: fixed list, or generate circles / lines / random mix")
        sp.add_argument("--n-obstacles", type=int, help="how many obstacles to generate")
        sp.add_argument("--obstacle-timing", choices=["before_park", "after_park", "before-park", "after-park"],
                        help="obstacles present from the start, or placed after the car has parked")
        return sp

    sub.add_parser("list")
    sp = sub.add_parser("ui", help="web interface: pick options, combine presets, run, compare")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (keep 127.0.0.1 behind a Cloudflare Tunnel)")
    sp.add_argument("--max-runs", type=int, default=200, help="max Monte Carlo runs per job")
    sp.add_argument("--max-sims", type=int, default=1000, help="max simulations per comparison")
    sp.add_argument("--workers", type=int, help="worker processes for MC/compare (default: CPUs - 1)")
    sp.add_argument("--keep-jobs", type=int, default=20, help="finished results kept in memory")
    common(sub.add_parser("config"))
    sp = common(sub.add_parser("run"))
    sp.add_argument("--animate", action="store_true", help="show live animation")
    sp.add_argument("--save-anim", help="write animation (.gif or .mp4)")
    sp.add_argument("--speed", type=float, default=2.0, help="animation playback speed-up")
    sp.add_argument("--fps", type=int, default=15)
    sp.add_argument("--summary", action="store_true", help="show summary plots")
    sp.add_argument("--save-summary", help="write summary figure (.png)")
    sp.add_argument("--json", help="write metrics JSON")
    sp.add_argument("--no-show", action="store_true")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp = common(sub.add_parser("plan"))
    sp.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"))
    sp.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"))
    sp.add_argument("--map", choices=["visible", "truth"], default="visible",
                    help="visible = only obstacles the LiDAR can see")
    sp.add_argument("--save")
    sp.add_argument("--no-show", action="store_true")
    sp = common(sub.add_parser("steer-test"))
    sp.add_argument("--save")
    sp.add_argument("--no-show", action="store_true")
    sp = common(sub.add_parser("mc"))
    sp.add_argument("-n", type=int, default=20)
    sp.add_argument("--workers", type=int)
    sp.add_argument("--csv")
    sp.add_argument("--json")
    sp = common(sub.add_parser("sweep"))
    sp.add_argument("--param", required=True)
    sp.add_argument("--values", nargs="+", required=True)
    sp.add_argument("-n", type=int, default=10)
    sp.add_argument("--workers", type=int)
    sp.add_argument("--save")
    sp.add_argument("--json")
    sp.add_argument("--no-show", action="store_true")

    args = p.parse_args(argv)
    handler = {"list": cmd_list, "config": cmd_config, "run": cmd_run, "plan": cmd_plan,
               "steer-test": cmd_steer, "mc": cmd_mc, "sweep": cmd_sweep, "ui": cmd_ui}[args.cmd]
    rc = handler(args)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
