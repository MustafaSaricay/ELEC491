"""Isolated Ackermann steering / actuator experiments (no planner, no estimator
unless asked). These mirror the bench calibrations in the proposal:

1. circle test   - constant steering -> measured turning radius vs L / tan(delta),
                   plus ideal inner/outer wheel angles (cot do - cot di = w / L)
2. servo step    - commanded vs actual steering angle (lag, slew, backlash, trim)
3. speed step    - PID speed loop on encoder feedback, incl. dead zone at low speed
4. tracking      - Pure Pursuit on a planned park path using *true* pose, isolating
                   control from localization
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g
from .config import Config
from .control import PathTracker, SpeedController
from .sensors import Encoder
from .vehicle import Vehicle


def _vehicle(cfg, seed=0, pose=(1.0, 1.0, 0.0)):
    rng = np.random.default_rng(seed)
    return Vehicle(cfg.vehicle, cfg.steering, cfg.drive, cfg.traction, rng, pose), rng


def fit_circle(x, y):
    """Algebraic (Kasa) least-squares circle fit -> radius."""
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    return math.sqrt(c[2] + cx * cx + cy * cy)


def ideal_wheel_angles(cfg, delta):
    L, w = cfg.vehicle.wheelbase, cfg.vehicle.track
    R = L / math.tan(delta)
    return math.atan(L / (R - w / 2)), math.atan(L / (R + w / 2))


def circle_test(cfg: Config, steer_degs=(5, 10, 15, 20, 25, 30), speed=0.2, duration=10.0, dt=0.005):
    rows = []
    for d in steer_degs:
        veh, _ = _vehicle(cfg)
        delta = math.radians(d)
        ctl = SpeedController(cfg.control)
        xs, ys = [], []
        for k in range(int(duration / dt)):
            if k % 4 == 0:
                veh.set_command(delta, ctl.step(speed, veh.v_wheel, 4 * dt))
            veh.step(dt)
            if k * dt > duration / 3:
                xs.append(veh.x)
                ys.append(veh.y)
        R_meas = fit_circle(np.array(xs), np.array(ys))
        R_th = cfg.vehicle.wheelbase / math.tan(delta)
        di, do = ideal_wheel_angles(cfg, delta)
        cot_check = 1 / math.tan(do) - 1 / math.tan(di)
        rows.append({"steer_deg": d, "R_theory": R_th, "R_measured": R_meas,
                     "R_error_pct": 100 * (R_meas - R_th) / R_th,
                     "inner_deg": math.degrees(di), "outer_deg": math.degrees(do),
                     "cot_do_minus_cot_di": cot_check, "w_over_L": cfg.vehicle.track / cfg.vehicle.wheelbase})
    return rows


def servo_step(cfg: Config, dt=0.002):
    seq = [(0.0, 0.0), (0.2, 20.0), (0.8, -20.0), (1.4, 5.0), (1.9, 3.0), (2.4, 0.0), (3.0, 0.0)]
    veh, _ = _vehicle(cfg)
    t_end = seq[-1][0]
    T, C, A = [], [], []
    j = 0
    for k in range(int(t_end / dt)):
        t = k * dt
        while j + 1 < len(seq) and t >= seq[j + 1][0]:
            j += 1
        cmd = math.radians(seq[j][1])
        veh.set_command(cmd, 0.0)
        veh.step(dt)
        T.append(t)
        C.append(seq[j][1])
        A.append(math.degrees(veh.delta))
    return np.array(T), np.array(C), np.array(A)


def speed_step(cfg: Config, dt=0.005):
    seq = [(0.0, 0.0), (0.5, 0.05), (2.5, 0.25), (4.5, 0.08), (6.5, 0.0), (7.5, -0.1), (9.5, 0.0), (10.5, 0.0)]
    veh, rng = _vehicle(cfg)
    enc = Encoder(cfg.encoder, rng)
    ctl = SpeedController(cfg.control)
    period = 1.0 / cfg.control.rate_hz
    T, R, V, E = [], [], [], []
    v_est, j, next_ctl = 0.0, 0, 0.0
    for k in range(int(seq[-1][0] / dt)):
        t = k * dt
        while j + 1 < len(seq) and t >= seq[j + 1][0]:
            j += 1
        veh.step(dt)
        if t >= next_ctl:
            next_ctl += period
            m = enc.sample(t, veh.wheel_dist)
            if m["dt"] > 0:
                a = cfg.control.speed_lpf
                v_est = (1 - a) * v_est + a * m["ds"] / m["dt"]
            veh.set_command(0.0, ctl.step(seq[j][1], v_est, period))
        T.append(t)
        R.append(seq[j][1])
        V.append(veh.v)
        E.append(v_est)
    return np.array(T), np.array(R), np.array(V), np.array(E)


def tracking_test(cfg: Config, dt=0.005):
    from .mapping import OccupancyGrid
    from .planning.hybrid_astar import HybridAStar
    from .world import World
    world = World(cfg.world)
    grid = OccupancyGrid(cfg.world.width, cfg.world.height, cfg.mapping)
    grid.mark_obstacles(world, cfg.lidar.mount_height)
    off = cfg.vehicle.center_offset
    start = g.center_to_rear(g.deg_pose(cfg.world.start), off)
    goal = g.center_to_rear(world.bay_pose, off)
    path, info = HybridAStar(cfg.vehicle, cfg.planner).plan(start, goal, grid)
    if path is None:
        raise RuntimeError(f"no path: {info['reason']}")
    veh, _ = _vehicle(cfg, pose=start)
    tr = PathTracker(cfg.vehicle, cfg.control, path)
    ctl = SpeedController(cfg.control)
    period = 1.0 / cfg.control.rate_hz
    xs, ys, cte = [], [], []
    next_ctl, t = 0.0, 0.0
    while t < 60.0:
        veh.step(dt)
        t += dt
        if t >= next_ctl:
            next_ctl += period
            delta, v_ref, status = tr.step(t, veh.pose, veh.v)
            veh.set_command(delta, ctl.step(v_ref, veh.v, period))
            if status == "done" and abs(veh.v) < 0.01:
                break
        xs.append(veh.x)
        ys.append(veh.y)
        d = np.hypot(path.x - veh.x, path.y - veh.y)
        cte.append(d.min())
    c = veh.center_pose
    gc = g.rear_to_center(goal, off)
    return {"path": path, "x": np.array(xs), "y": np.array(ys), "cte": np.array(cte),
            "final_pos_err": float(math.hypot(*(c[:2] - gc[:2]))),
            "final_heading_err_deg": math.degrees(g.wrap(c[2] - gc[2])), "time": t}


def report(cfg: Config, save=None, show=True):
    from . import viz
    plt = viz._plt()
    rows = circle_test(cfg)
    print("\nCircle test (bicycle angle -> turning radius of the rear axle)")
    print(f"  L = {cfg.vehicle.wheelbase} m, w = {cfg.vehicle.track} m, "
          f"R_min(theory) = {cfg.vehicle.min_turn_radius:.3f} m")
    print("  steer  R_theory  R_meas   err%   inner  outer   cot(do)-cot(di)  w/L")
    for r in rows:
        print(f"  {r['steer_deg']:4.0f}°  {r['R_theory']:7.3f}  {r['R_measured']:7.3f}  {r['R_error_pct']:+6.1f}"
              f"  {r['inner_deg']:5.1f}° {r['outer_deg']:5.1f}°   {r['cot_do_minus_cot_di']:.4f}        {r['w_over_L']:.4f}")
    trk = tracking_test(cfg)
    print(f"\nTracking test (true pose): mean CTE {trk['cte'].mean()*100:.2f} cm, "
          f"max {trk['cte'].max()*100:.2f} cm, final {trk['final_pos_err']*100:.2f} cm / "
          f"{trk['final_heading_err_deg']:+.1f} deg, {trk['time']:.1f} s")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor=viz.SURFACE)
    for a in axs.ravel():
        viz._style(a)
    a = axs[0, 0]
    sd = [r["steer_deg"] for r in rows]
    a.plot(sd, [r["R_theory"] for r in rows], color=viz.C_REV, marker="o", ms=4, lw=1.2, label="L / tan(δ)")
    a.plot(sd, [r["R_measured"] for r in rows], color=viz.C_TRUE, marker="o", ms=4, lw=1.2, label="measured")
    a.set_xlabel("steering angle [deg]")
    a.set_ylabel("turning radius [m]")
    a.set_title("Circle test", fontsize=10)
    a.legend(fontsize=8)
    T, C, A = servo_step(cfg)
    a = axs[0, 1]
    a.plot(T, C, color=viz.C_REV, lw=1.2, label="command")
    a.plot(T, A, color=viz.C_TRUE, lw=1.2, label="actual")
    a.set_xlabel("time [s]")
    a.set_ylabel("steering [deg]")
    a.set_title("Servo step response", fontsize=10)
    a.legend(fontsize=8)
    T, R, V, E = speed_step(cfg)
    a = axs[1, 0]
    a.plot(T, R, color=viz.C_REV, lw=1.2, label="reference")
    a.plot(T, V, color=viz.C_TRUE, lw=1.2, label="true")
    a.plot(T, E, color=viz.C_EST, lw=0.8, label="encoder est.")
    a.set_xlabel("time [s]")
    a.set_ylabel("speed [m/s]")
    a.set_title("Speed loop (PID + dead-zone comp.)", fontsize=10)
    a.legend(fontsize=8)
    a = axs[1, 1]
    from .world import World
    viz.draw_world(a, cfg, World(cfg.world).obstacles)
    a.grid(False)
    p = trk["path"]
    a.plot(*viz._path_xy(p, 1), color=viz.C_TRUE, lw=1.2, label="path fwd")
    a.plot(*viz._path_xy(p, -1), color=viz.C_REV, lw=1.2, label="path rev")
    a.plot(trk["x"], trk["y"], color=viz.INK, lw=0.8, ls="--", label="driven")
    a.set_title(f"Pure Pursuit tracking (true pose): max CTE {trk['cte'].max()*100:.1f} cm", fontsize=10)
    a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, facecolor=viz.SURFACE)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return rows, trk
