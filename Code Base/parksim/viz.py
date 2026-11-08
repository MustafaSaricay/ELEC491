"""Matplotlib visualisation: live/saved animation, run summary plots, planner plot."""
from __future__ import annotations

import math

import numpy as np

from . import geometry as g

# Validated categorical order (blue, orange, aqua) + recessive neutrals.
C_TRUE = "#2a78d6"      # truth / forward path
C_REV = "#eb6834"       # reverse path / command
C_EST = "#1baf7a"       # estimate
C_ALERT = "#e34948"     # collisions, invisible obstacles (always labelled)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
SURFACE = "#fcfcfb"
OBST = "#8a8984"


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def draw_world(ax, cfg, obstacles, t=None):
    """Testbed, bay, summon targets and obstacles. Returns {obstacle: patch}."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    w, h = cfg.world.width, cfg.world.height
    from .world import SIDES, wall_sides
    present = wall_sides(cfg.world.walls)
    edge = {"bottom": ((0, w), (0, 0)), "right": ((w, w), (0, h)), "top": ((0, w), (h, h)), "left": ((0, 0), (0, h))}
    for side in SIDES:
        xs, ys = edge[side]
        if side in present:
            ax.plot(xs, ys, color=INK, lw=2.0, solid_capstyle="butt")
        else:      # no physical wall: virtual boundary only
            ax.plot(xs, ys, color=INK2, lw=1.0, ls=(0, (4, 3)))
    if len(present) < 4:
        ax.text(w / 2, -0.035, "dashed edge = no wall (virtual boundary)", ha="center", va="top",
                fontsize=6, color=INK2)
    b = cfg.world.bay
    bp = np.array([b["x"], b["y"], math.radians(b["yaw_deg"])])
    c = g.rect_corners(*bp, b["depth"], b["width"])
    for a, z in ((c[1], c[0]), (c[2], c[3]), (c[0], c[3])):
        ax.plot([a[0], z[0]], [a[1], z[1]], color=INK2, lw=2, solid_capstyle="round")
    ax.text(bp[0], bp[1], "BAY", ha="center", va="center", fontsize=7, color=INK2)
    for cmd in cfg.mission.commands:
        if cmd.get("cmd") == "summon":
            p = g.deg_pose(cmd["pose"])
            pc = g.rect_corners(*p, cfg.vehicle.length, cfg.vehicle.width)
            ax.add_patch(Polygon(pc, closed=True, fill=False, ec=INK2, ls=":", lw=1))
            ax.text(p[0], p[1], "PICKUP", ha="center", va="center", fontsize=6, color=INK2)
    lidar_h = cfg.lidar.mount_height
    patches = {}
    for o in obstacles:
        invisible = o.h <= lidar_h
        kw = dict(fc="none" if invisible else OBST, ec=C_ALERT if invisible else INK2,
                  hatch="////" if invisible else None, lw=1)
        if o.kind == "circle":
            p = Circle((o.x, o.y), o.r, **kw)
        else:
            p = Polygon(o.corners(), closed=True, **kw)
        ax.add_patch(p)
        if invisible:
            ax.annotate("below LiDAR plane", (o.x, o.y), xytext=(4, 6), textcoords="offset points",
                        fontsize=6, color=C_ALERT)
        patches[id(o)] = (o, p)
        if t is not None:
            p.set_visible(o.spawn_t <= t)
    ax.set_xlim(-0.05, w + 0.05)
    ax.set_ylim(-0.05, h + 0.05)
    ax.set_aspect("equal")
    return patches


def _path_xy(path, direction):
    m = path.dir == direction
    x = np.where(m, path.x, np.nan)
    y = np.where(m, path.y, np.nan)
    # keep the cusp point so segments join
    return x, y


def _car_polys(cfg, rear_pose, wl, wr):
    vc = cfg.vehicle
    c = g.rear_to_center(rear_pose, vc.center_offset)
    body = g.rect_corners(*c, vc.length, vc.width)
    x, y, th = rear_pose
    wheels = []
    for lx, ly, ang in ((0, vc.track / 2, 0), (0, -vc.track / 2, 0),
                        (vc.wheelbase, vc.track / 2, wl), (vc.wheelbase, -vc.track / 2, wr)):
        wx = x + lx * math.cos(th) - ly * math.sin(th)
        wy = y + lx * math.sin(th) + ly * math.cos(th)
        wheels.append(g.rect_corners(wx, wy, th + ang, 0.06, 0.025))
    return body, wheels


# --------------------------------------------------------------------------- #
def animate(res, save=None, speed=1.0, fps=20):
    plt = _plt()
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon

    cfg, log = res.cfg, res.log
    t_log = log["t"]
    fig = plt.figure(figsize=(12, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.45, wspace=0.15)
    ax = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_e = fig.add_subplot(gs[2, 1], sharex=ax_v)
    for a in (ax, ax_v, ax_d, ax_e):
        _style(a)
    ax.grid(False)

    ext = None
    img = None
    if res.grids:
        from .mapping import OccupancyGrid
        og = OccupancyGrid(cfg.world.width, cfg.world.height, cfg.mapping)
        ext = (og.ox, og.ox + og.nx * og.res, og.oy, og.oy + og.ny * og.res)
        img = ax.imshow(res.grids[0][1], origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1,
                        alpha=0.55, zorder=0, interpolation="nearest")
    patches = draw_world(ax, cfg, res.obstacles, t=0.0)
    path_f, = ax.plot([], [], color=C_TRUE, lw=1.5, label="path (forward)")
    path_r, = ax.plot([], [], color=C_REV, lw=1.5, label="path (reverse)")
    trail_t, = ax.plot([], [], color=INK, lw=0.8, alpha=0.6, label="true trail")
    trail_e, = ax.plot([], [], color=C_EST, lw=0.8, ls="--", label="estimated trail")
    scan = ax.scatter([], [], s=3, color=C_ALERT, alpha=0.6, zorder=4,
                      label="LiDAR points" if cfg.lidar.enabled else None)
    from matplotlib.patches import Wedge
    uc = cfg.ultrasonic
    us_by = {s["name"]: s for s in uc.sensors} if uc.enabled else {}
    ping_t = {n: [] for n in us_by}
    ping_r = {n: [] for n in us_by}
    for tp, n, rr in res.pings:
        if n in ping_t:
            ping_t[n].append(tp)
            ping_r[n].append(rr)
    wedges = {}
    for i, n in enumerate(us_by):
        w = Wedge((0, 0), 0.01, 0, 1, fc=INK2, alpha=0.15, ec=INK2, lw=0.6, zorder=3,
                  label="ultrasonic cones" if i == 0 else None)
        w.set_visible(False)
        ax.add_patch(w)
        wedges[n] = w
    body = Polygon(np.zeros((4, 2)), closed=True, fc=C_TRUE, alpha=0.25, ec=C_TRUE, lw=1.2, zorder=5)
    est = Polygon(np.zeros((4, 2)), closed=True, fill=False, ec=C_EST, ls="--", lw=1.2, zorder=6)
    wheels = [Polygon(np.zeros((4, 2)), closed=True, fc=INK, zorder=6) for _ in range(4)]
    for p in [body, est, *wheels]:
        ax.add_patch(p)
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                   color=INK, bbox=dict(fc=SURFACE, ec=GRID, alpha=0.9))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title("Autonomous parking sim", fontsize=10)

    ax_v.set_ylabel("speed [m/s]", fontsize=8)
    lv_ref, = ax_v.plot([], [], color=C_REV, lw=1.2, label="reference")
    lv, = ax_v.plot([], [], color=C_TRUE, lw=1.2, label="true")
    lv_e, = ax_v.plot([], [], color=C_EST, lw=1.0, label="encoder est.")
    ax_v.legend(fontsize=6, loc="upper right", ncol=3)
    ax_d.set_ylabel("steer [deg]", fontsize=8)
    ld_c, = ax_d.plot([], [], color=C_REV, lw=1.2, label="command")
    ld, = ax_d.plot([], [], color=C_TRUE, lw=1.2, label="actual")
    ax_d.legend(fontsize=6, loc="upper right", ncol=2)
    ax_e.set_ylabel("loc. error [cm]", fontsize=8)
    ax_e.set_xlabel("time [s]", fontsize=8)
    le, = ax_e.plot([], [], color=C_TRUE, lw=1.2, label="|true - est|")
    ls, = ax_e.plot([], [], color=C_EST, lw=1.0, label="EKF 1-sigma")
    ax_e.legend(fontsize=6, loc="upper right", ncol=2)
    ax_v.set_ylim(-0.4, 0.4)
    dmax = cfg.vehicle.max_steer_deg + 3
    ax_d.set_ylim(-dmax, dmax)
    ax_e.set_ylim(0, max(2.0, float(np.percentile(log["loc_err"], 99)) * 100 * 1.3))

    states = {v: k for k, v in __import__("parksim.simulator", fromlist=["STATE_CODES"]).STATE_CODES.items()}
    frame_dt = speed / fps
    times = np.arange(0, t_log[-1], frame_dt)
    path_times = [p[0] for p in res.paths]
    scan_times = [s[0] for s in res.scans]
    grid_times = [gg[0] for gg in res.grids]
    ev_times = [e[0] for e in res.events]

    def frame(ti):
        i = min(int(np.searchsorted(t_log, ti)), len(t_log) - 1)
        rp = np.array([log["x"][i], log["y"][i], log["yaw"][i]])
        b, ws = _car_polys(cfg, rp, log["wl"][i], log["wr"][i])
        body.set_xy(b)
        for p, wv in zip(wheels, ws):
            p.set_xy(wv)
        ep = np.array([log["ex"][i], log["ey"][i], log["eyaw"][i]])
        est.set_xy(_car_polys(cfg, ep, 0, 0)[0])
        trail_t.set_data(log["x"][:i], log["y"][:i])
        trail_e.set_data(log["ex"][:i], log["ey"][:i])
        k = int(np.searchsorted(path_times, ti, side="right")) - 1
        if k >= 0:
            pth = res.paths[k][1]
            path_f.set_data(*_path_xy(pth, 1))
            path_r.set_data(*_path_xy(pth, -1))
        k = int(np.searchsorted(scan_times, ti, side="right")) - 1
        if k >= 0:
            _, _, sp, ang, rng = res.scans[k]
            ok = np.isfinite(rng)
            mx = cfg.lidar.mount_x
            pts = np.stack([mx + rng[ok] * np.cos(ang[ok]), rng[ok] * np.sin(ang[ok])], 1)
            scan.set_offsets(g.to_world(pts, sp) if len(pts) else np.zeros((0, 2)))
        for n, w in wedges.items():
            k = int(np.searchsorted(ping_t[n], ti, side="right")) - 1
            if k < 0 or not np.isfinite(ping_r[n][k]):
                w.set_visible(False)
                continue
            sx, sy, syaw = g.mount_pose(rp, us_by[n])
            w.set_center((sx, sy))
            w.set_radius(ping_r[n][k])
            w.set_theta1(math.degrees(syaw) - uc.half_angle_deg)
            w.set_theta2(math.degrees(syaw) + uc.half_angle_deg)
            w.set_visible(True)
        if img is not None:
            k = int(np.searchsorted(grid_times, ti, side="right")) - 1
            if k >= 0:
                img.set_data(res.grids[k][1])
        for o, p in patches.values():
            p.set_visible(o.spawn_t <= ti)
        k = int(np.searchsorted(ev_times, ti, side="right"))
        recent = [e[1] for e in res.events[max(0, k - 3):k]]
        link = "UP" if log["link"][i] else "DOWN"
        info.set_text(f"t={t_log[i]:6.2f}s  state={states[int(log['state'][i])]}\n"
                      f"v={log['v'][i]:+.2f} m/s  steer={math.degrees(log['delta'][i]):+5.1f} deg\n"
                      f"loc err={log['loc_err'][i]*100:4.1f} cm  wifi={link}\n" + "\n".join(recent))
        sl = slice(0, i + 1)
        tt = t_log[sl]
        lv_ref.set_data(tt, log["v_ref"][sl])
        lv.set_data(tt, log["v"][sl])
        lv_e.set_data(tt, log["v_est"][sl])
        ld_c.set_data(tt, np.degrees(log["delta_cmd"][sl]))
        ld.set_data(tt, np.degrees(log["delta"][sl]))
        le.set_data(tt, log["loc_err"][sl] * 100)
        ls.set_data(tt, log["sx"][sl] * 100)
        ax_v.set_xlim(max(0, ti - 15), max(15, ti))
        return []

    anim = FuncAnimation(fig, frame, frames=times, interval=1000 / fps, blit=False, repeat=False)
    if save:
        if save.endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(save, writer=PillowWriter(fps=fps), dpi=80)
        else:
            anim.save(save, fps=fps, dpi=100)
        plt.close(fig)
    else:
        plt.show()
    return anim


# --------------------------------------------------------------------------- #
def summary(res, save=None, show=True):
    """Static overview: trajectory + time series."""
    plt = _plt()
    cfg, log = res.cfg, res.log
    t = log["t"]
    fig = plt.figure(figsize=(13, 7.5), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for a in [ax, *axes]:
        _style(a)
    ax.grid(False)
    draw_world(ax, cfg, res.obstacles)
    for k, (tp, p) in enumerate(res.paths):
        alpha = 1.0 if k == len(res.paths) - 1 else 0.35
        ax.plot(*_path_xy(p, 1), color=C_TRUE, lw=1.2, alpha=alpha, label="planned fwd" if k == 0 else None)
        ax.plot(*_path_xy(p, -1), color=C_REV, lw=1.2, alpha=alpha, label="planned rev" if k == 0 else None)
    ax.plot(log["x"], log["y"], color=INK, lw=0.9, label="true (rear axle)")
    ax.plot(log["ex"], log["ey"], color=C_EST, lw=0.9, ls="--", label="estimate")
    for n_task, tk in enumerate(res.tasks):
        i = min(int(np.searchsorted(t, tk["t_end"])), len(t) - 1)
        b, _ = _car_polys(cfg, np.array([log["x"][i], log["y"][i], log["yaw"][i]]), 0, 0)
        from matplotlib.patches import Polygon
        ax.add_patch(Polygon(b, closed=True, fc=C_TRUE, alpha=0.2, ec=C_TRUE))
        mark = "OK" if tk["success"] else "FAIL"
        ax.annotate(f"{tk['kind']} {mark}\n{tk['err_pos']*100:.1f} cm, {tk['err_heading_deg']:+.1f} deg",
                    b.mean(0), xytext=(8, -18 - 22 * n_task), textcoords="offset points", fontsize=7,
                    color=INK)
    ax.legend(loc="lower right", fontsize=7)
    ok = res.metrics["all_success"]
    ax.set_title(f"{'SUCCESS' if ok else 'FAILED'}  |  loc RMSE {res.metrics['loc_rmse']*100:.1f} cm  |  "
                 f"min clearance {res.metrics['min_clearance']*100:.1f} cm", fontsize=9)

    a = axes[0]
    a.plot(t, log["v_ref"], color=C_REV, lw=1, label="reference")
    a.plot(t, log["v"], color=C_TRUE, lw=1, label="true")
    a.plot(t, log["v_est"], color=C_EST, lw=0.8, label="encoder est.")
    a.set_ylabel("speed [m/s]", fontsize=8)
    a.legend(fontsize=6, ncol=3, loc="upper right")
    a = axes[1]
    a.plot(t, np.degrees(log["delta_cmd"]), color=C_REV, lw=1, label="command")
    a.plot(t, np.degrees(log["delta"]), color=C_TRUE, lw=1, label="actual")
    a.set_ylabel("steer [deg]", fontsize=8)
    a.legend(fontsize=6, ncol=2, loc="upper right")
    a = axes[2]
    a.plot(t, log["loc_err"] * 100, color=C_TRUE, lw=1, label="position error")
    a.plot(t, log["sx"] * 100, color=C_EST, lw=0.8, label="EKF 1-sigma")
    a.set_ylabel("loc. [cm]", fontsize=8)
    a.legend(fontsize=6, ncol=2, loc="upper right")
    a = axes[3]
    yaw_err = np.degrees(g.wrap(log["eyaw"] - log["yaw"]))
    a.plot(t, yaw_err, color=C_TRUE, lw=1, label="heading error")
    a.set_ylabel("yaw err [deg]", fontsize=8)
    a.set_xlabel("time [s]", fontsize=8)
    # shade non-driving states
    from .simulator import STATE_CODES
    inv = {v: k for k, v in STATE_CODES.items()}
    for code, name in ((2, "BLOCKED"), (3, "PAUSED"), (5, "FAULT"), (4, "ESTOP")):
        m = log["state"] == code
        if m.any():
            for aa in axes:
                aa.fill_between(t, 0, 1, where=m, transform=aa.get_xaxis_transform(),
                                color=GRID, alpha=0.6, lw=0)
            axes[3].text(t[m][0], 0.9, name, transform=axes[3].get_xaxis_transform(), fontsize=6, color=INK2)
    a.legend(fontsize=6, loc="upper right")
    if save:
        fig.savefig(save, dpi=110, facecolor=SURFACE)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_plan(cfg, world, grid, path, info, start, goal, save=None, show=True):
    plt = _plt()
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)
    _style(ax)
    ax.grid(False)
    ext = (grid.ox, grid.ox + grid.nx * grid.res, grid.oy, grid.oy + grid.ny * grid.res)
    ax.imshow(grid.occupied_mask(), origin="lower", extent=ext, cmap="Greys", alpha=0.4,
              interpolation="nearest")
    draw_world(ax, cfg, world.obstacles)
    off = cfg.vehicle.center_offset
    for p, col in ((start, INK2), (goal, INK)):
        c = g.rear_to_center(p, off)
        ax.add_patch(Polygon(g.rect_corners(*c, cfg.vehicle.length, cfg.vehicle.width),
                             closed=True, fill=False, ec=col, lw=1.2))
    if path is not None:
        ax.plot(*_path_xy(path, 1), color=C_TRUE, lw=1.6, label="forward")
        ax.plot(*_path_xy(path, -1), color=C_REV, lw=1.6, label="reverse")
        for i in range(0, len(path), 12):
            c = g.rear_to_center(path.poses[i], off)
            ax.add_patch(Polygon(g.rect_corners(*c, cfg.vehicle.length, cfg.vehicle.width),
                                 closed=True, fill=False, ec=C_REV if path.dir[i] < 0 else C_TRUE,
                                 lw=0.5, alpha=0.5))
        ax.set_title(f"Hybrid A*: {path.length:.2f} m, {path.n_cusps} cusps, "
                     f"{info['expansions']} expansions, {info['time']*1e3:.0f} ms", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.set_title(f"No path: {info['reason']}", fontsize=9, color=C_ALERT)
    if save:
        fig.savefig(save, dpi=110, facecolor=SURFACE)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
