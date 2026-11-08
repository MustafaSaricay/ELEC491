"""Local web interface for parksim:  python -m parksim ui

A tiny stdlib HTTP server (bound to 127.0.0.1) that serves one HTML page and
a JSON API. Simulations run in a background thread; Monte Carlo / compare jobs
use worker processes. Only one job runs at a time.

API
  GET  /                      the interface (index.html)
  GET  /api/options           presets, defaults, every config key (for autocomplete)
  POST /api/config            spec -> resolved config JSON + preset conflicts (no run)
  POST /api/run               {"kind": "single" | "mc" | "compare", "spec": {...}, ...} -> {"job": id}
  GET  /api/job/<id>          status, progress, result
  POST /api/job/<id>/cancel   stop a Monte Carlo / compare job
"""
from __future__ import annotations

import base64
import io
import itertools
import json
import math
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from .. import scenarios  # noqa: E402
from ..config import Config, _default_ultrasonic_layout, parse_value, set_dotted  # noqa: E402
from ..simulator import STATE_CODES  # noqa: E402

HERE = Path(__file__).parent
WALL_CHOICES = {"all": ["bottom", "right", "top", "left"], "none": [],
                "bottom+left": ["bottom", "left"], "top only": ["top"]}
REAR_SENSOR = {"name": "B", "x": -0.045, "y": 0.0, "yaw_deg": 180.0}


# --------------------------------------------------------------------------- #
# spec -> Config
# --------------------------------------------------------------------------- #
def build_config(spec: dict):
    """The interface's choices -> (Config, notes). Empty / missing fields keep
    whatever the selected presets (or the defaults) say."""
    cfg, conflicts = scenarios.combine(spec.get("presets") or ["nominal"])
    notes = [f"presets disagree on '{k}' - the later preset wins" for k in conflicts]
    w = spec.get("walls")
    if w not in (None, "", "default"):
        cfg.world.walls = list(w) if isinstance(w, list) else WALL_CHOICES[w]
    for key, field in (("obstacle_mode", "obstacle_mode"), ("obstacle_timing", "obstacle_timing")):
        if spec.get(key) not in (None, "", "default"):
            setattr(cfg.world, field, spec[key])
    if spec.get("n_obstacles") not in (None, ""):
        cfg.world.n_obstacles = int(spec["n_obstacles"])
    sensors = spec.get("sensors")
    if sensors == "ultrasonic":
        cfg.ultrasonic.enabled, cfg.lidar.enabled = True, False
    elif sensors == "lidar":
        cfg.ultrasonic.enabled, cfg.lidar.enabled = False, True
    elif sensors == "both":
        cfg.ultrasonic.enabled = cfg.lidar.enabled = True
    uc = cfg.ultrasonic
    if spec.get("rear_sensor") and not any(s["name"] == "B" for s in uc.sensors):
        uc.sensors = list(uc.sensors) + [dict(REAR_SENSOR)]
        uc.fire_groups = [list(gr) for gr in uc.fire_groups] + [["B"]]
    names = [s["name"] for s in uc.sensors]
    firing = spec.get("firing")
    if firing == "all":
        uc.fire_groups = [names]
    elif firing == "one_by_one":
        uc.fire_groups = [[n] for n in names]
        uc.crosstalk_prob = 0.0
    elif firing == "grouped" and not spec.get("rear_sensor"):
        uc.fire_groups = [["F", "RL", "RR"], ["FL", "R"], ["FR", "L"]]
    for row in spec.get("overrides") or []:
        key, val = (row.get("key") or "").strip(), row.get("value")
        if key:
            set_dotted(cfg, key, parse_value(str(val)))
    if spec.get("seed") not in (None, ""):
        cfg.sim.seed = int(spec["seed"])
    return apply_clamps(cfg), notes


def describe(spec: dict) -> str:
    parts = ["+".join(spec.get("presets") or ["nominal"])]
    for key in ("walls", "obstacle_mode", "obstacle_timing", "sensors", "firing"):
        v = spec.get(key)
        if v not in (None, "", "default"):
            parts.append(f"{key.replace('obstacle_', '')}={v}")
    if spec.get("rear_sensor"):
        parts.append("+rear sensor")
    for row in spec.get("overrides") or []:
        if row.get("key"):
            parts.append(f"{row['key']}={row.get('value')}")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def clean(o, nd=4):
    """numpy / NaN / inf -> JSON-safe Python."""
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist(), nd)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return round(o, nd) if math.isfinite(o) else None
    if is_dataclass(o):
        return clean(asdict(o), nd)
    return o


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("bay",):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def serialize_single(res, spec):
    cfg, log = res.cfg, res.log
    t = log["t"]
    stride = max(1, int(round(cfg.sim.log_hz / 25)))           # ~25 Hz replay
    sl = slice(None, None, stride)
    keys = ["t", "x", "y", "yaw", "ex", "ey", "eyaw", "wl", "wr", "v", "v_ref", "v_est",
            "delta", "delta_cmd", "loc_err", "sx", "state", "link"]
    replay = {k: clean(np.asarray(log[k][sl], float)) for k in keys}

    paths = [{"t": tp, "x": clean(p.x[::2]), "y": clean(p.y[::2]), "dir": clean(p.dir[::2], 0)}
             for tp, p in res.paths]
    pings = {}
    for tp, name, r in res.pings:
        pings.setdefault(name, []).append([round(tp, 3), None if not np.isfinite(r) else round(r, 4)])
    scans = []
    if res.scans:
        mx = cfg.lidar.mount_x
        for t_arr, _, sp, ang, rng in res.scans[::2]:
            ok = np.isfinite(rng)
            ok[1::3] = ok[2::3] = False                        # every 3rd beam is plenty to draw
            px = mx + rng[ok] * np.cos(ang[ok])
            py = rng[ok] * np.sin(ang[ok])
            c, s = math.cos(sp[2]), math.sin(sp[2])
            scans.append({"t": round(t_arr, 3), "x": clean(sp[0] + px * c - py * s, 3),
                          "y": clean(sp[1] + px * s + py * c, 3)})
    grids = []
    from ..mapping import OccupancyGrid
    og = OccupancyGrid(cfg.world.width, cfg.world.height, cfg.mapping)
    last = -1.0
    for tg, arr in res.grids:
        if tg - last >= 1.0:
            last = tg
            grids.append({"t": round(tg, 2),
                          "data": base64.b64encode((np.clip(arr, 0, 1) * 255).astype(np.uint8).tobytes()).decode()})
    obstacles = [{"kind": o.kind, "x": o.x, "y": o.y, "r": o.r, "l": o.l, "w": o.w, "yaw": o.yaw,
                  "h": o.h, "spawn_t": o.spawn_t} for o in res.obstacles]
    summons = [c["pose"] for c in cfg.mission.commands if c.get("cmd") == "summon"]

    from .. import viz
    buf = io.BytesIO()
    viz.summary(res, save=buf, show=False)
    png = base64.b64encode(buf.getvalue()).decode()

    from ..world import wall_sides
    geom = {"width": cfg.world.width, "height": cfg.world.height, "walls": list(wall_sides(cfg.world.walls)),
            "bay": cfg.world.bay, "summons": summons, "vehicle": asdict(cfg.vehicle),
            "center_offset": cfg.vehicle.center_offset,
            "us": {"enabled": cfg.ultrasonic.enabled, "sensors": cfg.ultrasonic.sensors,
                   "half_angle_deg": cfg.ultrasonic.half_angle_deg, "mount_height": cfg.ultrasonic.mount_height},
            "lidar": {"enabled": cfg.lidar.enabled, "mount_height": cfg.lidar.mount_height},
            "grid": {"ox": og.ox, "oy": og.oy, "res": og.res, "nx": og.nx, "ny": og.ny}}
    return clean({"label": describe(spec), "geom": geom, "replay": replay, "paths": paths, "pings": pings,
                  "scans": scans, "grids": grids, "obstacles": obstacles,
                  "events": [[round(te, 2), e] for te, e in res.events], "tasks": res.tasks,
                  "metrics": {k: v for k, v in res.metrics.items() if k != "tasks"},
                  "states": {v: k for k, v in STATE_CODES.items()}, "summary_png": png})


def seed_row(m):
    if "error" in m:
        return {"seed": m["seed"], "ok": False, "error": m["error"], "tasks": []}
    return {"seed": m["seed"], "ok": m["all_success"], "collided": m.get("collided"),
            "left_testbed": m.get("left_testbed"), "loc_rmse": m.get("loc_rmse"),
            "tasks": [{"kind": t["kind"], "result": t["result"], "success": t["success"],
                       "err_pos": t["err_pos"], "err_heading_deg": t["err_heading_deg"], "time": t["time"]}
                      for t in m["tasks"]]}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.status = "running"
        self.progress = {"done": 0, "total": 1, "text": "starting..."}
        self.result = None
        self.error = None
        self.cancel = False
        self.started = time.time()

    def view(self):
        return clean({"id": self.id, "kind": self.kind, "status": self.status, "progress": self.progress,
                      "error": self.error, "result": self.result if self.status == "done" else None,
                      "elapsed": time.time() - self.started})


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()

# Limits protect the host machine when the interface is reachable by others
# (set by serve(); see `python -m parksim ui --help`).
LIMITS = {"max_runs": 200, "max_sims": 1000, "workers": None, "keep_jobs": 20}
# Settings that could make a single job run for hours are clamped to these ranges.
CLAMPS = {"sim.dt": (0.002, 0.02), "mission.max_time": (10.0, 300.0), "planner.max_expansions": (100, 20000),
          "ultrasonic.n_rays": (1, 101), "lidar.n_beams": (36, 720), "world.n_obstacles": (0, 15),
          "world.layout_retries": (1, 40)}
MAX_BODY = 1_000_000


def apply_clamps(cfg):
    from ..config import get_dotted
    for key, (lo, hi) in CLAMPS.items():
        v = get_dotted(cfg, key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            set_dotted(cfg, key, type(v)(min(max(v, lo), hi)))
    return cfg


def _prune_jobs():
    """Keep only the newest finished jobs (each single-run result is ~1 MB)."""
    done = sorted((j for j in JOBS.values() if j.status != "running"), key=lambda j: j.started)
    for j in done[:max(0, len(done) - LIMITS["keep_jobs"])]:
        JOBS.pop(j.id, None)


def _mc(cfg, n, job, label, offset=0, grand_total=None):
    from ..montecarlo import monte_carlo, summarize

    def cb(done, total, m):
        job.progress = {"done": offset + done, "total": grand_total or total,
                        "text": f"{label}: run {done}/{total}"}
        return not job.cancel
    res = monte_carlo(cfg, n=n, seed0=cfg.sim.seed, callback=cb, mp_context="spawn", workers=LIMITS["workers"])
    return res, summarize(res, cfg)


def run_job(job: Job, req: dict):
    try:
        kind = req["kind"]
        if kind == "single":
            from ..simulator import Simulation
            cfg, notes = build_config(req["spec"])
            job.progress = {"done": 0, "total": 1, "text": "simulating..."}
            res = Simulation(cfg).run()
            job.progress = {"done": 1, "total": 1, "text": "preparing replay..."}
            job.result = serialize_single(res, req["spec"])
            job.result["notes"] = notes
            job.result["spec"] = req["spec"]
        elif kind == "mc":
            cfg, notes = build_config(req["spec"])
            n = int(req.get("runs") or 20)
            res, summ = _mc(cfg, n, job, "Monte Carlo")
            job.result = clean({"label": describe(req["spec"]), "summary": summ, "notes": notes,
                                "seeds": [seed_row(m) for m in res], "spec": req["spec"]})
        elif kind == "compare":
            variants = req["variants"]
            n = int(req.get("runs") or 10)
            rows = []
            for i, v in enumerate(variants):
                if job.cancel:
                    break
                cfg, notes = build_config(v)
                res, summ = _mc(cfg, n, job, f"variant {i + 1}/{len(variants)}", offset=i * n,
                                grand_total=n * len(variants))
                rows.append({"label": describe(v), "spec": v, "summary": summ, "notes": notes,
                             "seeds": [seed_row(m) for m in res]})
            job.result = clean({"rows": rows, "runs": n})
        else:
            raise ValueError(f"unknown job kind {kind}")
        if job.cancel and isinstance(job.result, dict):
            job.result["cancelled"] = True
        job.status = "done"
    except Exception as exc:                       # report, don't kill the server
        job.status = "error"
        job.error = f"{exc}\n\n{traceback.format_exc()}"


def compare_variants(base: dict, grid: dict):
    """Cartesian product of the chosen values; empty dimensions keep the base value."""
    dims = [(k, grid[k]) for k in ("walls", "obstacle_mode", "obstacle_timing", "sensors", "rear_sensor")
            if grid.get(k)]
    if not dims:
        return [dict(base)]
    out = []
    for combo in itertools.product(*[vals for _, vals in dims]):
        v = json.loads(json.dumps(base))
        for (k, _), val in zip(dims, combo):
            v[k] = val
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def options_payload():
    cfg = Config()
    flat = flatten(clean(asdict(cfg)))
    return {"presets": [{"name": k, "description": d} for k, (d, _) in scenarios.SCENARIOS.items()],
            "keys": flat, "walls": list(WALL_CHOICES), "sensor_names": [s["name"] for s in _default_ultrasonic_layout()],
            "limits": {k: LIMITS[k] for k in ("max_runs", "max_sims")}}


class Handler(BaseHTTPRequestHandler):
    server_version = "parksim-ui"

    def log_message(self, fmt, *args):      # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise json.JSONDecodeError("request too large", "", 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html") or self.path.startswith("/?"):
            # options (and a finished job, for /?job=<id>) are inlined so the
            # page renders without waiting for extra requests
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            pre = {"options": options_payload()}
            job = JOBS.get((q.get("job") or [""])[0])
            if job and job.status == "done":
                pre["job"] = job.view()
            inline = "<script>window.PRELOAD = " + json.dumps(pre).replace("</", "<\\/") + ";</script>"
            html = (HERE / "index.html").read_text().replace("<!--PRELOAD-->", inline)
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        if self.path == "/api/options":
            return self._send(200, options_payload())
        if self.path.startswith("/api/job/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            return self._send(200, job.view()) if job else self._send(404, {"error": "no such job"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad JSON"})
        if self.path == "/api/config":
            try:
                cfg, notes = build_config(body.get("spec") or {})
                return self._send(200, {"config": clean(asdict(cfg)), "notes": notes,
                                        "label": describe(body.get("spec") or {})})
            except Exception as exc:
                return self._send(400, {"error": str(exc)})
        if self.path == "/api/variants":
            vs = compare_variants(body.get("spec") or {}, body.get("grid") or {})
            return self._send(200, {"variants": vs, "labels": [describe(v) for v in vs]})
        if self.path == "/api/run":
            kind = body.get("kind", "single")
            runs = int(body.get("runs") or (20 if kind == "mc" else 10))
            n_var = len(body.get("variants") or [None])
            if kind == "mc" and runs > LIMITS["max_runs"]:
                return self._send(400, {"error": f"at most {LIMITS['max_runs']} Monte Carlo runs per job on this server"})
            if kind == "compare" and runs * n_var > LIMITS["max_sims"]:
                return self._send(400, {"error": f"{n_var} variants x {runs} runs = {n_var * runs} simulations; "
                                                 f"this server allows {LIMITS['max_sims']} per comparison"})
            with LOCK:
                _prune_jobs()
                if any(j.status == "running" for j in JOBS.values()):
                    return self._send(409, {"error": "a job is already running - wait or cancel it"})
                try:                                   # validate before starting
                    for s in (body.get("variants") or [body.get("spec") or {}]):
                        build_config(s)
                except Exception as exc:
                    return self._send(400, {"error": f"invalid settings: {exc}"})
                job = Job(body.get("kind", "single"))
                JOBS[job.id] = job
            threading.Thread(target=run_job, args=(job, body), daemon=True).start()
            return self._send(200, {"job": job.id})
        if self.path.startswith("/api/job/") and self.path.endswith("/cancel"):
            job = JOBS.get(self.path.split("/")[3])
            if job:
                job.cancel = True
            return self._send(200, {"ok": bool(job)})
        self._send(404, {"error": "not found"})


def serve(port=8765, open_browser=True, host="127.0.0.1", max_runs=200, max_sims=1000, workers=None, keep_jobs=20):
    LIMITS.update(max_runs=max_runs, max_sims=max_sims, workers=workers, keep_jobs=keep_jobs)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"parksim interface running at {url}  (Ctrl+C to stop)")
    print(f"  limits: {max_runs} runs per Monte Carlo, {max_sims} simulations per comparison, "
          f"workers={workers or 'all CPUs - 1'}, keeping the last {keep_jobs} results")
    if host != "127.0.0.1":
        print(f"  WARNING: listening on {host} - anyone who can reach this port can use it")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
