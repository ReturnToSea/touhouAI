"""Live overview for a fight run (sim/train_fight.py) + its real-game transfer
(sim/fight_transfer_daemon.py). Two stacked charts:

  1. SURVIVAL (seconds)  - sim mean/median, sim kill-time, real-game survival
  2. BOSS KILL %          - sim kill-rate, real-game kill-rate

    .venv-cuda/Scripts/python sim/fight_hud.py fight_letty_seg

Reads runs_sim/<name>/{history.npy, meta.json, realtransfer.npy} every 2s.
history.npy 6-col: [wall_s, steps, med_s, mean_s, kill_rate, kill_time_s]
realtransfer.npy : [wall_epoch, steps, active_surv_s, killed01, dmg_frac]
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs_sim"
BG = "#0e1014"
GRID = "#1e2230"
AXIS = "#7a8394"
SIM = "#4da6ff"
KT = "#ffd24d"
REAL = "#ff5c8a"
REAL_DOT = "#5a2a3a"
KILL = "#5fd98a"


def _meta(d):
    try:
        return json.loads((d / "meta.json").read_text())
    except Exception:
        return {}


def load_log(name):
    """Live scrape of the training stdout log (runs_sim/<name>.log) so the HUD
    has something to show before history.npy exists / between evals."""
    for p in (RUNS / f"{name}.log", RUNS / name / "train.log"):
        if p.exists():
            break
    else:
        return None
    try:
        tail = p.read_text(errors="replace").splitlines()[-40:]
    except Exception:
        return None
    out = {"status": None, "upd": None}
    for ln in tail:
        s = ln.strip()
        if s.startswith("[ecl] building"):
            out["status"] = "building danmaku schedules..."
        elif s.startswith("[FightSim]"):
            out["status"] = "compiling / first rollout..."
        elif s.startswith("upd "):
            f = s.split()
            try:
                out["upd"] = int(f[1])
                out["steps_m"] = float(f[2].rstrip("M"))
                out["sps_k"] = float(f[3].rstrip("k/s"))
                out["med"] = float(f[f.index("med") + 1].rstrip("s"))
                out["kill"] = float(f[f.index("kill") + 1].rstrip("%")) / 100.0
                out["phase"] = float(f[f.index("phase") + 1].split("/")[0])
                out["status"] = "training"
            except (ValueError, IndexError):
                pass
    return out


def load(name):
    d = RUNS / name
    try:
        h = np.load(d / "history.npy")
    except Exception:
        r = {"name": name, "log": load_log(name), "meta": _meta(d)}
        _load_rt(d, r)
        return r
    if h.ndim != 2 or len(h) == 0 or h.shape[1] < 6:
        r = {"name": name, "log": load_log(name), "meta": _meta(d)}
        _load_rt(d, r)
        return r
    meta = {}
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        pass
    r = dict(name=name, meta=meta, log=load_log(name),
             wall=h[:, 0], steps=h[:, 1], med=h[:, 2], mean=h[:, 3],
             kill=h[:, 4], ktime=h[:, 5])
    _load_rt(d, r)
    return r


def _load_rt(d, r):
    parts = []
    for rp in sorted(d.glob("realtransfer*.npy")):   # 1 file, or 1 per daemon
        try:
            a = np.load(rp)
            if a.ndim == 2 and len(a):
                parts.append(a)
        except Exception:
            pass
    rt = np.vstack(parts) if parts else None
    if rt is not None:
        o = np.argsort(rt[:, 1])
        r["rt_steps"] = rt[o, 1] / 1e6
        r["rt_surv"] = rt[o, 2]
        r["rt_kill"] = rt[o, 3]
        # rolling window ~= one checkpoint's worth of episodes (all daemons)
        W = max(12, min(60, len(o) // max(1, len(np.unique(rt[:, 1])))))
        xs, sv, kl = [], [], []
        for i in range(len(o)):
            a = max(0, i - W + 1)
            xs.append(r["rt_steps"][i])
            sv.append(np.median(r["rt_surv"][a:i + 1]))
            kl.append(np.mean(r["rt_kill"][a:i + 1]) * 100.0)
        r["rt_x"] = np.array(xs)
        r["rt_sv_roll"] = np.array(sv)
        r["rt_kl_roll"] = np.array(kl)


class Hud:
    def __init__(self, name):
        self.name = name
        self.root = tk.Tk()
        self.root.title(f"fight: {name}")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.txt = tk.Label(self.root, justify="left", anchor="nw",
                            font=("Consolas", 10), fg="#dfe3ea", bg=BG)
        self.txt.pack(fill="x", padx=12, pady=(10, 4))
        self.c1 = tk.Canvas(self.root, width=720, height=250, bg=BG,
                            highlightthickness=0)
        self.c1.pack(padx=12)
        self.c2 = tk.Canvas(self.root, width=720, height=190, bg=BG,
                            highlightthickness=0)
        self.c2.pack(padx=12, pady=(0, 12))
        self._tick()
        self.root.mainloop()

    def _tick(self):
        r = load(self.name)
        self._text(r)
        self._chart(self.c1, r, "survival  (seconds)", kind="surv")
        self._chart(self.c2, r, "boss kill  (%)", kind="kill")
        self.root.after(2000, self._tick)

    def _text(self, r):
        if r is None:
            self.txt.config(text=f"({self.name}: run dir not found)")
            return
        meta = r.get("meta", {})
        lg = r.get("log") or {}
        L = [f"● {r.get('name', self.name)}   hidden={meta.get('hidden','?')}   "
             f"B={meta.get('B','?')}"]
        tgt = meta.get("steps")

        if "wall" in r:                              # history.npy present
            wall, steps = r["wall"][-1], r["steps"][-1]
            # instantaneous rate, not the since-startup average (which is dragged
            # down for ages by the ~5 min schedule-build + compile warmup). Prefer
            # the trainer's own printed k/s; else diff the last few history rows.
            sps = float("nan")
            if lg.get("sps_k"):
                sps = lg["sps_k"]
            elif len(r["wall"]) >= 2:
                w = np.asarray(r["wall"]); st = np.asarray(r["steps"])
                j = max(0, len(w) - 6)
                dw = w[-1] - w[j]
                if dw > 0:
                    sps = (st[-1] - st[j]) / dw / 1e3
            L.append(f"   time {wall/60:5.1f} min   steps {steps/1e6:6.1f}M"
                     + (f" / {tgt/1e6:.0f}M" if tgt else "")
                     + (f"   {sps:.0f}k/s" if sps == sps else ""))
            kt = r["ktime"][-1]
            ktxt = (f"   kill-time {kt:.0f}s (best {np.nanmin(r['ktime']):.0f})"
                    if kt == kt else "   kill-time --")
            L.append(f"   SIM   survival med {r['med'][-1]:5.1f}s  mean {r['mean'][-1]:5.1f}s"
                     f"    kill {r['kill'][-1]*100:3.0f}% (best {np.nanmax(r['kill'])*100:.0f}%)"
                     f"{ktxt}   best-surv {np.nanmax(r['med']):.0f}s")
        elif lg.get("status") == "training":         # log only, first eval seen
            L.append(f"   steps {lg['steps_m']:6.1f}M"
                     + (f" / {tgt/1e6:.0f}M" if tgt else "")
                     + f"   {lg['sps_k']:.0f}k/s   upd {lg['upd']}")
            L.append(f"   SIM   survival med {lg['med']:5.1f}s"
                     f"    kill {lg['kill']*100:3.0f}%    phase {lg['phase']:.2f}/4"
                     f"   (history.npy building - charts fill in shortly)")
        else:
            L.append(f"   {lg.get('status', 'starting up')}"
                     + (f" / {tgt/1e6:.0f}M target" if tgt else ""))
            L.append("   SIM   (no eval yet - first one lands ~1 min after compile)")

        if "rt_surv" in r and len(r["rt_surv"]):
            rs, rk = r["rt_surv"][-15:], r["rt_kill"][-15:]
            L.append(f"   REAL  survival med {np.median(rs):5.1f}s  "
                     f"best {r['rt_surv'].max():.0f}s"
                     f"    kill {rk.mean()*100:3.0f}%  (n={len(r['rt_surv'])})")
        else:
            L.append("   REAL  (transfer daemon warming up / no episodes yet)")
        self.txt.config(text="\n".join(L))

    def _chart(self, cv, r, title, kind):
        cv.delete("all")
        W = int(cv["width"]); H = int(cv["height"])
        x0, x1, y0, y1 = 52, W - 12, H - 22, 22
        cv.create_text((x0 + x1) / 2, 10, fill="#9aa4b4", font=("Consolas", 9),
                       text=title)
        if r is None or "steps" not in r:
            cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, fill="#5a6472",
                           font=("Consolas", 9),
                           text="waiting for the first eval...")
            return
        sm = r["steps"] / 1e6
        xmax = float(max(sm[-1], (r["rt_steps"][-1] if "rt_steps" in r and
                                  len(r["rt_steps"]) else 0), 1.0))

        if kind == "surv":
            series = [r["mean"], r["med"], r["ktime"]]
            if "rt_surv" in r:
                series.append(r["rt_surv"])
                series.append(r["rt_sv_roll"])
            ymax = max([np.nanmax(s) for s in series if s is not None
                        and np.isfinite(s).any()] + [10.0]) * 1.10
        else:
            ymax = 100.0

        def X(v):
            return x0 + (x1 - x0) * v / xmax

        def Y(v):
            return y0 + (y1 - y0) * min(v, ymax) / ymax

        for k in range(5):
            gy = y0 + (y1 - y0) * k / 4
            cv.create_line(x0, gy, x1, gy, fill=GRID)
            cv.create_text(x0 - 6, gy, anchor="e", fill=AXIS,
                           font=("Consolas", 8), text=f"{ymax*k/4:.0f}")
        for k in range(6):
            gx = x0 + (x1 - x0) * k / 5
            cv.create_text(gx, y0 + 11, fill=AXIS, font=("Consolas", 8),
                           text=f"{xmax*k/5:.0f}M")

        def line(xs, ys, col, wd=2, dash=()):
            pts = [c for a, b in zip(xs, ys) if b == b
                   for c in (X(a), Y(b))]
            if len(pts) >= 4:
                cv.create_line(*pts, fill=col, width=wd, smooth=True, dash=dash)

        if kind == "surv":
            line(sm, r["mean"], SIM, 2)
            line(sm, r["med"], SIM, 1, (4, 3))
            line(sm, r["ktime"], KT, 2)
            cv.create_text(x1, Y(np.nanmax(r["mean"])) - 8, anchor="e", fill=SIM,
                           font=("Consolas", 8),
                           text="sim  — mean  ·· median")
            if len(sm) and r["ktime"][-1] == r["ktime"][-1]:
                cv.create_text(X(sm[-1]), Y(r["ktime"][-1]) - 8, anchor="e",
                               fill=KT, font=("Consolas", 8), text="kill-time")
            if "rt_surv" in r and len(r["rt_surv"]):
                for a, b in zip(r["rt_steps"], r["rt_surv"]):
                    cv.create_oval(X(a) - 1.5, Y(b) - 1.5, X(a) + 1.5, Y(b) + 1.5,
                                   outline="", fill=REAL_DOT)
                line(r["rt_x"], r["rt_sv_roll"], REAL, 2)
                cv.create_text(X(r["rt_x"][-1]), Y(r["rt_sv_roll"][-1]) - 8,
                               anchor="e", fill=REAL, font=("Consolas", 8),
                               text=f"real {r['rt_sv_roll'][-1]:.0f}s")
        else:
            line(sm, r["kill"] * 100.0, KILL, 2)
            cv.create_text(x1, Y(np.nanmax(r["kill"] * 100.0)) - 8, anchor="e",
                           fill=KILL, font=("Consolas", 8), text="sim kill %")
            if "rt_kl_roll" in r and len(r["rt_kl_roll"]):
                for a, b in zip(r["rt_steps"], r["rt_kill"] * 100.0):
                    cv.create_oval(X(a) - 1.5, Y(b) - 1.5, X(a) + 1.5, Y(b) + 1.5,
                                   outline="", fill=REAL_DOT)
                line(r["rt_x"], r["rt_kl_roll"], REAL, 2)
                cv.create_text(X(r["rt_x"][-1]), Y(r["rt_kl_roll"][-1]) - 8,
                               anchor="e", fill=REAL, font=("Consolas", 8),
                               text=f"real {r['rt_kl_roll'][-1]:.0f}%")


def main():
    if len(sys.argv) < 2:
        dirs = sorted([p for p in RUNS.iterdir()
                       if (p / "history.npy").exists() and (p / "meta.json").exists()
                       and json.loads((p / "meta.json").read_text()).get("algo") == "ppo_fight"],
                      key=lambda p: (p / "history.npy").stat().st_mtime)
        if not dirs:
            print("usage: fight_hud.py <run-name>")
            return
        name = dirs[-1].name
    else:
        name = sys.argv[1]
    Hud(name)


if __name__ == "__main__":
    main()
