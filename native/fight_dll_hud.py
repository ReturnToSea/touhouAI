"""Live overview for a real-game Letty run (train_ppo_dll.py) and its
deterministic transfer eval (sim/fight_transfer_daemon.py --runsdir runs).

    .venv\\Scripts\\python native\\fight_dll_hud.py ppo_real_letty

Everything here is the REAL game - train_ppo_dll trains on hooked th07 instances,
so the "training" curve IS a real-game curve (just a stochastic-policy one). The
transfer daemon adds a greedy (argmax) number per checkpoint.

Reads runs/<name>/ every 2 s:
  history.npy  8-col: wall_s, total_steps, surv_s, entropy, value_expl,
                      boss_engaged_frac, boss_hp_floor_med, mean_return
  realtransfer*.npy : wall_epoch, train_steps, active_surv_s, killed01, dmg_frac
  train.log        : tailed for status + live steps/s
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
BASELINE_SURV = 103.0        # replay baseline, docs/de-letty-replay.md
BASELINE_KILL = 33.0

BG = "#0e1014"
GRID = "#1e2230"
AXIS = "#7a8394"
TRAIN = "#4da6ff"
DMG = "#ffd24d"
REAL = "#ff5c8a"
REAL_DOT = "#5a2a3a"
KILL = "#5fd98a"
BASE = "#54607a"


def _tail(p: Path, n: int = 40):
    try:
        return p.read_text(errors="replace").splitlines()[-n:]
    except Exception:
        return []


def load_log(d: Path):
    lines = _tail(d / "train.log")
    out = {"status": "starting up", "sps": None, "upd": None}
    for ln in lines:
        s = ln.strip()
        if s.startswith("[RealRolloutVec]"):
            out["status"] = "instances up - warming up"
        elif s.startswith("[Th07Env]"):
            out["status"] = "launching hooked games..."
        elif s.startswith("warm-started"):
            out["status"] = "warm-started - first rollout"
        elif s.startswith("upd "):
            f = s.split()
            try:
                out["upd"] = int(f[1])
                out["steps_m"] = float(f[2].rstrip("M"))
                out["sps"] = float(f[3].rstrip("/s"))
                out["status"] = ("critic-warmup" if "critic-warmup" in s
                                 else "training")
                if "ent" in f:
                    out["ent"] = float(f[f.index("ent") + 1])
                if "ev" in f:
                    out["ev"] = float(f[f.index("ev") + 1])
            except (ValueError, IndexError):
                pass
    return out


def load(name: str):
    d = RUNS / name
    r = {"name": name, "log": load_log(d), "meta": {}}
    try:
        r["meta"] = json.loads((d / "meta.json").read_text())
    except Exception:
        pass
    try:
        h = np.load(d / "history.npy")
        if h.ndim == 2 and len(h) and h.shape[1] >= 8:
            r["wall"] = h[:, 0]
            r["steps"] = h[:, 1]
            r["surv"] = h[:, 2]
            r["ent"] = h[:, 3]
            r["ev"] = h[:, 4]
            r["eng"] = h[:, 5]
            r["dmg"] = (1.0 - h[:, 6]) * 100.0     # boss HP drained, %
            r["ret"] = h[:, 7]
    except Exception:
        pass

    parts = []
    for rp in sorted(d.glob("realtransfer*.npy")):
        try:
            a = np.load(rp)
            if a.ndim == 2 and len(a):
                parts.append(a)
        except Exception:
            pass
    if parts:
        rt = np.vstack(parts)
        o = np.argsort(rt[:, 1])
        r["rt_x"] = rt[o, 1] / 1e6
        r["rt_surv"] = rt[o, 2]
        r["rt_kill"] = rt[o, 3]
        r["rt_dmg"] = rt[o, 4] * 100.0
        n_ck = max(1, len(np.unique(rt[:, 1])))
        W = max(8, min(40, len(o) // n_ck))
        sv, kl = [], []
        for i in range(len(o)):
            a = max(0, i - W + 1)
            sv.append(np.median(r["rt_surv"][a:i + 1]))
            kl.append(np.mean(r["rt_kill"][a:i + 1]) * 100.0)
        r["rt_sv_roll"] = np.array(sv)
        r["rt_kl_roll"] = np.array(kl)
    return r


class Hud:
    def __init__(self, name: str):
        self.name = name
        self.root = tk.Tk()
        self.root.title(f"letty (real): {name}")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.txt = tk.Label(self.root, justify="left", anchor="nw",
                            font=("Consolas", 10), fg="#dfe3ea", bg=BG)
        self.txt.pack(fill="x", padx=12, pady=(10, 4))
        self.c1 = tk.Canvas(self.root, width=740, height=250, bg=BG,
                            highlightthickness=0)
        self.c1.pack(padx=12)
        self.c2 = tk.Canvas(self.root, width=740, height=210, bg=BG,
                            highlightthickness=0)
        self.c2.pack(padx=12, pady=(0, 12))
        self._tick()
        self.root.mainloop()

    def _tick(self):
        try:
            r = load(self.name)
            self._text(r)
            self._chart(self.c1, r, "survival  (seconds)   — all real game", "surv")
            self._chart(self.c2, r, "boss kill  (%)   — all real game", "kill")
        except Exception as e:                       # never let the HUD die
            self.txt.config(text=f"{self.name}\n(hud error: {e})")
        self.root.after(2000, self._tick)

    def _text(self, r):
        lg = r.get("log") or {}
        m = r.get("meta", {})
        L = [f"● {r['name']}   {m.get('n_envs','?')} real games   "
             f"hidden={m.get('hidden','?')}   warm={Path(str(m.get('warmstart',''))).name or '-'}"]
        tgt = m.get("steps")

        if "steps" in r:
            wall, steps = r["wall"][-1], r["steps"][-1]
            sps = lg.get("sps")
            L.append(f"   {wall/60:5.1f} min   {steps/1e6:6.2f}M"
                     + (f" / {tgt/1e6:.0f}M" if tgt else "")
                     + (f"   {sps:.0f} steps/s" if sps else "")
                     + f"   [{lg.get('status','?')}]")
            L.append(f"   TRAIN  surv {r['surv'][-1]:5.1f}s (best {np.nanmax(r['surv']):.0f})"
                     f"   boss-bar seen {r['eng'][-1]*100:3.0f}%"
                     f"   HP drained {r['dmg'][-1]:3.0f}% (best {np.nanmax(r['dmg']):.0f}%)")
            L.append(f"          entropy {r['ent'][-1]:.2f}   value-expl {r['ev'][-1]:+.2f}"
                     f"   mean return {r['ret'][-1]:.0f}")
        else:
            L.append(f"   [{lg.get('status','starting up')}]"
                     + (f"  upd {lg['upd']}" if lg.get("upd") else "")
                     + "   (history.npy fills in after the first update)")

        if "rt_surv" in r:
            rs, rk = r["rt_surv"][-20:], r["rt_kill"][-20:]
            L.append(f"   GREEDY  surv med {np.median(rs):5.1f}s  best {r['rt_surv'].max():.0f}s"
                     f"   kill {rk.mean()*100:3.0f}%   dmg {np.median(r['rt_dmg'][-20:]):.0f}%"
                     f"   (n={len(r['rt_surv'])})")
        else:
            L.append("   GREEDY  (transfer daemon not running / no episodes yet)")
        L.append(f"   bar to beat:  {BASELINE_SURV:.0f}s  /  {BASELINE_KILL:.0f}% kill   (replay baseline)")
        self.txt.config(text="\n".join(L))

    def _chart(self, cv, r, title, kind):
        cv.delete("all")
        W = int(cv["width"]); H = int(cv["height"])
        x0, x1, y0, y1 = 52, W - 12, H - 22, 22
        cv.create_text((x0 + x1) / 2, 10, fill="#9aa4b4", font=("Consolas", 9),
                       text=title)
        if "steps" not in r and "rt_x" not in r:
            cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, fill="#5a6472",
                           font=("Consolas", 9), text="waiting for data...")
            return

        sm = r["steps"] / 1e6 if "steps" in r else np.array([0.0])
        xmax = float(max(sm[-1] if len(sm) else 0,
                         r["rt_x"][-1] if "rt_x" in r else 0, 1.0))
        if kind == "surv":
            cand = [r["surv"]] if "surv" in r else []
            if "rt_surv" in r:
                cand += [r["rt_surv"], r["rt_sv_roll"]]
            ymax = max([np.nanmax(c) for c in cand if c is not None and len(c)]
                       + [BASELINE_SURV, 30.0]) * 1.12
        else:
            ymax = 100.0

        def X(v): return x0 + (x1 - x0) * v / xmax
        def Y(v): return y0 + (y1 - y0) * min(v, ymax) / ymax

        for k in range(5):
            gy = y0 + (y1 - y0) * k / 4
            cv.create_line(x0, gy, x1, gy, fill=GRID)
            cv.create_text(x0 - 6, gy, anchor="e", fill=AXIS,
                           font=("Consolas", 8), text=f"{ymax*(4-k)/4:.0f}"
                           if kind == "surv" else f"{100*(4-k)/4:.0f}")
        for k in range(6):
            gx = x0 + (x1 - x0) * k / 5
            cv.create_text(gx, y0 + 11, fill=AXIS, font=("Consolas", 8),
                           text=f"{xmax*k/5:.0f}M")

        def line(xs, ys, col, wd=2, dash=()):
            pts = [c for a, b in zip(xs, ys) if b == b
                   for c in (X(a), Y(b))]
            if len(pts) >= 4:
                cv.create_line(*pts, fill=col, width=wd, smooth=True, dash=dash)

        base = BASELINE_SURV if kind == "surv" else BASELINE_KILL
        cv.create_line(x0, Y(base), x1, Y(base), fill=BASE, dash=(6, 4))
        cv.create_text(x1, Y(base) - 7, anchor="e", fill=BASE,
                       font=("Consolas", 8), text="baseline")

        if kind == "surv":
            if "surv" in r:
                line(sm, r["surv"], TRAIN, 2)
                cv.create_text(X(sm[-1]), Y(r["surv"][-1]) - 8, anchor="e",
                               fill=TRAIN, font=("Consolas", 8),
                               text="live  (training policy, every update)")
            if "rt_surv" in r:
                for a, b in zip(r["rt_x"], r["rt_surv"]):
                    cv.create_oval(X(a) - 1.5, Y(b) - 1.5, X(a) + 1.5, Y(b) + 1.5,
                                   outline="", fill=REAL_DOT)
                line(r["rt_x"], r["rt_sv_roll"], REAL, 2)
                cv.create_text(X(r["rt_x"][-1]), Y(r["rt_sv_roll"][-1]) - 8,
                               anchor="e", fill=REAL, font=("Consolas", 8),
                               text=f"eval  (greedy, per checkpoint)  {r['rt_sv_roll'][-1]:.0f}s")
        else:
            if "dmg" in r:
                line(sm, r["dmg"], DMG, 1, (4, 3))
                cv.create_text(X(sm[-1]), Y(r["dmg"][-1]) - 8, anchor="e",
                               fill=DMG, font=("Consolas", 8),
                               text="live: boss HP drained %")
            if "rt_kl_roll" in r:
                for a, b in zip(r["rt_x"], r["rt_kill"] * 100.0):
                    cv.create_oval(X(a) - 1.5, Y(b) - 1.5, X(a) + 1.5, Y(b) + 1.5,
                                   outline="", fill=REAL_DOT)
                line(r["rt_x"], r["rt_kl_roll"], KILL, 2)
                cv.create_text(X(r["rt_x"][-1]), Y(r["rt_kl_roll"][-1]) - 8,
                               anchor="e", fill=KILL, font=("Consolas", 8),
                               text=f"eval: kill-rate  {r['rt_kl_roll'][-1]:.0f}%")


def main():
    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        cands = sorted([p for p in RUNS.iterdir()
                        if (p / "meta.json").exists()
                        and json.loads((p / "meta.json").read_text()).get("algo")
                        == "ppo_real_dll"],
                       key=lambda p: p.stat().st_mtime) if RUNS.exists() else []
        if not cands:
            print("usage: fight_dll_hud.py <run-name>   (e.g. ppo_real_letty)")
            return
        name = cands[-1].name
    Hud(name)


if __name__ == "__main__":
    main()
