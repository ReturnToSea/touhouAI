"""Live training overview for the sim runs (sim/train.py). Standalone window,
reads runs_sim/<name>/history.npy + meta.json every couple of seconds.

    .venv-cuda\\Scripts\\python sim\\hud.py                # newest run in runs_sim/
    .venv-cuda\\Scripts\\python sim\\hud.py ppo_v3
    .venv-cuda\\Scripts\\python sim\\hud.py ppo_v3 es1     # overlay several runs

Shows: train time, steps (/target), speed, current + best survival, entropy,
mean return, and a survival-vs-steps curve per run.
"""
from __future__ import annotations

import json
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_sim"
FS = 3            # sim frame_skip: decisions -> frames
COLORS = ["#4da6ff", "#ff9d4d", "#5fd98a", "#e05c8a", "#c9a0ff"]


def load(name):
    d = RUNS / name
    hp = d / "history.npy"
    if not hp.exists():
        return None
    try:
        h = np.load(hp)
    except Exception:
        return None
    meta = {}
    if (d / "meta.json").exists():
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            pass
    if h.ndim != 2 or len(h) == 0:
        return None
    algo = meta.get("algo", "ppo" if h.shape[1] >= 5 else "es")
    n = len(h)
    nan = np.full(n, np.nan)
    med = p90 = f60 = f120 = f180 = wallf = enemyf = nan
    if algo == "ppo" and h.shape[1] >= 12:
        # v17+ cols: wall_s, steps, mean_s, sampled_dec, ent, med_s, p90_s,
        #            f60, f120, f180, wallf, enemyf   (survival cols already in seconds)
        wall, steps = h[:, 0], h[:, 1]
        surv = h[:, 2]
        ret = h[:, 3]
        ent = h[:, 4]
        med, p90 = h[:, 5], h[:, 6]
        f60, f120, f180 = h[:, 7], h[:, 8], h[:, 9]
        wallf, enemyf = h[:, 10], h[:, 11]
    elif algo == "ppo":
        # legacy cols: wall, steps, greedy_dec, sampled_dec, ent
        wall, steps = h[:, 0], h[:, 1]
        surv = h[:, 2] * FS / 60.0
        ret = h[:, 3] * FS / 60.0 if h.shape[1] >= 4 else surv
        ent = h[:, 4] if h.shape[1] >= 5 else nan
    else:
        wall, steps, dec = (h[:, 0], h[:, 1], h[:, 2]) if h.shape[1] == 4 else \
                           (nan, h[:, 0], h[:, 1])
        surv = dec * FS / 60.0
        ret = h[:, -1]
        ent = nan
    return dict(name=name, algo=algo, meta=meta, wall=wall, steps=steps,
               surv=surv, ret=ret, ent=ent, dcap=nan,
               med=med, p90=p90, f60=f60, f120=f120, f180=f180,
               wallf=wallf, enemyf=enemyf)


class Hud:
    def __init__(self, names):
        self.names = names
        self.root = tk.Tk()
        self.root.title("sim training")
        self.root.configure(bg="#0e1014")
        self.root.attributes("-topmost", True)
        self.txt = tk.Label(self.root, justify="left", anchor="nw", font=("Consolas", 10),
                            fg="#dfe3ea", bg="#0e1014")
        self.txt.pack(fill="x", padx=12, pady=(10, 4))
        self.cv = tk.Canvas(self.root, width=680, height=300, bg="#0e1014",
                            highlightthickness=0)
        self.cv.pack(padx=12, pady=(0, 12))
        self._tick()
        self.root.mainloop()

    def _tick(self):
        runs = [r for r in (load(n) for n in self.names) if r]
        self._text(runs)
        self._plot(runs)
        self.root.after(2000, self._tick)

    def _text(self, runs):
        if not runs:
            self.txt.config(text="(no runs found in runs_sim/)")
            return
        lines = []
        for i, r in enumerate(runs):
            wall = r["wall"][-1]
            steps = r["steps"][-1]
            tgt = r["meta"].get("steps")
            sps = steps / wall / 1e3 if wall and not np.isnan(wall) and wall > 0 else float("nan")
            head = (f"{'●'} {r['name']}  [{r['algo']}]  "
                    f"hidden={r['meta'].get('hidden', '?')}")
            lines.append(head)
            lines.append(
                f"    time {('%.1f min' % (wall / 60)) if not np.isnan(wall) else '  ? '}"
                f"   steps {steps / 1e6:6.1f}M" + (f" / {tgt / 1e6:.0f}M" if tgt else "") +
                (f"   {sps:.0f}k/s" if not np.isnan(sps) else ""))
            if not np.isnan(r["med"][-1]):
                lines.append(
                    f"    median {r['med'][-1]:5.1f}s   p90 {r['p90'][-1]:6.1f}s"
                    f" (best p90 {np.nanmax(r['p90']):.0f})"
                    f"   >60s {r['f60'][-1]*100:.0f}%  >120s {r['f120'][-1]*100:.0f}%"
                    f"  >180s {r['f180'][-1]*100:.0f}%")
                lines.append(
                    f"    deaths: wall {r['wallf'][-1]*100:.0f}%  enemy {r['enemyf'][-1]*100:.0f}%"
                    f"   ent {r['ent'][-1]:.2f}   mean {r['surv'][-1]:.0f}s")
            else:
                lines.append(
                    f"    greedy survival {r['surv'][-1]:5.1f}s (best {r['surv'].max():4.1f})"
                    + (f"   ent {r['ent'][-1]:.2f}" if not np.isnan(r['ent'][-1]) else ""))
        self.txt.config(text="\n".join(lines))

    def _plot(self, runs):
        cv = self.cv
        cv.delete("all")
        W, H, PL, PR, PT, PB = 680, 300, 52, 12, 12, 24
        if not runs:
            return
        xmax = max(r["steps"][-1] for r in runs) / 1e6 or 1.0
        def _mx(r):
            return np.nanmax(r["p90"]) if not np.all(np.isnan(r["p90"])) else r["surv"].max()
        ymax = max(2.0, max(_mx(r) for r in runs)) * 1.1
        x0, x1, y0, y1 = PL, W - PR, H - PB, PT

        def X(v):
            return x0 + (x1 - x0) * v / xmax

        def Y(v):
            return y0 + (y1 - y0) * v / ymax

        for k in range(5):
            gy = y0 + (y1 - y0) * k / 4
            cv.create_line(x0, gy, x1, gy, fill="#1e2230")
            cv.create_text(x0 - 6, gy, anchor="e", fill="#7a8394", font=("Consolas", 8),
                           text=f"{ymax * k / 4:.0f}s")
        for k in range(6):
            gx = x0 + (x1 - x0) * k / 5
            cv.create_text(gx, y0 + 12, fill="#7a8394", font=("Consolas", 8),
                           text=f"{xmax * k / 5:.0f}M")

        for i, r in enumerate(runs):
            c = COLORS[i % len(COLORS)]
            sm = r["steps"] / 1e6
            has_pct = not np.all(np.isnan(r["med"]))
            for key, w, dash in ((("p90" if has_pct else "surv"), 2, ()),
                                 ("med", 2, (4, 3))):
                v = r[key]
                pts = [coord for s, y in zip(sm, v) if not np.isnan(y)
                       for coord in (X(s), Y(y))]
                if len(pts) >= 4:
                    cv.create_line(*pts, fill=c, width=w, smooth=True, dash=dash)
            lab = r["name"] + ("  (— p90  ·· med)" if has_pct else "")
            yv = r["p90"][-1] if has_pct else r["surv"][-1]
            cv.create_text(X(sm[-1]), Y(yv) - 8, anchor="e",
                           fill=c, font=("Consolas", 8), text=lab)
        cv.create_text((x0 + x1) / 2, PT, fill="#9aa4b4", font=("Consolas", 9),
                       text="survival: p90 (solid) + median (dashed)  vs  steps")


def main():
    args = sys.argv[1:]
    if not args:
        if not RUNS.exists():
            print("no runs_sim/ yet")
            return
        dirs = sorted([p for p in RUNS.iterdir() if (p / "history.npy").exists()],
                      key=lambda p: (p / "history.npy").stat().st_mtime)
        if not dirs:
            print("no runs with history.npy in runs_sim/")
            return
        args = [dirs[-1].name]
    Hud(args)


if __name__ == "__main__":
    main()
