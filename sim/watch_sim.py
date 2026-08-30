"""Watch a policy play the danmaku sim, in real time, with the same debug
overlay as native/viz.py (detection box, danger-grid heatmap, in/out-range
bullets, action arrow) PLUS the 9 escape scalars drawn as rays from the player.

    .venv-cuda\\Scripts\\python sim\\watch_sim.py runs_sim\\ppo_v1\\best.pt
    .venv-cuda\\Scripts\\python sim\\watch_sim.py runs_sim\\ppo_v1\\best.pt --follow
    .venv-cuda\\Scripts\\python sim\\watch_sim.py --random

Keys:  g grid on/off   space pause   r reset episode   esc quit
--follow reloads the .pt every few seconds so you can watch it improve mid-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from danmaku import DanmakuSim  # noqa: E402
from obs import (HEAD_DIM, NDIRS, GCELLS, GRID, GRID_CELL,  # noqa: E402
                 PX_LO, PX_HI, PY_LO, PY_HI, OBS_DIRS)
from policy import MLPPolicy  # noqa: E402

PW, PH = 384.0, 448.0
SCALE = 1.9
PAD = 34
REDRAW_MS = 33
DIRS = OBS_DIRS.tolist()   # [[0,0],[0,-1],...]


def heat(d):
    d = 0.0 if d < 0 else 1.0 if d > 1 else d
    return f"#{int(110 + 145 * d):02x}{int(95 * (1 - d)):02x}{int(45 * (1 - d)):02x}"


def safe_col(v):
    v = 0.0 if v < 0 else 1.0 if v > 1 else v
    return f"#{int(230 * (1 - v)):02x}{int(70 + 150 * v):02x}40"


class Watcher:
    def __init__(self, args):
        self.args = args
        self.sim = DanmakuSim(B=1, device="cpu", max_frames=100000,
                              seed=args.seed, compile=False)
        self.pol = None
        self.pol_mtime = 0
        self.run_dir = None if args.random else Path(args.model).parent
        self.meta = {}
        if self.run_dir and (self.run_dir / "meta.json").exists():
            try:
                self.meta = json.loads((self.run_dir / "meta.json").read_text())
            except Exception:
                pass
        self._hist = None
        self._hist_mt = 0.0
        self._load()
        self.obs = self.sim.reset()
        self.paused = False
        self.show_grid = True
        self.show_gmap = True
        self.last_load = time.time()
        self.ep_start_frame = 0

        self.root = tk.Tk()
        self.root.title("danmaku sim - watch")
        self.root.configure(bg="#0a0b0e")
        self.root.attributes("-topmost", True)
        w = int(PW * SCALE + PAD + 296)      # + right gutter for the stats panel
        h = int(PH * SCALE + 2 * PAD)
        self.cv = tk.Canvas(self.root, width=w, height=h, bg="#0a0b0e", highlightthickness=0)
        self.cv.pack()
        self.root.bind("g", lambda e: setattr(self, "show_grid", not self.show_grid))
        self.root.bind("G", lambda e: setattr(self, "show_gmap", not self.show_gmap))
        self.root.bind("<space>", lambda e: setattr(self, "paused", not self.paused))
        self.root.bind("r", lambda e: self._reset())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.after(REDRAW_MS, self._tick)
        self.root.mainloop()

    def _load(self):
        if self.args.random:
            return
        p = Path(self.args.model)
        if not p.exists():
            print(f"waiting for {p} ...")
            return
        m = p.stat().st_mtime
        if m == self.pol_mtime:
            return
        try:
            self.pol = MLPPolicy.load(p)
            self.pol_mtime = m
            print(f"loaded {p}  hidden={self.pol.hidden}  (mtime {time.ctime(m)})")
        except Exception as e:
            print("load failed:", e)

    def _reset(self):
        self.obs = self.sim.reset()
        self.ep_start_frame = 0

    def _stats_lines(self):
        """Training stats from runs_sim/<name>/history.npy (+ meta.json).
        history cols: ppo (wall, steps, dec, ret, ent); es (wall, steps, surv, bestret)."""
        if self.run_dir is None:
            return ["random policy"]
        import numpy as np
        hp = self.run_dir / "history.npy"
        if not hp.exists():
            return [self.run_dir.name, "(no history yet)"]
        try:
            mt = hp.stat().st_mtime
            if mt != self._hist_mt:
                self._hist = np.load(hp)
                self._hist_mt = mt
        except Exception:
            return [self.run_dir.name, "(reading...)"]
        h = self._hist
        if h is None or len(h) == 0 or h.shape[1] < 4:
            return [self.run_dir.name, "(warming up)"]
        algo = self.meta.get("algo", "ppo" if h.shape[1] == 5 else "es")
        wall, steps = float(h[-1, 0]), float(h[-1, 1])
        fs = self.sim.frame_skip
        tgt = self.meta.get("steps")
        out = [f"{self.run_dir.name}   [{algo}]   {self.meta.get('hidden', '')}",
               f"train time   {wall / 60:6.1f} min",
               f"steps        {steps / 1e6:6.1f} M" + (f" / {tgt / 1e6:.0f}M" if tgt else ""),
               f"speed        {steps / max(wall, 1) / 1e3:6.0f} k/s"]
        if algo == "ppo" and h.shape[1] >= 12:
            # v17+: wall_s, steps, mean_s, sampled_dec, ent, med_s, p90_s, f60, f120, f180, wallf, enemyf
            out += [f"median surv  {h[-1, 5]:6.1f} s",
                    f"p90 surv     {h[-1, 6]:6.1f} s   (best {np.nanmax(h[:, 6]):.0f})",
                    f">60/120/180s {h[-1,7]*100:.0f}/{h[-1,8]*100:.0f}/{h[-1,9]*100:.0f} %",
                    f"deaths wall/en {h[-1,10]*100:.0f}/{h[-1,11]*100:.0f} %",
                    f"entropy      {h[-1, 4]:6.3f}"]
        elif algo == "ppo":
            gd = h[-1, 2]
            sm = h[-1, 3] if h.shape[1] >= 4 else gd
            ent = h[-1, 4] if h.shape[1] >= 5 else float("nan")
            out += [f"greedy surv  {gd * fs / 60:6.1f} s   (best {h[:, 2].max() * fs / 60:.1f})",
                    f"sampled surv {sm * fs / 60:6.1f} s",
                    f"entropy      {ent:6.3f}"]
        else:
            surv, bret = h[-1, 2], h[-1, 3]
            out += [f"survival ~   {surv * fs / 60:6.1f} s",
                    f"best ep ret  {bret:6.2f}"]
        return out

    def _cx(self, x):
        return x * SCALE + PAD

    def _cy(self, y):
        return y * SCALE + PAD

    def _tick(self):
        try:
            if self.args.follow and time.time() - self.last_load > 3:
                self._load()
                self.last_load = time.time()
            if not self.paused:
                if self.pol is not None:
                    a = int(self.pol.act(self.obs[0].numpy()))
                elif self.args.random:
                    a = np.random.randint(36)
                else:
                    a = 0
                self.obs, rew, done = self.sim.step(torch.tensor([a]))
                self.last_action = a
                if bool(done[0]):
                    self.ep_start_frame = float(self.sim.frame[0])
            self._draw()
        except Exception:
            import traceback
            traceback.print_exc()
        self.root.after(REDRAW_MS, self._tick)

    def _draw(self):
        cv = self.cv
        cv.delete("all")
        s = self.sim
        px, py = float(s.player[0, 0]), float(s.player[0, 1])
        cx, cy = self._cx, self._cy
        act = s.b_active[0].numpy() > 0.5
        bpos = s.b_pos[0].numpy()[act]
        brad = s.b_rad[0].numpy()[act]
        obs = self.obs[0].numpy()
        esc = obs[HEAD_DIM:HEAD_DIM + NDIRS]
        grid = obs[HEAD_DIM + NDIRS:HEAD_DIM + NDIRS + GCELLS].reshape(GRID, GRID)
        a = getattr(self, "last_action", 0)

        cv.create_rectangle(cx(0), cy(0), cx(PW), cy(PH), outline="#22262e")
        cv.create_rectangle(cx(PX_LO), cy(PY_LO), cx(PX_HI), cy(PY_HI), outline="#333a44")

        # macro view (toggle G): predicted bullet tracks - each active bullet's
        # near-future path drawn as a faint line, so walls / dense streams read
        # as clear bands without washing the whole field red.
        if self.show_gmap:
            bvel = s.b_vel[0].numpy()[act]
            H_G = 24.0
            for (bx, by), (vx, vy) in zip(bpos, bvel):
                if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                    continue
                cv.create_line(cx(bx), cy(by), cx(bx + vx * H_G), cy(by + vy * H_G),
                               fill="#5a3030", width=1)

        # danger grid heatmap
        if self.show_grid:
            R = GRID // 2
            for j in range(GRID):
                for i in range(GRID):
                    d = float(grid[j, i])
                    if d <= 0.03:
                        continue
                    x0 = px + (i - R - 0.5) * GRID_CELL
                    y0 = py + (j - R - 0.5) * GRID_CELL
                    cv.create_rectangle(cx(x0), cy(y0), cx(x0 + GRID_CELL), cy(y0 + GRID_CELL),
                                        fill=heat(d), outline="", stipple="gray50")

        box = (GRID // 2 + 0.5) * GRID_CELL
        cv.create_rectangle(cx(px - box), cy(py - box), cx(px + box), cy(py + box),
                            outline="#3fa7ff", width=1, dash=(3, 3))

        # bullets: b_rad is the real th07 hitbox (2-3 px), sprite is ~2.4x that.
        # all drawn at real sprite size; near the player also gets a bright hitbox
        # dot. far ones = one flat oval (skip the stipple + dot) so a spam flood
        # still draws fast.
        if len(bpos):
            nearm = (np.abs(bpos[:, 0] - px) < box) & (np.abs(bpos[:, 1] - py) < box)
            for (bx, by), r in zip(bpos[~nearm], brad[~nearm]):
                spr = max(2.2, r * 2.4 * SCALE)
                cv.create_oval(cx(bx) - spr, cy(by) - spr, cx(bx) + spr, cy(by) + spr,
                               fill="#5b6472", outline="")
            for (bx, by), r in zip(bpos[nearm], brad[nearm]):
                spr = max(2.5, r * 2.4 * SCALE)
                hb = max(1.2, r * SCALE)
                cv.create_oval(cx(bx) - spr, cy(by) - spr, cx(bx) + spr, cy(by) + spr,
                               fill="#ff4d4d", outline="", stipple="gray50")
                cv.create_oval(cx(bx) - hb, cy(by) - hb, cx(bx) + hb, cy(by) + hb,
                               fill="#ff4d4d", outline="")

        # spam-phase spawners (orange rings near the top) + a phase banner
        sph = float(s.spam_phase[0])
        if sph > 0.5:
            n = int(s.spam_n[0].item())
            spx = s.sp_x[0].numpy()
            spy = s.sp_y[0].numpy()
            for k in range(n):
                cv.create_oval(cx(spx[k]) - 7, cy(spy[k]) - 7, cx(spx[k]) + 7, cy(spy[k]) + 7,
                               outline="#ff9c33", width=2)
            lab = f"SPAM  fire  {n} spawners" if sph < 1.5 else "SPAM  cooldown"
            cv.create_text(cx(PW / 2), cy(6), fill="#ff9c33", font=("Consolas", 10, "bold"),
                           text=lab)

        # front-only shot column: the shot only hits an enemy within +-26px of
        # the player's x and above it (teaches "position under the target").
        ALIGN = 26.0
        cv.create_line(cx(px - ALIGN), cy(0), cx(px - ALIGN), cy(py), fill="#3a3a22", dash=(2, 4))
        cv.create_line(cx(px + ALIGN), cy(0), cx(px + ALIGN), cy(py), fill="#3a3a22", dash=(2, 4))

        # enemies (magenta = active), hp bar above; line to the one being shot
        en_act = s.en_active[0].numpy() > 0.5
        en_pos = s.en_pos[0].numpy()
        en_hp = s.en_hp[0].numpy()
        shooting = (a // 18) % 2
        near_i, near_d = -1, 1e9
        for k in range(len(en_act)):
            if not en_act[k]:
                continue
            ex, ey = en_pos[k]
            d = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
            if abs(ex - px) < ALIGN and ey < py and d < near_d:   # aligned + above
                near_i, near_d = k, d
            cv.create_oval(cx(ex) - 9, cy(ey) - 9, cx(ex) + 9, cy(ey) + 9,
                           fill="#d54de0", outline="#ffffff")
            hpf = max(0.0, min(1.0, en_hp[k] / 2.0))
            cv.create_rectangle(cx(ex) - 10, cy(ey) - 16, cx(ex) - 10 + 20 * hpf, cy(ey) - 13,
                                fill="#66ff88", outline="")
        if shooting and near_i >= 0:
            ex, ey = en_pos[near_i]
            cv.create_line(cx(px), cy(py), cx(ex), cy(ey), fill="#ffe14d", width=2)

        # P items (falling) + the collect radius
        it_act = s.it_active[0].numpy() > 0.5
        it_pos = s.it_pos[0].numpy()
        for k in range(len(it_act)):
            if not it_act[k]:
                continue
            ix, iy = it_pos[k]
            cv.create_rectangle(cx(ix) - 3, cy(iy) - 3, cx(ix) + 3, cy(iy) + 3,
                                fill="#ff4fa3", outline="#ffd0e6")
        cv.create_oval(cx(px) - 14 * SCALE, cy(py) - 14 * SCALE,
                       cx(px) + 14 * SCALE, cy(py) + 14 * SCALE,
                       outline="#7a3350", dash=(2, 3))

        # escape rays (dirs 1..8), length ~ escape value
        for i in range(1, NDIRS):
            dx, dy = DIRS[i]
            n = (dx * dx + dy * dy) ** 0.5
            ln = 12 + esc[i] * 62
            ex, ey = px + dx / n * ln, py + dy / n * ln
            cv.create_line(cx(px), cy(py), cx(ex), cy(ey), fill=safe_col(esc[i]), width=3)
        # stay-still escape as a ring
        cv.create_oval(cx(px) - 9, cy(py) - 9, cx(px) + 9, cy(py) + 9,
                       outline=safe_col(esc[0]), width=2)

        # player + hitbox
        cv.create_oval(cx(px) - 2.5, cy(py) - 2.5, cx(px) + 2.5, cy(py) + 2.5,
                       fill="#ffffff", outline="")

        # action arrow
        di = a % 9
        focus = (a // 9) % 2
        dx, dy = DIRS[di]
        if dx or dy:
            n = (dx * dx + dy * dy) ** 0.5
            sp = 1.6 if focus else 4.0
            ex, ey = px + dx / n * sp * 10, py + dy / n * sp * 10
            cv.create_line(cx(px), cy(py), cx(ex), cy(ey), fill="#00e5ff", width=4, arrow="last")
        if focus:
            cv.create_oval(cx(px) - 14, cy(py) - 14, cx(px) + 14, cy(py) + 14, outline="#66ffcc")
        if (a // 18) % 2:
            cv.create_text(cx(px), cy(py) + 22, fill="#ffe14d", font=("Consolas", 8), text="SHOOT")

        cv.create_text(cx(0), 14, anchor="w", fill="#c8ccd4", font=("Consolas", 11),
                       text=f"frame {int(s.frame[0]):5d}  ~{float(s.frame[0]) / 60:5.1f}s   "
                            f"diff {float(s.diff[0]):.2f}   bullets {len(bpos):3d}   "
                            f"{'PAUSED' if self.paused else ''}")

        # training-stats panel (top-right)
        lines = self._stats_lines()
        x = cx(PW) + 6
        cv.create_rectangle(x - 4, 24, x + 260, 24 + 16 * len(lines) + 8,
                            fill="#12151b", outline="#2a2f3a")
        for i, ln in enumerate(lines):
            cv.create_text(x, 32 + i * 16, anchor="nw", fill="#9fd0ff" if i == 0 else "#c8ccd4",
                           font=("Consolas", 9), text=ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="runs_sim/ppo_v1/best.pt")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--follow", action="store_true", help="reload the .pt every 3s")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Watcher(args)


if __name__ == "__main__":
    main()
