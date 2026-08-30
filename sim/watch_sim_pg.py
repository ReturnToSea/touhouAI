"""pygame build of watch_sim.py - same overlay, SDL renderer so it stays at
60 fps under thousands of bullets (the tkinter Canvas recreates every item each
frame and chokes). Keep both so you can compare: watch_sim.py (tkinter) vs this.

    .venv-cuda\\Scripts\\python sim\\watch_sim_pg.py runs_sim\\ppo_v27\\best.pt --follow
    .venv-cuda\\Scripts\\python sim\\watch_sim_pg.py --random
    .venv-cuda\\Scripts\\python sim\\watch_sim_pg.py <model> --gpu   # sim on cuda

Keys:  g grid   G macro tracks   space pause   r reset   esc quit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import torch

torch.set_num_threads(1)   # B=1 sim - default OMP threads thrash all cores for ~nothing

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from danmaku import (DanmakuSim, E_CONE, E_SPRAY, E_LINE, E_BRING, PLAYER_HB,  # noqa: E402
                     SHOOT_ALIGN_DX)
from obs import (HEAD_DIM, NDIRS, GCELLS, GRID, GRID_CELL,  # noqa: E402
                 PX_LO, PX_HI, PY_LO, PY_HI, OBS_DIRS)
from policy import MLPPolicy  # noqa: E402

PW, PH = 384.0, 448.0
SCALE = 1.9
PAD = 34
DIRS = OBS_DIRS.tolist()
BG = (10, 11, 14)


def hx(s):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


_SRC_COL = {k: hx(v) for k, v in {
    "cone": "#4fa3ff", "spray": "#2f6fd0", "line": "#e6e6e6",
    "bring": "#c07fff", "spam": "#ff9c33", "enemy": "#4fe08a"}.items()}
_ET = {E_CONE: "cone", E_SPRAY: "spray", E_LINE: "line", E_BRING: "bring"}
_MT = {0: "", 1: "acc", 2: "dec", 3: "snake", 4: "arc", 5: "home", 6: "puls", 7: "freeze"}


def heat(d):
    d = float(np.clip(d, 0.0, 1.0))
    return (int(110 + 145 * d), int(95 * (1 - d)), int(45 * (1 - d)))


def safe_col(v):
    v = float(np.clip(v, 0.0, 1.0))
    return (int(230 * (1 - v)), int(70 + 150 * v), 64)


def _cx(x):
    return float(x) * SCALE + PAD


def _cy(y):
    return float(y) * SCALE + PAD


def dashed(surf, col, p0, p1, dash=5, gap=4, width=1):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    ln = (dx * dx + dy * dy) ** 0.5
    if ln < 1:
        return
    ux, uy = dx / ln, dy / ln
    t = 0.0
    while t < ln:
        a = (x0 + ux * t, y0 + uy * t)
        b = (x0 + ux * min(t + dash, ln), y0 + uy * min(t + dash, ln))
        pygame.draw.line(surf, col, a, b, width)
        t += dash + gap


class Watcher:
    def __init__(self, args):
        self.args = args
        dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        self.sim = DanmakuSim(B=1, device=dev, max_frames=100000,
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
        self._slot_col = self._build_slot_colours()
        self._load()
        self.obs = self.sim.reset()
        self.paused = False
        self.show_grid = True
        self.show_gmap = True
        self.last_load = time.time()
        self.last_action = 0

        pygame.init()
        pygame.display.set_caption("danmaku sim - watch (pygame)")
        self.W = int(PW * SCALE + PAD + 296)
        self.H = int(PH * SCALE + 2 * PAD)
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.f8 = pygame.font.SysFont("consolas", 12)
        self.f7 = pygame.font.SysFont("consolas", 11)
        self.fsmall = pygame.font.SysFont("consolas", 10)
        # a reusable per-pixel-alpha layer for the stippled fills
        self.alpha = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------ policy
    def _load(self):
        if self.args.random:
            return
        p = Path(self.args.model)
        if not p.exists():
            return
        m = p.stat().st_mtime
        if m == self.pol_mtime:
            return
        try:
            self.pol = MLPPolicy.load(p)
            self.pol_mtime = m
            print(f"loaded {p}  hidden={self.pol.hidden}  ({time.ctime(m)})")
        except Exception as e:
            print("load failed:", e)

    def _build_slot_colours(self):
        s = self.sim
        col = np.zeros((s.N, 3), np.uint8)
        col[:] = _SRC_COL["cone"]
        rt = s._R_type.cpu().numpy()
        for i in range(s._spam_base):
            e = int(rt[min(i // s.SPE, s.E - 1)])
            col[i] = _SRC_COL[_ET.get(e, "cone")]
        col[s._spam_base:s._en_base] = _SRC_COL["spam"]
        col[s._en_base:s.dump] = _SRC_COL["enemy"]
        return col

    def _reset(self):
        self._reset_req = True   # picked up by the sim thread

    # ------------------------------------------------------------------ stats
    def _stats_lines(self):
        if self.run_dir is None:
            return ["random policy"]
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
            sp, en = h[-1, 10] * 100, h[-1, 11] * 100
            out += [f"median surv  {h[-1, 5]:6.1f} s",
                    f"p90 surv     {h[-1, 6]:6.1f} s   (best {np.nanmax(h[:, 6]):.0f})",
                    f">60/120/180s {h[-1,7]*100:.0f}/{h[-1,8]*100:.0f}/{h[-1,9]*100:.0f} %",
                    f"deaths  emit {max(0,100-sp-en):.0f}%  spam {sp:.0f}%  en {en:.0f}%",
                    f"entropy      {h[-1, 4]:6.3f}"]
        elif algo == "ppo":
            gd = h[-1, 2]
            sm = h[-1, 3] if h.shape[1] >= 4 else gd
            ent = h[-1, 4] if h.shape[1] >= 5 else float("nan")
            out += [f"greedy surv  {gd * fs / 60:6.1f} s",
                    f"sampled surv {sm * fs / 60:6.1f} s",
                    f"entropy      {ent:6.3f}"]
        else:
            out += [f"survival ~   {h[-1,2] * fs / 60:6.1f} s",
                    f"best ep ret  {h[-1,3]:6.2f}"]
        return out

    # ------------------------------------------------------------------ sim thread
    # The B=1 sim step (frame_skip x _advance) is ~30 ms on CPU and dominates.
    # Run it on its own thread so the render loop stays a locked 60 fps; torch
    # releases the GIL during tensor ops so the draw thread isn't starved.
    def _snapshot(self):
        s = self.sim
        n = SimpleNamespace()
        act = s.b_active[0].cpu().numpy() > 0.5
        n.bpos = s.b_pos[0].cpu().numpy()[act]
        n.bvel = s.b_vel[0].cpu().numpy()[act]
        n.brad = self._brad[act]
        n.bcol = self._slot_col[act]
        n.player = s.player[0].cpu().numpy().astype(float)
        n.obs = self.obs[0].cpu().numpy()
        n.e_pos = s.e_pos[0].cpu().numpy()
        n.e_on = s.e_on[0].cpu().numpy() > 0.5
        n.e_type = s.e_type[0].cpu().numpy()
        n.e_mt = s.e_mtype[0].cpu().numpy()
        n.en_act = s.en_active[0].cpu().numpy() > 0.5
        n.en_pos = s.en_pos[0].cpu().numpy()
        n.en_hp = s.en_hp[0].cpu().numpy()
        n.it_act = s.it_active[0].cpu().numpy() > 0.5
        n.it_pos = s.it_pos[0].cpu().numpy()
        n.spam_phase = float(s.spam_phase[0])
        n.spam_n = int(s.spam_n[0].item())
        n.sp_x = s.sp_x[0].cpu().numpy()
        n.sp_y = s.sp_y[0].cpu().numpy()
        n.frame = int(s.frame[0])
        n.diff = float(s.diff[0])
        n.action = self.last_action
        return n

    def _sim_loop(self):
        target_dt = self.sim.frame_skip / 60.0   # one step == this much real time
        while self.running:
            t0 = time.perf_counter()
            if getattr(self, "_reset_req", False):
                self._reset_req = False
                self.obs = self.sim.reset()
            if self.paused:
                time.sleep(0.10)              # STOPPED: ~no CPU
                continue
            if self.args.follow and time.time() - self.last_load > 3:
                self._load()
                self.last_load = time.time()
            if self.pol is not None:
                a = int(self.pol.act(self.obs[0].cpu().numpy()))
            elif self.args.random:
                a = int(np.random.randint(36))
            else:
                a = 0
            self.obs, _, _ = self.sim.step(torch.tensor([a], device=self.sim.dev))
            self.last_action = a
            snap = self._snapshot()
            with self.lock:
                self.snap = snap
            # pace to real time - no reason to sim faster than the game runs, and
            # the sleep hands the core back
            lag = target_dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)

    # ------------------------------------------------------------------ loop
    def run(self):
        import threading
        self._brad = self.sim._slot_rad.cpu().numpy()
        self.running = True
        self.lock = threading.Lock()
        self.snap = self._snapshot()
        th = threading.Thread(target=self._sim_loop, daemon=True)
        th.start()
        self._btn = pygame.Rect(self.W - 300, 4, 128, 18)   # start/stop button
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self._btn.collidepoint(ev.pos):
                        self.paused = not self.paused
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running = False
                    elif ev.key == pygame.K_g and (ev.mod & pygame.KMOD_SHIFT):
                        self.show_gmap = not self.show_gmap
                    elif ev.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_r:
                        self._reset()
            with self.lock:
                snap = self.snap
            self._draw(snap)
            b = self._btn
            pygame.draw.rect(sc := self.screen, (60, 26, 30) if not self.paused else (26, 46, 30), b)
            pygame.draw.rect(sc, (120, 90, 90) if not self.paused else (90, 120, 90), b, 1)
            lbl = "|| STOPPED  (click)" if self.paused else ">  running  (click)"
            sc.blit(self.f7.render(lbl, True, (240, 180, 180) if not self.paused else (180, 240, 180)),
                    (b.x + 6, b.y + 2))
            pygame.display.flip()
            # sim only advances ~20/s, so 30 fps redraw is already smooth; STOPPED
            # idles the render loop right down.
            self.clock.tick(12 if self.paused else 30)
        th.join(timeout=1.0)
        pygame.quit()

    # ------------------------------------------------------------------ draw
    def _draw(self, d):
        sc = self.screen
        sc.fill(BG)
        self.alpha.fill((0, 0, 0, 0))
        px, py = float(d.player[0]), float(d.player[1])
        bpos, bvel, brad, bcol = d.bpos, d.bvel, d.brad, d.bcol
        esc = d.obs[HEAD_DIM:HEAD_DIM + NDIRS]
        grid = d.obs[HEAD_DIM + NDIRS:HEAD_DIM + NDIRS + GCELLS].reshape(GRID, GRID)
        a = d.action

        pygame.draw.rect(sc, (34, 38, 46), (_cx(0), _cy(0), PW * SCALE, PH * SCALE), 1)
        pygame.draw.rect(sc, (51, 58, 68),
                         (_cx(PX_LO), _cy(PY_LO), (PX_HI - PX_LO) * SCALE, (PY_HI - PY_LO) * SCALE), 1)

        # emitter origins
        e_pos, e_on, e_type, e_mt = d.e_pos, d.e_on, d.e_type, d.e_mt
        for e in range(len(e_on)):
            if not e_on[e]:
                continue
            ex, ey = float(e_pos[e, 0]), float(e_pos[e, 1])
            ec = _SRC_COL.get(_ET.get(int(round(e_type[e])), "cone"), (79, 163, 255))
            X, Y, r = _cx(ex), _cy(ey), 6
            pygame.draw.line(sc, (43, 48, 56), (X, Y), (_cx(192), _cy(224)), 1)
            pygame.draw.polygon(sc, ec, [(X, Y - r), (X + r, Y), (X, Y + r), (X - r, Y)])
            pygame.draw.polygon(sc, BG, [(X, Y - r), (X + r, Y), (X, Y + r), (X - r, Y)], 1)
            tag = _MT.get(int(round(e_mt[e])), "")
            lab = f"{_ET.get(int(round(e_type[e])), '?')[:2]}{('·' + tag) if tag else ''}"
            sc.blit(self.fsmall.render(lab, True, ec), (X - 10, Y - r - 12))

        # macro tracks
        if self.show_gmap and len(bpos):
            for (bx, by), (vx, vy) in zip(bpos, bvel):
                if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                    continue
                pygame.draw.line(sc, (90, 48, 48), (_cx(bx), _cy(by)),
                                 (_cx(bx + vx * 24.0), _cy(by + vy * 24.0)), 1)

        # danger grid heatmap (alpha layer)
        if self.show_grid:
            R = GRID // 2
            gc = GRID_CELL * SCALE
            for j in range(GRID):
                for i in range(GRID):
                    gv = float(grid[j, i])
                    if gv <= 0.03:
                        continue
                    x0 = _cx(px + (i - R - 0.5) * GRID_CELL)
                    y0 = _cy(py + (j - R - 0.5) * GRID_CELL)
                    pygame.draw.rect(self.alpha, heat(gv) + (120,), (x0, y0, gc, gc))

        box = (GRID // 2 + 0.5) * GRID_CELL
        dashed(sc, (63, 167, 255), (_cx(px - box), _cy(py - box)),
               (_cx(px + box), _cy(py - box)), width=1)
        dashed(sc, (63, 167, 255), (_cx(px - box), _cy(py + box)),
               (_cx(px + box), _cy(py + box)), width=1)
        dashed(sc, (63, 167, 255), (_cx(px - box), _cy(py - box)),
               (_cx(px - box), _cy(py + box)), width=1)
        dashed(sc, (63, 167, 255), (_cx(px + box), _cy(py - box)),
               (_cx(px + box), _cy(py + box)), width=1)

        # bullets at true hitbox size; inside the danger box -> red + ring
        if len(bpos):
            near = (np.abs(bpos[:, 0] - px) < box) & (np.abs(bpos[:, 1] - py) < box)
            onf = ((bpos[:, 0] > -20) & (bpos[:, 0] < PW + 20) &
                   (bpos[:, 1] > -20) & (bpos[:, 1] < PH + 20))
            for k in np.where(~near & onf)[0]:
                bx, by = bpos[k]
                pygame.draw.circle(sc, tuple(int(c) for c in bcol[k]),
                                   (_cx(bx), _cy(by)), float(max(1.5, brad[k] * SCALE)))
            for k in np.where(near)[0]:
                bx, by = bpos[k]
                rr = float(max(1.5, brad[k] * SCALE))
                pygame.draw.circle(sc, (255, 77, 77), (_cx(bx), _cy(by)), rr)
                pygame.draw.circle(sc, (255, 208, 208), (_cx(bx), _cy(by)), rr, 1)

        # spam spawners + banner
        sph = d.spam_phase
        if sph > 0.5:
            n = d.spam_n
            spx, spy = d.sp_x, d.sp_y
            for k in range(n):
                pygame.draw.circle(sc, (255, 156, 51), (_cx(spx[k]), _cy(spy[k])), 7, 2)
            lab = f"SPAM  fire  {n} spawners" if sph < 1.5 else "SPAM  cooldown"
            sc.blit(self.f8.render(lab, True, (255, 156, 51)), (_cx(PW / 2) - 60, _cy(2)))

        # front-only shot column
        ALIGN = SHOOT_ALIGN_DX
        shooting = (a // 18) % 2
        fc = (107, 99, 32, 90) if shooting else (51, 50, 28, 70)
        pygame.draw.rect(self.alpha, fc, (_cx(px - ALIGN), _cy(0), 2 * ALIGN * SCALE, py * SCALE))
        edge = (255, 225, 77) if shooting else (90, 85, 51)
        dashed(sc, edge, (_cx(px - ALIGN), _cy(0)), (_cx(px - ALIGN), _cy(py)), width=1)
        dashed(sc, edge, (_cx(px + ALIGN), _cy(0)), (_cx(px + ALIGN), _cy(py)), width=1)

        # enemies
        en_act, en_pos, en_hp = d.en_act, d.en_pos, d.en_hp
        near_i, near_d = -1, 1e9
        for k in range(len(en_act)):
            if not en_act[k]:
                continue
            ex, ey = en_pos[k]
            ed = ((ex - px) ** 2 + (ey - py) ** 2) ** 0.5
            if abs(ex - px) < ALIGN and ey < py and ed < near_d:
                near_i, near_d = k, ed
            pygame.draw.circle(sc, (213, 77, 224), (_cx(ex), _cy(ey)), 9)
            pygame.draw.circle(sc, (255, 255, 255), (_cx(ex), _cy(ey)), 9, 1)
            hpf = float(max(0.0, min(1.0, en_hp[k] / 2.0)))
            if hpf > 0:
                pygame.draw.rect(sc, (102, 255, 136),
                                 (_cx(ex) - 10, _cy(ey) - 16, 20.0 * hpf, 3.0))
        if near_i >= 0:
            ex, ey = en_pos[near_i]
            col = (255, 225, 77) if shooting else (122, 116, 64)
            pygame.draw.circle(sc, col, (_cx(ex), _cy(ey)), 12, 2)
            if shooting:
                pygame.draw.line(sc, (255, 225, 77), (_cx(px), _cy(py)), (_cx(ex), _cy(ey)), 3)

        # P items + collect radius
        it_act, it_pos = d.it_act, d.it_pos
        for k in range(len(it_act)):
            if not it_act[k]:
                continue
            ix, iy = it_pos[k]
            pygame.draw.rect(sc, (255, 79, 163), (_cx(ix) - 3, _cy(iy) - 3, 6, 6))
            pygame.draw.rect(sc, (255, 208, 230), (_cx(ix) - 3, _cy(iy) - 3, 6, 6), 1)
        dashed(sc, (122, 51, 80), (_cx(px - 14), _cy(py)), (_cx(px + 14), _cy(py)), 2, 3)

        # escape rays + stay ring
        for i in range(1, NDIRS):
            dx, dy = DIRS[i]
            nn = (dx * dx + dy * dy) ** 0.5
            ln = 12 + esc[i] * 62
            pygame.draw.line(sc, safe_col(esc[i]), (_cx(px), _cy(py)),
                             (_cx(px + dx / nn * ln), _cy(py + dy / nn * ln)), 3)
        pygame.draw.circle(sc, safe_col(esc[0]), (_cx(px), _cy(py)), 9, 2)

        # player at true hitbox size
        pygame.draw.circle(sc, (255, 255, 255), (_cx(px), _cy(py)), float(max(2.0, PLAYER_HB * SCALE)))

        # action arrow / focus / shoot
        di, focus = a % 9, (a // 9) % 2
        dx, dy = DIRS[di]
        if dx or dy:
            nn = (dx * dx + dy * dy) ** 0.5
            spd = 1.6 if focus else 4.0
            tip = (_cx(px + dx / nn * spd * 10), _cy(py + dy / nn * spd * 10))
            pygame.draw.line(sc, (0, 229, 255), (_cx(px), _cy(py)), tip, 4)
        if focus:
            pygame.draw.circle(sc, (102, 255, 204), (_cx(px), _cy(py)), 14, 1)
        if (a // 18) % 2:
            sc.blit(self.fsmall.render("SHOOT", True, (255, 225, 77)),
                    (_cx(px) - 16, _cy(py) + 14))

        sc.blit(self.alpha, (0, 0))

        # status line
        st = (f"frame {d.frame:5d}  ~{d.frame / 60:5.1f}s   "
              f"diff {d.diff:.2f}   bullets {len(bpos):3d}   "
              f"{int(self.clock.get_fps())} fps  {'PAUSED' if self.paused else ''}")
        sc.blit(self.f8.render(st, True, (200, 204, 212)), (_cx(0), 8))

        # stats panel
        lines = self._stats_lines()
        x = int(_cx(PW) + 8)
        pygame.draw.rect(sc, (18, 21, 27), (x - 4, 24, 264, 16 * len(lines) + 10))
        pygame.draw.rect(sc, (42, 47, 58), (x - 4, 24, 264, 16 * len(lines) + 10), 1)
        for i, ln in enumerate(lines):
            c = (159, 208, 255) if i == 0 else (200, 204, 212)
            sc.blit(self.f7.render(ln, True, c), (x, 30 + i * 16))

        # legend
        ly = 24 + 16 * len(lines) + 26
        for name, key in (("cone", "cone"), ("spray", "spray"), ("line", "line"),
                          ("bounce/orbit", "bring"), ("spam", "spam"), ("enemy", "enemy"),
                          ("in danger box", None)):
            col = (255, 77, 77) if key is None else _SRC_COL[key]
            pygame.draw.circle(sc, col, (x + 7, ly + 8), 5)
            sc.blit(self.f7.render(name, True, (200, 204, 212)), (x + 20, ly + 2))
            ly += 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="runs_sim/ppo_v1/best.pt")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="run the B=1 sim on cuda (frees a CPU core)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Watcher(args).run()


if __name__ == "__main__":
    main()
