"""Live top-down debug view of what the evolution policy sees.

Attaches to a running hooked th07.exe by pid and reads the shared-memory obs
the DLL already publishes (player, every bullet slot, enemies, boss, the
current action). No game hooking.

    .venv\\Scripts\\python native\\viz.py <pid>
    .venv\\Scripts\\python watch.py runs\\evo_v8\\best.pt --evo --viz

Shows, in playfield space:
  * the playfield + player + hitbox
  * the +-78 px danger-grid window (dashed blue box)
  * every live bullet: red = inside the window now, or its straight-line path
    enters it within the 24-frame horizon (i.e. the net sees it); grey = not
  * the 13x13 danger grid as a translucent heatmap  (toggle: g)
  * a cyan arrow for the direction the policy is currently commanding, a focus
    ring when slow is held

Keys:  g = grid on/off   esc = quit
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shm as S  # noqa: E402

# mirror of th07hook.cpp build_obs / env.py
PW, PH = 384.0, 448.0
GRID, GRID_CELL, GRID_HORIZON = 13, 12.0, 24.0
GRID_R = GRID // 2
BOX = (GRID_R + 0.5) * GRID_CELL          # +-78 px

SCALE = 1.8
PAD = 40
REDRAW_MS = 40                            # ~25 Hz


def _heat(d: float) -> str:
    d = 0.0 if d < 0 else 1.0 if d > 1 else d
    r = int(110 + 145 * d)
    g = int(95 * (1.0 - d))
    b = int(45 * (1.0 - d))
    return f"#{r:02x}{g:02x}{b:02x}"


class Viz:
    def __init__(self, pid: int):
        import tkinter as tk
        self.h = S.Hook(pid, timeout=20)
        self.mm = self.h._mm
        self.bview = np.frombuffer(
            self.mm, np.float32, S.MAX_BULLETS * 4, S.Shm.bullets.offset
        ).reshape(S.MAX_BULLETS, 4)
        self.prev = np.full((S.MAX_BULLETS, 2), np.nan, np.float32)
        self.prev_t = None
        self.show_grid = True
        self.miss = 0

        print(f"viz: attached to pid {pid}", flush=True)
        self.root = tk.Tk()
        self.root.title(f"th07 viz - pid {pid}")
        self.root.configure(bg="#0a0b0e")
        self.root.attributes("-topmost", True)
        self.root.geometry("+900+20")
        w = int(PW * SCALE + 2 * PAD)
        ht = int(PH * SCALE + 2 * PAD)
        self.cv = tk.Canvas(self.root, width=w, height=ht, bg="#0a0b0e",
                            highlightthickness=0)
        self.cv.pack()
        self.root.bind("g", lambda e: setattr(self, "show_grid", not self.show_grid))
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.after(REDRAW_MS, self._tick)
        self.root.mainloop()

    # ------------------------------------------------------------------
    def _tick(self):
        try:
            self._draw()
            self.miss = 0
        except Exception:
            self.miss += 1
            if self.miss > 50:              # game gone / mapping closed
                self.root.destroy()
                return
        self.root.after(REDRAW_MS, self._tick)

    def _cx(self, x):
        return x * SCALE + PAD

    def _cy(self, y):
        return y * SCALE + PAD

    def _bullet_vel(self, live, b):
        """per-frame velocity from the position delta since the last redraw."""
        now = time.perf_counter()
        dt = (now - self.prev_t) if self.prev_t else 1 / 60.0
        self.prev_t = now
        pv = self.prev[live]
        raw = np.where(np.isfinite(pv), b - pv, 0.0)
        v = raw / max(dt * 60.0, 1e-3)
        v[np.abs(v) > 24.0] = 0.0
        self.prev[:] = np.nan
        self.prev[live] = b
        return v

    def _grid(self, px, py, b, v):
        g = np.zeros((GRID, GRID), np.float32)
        ii, jj = np.meshgrid(np.arange(GRID), np.arange(GRID))
        wx = px + (ii - GRID_R) * GRID_CELL
        wy = py + (jj - GRID_R) * GRID_CELL
        g[(wx < 4) | (wx > PW - 4) | (wy < 4) | (wy > PH - 4)] = 0.5
        for it in range(49):
            t = it * 0.5
            cx = np.floor((b[:, 0] + v[:, 0] * t - px) / GRID_CELL + 0.5).astype(int) + GRID_R
            cy = np.floor((b[:, 1] + v[:, 1] * t - py) / GRID_CELL + 0.5).astype(int) + GRID_R
            m = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
            if m.any():
                np.maximum.at(g, (cy[m], cx[m]), 1.0 - t / GRID_HORIZON)
        return g

    def _in_range(self, px, py, b, v):
        m = np.zeros(len(b), bool)
        for it in range(49):
            t = it * 0.5
            x = np.abs(b[:, 0] + v[:, 0] * t - px)
            y = np.abs(b[:, 1] + v[:, 1] * t - py)
            m |= (x <= BOX) & (y <= BOX)
        return m

    # ------------------------------------------------------------------
    def _draw(self):
        s = self.h.s
        cv = self.cv
        cv.delete("all")
        px, py = s.player_x, s.player_y
        cx, cy = self._cx, self._cy

        xy = self.bview[:, :2]
        live = xy[:, 0] > -9000.0
        b = xy[live].astype(np.float32)
        v = self._bullet_vel(live, b)

        cv.create_rectangle(cx(0), cy(0), cx(PW), cy(PH), outline="#2a2f3a")

        # danger grid heatmap
        if self.show_grid:
            g = self._grid(px, py, b, v)
            for j in range(GRID):
                for i in range(GRID):
                    d = float(g[j, i])
                    if d <= 0.02:
                        continue
                    x0 = px + (i - GRID_R - 0.5) * GRID_CELL
                    y0 = py + (j - GRID_R - 0.5) * GRID_CELL
                    cv.create_rectangle(cx(x0), cy(y0), cx(x0 + GRID_CELL),
                                        cy(y0 + GRID_CELL), fill=_heat(d),
                                        outline="", stipple="gray50")

        # detection window
        cv.create_rectangle(cx(px - BOX), cy(py - BOX), cx(px + BOX), cy(py + BOX),
                            outline="#3fa7ff", width=2, dash=(4, 3))

        # bullets
        inr = self._in_range(px, py, b, v) if len(b) else np.zeros(0, bool)
        for (bx, by), ir in zip(b, inr):
            col = "#ff4444" if ir else "#59616f"
            rr = 3.0 if ir else 2.0
            cv.create_oval(cx(bx) - rr, cy(by) - rr, cx(bx) + rr, cy(by) + rr,
                           fill=col, outline="")

        # enemies
        for k in range(min(s.enemy_count, S.MAX_ENEMIES)):
            e = s.enemies[k]
            if e.y < -50:
                continue
            big = e.maxlife >= 200
            cv.create_rectangle(cx(e.x) - 5, cy(e.y) - 5, cx(e.x) + 5, cy(e.y) + 5,
                                outline="#ff9c33" if big else "#ffd23f", width=2)

        # player + focus ring
        cv.create_oval(cx(px) - 3, cy(py) - 3, cx(px) + 3, cy(py) + 3,
                       fill="#ffffff", outline="")
        if s.action & S.SLOW:
            cv.create_oval(cx(px) - 11 * SCALE, cy(py) - 11 * SCALE,
                           cx(px) + 11 * SCALE, cy(py) + 11 * SCALE, outline="#66ffcc")

        # movement arrow
        dx = bool(s.action & S.RIGHT) - bool(s.action & S.LEFT)
        dy = bool(s.action & S.DOWN) - bool(s.action & S.UP)
        if dx or dy:
            spd = 1.9 if s.action & S.SLOW else 3.6
            n = (dx * dx + dy * dy) ** 0.5
            ex, ey = px + dx / n * spd * 9, py + dy / n * spd * 9
            cv.create_line(cx(px), cy(py), cx(ex), cy(ey),
                           fill="#00e5ff", width=3, arrow="last")

        # readout
        nd = float(np.hypot(b[:, 0] - px, b[:, 1] - py).min()) if len(b) else 999.0
        acts = "+".join(name for bit, name in (
            (S.LEFT, "L"), (S.RIGHT, "R"), (S.UP, "U"), (S.DOWN, "D"),
            (S.SLOW, "slow"), (S.SHOOT, "shoot")) if s.action & bit) or "-"
        cv.create_text(cx(0), 16, anchor="w", fill="#c8ccd4", font=("Consolas", 11),
                       text=f"bullets {int(live.sum())}   in-range {int(inr.sum())}"
                            f"   nearest {nd:.0f}px      lives {s.lives:.0f}"
                            f"   score {s.score}   act [{acts}]")


def main():
    if len(sys.argv) < 2:
        print("usage: python native/viz.py <pid>")
        sys.exit(2)
    try:
        Viz(int(sys.argv[1]))
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nviz failed to attach - is that pid a hooked th07.exe?",
              flush=True)
        time.sleep(4)   # keep the console up if double-clicked


if __name__ == "__main__":
    main()
