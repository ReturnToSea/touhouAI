"""Status HUD for the island-evolution run (tkinter only).

Two windows:
  * summary  - wall/sim time, speed, generation, best-ever/this-gen, mean, ...
  * population - the current generation's individuals, filling in live; double-
    click a row to launch watch.py on that individual.

All tkinter access is crash-guarded: if a window is closed or errors, the HUD
goes dark and never disturbs training. After training, call finish() to keep
the windows open (enters mainloop).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time


def _hms(sec: float) -> str:
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"


_SUMMARY = ["wall", "sim time", "speed", "generation", "sec/gen",
            "evaluated", "best ever", "best this gen", "mean so far",
            "best score", "best survival"]


class EvoHud:
    def __init__(self, total=0, total_gens=0, hidden=(64, 64), run_dir="."):
        self.total = total
        self.total_gens = total_gens
        self.hidden = tuple(hidden)
        self.run_dir = str(run_dir)
        self._t0 = time.time()
        self._gen_t0 = self._t0
        self._sim_frames = 0
        self._best_ever = -1e18
        self._best_score = 0
        self._pop = None            # [island][idx] -> flat weights, this gen
        self._gen = 0
        self._gbest = -1e18
        self._n_eval = 0
        self._sum_fit = 0.0
        self._root = None
        self._rows = {}
        self._tree = None
        self._items = {}            # (island, idx) -> tree item id
        self._build()

    # ---- window plumbing -------------------------------------------------
    def _dead(self):
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        self._root = None
        self._rows = {}
        self._tree = None

    def _build(self):
        try:
            import tkinter as tk
            from tkinter import ttk
            root = tk.Tk()
        except Exception:
            self._root = None
            return
        try:
            root.title("Th07 evolution")
            root.attributes("-topmost", True)
            root.configure(bg="#12141a", padx=16, pady=12)
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self._dead)
            root.grid_columnconfigure(1, minsize=200)
            lf, vf = ("Consolas", 11), ("Consolas", 13, "bold")
            for i, name in enumerate(_SUMMARY):
                tk.Label(root, text=name, anchor="w", width=14, fg="#8a93a3",
                         bg="#12141a", font=lf).grid(row=i, column=0, sticky="w", pady=2)
                v = tk.Label(root, text="-", anchor="e", fg="#e6e9ef",
                             bg="#12141a", font=vf)
                v.grid(row=i, column=1, sticky="e", padx=(16, 0), pady=2)
                self._rows[name] = v

            pop = tk.Toplevel(root)
            pop.title("population - double-click to watch")
            pop.configure(bg="#12141a")
            pop.geometry("460x560")
            cols = ("rank", "island", "fitness", "survival", "score")
            tv = ttk.Treeview(pop, columns=cols, show="headings", height=26)
            for c, w in zip(cols, (48, 54, 90, 90, 100)):
                tv.heading(c, text=c)
                tv.column(c, width=w, anchor="e")
            tv.pack(fill="both", expand=True)
            tv.bind("<Double-1>", self._on_click)
            self._tree = tv
            self._pop_win = pop
            root.update_idletasks()
            root.update()
            self._root = root
        except Exception:
            self._dead()

    def _set(self, name, txt):
        w = self._rows.get(name)
        if w is not None:
            w.config(text=txt)

    def pump(self):
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:
            self._dead()

    # ---- click -> watch -------------------------------------------------
    def _on_click(self, _evt):
        if self._tree is None or self._pop is None:
            return
        try:
            item = self._tree.focus()
            isl, idx = self._tree.item(item, "tags")[:2]
            flat = self._pop[int(isl)][int(idx)]
            from policy import MLPPolicy
            p = MLPPolicy(hidden=self.hidden)
            p.set_flat(flat)
            path = os.path.join(tempfile.gettempdir(),
                                f"th07_watch_i{isl}_p{idx}.pt")
            p.save(path)
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # keep it short: a watched game RENDERS (no headless stubs) and
            # oversubscribes the CPU, dragging the whole run's speed down while
            # it's alive. 3 episodes is enough of a look.
            subprocess.Popen([sys.executable, os.path.join(here, "watch.py"),
                              path, "--evo", "--episodes", "3", "--viz"],
                             cwd=here)
        except Exception:
            pass

    # ---- per-generation lifecycle -------------------------------------
    def gen_start(self, gen, pop):
        """pop: [island][idx] -> flat weights for the generation about to run."""
        self._gen = gen
        self._pop = pop
        self._gen_t0 = time.time()
        self._gbest = -1e18
        self._n_eval = 0
        self._sum_fit = 0.0
        if self._tree is None:
            return
        try:
            self._tree.delete(*self._tree.get_children())
            self._items.clear()
        except Exception:
            self._dead()

    def record(self, island, idx, fitness, frames, score):
        self._n_eval += 1
        self._sum_fit += fitness
        self._gbest = max(self._gbest, fitness)
        if fitness > self._best_ever:
            self._best_ever = fitness
        self._best_score = max(self._best_score, int(score))
        self._sim_frames += int(frames)
        if self._tree is None:
            return
        try:
            self._tree.insert(
                "", "end", tags=(str(island), str(idx)),
                values=("", island, f"{fitness:.1f}",
                        f"{frames/60:.1f}s", f"{int(score):,}"))
            # keep it roughly sorted by fitness, cheaply: re-rank every ~8 rows
            if self._n_eval % 8 == 0 or self._n_eval == self.total:
                self._rerank()
            self._paint_summary()
        except Exception:
            self._dead()

    def _rerank(self):
        try:
            rows = [(float(self._tree.set(i, "fitness")), i)
                    for i in self._tree.get_children()]
            rows.sort(reverse=True)
            for rank, (_, i) in enumerate(rows, 1):
                self._tree.move(i, "", rank - 1)
                self._tree.set(i, "rank", rank)
        except Exception:
            self._dead()

    def _paint_summary(self):
        wall = time.time() - self._t0
        sim = self._sim_frames / 60.0
        mean = self._sum_fit / max(1, self._n_eval)
        vals = {
            "wall": _hms(wall),
            "sim time": _hms(sim),
            "speed": f"{sim/wall:,.0f}x" if wall > 0 else "-",
            "generation": f"{self._gen:,}" + (f" / {self.total_gens:,}"
                                              if self.total_gens else ""),
            "sec/gen": f"{time.time() - self._gen_t0:.0f}",
            "evaluated": f"{self._n_eval} / {self.total}",
            "best ever": f"{self._best_ever:.1f}",
            "best this gen": f"{self._gbest:.1f}",
            "mean so far": f"{mean:.1f}",
            "best score": f"{self._best_score:,}",
            "best survival": "-",
        }
        for k, v in vals.items():
            self._set(k, v)

    def gen_end(self, best_survival_frames=0):
        if self._root is None:
            return
        try:
            self._rerank()
            self._set("best survival",
                      f"{best_survival_frames/60:.1f}s ({int(best_survival_frames)}f)")
            self._paint_summary()
            self._root.update()
        except Exception:
            self._dead()

    def finish(self):
        if self._root is None:
            return
        try:
            self._root.title("Th07 evolution - done (close to exit)")
            self._root.attributes("-topmost", False)
            self._root.mainloop()
        except Exception:
            pass

    def close(self):
        self._dead()
