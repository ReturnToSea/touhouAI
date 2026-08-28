"""Tiny always-on-top status window for the evolution run. tkinter only.

    from evohud import EvoHud
    hud = EvoHud(pop=128, total_gens=5000)
    ...                       # call hud.pump() often so the window stays live
    hud.update(gen, best_all, best_gen, mean, median, sim_frames, best_score,
               best_frames)

Best-effort: if the window is closed or tkinter errors, it goes dark and never
disturbs training.
"""
from __future__ import annotations

import time


def _hms(sec: float) -> str:
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"


_FIELDS = ["wall", "sim time", "speed", "generation", "sec/gen",
           "best ever", "best this gen", "mean",
           "best score", "best survival"]


class EvoHud:
    def __init__(self, pop: int = 0, total_gens: int = 0):
        self.pop = pop
        self.total_gens = total_gens
        self._t0 = time.time()
        self._last_gen_t = self._t0
        self._root = None
        self._rows = {}
        self._build()

    def _kill(self):
        try:
            if self._root is not None:
                self._root.destroy()
        except Exception:
            pass
        self._root = None
        self._rows = {}

    def _build(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception:
            self._root = None
            return
        try:
            title = "Th07 evolution" + (f"  (pop {self.pop})" if self.pop else "")
            root.title(title)
            root.attributes("-topmost", True)
            root.configure(bg="#12141a", padx=18, pady=14)
            root.resizable(False, False)
            root.minsize(400, 320)
            root.protocol("WM_DELETE_WINDOW", self._kill)
            root.grid_columnconfigure(1, minsize=220)
            lf = ("Consolas", 11)
            vf = ("Consolas", 14, "bold")
            for i, name in enumerate(_FIELDS):
                tk.Label(root, text=name, anchor="w", width=14, fg="#8a93a3",
                         bg="#12141a", font=lf).grid(row=i, column=0, sticky="w",
                                                     pady=3)
                v = tk.Label(root, text="-", anchor="e", fg="#e6e9ef",
                             bg="#12141a", font=vf)
                v.grid(row=i, column=1, sticky="e", padx=(20, 0), pady=3)
                self._rows[name] = v
            root.update_idletasks()
            root.update()
            self._root = root
        except Exception:
            self._kill()

    def pump(self):
        """Process pending window events - call frequently so it stays live."""
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:
            self._kill()

    def update(self, gen, best_all, best_gen, mean, median,
               sim_frames=0, best_score=0, best_frames=0):
        if self._root is None:
            return
        now = time.time()
        wall = now - self._t0
        spg = now - self._last_gen_t
        self._last_gen_t = now
        sim = sim_frames / 60.0
        g = f"{gen:,}" + (f" / {self.total_gens:,}" if self.total_gens else "")
        vals = {
            "wall": _hms(wall),
            "sim time": _hms(sim),
            "speed": f"{sim / wall:,.0f}x" if wall > 0 else "-",
            "generation": g,
            "sec/gen": f"{spg:.0f}",
            "best ever": f"{best_all:.1f}",
            "best this gen": f"{best_gen:.1f}",
            "mean": f"{mean:.1f}",
            "best score": f"{int(best_score):,}",
            "best survival": f"{best_frames/60:.1f}s ({int(best_frames)}f)",
        }
        try:
            for k, txt in vals.items():
                self._rows[k].config(text=txt)
            self._root.update()
        except Exception:
            self._kill()

    def close(self):
        self._kill()
