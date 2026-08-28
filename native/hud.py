"""A small always-on-top status window for a PPO run. No extra deps (tkinter).

    from hud import TrainHud
    model.learn(..., callback=[ckpt, TrainHud(frame_skip=3, total_steps=5_000_000)])

PPO is not generational - it's one policy improved by a gradient update every
rollout. "iters" = collect-then-update cycles; "steps" = agent steps summed
over all envs; "sim" = in-game time those steps represent.

The HUD is best-effort: if the window is closed or tkinter errors for any
reason, it silently stops updating - it must never disturb training.
"""
from __future__ import annotations

import time

from stable_baselines3.common.callbacks import BaseCallback


def _hms(sec: float) -> str:
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{sec:02d}"


_FIELDS = ["wall", "sim", "speed", "steps", "iters",
           "episodes", "best score", "best return", "recent return"]


class TrainHud(BaseCallback):
    def __init__(self, frame_skip: int = 3, total_steps: int = 0,
                 refresh: float = 0.25, verbose: int = 0):
        super().__init__(verbose)
        self.frame_skip = frame_skip
        self.total_steps = total_steps
        self.refresh = refresh
        self._t0 = 0.0
        self._last = 0.0
        self._iters = 0
        self._ep_total = 0
        self.best_score = 0
        self.best_return = float("-inf")
        self._root = None
        self._rows = {}

    # ---- window (all tkinter access funnels through _safe) ---------------
    def _kill_window(self):
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
            root.title("Th07 PPO")
            root.attributes("-topmost", True)
            root.configure(bg="#12141a", padx=18, pady=14)
            root.resizable(False, False)
            root.minsize(420, 300)
            root.protocol("WM_DELETE_WINDOW", self._kill_window)
            lab_font = ("Consolas", 11)
            val_font = ("Consolas", 14, "bold")
            # no fixed width on the value label - a tk width= HARD-clips longer
            # text (that was the bug); let it size to content, hold a column
            # minsize so it doesn't jitter.
            root.grid_columnconfigure(1, minsize=260)
            for i, name in enumerate(_FIELDS):
                tk.Label(root, text=name, anchor="w", width=13, fg="#8a93a3",
                         bg="#12141a", font=lab_font).grid(
                             row=i, column=0, sticky="w", pady=3)
                val = tk.Label(root, text="-", anchor="e", fg="#e6e9ef",
                               bg="#12141a", font=val_font)
                val.grid(row=i, column=1, sticky="e", padx=(20, 0), pady=3)
                self._rows[name] = val
            root.update_idletasks()
            root.update()
            self._root = root
        except Exception:
            self._kill_window()

    def _paint(self):
        if self._root is None:
            return
        wall = time.time() - self._t0
        frames = self.num_timesteps * self.frame_skip
        sim = frames / 60.0
        speed = sim / wall if wall > 0 else 0.0
        ep_buf = getattr(self.model, "ep_info_buffer", None)
        recent = (sum(e["r"] for e in ep_buf) / len(ep_buf)) if ep_buf else 0.0
        tot = f" / {self.total_steps:,}" if self.total_steps else ""
        vals = {
            "wall": _hms(wall),
            "sim": _hms(sim),
            "speed": f"{speed:,.0f}x",
            "steps": f"{self.num_timesteps:,}{tot}",
            "iters": f"{self._iters:,}",
            "episodes": f"{self._ep_total:,}",
            "best score": f"{self.best_score:,}",
            "best return": (f"{self.best_return:.1f}"
                            if self.best_return > -1e17 else "-"),
            "recent return": f"{recent:.1f}",
        }
        try:
            for name, text in vals.items():
                self._rows[name].config(text=text)
            self._root.update()
        except Exception:
            # window closed / destroyed / tkinter unhappy - let it go
            self._kill_window()

    # ---- callback hooks -------------------------------------------------
    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._build()

    def _on_rollout_end(self) -> None:
        self._iters += 1

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            sc = info.get("score")
            if sc is not None and sc > self.best_score:
                self.best_score = sc
            ep = info.get("episode")
            if ep is not None:
                self._ep_total += 1
                if ep["r"] > self.best_return:
                    self.best_return = ep["r"]
        now = time.time()
        if now - self._last >= self.refresh:
            self._last = now
            try:
                self._paint()
            except Exception:
                self._kill_window()
        return True

    def _on_training_end(self) -> None:
        try:
            self._paint()
            if self._root is not None:
                self._root.title("Th07 PPO - done")
                self._root.update()
        except Exception:
            self._kill_window()
