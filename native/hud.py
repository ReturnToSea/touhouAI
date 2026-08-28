"""A tiny always-on-top status window for a PPO run. No extra deps (tkinter).

    from hud import TrainHud
    model.learn(..., callback=[ckpt, TrainHud(frame_skip=3, total_steps=5_000_000)])

PPO is not generational - it's one policy improved by a gradient update every
rollout. "iters" = collect-then-update cycles; "steps" = agent steps summed
over all envs; "sim" = in-game time those steps represent.
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
        self.best_score = 0
        self.best_return = float("-inf")
        self._root = None
        self._rows = {}

    # ---- window -------------------------------------------------------------
    def _build(self):
        try:
            import tkinter as tk
        except Exception:
            return
        try:
            self._root = tk.Tk()
        except Exception:
            self._root = None
            return
        r = self._root
        r.title("Th07 PPO")
        r.attributes("-topmost", True)
        r.configure(bg="#12141a")
        r.resizable(False, False)
        fields = ["wall", "sim", "steps", "iters", "episodes",
                  "best score", "best return", "recent return"]
        for i, name in enumerate(fields):
            tk.Label(r, text=name, anchor="w", width=12, fg="#8a93a3",
                     bg="#12141a", font=("Consolas", 10)).grid(
                         row=i, column=0, sticky="w", padx=(12, 6), pady=2)
            val = tk.Label(r, text="-", anchor="e", width=16, fg="#e6e9ef",
                           bg="#12141a", font=("Consolas", 11, "bold"))
            val.grid(row=i, column=1, sticky="e", padx=(6, 12), pady=2)
            self._rows[name] = val
        r.update()

    def _set(self, name, text):
        w = self._rows.get(name)
        if w is not None:
            w.config(text=text)

    def _paint(self):
        if self._root is None:
            return
        wall = time.time() - self._t0
        frames = self.num_timesteps * self.frame_skip
        sim = frames / 60.0
        speed = sim / wall if wall > 0 else 0.0
        ep_buf = self.model.ep_info_buffer
        recent = (sum(e["r"] for e in ep_buf) / len(ep_buf)) if ep_buf else 0.0

        self._set("wall", _hms(wall))
        self._set("sim", f"{_hms(sim)} {speed:.0f}x")
        tot = f" / {self.total_steps:,}" if self.total_steps else ""
        self._set("steps", f"{self.num_timesteps:,}{tot}")
        self._set("iters", f"{self._iters:,}")
        self._set("episodes", f"{self._ep_total:,}")
        self._set("best score", f"{self.best_score:,}")
        self._set("best return",
                  f"{self.best_return:.1f}" if self.best_return > -1e17 else "-")
        self._set("recent return", f"{recent:.1f}")
        try:
            self._root.update()
        except Exception:
            self._root = None  # window closed - stop touching it

    # ---- callback hooks ---------------------------------------------------
    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._ep_total = 0
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
            self._paint()
        return True

    def _on_training_end(self) -> None:
        self._paint()
        if self._root is not None:
            try:
                self._rows["wall"].master.title("Th07 PPO - done")
                self._root.update()
            except Exception:
                pass
