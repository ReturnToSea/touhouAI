"""Build a SubprocVecEnv of Th07Env - N parallel game processes."""
from __future__ import annotations

import os
import sys
import time

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

_NATIVE = os.path.dirname(os.path.abspath(__file__))


def _make(rank: int, frame_skip: int, max_seconds: float, stagger: float):
    def thunk():
        if _NATIVE not in sys.path:
            sys.path.insert(0, _NATIVE)
        # small offset so N workers don't hit the build lock at the same instant;
        # Th07Env's cross-process _BuildLock does the real serialisation.
        time.sleep(rank * stagger)
        from env import Th07Env
        return Th07Env(frame_skip=frame_skip, max_seconds=max_seconds)
    return thunk


def make_vec(n_envs: int = 8, frame_skip: int = 3, max_seconds: float = 60.0,
             stagger: float = 0.3) -> VecMonitor:
    venv = SubprocVecEnv(
        [_make(i, frame_skip, max_seconds, stagger) for i in range(n_envs)],
        start_method="spawn",
    )
    return VecMonitor(venv)
