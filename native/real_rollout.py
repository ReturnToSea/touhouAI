"""N parallel real th07 instances, each running ST_ROLLOUT in the DLL (whole
trajectory, no per-step Python). The training loop ships actor weights, calls
collect(), gets back stacked [T, N] buffers.

    from real_rollout import RealRolloutVec
    vec = RealRolloutVec(n_envs=12, frame_skip=3)
    obs, act, rew, done, last_obs, ep_ends = vec.collect(flat_weights, T=256, h1=256, h2=256)
"""
from __future__ import annotations

import time

import numpy as np

from env import Th07Env


class RealRolloutVec:
    def __init__(self, n_envs: int = 12, frame_skip: int = 3,
                 max_ep_frames: int = 10800):
        self.n = n_envs
        self.frame_skip = frame_skip
        self.max_ep_frames = max_ep_frames
        self._seed = 1
        self.envs = []
        for _ in range(n_envs):
            # Th07Env handles the staggered launch + _BuildLock + autonav +
            # snapshot; we then drive it via .h (the Hook) only.
            self.envs.append(Th07Env(frame_skip=frame_skip, max_seconds=999,
                                     hard_reset=True))
        print(f"[RealRolloutVec] {n_envs} instances up", flush=True)

    def collect(self, flat_weights, T: int, h1: int, h2: int, timeout: float = 300.0):
        for e in self.envs:
            self._seed += 1
            e.h.rollout_start(flat_weights, T, h1, h2, self.frame_skip,
                              self._seed, self.max_ep_frames)
        deadline = time.perf_counter() + timeout
        pending = list(self.envs)
        while pending:
            pending = [e for e in pending if not e.h.rollout_done()]
            if not pending:
                break
            if time.perf_counter() > deadline:
                for e in pending:
                    if e.h.s.crash_code:
                        raise RuntimeError(
                            f"game crashed mid-rollout: exc {e.h.s.crash_code:#x} "
                            f"eip {e.h.s.crash_eip:#x}")
                raise RuntimeError(f"{len(pending)}/{self.n} rollouts timed out")
            time.sleep(0.02)

        obs, act, rew, done, last = [], [], [], [], []
        ep_ends = 0
        for e in self.envs:
            o, a, r, d, l, ee = e.h.rollout_result()
            obs.append(o); act.append(a); rew.append(r); done.append(d); last.append(l)
            ep_ends += ee
        return (np.stack(obs, 1), np.stack(act, 1), np.stack(rew, 1),
                np.stack(done, 1), np.stack(last, 0), ep_ends)

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass
