"""Smoke-test Th07Env with a random policy.

    python native\\test_env.py

Navigate into a stage. Runs a few episodes with random actions, printing
length / return / score, and the per-step rate.
"""
from __future__ import annotations

import time

import numpy as np

from env import Th07Env


def main() -> None:
    env = Th07Env(frame_skip=3, max_seconds=40)
    print(f"obs {env.observation_space.shape}  actions {env.action_space.n}")

    for ep in range(4):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all(), "non-finite obs"
        ret, t0 = 0.0, time.perf_counter()
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            ret += r
            steps += 1
            if term or trunc:
                break
        dt = time.perf_counter() - t0
        print(f"ep {ep}: {steps:4d} steps  return {ret:7.2f}  score {info['score']:>8}  "
              f"{steps / dt:.0f} steps/s ({steps * env.frame_skip / dt / 60:.1f}x)  "
              f"{'CLEAR' if info['tick_status'] else 'died/timeout'}")

    env.close()


if __name__ == "__main__":
    main()
