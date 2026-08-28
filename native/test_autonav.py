"""Test fully-autonomous env creation (no human navigation).

    python native\\test_autonav.py
"""
from __future__ import annotations

import time

from env import Th07Env


def main() -> None:
    t0 = time.perf_counter()
    env = Th07Env(frame_skip=3, max_seconds=30)
    print(f"env ready in {time.perf_counter() - t0:.1f}s")

    for ep in range(3):
        obs, _ = env.reset()
        s = env.h.s
        print(f"reset {ep}: stage {s.stage} diff {s.difficulty} "
              f"pl=({s.player_x:.0f},{s.player_y:.0f}) lives {s.lives:.0f} "
              f"bul {s.bullet_count}")
        ret = 0.0
        while True:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            ret += r
            if term or trunc:
                break
        print(f"   ep {ep}: {info['frame']} frames, return {ret:.1f}, "
              f"score {info['score']}")
    env.close()


if __name__ == "__main__":
    main()
