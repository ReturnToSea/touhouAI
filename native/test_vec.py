"""Vectorised throughput smoke test - N parallel game processes, random actions.

    python native\\test_vec.py --n 4 --steps 400
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from vec import make_vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    t0 = time.perf_counter()
    venv = make_vec(n_envs=args.n, frame_skip=3, max_seconds=30)
    print(f"{args.n} envs up in {time.perf_counter() - t0:.1f}s")

    venv.reset()
    t0 = time.perf_counter()
    ep_returns = []
    for i in range(args.steps):
        acts = np.random.randint(0, 36, size=args.n)
        obs, rews, dones, infos = venv.step(acts)
        for info in infos:
            if "episode" in info:
                ep_returns.append(info["episode"]["r"])
    dt = time.perf_counter() - t0
    total = args.steps * args.n
    print(f"{args.steps} vec-steps, {total} transitions, {dt:.1f}s")
    print(f"  {total / dt:.0f} transitions/s  = {total * 3 / dt / 60:.0f}x real-time aggregate")
    if ep_returns:
        print(f"  {len(ep_returns)} episodes, mean return {np.mean(ep_returns):.1f}")
    venv.close()


if __name__ == "__main__":
    main()
