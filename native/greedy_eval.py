"""Fast greedy eval of one or more MLPPolicy .pt on the real game: one hooked
process, hard-reset between episodes, argmax policy, N episodes each, prints the
survival distribution. Much faster than probe_deathcam (no per-episode relaunch).

    .venv/Scripts/python native/greedy_eval.py --eps 6 A.pt B.pt ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from env import Th07Env       # noqa: E402
from policy import MLPPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--eps", type=int, default=6)
    ap.add_argument("--cap", type=float, default=600.0)
    args = ap.parse_args()

    env = Th07Env(frame_skip=3, max_seconds=args.cap + 5, render=False,
                  hard_reset=True)
    for m in args.models:
        pol = MLPPolicy.load(m)
        survs, scores = [], []
        for ep in range(args.eps):
            obs, _ = env.reset(options={"hard": True})
            steps = 0
            while True:
                obs, r, term, trunc, info = env.step(pol.act(obs))
                steps += 1
                if term or trunc:
                    break
            survs.append(steps * 3 / 60.0)
            scores.append(int(info.get("score", 0)))
        s = np.array(survs)
        print(f"\n{m.name:28s}  n={args.eps}")
        print(f"   survival: med {np.median(s):6.1f}s  mean {s.mean():6.1f}s  "
              f"min {s.min():.0f}  max {s.max():.0f}   [{', '.join(f'{x:.0f}' for x in survs)}]")
        print(f"   score:    med {int(np.median(scores)):>8d}  max {max(scores):>8d}")
    env.close()


if __name__ == "__main__":
    main()
