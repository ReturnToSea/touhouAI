"""Watch a trained policy (or a random one) play, in real time, with the game
window visible + sound on.

    .venv\\Scripts\\python watch.py runs\\ppo_st1\\final.zip
    .venv\\Scripts\\python watch.py runs\\evo_st1\\best.pt --evo
    .venv\\Scripts\\python watch.py --random          # no model, just look at the env

The game runs headless+muted during training; here it renders and presents so
you can actually see it. Playback is paced to ~60 Hz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))

import numpy as np  # noqa: E402

from env import Th07Env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", type=Path,
                    help="PPO .zip checkpoint, or an evo .pt with --evo")
    ap.add_argument("--evo", action="store_true", help="model is an evo policy .pt")
    ap.add_argument("--random", action="store_true", help="ignore model, act randomly")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of taking the argmax")
    args = ap.parse_args()

    act_fn = None
    if not args.random:
        if not args.model or not args.model.exists():
            ap.error("give a model path, or pass --random")
        if args.evo:
            from policy import MLPPolicy
            pol = MLPPolicy.load(args.model)
            act_fn = lambda obs: pol.act(obs)  # noqa: E731
        else:
            from stable_baselines3 import PPO
            ppo = PPO.load(args.model, device="cpu")
            act_fn = lambda obs: int(  # noqa: E731
                ppo.predict(obs, deterministic=not args.stochastic)[0])
        print(f"loaded {args.model}")

    env = Th07Env(frame_skip=args.frame_skip, max_seconds=600, render=True)
    dt = args.frame_skip / 60.0

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = ret = 0.0
            steps = 0
            while not done:
                t0 = time.perf_counter()
                action = env.action_space.sample() if act_fn is None else act_fn(obs)
                obs, r, term, trunc, info = env.step(int(action))
                ret += r
                steps += 1
                done = term or trunc
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            print(f"ep {ep}: {steps} steps  return {ret:6.1f}  "
                  f"score {info['score']}  {'died' if term else 'timeout'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
