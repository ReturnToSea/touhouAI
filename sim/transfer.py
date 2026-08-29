"""Phase 3: drop a sim-trained policy into the REAL game and measure how it
does on Lunatic stage 1.

    .venv\\Scripts\\python sim\\transfer.py runs_sim\\ppo_v1\\best.pt --episodes 20
    .venv\\Scripts\\python sim\\transfer.py runs_sim\\ppo_v1\\best.pt --watch     # render + viz

Uses the Python step loop (env._obs -> the shared native/obs.py builder), so the
observation matches exactly what the policy trained on. The DLL's in-C build_obs
is NOT used here (it still has stale constants until it can be rebuilt).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "native"))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--watch", action="store_true", help="render + launch viz.py")
    ap.add_argument("--max-seconds", type=float, default=180)
    args = ap.parse_args()

    pol = MLPPolicy.load(args.model)
    print(f"loaded {args.model}  hidden={pol.hidden}  params={pol.n_params()}")

    env = Th07Env(frame_skip=args.frame_skip, max_seconds=args.max_seconds,
                  render=args.watch)
    if args.watch:
        import subprocess
        subprocess.Popen([sys.executable, str(HERE.parent / "native" / "viz.py"),
                          str(env.pid)])
    dt = args.frame_skip / 60.0

    survived, scores, actions_hist = [], [], np.zeros(36, dtype=np.int64)
    move_frac = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            steps = moves = 0
            done = False
            while not done:
                t0 = time.perf_counter()
                a = int(pol.act(obs))
                actions_hist[a] += 1
                if a % 9 != 0:          # non-neutral direction
                    moves += 1
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                done = term or trunc
                if args.watch:
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            frames = steps * args.frame_skip
            survived.append(frames)
            scores.append(info["score"])
            move_frac.append(moves / max(steps, 1))
            print(f"ep {ep:2d}: {frames:5d} frames  {frames/60:5.1f}s  "
                  f"score {info['score']:7d}  moved {moves/max(steps,1)*100:3.0f}% of decisions  "
                  f"{'died' if term else 'timeout'}")
    finally:
        env.close()

    s = np.array(survived) / 60.0
    print(f"\n=== {args.model.parent.name} on real Lunatic stage 1 "
          f"({args.episodes} eps) ===")
    print(f"survival:  mean {s.mean():.1f}s   median {np.median(s):.1f}s   "
          f"min {s.min():.1f}s   max {s.max():.1f}s")
    print(f"score:     mean {np.mean(scores):.0f}   max {max(scores)}")
    print(f"movement:  {np.mean(move_frac)*100:.0f}% of decisions were a move "
          f"(low % = it mostly sits still)")
    top = actions_hist.argsort()[::-1][:6]
    print("top actions (idx: dir/focus/shoot, count):")
    for a in top:
        print(f"  {a:2d}: dir{a%9} focus{(a//9)%2} shoot{(a//18)%2}  x{actions_hist[a]}")


if __name__ == "__main__":
    main()
