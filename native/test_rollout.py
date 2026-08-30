"""Smoke-test ST_ROLLOUT: one game, one DLL-collected trajectory. Checks the
buffers are sane and the reward matches env.py's formula.

    .venv/Scripts/python native/test_rollout.py [n_envs]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from real_rollout import RealRolloutVec  # noqa: E402
from policy import MLPPolicy             # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
T = 256


def main():
    pol = MLPPolicy.load(HERE.parent / "runs_sim" / "ppo_v29" / "snap_0092M.pt")
    w = np.concatenate([p.detach().numpy().reshape(-1) for p in pol.net.parameters()]).astype(np.float32)
    print(f"flat actor weights: {w.size}")

    vec = RealRolloutVec(n_envs=N, frame_skip=3, max_ep_frames=180 * 60)
    for it in range(3):
        t0 = time.perf_counter()
        obs, act, rew, done, last, ep_ends = vec.collect(w, T, 256, 256)
        dt = time.perf_counter() - t0
        frames = T * N * 3
        print(f"\nrollout {it}: {dt:.1f}s  {frames/dt/60:.0f}x realtime aggregate "
              f"({frames/dt/60/N:.0f}x/env)")
        print(f"  obs {obs.shape}  act {act.shape}  range [{act.min()},{act.max()}]")
        print(f"  reward: mean {rew.mean():.3f}  min {rew.min():.2f}  max {rew.max():.2f}  "
              f"deaths(-5ish) {(rew < -3).sum()}")
        print(f"  done flags: {int(done.sum())}   ep_ends: {ep_ends}")
        print(f"  obs sane: min {obs.min():.2f} max {obs.max():.2f}  "
              f"nan {np.isnan(obs).any()}  last_obs {last.shape}")
        # per-step reward should be ~0.06 baseline + score/1e-4 + boss + death
        base = (np.abs(rew - 0.06) < 0.001).mean()
        print(f"  frac steps at bare-alive 0.06: {base*100:.0f}%")
    vec.close()
    print("\nOK" if not np.isnan(obs).any() else "\nNAN IN OBS")


if __name__ == "__main__":
    main()
