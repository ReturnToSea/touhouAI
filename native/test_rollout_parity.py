"""Definitive check that ST_ROLLOUT's buffered obs == what ST_STEP produces for
the same actions (i.e. the obs PPO trains on is correct).

  1. hard-reset -> snapshot
  2. ST_ROLLOUT (fixed seed) -> roll_obs[T], roll_act[T], roll_done[T]
  3. restore snapshot, replay roll_act via ST_STEP, compare obs each step
     (up to the first episode end - a death triggers an in-DLL engine reload
     that the snapshot can't reproduce).

    .venv/Scripts/python native/test_rollout_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
import shm as S                       # noqa: E402
from env import Th07Env               # noqa: E402
from policy import MLPPolicy          # noqa: E402
from obs import HEAD_DIM, NDIRS, GCELLS, M_ENEMIES  # noqa: E402

T = 128
_seg = {"head": (0, HEAD_DIM), "escape": (HEAD_DIM, HEAD_DIM + NDIRS),
        "grid": (HEAD_DIM + NDIRS, HEAD_DIM + NDIRS + GCELLS),
        "enemies": (HEAD_DIM + NDIRS + GCELLS, HEAD_DIM + NDIRS + GCELLS + M_ENEMIES * 3),
        "items": (HEAD_DIM + NDIRS + GCELLS + M_ENEMIES * 3, S.OBS_DIM)}


def main():
    pol = MLPPolicy.load(HERE.parent / "runs_sim" / "ppo_v29" / "snap_0092M.pt")
    w = np.concatenate([p.detach().numpy().reshape(-1)
                        for p in pol.net.parameters()]).astype(np.float32)

    env = Th07Env(frame_skip=3, max_seconds=999, hard_reset=True, dll_obs=True)
    env.reset(options={"hard": True})
    assert env.h.snapshot(), "snapshot failed"

    env.h.rollout_start(w, T, 256, 256, frame_skip=3, seed=777,
                        max_ep_frames=180 * 60)
    import time
    while not env.h.rollout_done():
        time.sleep(0.02)
    r_obs, r_act, r_rew, r_done, r_last, ee = env.h.rollout_result()
    first_done = int(np.argmax(r_done)) if r_done.any() else T
    print(f"rollout: {len(r_act)} steps, first episode end at {first_done}, "
          f"grid nonzero/step {np.count_nonzero(r_obs[:, 25:194], 1).mean():.0f}/169")

    # replay the actions from the same snapshot
    assert env.h.reset(), "reset failed"
    worst = {k: 0.0 for k in _seg}
    ncmp = min(first_done, T)
    for t in range(ncmp):
        step_obs = env._obs()                      # DLL step_obs (== obs.py, verified)
        d = np.abs(step_obs - r_obs[t])
        for k, (a, b) in _seg.items():
            worst[k] = max(worst[k], d[a:b].max())
        if d.max() > 0.05 and t < 8:
            j = int(d.argmax())
            print(f"  step {t}: diff {d.max():.3f} idx {j} step={step_obs[j]:.3f} roll={r_obs[t][j]:.3f}")
        env.h.step(action=env_bits(int(r_act[t])), repeat=3)
    env.close()

    print(f"\ncompared {ncmp} steps (rollout obs vs ST_STEP replay):")
    for k, v in worst.items():
        print(f"  {k:8s}: max abs diff {v:.5f}{'   <-- CHECK' if v > 0.05 else ''}")
    print("OK - rollout obs is correct" if max(worst.values()) < 0.05
          else "MISMATCH - do NOT train on ST_ROLLOUT")


def env_bits(a: int) -> int:
    from env import _decode_action
    return _decode_action(a)


if __name__ == "__main__":
    main()
