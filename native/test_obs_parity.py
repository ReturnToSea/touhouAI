"""Verify the DLL's C-built observation (shm.step_obs) matches the Python
builder (native/obs.py) frame-for-frame. Runs a policy on the real game and
diffs the two on every step.

    .venv/Scripts/python native/test_obs_parity.py [steps]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from env import Th07Env  # noqa: E402
from obs import HEAD_DIM, NDIRS, GCELLS, M_ENEMIES  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
_O_ESC, _O_GRID = HEAD_DIM, HEAD_DIM + NDIRS
_O_ENE = _O_GRID + GCELLS
_O_ITEM = _O_ENE + M_ENEMIES * 3


def main():
    try:
        from policy import MLPPolicy
        pol = MLPPolicy.load(HERE.parent / "runs_sim" / "ppo_v29" / "snap_0092M.pt")
        act = pol.act
    except Exception:
        act = lambda o: np.random.randint(0, 36)  # noqa: E731

    env = Th07Env(frame_skip=3, max_seconds=120, render=False, dll_obs=True)
    obs, _ = env.reset()
    segs = {"head": (0, HEAD_DIM), "escape": (_O_ESC, _O_GRID),
            "grid": (_O_GRID, _O_ENE), "enemies": (_O_ENE, _O_ITEM),
            "items": (_O_ITEM, obs.shape[0])}
    worst = {k: 0.0 for k in segs}
    n_big = 0
    for i in range(N):
        dll = env._obs()            # shm.step_obs
        py = env._obs_python()      # rebuilt
        d = np.abs(dll - py)
        for k, (a, b) in segs.items():
            worst[k] = max(worst[k], d[a:b].max())
        if d.max() > 0.02:
            n_big += 1
            if n_big <= 5:
                j = int(d.argmax())
                seg = next(k for k, (a, b) in segs.items() if a <= j < b)
                print(f"  step {i}: max diff {d.max():.4f} at idx {j} ({seg})  "
                      f"dll={dll[j]:.4f} py={py[j]:.4f}")
        obs, r, term, trunc, _ = env.step(int(act(dll)))
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    print("\nmax abs diff per segment (DLL vs Python):")
    for k, v in worst.items():
        flag = "  <-- CHECK" if v > 0.02 else ""
        print(f"  {k:8s}: {v:.5f}{flag}")
    print(f"\nsteps with any diff > 0.02: {n_big}/{N}")
    print("OK" if max(worst.values()) < 0.05 else "MISMATCH - do not train on DLL obs")


if __name__ == "__main__":
    main()
