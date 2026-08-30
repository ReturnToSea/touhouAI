"""Smoke-test ST_HARD_RESET: construct one env, run a few short episodes using
the engine-level Stage 1 reload (no relaunch), check the game comes back clean.

    .venv/Scripts/python native/test_hard_reset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env  # noqa: E402


def snap(env):
    s = env.h.s
    return (f"stage={s.stage} gm={s.gamemode} frame={s.frame} "
            f"score={s.score} lives={s.lives:.1f} power={s.power:.0f} "
            f"pstate={s.player_state} pos=({s.player_x:.0f},{s.player_y:.0f}) "
            f"bullets={s.bullet_count}")


def main():
    env = Th07Env(frame_skip=3, max_seconds=60, render=False)
    print("constructed:", snap(env))
    for ep in range(4):
        t0 = time.time()
        if ep == 0:
            obs, _ = env.reset()               # soft (snapshot) - baseline
            kind = "soft"
        else:
            obs, _ = env.reset(options={"hard": True})
            kind = "hard"
        s = env.h.s
        print(f"\nep {ep} [{kind}] reset in {time.time()-t0:.2f}s  nav={s.nav_frames}")
        print("  after reset:", snap(env))
        assert s.stage == 1, f"stage != 1 after {kind} reset"
        assert s.gamemode == 2, f"gamemode != 2 after {kind} reset"
        assert s.player_state in (0, 3), f"player not alive (state {s.player_state})"
        assert abs(obs).max() < 50, f"obs blew up: max {abs(obs).max()}"
        # step ~8s and make sure it's really running
        for i in range(160):
            obs, r, term, trunc, info = env.step(0)   # sit still, no shoot
            if term or trunc:
                print(f"  died/ended at step {i} ({info})")
                break
        print("  after ~8s idle:", snap(env))
    env.close()
    print("\nOK - hard reset works, one process, no relaunch")


if __name__ == "__main__":
    main()
