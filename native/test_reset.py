"""Validate snapshot/reset: two runs of the same input sequence from a snapshot
must produce identical observations.

    python native\\test_reset.py

Navigate into a stage, then it snapshots and does 2x400 steps with a fixed
pseudo-random action sequence, comparing player pos / bullet count / score /
enemy count frame-by-frame.
"""
from __future__ import annotations

import random
import time

import shm as S
from inject import inject

MOVES = [0x00, 0x40, 0x80, 0x10, 0x20, 0x41, 0x81, 0x01, 0x05]  # dirs + shoot/slow
N = 400


def run(h, actions):
    trace = []
    for a in actions:
        h.step(action=a, repeat=1)
        s = h.s
        trace.append((round(s.player_x, 2), round(s.player_y, 2),
                      s.bullet_count, s.enemy_count, s.score, s.player_state))
    return trace


def main() -> None:
    pid = inject()
    print(f"injected pid {pid}")
    h = S.Hook(pid)
    s = h.s

    print("navigate into a stage...")
    while not (s.gamemode == 2 and s.stage >= 1):
        time.sleep(0.5)
    time.sleep(0.5)
    print(f"in stage st{s.stage} pl=({s.player_x:.0f},{s.player_y:.0f}). snapshotting.")
    assert h.snapshot(), "snapshot timed out"
    assert s.have_snapshot

    rng = random.Random(1234)
    actions = [rng.choice(MOVES) for _ in range(N)]

    t0 = time.perf_counter()
    a = run(h, actions)
    dt_a = time.perf_counter() - t0
    print(f"run A done ({N/dt_a:.0f} steps/s). last={a[-1]}")

    t0 = time.perf_counter()
    assert h.reset(), "reset timed out"
    dt_r = time.perf_counter() - t0
    print(f"reset in {dt_r*1000:.1f} ms. post-reset pl=({s.player_x:.0f},{s.player_y:.0f}) "
          f"bul={s.bullet_count} score={s.score}")

    b = run(h, actions)
    print(f"run B done. last={b[-1]}")

    mism = [(i, a[i], b[i]) for i in range(N) if a[i] != b[i]]
    if not mism:
        print(f"\nPASS - {N} frames identical after reset. snapshot is faithful.")
    else:
        print(f"\nFAIL - {len(mism)}/{N} frames differ. first few:")
        for i, x, y in mism[:8]:
            print(f"  frame {i}: A={x}  B={y}")
    h.set_free()
    h.close()


if __name__ == "__main__":
    main()
