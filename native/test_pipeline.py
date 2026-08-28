"""Milestone-1 smoke test: inject, wait for a stage, drive STEP ticks headless.

    python native\\test_pipeline.py

Navigate into a stage whenever you like - it waits (FREE mode, rendered).
Once gamemode==2 it flips to STEP and drives 1000 logic ticks, checking that
`frame` advances, obs update, and measuring the tick rate.
"""
from __future__ import annotations

import time

import shm as S
from inject import inject

BTN_SHOOT, BTN_LEFT, BTN_RIGHT, BTN_UP, BTN_DOWN = 0x01, 0x40, 0x80, 0x10, 0x20


def main() -> None:
    print(f"sizeof(Shm) = {S.ctypes.sizeof(S.Shm)}")
    pid = inject()
    print(f"injected, pid {pid}")
    h = S.Hook(pid)
    s = h.s
    print(f"shm magic={s.magic:#x} v{s.version} state={s.state}")

    print("\nFREE mode - navigate into a stage (waiting for gamemode==2)...")
    for i in range(600):  # up to 5 min
        time.sleep(0.5)
        if s.gamemode == 2 and s.stage >= 1:
            break
        if i % 6 == 0:
            print(f"  alive={s.alive} mode={s.gamemode} st={s.stage} "
                  f"pl=({s.player_x:.0f},{s.player_y:.0f}) score={s.score}")
    else:
        print("never reached a stage; abort")
        return
    print(f"  in stage: mode={s.gamemode} st={s.stage} "
          f"pl=({s.player_x:.0f},{s.player_y:.0f}) bul={s.bullet_count}")

    print("\nSTEP mode - 1000 ticks, action=SHOOT+LEFT:")
    act = BTN_SHOOT | BTN_LEFT
    f0, t0, ok = s.frame, time.perf_counter(), 0
    for i in range(1000):
        if h.step(action=act, repeat=1):
            ok += 1
        else:
            print(f"  step {i}: TIMEOUT (done never set)")
            break
        if i % 100 == 0:
            print(f"  step {i:4d} frame={s.frame} pl=({s.player_x:.1f},{s.player_y:.1f}) "
                  f"st={s.stage} pstate={s.player_state} bul={s.bullet_count} "
                  f"ene={s.enemy_count} score={s.score} boss={s.boss_hp:.0f}/{s.boss_hp_max:.0f} "
                  f"status={s.tick_status}")
    dt = time.perf_counter() - t0
    df = s.frame - f0
    print(f"\n{ok}/1000 steps, frame +{df}, {df / dt:.0f} ticks/sec "
          f"({df / dt / 60:.1f}x real-time)")
    h.set_free()
    h.close()


if __name__ == "__main__":
    main()
