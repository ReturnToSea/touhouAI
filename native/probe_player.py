"""Nail down ReimuA's move speed + hitbox: read the exact params out of th07.exe
AND measure movement empirically, then compare to the FightSim constants.

    .venv/Scripts/python native/probe_player.py

From FUN_0043e260 / the player-move fn (Ghidra, decomp_all.c ~27700-28890):
  shot_root = *(u32*)0x00575948            (== PLAYER + 0xb7e70, unfocused)
  player hitbox half  = *(f32*)(shot_root + 0x0C) / 2      # square: hb_x == hb_y
  graze box   half    = *(f32*)(shot_root + 0x10) / 2
  move speed (px/f)   = *(f32*)(shot_root + 0x24)  unfocused cardinal
                        *(f32*)(shot_root + 0x28)  focused   cardinal
                        *(f32*)(shot_root + 0x2C)  unfocused diagonal (pre-/sqrt2)
                        *(f32*)(shot_root + 0x30)  focused   diagonal
  per-frame speed mult = PLAYER + 0x23F0 / 0x23F4  (1.0 idle; <1 while firing)
  global timescale     = *(f32*)0x00575AC8
  playfield clamp: x in [DAT_0062F874, +DAT_0062F87C], y in [DAT_0062F878, +DAT_0062F880]
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from env import Th07Env                       # noqa: E402
import shm as S                               # noqa: E402

PLAYER = 0x004BDAD8
PL_X, PL_Y = 0x930, 0x934
SR_UNFOCUSED = 0x00575948
TIMESCALE = 0x00575AC8
CLAMP_X0, CLAMP_XW = 0x0062F874, 0x0062F87C
CLAMP_Y0, CLAMP_YW = 0x0062F878, 0x0062F880


def f32(pm, addr):
    return struct.unpack("<f", pm.read_bytes(addr, 4))[0]


def u32(pm, addr):
    return struct.unpack("<I", pm.read_bytes(addr, 4))[0]


def player_xy(pm):
    b = pm.read_bytes(PLAYER + PL_X, 8)
    return struct.unpack("<ff", b)


def hold(env, bits, n):
    """Step n single frames with `bits` held; return the list of (x, y)."""
    out = []
    for _ in range(n):
        env.h.step(action=bits, repeat=1)
        out.append(player_xy(env._pm))
    return out


def per_frame_delta(track):
    """Steady-state |displacement| per frame (skip the first 3 frames: ramp)."""
    import math
    d = [math.hypot(track[i + 1][0] - track[i][0], track[i + 1][1] - track[i][1])
         for i in range(len(track) - 1)]
    steady = d[3:] if len(d) > 6 else d
    return sum(steady) / max(len(steady), 1), d


def main():
    env = Th07Env(frame_skip=1, max_seconds=120, render=False, dll_obs=True,
                  hard_reset=False)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)
        env._pm = pm

    # let the stage settle so the player is fully in control
    for _ in range(90):
        env.h.step(action=0, repeat=1)

    root = u32(pm, SR_UNFOCUSED)
    print(f"\n=== th07.exe, read straight from memory (shot_root {root:#x}) ===")
    hb = f32(pm, root + 0x0C) / 2.0
    grz = f32(pm, root + 0x10) / 2.0
    su_c, sf_c = f32(pm, root + 0x24), f32(pm, root + 0x28)
    su_d, sf_d = f32(pm, root + 0x2C), f32(pm, root + 0x30)
    mult = f32(pm, PLAYER + 0x23F0)
    ts = f32(pm, TIMESCALE)
    cx0, cxw = f32(pm, CLAMP_X0), f32(pm, CLAMP_XW)
    cy0, cyw = f32(pm, CLAMP_Y0), f32(pm, CLAMP_YW)
    print(f"  player hitbox half-extent : {hb:.4f}  (full box {hb*2:.3f})")
    print(f"  graze  box   half-extent  : {grz:.4f}")
    print(f"  move speed  unfocused card : {su_c:.4f} px/f     focused card : {sf_c:.4f}")
    print(f"  move speed  unfocused diag : {su_d:.4f} px/f     focused diag : {sf_d:.4f}")
    print(f"  per-frame speed mult (idle): {mult:.4f}    global timescale: {ts:.4f}")
    print(f"  playfield clamp: x [{cx0:.1f}, {cx0+cxw:.1f}]   y [{cy0:.1f}, {cy0+cyw:.1f}]")

    # ---- empirical: hold a direction, measure px/frame ----
    print(f"\n=== measured (hold key 40 frames, steady-state |delta|/frame) ===")
    tests = [
        ("unfocused  RIGHT", S.RIGHT, 40),
        ("unfocused  DOWN ", S.DOWN, 40),
        ("unfocused  DR-diag", S.DOWN | S.RIGHT, 40),
        ("focused    RIGHT", S.RIGHT | S.SLOW, 40),
        ("focused    DOWN ", S.DOWN | S.SLOW, 40),
        ("focused    DR-diag", S.DOWN | S.RIGHT | S.SLOW, 40),
    ]
    # re-centre before each run so it doesn't sit on a wall
    for name, bits, n in tests:
        struct.pack_into  # noqa
        pm.write_bytes(PLAYER + PL_X, struct.pack("<ff", 192.0, 240.0), 8)
        env.h.step(action=0, repeat=2)
        tr = hold(env, bits, n)
        v, _d = per_frame_delta(tr)
        print(f"  {name}: {v:.4f} px/f   ({tr[0][0]:.1f},{tr[0][1]:.1f}) -> "
              f"({tr[-1][0]:.1f},{tr[-1][1]:.1f})")

    # ---- corner to corner ----
    print(f"\n=== corner-to-corner (unfocused) ===")
    pm.write_bytes(PLAYER + PL_X, struct.pack("<ff", cx0 + 1, cy0 + 1), 8)
    env.h.step(action=0, repeat=2)
    x0, y0 = player_xy(pm)
    frames = 0
    for _ in range(600):
        env.h.step(action=S.DOWN | S.RIGHT, repeat=1)
        frames += 1
        x, y = player_xy(pm)
        if x >= cx0 + cxw - 0.5 and y >= cy0 + cyw - 0.5:
            break
    x1, y1 = player_xy(pm)
    import math
    dist = math.hypot(x1 - x0, y1 - y0)
    print(f"  ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f})  {frames} frames  "
          f"diag {dist:.1f}px  -> {dist/frames:.3f} px/f along the diagonal")
    span_x = cxw
    for _ in range(600):                       # pure horizontal sweep
        pass
    pm.write_bytes(PLAYER + PL_X, struct.pack("<ff", cx0 + 1, cy0 + cyw / 2), 8)
    env.h.step(action=0, repeat=2)
    fx = 0
    for _ in range(600):
        env.h.step(action=S.RIGHT, repeat=1)
        fx += 1
        x, _y = player_xy(pm)
        if x >= cx0 + cxw - 0.5:
            break
    print(f"  full width {span_x:.0f}px swept in {fx} frames -> {span_x/fx:.3f} px/f")

    # ---- compare to the sim ----
    from fight_replay import (PLAYER_HB, PX_LO, PX_HI, PY_LO, PY_HI)
    from obs import DIR_SPEED, DIR_SPEED_FOCUS
    print(f"\n=== FightSim constants ===")
    print(f"  PLAYER_HB (sim collision half) : {PLAYER_HB}")
    print(f"  DIR_SPEED / DIR_SPEED_FOCUS    : {DIR_SPEED} / {DIR_SPEED_FOCUS}")
    print(f"  playfield PX[{PX_LO},{PX_HI}] PY[{PY_LO},{PY_HI}]  "
          f"(w {PX_HI-PX_LO:.0f} h {PY_HI-PY_LO:.0f})")

    env.close()


if __name__ == "__main__":
    main()
