"""Write-test th07's suspected in-place retry trigger.

Attach to a th07 that is IN a Stage 1 run (playing, NOT paused). It reads the
stage-frame counters, writes <addr>=<value>, then watches for ~3s to see whether
a stage (re)load fires - detected by GAME_MANAGER frame counters snapping back
to ~0.

    .venv\\Scripts\\python native\\probe_write.py 0x575aa8 10
    .venv\\Scripts\\python native\\probe_write.py 0x62f640 0
    .venv\\Scripts\\python native\\probe_write.py 0x575c30 0

The game is disposable - if a write corrupts it, just relaunch. Run from a live
stage each time (get shot at / move around so the counters are clearly ticking).
"""
from __future__ import annotations

import struct
import sys
import time

import pymem

# stage-frame counters - snap to ~0 on a (re)load
GM_FRAME = [0x0062F858, 0x0062F898]
SUP_MODE = 0x00575AA8
SUP_TIMER = 0x00575AB4
GM_STAGE = 0x0062F85C
WATCH = GM_FRAME + [SUP_MODE, SUP_TIMER, GM_STAGE]

# GUI = 0x0049FBF0: score +0x00 (i32), life +0x5C (f32), power +0x7C (f32)
GUI = 0x0049FBF0
A_SCORE = GUI + 0x00
A_LIFE = GUI + 0x5C
A_POWER = GUI + 0x7C


def rd(pm, a):
    try:
        return struct.unpack("<i", pm.read_bytes(a, 4))[0]
    except Exception:
        return None


def rf(pm, a):
    try:
        return struct.unpack("<f", pm.read_bytes(a, 4))[0]
    except Exception:
        return None


def run_state(pm):
    return (f"score={rd(pm, A_SCORE)}  life={rf(pm, A_LIFE)}  "
            f"power={rf(pm, A_POWER)}  stage={rd(pm, GM_STAGE)}")


def main():
    if len(sys.argv) != 3:
        print("usage: probe_write.py <addr hex> <int value>")
        return
    addr = int(sys.argv[1], 16)
    val = int(sys.argv[2])

    pm = pymem.Pymem("th07.exe")
    before = {a: rd(pm, a) for a in WATCH}
    old = rd(pm, addr)
    print(f"before: {addr:#010x} = {old}", flush=True)
    for a in WATCH:
        print(f"  {a:#010x} = {before[a]}", flush=True)
    print(f"  RUN STATE  {run_state(pm)}", flush=True)

    # confirm the frame counters are actually moving
    time.sleep(0.3)
    mid = {a: rd(pm, a) for a in GM_FRAME}
    moving = any(mid[a] != before[a] for a in GM_FRAME)
    print(f"\nframe counters moving: {moving}  ({[mid[a] for a in GM_FRAME]})",
          flush=True)
    if not moving:
        print("  -> counters frozen. Are you in a live stage (not paused, "
              "not in dialogue)? Continuing anyway.", flush=True)

    print(f"\nWRITE {addr:#010x} <- {val}", flush=True)
    pm.write_bytes(addr, struct.pack("<i", val), 4)

    t0 = time.perf_counter()
    reload_seen = False
    last = None
    while time.perf_counter() - t0 < 3.0:
        cur = {a: rd(pm, a) for a in WATCH}
        line = "  ".join(f"{a & 0xffff:#06x}={cur[a]}" for a in WATCH)
        if line != last:
            print(f"  t+{time.perf_counter()-t0:4.2f}  {line}", flush=True)
            last = line
        # reload = a GM frame counter dropped hard toward 0
        for a in GM_FRAME:
            if cur[a] is not None and before[a] is not None \
                    and cur[a] < before[a] - 30 and cur[a] < 60:
                reload_seen = True
        time.sleep(1 / 120)

    print(f"\n{'*** STAGE RELOAD DETECTED ***' if reload_seen else 'no reload'}",
          flush=True)
    time.sleep(0.5)
    print(f"RUN STATE after  {run_state(pm)}", flush=True)
    print("  (want: score=0, life back to starting count, power reset, "
          "stage=1 - if stage != 1 it retried the CURRENT stage, not stage 1)",
          flush=True)
    if not reload_seen and rd(pm, addr) == val:
        # restore if the game just ignored it and kept our poke sitting there
        print(f"restoring {addr:#010x} <- {old}", flush=True)
        pm.write_bytes(addr, struct.pack("<i", old), 4)


if __name__ == "__main__":
    main()
