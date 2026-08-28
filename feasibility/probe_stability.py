"""Gate B follow-up: does the patched game run FAST and STABLE for a while?

Prereqs: launch with launch_uncapped.py and get into a stage.

This sets render frameskip once, then only *reads* for --minutes, logging the
logic tick rate + player state + whether the process is still alive every few
seconds. It never pokes memory after the initial frameskip write, so a crash
here is the game's, not ours.

  - "process exited"  -> real crash. Note how long it lasted / last state.
  - calc_count frozen but process alive -> just the Continue screen (fine).
  - tick rate steady over the whole run -> stable; that rate is the ceiling
    for this frameskip.

You do NOT need to survive - deaths / continue screens are fine, we only care
whether the process stays up. If you want a cleaner rate reading, sit in a
low-bullet spot.

Usage:
    .venv\\Scripts\\python feasibility\\probe_stability.py --frameskip 9 --minutes 3
    .venv\\Scripts\\python feasibility\\probe_stability.py --frameskip 59 --minutes 2
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

import pymem

import th07_data as D

CALC_COUNT = D.SUPERVISOR + 0x150
FRAMESKIP  = 0x00575A8B

# skip all per-frame D3D (needs the limiter patches from launch_uncapped.py)
HEADLESS_PATCHES = [
    (0x00434718, b"\x7F\x74", b"\xEB\x74"),
    (0x00434A0E, b"\xE8\xAD\xFB\xFF\xFF", b"\x90" * 5),
    (0x00434A18, b"\xE8\xA3\xFB\xFF\xFF", b"\x90" * 5),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameskip", type=int, default=0,
                    help="render frameskip to set (0 = leave as-is)")
    ap.add_argument("--headless", "-H", action="store_true",
                    help="apply render+Present skip patches to the running game")
    ap.add_argument("--keepalive", "-k", action="store_true",
                    help="rewrite lives high every poll so the stage keeps running")
    ap.add_argument("--minutes", type=float, default=1.5)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    try:
        pm = pymem.Pymem("th07.exe")
    except Exception as exc:  # noqa: BLE001
        print(f"could not attach ({exc}); launch_uncapped.py then get into a stage")
        sys.exit(1)

    if args.headless:
        for va, want, new in HEADLESS_PATCHES:
            cur = pm.read_bytes(va, len(want))
            if cur != want:
                print(f"  {va:#x}: expected {want.hex()} got {cur.hex()} - skipping")
                continue
            pm.write_bytes(va, new, len(new))
            print(f"  headless patch {va:#x}: {want.hex()} -> "
                  f"{pm.read_bytes(va, len(new)).hex()}")
    if args.frameskip:
        pm.write_bytes(FRAMESKIP, bytes([args.frameskip & 0xFF]), 1)

    print(f"pid {pm.process_id}, frameskip={pm.read_bytes(FRAMESKIP, 1)[0]}, "
          f"headless={args.headless}. monitoring {args.minutes:.0f} min (reads only)...")

    t_start = time.perf_counter()
    end = t_start + args.minutes * 60
    last_c = struct.unpack("<i", pm.read_bytes(CALC_COUNT, 4))[0]
    last_t = time.perf_counter()
    rates = []
    frozen_streak = 0

    while time.perf_counter() < end:
        time.sleep(args.interval)
        now = time.perf_counter()
        try:
            if args.keepalive:
                gp = struct.unpack("<I", pm.read_bytes(D.GAME_MANAGER + D.GM_GLOBALS_PTR, 4))[0]
                if gp:
                    pm.write_bytes(gp + D.G_LIFE_COUNT, struct.pack("<f", 8.0), 4)
            c = struct.unpack("<i", pm.read_bytes(CALC_COUNT, 4))[0]
            state = D.PLAYER_STATE_NAMES.get(pm.read_bytes(D.PLAYER + D.PLAYER_STATE, 1)[0], "?")
            px = struct.unpack("<f", pm.read_bytes(D.PLAYER + D.PLAYER_POS_X, 4))[0]
            py = struct.unpack("<f", pm.read_bytes(D.PLAYER + D.PLAYER_POS_Y, 4))[0]
            stage = struct.unpack("<i", pm.read_bytes(D.GAME_MANAGER + D.GM_STAGE, 4))[0]
            bm = struct.unpack("<i", pm.read_bytes(D.BULLET_MANAGER + D.BM_BULLET_COUNT, 4))[0]
            gptr = struct.unpack("<I", pm.read_bytes(D.GAME_MANAGER + D.GM_GLOBALS_PTR, 4))[0]
            if gptr:
                gb = pm.read_bytes(gptr, 0x80)
                lives = struct.unpack_from("<f", gb, D.G_LIFE_COUNT)[0]
                score = struct.unpack_from("<i", gb, D.G_DISPLAYED_SCORE)[0]
            else:
                lives, score = -1, -1
        except pymem.exception.MemoryReadError:
            elapsed = now - t_start
            print(f"\n*** PROCESS EXITED after {elapsed:.0f}s "
                  f"({elapsed/60:.1f} min) — real crash ***")
            if rates:
                print(f"    tick rate before crash: {rates[-1]:.0f}/s "
                      f"(avg {sum(rates)/len(rates):.0f})")
            sys.exit(2)
        rate = (c - last_c) / (now - last_t)
        last_c, last_t = c, now
        if rate < 5:
            frozen_streak += 1
            tag = f"  (frozen x{frozen_streak} - Continue screen / paused, process OK)"
        else:
            frozen_streak = 0
            rates.append(rate)
            tag = ""
        print(f"  t+{now - t_start:5.0f}s {rate:9.0f}/s  st{stage} "
              f"pl=({px:5.0f},{py:5.0f}) {state:<10} L={lives:.0f} "
              f"bul={bm:<4} score={score}{tag}")

    print(f"\nSURVIVED {args.minutes:.0f} min at frameskip {args.frameskip}.")
    if rates:
        print(f"  tick rate: avg {sum(rates)/len(rates):.0f}/s, "
              f"min {min(rates):.0f}, max {max(rates):.0f}  "
              f"(~{sum(rates)/len(rates)/60:.0f}x real-time)")
        drift = rates[-1] - rates[0]
        print(f"  drift first->last: {drift:+.0f}/s "
              f"({'stable' if abs(drift) < 0.1 * rates[0] else 'DEGRADING'})")


if __name__ == "__main__":
    main()
