"""Gate B experiment: can a plain memory write make the sim run faster?

Measures the game's real logic-tick rate (Supervisor.calc_count) over a few
seconds, then does the same after writing candidate speed knobs:
  - framerate_multiplier  (float @ Supervisor+0x178)
  - replay_fps            (int16 @ Supervisor+0x188; the value vpatch bumps to
                           240 for replay fast-forward)

All original values are restored at the end (and on Ctrl+C).

IMPORTANT: the game only ticks while its window is focused. So:
  1. start a stage
  2. run this
  3. when it says GO, click the game and leave it focused, hands off the
     keyboard, until it says done (~30 s). Don't alt-tab.

If the game desyncs / freezes / crashes, just restart it - it's an offline
single-player game, no harm done.

Usage (one line):
    .venv\\Scripts\\python feasibility\\probe_speed.py
"""
from __future__ import annotations

import struct
import sys
import time

import pymem

import th07_data as D

SV_CALC_COUNT     = 0x150   # int32
SV_FRAMERATE_MULT = 0x178   # float
SV_REPLAY_FPS     = 0x188   # int16

MEASURE_SECONDS = 6.0


def main() -> None:
    try:
        pm = pymem.Pymem("th07.exe")
    except Exception as exc:  # noqa: BLE001
        print(f"could not attach to th07.exe ({exc}); start the game first")
        sys.exit(1)

    def calc_count() -> int:
        return struct.unpack("<i", pm.read_bytes(D.SUPERVISOR + SV_CALC_COUNT, 4))[0]

    def get_mult() -> float:
        return struct.unpack("<f", pm.read_bytes(D.SUPERVISOR + SV_FRAMERATE_MULT, 4))[0]

    def set_mult(v: float) -> None:
        pm.write_bytes(D.SUPERVISOR + SV_FRAMERATE_MULT, struct.pack("<f", v), 4)

    def get_rfps() -> int:
        return struct.unpack("<h", pm.read_bytes(D.SUPERVISOR + SV_REPLAY_FPS, 2))[0]

    def set_rfps(v: int) -> None:
        pm.write_bytes(D.SUPERVISOR + SV_REPLAY_FPS, struct.pack("<h", v), 2)

    orig_mult = get_mult()
    orig_rfps = get_rfps()
    print(f"originals: framerate_multiplier={orig_mult}  replay_fps={orig_rfps}")

    def measure(label: str) -> float:
        c0 = calc_count()
        t0 = time.perf_counter()
        time.sleep(MEASURE_SECONDS)
        dt = time.perf_counter() - t0
        rate = (calc_count() - c0) / dt
        flag = ""
        if rate < 30:
            flag = "  <-- game not focused? (too low)"
        print(f"  {label:<24} {rate:6.1f} ticks/sec{flag}")
        return rate

    try:
        for i in range(5, 0, -1):
            print(f"GO: click the game, hands off keyboard... {i}", end="\r", flush=True)
            time.sleep(1)
        print(" " * 60)

        base = measure("baseline (1.0x)")

        set_mult(3.0)
        time.sleep(0.2)
        m3 = measure("framerate_multiplier=3")
        set_mult(orig_mult)
        time.sleep(0.2)

        set_rfps(180)
        time.sleep(0.2)
        r180 = measure("replay_fps=180")
        set_rfps(orig_rfps)
        time.sleep(0.2)

        both_mult = 4.0
        set_mult(both_mult)
        set_rfps(240)
        time.sleep(0.2)
        both = measure("mult=4 + replay_fps=240")
        set_mult(orig_mult)
        set_rfps(orig_rfps)

    finally:
        set_mult(orig_mult)
        set_rfps(orig_rfps)
        print(f"restored: framerate_multiplier={get_mult()}  replay_fps={get_rfps()}")

    print()
    print(f"baseline {base:.0f}/s.  A knob 'works' if its rate is a clear multiple.")
    print("If nothing beat the baseline, the frame limiter gates the loop and the")
    print("hook has to patch that directly (still doable, just more work).")


if __name__ == "__main__":
    main()
