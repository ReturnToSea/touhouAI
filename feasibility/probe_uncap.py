"""Gate B experiment 2: can we uncap the sim with a memory write (no DLL)?

Static analysis of Window::do_tick (th07.exe 0x4346E0) found the frame limiter:

    0x4348C1  je 0x4348CE                 ; elapsed >= 1/60 -> run a frame
    0x4348C3  movzx eax, byte [0x575C3C]  ; else check the uncap flag
    0x4348CC  je 0x434924                 ; flag == 0 -> skip (hold 60 fps)
                                          ; flag != 0 -> run frame anyway

[0x575C3C] gates BOTH the QPC and timeGetTime limiter paths, and the loop it
unblocks runs run_all_on_tick (real game logic). So a nonzero write there
*should* uncap logic + render. [0x575A8B] is render frameskip (skip drawing
N frames between logic ticks).

This measures Supervisor.calc_count over a few seconds for each setting. The
game only ticks while focused, so: start a stage, run this, and when it says
GO, click the game and leave it focused with hands off the keyboard.

Everything is restored at the end and on Ctrl+C. If the game misbehaves just
restart it.

Usage (one line):
    .venv\\Scripts\\python feasibility\\probe_uncap.py
"""
from __future__ import annotations

import struct
import sys
import time

import pymem

import th07_data as D

CALC_COUNT = D.SUPERVISOR + 0x150   # int32
UNCAP_FLAG = 0x00575C3C             # uint8   -- the limiter bypass
FRAMESKIP  = 0x00575A8B             # uint8   -- render frameskip
TURBO_FLAG = 0x00575A8A             # uint8   -- ? also checked around the limiter

MEASURE_SECONDS = 5.0


def main() -> None:
    try:
        pm = pymem.Pymem("th07.exe")
    except Exception as exc:  # noqa: BLE001
        print(f"could not attach to th07.exe ({exc}); start a stage first")
        sys.exit(1)

    def calc() -> int:
        return struct.unpack("<i", pm.read_bytes(CALC_COUNT, 4))[0]

    def rb(va):
        return pm.read_bytes(va, 1)[0]

    def wb(va, v):
        pm.write_bytes(va, bytes([v & 0xFF]), 1)

    o_uncap, o_skip, o_turbo = rb(UNCAP_FLAG), rb(FRAMESKIP), rb(TURBO_FLAG)
    print(f"originals: uncap[0x575c3c]={o_uncap}  frameskip[0x575a8b]={o_skip}  "
          f"turbo[0x575a8a]={o_turbo}")

    def measure(label: str) -> float:
        try:
            c0 = calc()
            t0 = time.perf_counter()
            time.sleep(MEASURE_SECONDS)
            dt = time.perf_counter() - t0
            rate = (calc() - c0) / dt
        except pymem.exception.MemoryReadError:
            print(f"  {label:<32}  game exited / unreadable")
            raise SystemExit(1)
        note = "   <-- not focused?" if rate < 30 else ""
        print(f"  {label:<32} {rate:7.1f} ticks/sec{note}")
        return rate

    trials = [
        ("frameskip=1  (baseline)",     {FRAMESKIP: 1}),
        ("frameskip=9",                 {FRAMESKIP: 9}),
        ("frameskip=29",                {FRAMESKIP: 29}),
        ("frameskip=59",                {FRAMESKIP: 59}),
        ("frameskip=119",               {FRAMESKIP: 119}),
        ("frameskip=249",               {FRAMESKIP: 249}),
    ]

    try:
        for i in range(5, 0, -1):
            print(f"GO: click the game, hands off keyboard... {i}", end="\r", flush=True)
            time.sleep(1)
        print(" " * 60)

        base = None
        for label, writes in trials:
            wb(UNCAP_FLAG, o_uncap)
            wb(FRAMESKIP, o_skip)
            wb(TURBO_FLAG, o_turbo)
            for va, v in writes.items():
                wb(va, v)
            time.sleep(0.3)
            rate = measure(label)
            if base is None:
                base = rate
    finally:
        try:
            wb(UNCAP_FLAG, o_uncap)
            wb(FRAMESKIP, o_skip)
            wb(TURBO_FLAG, o_turbo)
            print(f"restored: uncap={rb(UNCAP_FLAG)} frameskip={rb(FRAMESKIP)} "
                  f"turbo={rb(TURBO_FLAG)}")
        except Exception:  # noqa: BLE001
            print("(could not restore - game already exited)")

    if base:
        print(f"\nbaseline ~{base:.0f}/s. Any row well above that = a working "
              f"memory-only speedup (no DLL needed for the frame limiter).")
        print("If everything sticks near 60, Present() is vsync-locked and the "
              "hook must also force D3DPRESENT_INTERVAL_IMMEDIATE.")


if __name__ == "__main__":
    main()
