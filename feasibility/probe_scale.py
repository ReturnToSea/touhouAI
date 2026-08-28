"""Gate B recon: what will it take to run faster-than-realtime and/or many copies.

This is read-only inspection plus an optional second-instance test. It does not
patch anything. It reports the levers we would use for the C++ hook later.

Usage:
    .venv\\Scripts\\python feasibility\\probe_scale.py
    .venv\\Scripts\\python feasibility\\probe_scale.py --spawn-second "<path to th07.exe>"
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys
import time

import pymem

import th07_data as D

# zSupervisor fields relevant to timing (offsets from thprac_th07.h / th-re-data)
SV_CALC_COUNT        = 0x150   # int32  frames simulated
SV_GAMEMODE          = 0x154
SV_GUI_UPDATE_FRAMES = 0x164
SV_DISABLE_VSYNC     = 0x16C   # int32
SV_LAST_FRAME_TIME   = 0x174   # int32
SV_FRAMERATE_MULT    = 0x178   # float  <-- game multiplies its timestep by this
SV_LAG_NUM           = 0x180   # float
SV_LAG_DEN           = 0x184   # float
SV_REPLAY_FPS        = 0x188   # int16  (vpatch's ReplaySkipFPS lands here)

# Update / draw callback chain node heads (struct zUpdateFunc).
# The C++ hook would walk/replace these to skip rendering or double-tick logic.
UPDATE_FUNCS = {
    "GAME_MANAGER_ON_TICK": 0x0062F8B4,
    "GAME_MANAGER_ON_DRAW": 0x0062F8D4,
    "GUI_ON_TICK":          0x0062F914,
    "GUI_ON_DRAW":          0x0062F8F4,
    "BULLET_MANAGER_ON_TICK": 0x009A9ABC,
    "BULLET_MANAGER_ON_DRAW": 0x0062F934,
    "ENEMY_MANAGER_ON_TICK": 0x012FE210,
    "STAGE_ON_TICK":        0x0134CDD4,
}

# thprac disables the single-instance mutex by forcing EIP 0x435bff -> 0x435c1b
# (function `th07_disable_mutex`). So: a mutex DOES exist, and its check lives at
# th07.exe+0x35bff. Our hook can NOP the same site to allow parallel instances.
MUTEX_CHECK_SITE = 0x00435BFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn-second", metavar="TH07_EXE",
                    help="launch a second copy of the given exe and report if it runs")
    args = ap.parse_args()

    try:
        pm = pymem.Pymem("th07.exe")
    except Exception as exc:  # noqa: BLE001
        print(f"attach failed ({exc}); start the game first")
        sys.exit(1)

    def rd(off: int, n: int) -> bytes:
        return pm.read_bytes(D.SUPERVISOR + off, n)

    print(f"pid {pm.process_id}")
    print("--- timing state (zSupervisor) ---")
    print(f"  calc_count          {struct.unpack('<i', rd(SV_CALC_COUNT, 4))[0]}")
    print(f"  gamemode            {struct.unpack('<I', rd(SV_GAMEMODE, 4))[0]}")
    print(f"  disable_vsync       {struct.unpack('<i', rd(SV_DISABLE_VSYNC, 4))[0]}")
    print(f"  last_frame_time     {struct.unpack('<i', rd(SV_LAST_FRAME_TIME, 4))[0]} (ms)")
    print(f"  framerate_multiplier {struct.unpack('<f', rd(SV_FRAMERATE_MULT, 4))[0]:.3f}")
    print(f"  lag_pct             {struct.unpack('<f', rd(SV_LAG_NUM, 4))[0]:.2f}"
          f" / {struct.unpack('<f', rd(SV_LAG_DEN, 4))[0]:.2f}")
    print(f"  replay_fps          {struct.unpack('<h', rd(SV_REPLAY_FPS, 2))[0]}")

    # measure the real logic rate over 1s
    c0 = struct.unpack("<i", rd(SV_CALC_COUNT, 4))[0]
    time.sleep(1.0)
    c1 = struct.unpack("<i", rd(SV_CALC_COUNT, 4))[0]
    print(f"  observed logic rate ~{c1 - c0} ticks/sec")

    print("--- update/draw chain heads (for the headless hook) ---")
    for name, va in UPDATE_FUNCS.items():
        try:
            first_cb = struct.unpack("<I", pm.read_bytes(va + 0x4, 4))[0]
            print(f"  {name:<24} node@{va:#010x}  first_cb={first_cb:#010x}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<24} node@{va:#010x}  <unreadable: {exc}>")

    print(f"--- single-instance mutex check lives at th07.exe+{MUTEX_CHECK_SITE - D.IMAGE_BASE:#x} "
          f"(VA {MUTEX_CHECK_SITE:#x}); hook NOPs it for parallel instances ---")

    if args.spawn_second:
        print(f"--- launching second instance: {args.spawn_second} ---")
        p = subprocess.Popen([args.spawn_second],
                             cwd=str(pathlib.Path(args.spawn_second).parent))
        time.sleep(3.0)
        rc = p.poll()
        if rc is None:
            print("  second instance is still running -> no hard single-instance lock,"
                  " OR it silently shares. Check the screen.")
            p.terminate()
        else:
            print(f"  second instance exited quickly (rc={rc}) -> single-instance"
                  " mutex is blocking it; hook will be required.")


if __name__ == "__main__":
    main()
