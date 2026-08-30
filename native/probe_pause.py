"""RE the th07 pause menu for ST_HARD_RESET.

Launches the game (ST_FREE - you play normally), then:
  * logs the INPUT_CUR word (0x4B9E4C) whenever it changes -> the PAUSE bit
  * watches a scan window of .data ints and prints any that change -> the pause
    menu cursor / a "pause active" flag

    .venv\\Scripts\\python native\\probe_pause.py

Then: click the game, get into Stage 1, press ESC (pause), tap DOWN a few times,
tap UP, press Z on an option (Return to Game), pause again, etc. Ctrl-C to stop.
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import pymem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inject import inject  # noqa: E402

INPUT_CUR = 0x004B9E4C
# scan these .data regions for small ints that toggle with the pause menu
SCAN = [(0x00575900, 0x00575D00), (0x00625E00, 0x00626400), (0x004B9C00, 0x004BA200)]
BTN = {0x01: "Z", 0x02: "X", 0x04: "shift", 0x08: "ctrl", 0x10: "up",
       0x20: "down", 0x40: "left", 0x80: "right"}


def decode(w):
    lo = "+".join(n for b, n in BTN.items() if w & b) or "-"
    hi = w & ~0xFF
    return f"{w:#06x}  [{lo}]" + (f"  HIGH={hi:#x}" if hi else "")


def main():
    pid = inject()
    print(f"th07 pid {pid} - click it, get to Stage 1, then press ESC and navigate "
          f"the pause menu. Ctrl-C to stop.\n", flush=True)
    time.sleep(3)
    pm = pymem.Pymem(pid)

    def snap():
        d = {}
        for lo, hi in SCAN:
            try:
                blob = pm.read_bytes(lo, hi - lo)
                for o in range(0, len(blob) - 4, 4):
                    v = struct.unpack_from("<i", blob, o)[0]
                    if -8 < v < 64:          # small ints - cursors, flags, small enums
                        d[lo + o] = v
            except Exception:
                pass
        return d

    last_w = -1
    base = snap()
    print(f"baseline: tracking {len(base)} small-int addresses", flush=True)
    while True:
        try:
            w = pm.read_ushort(INPUT_CUR)
        except Exception:
            time.sleep(0.2)
            continue
        if w != last_w:
            print(f"INPUT  {decode(w)}", flush=True)
            last_w = w
        cur = snap()
        for a, v in cur.items():
            if base.get(a) != v:
                print(f"  mem {a:#010x}: {base.get(a)} -> {v}", flush=True)
                base[a] = v
        time.sleep(0.03)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
