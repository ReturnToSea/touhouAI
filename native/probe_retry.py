"""Find the 'Give Up and Retry' trigger by diffing th07's .data across the action.

Attach to a th07 that's IN a Stage 1 run (paused or playing). It snapshots the
supervisor/menu region, gives you a countdown, you do:
    ESC -> DOWN to 'Give Up and Retry' -> Z -> UP to 'Yes' -> Z
then it prints the ints that changed and stuck (60Hz togglers filtered).

    .venv\\Scripts\\python native\\probe_retry.py
"""
from __future__ import annotations

import struct
import time

import pymem

LO, HI = 0x00575800, 0x00576400
STEP = 4


def snap(pm):
    b = pm.read_bytes(LO, HI - LO)
    return {LO + o: struct.unpack_from("<i", b, o)[0] for o in range(0, len(b) - 4, STEP)}


def stable_snap(pm, dwell=1.4):
    a = snap(pm)
    time.sleep(dwell)
    b = snap(pm)
    return {k: v for k, v in b.items() if a.get(k) == v}


def main():
    pm = pymem.Pymem("th07.exe")
    before = stable_snap(pm)
    print(f"{len(before)} stable ints. CLICK THE GAME.", flush=True)
    for i in range(5, 0, -1):
        print(f"  do: ESC -> DOWN to 'Give Up and Retry' -> Z -> UP to Yes -> Z ...{i}",
              flush=True)
        time.sleep(1)
    print("  GO (12s)", flush=True)
    time.sleep(12)
    after = stable_snap(pm)
    changed = [(k, before[k], after[k]) for k in before
               if k in after and after[k] != before[k]]
    print(f"\nchanged & stuck ({len(changed)}):")
    for k, o, n in sorted(changed):
        print(f"  {k:#010x}: {o} -> {n}")


if __name__ == "__main__":
    main()
