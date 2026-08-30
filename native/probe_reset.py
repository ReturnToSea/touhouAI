"""Find th07's in-place retry trigger (scene-transition request word / pause flag).

Attach to a th07 that is IN a Stage 1 run. It baselines a wide .data window,
gives you a countdown, then samples that window EVERY FRAME for ~18s while you do:

    ESC  ->  DOWN to 'Give Up and Retry'  ->  Z  ->  UP to 'Yes'  ->  Z

Unlike probe_retry.py (before/after stable diff only) this records, per address:
  * baseline value
  * every distinct value seen and the sample index it first appeared
  * whether it ended != baseline (sticky end-state) or blipped and moved on
    (transient -> candidate request flag)

    .venv\\Scripts\\python native\\probe_reset.py

Say roughly when you pressed the final Z ("yes at ~8s") so we can correlate.
"""
from __future__ import annotations

import struct
import time

import pymem

# SUPERVISOR ~0x575950, GAME_MANAGER 0x626270 (+0x95EC lands at 0x62F85C, so the
# struct runs long). Cover both plus the input/global area.
REGIONS = [
    (0x00575800, 0x00576400),
    (0x004B9C00, 0x004BA200),
    (0x00626200, 0x00630000),
]
SAMPLE_HZ = 60
DURATION_S = 18.0
# only care about small-magnitude ints (cursors, enums, flags, request codes)
SMALL = 4096


def read_region(pm, lo, hi):
    return pm.read_bytes(lo, hi - lo)


def main():
    pm = pymem.Pymem("th07.exe")

    def snap():
        d = {}
        for lo, hi in REGIONS:
            try:
                b = read_region(pm, lo, hi)
            except Exception:
                continue
            for o in range(0, len(b) - 4, 4):
                v = struct.unpack_from("<i", b, o)[0]
                if -SMALL < v < SMALL:
                    d[lo + o] = v
        return d

    base = snap()
    print(f"baseline: {len(base)} small-int addresses", flush=True)
    for i in range(5, 0, -1):
        print(f"  ready: ESC -> DOWN to 'Give Up and Retry' -> Z -> UP to 'Yes' -> Z ...{i}",
              flush=True)
        time.sleep(1)
    print(f"  GO ({DURATION_S:.0f}s) - note when you press the final Z", flush=True)

    # history[addr] = list of (sample_idx, value) at each change
    hist: dict[int, list[tuple[int, int]]] = {}
    last: dict[int, int] = dict(base)
    dt = 1.0 / SAMPLE_HZ
    n = int(DURATION_S * SAMPLE_HZ)
    t0 = time.perf_counter()
    for k in range(n):
        cur = snap()
        for a, v in cur.items():
            if a not in last or last[a] != v:
                hist.setdefault(a, []).append((k, v))
                last[a] = v
        target = t0 + (k + 1) * dt
        slack = target - time.perf_counter()
        if slack > 0:
            time.sleep(slack)

    print(f"\n--- {len(hist)} addresses changed over {n} samples "
          f"({SAMPLE_HZ}Hz) ---\n", flush=True)

    sticky, transient = [], []
    for a, changes in sorted(hist.items()):
        b = base.get(a)
        final = changes[-1][1]
        seq = ", ".join(f"{v}@{i}" for i, v in changes)
        row = (a, b, final, len(changes), seq)
        if final != b:
            sticky.append(row)
        else:
            transient.append(row)

    print(f"STICKY end-state ({len(sticky)}) - final != baseline:")
    for a, b, final, ncng, seq in sticky:
        print(f"  {a:#010x}: {b} -> {final}   ({ncng} changes: {seq})", flush=True)

    print(f"\nTRANSIENT ({len(transient)}) - blipped, returned to baseline "
          f"(request-flag candidates):")
    for a, b, final, ncng, seq in transient:
        print(f"  {a:#010x}: base {b}   ({ncng} changes: {seq})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
