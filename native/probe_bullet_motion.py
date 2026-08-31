"""Find the zBullet motion fields (velocity / speed / angle / accel / delay
timer) by tracking each bullet slot across frames and correlating struct floats
with the observed per-frame motion.

Attach to a th07.exe that's IN a boss fight (Cirno midboss or Letty). Best via
thprac (jump to Stage 1 boss) or just play there.

    .venv/Scripts/python native/probe_bullet_motion.py [frames]
"""
from __future__ import annotations

import struct
import sys
import time
from collections import defaultdict

import numpy as np
import pymem

BULLET_MANAGER = 0x0062F958
BM_BULLETS = 0x0000B8C0
BM_STRIDE = 0x00000D68
BM_MAX = 0x401
B_POS = 0xB8C            # float x, y, z
B_STATE = 0xBFC          # uint16
LIVE = (1, 2, 3, 4, 5)
LO, HI = 0xB40, 0xC60    # struct window we scan


def read_bullets(pm):
    blob = pm.read_bytes(BULLET_MANAGER + BM_BULLETS, BM_STRIDE * BM_MAX)
    out = {}
    for i in range(BM_MAX):
        o = i * BM_STRIDE
        st = struct.unpack_from("<H", blob, o + B_STATE)[0]
        if st not in LIVE:
            continue
        x, y = struct.unpack_from("<ff", blob, o + B_POS)
        if not (-80 < x < 480 and -80 < y < 560):
            continue
        out[i] = (x, y, blob[o + LO:o + HI])
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    pm = pymem.Pymem("th07.exe")
    print("tracking bullets... move around so patterns vary", flush=True)

    prev = {}
    # per struct-offset: how well does a float-pair there predict (vx, vy)?
    vel_err = defaultdict(list)     # offset -> list of |struct - actual| for pair
    spd_err = defaultdict(list)     # single float ~ speed
    ang_err = defaultdict(list)     # single float ~ angle
    samples = 0
    off_pairs = list(range(0, HI - LO - 8, 4))
    off_singles = list(range(0, HI - LO - 4, 4))

    for _ in range(n):
        try:
            cur = read_bullets(pm)
        except Exception:
            time.sleep(0.02)
            continue
        for idx, (x, y, buf) in cur.items():
            if idx not in prev:
                continue
            px, py, _ = prev[idx]
            vx, vy = x - px, y - py
            if abs(vx) > 20 or abs(vy) > 20 or (vx == 0 and vy == 0):
                continue
            spd = (vx * vx + vy * vy) ** 0.5
            ang = np.arctan2(vy, vx)
            samples += 1
            for o in off_pairs:
                a, b = struct.unpack_from("<ff", buf, o)
                if abs(a) < 25 and abs(b) < 25:
                    vel_err[o].append(abs(a - vx) + abs(b - vy))
            for o in off_singles:
                v = struct.unpack_from("<f", buf, o)[0]
                if 0 <= v < 25:
                    spd_err[o].append(abs(v - spd))
                if -7 < v < 7:
                    ang_err[o].append(min(abs(v - ang), abs(abs(v - ang) - 2 * np.pi)))
        prev = cur
        time.sleep(1 / 60)

    print(f"\n{samples} bullet-frame samples\n")

    def best(errs, label, thresh, minn):
        rows = [(o, np.mean(e), len(e)) for o, e in errs.items() if len(e) >= minn]
        rows.sort(key=lambda r: r[1])
        print(f"--- {label} (offset from +0x{LO:X}) ---")
        for o, m, c in rows[:6]:
            tag = "  <== MATCH" if m < thresh else ""
            print(f"  +0x{LO + o:03X}  mean err {m:.4f}  n={c}{tag}")
        print()

    best(vel_err, "velocity (vx,vy float pair)", 0.02, samples // 4)
    best(spd_err, "speed (single float)", 0.02, samples // 4)
    best(ang_err, "angle (single float, rad)", 0.02, samples // 4)

    # dump a couple of raw structs for manual inspection of accel / timer fields
    cur = read_bullets(pm)
    print("--- raw struct window of 2 live bullets ---")
    for k, (x, y, buf) in list(cur.items())[:2]:
        print(f"\n slot {k} pos ({x:.1f},{y:.1f})")
        for o in range(0, HI - LO, 4):
            iv = struct.unpack_from("<i", buf, o)[0]
            fv = struct.unpack_from("<f", buf, o)[0]
            fs = f"{fv:11.4f}" if abs(fv) < 1e6 else f"{fv:11.1e}"
            print(f"  +0x{LO + o:03X}: i={iv:<11d} f={fs}")


if __name__ == "__main__":
    main()
