"""Verify the th07 ItemManager layout (from exphp-share/th-re-data) against the
live game: launch hooked, autonav into stage 1, hold SHOOT so fairies die and
drop power items, then walk ITEM_MANAGER.items[] and dump active entries.

    .venv\\Scripts\\python native\\probe_items.py

Layout under test (th07 v1.00b):
  ITEM_MANAGER   = 0x00575C70   (struct zItemManager)
  zItemManager:  items  = +0x0        zItem[0x44C]
                 next_index = +0xAE2E8 (int32)
                 item_count = +0xAE2EC (int32)
  zItem (stride 0x288):
                 pos     = +0x24C     (float x, y, z)
                 velocity= +0x258
                 item_type = +0x27C   (uint8)
                 in_use    = +0x27D   (uint8)
                 state     = +0x27F   (uint8)
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pymem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env  # noqa: E402

ITEM_MANAGER = 0x00575C70
ITEMS_OFF = 0x0
STRIDE = 0x288
N_SLOTS = 0x44C
NEXT_INDEX_OFF = 0xAE2E8
ITEM_COUNT_OFF = 0xAE2EC

I_POS = 0x24C
I_VEL = 0x258
I_TYPE = 0x27C
I_INUSE = 0x27D
I_STATE = 0x27F

TYPE_NAMES = {0: "P-small", 1: "point", 2: "P-big", 3: "bomb", 4: "full-power",
              5: "1up", 6: "star", 7: "cherry", 8: "cherry-petal", 9: "cherry-bullet"}


def scan(pm, want_raw=False):
    mgr = ITEM_MANAGER
    item_count = pm.read_int(mgr + ITEM_COUNT_OFF)
    next_index = pm.read_int(mgr + NEXT_INDEX_OFF)
    blob = pm.read_bytes(mgr + ITEMS_OFF, STRIDE * N_SLOTS)
    import struct
    active = []
    for i in range(N_SLOTS):
        b = i * STRIDE
        in_use = blob[b + I_INUSE]
        if not in_use:
            continue
        x, y, z = struct.unpack_from("<fff", blob, b + I_POS)
        vx, vy, vz = struct.unpack_from("<fff", blob, b + I_VEL)
        t = blob[b + I_TYPE]
        st = blob[b + I_STATE]
        active.append((i, t, st, x, y, vx, vy))
    return item_count, next_index, active


def main():
    env = Th07Env(frame_skip=1, max_seconds=600)
    pm = pymem.Pymem()
    pm.open_process_from_id(env.pid)
    print(f"attached to pid {env.pid}")

    env.reset()
    seen_types = Counter()
    max_active = 0
    samples = []
    field_count_matches = 0
    checks = 0

    A_SHOOT = 18          # dir 0, focus 0, shoot 1
    A_SHOOT_LEFT = 18 + 7  # nudge so we don't instantly die sitting still
    for step in range(1500):
        a = A_SHOOT if (step // 40) % 2 == 0 else (18 + 3)  # alternate shoot / shoot+right
        obs, r, term, trunc, info = env.step(a)
        try:
            cnt, nxt, active = scan(pm)
        except Exception as e:
            print(f"  read failed @step {step}: {e}")
            if term or trunc:
                env.reset()
            continue
        checks += 1
        if len(active) == cnt:
            field_count_matches += 1
        max_active = max(max_active, len(active))
        for (idx, t, st, x, y, vx, vy) in active:
            seen_types[t] += 1
            if len(samples) < 40:
                samples.append((step, idx, t, st, round(x, 1), round(y, 1),
                                round(vx, 2), round(vy, 2)))
        if step % 150 == 0:
            print(f"step {step:4d}  field item_count={cnt:3d} next_index={nxt:4d}  "
                  f"active_slots={len(active):3d}  score={info.get('score')}")
        if term or trunc:
            env.reset()

    print("\n=== ItemManager probe result ===")
    print(f"checks: {checks}   field item_count == active-slot count: "
          f"{field_count_matches}/{checks}")
    print(f"max active items at once: {max_active}")
    print("item_type histogram (raw byte -> name -> frames-seen):")
    for t, n in seen_types.most_common():
        print(f"  {t:3d}  {TYPE_NAMES.get(t, '???'):14s}  {n}")
    print("\nsample active items (step, slot, type, state, x, y, vx, vy):")
    for s in samples:
        print("  ", s)

    x_ok = all(0 - 40 <= s[4] <= 384 + 40 for s in samples)
    y_ok = all(-80 <= s[5] <= 448 + 60 for s in samples)
    print(f"\npositions in/near playfield:  x_ok={x_ok}  y_ok={y_ok}")
    env.close()


if __name__ == "__main__":
    main()
