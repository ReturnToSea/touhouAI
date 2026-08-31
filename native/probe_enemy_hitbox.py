"""Reach Letty, wait for the spell-phase satellite burst (EM_ENEMY_COUNT high),
dump a live satellite's struct + Letty's struct to locate the hitbox +
collidable fields. Retries the whole fight until it catches the burst.

    .venv/Scripts/python native/probe_enemy_hitbox.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402

EM = 0x009A9B00
EM_ENEMIES, STRIDE = 0x00004F50, 0x00004F48
EM_COUNT = 0x009545BC
EM_BOSSES = 0x00954598
E_POS, E_LIFE = 0x2B0C, 0x2BB8

BM_BASE = 0x0062F958 + 0x0000B8C0
BM_STRIDE, BM_MAX = 0x00000D68, 0x401
B_POS, B_STATE = 0xB8C, 0xBFC


def u32(pm, a):
    return struct.unpack("<I", pm.read_bytes(a, 4))[0]


def dump(pm, base, lo, hi, label):
    print(f"    {label}  [+{lo:#06x}..+{hi:#06x}]")
    blob = pm.read_bytes(base + lo, hi - lo)
    for o in range(0, len(blob), 16):
        row = blob[o:o + 16]
        fl = " ".join(f"{struct.unpack_from('<f', row, k)[0]:>10.3f}"
                      for k in range(0, len(row) - 3, 4))
        print(f"      +{lo + o:#06x}: {row.hex(' '):<47}  f: {fl}")


def scan(pm, base, span, wants):
    blob = pm.read_bytes(base, span)
    for o in range(0, len(blob) - 3, 4):
        v = struct.unpack_from("<f", blob, o)[0]
        if any(abs(v - w) < 0.02 for w in wants):
            iv = struct.unpack_from("<i", blob, o)[0]
            print(f"      f {v:9.3f} @ +{o:#06x}")


def main():
    pol = MLPPolicy.load(HERE.parent / "runs_sim/ppo_v29/snap_0092M.pt")
    env = Th07Env(frame_skip=1, max_seconds=400, render=False, dll_obs=True)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)

    def boss0():
        p = u32(pm, EM + EM_BOSSES)
        if 0x400000 < p < 0x7FFFFFFF:
            try:
                x, y = struct.unpack("<ff", pm.read_bytes(p + E_POS, 8))
                if -80 < x < 480 and -80 < y < 520:
                    return p, x, y
            except Exception:
                pass
        return None

    for attempt in range(8):
        obs, _ = env.reset(options={"hard": True})
        step, appear, present, nullrun, at_letty = 0, 0, False, 999, False
        while step < 20000:
            obs, r, term, trunc, info = env.step(int(pol.act(obs)))
            step += 1
            b = boss0()
            if b is None:
                nullrun += 1
                if nullrun > 90:
                    present = False
            else:
                if not present and nullrun > 90:
                    appear += 1
                present = True
                nullrun = 0
            if appear >= 2:
                at_letty = True
            if term or trunc:
                break
            if not at_letty:
                continue

            cnt = struct.unpack("<i", pm.read_bytes(EM + EM_COUNT, 4))[0]
            if cnt < 8:
                continue

            # burst! grab live satellites
            bpa = b[0] if b else u32(pm, EM + EM_BOSSES)
            sats = []
            for i in range(80):
                ea = EM + EM_ENEMIES + i * STRIDE
                x, y, z = struct.unpack("<fff", pm.read_bytes(ea + E_POS, 12))
                life = struct.unpack("<i", pm.read_bytes(ea + E_LIFE, 4))[0]
                if ea != bpa and life == 1 and -40 < x < 420 and 0 < y < 440:
                    sats.append((i, ea, x, y))
            if len(sats) < 3:
                continue

            print(f"\n### burst at step {step} (attempt {attempt}): "
                  f"EM_COUNT={cnt}, {len(sats)} live satellites ###")
            for (i, ea, x, y) in sats[:4]:
                print(f"\n-- satellite slot {i} @ {ea:#010x} pos=({x:.1f},{y:.1f})")
                scan(pm, ea, STRIDE, (8.0, 4.0, 2.6667, 5.3333, 16.0, 32.0))
                dump(pm, ea, 0x2B00, 0x2BC8, "struct POS(+0x2B0C) LIFE(+0x2BB8)")
            print(f"\n-- BOSS Letty @ {bpa:#010x}")
            scan(pm, bpa, STRIDE, (4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0))
            dump(pm, bpa, 0x2B00, 0x2BC8, "boss struct")

            # bullet hitbox
            blob = pm.read_bytes(BM_BASE, BM_STRIDE * BM_MAX)
            for i in range(BM_MAX):
                o = i * BM_STRIDE
                if struct.unpack_from("<H", blob, o + B_STATE)[0] in (1, 2, 3, 4, 5):
                    x, y = struct.unpack_from("<ff", blob, o + B_POS)
                    if 0 < x < 380 and 0 < y < 440:
                        cls = struct.unpack_from("<h", blob, o + 0xB8A)[0]
                        print(f"\n-- bullet slot {i} cls={cls} pos=({x:.1f},{y:.1f})")
                        dump(pm, BM_BASE + o, 0xB70, 0xB90, "bullet hitbox region")
                        break
            env.close()
            return
        print(f"  attempt {attempt}: no burst (died/ended at step {step})")

    print("never caught the satellite burst")
    env.close()


if __name__ == "__main__":
    main()
