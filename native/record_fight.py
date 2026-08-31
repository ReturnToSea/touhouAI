"""Record a real th07 boss fight frame-by-frame: every live bullet's
(x, y, vx, vy, class, fx_flag) + the boss position + the player position, keyed
by the game's own frame counter. This is the ground-truth pattern for sim
replay - the exact engine output, hangs / accel / curves and all.

Attach to a th07 that's IN the fight (thprac jump, or just play there).

    .venv/Scripts/python native/record_fight.py cirno   [seconds]
    .venv/Scripts/python native/record_fight.py letty

Writes sim/fights/<name>.npz : bullets (N x 8: frame,slot,x,y,vx,vy,cls,fxflag),
boss (F x 3: frame,x,y), player (F x 3), meta.
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import numpy as np
import pymem

BULLET_MANAGER = 0x0062F958
BM_BULLETS, BM_STRIDE, BM_MAX = 0x0000B8C0, 0x00000D68, 0x401
B_POS, B_VEL, B_ANGLE, B_STATE = 0xB8C, 0xB98, 0xBBC, 0xBFC
B_CLASS = 0xB8A
B_FXFLAG = 0xC3C
LIVE = (1, 2, 3, 4, 5)

ENEMY_MANAGER = 0x009A9B00
EM_BOSSES = 0x00954598          # &EM_BOSSES[0]
E_POS = 0x2B0C
PLAYER = 0x004BDAD8
PL_POS = 0x0930
GAME_MANAGER = 0x00626270
GM_STAGE_TIMER = 0x95E8         # per-stage frame counter


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fight"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    pm = pymem.Pymem("th07.exe")

    def rd_f(a):
        return struct.unpack("<f", pm.read_bytes(a, 4))[0]

    bullets, boss, player = [], [], []
    last_frame = -1
    t_end = time.time() + secs
    print(f"recording '{name}' for up to {secs:.0f}s - fight normally", flush=True)
    n_static = 0
    while time.time() < t_end:
        try:
            fr = struct.unpack("<i", pm.read_bytes(GAME_MANAGER + GM_STAGE_TIMER, 4))[0]
        except Exception:
            time.sleep(0.001)
            continue
        if fr == last_frame:
            n_static += 1
            time.sleep(0.0005)
            continue
        last_frame = fr

        blob = pm.read_bytes(BULLET_MANAGER + BM_BULLETS, BM_STRIDE * BM_MAX)
        cnt = 0
        for i in range(BM_MAX):
            o = i * BM_STRIDE
            st = struct.unpack_from("<H", blob, o + B_STATE)[0]
            if st not in LIVE:
                continue
            x, y = struct.unpack_from("<ff", blob, o + B_POS)
            if not (-100 < x < 500 and -100 < y < 580):
                continue
            vx, vy = struct.unpack_from("<ff", blob, o + B_VEL)
            cls = struct.unpack_from("<h", blob, o + B_CLASS)[0]
            fxf = struct.unpack_from("<i", blob, o + B_FXFLAG)[0]
            bullets.append((fr, i, x, y, vx, vy, cls, fxf))
            cnt += 1

        try:
            bptr = struct.unpack("<I", pm.read_bytes(ENEMY_MANAGER + EM_BOSSES, 4))[0]
            if 0x400000 < bptr < 0x7FFFFFFF:
                bx, by = struct.unpack("<ff", pm.read_bytes(bptr + E_POS, 8))
                boss.append((fr, bx, by))
        except Exception:
            pass
        try:
            px, py = struct.unpack("<ff", pm.read_bytes(PLAYER + PL_POS, 8))
            player.append((fr, px, py))
        except Exception:
            pass

        if fr % 300 == 0:
            print(f"  frame {fr}: {cnt} bullets, {len(bullets)} rows", flush=True)

    out = Path(__file__).resolve().parent.parent / "sim" / "fights"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.npz"
    np.savez_compressed(
        p,
        bullets=np.array(bullets, np.float32),
        boss=np.array(boss, np.float32),
        player=np.array(player, np.float32),
    )
    print(f"\nsaved {p}")
    b = np.array(bullets, np.float32)
    if len(b):
        print(f"  {len(b)} bullet-frames, frames {int(b[:,0].min())}..{int(b[:,0].max())}, "
              f"classes {sorted(set(b[:,6].astype(int)))[:10]}, "
              f"fx flags {sorted(set(b[:,7].astype(int)))}")


if __name__ == "__main__":
    main()
