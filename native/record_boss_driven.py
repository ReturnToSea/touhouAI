"""Launch a hooked th07, drive it with a sim policy until a boss is on screen,
then record the fight frame-by-frame (every live bullet's x,y,vx,vy,class,fxflag
+ boss + player). Reliable - the hook keeps the game ticking.

    .venv/Scripts/python native/record_boss_driven.py cirno   [--secs 60]
    .venv/Scripts/python native/record_boss_driven.py letty   --secs 90

`cirno` stops recording when the midboss dies; `letty` keeps going to the boss.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402

BM_BASE = 0x0062F958 + 0x0000B8C0
BM_STRIDE, BM_MAX = 0x00000D68, 0x401
B_POS, B_VEL, B_ANGLE, B_CLASS, B_STATE, B_FXFLAG = 0xB8C, 0xB98, 0xBBC, 0xB8A, 0xBFC, 0xC3C
LIVE = (1, 2, 3, 4, 5)
EM_BOSSES0 = 0x009A9B00 + 0x00954598
E_POS, E_LIFE = 0x2B0C, 0x2BB8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--secs", type=float, default=75)
    ap.add_argument("--policy", default="runs_sim/ppo_v29/snap_0092M.pt")
    args = ap.parse_args()

    pol = MLPPolicy.load(HERE.parent / args.policy)
    env = Th07Env(frame_skip=1, max_seconds=400, render=False, dll_obs=True)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)
    obs, _ = env.reset()

    def boss():
        try:
            p = struct.unpack("<I", pm.read_bytes(EM_BOSSES0, 4))[0]
            if not (0x400000 < p < 0x7FFFFFFF):
                return None
            x, y = struct.unpack("<ff", pm.read_bytes(p + E_POS, 8))
            life = struct.unpack("<i", pm.read_bytes(p + E_LIFE, 4))[0]
            if -80 < x < 480 and -80 < y < 520:
                return x, y, life
        except Exception:
            pass
        return None

    bullets, bosslog, playerlog = [], [], []
    recording = False
    max_steps = int(args.secs * 60) + 4000
    step = 0
    seen_boss = False
    while step < max_steps:
        obs, r, term, trunc, info = env.step(int(pol.act(obs)))
        step += 1
        b = boss()
        if b and not recording:
            recording = True
            seen_boss = True
            f0 = int(info.get("frame", step))
            print(f"boss up at game-frame {f0} - recording", flush=True)
        if recording:
            if b is None:
                if seen_boss:
                    print("boss gone - done", flush=True)
                    break
            else:
                bosslog.append((step, b[0], b[1]))
            s = env.h.s
            playerlog.append((step, s.player_x, s.player_y))
            blob = pm.read_bytes(BM_BASE, BM_STRIDE * BM_MAX)
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
                bullets.append((step, i, x, y, vx, vy, cls, fxf))
            if step % 300 == 0:
                print(f"  step {step}: {len(bullets)} rows", flush=True)
        if term or trunc:
            print("player died / episode end", flush=True)
            break

    env.close()
    out = HERE.parent / "sim" / "fights"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{args.name}.npz"
    np.savez_compressed(p, bullets=np.array(bullets, np.float32),
                        boss=np.array(bosslog, np.float32),
                        player=np.array(playerlog, np.float32))
    b = np.array(bullets, np.float32)
    print(f"\nsaved {p}")
    if len(b):
        fr = b[:, 0]
        print(f"  {len(b)} bullet-frames over {int(fr.max() - fr.min())} frames "
              f"(~{(fr.max()-fr.min())/60:.0f}s), avg {len(b)/len(set(fr)):.0f}/frame")
        print(f"  classes {sorted(set(b[:,6].astype(int)))[:12]}")
        print(f"  fx flags {sorted(set(b[:,7].astype(int)))}")


if __name__ == "__main__":
    main()
