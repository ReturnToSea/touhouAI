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
B_HITBOX, B_POS, B_VEL, B_CLASS, B_STATE, B_FXFLAG = 0xB7C, 0xB8C, 0xB98, 0xB8A, 0xBFC, 0xC3C
LIVE = (1, 2, 3, 4, 5)

# one strided view over the whole bullet pool - lets us pull every live bullet
# per frame with numpy instead of 1025 x N struct.unpack_from calls (~10x faster,
# which is most of the recording-time cost).
BULLET_DT = np.dtype({
    "names":    ["cls", "hb", "pos", "vel", "state", "fxf"],
    "formats":  ["<i2", "<2f4", "<2f4", "<2f4", "<u2", "<i4"],
    "offsets":  [B_CLASS, B_HITBOX, B_POS, B_VEL, B_STATE, B_FXFLAG],
    "itemsize": BM_STRIDE,
})

EM = 0x009A9B00
EM_ENEMIES, EM_STRIDE, EM_MAX = 0x00004F50, 0x00004F48, 0x1E1
EM_SCAN = 128            # only the low slots are ever used on stage 1; the full
#                          481-slot pool is ~10MB/frame to read, this is ~2.6MB
EM_BOSSES0 = EM + 0x00954598
E_POS, E_HITBOX, E_LIFE = 0x2B0C, 0x2B3C, 0x2BB8
# strided view over the enemy pool for the per-frame satellite scan
ENEMY_DT = np.dtype({
    "names":    ["pos", "hb", "life"],
    "formats":  ["<3f4", "<3f4", "<i4"],
    "offsets":  [E_POS, E_HITBOX, E_LIFE],
    "itemsize": EM_STRIDE,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--secs", type=float, default=600,
                    help="hard safety cap on recording length (s). The recorder "
                    "normally stops on its own when the boss despawns; this is "
                    "just a backstop so a stuck run can't record forever.")
    ap.add_argument("--policy", default="runs_sim/ppo_v29/snap_0092M.pt")
    ap.add_argument("--shoot", choices=("auto", "off", "on"), default="auto",
                    help="off = dodge only (phases run to their full timer)")
    ap.add_argument("--dodge-after-boss", action="store_true",
                    help="shoot to reach the boss, then dodge-only so all its "
                    "phases play their full timer (no damage-phasing)")
    ap.add_argument("--which", type=int, default=1,
                    help="which EM_BOSSES[0] appearance to record: 1 Cirno "
                    "(S1 mid), 2 Letty (S1 boss), 3 Chen (S2 mid), 4 Chen (S2 boss)")
    ap.add_argument("--godmode", action="store_true",
                    help="player can't die (RECORDING ONLY) - lets a weak driver "
                    "reach a Stage 2+ boss. The trained policy never sees this.")
    ap.add_argument("--n", type=int, default=1, help="record N runs -> <name>_0.npz ...")
    args = ap.parse_args()

    pol = MLPPolicy.load(HERE.parent / args.policy)
    env = Th07Env(frame_skip=1, max_seconds=max(args.secs + 120, 400),
                  render=False, dll_obs=True, godmode=args.godmode)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)

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

    out = HERE.parent / "sim" / "fights"
    out.mkdir(parents=True, exist_ok=True)
    for run in range(args.n):
        _record_one(env, pm, pol, boss, args, out, run)
    env.close()


def _record_one(env, pm, pol, boss, args, out, run):
    obs, _ = env.reset(options={"hard": True})
    frames, enemyframes, bosslog, playerlog = [], [], [], []
    recording = False
    max_steps = int(args.secs * 60) + 6000
    step = 0
    appearance = 0                 # how many distinct bosses we've seen
    boss_present = False
    null_run = 999                 # consecutive no-boss frames
    gone = 0
    while step < max_steps:
        a = int(pol.act(obs))
        if args.shoot == "off" or (recording and args.dodge_after_boss):
            a = a % 18
        elif args.shoot == "on":
            a = a % 18 + 18
        obs, r, term, trunc, info = env.step(a)
        step += 1
        b = boss()
        if b is None:
            null_run += 1
            if null_run > 90:
                boss_present = False
        else:
            if not boss_present and null_run > 90:     # a NEW boss appeared
                appearance += 1
                if appearance == args.which and not recording:
                    recording = True
                    print(f"boss #{appearance} up at step {step} - recording"
                          f"{' (dodge-only)' if args.dodge_after_boss else ''}",
                          flush=True)
            boss_present = True
            null_run = 0
        if recording:
            if b is None:
                gone += 1
                if gone > 90:                     # ~1.5s null -> this boss done
                    print(f"boss gone at step {step} - done", flush=True)
                    break
            else:
                gone = 0
                bosslog.append((step, b[0], b[1]))
            s = env.h.s
            playerlog.append((step, s.player_x, s.player_y))

            blob = pm.read_bytes(BM_BASE, BM_STRIDE * BM_MAX)
            arr = np.frombuffer(blob, dtype=BULLET_DT, count=BM_MAX)
            pos = arr["pos"]
            keep = ((arr["state"] >= 1) & (arr["state"] <= 5) &
                    (pos[:, 0] > -100) & (pos[:, 0] < 500) &
                    (pos[:, 1] > -100) & (pos[:, 1] < 580))
            sel = np.nonzero(keep)[0]
            if sel.size:
                fr = np.empty((sel.size, 10), np.float32)
                fr[:, 0] = step
                fr[:, 1] = sel
                fr[:, 2:4] = pos[sel]
                fr[:, 4:6] = arr["vel"][sel]
                fr[:, 6] = arr["cls"][sel]
                fr[:, 7] = arr["fxf"][sel]
                fr[:, 8:10] = arr["hb"][sel]          # AABB full size (x, y)
                frames.append(fr)

            # satellite sub-enemies (Letty's orbs etc.) - they contact-kill the
            # player and FightSim otherwise ignores them entirely
            eblob = pm.read_bytes(EM + EM_ENEMIES, EM_STRIDE * EM_SCAN)
            en = np.frombuffer(eblob, dtype=ENEMY_DT, count=EM_SCAN)
            epos = en["pos"]
            ek = np.nonzero((en["life"] != 0) &
                            (epos[:, 0] > -60) & (epos[:, 0] < 440) &
                            (epos[:, 1] > -40) & (epos[:, 1] < 500))[0]
            if ek.size:
                ef = np.empty((ek.size, 8), np.float32)
                ef[:, 0] = step
                ef[:, 1] = ek
                ef[:, 2:4] = epos[ek][:, :2]
                ef[:, 4] = en["life"][ek]
                ef[:, 5:8] = en["hb"][ek]             # ECL enemy_set_hitbox args
                enemyframes.append(ef)

            if step % 600 == 0:
                nrows = sum(len(f) for f in frames)
                ne = sum(len(f) for f in enemyframes)
                print(f"  step {step}: {nrows} bullet rows, {ne} enemy rows",
                      flush=True)
        if term or trunc:
            print(f"  run {run}: player died / episode end at step {step}", flush=True)
            break

    b = (np.concatenate(frames) if frames
         else np.zeros((0, 10), np.float32))
    e = (np.concatenate(enemyframes) if enemyframes
         else np.zeros((0, 8), np.float32))
    p = out / (f"{args.name}_{run}.npz" if args.n > 1 else f"{args.name}.npz")
    np.savez_compressed(p, bullets=b, enemies=e,
                        boss=np.array(bosslog, np.float32),
                        player=np.array(playerlog, np.float32))
    if len(b):
        fr = b[:, 0]
        elethal = e[(e[:, 5] > 0.5) & (e[:, 6] > 0.5)] if len(e) else e
        print(f"  saved {p.name}: {len(b)} bullet rows / {int(fr.max()-fr.min())}f "
              f"(~{(fr.max()-fr.min())/60:.0f}s) / avg {len(b)/len(set(fr)):.0f}/f "
              f"/ fx {sorted(set(b[:,7].astype(int)))} / hb {sorted({round(float(v),1) for v in b[:,8]})[:6]} "
              f"/ {len(e)} enemy rows ({len(elethal)} lethal), "
              f"enemy hb {sorted({(round(float(r[5]),0),round(float(r[6]),0)) for r in e})[:6] if len(e) else []}",
              flush=True)
    else:
        print(f"  run {run}: NO BULLETS recorded (boss not reached?)", flush=True)


if __name__ == "__main__":
    main()
