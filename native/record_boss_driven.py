"""Launch a hooked th07, drive it with a sim policy until a boss is on screen,
then record the fight frame-by-frame. Reliable - the hook keeps the game ticking.

Per live bullet, per frame (the `bullets` array, one row each):
    0 step  1 slot  2,3 x,y  4,5 vx,vy  6 class  7 fx_flag  8,9 hitbox_x,y
    10 speed  11 accel  12 ang_vel  13 angle  14,15,16 bullet_effects p1/p2/interval
Offsets are all in native/th07_addrs.h. Cols 0-9 are stable (older recordings
have only those); 10-16 were added for the ECL-VM motion models.

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
#   full motion state (all in th07_addrs.h, offsets verified via probe_bullet_motion):
B_SPEED, B_ACCEL, B_ANGVEL, B_ANGLE = 0xBB0, 0xBB4, 0xBB8, 0xBBC
B_FX_P1, B_FX_P2, B_FX_INT = 0xC2C, 0xC30, 0xC34   # == bullet_effects staging entry 1

# --- from the th07.exe RE (docs/th07-re-notes.md) -------------------------
#   B_TFLAG  = type-word flags: bits 0x2/0x4/0x8 = hang (spawn state 2/3/4)
#   B_AFLAG  = live active fx-flag word (0x10 dir-accel, 0x20 turn+accel,
#              0x40 pause-redirect, 0x80 pause-reaim, 0xc00 wall-bounce)
#   B_EIDX   = how many staging entries FUN_00424290 has processed
#   B_STG    = the 5 bullet_effects staging entries, 6 floats each
#              [p1, p2, interval, repeat, flag(int-as-float), gate]
B_TFLAG, B_AFLAG, B_YOUNG, B_EIDX, B_STG = 0xBF6, 0xBF4, 0xBF0, 0xC10, 0xC14
NSTG = 5 * 6                       # 30 floats
LIVE = (1, 2, 3, 4, 5)
NBCOL = 22 + NSTG                 # per-bullet row width (see the assembly below)

# one strided view over the whole bullet pool - lets us pull every live bullet
# per frame with numpy instead of 1025 x N struct.unpack_from calls (~10x faster,
# which is most of the recording-time cost).
BULLET_DT = np.dtype({
    "names":    ["cls", "hb", "pos", "vel", "state", "fxf",
                 "speed", "accel", "angvel", "angle", "fxp1", "fxp2", "fxint",
                 "tflag", "aflag", "young", "eidx", "stg"],
    "formats":  ["<i2", "<2f4", "<2f4", "<2f4", "<u2", "<i4",
                 "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<i4",
                 "<u2", "<u2", "<i4", "<i4", f"<{NSTG}f4"],
    "offsets":  [B_CLASS, B_HITBOX, B_POS, B_VEL, B_STATE, B_FXFLAG,
                 B_SPEED, B_ACCEL, B_ANGVEL, B_ANGLE, B_FX_P1, B_FX_P2, B_FX_INT,
                 B_TFLAG, B_AFLAG, B_YOUNG, B_EIDX, B_STG],
    "itemsize": BM_STRIDE,
})
assert B_STG + NSTG * 4 <= BM_STRIDE

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
    ap.add_argument("--secs", type=float, default=800,
                    help="hard safety cap on the whole drive-through (s). Each "
                    "boss normally stops on its own when it despawns; this is "
                    "just a backstop. Multi-boss to Chen needs ~600+.")
    ap.add_argument("--policy", default="runs_sim/ppo_v29/snap_0092M.pt")
    ap.add_argument("--shoot", choices=("auto", "off", "on"), default="auto",
                    help="off = dodge only (phases run to their full timer)")
    ap.add_argument("--dodge-after-boss", action="store_true",
                    help="shoot to reach the boss, then dodge-only so all its "
                    "phases play their full timer (no damage-phasing)")
    ap.add_argument("--which", default="1",
                    help="EM_BOSSES[0] appearance(s) to record, comma-separated: "
                    "1 Cirno (S1 mid), 2 Letty (S1 boss), 3 Chen (S2 mid), "
                    "4 Chen (S2 boss). Multiple -> one drive-through records each "
                    "to its own <bossname>_<run>.npz (needs --godmode past S1).")
    ap.add_argument("--godmode", action="store_true",
                    help="player can't die (RECORDING ONLY) - lets a weak driver "
                    "reach a Stage 2+ boss. The trained policy never sees this.")
    ap.add_argument("--n", type=int, default=1, help="record N drive-throughs")
    ap.add_argument("--maxpower", action="store_true",
                    help="lock player power to 128 (RECORDING ONLY) - the drive "
                    "policy doesn't collect power items, but a 1CC run is at max "
                    "power for the Stage-1 boss, so DPS must be measured there")
    args = ap.parse_args()
    args.which_list = [int(w) for w in str(args.which).split(",")]

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


BOSS_NAME = {1: "cirno", 2: "letty", 3: "chenmid", 4: "chen"}


def _record_one(env, pm, pol, boss, args, out, run):
    obs, _ = env.reset(options={"hard": True})
    want = list(args.which_list)          # appearances still to record
    frames, enemyframes, bosslog, playerlog = [], [], [], []
    recording = False
    cur_which = None
    max_steps = int(args.secs * 60) + 6000
    step = 0
    appearance = 0                 # how many distinct bosses we've seen
    boss_present = False
    null_run = 999                 # consecutive no-boss frames
    gone = 0
    while step < max_steps and want:
        a = int(pol.act(obs))
        if args.shoot == "off" or (recording and args.dodge_after_boss):
            a = a % 18
        elif args.shoot == "on":
            a = a % 18 + 18
        obs, r, term, trunc, info = env.step(a)
        step += 1
        if args.maxpower:
            try:
                g = struct.unpack("<I", pm.read_bytes(0x00626270 + 8, 4))[0]
                pm.write_bytes(g + 0x7C, struct.pack("<f", 128.0), 4)
            except Exception:
                pass
        b = boss()
        if b is None:
            null_run += 1
            if null_run > 90:
                boss_present = False
        else:
            if not boss_present and null_run > 90:     # a NEW boss appeared
                appearance += 1
                if appearance in want and not recording:
                    recording = True
                    cur_which = appearance
                    print(f"boss #{appearance} ({BOSS_NAME.get(appearance,'?')}) "
                          f"up at step {step} - recording"
                          f"{' (dodge-only)' if args.dodge_after_boss else ''}",
                          flush=True)
            boss_present = True
            null_run = 0
        if recording:
            if b is None:
                gone += 1
            else:
                gone = 0
                bosslog.append((step, b[0], b[1], b[2]))     # b[2] = boss HP (+0x2BB8)
            s = env.h.s
            playerlog.append((step, s.player_x, s.player_y,
                              getattr(s, "power", 0)))         # player power for DPS-vs-power

            blob = pm.read_bytes(BM_BASE, BM_STRIDE * BM_MAX)
            arr = np.frombuffer(blob, dtype=BULLET_DT, count=BM_MAX)
            pos = arr["pos"]
            keep = ((arr["state"] >= 1) & (arr["state"] <= 5) &
                    (pos[:, 0] > -100) & (pos[:, 0] < 500) &
                    (pos[:, 1] > -100) & (pos[:, 1] < 580))
            sel = np.nonzero(keep)[0]
            if sel.size:
                fr = np.empty((sel.size, NBCOL), np.float32)
                fr[:, 0] = step
                fr[:, 1] = sel
                fr[:, 2:4] = pos[sel]
                fr[:, 4:6] = arr["vel"][sel]
                fr[:, 6] = arr["cls"][sel]
                fr[:, 7] = arr["fxf"][sel]
                fr[:, 8:10] = arr["hb"][sel]          # AABB full size (x, y)
                fr[:, 10] = arr["speed"][sel]         # cols 10-16: full motion state
                fr[:, 11] = arr["accel"][sel]
                fr[:, 12] = arr["angvel"][sel]
                fr[:, 13] = arr["angle"][sel]         # radians
                fr[:, 14] = arr["fxp1"][sel]          # bullet_effects redirect angle / accel
                fr[:, 15] = arr["fxp2"][sel]          # bullet_effects redirect speed (-999 = keep)
                fr[:, 16] = arr["fxint"][sel]         # bullet_effects interval / duration
                fr[:, 17] = arr["state"][sel]         # 1 live, 2/3/4 hang, 5 dying
                fr[:, 18] = arr["tflag"][sel]         # type-word flags (hang bits 0x2/4/8)
                fr[:, 19] = arr["aflag"][sel]         # live active fx-flag word
                fr[:, 20] = arr["young"][sel]         # +0xBF0 young countdown
                fr[:, 21] = arr["eidx"][sel]          # staging entries processed
                fr[:, 22:22 + NSTG] = arr["stg"][sel]  # 5 x [p1,p2,int,rep,flag,gate]
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

            if gone > 90:                          # ~1.5s null -> this boss done
                print(f"  {BOSS_NAME.get(cur_which,'?')} gone at step {step}",
                      flush=True)
                _save(out, args, run, cur_which,
                      frames, enemyframes, bosslog, playerlog)
                want.remove(cur_which)
                frames, enemyframes, bosslog, playerlog = [], [], [], []
                recording, cur_which, gone = False, None, 0

            if step % 900 == 0:
                nrows = sum(len(f) for f in frames)
                print(f"  step {step}: {nrows} bullet rows", flush=True)
        if term or trunc:
            print(f"  run {run}: episode end at step {step} "
                  f"(still wanted: {want})", flush=True)
            break

    if recording:                     # loop hit max_steps mid-fight - keep it
        print(f"  run {run}: stopped at step {step}, saving partial {cur_which}",
              flush=True)
        _save(out, args, run, cur_which,
              frames, enemyframes, bosslog, playerlog)


def _save(out, args, run, which, frames, enemyframes, bosslog, playerlog):
    b = np.concatenate(frames) if frames else np.zeros((0, NBCOL), np.float32)
    e = np.concatenate(enemyframes) if enemyframes else np.zeros((0, 8), np.float32)
    stem = BOSS_NAME.get(which, f"b{which}")
    if len(args.which_list) == 1:
        stem = args.name          # single-boss: keep the caller's name
    p = out / f"{stem}_{run}.npz"
    np.savez_compressed(p, bullets=b, enemies=e,
                        boss=np.array(bosslog, np.float32),
                        player=np.array(playerlog, np.float32))
    if len(b):
        fr = b[:, 0]
        eleth = e[(e[:, 5] > 0.5) & (e[:, 6] > 0.5)] if len(e) else e
        print(f"  saved {p.name}: {len(b)} rows / {int(fr.max()-fr.min())}f "
              f"(~{(fr.max()-fr.min())/60:.0f}s) / avg {len(b)/len(set(fr)):.0f}/f "
              f"/ hb {sorted({round(float(v),1) for v in b[:,8]})[:6]} "
              f"/ {len(e)} enemy ({len(eleth)} lethal) "
              f"hb {sorted({(round(float(r[5])),round(float(r[6]))) for r in e})[:6] if len(e) else []}",
              flush=True)
    else:
        print(f"  {p.name}: NO BULLETS", flush=True)


if __name__ == "__main__":
    main()
