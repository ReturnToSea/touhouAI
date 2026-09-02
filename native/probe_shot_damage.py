"""Measure ReimuA's shot damage on a boss - drive to Letty, shoot continuously
through her first non-spell (not armored), log boss HP + player power each
frame. Reports damage/frame and damage/frame per power unit, for FightSim's
synthetic phasing.

    .venv/Scripts/python native/probe_shot_damage.py [--which 2] [--power -1]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402

EM = 0x009A9B00
EM_BOSSES0 = EM + 0x00954598
E_LIFE = 0x2BB8
GAME_MANAGER = 0x00626270
GM_GLOBALS_PTR = 0x08
G_POWER = 0x7C
BTN_SHOOT = 0x01
FOCUS_SHOOT = 0x01 | 0x04


# PLAYER (0x4BDAD8) + 0xb7e70 -> a fixed global holding the shot-table root ptr
# (unfocused); +0xb7e74 is the focused root.  FUN_0043d160 walks root+0x34 as
# (u32 table, i32 power_threshold) pairs, advancing while threshold <= power, then
# fires 52-byte descriptor entries until a leading i16 period < 0.  Damage is the
# i16 at entry+0x1c (FUN_0043bbd0: -> shot+0x348, summed per frame, /2 focused,
# clamped 70/frame in FUN_00420620).
SHOT_ROOT_UNFOCUSED = 0x00575948
SHOT_ROOT_FOCUSED = 0x0057594C
_ENTRY = 52


def _dump_shot_table(pm):
    import struct as _s
    for name, root_addr in (("UNFOCUSED", SHOT_ROOT_UNFOCUSED),
                            ("FOCUSED", SHOT_ROOT_FOCUSED)):
        try:
            root = _s.unpack("<I", pm.read_bytes(root_addr, 4))[0]
            if not (0x400000 < root < 0x7FFFFFFF):
                print(f"  [{name}] root ptr {root:#x} not mapped"); continue
            pairs = root + 0x34
            print(f"  === {name} shot table  (root {root:#x}) ===")
            for pi in range(16):
                tbl, thr = _s.unpack("<Ii", pm.read_bytes(pairs + pi * 8, 8))
                if not (0x400000 < tbl < 0x7FFFFFFF):
                    break
                blob = pm.read_bytes(tbl, _ENTRY * 96)
                dmg_per_frame = 0.0
                lines = []
                for ei in range(96):
                    off = ei * _ENTRY
                    period, phase = _s.unpack_from("<hh", blob, off)
                    if period < 0:
                        break
                    xoff, yoff = _s.unpack_from("<ff", blob, off + 4)
                    speed = _s.unpack_from("<f", blob, off + 0x18)[0]
                    dmg = _s.unpack_from("<h", blob, off + 0x1c)[0]
                    muzzle, btype = _s.unpack_from("<BB", blob, off + 0x1e)
                    per = period if period else 1
                    dmg_per_frame += dmg / per
                    lines.append(f"      e{ei}: every {period:>3}f@{phase:<2} "
                                 f"dmg {dmg:>4}  off=({xoff:+.0f},{yoff:+.0f}) "
                                 f"spd {speed:.1f} muzzle {muzzle} type {btype}")
                print(f"   power < {thr:<5}: {len(lines)} shots, "
                      f"{dmg_per_frame:.2f} dmg/frame  (= {dmg_per_frame*60:.0f} DPS "
                      f"uncapped, {min(dmg_per_frame,70)*60:.0f} capped)")
                for ln in lines:
                    print(ln)
        except Exception as e:
            print(f"  [{name}] dump failed: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, default=2, help="1 Cirno, 2 Letty, ...")
    ap.add_argument("--power", type=float, default=-1,
                    help="pin power to this (0-128) via godmode's life write path; "
                    "-1 = leave as-is and just record it")
    ap.add_argument("--focus", action="store_true", help="focused shot")
    ap.add_argument("--dumponly", action="store_true",
                    help="dump the shot table and exit (no fight loop)")
    ap.add_argument("--camp", action="store_true",
                    help="ignore the policy: park under the boss and full-auto "
                    "(unfocused). NOTE: player_x tracking is unreliable here "
                    "(reads stale) - prefer --dumponly for the exact table")
    ap.add_argument("--policy", default="runs_sim/ppo_v29/snap_0092M.pt")
    args = ap.parse_args()

    pol = MLPPolicy.load(HERE.parent / args.policy)
    env = Th07Env(frame_skip=1, max_seconds=400, render=False, dll_obs=True,
                  godmode=True)                 # godmode: survive to the boss
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)

    def boss_hp():
        try:
            p = struct.unpack("<I", pm.read_bytes(EM_BOSSES0, 4))[0]
            if 0x400000 < p < 0x7FFFFFFF:
                return struct.unpack("<i", pm.read_bytes(p + E_LIFE, 4))[0]
        except Exception:
            pass
        return None

    def power():
        try:
            g = struct.unpack("<I", pm.read_bytes(GAME_MANAGER + GM_GLOBALS_PTR, 4))[0]
            return struct.unpack("<f", pm.read_bytes(g + G_POWER, 4))[0]
        except Exception:
            return -1.0

    def boss_ptr():
        p = struct.unpack("<I", pm.read_bytes(EM_BOSSES0, 4))[0]
        return p if 0x400000 < p < 0x7FFFFFFF else 0

    def hp_candidates():
        p = boss_ptr()
        if not p:
            return {}
        blob = pm.read_bytes(p + 0x2B80, 0x80)         # 0x2B80 .. 0x2C00
        out = {}
        for o in range(0, len(blob) - 3, 4):
            v = struct.unpack_from("<i", blob, o)[0]
            if 5000 < v < 40000:                       # real boss-HP range only
                out[0x2B80 + o] = v
        return out

    obs, _ = env.reset(options={"hard": True})
    step, appear, present, nullrun, at_boss = 0, 0, False, 999, False
    while step < 16000:
        obs, r, t, tr, info = env.step(int(pol.act(obs)))
        step += 1
        p = boss_ptr()
        if not p:
            nullrun += 1
            if nullrun > 90:
                present = False
        else:
            if not present and nullrun > 90:
                appear += 1
            present = True
            nullrun = 0
        # wait until we're at the target boss AND an HP field has activated
        if appear == args.which and hp_candidates():
            at_boss = True
            break
    if not at_boss:
        print("never reached an activated boss"); env.close(); return

    # optional: pin power by writing it each frame
    def set_power(v):
        try:
            g = struct.unpack("<I", pm.read_bytes(GAME_MANAGER + GM_GLOBALS_PTR, 4))[0]
            pm.write_bytes(g + G_POWER, struct.pack("<f", v), 4)
        except Exception:
            pass

    HP = 0x2BB8
    print(f"boss up at step {step}, power {power():.1f}, HP "
          f"{struct.unpack('<i', pm.read_bytes(boss_ptr()+HP,4))[0]} - "
          f"letting the policy damage-phase the fight", flush=True)
    _dump_shot_table(pm)
    if args.dumponly:
        env.close(); return
    E_POS = 0x2B0C
    hp_log, pw_log, lined_log = [], [], []
    for i in range(13000):                     # the whole fight
        if args.power >= 0:
            set_power(args.power)
        pp = boss_ptr()
        bx = struct.unpack("<f", pm.read_bytes(pp + E_POS, 4))[0] if pp else 192.0
        px = env.h.s.player_x
        if args.camp:                          # park under the boss, full-auto
            d = 3 if px < bx - 3 else (7 if px > bx + 3 else 0)
            a = d + (9 if args.focus else 0) + 18
        else:
            # policy plays normally (positions + shoots), force the shoot bit on
            a = int(pol.act(obs))
            a = (a % 18) + 18                   # keep dir/focus, force shoot
        lined_log.append(abs(px - bx) < 26.0)
        obs, r, t, tr, info = env.step(a)
        step += 1
        p = boss_ptr()
        if not p:
            print(f"  boss gone at frame {i}"); break
        hp_log.append(struct.unpack("<i", pm.read_bytes(p + HP, 4))[0])
        pw_log.append(power())
        if i % 1200 == 0:
            print(f"  f{i}: HP {hp_log[-1]}  pow {pw_log[-1]:.1f}", flush=True)
    env.close()

    hp = np.array(hp_log, float)
    pw = np.array(pw_log)
    dd = -np.diff(hp)
    lined = np.array(lined_log, bool)[:len(dd)]
    ok = (dd > 0) & (dd < 500)                  # damage frames (ignore resets)
    hits = dd[ok]
    print(f"\n=== fight: {len(hp)} frames, power {pw.min():.0f}-{pw.max():.0f} ===")
    print(f"HP {hp[0]:.0f} -> {hp.min():.0f}, {ok.sum()} damage frames "
          f"({ok.mean()*100:.0f}% of frames), lined-up {lined.mean()*100:.0f}% of frames")
    print(f"dmg/frame when landing:   mean {hits.mean():.1f}  median {np.median(hits):.1f}")
    print(f"effective dmg/frame:      {hits.sum()/len(dd):.1f}  "
          f"(= a 15000-HP phase in ~{15000/(hits.sum()/len(dd))/60:.0f}s)")
    for tag, msk in (("lined up", lined), ("homing only", ~lined)):
        seg = dd[ok & msk]
        fr = (ok & msk).sum() / max(1, msk.sum())
        if seg.size:
            print(f"  {tag:12s}: {seg.sum()/max(1,msk.sum()):.1f} dmg/frame "
                  f"(landing {fr*100:.0f}% of those frames, "
                  f"{seg.mean():.1f}/hit)")


if __name__ == "__main__":
    main()
