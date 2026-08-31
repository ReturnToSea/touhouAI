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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, default=2, help="1 Cirno, 2 Letty, ...")
    ap.add_argument("--power", type=float, default=-1,
                    help="pin power to this (0-128) via godmode's life write path; "
                    "-1 = leave as-is and just record it")
    ap.add_argument("--focus", action="store_true", help="focused shot")
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
    hp_log, pw_log = [], []
    for i in range(13000):                     # the whole fight
        if args.power >= 0:
            set_power(args.power)
        # policy plays normally (positions + shoots), but force the shoot bit on
        a = int(pol.act(obs))
        a = (a % 18) + 18                       # keep dir/focus, force shoot
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
    hits = dd[(dd > 0) & (dd < 500)]            # damage frames (ignore resets)
    print(f"\n=== fight: {len(hp)} frames, power {pw.min():.0f}-{pw.max():.0f} ===")
    print(f"HP {hp[0]:.0f} -> {hp.min():.0f}, {(dd > 0).sum()} damage frames "
          f"({(dd > 0).mean()*100:.0f}% of frames)")
    print(f"dmg/frame when landing:   mean {hits.mean():.1f}  median {np.median(hits):.1f}")
    print(f"effective dmg/frame:      {hits.sum()/len(dd):.1f}  "
          f"(= a 15000-HP phase in ~{15000/(hits.sum()/len(dd))/60:.0f}s)")


if __name__ == "__main__":
    main()
