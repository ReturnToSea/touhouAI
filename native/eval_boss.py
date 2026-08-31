"""Measure a policy's survival IN a real th07 boss fight (--which 1=Cirno midboss, 2=Letty). Drive to the
midboss with a strong stage-1 policy, then hand control to the policy under
test and time how long it lasts against Cirno.

    .venv/Scripts/python native/eval_cirno.py POLICY.pt [--eps 6] [--driver snap]

Compares real-Cirno transfer of a FightSim-trained policy vs a made-up-danmaku
policy.
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

EM_BOSSES0 = 0x009A9B00 + 0x00954598
E_POS, E_LIFE = 0x2B0C, 0x2BB8


def boss_state(pm):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=Path)
    ap.add_argument("--eps", type=int, default=6)
    ap.add_argument("--driver", type=Path,
                    default=HERE.parent / "runs_sim/ppo_v29/snap_0092M.pt")
    ap.add_argument("--which", type=int, default=1, help="1=Cirno midboss, 2=Letty")
    ap.add_argument("--no-shoot", action="store_true",
                    help="force the test policy to dodge only (a %% 18) - clean "
                    "survival metric, no boss damage confound")
    args = ap.parse_args()
    _mask = (lambda a: a % 18) if args.no_shoot else (lambda a: a)

    test = MLPPolicy.load(args.policy)
    drive = MLPPolicy.load(args.driver)
    env = Th07Env(frame_skip=1, max_seconds=400, render=False, dll_obs=True,
                  hard_reset=True)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)

    survs = []
    for ep in range(args.eps):
        obs, _ = env.reset(options={"hard": True})
        # drive until the Nth boss appearance
        step, appear, present, nullrun, in_fight = 0, 0, False, 999, False
        while step < 12000:
            obs, r, term, trunc, info = env.step(int(drive.act(obs)))
            step += 1
            b = boss_state(pm)
            if b is None:
                nullrun += 1
                if nullrun > 90:
                    present = False
            else:
                if not present and nullrun > 90:
                    appear += 1
                    if appear == args.which:
                        in_fight = True
                        break
                present = True
                nullrun = 0
            if term or trunc:
                break
        if not in_fight:
            print(f"  ep {ep}: never reached boss #{args.which}", flush=True)
            continue
        # hand over to the test policy, time the fight (Letty dodge-only runs
        # ~179s to the natural end, so give the window room to see a real clear)
        fstart, nf = step, 0
        while step < fstart + 11400:
            obs, r, term, trunc, info = env.step(_mask(int(test.act(obs))))
            step += 1
            b = boss_state(pm)
            if b is None:
                nf += 1
                if nf > 90:
                    break
                continue
            nf = 0
            if term or trunc:
                break
        surv = max(0.0, (step - fstart - nf) / 60.0)
        survs.append(surv)
        print(f"  ep {ep}: boss-fight survival {surv:.1f}s"
              f"{'  (fight ended - cleared/timed out)' if nf > 90 else '  (died)'}",
              flush=True)
    env.close()

    s = np.array(survs)
    if len(s):
        print(f"\n{args.policy.name} vs real boss (n={len(s)}): "
              f"median {np.median(s):.1f}s  mean {s.mean():.1f}s  "
              f"[{', '.join(f'{x:.0f}' for x in survs)}]")


if __name__ == "__main__":
    main()
