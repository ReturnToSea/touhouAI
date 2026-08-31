"""Where does a sim-trained policy actually die on the real game? Run it
NORMALLY (shooting) from Stage 1 to its first death, N times, and log the
location. Tells us which boss to record + train against - the PoC needs a
boss the policy FAILS, not one it already clears (Cirno/Letty).

    .venv/Scripts/python native/survey_deaths.py runs_sim/ppo_v29/snap_0092M.pt --eps 15
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=Path)
    ap.add_argument("--eps", type=int, default=15)
    ap.add_argument("--max-seconds", type=float, default=1600)
    ap.add_argument("--no-shoot", action="store_true",
                    help="dodge only (a %% 18) - matches the recording method; "
                    "boss dies by timeout, so we see the pure-dodge wall")
    args = ap.parse_args()
    mask = (lambda a: a % 18) if args.no_shoot else (lambda a: a)

    pol = MLPPolicy.load(args.policy)
    env = Th07Env(frame_skip=1, max_seconds=args.max_seconds, render=False,
                  dll_obs=True, hard_reset=True)
    s = env.h.s

    rows = []
    for ep in range(args.eps):
        obs, _ = env.reset(options={"hard": True})   # fresh RNG each run
        step = 0
        peak_stage = int(s.stage)
        while True:
            obs, r, term, trunc, info = env.step(mask(int(pol.act(obs))))
            step += 1
            peak_stage = max(peak_stage, int(s.stage))
            if term or trunc:
                break
        boss = bool(s.boss_present)
        hpf = (s.boss_hp / s.boss_hp_max) if (boss and s.boss_hp_max > 0) else -1.0
        where = f"S{s.stage} {'BOSS %.0f%%hp' % (hpf * 100) if boss else 'stage-portion'}"
        rows.append((int(s.stage), boss, hpf, peak_stage, step, int(s.score),
                     "trunc" if trunc else "died"))
        print(f"  ep {ep:2d}: {where:22s} peak S{peak_stage}  "
              f"{step:6d} steps  score {s.score:>9d}  "
              f"{'(timed out)' if trunc else ''}", flush=True)
    env.close()

    print("\n=== where it ended ===")
    key = Counter()
    for st, boss, hpf, peak, step, sc, how in rows:
        k = f"S{st} " + (f"boss ({'cleared-ish' if hpf < 0.05 else 'mid-fight'})"
                         if boss else "stage-portion")
        key[k] += 1
    for k, n in key.most_common():
        print(f"  {n:2d}x  {k}")
    peaks = [r[3] for r in rows]
    print(f"\n  deepest stage reached: S{max(peaks)}   "
          f"median end stage: S{int(np.median([r[0] for r in rows]))}")


if __name__ == "__main__":
    main()
