"""Continuously play the newest checkpoint of a fight run against the REAL game
and append the result to runs_sim/<name>/realtransfer.npy, so sim/fight_hud.py
can plot real-game transfer live next to the sim curve.

    .venv/Scripts/python sim/fight_transfer_daemon.py fight_letty_seg --which 2

One persistent hooked th07 (env hard-reset between episodes, no relaunch). Each
time a newer mlp_*.pt / last_mlp.pt appears it plays K episodes and logs, per
episode, a row:

    [wall_epoch, train_steps, active_survival_s, killed(0/1), boss_dmg_frac]

Runs fine alongside training - the game is render-less and the policies are
tiny MLPs; the GPU stays with the trainer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from eval_boss import run_episode, boss_state          # noqa: E402
from env import Th07Env                                 # noqa: E402
from policy import MLPPolicy                            # noqa: E402

RUNS = HERE.parent / "runs_sim"


def newest_ckpt(run):
    snaps = sorted(run.glob("mlp_*.pt"),
                   key=lambda p: p.stat().st_mtime)
    last = run / "last_mlp.pt"
    if last.exists() and (not snaps or last.stat().st_mtime >= snaps[-1].stat().st_mtime):
        return last
    return snaps[-1] if snaps else None


def train_steps(run):
    try:
        h = np.load(run / "history.npy")
        return float(h[-1, 1])          # 6-col fight schema: col 1 = total steps
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--which", type=int, default=2, help="1=Cirno, 2=Letty")
    ap.add_argument("--driver", type=Path,
                    default=RUNS / "ppo_v29/snap_0092M.pt")
    ap.add_argument("--k", type=int, default=5, help="episodes per checkpoint")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--tag", default="", help="suffix so N daemons can run in "
                    "parallel writing realtransfer_<tag>.npy (fight_hud merges "
                    "all realtransfer*.npy)")
    args = ap.parse_args()

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    out = run / (f"realtransfer_{args.tag}.npy" if args.tag else "realtransfer.npy")

    drive = MLPPolicy.load(args.driver)
    env = Th07Env(frame_skip=1, max_seconds=400, render=False, dll_obs=True,
                  hard_reset=True)
    pm = env._pm
    if pm is None:
        import pymem
        pm = pymem.Pymem(); pm.open_process_from_id(env.pid)

    # a boss HP scale for the dmg fraction - Letty's activated life field peaks
    # ~15000; use the max hp0 we ever see as the running denominator.
    hp_scale = 15000.0
    seen_mtime = 0.0
    print(f"[transfer] watching {run}/  (K={args.k} eps / checkpoint)", flush=True)

    while True:
        c = newest_ckpt(run)
        if c is None or c.stat().st_mtime == seen_mtime:
            time.sleep(args.poll)
            continue
        seen_mtime = c.stat().st_mtime
        steps = train_steps(run)
        try:
            test = MLPPolicy.load(c)
        except Exception:
            time.sleep(args.poll)
            continue

        rows = []
        for k in range(args.k):
            r = {"reached": False}
            for _try in range(4):              # drive policy sometimes dies en
                r = run_episode(env, pm, drive, test, args.which, lambda a: a)
                if r["reached"]:
                    break                      # route to Letty - just retry
            if not r["reached"]:
                print(f"[transfer] {steps/1e6:.0f}M ep{k}: drive failed x4, skip",
                      flush=True)
                continue
            if r["hp0"] > hp_scale:
                hp_scale = r["hp0"]
            dmg_frac = min(1.0, r["dmg"] / hp_scale) if r["dmg"] > 0 else 0.0
            killed = 1.0 if r["killed"] else 0.0
            rows.append([time.time(), steps, r["active_s"], killed, dmg_frac])
            kk, kb, kd = r.get("killer_kind", 0), r.get("killer_box", 0.0), r.get("killer_dist", 0.0)
            where = (f"phase {r.get('death_phase','?')} (bar {r.get('n_bars','?')}, "
                     f"HP {r.get('death_life',0):.0f})") if not killed else ""
            killer = (f"  killer: class {kk} box {kb:.1f} @ {kd:.1f}px"
                      if not killed and kd > 0 else "")
            hx = r.get("live_xy", (0, 0))
            print(f"[transfer] {steps/1e6:5.0f}M ep{k}: {r['active_s']:5.1f}s active "
                  f"(lead-in {r['lead_in_s']:.0f}s, total {r['total_s']:.0f}s)  "
                  f"{'KILL' if killed else 'died ' + where}  "
                  f"lives {r['lives0']:.0f}->{r['lives_end']:.0f}  "
                  f"start@({hx[0]:.0f},{hx[1]:.0f}){killer}",
                  flush=True)

        if rows:
            arr = np.array(rows, np.float64)
            if out.exists():
                arr = np.vstack([np.load(out), arr])
            np.save(out, arr)


if __name__ == "__main__":
    main()
