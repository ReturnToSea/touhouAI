"""Honest sim eval for the snap_*.pt / best.pt / last.pt checkpoints of a run.

train.py's live greedy_eval pools every episode that finishes inside a fixed
decision window WITH auto-reset, so fast-dying spawns (which cycle many times)
dominate the median -> the number is biased low, worst early in training. This
runs the standard eval instead: each of B envs plays exactly ONE episode to
death or the cap, equal weight. Directly comparable to the real-game daemon.

    .venv-cuda\\Scripts\\python sim\\eval_snaps.py ppo_v27
    .venv-cuda\\Scripts\\python sim\\eval_snaps.py ppo_v27 --B 8192 --cap 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from danmaku import DanmakuSim  # noqa: E402
from policy import MLPPolicy    # noqa: E402

RUNS = HERE.parent / "runs_sim"
FS = 3.0 / 60.0


@torch.no_grad()
def eval_one(net, sim, n_dec):
    """Each env: one episode to death or n_dec cap. -> array of lengths (seconds)."""
    o = sim.reset()
    B = sim.B
    dev = sim.dev
    el = torch.zeros(B, device=dev)
    done_ever = torch.zeros(B, dtype=torch.bool, device=dev)
    rec = torch.zeros(B, device=dev)
    for _ in range(n_dec):
        a = net(o).argmax(-1)
        o, _, done = sim.step(a)
        el = el + 1.0
        newly = done.bool() & ~done_ever
        rec = torch.where(newly, el, rec)
        done_ever = done_ever | done.bool()
        if bool(done_ever.all()):
            break
    rec = torch.where(done_ever, rec, el)      # censor survivors at the cap
    return (rec * FS).sort().values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--B", type=int, default=2048, help="raise if training isn't using the GPU")
    ap.add_argument("--cap", type=float, default=300.0, help="episode cap, seconds")
    args = ap.parse_args()

    d = RUNS / args.run
    pts = sorted(d.glob("snap_*.pt")) + [p for p in (d / "last.pt", d / "best.pt") if p.exists()]
    if not pts:
        print(f"no checkpoints in {d}")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_dec = int(args.cap / FS)
    # compile=False: a second compiled sim fights the training run's inductor
    # cache and takes minutes to warm up. eager on the GPU is fine at this B.
    sim = DanmakuSim(B=args.B, device=dev, max_frames=n_dec * 3 + 300, compile=False)
    print(f"{args.run}   B={args.B}  cap={args.cap:.0f}s   (first-episode-only, equal weight)\n")
    print(f"{'checkpoint':>16}   {'median':>7} {'p90':>7} {'mean':>7}  "
          f"{'>60s':>6} {'>120s':>6} {'>180s':>6}")
    for p in pts:
        net = MLPPolicy.load(p).net.to(dev)
        a = eval_one(net, sim, n_dec)
        print(f"{p.stem:>16}   {np.median(a):7.1f} {np.percentile(a,90):7.1f} "
              f"{a.mean():7.1f}  {(a>60).mean()*100:5.0f}% "
              f"{(a>120).mean()*100:5.0f}% {(a>180).mean()*100:5.0f}%")


if __name__ == "__main__":
    main()
