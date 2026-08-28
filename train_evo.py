"""Neuroevolution (Deep GA) on Th07Env. Run from repo root.

    .venv\\Scripts\\python train_evo.py --pop 128 --gens 2000

Every policy is evaluated on ONE game instance, sequentially. The snapshot makes
that instance perfectly deterministic, so one rollout per policy is an exact
fitness with zero noise - which is what makes selection clean. (Running the 8
instances in parallel was tried and abandoned: they take independent snapshots,
so the same policy scored 19-47 depending on which instance it landed on.)

Each generation: score every policy, keep the top `elite` unchanged, refill by
copying a top-`parents` policy and adding Gaussian noise. Fitness = episode
return.

Checkpoints (best.pt + resume.npz) land in runs/<name>/. Watch with:
    watch.py runs/<name>/best.pt --evo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))

from env import Th07Env  # noqa: E402
from policy import MLPPolicy  # noqa: E402


def rollout(env, pol, max_steps) -> float:
    obs, _ = env.reset()
    ret = 0.0
    for _ in range(max_steps):
        obs, r, term, trunc, _ = env.step(pol.act(obs))
        ret += r
        if term or trunc:
            break
    return ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--parents", type=int, default=32, help="truncation-selection cut")
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.02, help="mutation noise std")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--name", default="evo_st1")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    hidden = tuple(args.hidden)
    print(f"policy {hidden}: {MLPPolicy(hidden=hidden).n_params()} params")

    if args.resume:
        blob = np.load(args.resume)
        pop = [np.asarray(v, np.float32) for v in blob["pop"]]
        gen0 = int(blob["gen"])
        print(f"resumed {len(pop)} policies at gen {gen0}")
    else:
        pop = [MLPPolicy(hidden=hidden).get_flat().astype(np.float32)
               for _ in range(args.pop)]
        gen0 = 0

    env = Th07Env(frame_skip=args.frame_skip, max_seconds=args.max_seconds)
    max_steps = env.max_steps
    pol = MLPPolicy(hidden=hidden)
    best_fit = -1e9
    hist = []

    try:
        for gen in range(gen0, args.gens):
            t0 = time.perf_counter()
            fit = np.empty(len(pop))
            for i, flat in enumerate(pop):
                pol.set_flat(flat)
                fit[i] = rollout(env, pol, max_steps)
            order = np.argsort(fit)[::-1]
            dt = time.perf_counter() - t0

            fbest = fit[order[0]]
            hist.append((gen, fbest, fit.mean()))
            print(f"gen {gen:4d}  best {fbest:7.1f}  mean {fit.mean():6.1f}  "
                  f"med {np.median(fit):6.1f}  worst {fit[order[-1]]:6.1f}  "
                  f"({dt:.0f}s)", flush=True)

            if fbest > best_fit:
                best_fit = fbest
                pol.set_flat(pop[order[0]])
                pol.save(run / "best.pt")

            parents = order[:args.parents]
            new_pop = [pop[order[j]].copy() for j in range(args.elite)]
            while len(new_pop) < args.pop:
                src = pop[parents[rng.integers(len(parents))]]
                new_pop.append(src + rng.normal(
                    0.0, args.sigma, src.shape).astype(np.float32))
            pop = new_pop

            if gen % 5 == 0:
                np.savez(run / "resume.npz", pop=np.array(pop), gen=gen + 1)
                np.save(run / "history.npy", np.array(hist))
    finally:
        np.savez(run / "resume.npz", pop=np.array(pop), gen=gen + 1)
        np.save(run / "history.npy", np.array(hist))
        env.close()


if __name__ == "__main__":
    main()
