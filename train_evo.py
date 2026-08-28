"""Island-model neuroevolution on Th07Env. Run from repo root.

    .venv\\Scripts\\python train_evo.py --islands 8 --island-pop 24 --gens 5000

Each island owns one game instance and its own sub-population. A generation
evaluates all islands in parallel (one Python process, work-queue over the N
instances - the episode runs inside the DLL so Python barely participates).
Within an island: truncation selection + Gaussian mutation + elitism. Every
`--migrate-every` generations the best few individuals ring-migrate to the
next island.

Fitness (episode return) is only comparable WITHIN an island - the instances
take independent snapshots. best.pt is the highest-scoring island champion.

Checkpoints: runs/<name>/best.pt, resume.npz, history.npy, generations.npz.
Watch:  watch.py runs/<name>/best.pt --evo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))

from env import Th07Env       # noqa: E402
from evohud import EvoHud     # noqa: E402
from policy import MLPPolicy  # noqa: E402


def fitness_of(st) -> float:
    return 0.02 * st["frames"] + st["score"] * 1e-4 - (5.0 if st["died"] else 0.0)


class Island:
    def __init__(self, idx, hidden, pop_size, frame_skip, max_seconds):
        self.idx = idx
        self.hidden = hidden
        self.frame_skip = frame_skip
        self.max_seconds = max_seconds
        self.env = Th07Env(frame_skip=frame_skip, max_seconds=max_seconds)
        self.max_frames = self.env.max_steps * frame_skip
        self.pop = [MLPPolicy(hidden=hidden).get_flat().astype(np.float32)
                    for _ in range(pop_size)]
        self.fit = np.zeros(pop_size)
        self.frames = np.zeros(pop_size, dtype=np.int64)
        self.scores = np.zeros(pop_size, dtype=np.int64)

    def relaunch(self):
        try:
            self.env.close()
        except Exception:
            pass
        self.env = Th07Env(frame_skip=self.frame_skip, max_seconds=self.max_seconds)
        self.max_frames = self.env.max_steps * self.frame_skip

    def next_gen(self, parents_cut, elite, sigma, rng):
        order = np.argsort(self.fit)[::-1]
        parents = order[:parents_cut]
        new = [self.pop[order[j]].copy() for j in range(elite)]
        while len(new) < len(self.pop):
            src = self.pop[parents[rng.integers(len(parents))]]
            new.append(src + rng.normal(0.0, sigma, src.shape).astype(np.float32))
        self.pop = new


def evaluate(islands, hidden, hud=None, stall_timeout=90.0):
    """Work-queue: each island churns through its sub-population on its own
    instance; all N run concurrently. Blocks until every policy is scored.
    Raises RuntimeError if any island makes no progress for stall_timeout s
    (a crashed / hung instance)."""
    h1, h2 = hidden
    N = len(islands)
    nxt = [0] * N
    running = [None] * N
    last_ok = [time.perf_counter()] * N
    remaining = 0
    for i, isl in enumerate(islands):
        remaining += len(isl.pop)
        isl.env.h.eval_start(isl.pop[0], h1, h2, isl.frame_skip, isl.max_frames)
        running[i] = 0
        nxt[i] = 1

    last_pump = time.perf_counter()
    while remaining:
        progressed = False
        now = time.perf_counter()
        for i, isl in enumerate(islands):
            if running[i] is None:
                continue
            if now - last_ok[i] > stall_timeout:
                raise RuntimeError(f"island {i} stalled "
                                   f"(crash_code={isl.env.h.s.crash_code:#x})")
            if not isl.env.h.eval_done():
                continue
            last_ok[i] = now
            r = isl.env.h.eval_result()
            j = running[i]
            f = fitness_of(r)
            isl.fit[j] = f
            isl.frames[j] = r["frames"]
            isl.scores[j] = r["score"]
            remaining -= 1
            progressed = True
            if hud is not None:
                hud.record(i, j, f, r["frames"], r["score"])
            if nxt[i] < len(isl.pop):
                isl.env.h.eval_start(isl.pop[nxt[i]], h1, h2, isl.frame_skip,
                                     isl.max_frames)
                running[i] = nxt[i]
                nxt[i] += 1
            else:
                running[i] = None
        now = time.perf_counter()
        if hud is not None and now - last_pump > 0.05:
            hud.pump()
            last_pump = now
        if not progressed:
            time.sleep(0.0005)


def migrate(islands, n_migrants):
    champs = []
    for isl in islands:
        order = np.argsort(isl.fit)[::-1]
        champs.append([isl.pop[order[k]].copy() for k in range(n_migrants)])
    for i, isl in enumerate(islands):
        incoming = champs[(i - 1) % len(islands)]
        worst = np.argsort(isl.fit)[:n_migrants]
        for k, w in enumerate(worst):
            isl.pop[w] = incoming[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--islands", type=int, default=8)
    ap.add_argument("--island-pop", type=int, default=24)
    ap.add_argument("--parents", type=int, default=8, help="truncation cut per island")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--gens", type=int, default=5000)
    ap.add_argument("--migrate-every", type=int, default=10)
    ap.add_argument("--n-migrants", type=int, default=2)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=120.0)
    ap.add_argument("--name", default="evo")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hud", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    hidden = tuple(args.hidden)
    print(f"policy {hidden}: {MLPPolicy(hidden=hidden).n_params()} params  "
          f"| {args.islands} islands x {args.island_pop} = "
          f"{args.islands * args.island_pop} total")

    print(f"launching {args.islands} game instances...")
    islands = [Island(i, hidden, args.island_pop, args.frame_skip,
                      args.max_seconds) for i in range(args.islands)]
    if args.resume:
        blob = np.load(args.resume)
        saved = blob["pop"]              # (islands, island_pop, nparams)
        gen0 = int(blob["gen"])
        for i, isl in enumerate(islands):
            if i < len(saved):
                isl.pop = [np.asarray(v, np.float32) for v in saved[i]]
        print(f"resumed at gen {gen0}")
    else:
        gen0 = 0
    print("all instances up.\n")

    pol = MLPPolicy(hidden=hidden)
    best_fit = -1e9
    best_score = 0
    sim_frames = 0
    hist = []
    genlog = []
    hud = None if args.no_hud else EvoHud(
        total=args.islands * args.island_pop, total_gens=args.gens,
        hidden=hidden, run_dir=run)
    gen = gen0
    try:
        while gen < args.gens:
            t0 = time.perf_counter()
            if hud is not None:
                hud.gen_start(gen, [isl.pop for isl in islands])
            try:
                evaluate(islands, hidden, hud)
            except RuntimeError as e:
                print(f"gen {gen}: island eval failed ({e}); relaunching all",
                      flush=True)
                for isl in islands:
                    isl.relaunch()
                continue
            dt = time.perf_counter() - t0

            allfit = np.concatenate([isl.fit for isl in islands])
            champs = np.array([isl.fit.max() for isl in islands])
            gbest = champs.max()
            sim_frames += int(sum(isl.frames.sum() for isl in islands))
            best_score = max(best_score, max(int(isl.scores.max()) for isl in islands))
            hist.append((gen, gbest, allfit.mean()))
            genlog.append(np.stack([np.concatenate([isl.fit for isl in islands]),
                                    np.concatenate([isl.frames for isl in islands]),
                                    np.concatenate([isl.scores for isl in islands])]))
            bi = int(champs.argmax())
            bframes = int(islands[bi].frames[islands[bi].fit.argmax()])

            print(f"gen {gen:4d}  gbest {gbest:7.1f}  mean {allfit.mean():6.1f}  "
                  f"island-bests {np.round(champs,0)}  ({dt:.0f}s)", flush=True)
            if hud is not None:
                hud.gen_end(bframes)

            # island fitness isn't comparable across instances - every
            # --migrate-every gens, re-score all island champions on ONE
            # reference instance so best.pt is a fair pick with a real number.
            if gen % max(1, args.migrate_every) == 0:
                ref = islands[0].env.h
                champ_w = []
                ref_fit = []
                for isl in islands:
                    w = isl.pop[int(isl.fit.argmax())]
                    r = ref.eval_policy(w, *hidden, frame_skip=args.frame_skip,
                                        max_frames=islands[0].max_frames)
                    champ_w.append(w)
                    ref_fit.append(fitness_of(r) if r else -1e18)
                k = int(np.argmax(ref_fit))
                if ref_fit[k] > best_fit:
                    best_fit = ref_fit[k]
                    pol.set_flat(champ_w[k])
                    pol.save(run / "best.pt")
                    print(f"  new best.pt: island {k} champ, ref-fitness "
                          f"{ref_fit[k]:.1f}", flush=True)

            for isl in islands:
                isl.next_gen(args.parents, args.elite, args.sigma, rng)
            if args.migrate_every and (gen + 1) % args.migrate_every == 0:
                migrate(islands, args.n_migrants)

            gen += 1
            if gen % 5 == 0:
                np.savez(run / "resume.npz",
                         pop=np.array([isl.pop for isl in islands]), gen=gen)
                np.save(run / "history.npy", np.array(hist))
                np.savez_compressed(run / "generations.npz",
                                    data=np.array(genlog[-200:]))
    finally:
        np.savez(run / "resume.npz",
                 pop=np.array([isl.pop for isl in islands]), gen=gen)
        np.save(run / "history.npy", np.array(hist))
        for isl in islands:
            try:
                isl.env.close()
            except Exception:
                pass
        print(f"stopped at gen {gen}. best.pt fitness {best_fit:.1f}", flush=True)
        if hud is not None:
            hud.finish()   # keep the windows up until the user closes them


if __name__ == "__main__":
    main()
