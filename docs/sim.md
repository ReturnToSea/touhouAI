# The danmaku simulator

Even headless at [~80× real-time](hook.md), the real game is far too slow for
PPO, which needs hundreds of millions of frames to learn a hard pattern. The
answer is a **GPU danmaku simulator** that runs ~1000 games in parallel as one
batched tensor program — **~273 k env-frames per second**, minutes instead of
days.

Everything downstream of the simulator — [collision](collision.md),
[observation](obs.md), [PPO](ppo.md), the real-game transfer daemon — is shared.
What changes between approaches is only **where the bullet positions come from**.
Two approaches have been used; both plateau in the same place
([Chapter 11](ceiling.md)).

## Approach 1 — procedural generation

A hand-written bullet-hell generator. Bullet motion is ported from PyTouhou's
`Bullet.update()`; patterns — fans, rings, spirals, aimed spreads — are
**generated fresh every episode** with heavy randomisation of density, speed,
aim, and emitter layout. There is no real boss, just a distribution of made-up
danmaku tuned to be *harder* than anything in the real game.

**Why it works:** the generator never repeats a pattern, so there is nothing to
memorise. A policy trained on it learns general dodging reflexes that transfer.

| Run | Real transfer |
|---|---|
| `ppo_v12` (retired 212-d obs) | mid-Stage 2, ~1.63 M score — cleared the Chen midboss |
| `ppo_v27` / `ppo_v29` (236-d) | ~225–231 s median — clears Stage 1, dies in Stage 2 |

**Why it tops out:** the generator produces plausible *bullets* but not real
*boss structure* — no phase transitions, no spellcard declarations, no
satellite-enemy choreography, and nothing about the stage portions (enemy waves,
power management). Every domain-randomised run plateaus around "clears Stage 1,
dies in Stage 2." The run-by-run arc — and every attempt to make it teach *more*
than dodging — is in the [experiment log](experiment-log.md).

## Approach 2 — recorded replay

Record the real game's bullet positions frame by frame and replay them on the
GPU. This reproduces a **specific** boss exactly, including the undocumented
motion effects the [ECL interpreter](de-ecl-vm.md) couldn't, and it is what
landed the first real *kills* on Letty. It has its own ceiling — a fixed set of
recordings and their symmetries. [Chapter 10](recording.md) covers the recorder;
the full pipeline and the transfer analysis are in
[Porting Letty into the sim](de-letty-replay.md).

## Throughput

| Change | env-frames/s |
|---|---|
| eager PyTorch | 115 k |
| `+ torch.compile`, deduped `_now()`, grid-march graph-break fixed | **273 k** |
| `+ bigger batch (24576)` | 277 k — no gain, compute-bound |

End-to-end training went ~40 k → ~250 k frames/s.
