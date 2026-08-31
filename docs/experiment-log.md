# 13 · Experiment log

The dated, quantitative companion to [Dead ends & lessons](dead-ends.md). Every
training run: what changed, what the sim curve did, what transferred, verdict.

!!! note "How this is kept"
    After a run, `python tools/plot_run.py runs_sim/<name>` writes the curve to
    `docs/assets/curves/<name>.png` and copies `history.npy` to `runs_meta/`
    (so the plot is reproducible from the repo — `.pt` weights stay out of git).
    Then add a row below. The *what changed* and *verdict* columns are written
    by hand.

## Current best

| Metric | Value | From |
|---|---|---|
| Real playthrough | `~470 s, stages 1–3` | `ppo_v12` (retired 212-d obs) |
| Real Letty, dodge-only | `6/6 clears` | procedural-sim baseline (`snap_0092M`) |
| Real Letty, dodge-only | `3/6 clears` | `fight_letty` (recorded) |
| Current-obs general policy | `dies at S1 boss` | best of v27 / v29 |

The headline number (`ppo_v12`) has not been reproduced on the current 236-d
observation. See [the base-policy regression](dead-ends.md#the-base-policy-regression-open).

## Runs

Reverse-chronological. "sim" = greedy median survival at the end of training;
"real" = survival on the actual game.

| Run | Date | What changed | Sim | Real | Verdict |
|---|---|---|---|---|---|
| `fight_letty` | 2026-08-31 | recorded-Letty FightSim + re-aiming + per-bullet hitboxes + lethal enemy bodies; from scratch | 70 s median (capped view) | 150–190 s on real Letty, but the plain baseline clears it 6/6 | **inconclusive** — Letty is too easy to show a benefit |
| `ppo_v29` | 2026-08-30 | v27 randomization + AABB collision + bottom-camp penalty + focus-aware escape + reward-norm + LR anneal; **no** obs-norm | ~35 s median | dies at S1 boss (normal play) | regressed vs v12; snapshots every 40 M for transfer-testing |
| `ppo_v28` | 2026-08-29 | added obs normalization | healthy curve | **15–40 s** real | **killed** — sim-constant features got `1e4×` weights ([why](dead-ends.md#observation-normalization-v28-abandoned)) |
| `ppo_v27` | 2026-08-28 | domain-randomization rewrite | ~39 s median | 200–500 s real (per notes) | transferred well; the last current-obs run that did |
| `ppo_v26` | — | fixed sim stage, stronger enemy/item rewards | rising | got **worse** as sim rose | overfit — `best.pt` is the most overfit checkpoint |
| `ppo_v25` | — | front-only shot + aimed enemy bursts + top-down spam phase, 1000 M steps | high | **159 s** real (down from 470) | regressed — reward changes made it stop engaging |
| `ppo_v22` | — | (212-d era) | 88 s median | **368 s real**, stages 1–3, at only 82 M steps | strong; preserved |
| `ppo_v12` | — | (212-d era) | — | **470 s real**, stages 1–3 | best real result to date; obs since retired |

### fight_letty

![fight_letty survival curve](assets/curves/fight_letty.png)

Climbed the whole run (42 → 70 s median), never plateaued. The in-sim eval is
capped at 83 s so full clears are invisible here — the real signal is
`eval_boss.py`.

### ppo_v27

![ppo_v27 survival curve](assets/curves/ppo_v27.png)

### ppo_v29

![ppo_v29 survival curve](assets/curves/ppo_v29.png)

## Sim throughput

| Change | env-frames/s | Notes |
|---|---|---|
| FightSim, eager PyTorch | 115k | ~40 kernel launches/step, overhead-bound |
| `+ torch.compile` + dedup `_now()` | **273k** | build_obs fused; grid-march graph-break fixed |
| `+ bigger batch (24576)` | 277k | no gain — compute-bound now |

End-to-end training went ~40k → ~250k frames/s (~6×).
