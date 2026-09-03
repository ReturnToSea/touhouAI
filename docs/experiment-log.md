# Experiment log

The dated, quantitative companion to the [Extras](dead-ends.md) section. Every
training run: what changed, what the sim curve did, what transferred, verdict.

!!! warning "Sim score is not real transfer"
    The procedural sim is deliberately brutal (domain randomization), so its
    survival numbers run ~20–40 s while the *same policy* survives ~200–300 s on
    the real game. Always read the **real** line, not the sim line. `ppo_v26` got
    *worse* on the real game while its sim median rose.

!!! note "How this is kept"
    After a run, `python tools/plot_run.py runs_sim/<name>` writes the curve to
    `docs/assets/curves/<name>.png` (sim series + the `realtransfer.npy` overlay)
    and copies `history.npy` to `runs_meta/`. `.pt` weights stay out of git.
    The *what changed* / *verdict* columns are written by hand.

## Current best

| Metric | Value | From |
|---|---|---|
| Best real playthrough | mid-Stage 2, ~1.63 M score | `ppo_v12` — cleared the Chen midboss, died before the Stage 2 boss (watched, not logged) |
| Real transfer, current obs | **~225 s** median (stages 1–2) | `ppo_v27` / `ppo_v29` |
| Recorded-Letty, full 1 B-step run | 60 s active-fight median / 7% kill-rate over 631 real fights; best checkpoint (~724 M) ~103 s / 33% | `fight_letty_seg` v9 |

No current-obs run has matched `ppo_v12`'s mid-Stage-2 reach. Every
domain-randomized run plateaus around "clears stage 1, dies in stage 2." The
recorded-Letty runs move past pure dodging (they land real kills) but
[hit their own ceiling](ceiling.md).

## The PPO arc

Read top-to-bottom.

### v12 and the 212-d era — peak

`ppo_v12` (212-d obs, auto-aim) is the deepest a policy has reached on the real
game: it **cleared Stage 1 and the Chen midboss, then died in the Stage 2
stage portion just before the Chen boss fight** — mid-Stage 2, ~1.63 M score.
Watched, not logged (no `realtransfer.npy`). The old note said "470 s / stages
1–3"; that overstates the reach, and **"470 s" exactly matches this run's
`history.npy` sim-survival column** — it was the sim number, not a real
duration.

`ppo_v22` (an early 236-d re-run) hit **~368 s at only 82 M steps**. The 212-d
obs was later widened to 236-d and `ppo_v12` no longer loads against the current
code.

### v23–v25 — trying to teach shooting, the policy games it

Added a front-only shot model, aimed enemy bursts, and a top-down "spam phase"
to push the agent to engage enemies. Over 1000 M steps `ppo_v25` **regressed to
~159 s**. The kill and power rewards were too weak relative to the survival
bonus, so the optimal policy was to *stop shooting and just dodge* — which is
fine in the sim and useless for a real run where you need power.

> **Realisation.** You cannot bolt "shoot enemies" onto a survival objective with
> a small reward term. The agent will find the survive-only local optimum.

### v26 — stronger engagement rewards, overfits

Cranked the enemy-damage reward (`0.10 → 0.35`), item reward (`0.30 → 0.60`),
added a power-standing bonus, widened the enemy hit radius, capped episodes at
240 s, split the death signal by cause. On a **fixed** sim stage it learned to
engage — and transfer got *worse* as the sim score climbed. `best.pt`, ranked by
sim score, was the most overfit checkpoint in the run.

> **Realisation.** A fixed sim stage is a memorisation target. Rank checkpoints
> by transfer tests, not sim score.

### v27 — domain-randomization rewrite, recovers

Threw out the hand-tuned stage. Patterns are now procedurally generated per
episode with heavy randomization of density, speed, aim, and emitter layout.
Real transfer came back to **~223 s median** (41–505 s range). The last
current-obs run that clearly worked.

### v28 — observation normalization, catastrophic

Standardized each obs feature to zero-mean/unit-variance using sim statistics.
Features that are constant in the sim (empty item slots, walls) got divided by a
near-zero std, folding their weights up ~`1e4×`. On real obs the network
exploded. Real transfer **collapsed to 18 s median** (13–42 s) despite a
perfectly healthy sim curve. Reverted (`65a4d78`).

### v29 — v27 plus a batch of refinements, marginal

v27 randomization + AABB collision (was circular) + a bottom-camp penalty +
focus-aware escape scalars + reward normalization + LR annealing, **no** obs
normalization. Real transfer **~231 s median** (68–645 s), flat after ~100 M
steps. Basically a wash versus v27 — the plateau is the sim's, not the run's.
Snapshots saved every 40 M for transfer-testing (`snap_*.pt`); `best.pt` still
underperforms mid-training snapshots, per the v26 lesson.

## The recorded-Letty era — `fight_letty_seg`

A from-scratch policy trained on ~20 [recorded Letty fights](recording.md) with
[synthetic damage-phasing](de-letty-replay.md#synthetic-damage-phasing), aiming
to be the first run that *kills* a real boss rather than just outlasting it. The
full pipeline and the transfer analysis are in
[Porting Letty into the sim](de-letty-replay.md); this is the run-by-run.
Numbers are **active fight time** — the ~42 s of dialogue is excluded.

### v1–v4 — the mechanics come together

Real phase detection from screen-clears; per-phase armor windows; the
[realistic ReimuA shot model](de-letty-replay.md#synthetic-damage-phasing) (20%
homing + 80% forward needle, ±17.5 px lane); the
[kill-only reward](de-shooting.md). Sim
kill-rate climbed 19% → a noisy 40–100%. Real transfer: from **one** kill in a
full billion steps (memorisation-prone) to ~11 real kills by 250 M.

### v5–v7 — the re-aim saga

Three attempts to re-aim replayed bullets at the live policy, all abandoned — the
[full story](de-reaim.md). Each fixed one flicker and introduced another. v7's
best real checkpoint hit **88 s active-fight median** — the best of any run — then
PPO thrashed and destroyed it, dropping to ~20 s.

### v8 — stability + a survival floor

γ `0.997 → 0.995`, entropy `0.004 → 0.002`, cosine LR decay, `best_mlp.pt` saved
on eval-score peaks, and a small `+0.004/frame` survival floor added back (the
[over-correction](de-shooting.md#the-over-correction-put-a-small-survival-term-back)).
Still bimodal on the real game.

### v9 — re-aim removed, rigid field rotation

Dropped per-bullet re-aim entirely; replaced with a rigid ±10° per-episode
rotation of the whole danmaku field. Deleted `spawn`, `aimed`, `launch_ang`,
`birth`, and the per-episode buffers — freed ~8 GB VRAM, training 94k → 136k
frames/s. **Trained the full 1 B steps.** Sim kill-rate **stable at 50–83%** (vs
the 0–35% oscillation of v6–v8); median drifted ~128 s → ~118 s as the LR
annealed. Real transfer over 631 daemon fights: **7% kill-rate (47 kills),
60 s median**, but per-checkpoint kill-rate swings 0% → 33% with no trend —
best at ~724 M / ~818 M (4/12), while the sim's `best_mlp.pt` (~598 M, 83% sim)
lands 1/10 real.

> **Realisation.** The anti-memorisation bundle genuinely helps — v9 lands real
> kills where v5 got one in a billion steps. But over a full billion steps the
> sim eval never predicts real performance, which means the policy is exploiting
> structure that only exists in 20 recordings and their symmetries. This is
> [the ceiling](ceiling.md). The response was
> [generative danmaku](de-generative-danmaku.md) — which hit the same wall from
> the other side — and then [real-game training](plan.md).

## Runs

| Run | Date | What changed | Real transfer | Verdict |
|---|---|---|---|---|
| `ppo_real_letty` | 2026-09-02 | [real-game PPO](plan.md) on the Letty fight; `ST_ROLLOUT` × 12 hooked games; warm-start `ppo_v29` | *running* | the [current plan](plan.md) — accept ~15× slower for zero fidelity gap |
| `fight_letty_ecl` run 1–8 | 2026-09-01/02 | [ECL VM](de-generative-danmaku.md) danmaku: real bytecode, measured bullet motion, ~2 px fidelity; streaming schedules; per-bullet hitboxes; full RE pass | flat **~50 s / 0% kill** every checkpoint; run 2 once reached Table-Turning (127 s) | [dead end](de-generative-danmaku.md) — reimplemented engine is still an artefact; x87 float error compounds |
| `fight_letty_seg` v9 | 2026-08-31 | re-aim removed → rigid field rotation; stability fixes; ~20 recs; full 1 B steps | 60 s median / 7% kill-rate over 631 fights; best ckpt ~103 s / 33% | best recorded-boss run; hits the [ceiling](ceiling.md) |
| `fight_letty_seg` v1–v8 | 2026-08-31 | phase detection + synthetic damage-phasing + kill-only reward + the [re-aim saga](de-reaim.md) | 1 → ~11 real kills; v7 peaked 88 s then thrashed | superseded by v9 |
| `fight_letty` | 2026-08-31 | recorded-Letty FightSim + re-aiming + real hitboxes + lethal enemy bodies; from scratch | 150–190 s on real Letty | inconclusive — Letty too easy (baseline 6/6) |
| `ppo_v29` | 2026-08-30 | v27 + AABB collision + focus-aware escape + reward-norm + LR anneal | ~231 s median | wash vs v27 |
| `ppo_v28` | 2026-08-29 | + obs normalization | ~18 s median | **killed** |
| `ppo_v27` | 2026-08-28 | domain-randomization rewrite | ~223 s median | works; the current baseline |
| `ppo_v26` | — | stronger enemy/item rewards, fixed sim stage | worse as sim rose | overfit |
| `ppo_v25` | — | front-only shot + aimed bursts + spam phase, 1000 M | ~159 s | regressed — stopped engaging |
| `ppo_v22` | — | early 236-d re-run | ~368 s, 82 M steps | strong; preserved |
| `ppo_v12` | — | 212-d era, auto-aim | mid-Stage 2 (past Chen midboss), ~1.63 M score | deepest real reach; "470 s" = its sim column |

### ppo_v27

![ppo_v27 survival curve](assets/curves/ppo_v27.png)

### ppo_v28 — obs normalization

![ppo_v28 survival curve](assets/curves/ppo_v28.png)

The sim curve is fine; real transfer sits at the floor the whole run.

### ppo_v29

![ppo_v29 survival curve](assets/curves/ppo_v29.png)

The gap between the sim lines (~20–35 s) and the real line (~230 s) is the whole
point of this chapter.

### fight_letty

![fight_letty survival curve](assets/curves/fight_letty.png)

No `realtransfer.npy` for this run — the real number is one `eval_boss.py` pass:
150–190 s on real Letty, 3/6 full clears.

## Sim throughput

| Change | env-frames/s | Notes |
|---|---|---|
| FightSim, eager PyTorch | 115k | ~40 kernel launches/step, overhead-bound |
| `+ torch.compile` + dedup `_now()` | **273k** | build_obs fused; grid-march graph-break fixed |
| `+ bigger batch (24576)` | 277k | no gain — compute-bound now |

End-to-end training went ~40k → ~250k frames/s (~6×).
