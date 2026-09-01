# Porting Letty into the sim

**Verdict:** record-and-replay is the furthest a policy has gone against a real
boss — over a full 1 B-step run it killed real Letty **47 times in 631 fights**,
where a memorisation-prone earlier run got one kill in a billion steps. It still
didn't transfer *reliably* (7 % overall, wildly checkpoint-dependent), for a
reason that turned out to be structural — and that reason is
[the plan](ecl-vm.md)'s whole justification.

This is the end-to-end walkthrough: how ~20 recorded Letty fights became a GPU
training environment, and why the policy it produced swings between "kills
Letty" and "faceplants" on the real game.

## Why replay at all

Two earlier routes to "train on Letty specifically":

- The [procedural sim](sim.md) generates plausible danmaku but has no concept of
  a named boss — no phases, no spellcard declarations, no satellite-orb
  choreography.
- The [first ECL interpreter](de-ecl-vm.md) ran Letty's actual script correctly
  for control flow but couldn't reproduce **bullet motion** — the undocumented
  multi-slot `bullet_effects` system that makes delay bullets hang and aimed
  bullets curve.

Replay sidesteps both: record every live bullet's *position* each frame, and the
motion effects come along for free because they're already baked into the
recorded path. No engine to reimplement.

## Recording the fight

`record_boss_driven.py` drives to Letty with a weak god-mode policy and logs
every frame from boss-pointer-valid to boss-pointer-null — see
[Recording real bosses](recording.md) for the recorder itself. ~20 Letty fights,
each ~224 s (42 s dialogue + ~179 s / 10,750 frames of danmaku).

## The 42-second alignment bug

The boss / player / enemy logs start ~42 s *before* the first bullet — during
the dialogue. The loader (`_load_dense`) aligned the player, boss, and enemy
arrays each to its own first frame, but the bullet array to the first-bullet
frame: a **2548-frame (42 s) skew**. Every downstream check comparing a bullet to
the player was 42 seconds out of register — which is why the
[re-aim attempts](de-reaim.md) pointed bullets at a stale player for the whole
fight. Fixed by aligning every array to `f0 = first-bullet frame` and dropping
everything before it.

!!! note "Measure the fight, not the intro"
    Because of that lead-in, every transfer number on this page is **active
    fight time** — total minus the entrance and dialogue, timer starting when
    bullets appear. A quoted "102 s" was really ~60 s of danmaku.

## Phase detection

The recordings are dodge-only, so Letty never dies — every phase runs its full
duration, times out, and clears the screen. `boss_phases.phase_windows()` finds
those screen-clears (bullet count collapsing to ~0) and snaps them to Letty's
four known phase times, consistent to ~15 frames across all 20 recordings:

| Phase | Attack window | Repositioning lull before it |
|---|---|---|
| Non-spell 1 | 0 – 38 s | — |
| Cold Sign "Lingering Cold" | 44 – 89 s | 5.5 s |
| Non-spell 2 | 93 – 129 s | 3.8 s |
| "Table-Turning" | 132 – 179 s | 3.0 s |

Between a phase's `clear_start` and its `first_attack`, Letty is repositioning or
declaring the spell — she deals and takes no damage (**armored**).

## Synthetic damage-phasing

The recordings carry no boss HP: the recorder logs only `(step, x, y)` for the
boss, and Letty's `life` field reads `1` for most of the fight. So the sim
**invents** an HP pool per phase:

```
phase_HP = SHOT_DPS · attack_duration · KILL_FRAC
```

and lets the agent drain it. ReimuA's shot is modelled as **20 % homing** (lands
from anywhere) + **80 % forward needle** (lands only when the player is lined up
in x under the boss, within ±17.5 px):

```
dps = SHOT_DPS · dps_mult · (0.20 + 0.80 · aligned)
```

Drain a phase's HP to zero, *or* let its recorded timer expire, and the sim
screen-clears the bullets and jumps the recording to the next phase's
`first_attack`. Beat the last phase and the episode ends with a kill bonus.

!!! warning "The HP numbers are guessed"
    `KILL_FRAC` and `SHOT_DPS` were calibrated to produce plausible phase
    lengths, not measured. The [ECL VM plan](ecl-vm.md) replaces them with
    Letty's real `enemy_life_set` thresholds.

## FightSim

The recordings pack into a `[n_rec, F, 1025, 2]` position tensor on the GPU. Each
of *B* parallel episodes picks a random recording and a start point; bullets
follow their exact recorded path. Player physics and AABB
[collision](collision.md) run on top, feeding the same observation builder as the
procedural sim. **~130–270 k env-frames/s** depending on the feature set.

## Fighting memorisation

20 recordings is a tiny distribution, and the policy's cheapest win is to
memorise it: "the aimed fan always opens to the left, so start on the left." Four
measures widen the distribution **without distorting any trajectory**:

| Measure | What it does |
|---|---|
| **x-mirror** | flips each fight left↔right — 10 recordings become 20 |
| **rigid field rotation ±10°** | rotates the *whole* field — every bullet, every orb, the boss — by one random angle per episode, about screen-centre. A rigid rotation preserves every path and every gap exactly |
| **mid-phase random starts** | 50 % of episodes drop in at any frame of a phase with a random slice of HP left — a phase can't be learned as a fixed sequence |
| **damage randomisation** | random ±50 % DPS per episode + 20 % of episodes deal zero damage (pure survival), so a fast kill can't skip training on the back half of a phase |

### The re-aim detour

Pointing aimed bullets at the *live* policy — instead of the recorded player —
looked like the real fix for the biggest memorisation vector. **Three
implementations, all abandoned:** a re-aimed bullet's screen-lifetime stops
matching the recording's slot bookkeeping, so bullets vanish mid-screen or
flicker. The [full saga](de-reaim.md) is its own page. Rigid field rotation was
the escape hatch — and deleting the re-aim machinery freed **~8 GB of VRAM** and
sped training from 94 k to 136 k frames/s.

## The reward

Kill is the goal, survival is instrumental:

```
rew =  DMG_REW · boss-HP drained  +  KILL_BONUS · killed  −  HIT_PEN · hit  +  SURV_REW · alive
```

with `SURV_REW` a thin floor ~30× smaller than the aligned-shooting damage
reward. Getting here from a survival-weighted reward was
[its own saga](de-shooting.md).

## Training: `fight_letty_seg` v1–v9

| Runs | What | Result |
|---|---|---|
| v1–v4 | mechanics: phase detection, armor windows, the 20/80 shot model, kill-only reward | sim kill-rate 19 % → noisy 40–100 %; real kills 1-in-a-billion → ~11 by 250 M |
| v5–v7 | the [re-aim saga](de-reaim.md) | v7 peaked at **88 s** active-fight median — best of any run — then PPO thrashed it to ~20 s |
| v8 | stability: γ 0.997→0.995, entropy 0.004→0.002, cosine LR, `best_mlp.pt` on eval peaks, + the survival floor | still bimodal |
| v9 | re-aim removed → rigid field rotation; trained the full 1 B steps | sim kill-rate **stable 50–83 %** (vs the 0–35 % oscillation of v6–v8) |

The [experiment log](experiment-log.md) has the full run-by-run.

## The transfer daemon

`fight_transfer_daemon.py` runs a persistent hooked game alongside training. Each
time a checkpoint drops it plays ~12–15 real Letty fights, timed from the first
bullet, and appends `[wall, steps, active_survival_s, killed, damage_frac]` to
`realtransfer_*.npy`. Several daemons run in parallel under different tags;
`fight_hud.py` merges them into a live real-transfer curve next to the sim curve.
It exists because **the sim score stopped predicting transfer.**

## Why it didn't transfer

### What "it works" looks like

`fight_letty_seg` v9 trained the full **1 billion steps**. The transfer daemon
played **631 real Letty fights** across 53 checkpoints along the way:

| | Sim eval | Real game (all 631 fights) |
|---|---|---|
| survival | ~120 s median, drifting to ~118 s | median **60 s** active-fight, p90 114 s, max 148 s |
| kill-rate | flat 50–83 % the whole run | **7 %** overall (47 kills); **26 %** of runs clear 100 s+, **11 %** faceplant under 10 s |

That is still a real improvement — a memorisation-prone earlier run got exactly
**one** real kill in a full billion steps; v9 got 47. The mitigations help; they
don't close the gap.

### The tell: sim score stops predicting real score

Per-checkpoint real kill-rate swings **0 % → 33 %** with no trend over the run,
while the sim eval sits flat at 50–83 %:

| Checkpoint | Real median | Faceplants | Real kills / 12 |
|---|---|---|---|
| 235 M | **2.8 s** | 7 | 0 |
| 251 M | 83 s | 0 | 2 |
| 707 M | 17 s | 0 | 0 |
| 723 M | **103 s** | 0 | 4 |
| 927 M | 19 s | 2 | 0 |

The best checkpoint for transfer was ~724 M (also ~818 M) — 4/12 kills — but the
sim's own `best_mlp.pt`, picked on an 83 % sim kill-rate at ~598 M, lands 1/10 on
the real game. Consecutive checkpoints swing from "kills Letty a third of the
time" to "faceplants every run" while the sim eval barely moves. When the sim
eval can't tell a good policy from a bad one, the policy is exploiting structure
that only exists in the sim.

### Why the ceiling exists

Every anti-memorisation measure is an **affine transform of a fixed dataset**,
not new data:

| Measure | What it can't do |
|---|---|
| x-mirror | still the same 20 fights |
| rigid field rotation | a rotation of fight #3 is still fight #3 |
| mid-phase random starts | doesn't change what the bullets *are* |
| damage randomisation | doesn't change the pattern |

On the real game Letty's RNG produces **genuinely different** danmaku every run —
different spread widths, bullet counts, sub-wave timing. No flip or rotation of
20 recordings covers that. The policy learns "handle these 20 fights and their
symmetries", a narrow slice of "handle real Letty", and small weight changes that
don't hurt the sim wreck the transfer.

## What it left us

The dead end was productive:

- The **recorder**, FightSim's **collision / observation / damage-phasing** code,
  and the **transfer daemon** are all reused unchanged by [the plan](ecl-vm.md).
- The 20 recordings become **validation ground truth** for the VM — its output is
  checked against them at every stage.
- The re-aim failure is a direct argument for **generative** bullets: a bullet
  created from `(spawn, angle, speed)` can just be aimed at this episode's policy
  and integrated forward. Nothing to desync.
- "More recordings" (20 → 60–100) would widen the distribution for real, but only
  at the margin. Running Letty's **actual bytecode** so every episode is a novel,
  correct pattern is the real fix — [the plan](ecl-vm.md).
