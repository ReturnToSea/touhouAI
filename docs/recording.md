# Recording & the replay sim

The [procedural sim](sim.md) can't reproduce a specific boss. The
[ECL interpreter](de-ecl-vm.md) couldn't reproduce the engine. So the third
approach: record the real game's bullet positions frame by frame and replay them
on the GPU.

This is the pipeline that clears Stage 1 and lands real kills on Letty. It also
[hits a ceiling](ceiling.md) — which is why [the plan](ecl-vm.md) exists — but it
is the current data source and the validation ground truth for everything after.

## Why replay beats interpretation

A bullet's hang, curve, and redirect state all live in the
[zBullet struct](re.md#zbullet-stride-0xd68-1025-slots) at `+0xC2C` onward.
Reading every live bullet's position each frame captures that motion exactly,
with no need to understand the undocumented `bullet_effects` semantics that sank
the ECL VM.

## The recorder

`record_boss_driven.py` launches the [hooked game](hook.md), drives to the target
boss with a policy (god-mode on, so a weak driver reaches a deep boss), and from
the moment the boss pointer goes valid until it goes null again, logs per frame:

```python
# per live bullet
(step, slot, x, y, vx, vy, class, fx_flag, hitbox_x, hitbox_y)
# per live enemy — satellite orbs and the boss
(step, slot, x, y, life, hb_x, hb_y, hb_z)
# plus the boss track and the player track
```

Start and stop are memory-driven — the boss appearing and despawning — not a
fixed timer. A full dodge-only Letty fight is **~224 s**: ~42 s of boss entrance
and un-skippable dialogue (no bullets), then ~179 s / 10,750 frames of danmaku,
ending when the last spell times out and she is defeated.

!!! note "Everything aligns to the first bullet"
    The boss / player / enemy logs start ~42 s before the first bullet (the
    dialogue). `_load_dense` aligns every array to the first-bullet frame — an
    earlier bug aligned each to its own first frame, a 42 s skew that pointed
    re-aimed bullets at a stale player position for the whole fight.

## Phase detection

The recordings are dodge-only, so every phase times out and clears the screen.
`boss_phases.phase_windows()` finds those screen-clears — bullet count collapsing
to ~0 — per recording and snaps them to Letty's known phase times. Consistent
across all 20 recordings to within ~15 frames:

| Phase | Attack frames | Repositioning lull before it |
|---|---|---|
| Non-spell 1 | 0 – 38 s | — |
| Lingering Cold | 44 – 89 s | 5.5 s |
| Non-spell 2 | 93 – 129 s | 3.8 s |
| Table-Turning | 132 – 179 s | 3.0 s |

Between `clear_start` and `first_attack` the boss is repositioning / declaring —
she deals **and takes** no damage (armored).

## Synthetic damage-phasing

The recordings have no boss HP — the recorder logs only `(step, x, y)` for the
boss, and Letty's `life` field reads `1` for most of the fight. So per phase the
sim synthesises an HP pool (`SHOT_DPS · attack_duration · KILL_FRAC`) and lets
the agent drain it by shooting from under the boss.

ReimuA's shot is modelled as **20% homing** (lands from anywhere) + **80% forward
needle** (lands only when lined up in x under the boss, ±17.5 px). Draining a
phase, or its recorded timer expiring, screen-clears the bullets and jumps the
recording to the next phase's start. Beating the last phase ends the episode with
a kill bonus. See [Bolting shooting onto survival](de-shooting.md) for the reward
structure.

!!! warning "The HP numbers are guessed"
    `KILL_FRAC` and `SHOT_DPS` are calibrated to give plausible phase lengths,
    not measured. [The ECL VM plan](ecl-vm.md#7-hp-life-callbacks-spellcard-phases-1-day-autonomy-82)
    replaces them with Letty's real `enemy_life_set` thresholds.

## FightSim

The recordings pack into a `[n_rec, F, 1025, 2]` position tensor on the GPU. Each
of *B* parallel episodes picks a random recording and a start point; bullets
follow their exact recorded path. Player physics and AABB
[collision](collision.md) run on top, feeding the same `build_obs_batch` as the
procedural sim. ~130–270k env-frames/s depending on the feature set.

### Anti-memorisation

20 recordings is a narrow distribution. Four measures widen it *without*
distorting any trajectory:

| Measure | What it does |
|---|---|
| **x-mirror** | flips the fight left↔right — 10 recordings become 20 |
| **rigid field rotation** | rotates the *whole* danmaku field (bullets + orbs + boss) by one random ±10° angle per episode, about screen-centre. A rigid rotation preserves every trajectory and gap exactly |
| **mid-phase random starts** | 50% of episodes drop in at any frame of a phase with a random slice of HP left — a phase can't be memorised as a fixed sequence |
| **damage randomisation** | random ±50% DPS multiplier per episode + 20% of episodes deal zero damage (pure survival — forces training on the parts of a phase a fast kill skips) |

Per-bullet re-aiming — pointing aimed bullets at the live policy — was tried
three times and abandoned; the [re-aim saga](de-reaim.md) is the whole story.
Rigid field rotation is what replaced it.

## The transfer daemon

`fight_transfer_daemon.py` runs a persistent hooked game alongside training. Each
time a new checkpoint drops it plays ~12–15 real Letty fights, timed from the
first bullet (dialogue excluded), and appends
`[wall, steps, active_survival_s, killed, damage_frac]` to `realtransfer_*.npy`.
Multiple daemons run in parallel with different tags; `fight_hud.py` merges them
into a live real-transfer curve next to the sim curve.

This exists because [the sim score doesn't predict transfer](de-checkpoints.md) —
the daemon gives the "which checkpoint is actually good" question a data-driven
answer.
