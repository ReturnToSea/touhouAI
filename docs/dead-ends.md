# 11 · Dead ends & lessons

The current design is mostly a fossil record of these. Each one cost real time;
each one is why something downstream looks the way it does.

## The ECL interpreter — *reference only*

Plan: parse every boss script, run it in a CPU VM, get a spawn schedule to feed
the sim. The parser and phase-chaining worked. The *motion* did not — TH07's
bullet-type launch data, difficulty coefficients, and multi-slot `bullet_effects`
are undocumented, and reversing them is weeks of work. Watching the output: "way
too fast, they don't pause, the wrong ones rotate."

> **Lesson.** Don't reimplement an undocumented engine. Record its output instead
> — the hang/curve state is already sitting in the bullet struct.

## Observation normalization (v28) — *abandoned*

Standardizing each obs feature to zero mean / unit variance looked principled.
But features that are constant in the sim (empty item slots, walls, a sim-only
zero) got divided by a near-zero std — folding weights up by `1e4×`. On real-game
obs, where those features aren't constant, the network exploded. Real transfer
fell to 15–40 s despite a healthy sim curve.

> **Lesson.** Normalization statistics computed in the sim are a hidden channel
> for sim-only structure to leak into the weights. Removed entirely; kept
> reward-norm and LR annealing.

## Cirno as a proof-of-concept target — *inconclusive*

First test of the recording pipeline: does a policy trained on replayed Cirno
beat the procedural-sim policy at real Cirno? Both cleared the fight at ~66 s,
identically. Cirno is too easy — generic dodging already handles her, so there
was no gap for the recordings to close.

> **Lesson.** A PoC target has to be a fight the current policy *fails*. We then
> repeated the mistake with Letty.

## Letty as a proof-of-concept target — *inconclusive*

`fight_letty` transferred — 150–190 s on real Letty. But the baseline, a plain
procedural-sim policy on pure-dodge, cleared real Letty **6 / 6**. The recorded
policy was actually *worse* (3 / 6). No room to show benefit where the generic
policy already scores 100%.

> **Lesson.** The pipeline is validated (it transfers). Whether recordings *help*
> is still unanswered — needs a Stage 3+ boss the policy can't already dodge.

## Best-checkpoint selection — *abandoned*

`best.pt` is ranked by sim score. But transfer and sim score diverge — v26 got
*worse* on the real game as its sim number rose. So `best.pt` is, by
construction, the most overfit checkpoint in the run.

> **Lesson.** Snapshot every N million steps and transfer-test the snapshots. Sim
> score does not track transfer.

## Real-game fine-tuning on Stage 1 — *wash*

12M steps of real-game PPO on Stage 1: 238 s → 223 s greedy survival — flat.
Score doubled, survival didn't. Stage 1 was already at the sim policy's ceiling;
there was nothing left to fine-tune.

> **Lesson.** Real-game RL can't improve an already-solved stage. Its value is on
> the parts the sim policy fails, not the parts it passes.

## The base-policy regression — *open*

`ppo_v12` reached ~470 s (stages 1–3) on a 212-d observation that no longer
exists. The current 236-d runs (v27, v29) transfer to **~225 s median** — they
clear Stage 1 and die somewhere in Stage 2. A regression from v12, though not
the "dies at Stage 1 boss" I first reported (that was `best.pt`, which the
[v26 lesson](#v26-stronger-engagement-rewards-overfits) says is the wrong
checkpoint to pick).

Every domain-randomized run plateaus at roughly the same place. Pure-dodge, with
no shooting at all, dies in the Stage 1 *stage portion* at ~33 s — the
procedural sim never taught enemy management.

> **Lesson (provisional).** The sim teaches boss-dodging and tops out around
> "into Stage 2." A 1cc also needs stage-portion enemy management and boss
> damage-routing, and recording bosses fixes neither. The path being taken:
> record bosses for dodging *and* add synthetic HP phasing so the policy learns
> to shoot.
