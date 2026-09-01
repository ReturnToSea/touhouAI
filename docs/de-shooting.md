# Bolting shooting onto survival

**Verdict:** every attempt to add "engage the boss / manage power" as a small
reward term to a survival objective was gamed by the policy. It took a full
reward-structure rethink — and one over-correction — to get a policy that both
dodges and kills.

## v23–v25 — the policy stops shooting

Added a front-only shot model, aimed enemy bursts, and a top-down "spam phase" to
push the agent to engage enemies. Over 1000 M steps `ppo_v25` **regressed to
~159 s**. The kill and power rewards were too weak relative to the survival
bonus, so the optimal policy was to *stop shooting and just dodge* — fine in the
sim, useless on a real run where you need power for later stages.

> You cannot bolt "shoot enemies" onto a survival objective with a small reward
> term. The agent finds the survive-only local optimum.

## v26 — stronger rewards, overfits

Cranked the enemy-damage reward (`0.10 → 0.35`), item reward (`0.30 → 0.60`),
added a power-standing bonus, capped episodes at 240 s, split the death signal by
cause. On a **fixed** sim stage it learned to engage — and transfer got *worse*
as the sim score climbed. (That's the [checkpoint lesson](de-checkpoints.md) and
the [transfer-ceiling](ceiling.md) argument in miniature.)

## `fight_letty_seg` — remove the survival reward entirely

The [damage-phased Letty sim](recording.md#synthetic-damage-phasing) went the other way:
**no per-frame survival reward at all**. Just

```
+ DMG_REW · boss-HP drained      (dense progress signal)
+ KILL_BONUS on final-phase kill
− HIT_PEN on death
```

The reasoning: rewarding survival *and* a kill is self-defeating — the survival
tick pays the policy to drag the fight out. With survival only instrumental (you
survive in order to keep dealing damage), the policy learned to damage-phase.
Kill-rate climbed from ~19% to a noisy 50–70% in the sim, and real-game kills
started happening — where the memorisation-prone runs got one kill in a billion
steps.

## The over-correction — put a small survival term back

Pure "no survival reward" left the 60–70% of episodes that die before they can
kill with **no gradient** — nothing telling them "don't die in Lingering Cold".
Fix: a small floor, `+0.004/frame` alive — about 30× smaller than the
aligned-shooting damage reward, so it can't cause stalling, but it gives the
dying-early majority something to climb.

## The lesson

> A composite "dodge *and* shoot" objective is not a weighting problem. Decide
> which one is the goal and which is instrumental. Here: **kill the boss** is the
> goal, survival is instrumental (dead policies deal no damage), so damage is the
> dense reward and survival is a thin floor — not the other way around.

## Bonus: the reward oscillation

Even with the kill-focused reward, consecutive checkpoints swung wildly on the
real game (median 104 s → 2.8 s → 83 s) while the sim eval stayed flat. Stability
fixes: γ `0.997 → 0.995`, entropy `0.004 → 0.002`, cosine LR decay, and a
`best_mlp.pt` saved on eval-score peaks so a later training thrash can't destroy
the good policy. Helped the *sim* stability; the real-game swing is the
[ceiling](ceiling.md), not the optimiser.
