# Why simulation isn't enough

Both ways of feeding the [simulator](sim.md) — procedurally generated danmaku and
[replayed recordings](recording.md) — plateau in the same place, and the failure
mode is the same.

## The plateau

| Approach | Best real transfer | Where it stops |
|---|---|---|
| procedural generation | ~225–231 s median (`ppo_v27` / `ppo_v29`) | clears Stage 1, dies in Stage 2 |
| recorded replay | 60 s active-fight median / 7 % kill-rate over 631 fights; best checkpoint ~103 s / 33 % (`fight_letty_seg` v9, full 1 B steps) | lands real Letty kills, but bimodal — consecutive checkpoints swing "kills Letty" ↔ "faceplants" |

Neither is a tuning problem. In both cases the **sim eval stops predicting real
performance**: the procedural sim's score keeps rising while transfer stalls or
regresses ([obs normalisation](de-obsnorm.md),
[fixed-stage overfitting](de-checkpoints.md)), and the replay sim's kill-rate
sits flat at 50–83 % while real per-checkpoint kill-rate swings between 0 % and
33 % with no trend across a billion steps.

## The reason

When the sim eval can't tell a good policy from a bad one, the policy is
exploiting structure that only exists in the sim. That structure is unavoidable
with either approach, because both train on a **fixed artefact**:

- procedural generation makes plausible bullets but no real boss *structure* — no
  phases, no spellcard declarations, no satellite-orb choreography;
- replay is 20 recordings, and every anti-memorisation measure
  ([mirror, rotation, mid-phase starts, damage randomisation](de-letty-replay.md#fighting-memorisation))
  is an affine transform of those 20 — not new data.

Real Letty's RNG produces genuinely different danmaku every run. No transform of
a fixed set covers that.

The full replay-training postmortem — every step, and the
checkpoint-by-checkpoint transfer numbers — is
[Porting Letty into the sim](de-letty-replay.md).

## The response that didn't work either

Generate the danmaku instead: run Letty's **actual PCB bytecode** in a
reimplemented VM, so every episode is a novel, correct pattern. The
[first attempt](de-ecl-vm.md) failed on bullet motion; the
[second](de-generative-danmaku.md) solved that by measuring the engine and got
the VM to a ~2 px per-bullet fidelity — and **still didn't transfer**. Eight
runs, a flat 0 % real kill-rate, worse than either simulator it replaced. A
reimplemented engine is still a fixed artefact; the policy found the seam between
it and reality just the same, and the seam — 80-bit x87 float error compounding
across 500 bullets — was wide enough to swallow the whole transfer.

## The response after that

Stop simulating Letty. Train the fight directly on
[the real game](de-realgame.md#the-letty-fight-revisited) — many hooked instances
running whole PPO trajectories in parallel, warm-started from the procedural-sim
policy. Slower per step, but there is no fidelity gap to fall through.
