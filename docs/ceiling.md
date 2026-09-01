# Why simulation isn't enough

Both ways of feeding the [simulator](sim.md) — procedurally generated danmaku and
[replayed recordings](recording.md) — plateau in the same place, and the failure
mode is the same.

## The plateau

| Approach | Best real transfer | Where it stops |
|---|---|---|
| procedural generation | ~225–231 s median (`ppo_v27` / `ppo_v29`) | clears Stage 1, dies in Stage 2 |
| recorded replay | ~100 s active-fight median, ~15 % real kill-rate (`fight_letty_seg` v9) | lands real Letty kills, but bimodal — consecutive checkpoints swing "kills Letty" ↔ "faceplants" |

Neither is a tuning problem. In both cases the **sim eval stops predicting real
performance**: the procedural sim's score keeps rising while transfer stalls or
regresses ([obs normalisation](de-obsnorm.md),
[fixed-stage overfitting](de-checkpoints.md)), and the replay sim's kill-rate
sits flat at 50–70 % while real checkpoints swing from 104 s to 2.8 s to 83 s.

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

## The response

Generate the danmaku instead: run each boss's **actual PCB bytecode**, so every
episode is a novel, correct pattern. The [first attempt](de-ecl-vm.md) at that
failed on bullet motion; [the plan](ecl-vm.md) gets the motion by hooking the
engine and measuring it, the same way recording did.
