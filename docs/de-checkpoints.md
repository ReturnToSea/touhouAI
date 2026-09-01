# Ranking checkpoints by sim score

**Verdict:** `best.pt` — the checkpoint with the best sim survival — is, by
construction, the most overfit checkpoint in the run. Abandoned.

## The observation

`ppo_v26` was trained on a fixed sim stage. As training went on, its **sim
survival rose** and its **real-game transfer fell**. The `best.pt` it saved —
picked on the peak sim number — transferred worse than checkpoints from halfway
through the run.

`fight_letty_seg` (v9) made this unmistakable: the sim eval sat at ~50–70%
kill-rate the whole run, while consecutive real-game checkpoints swung from
"median 104 s, kills Letty" to "median 2.8 s, faceplants every run". The sim
score genuinely could not tell a good policy from a bad one.

## Why

A sim — especially a fixed or narrowly-randomised one — is a **memorisation
target**. The longer you train, the more the policy fits sim-specific structure
that the sim eval rewards and the real game punishes. The sim score and the
transfer score are measuring different things, and past a point they anti-correlate.

## The lesson

> Snapshot every N million steps (`snap_*.pt`, `mlp_*M.pt`). Rank checkpoints by
> **transfer tests** — run them on the real game — not by sim score. Keep a
> `best_mlp.pt` if you like, but treat it as one candidate, not the answer.

The [real-game transfer daemon](de-letty-replay.md#the-transfer-daemon) exists
for exactly this — it plays every new checkpoint against the real boss and logs
the result, so the "which checkpoint" question has a data-driven answer instead
of a sim number.

See also [Why simulation isn't enough](ceiling.md) — this is one symptom of the
same underlying problem.
