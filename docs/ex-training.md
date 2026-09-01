# Training & reward lessons

Four times the policy or the training setup did something we didn't intend. Each
one changed how the current system is built.

<div class="grid cards" markdown>

- __[Bolting shooting onto survival](de-shooting.md)__

    ---

    Every attempt to add "engage the boss / manage power" as a small reward term
    (v23–v26) was gamed — the policy found the survive-only local optimum and
    stopped shooting. Fixed only by a full reward-structure rethink: make the
    kill the goal and survival instrumental.

- __[Observation normalization](de-obsnorm.md)__

    ---

    `ppo_v28`. Standardising each obs feature by its sim variance folded the
    weights of sim-constant features up ~`1e4×`. Healthy sim curve, real transfer
    collapsed to ~18 s. Reverted; there is now **no** obs normalisation.

- __[Ranking checkpoints by sim score](de-checkpoints.md)__

    ---

    `best.pt` — the checkpoint with the best sim survival — is by construction
    the most overfit one in the run. Snapshot on a step schedule and rank by
    real-game transfer instead.

- __[Real-game fine-tuning](de-realgame.md)__

    ---

    12 M steps of genuine on-policy PPO against the retail game moved Stage 1
    survival not at all — it was already at the sim policy's ceiling. Real-game
    RL only pays off pointed at what the sim can't teach.

</div>
