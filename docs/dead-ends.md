# Extras

Everything off the main path — the approaches that didn't work, the lessons they
cost, and the engine-internals reference. The design of the working system is
mostly a fossil record of what's below: each dead end is why something downstream
looks the way it does.

## How this section is organised

<div class="grid cards" markdown>

- __[Training & reward lessons](ex-training.md)__

    ---

    Four ways the policy or the training setup fought back: gaming the reward,
    exploding on normalised inputs, overfitting the sim score, and wasting
    real-game rollouts on a solved stage.

- __[Reproducing real danmaku](ex-danmaku.md)__

    ---

    Getting a specific boss into the sim: the first ECL interpreter, the full
    [replay pipeline for Letty](de-letty-replay.md) (it landed real kills then
    plateaued), the re-aim detour, the proof-of-concept boss that never proved
    anything, and the
    [near-complete engine reimplementation](de-generative-danmaku.md) that hit
    ~2 px fidelity and still transferred at 0 %.

- __[Engine internals](ex-internals.md)__

    ---

    Reference material — the `th07.exe` memory map and the full address table.

- __[Experiment log](experiment-log.md)__

    ---

    The dated, quantitative companion: every training run, what changed, what
    the sim curve did, what transferred.

</div>

## The lessons, in one table

| What we tried | Verdict | Lesson |
|---|---|---|
| [First ECL interpreter](de-ecl-vm.md) | motion failed | don't reimplement an undocumented engine — record or measure its output instead |
| [Porting Letty into the sim](de-letty-replay.md) | real kills, then plateaued | 20 recordings + their symmetries is an affine transform of a fixed set, not real Letty; the sim score stops predicting transfer |
| [The re-aim saga](de-reaim.md) | all three abandoned | a re-aimed replayed bullet desyncs from the recording's slot bookkeeping; rigid field rotation was the escape hatch |
| [Generative danmaku — the ECL VM](de-generative-danmaku.md) | ~2 px fidelity, 0 % transfer | a reimplemented engine is still a fixed artefact; a 2003 x87-FPU game can't be reproduced bit-exact and the sub-pixel error compounds across a 500-bullet screen |
| [PoC boss choice](de-poc.md) | inconclusive twice | a proof-of-concept target has to be a fight the current policy *fails* |
| [Bolting shooting onto survival](de-shooting.md) | policy games it | you can't add "shoot" as a minor term to a survival objective; it finds the survive-only optimum |
| [Observation normalization](de-obsnorm.md) | catastrophic | sim-computed normalisation stats leak sim-only structure into the weights |
| [Sim-score checkpoints](de-checkpoints.md) | most-overfit checkpoint | snapshot every N M steps, rank by transfer tests, not sim score |
| [Real-game fine-tuning](de-realgame.md) | wash on a solved stage | real-game RL only pays off on what the sim policy *can't* do |

## The through-line

Every one of these is the same shape: **a fixed artefact — a hand-tuned sim
stage, 20 recordings and their symmetries, an entire engine reimplemented — that
the policy learns to exploit in ways the sim eval can't see.**
[Porting Letty into the sim](de-letty-replay.md) is that argument with the
receipts; the [ECL VM](de-generative-danmaku.md) is its strongest form — even a
2 px-faithful reimplementation is *an artefact*, and the policy found the seam.
[Why simulation isn't enough](ceiling.md) is the short version.

The response is not another simulator. It's to
[train the fight on the real game](de-realgame.md#the-letty-fight-revisited) —
the one danmaku source with no seam to exploit.
