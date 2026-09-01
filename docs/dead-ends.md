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

- __[Danmaku-generation attempts](ex-danmaku.md)__

    ---

    The efforts to produce novel-but-correct patterns: the first ECL
    interpreter, the re-aim saga, and the proof-of-concept boss that never
    proved anything.

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
| [The re-aim saga](de-reaim.md) | all three abandoned | a re-aimed replayed bullet desyncs from the recording's slot bookkeeping; rigid field rotation was the escape hatch |
| [PoC boss choice](de-poc.md) | inconclusive twice | a proof-of-concept target has to be a fight the current policy *fails* |
| [Bolting shooting onto survival](de-shooting.md) | policy games it | you can't add "shoot" as a minor term to a survival objective; it finds the survive-only optimum |
| [Observation normalization](de-obsnorm.md) | catastrophic | sim-computed normalisation stats leak sim-only structure into the weights |
| [Sim-score checkpoints](de-checkpoints.md) | most-overfit checkpoint | snapshot every N M steps, rank by transfer tests, not sim score |
| [Real-game fine-tuning](de-realgame.md) | wash on a solved stage | real-game RL only pays off on what the sim policy *can't* do |

## The through-line

Every one of these is the same shape: **a fixed artefact — a hand-tuned sim
stage, an undocumented engine reimplemented, 20 recordings and their symmetries —
that the policy learns to exploit in ways the sim eval can't see.**
[The transfer ceiling](ceiling.md) makes that argument in full, and
[the plan](ecl-vm.md) is the response: generate the danmaku instead of
transforming a fixed set of it.
