# Reproducing real danmaku

The recurring problem: get a *specific* boss's patterns into the training sim —
novel every episode (nothing to memorise) but still correct (so they transfer).
Three attempts, in order. The third got furthest, and where it stopped is what
[the plan](ecl-vm.md) responds to.

<div class="grid cards" markdown>

- __[The first ECL interpreter](de-ecl-vm.md)__

    ---

    Parse the boss's bytecode, run it in a CPU VM, emit a spawn schedule. The
    VM's control flow worked; **bullet motion** did not — the undocumented
    multi-slot `bullet_effects` system lives in the executable, not the script.

- __[Porting Letty into the sim](de-letty-replay.md)__

    ---

    The full replay pipeline, end to end: record ~20 real Letty fights, align
    them, detect phases, synthesise boss HP, replay on the GPU, and fight
    memorisation four ways. It landed the first real boss kills — then plateaued
    because 20 recordings and their symmetries aren't real Letty. **The main
    postmortem.**

- __[The re-aim saga](de-reaim.md)__

    ---

    A detour within the replay pipeline: three implementations of "point replayed
    aimed bullets at the *live* policy", all abandoned. A re-aimed bullet's
    screen-lifetime stops matching the recording's slot bookkeeping.

- __[Choosing a proof-of-concept boss](de-poc.md)__

    ---

    Cirno, then Letty — both fights the generic procedural policy already
    cleared, so "the recorded policy is better" was never a measurable claim. A
    PoC target has to be a fight the current policy *fails*.

</div>
