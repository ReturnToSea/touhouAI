# Danmaku-generation attempts

The recurring problem: produce patterns that are **novel every episode** (so
there's nothing to memorise) but **still correct** (so they transfer). These are
the attempts that didn't get there — and that motivate [the plan](ecl-vm.md).

<div class="grid cards" markdown>

- __[The first ECL interpreter](de-ecl-vm.md)__

    ---

    Parse every boss's bytecode, run it in a CPU VM, emit a spawn schedule. The
    VM's control flow worked; **bullet motion** did not — the undocumented
    multi-slot `bullet_effects` system lives in the executable, not the script.
    Shelved for [recording](recording.md); revived by
    [the plan](ecl-vm.md) with hook-and-measure instead of static RE.

- __[The re-aim saga](de-reaim.md)__

    ---

    Three implementations of "point replayed aimed bullets at the *live* policy"
    instead of the recorded player. All abandoned: a re-aimed bullet's
    screen-lifetime stops matching the recording's slot bookkeeping, so bullets
    vanish or flicker. Rigid per-episode field rotation was the escape hatch.

- __[Choosing a proof-of-concept boss](de-poc.md)__

    ---

    Cirno, then Letty — both fights the generic procedural policy already
    cleared, so "the recorded policy is better" was never a measurable claim. A
    PoC target has to be a fight the current policy *fails*.

</div>
