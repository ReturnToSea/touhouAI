# Why replay hit a ceiling

The [record → FightSim → real](recording.md) pipeline works — a policy trained on
nothing but replayed Letty transfers to the real game and dodges real Letty. But
across a dozen training variants it plateaus in the same place, and the failure
mode points at a hard limit of the approach rather than a tuning problem.

## What "it works" looks like

`fight_letty_seg` — a from-scratch policy trained on ~20 recorded Letty fights
with synthetic damage-phasing — reaches, on the real game:

| | Sim eval | Real game |
|---|---|---|
| survival | ~120 s median | **bimodal** — many runs 100–130 s (into Spell 2), many 15–40 s, ~2–7 outright faceplants per checkpoint |
| kill-rate | 45–70%, stable | ~15 real kills across checkpoints 157–315 M |

That is a real improvement — a memorisation-prone earlier run got exactly **one**
real kill in a full billion steps; the version with the anti-memorisation bundle
got ~15 by 315 M. So the mitigations below help. They just do not close the gap.

## The tell: sim score stops predicting real score

The sim eval is stable — ~50–70% kill-rate the whole run. The **real** number is
not:

| Checkpoint | Real median | Faceplants | Real kills |
|---|---|---|---|
| 189 M | **104 s** | 2 | 1 |
| 236 M | **2.8 s** | 7 | 0 |
| 252 M | **83 s** | 0 | 2 |
| 315 M | 23 s | 1 | 0 |

Consecutive checkpoints swing from "kills Letty" to "faceplants every run" while
the sim eval barely moves. When the sim eval can no longer tell a good policy
from a bad one, the policy is exploiting structure that only exists in the sim.

!!! note "Measure the fight, not the intro"
    PCB puts ~42 s of boss entrance + un-skippable dialogue before Letty's first
    bullet. Early transfer numbers counted it. The real fight numbers above
    exclude it — the timer starts when bullets appear. A quoted "102 s" was
    really ~60 s of danmaku.

## Why the ceiling exists

Every anti-memorisation measure we added is an **affine transform of a fixed
dataset**, not new data:

| Measure | What it does | What it can't do |
|---|---|---|
| x-mirror | flips the fight left↔right (10 recs → 20) | still the same 20 fights |
| rigid field rotation ±10° | rotates the whole danmaku field per episode | a rotation of fight #3 is still fight #3 |
| mid-phase random starts | drops the policy in at any frame with random HP | doesn't change what the bullets *are* |
| damage randomisation | fast/slow kill, 20% pure-survival episodes | doesn't change the pattern |

On the real game Letty's RNG produces **genuinely different** danmaku every run —
different spread widths, bullet counts, sub-wave timing. No flip or rotation of
20 recordings covers that. The policy learns "handle these 20 fights and their
symmetries", which is a narrow slice of "handle real Letty", and small weight
changes that don't hurt the sim can wreck the transfer.

!!! warning "The re-aim detour"
    Re-aiming replayed bullets at the *live* policy — instead of the recorded
    player — seemed like the fix for the biggest memorisation vector. Three
    implementations, all abandoned: a rotated bullet's screen-lifetime stops
    matching the recording's slot bookkeeping, so bullets vanish mid-screen or
    flicker. The full story is in
    [The re-aim saga](de-reaim.md). Rigid field rotation was
    the escape hatch — it moves every bullet without changing any trajectory.

## What actually moves the needle

1. **More recordings.** 20 → 60–100. More RNG samples is a genuinely wider
   distribution, not symmetries of a fixed one. Cheap now that the re-aim
   machinery is gone (~8 GB of VRAM freed). Helps at the margin.
2. **Generative danmaku.** Run Letty's actual bytecode so every episode is a
   novel, correct pattern. This is [the plan](ecl-vm.md).

The [old ECL VM attempt](de-ecl-vm.md) failed on bullet *motion*, not on the VM. The new attempt gets motion by hooking the
game and measuring it, the same way [recording](recording.md) got bullet physics
without understanding `bullet_effects`.
