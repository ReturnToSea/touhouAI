# Choosing a proof-of-concept boss

**Verdict:** inconclusive, twice. The [recording pipeline](recording.md) was
validated — it transfers — but whether recordings *help* over the generic
procedural policy was never shown, because both PoC targets were fights the
generic policy already handled.

## Cirno — inconclusive

First test of the pipeline: does a policy trained on replayed Cirno beat the
[procedural-sim](sim.md) policy at real Cirno?

Both cleared the fight at ~66 s, **identically**. Cirno is too easy — generic
dodging already handles her, so there was no gap for the recordings to close.

## Letty — inconclusive

`fight_letty` transferred: 150–190 s on real Letty, essentially clearing the
fight. But the baseline — a plain procedural-sim policy on pure-dodge — cleared
real Letty **6 / 6**. The recorded-boss policy was actually *worse*, 3 / 6.

No room to show benefit where the generic policy already scores 100%.

## The lesson

> A proof-of-concept target has to be a fight the current policy **fails**. Then
> "the recorded policy beats the generic one" is a measurable claim. We picked
> Cirno, learned nothing, and repeated the mistake with Letty.

The right target is a Stage 3+ boss (Alice, Prismriver, Youmu) that the
procedural policy dies to. Getting there needs god-mode recording — a weak driver
survives to a deep boss with the invincibility flag on
([recording](recording.md)).

Note that the later `fight_letty_seg` runs — the ones with synthetic
damage-phasing and the anti-memorisation bundle — did move past "just dodging":
they [land real kills on Letty](ceiling.md). But the deeper point stands: Letty
is a poor yardstick because the floor is already so high.
