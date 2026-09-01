# The re-aim saga

**Verdict:** three implementations of "point replayed aimed bullets at the *live*
policy", all abandoned. A re-aimed bullet's screen-lifetime stops matching the
recording's slot bookkeeping — it vanishes mid-screen or flickers. Rigid
per-episode field rotation was the escape hatch, and it's what runs now.

## The problem it was trying to solve

A [replayed](recording.md) aimed bullet points at wherever the *recording's*
player stood, not at the policy being trained. So an aimed stream always goes the
same way every episode — a clean memorisation vector: "the aimed fan is always to
the left, so start left." On the real game Letty aims at *you*, and the policy
has never had to actually dodge a tracking pattern.

Lingering Cold is `bullet_fan_aimed` [in the ECL](ecl.md#what-letty-actually-does)
— ~44 seconds of aimed fans. If any phase needed re-aiming right, it was that
one.

## Attempt 1 — the rolling player-history window

Keep a 320-frame ring buffer of the sim player's positions. An aimed bullet born
at sim-step *s* reads `pl_ring[s % 320]` and rotates its recorded path to point
at where the player was at spawn.

**Bug:** a hard `age < 320` cutoff. Once an aimed bullet was older than 320
frames (5.3 s), `use` went false and it **snapped back** to its un-rotated
recorded path — a discontinuous jump of 100–200 px. Lingering Cold's bullets are
slow orb streams that live well past 5.3 s, so lots of them snapped.

> *"Projectiles are despawning and then randomly appearing near the player."*

## Attempt 2 — spawn-locked rotation

Snapshot the rotation angle **once** at the bullet's birth (`slot_dth[B, POOL]`),
hold it for the bullet's whole life, no window, no cutoff. Verified: worst
same-bullet frame-to-frame jump dropped from ~200 px to **9 px** (normal bullet
speed). The snap was gone.

**Bug:** rotating a bullet's whole recorded path about its spawn point sends it
somewhere the recording never tracked. Its screen-lifetime no longer matches the
recording's. When `pos[frame, slot]` goes `NaN` — the recording stopped tracking
the *un-rotated* bullet — the re-aimed bullet despawns, even though it's still
mid-screen and "should" be flying.

## Attempt 3 — straight-line re-aim

An aimed bullet becomes a genuine straight line: `spawn + dir(snapshot_aim) ·
speed · age`, which is what `bullet_fan_aimed` actually *is*. It lives until it
flies off-screen or the recording hands its slot to a new bullet.

**Bug:** the recording churns slots fast. In Lingering Cold a slot's bullet
despawns and a new one spawns in it ~13 frames later, ~60 times a fight. The
"slot reused → evict my re-aimed bullet" rule fired constantly, so re-aimed
bullets got evicted mid-flight — right in front of the player.

> *"Projectiles are despawning and then randomly appearing later. None of this is
> looking right."*

## Aggressive re-aim made it worse

Separately, we tried marking **every** bullet in Lingering Cold as aimed (from
the ECL), not just the ~5% the 24° velocity heuristic catches. That took the
in-recording aimed fraction from ~5% to ~44%. It also took the desync problem
from "a few bullets flicker" to "a quarter of the fight flickers".

## The resolution — rigid field rotation

Drop per-bullet re-aim entirely. Instead, each episode, rotate the **whole
danmaku field** — every bullet, every orb, the boss — by one random angle (±10°)
about screen-centre.

A rigid rotation of the whole field:

- preserves **every** trajectory and **every** gap, exactly — no bullet's
  lifetime, speed, or shape changes
- makes the absolute positions different every episode, so there's nothing fixed
  to memorise
- verified: worst same-bullet frame jump 9 px, no snapping, no flicker

It's not "the bullets aim at you" — but combined with x-mirror, mid-phase random
starts, and damage randomisation, the pattern the policy has to solve is
different enough each episode.

**Bonus:** deleting the re-aim machinery — `spawn`, `aimed`, `launch_ang`,
`birth`, and the per-episode rotation buffers — freed **~8 GB of VRAM** and sped
training from 94k to 136k env-frames/s.

## The lesson

> Re-aiming replayed bullets can't work in a slot-based replay architecture: a
> re-aimed bullet's lifetime no longer matches the recording's slot bookkeeping,
> and every fix for that reintroduces the flicker somewhere else. If you need
> per-bullet player-tracking, you need **generative** bullets — the bullet is
> created from `(spawn, angle, speed)`, so you just set the angle and integrate
> forward. That's [the plan](ecl-vm.md), and re-aiming is one of the reasons for
> it.
