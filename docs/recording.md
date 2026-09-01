# Recording real bosses

The [procedural sim](sim.md) can't reproduce a *named* boss, and the
[first ECL interpreter](de-ecl-vm.md) couldn't reproduce the engine. The third
option is the simplest: record the real game's bullet positions frame by frame,
and keep them.

Those recordings did two jobs. They were the training data for the
[replay experiment](de-letty-replay.md) — the run that landed the first real
kills on a boss and then plateaued. And they are the **validation ground truth**
for [the plan](ecl-vm.md): the VM's output is checked against them at every
stage.

## Why recording captures the motion

A bullet's hang, curve, and redirect state all live in the
[zBullet struct](re.md#zbullet-stride-0xd68-1025-slots) at `+0xC2C` onward.
Reading every live bullet's *position* each frame captures that motion exactly,
with no need to understand the undocumented `bullet_effects` semantics that sank
the [ECL VM](de-ecl-vm.md).

## The recorder

`record_boss_driven.py` launches the [hooked game](hook.md), drives to the target
boss with a weak policy that has **god-mode on** (so a poor driver still reaches
a deep boss), and from the frame the boss pointer goes valid to the frame it
goes null, logs per frame:

```
per live bullet   (step, slot, x, y, vx, vy, class, fx_flag, hitbox_x, hitbox_y)
per live enemy    (step, slot, x, y, life, hb_x, hb_y, hb_z)   # satellite orbs + boss
plus              the boss track and the player track
```

Start and stop are **memory-driven** — the boss appearing and despawning — not a
fixed timer. A full dodge-only Letty fight is **~224 s**: ~42 s of boss entrance
and un-skippable dialogue (no bullets), then ~179 s / 10,750 frames of danmaku,
ending when the last spell times out. About 20 Letty fights were recorded.

## What the recordings feed

| Consumer | Use |
|---|---|
| [the replay experiment](de-letty-replay.md) | packed into a GPU tensor and replayed as a training environment — how, and why it didn't transfer, is that page |
| [the ECL VM plan](ecl-vm.md) | ground truth — spawn counts and timing (Part 5), sub-enemy tracks (Part 6), phase-transition frames (Part 7), the boss track (Part 8), and per-type motion traces (Part 10) all validate against these arrays |

!!! note "Two things to know about using them"
    The non-bullet logs start ~42 s before the first bullet (the dialogue), so
    every array has to be aligned to the first-bullet frame — an earlier
    misalignment cost 42 s of skew. And the recordings are dodge-only, so every
    phase times out and clears the screen, which is how phase boundaries are
    detected. Both are covered in
    [Porting Letty into the sim](de-letty-replay.md).
