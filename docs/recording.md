# 9 · Recording real fights

The procedural sim can't reproduce a specific boss. The ECL interpreter couldn't
reproduce the engine. So the third approach: record the real game's bullet
positions frame by frame and replay them on the GPU.

## Why replay beats interpretation

A bullet's hang, curve, and redirect state all live in the zBullet struct
(`+0xC2C` onward). Reading every live bullet's position each frame captures that
motion exactly, with no need to understand the undocumented `bullet_effects`
semantics that sank the ECL VM.

## The recorder

`record_boss_driven.py` launches the hooked game, drives to the target boss with
a policy, and from the moment the boss pointer goes valid until it goes null
again, logs per frame:

```python
# per live bullet
(step, slot, x, y, vx, vy, class, fx_flag, hitbox_x, hitbox_y)
# per live enemy (satellite orbs + the boss)
(step, slot, x, y, life, hb_x, hb_y, hb_z)
# plus the boss track and the player track
```

Start and stop are memory-driven — the boss appearing and despawning — not a
fixed timer. A full dodge-only Letty fight is **179 s / 10,750 frames**, ending
when the last spell times out and she is defeated.

## FightSim

The recordings pack into a `[n_rec, F, 1025, 2]` position tensor on the GPU. Each
of *B* parallel episodes picks a random recording and a random start offset;
bullets follow their exact recorded path. Player physics and AABB collision run
on top, feeding the same `build_obs_batch` as the procedural sim. After the
`torch.compile` pass, ~**273k env-frames/s**.

## Re-aiming

A replayed aimed bullet points at wherever the *recording's* player stood, not at
the policy being trained. Fix: a bullet is "aimed" if its spawn velocity points
within **24°** of the recorded player at spawn. Those bullets are rotated about
their spawn point to point at where the *sim* player was **at that spawn frame**
— then they fly straight. Spawn-locked, not tracking. `fight_play.py` is an
interactive viewer for eyeballing it.

!!! success "Transfer"
    `fight_letty`, trained from scratch on nothing but replayed Letty recordings,
    went to the real game and dodged real Letty for 150–190 s — essentially
    clearing the fight. The record → FightSim → real pipeline works.
