# 8 · The GPU danmaku sim

A hand-written bullet-hell simulator that runs thousands of episodes as one
batched tensor op. Bullet motion is ported from PyTouhou's `Bullet.update()`;
patterns are procedurally generated with heavy domain randomization.

!!! note "Draft"
    **To write.** The emitter model. Domain randomization and why v27's version
    transferred well (200–500 s). The bullet-type → {hitbox, draw-size} table.
    Why procedural danmaku tops out around stages 1–3 on transfer.
