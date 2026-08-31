# 6 · The observation

A 236-float vector: a scalar head, nine "escape" ray-cast scalars
(frames-until-hit if the player commits to each of nine directions), a local
danger grid, and blocks for the nearest enemies and items.

!!! note "Draft"
    **To write.** The escape-scalar geometry and why it turns "which way do I
    dodge" into an argmax. The danger-grid march. The K-nearest-128 pre-filter.
    Why the design survived four obs revisions. The parity harness. The
    `torch.compile` graph-break fix in the grid-march loop.
