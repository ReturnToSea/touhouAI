# What the agent sees

The agent never sees the screen. Every frame it receives a **236-float
observation vector** built from process-memory reads: a scalar head, nine
"escape" ray-casts, a local danger grid, and blocks for the nearest enemies and
items. The design is deliberately hand-engineered — the spatial reasoning is
precomputed into the features so a [small MLP](ch-policy.md) is enough.

## The pieces

### Scalar head

Player position and velocity, the focus bit, power, lives, bombs, whether a boss
is present and its HP fraction, and the per-stage frame counter. Cheap context
the policy needs but can't derive from bullet geometry alone.

### Nine escape scalars

For each of the **9 movement choices** (8 compass directions + "stay"), a
short forward simulation asks: *if the agent commits to this direction, how many
frames until it gets hit?* The nine answers turn the entire question "which way
do I dodge" into an **argmax over nine numbers** — the most important nine
numbers in the vector. They are recomputed with the same bullet physics the
[simulator](sim.md) uses, and they are **focus-aware**: the ray-cast uses the
speed the agent would actually move at given the current focus bit.

### Local danger grid

A grid of cells centred on the player. Each cell is marched against every nearby
bullet's trajectory and scored by **time-to-impact / local density** — a
coarse "heat map" of where the curtain is closing in. This is the feature that
gives the policy its picture of pattern *shape* rather than individual bullets.

### Nearest-enemy and nearest-item blocks

Up to a fixed number of the closest enemies (position, velocity, hitbox, life)
and items (position, type). Enemies are pre-filtered by a **K-nearest cut over
the 128 candidate slots** before the block is filled, so the vector length is
fixed regardless of how full the screen is.

## Parity

The DLL rebuilds the full 236-float vector in C so the real game can be driven at
[~80× real-time](hook.md). `test_obs_parity.py` checks it against the Python
builder byte-for-byte — current difference **`0.00000`**. Any drift here is a
silent transfer bug, so the check runs on every build.

## History

The vector has been revised four times (212 → 236 dimensions); the escape-scalar
+ danger-grid + nearest-blocks skeleton survived every revision. A
`torch.compile` graph-break in the grid-march loop — which had quietly halved sim
throughput — was fixed in the process (see [PPO](ppo.md)).

!!! warning "Features that are constant in the sim"
    A few slots (empty item blocks, wall-distance scalars in configurations the
    sim never produces) never vary during training but do on the real game.
    Normalising them by their sim variance was
    [catastrophic](de-obsnorm.md) — it is why there is **no observation
    normalisation** in the current design.
