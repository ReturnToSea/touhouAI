# Reverse-engineering th07

`th07.exe` is a 32-bit DirectX 8 binary from 2003 with no meaningful ASLR. Every
manager and pool is at a static address; every struct field is a fixed offset.
Most of the map below was found by probing — correlating struct floats against
observed per-frame motion, or scanning memory for values we already knew from the
scripts.

## The managers

| Address | Manager | Holds |
|---|---|---|
| `0x0062F958` | BulletManager | the 1025-slot bullet pool at `+0xB8C0`, per-type templates |
| `0x009A9B00` | EnemyManager | the enemy pool at `+0x4F50`, the boss pointer array at `+0x954598` |
| `0x00626270` | GameManager | `zGlobals*` at `+0x08`, per-stage frame counter at `+0x95E8`, stage number at `+0x95EC` |
| `0x00575950` | Supervisor | game mode at `+0x154`; the retry trigger at `+0x158` |
| `0x00575C70` | ItemManager | the item pool, stride `0x288`, 1100 slots |

## zBullet — stride 0xD68, 1025 slots

A flat array. State `1–5` means live; `0` is a free slot; `6` is a dying
sentinel. A bullet keeps its slot for its whole lifetime, which is what makes
velocity-diffing and per-bullet tracking possible.

| Offset | Type | Field |
|---|---|---|
| `+0x0B7C` | `float×2` | hitbox AABB, full size — collision uses ±½ (see [Collision](collision.md)) |
| `+0x0B8A` | `int16` | bullet class — stage 1: 3 ball, 4/5 pellet |
| `+0x0B8C` | `float×3` | position x, y, z |
| `+0x0B98` | `float×2` | velocity — RE'd by diffing position frame to frame |
| `+0x0BB0` | `float` | speed |
| `+0x0BB4` | `float` | acceleration (0 unless a speed effect is active) |
| `+0x0BB8` | `float` | angular velocity |
| `+0x0BBC` | `float` | angle, radians |
| `+0x0BFC` | `uint16` | state |
| `+0x0C2C…` | mixed | live `bullet_effects` state — redirect angle/speed, interval, flag (16/32/64/128/256). The hang & curve state is *in the struct*; read it each frame and you capture the motion with zero interpretation. |

## zEnemy — stride 0x4F48

Bosses are ordinary enemies; `EnemyManager + 0x954598` is just a pointer array
into the pool. Letty's `life` field reads `1` early in her fight and only becomes
`15000` once she "activates" — so satellite enemies and the boss are
distinguished by *hitbox size*, not life.

| Offset | Type | Field |
|---|---|---|
| `+0x2B0C` | `float×3` | position |
| `+0x2B3C` | `float×3` | hitbox x, y, z — **exactly the arguments to the ECL `enemy_set_hitbox` call**. Lingering Cold orbs read `(0, 0, 32)`; the lethal orbs read `(8, 8, 32)`; Letty's body `(48, 64, 32)`. |
| `+0x2BB8` | `int32` | life |
| `+0x2BBC` | `int32` | max life |

!!! note "Probe method"
    `native/probe_bullet_motion.py` steps the game one frame at a time and
    correlates every candidate float against the observed position delta.
    `native/probe_enemy_hitbox.py` drives to a boss, waits for the satellite-orb
    burst, and scans each live enemy's struct for the values the ECL says the
    hitbox should be. When the scan finds `0x42000000` (32.0) at `+0x2B44` on an
    orb whose script calls `enemy_set_hitbox(0, 0, 32)`, that is the field.
