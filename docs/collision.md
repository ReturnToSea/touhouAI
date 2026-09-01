# Collision

Touhou's reputation for a "tiny circular hitbox" is a simplification. The check
at `0x43e260` is an **axis-aligned box overlap**, and it is not only bullets that
kill you.

## The bullet check

Called per bullet from `BulletManager::on_tick`. The bullet's box is
`pos.axis ± size[axis] / 2` — the divisor `2.0` is a literal at `0x498a70`. The
player's box is a world-coordinate rectangle from the player struct. A hit sets
the bullet's state to 5. A separate graze test uses the same box widened by
**20 px** on each side, which matches PCB's visible graze range.

| Class | size @ `+0xB7C` | half-extent | What |
|---|---|---|---|
| `4` | `4.0` | `±2.0` | small pellet (fairy popcorn) |
| `3` | `6.0` | `±3.0` | medium ball (aimed shots, Letty orbs) |
| `5` | `4.0` | `±2.0` | rice variant |

Death-frame measurement puts the class-3 kill at ~4.8 px centre to centre
head-on, so the **player half-extent is ~1.8 px** (4.8 − 3.0).

## Enemy bodies kill the player

Confirmed against PyTouhou's faithful reimplementation: for any *collidable*
enemy — including the boss — `player.collide()` fires on overlap. The
player-versus-enemy box is **⅔ of the set hitbox**. `collidable` defaults to
true; only `enemy_flag_collision(0)` disables it.

So Letty's non-spell and Table-Turning orbs (`8×8`, collidable) instant-kill on
contact, right in the space you're trying to dodge through. Lingering Cold's orbs
(`0×0`, collision disabled) are harmless. The simulator models the lethal ones;
the boss body is currently excluded (point-blanking a PCB boss is legitimate).

!!! danger "Cost of getting this wrong"
    Before the sim modelled enemy bodies, a policy trained on it would fly
    straight through an orbiting orb and die instantly on the real game. This was
    a real, invisible chunk of the transfer gap.
