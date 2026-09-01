# Address reference

Everything RE'd so far, for `th07.exe` v1.00b. Offsets marked "from" are relative
to the named base. Mirrors `native/th07_addrs.h`.

## Statics & functions

| Address | Name | Notes |
|---|---|---|
| `0x0062F958` | BULLET_MANAGER | pool at `+0xB8C0`, stride `0xD68`, 1025 slots |
| `0x009A9B00` | ENEMY_MANAGER | pool at `+0x4F50`, stride `0x4F48` |
| `0x00626270` | GAME_MANAGER | `zGlobals*` at `+0x08` |
| `0x00575950` | SUPERVISOR | game mode `+0x154` (2 = in a run) |
| `0x00575C70` | ITEM_MANAGER | stride `0x288`, 1100 slots |
| `0x004346E0` | fn `Window::do_tick` | hooked — one logic tick per call in STEP |
| `0x00430B50` | fn `read_input` | hooked — returns the agent action word |
| `0x004345C0` | fn `present` | hooked — skipped while driving |
| `0x0042FE20` | fn `run_all_on_draw` | hooked — stubbed while driving |
| `0x0043E260` | fn player–bullet collision | AABB; box = `pos ± size/2`, divisor `0x498A70` |
| `0x0043E3B0` | fn graze test | same box + 20 px (`0x498B84`) |
| `0x00435BFF` | single-instance guard `jnz` | patched to `jmp` at launch |
| `0x004348CC` | frame-limiter skip A | NOP'd after auto-nav |
| `0x00434997` | frame-limiter skip B | NOP'd after auto-nav |

## zBullet — from `BULLET_MANAGER + 0xB8C0`

| from | type | field |
|---|---|---|
| `+0xB7C` | `f32×2` | hitbox AABB (full size) |
| `+0xB8A` | `i16` | class |
| `+0xB8C` | `f32×3` | position |
| `+0xB98` | `f32×2` | velocity |
| `+0xBB0` | `f32` | speed |
| `+0xBB4` | `f32` | acceleration |
| `+0xBB8` | `f32` | angular velocity |
| `+0xBBC` | `f32` | angle (rad) |
| `+0xBFC` | `u16` | state (1–5 live) |
| `+0xC2C` | `f32` | `bullet_effects` p1 — redirect angle / accel |
| `+0xC30` | `f32` | p2 — redirect speed (−999 = keep) |
| `+0xC34` | `i32` | interval / duration |
| `+0xC3C` | `i32` | effect flag (16/32/64/128/256) |

## zEnemy — from a pool slot

| from | type | field |
|---|---|---|
| `+0x2B0C` | `f32×3` | position |
| `+0x2B3C` | `f32×3` | hitbox x, y, z (= ECL `enemy_set_hitbox` args) |
| `+0x2BB8` | `i32` | life |
| `+0x2BBC` | `i32` | max life |

## Supervisor / GameManager fields

| from | base | field |
|---|---|---|
| `+0x154` | SUPERVISOR | game mode (2 = in a run) |
| `+0x158` | SUPERVISOR | retry trigger — write `10` to reload the stage |
| `+0x08` | GAME_MANAGER | `zGlobals*` |
| `+0x95E8` | GAME_MANAGER | per-stage frame counter (→ 0 on reload) |
| `+0x95EC` | GAME_MANAGER | stage number |
| `+0x954598` | ENEMY_MANAGER | boss pointer array `zEnemy*[8]` |
