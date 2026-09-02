# th07.exe reverse-engineering notes

Static analysis of `th07.exe` v1.00b (unpacked, MD5 `0126afce1e805370d36c3482445e98da`,
ImageBase `0x400000`) with Ghidra 12.1.3 headless. Decompiled all 1573 functions
(`tools/ghidra_scripts/th07_dump.java` → `C:\Users\spore\th07_re\decomp_all.c`).

This is the reverse-engineering session the ECL-VM plan kept deferring. It settled
the bullet-motion model (the Part 10 blocker), confirmed the Part 8 orbit
implementation, and turned up the real danmaku PRNG (Part 4) and a `move_point`
easing correction.

## Key function addresses

| addr | what |
|---|---|
| `0x00425a50` | **per-frame bullet update loop** — iterates all 1024 slots |
| `0x00426f60` | bullet graphic / hitbox init per bullet type |
| `0x00423730` | **per-bullet spawn/instantiate** (the "bullet constructor") |
| `0x00424d20` | bullet-pattern emit wrapper (loops nway × nshots → `0x00423730`) |
| `0x00424290` | `bullet_effects` staging processor (reads the pending-effect list) |
| `0x004251a0` | fx flag `0x10` — directional acceleration |
| `0x00425310` | fx flag `0x20` — turn + accelerate |
| `0x00425400` | fx flag `0x40` — decelerate-pause-redirect |
| `0x00425700` | fx flag `0x80` — decelerate-pause-**re-aim at player** |
| `0x004258a0` | fx flag `0xc00` — **wall bounce** |
| `0x004325c0` | `set_vel_from(angle, speed)` → `vel = (cos,sin)(angle)·speed` |
| `0x00431930` | `norm_angle(a + b)` into `(-π, π]` |
| `0x00431870` | **danmaku PRNG** (16-bit) — see Part 4 below |
| `0x00431900` | `prng_float()` = `rand_u32() / 2³²` |
| `0x00410520` | enemy tick / ECL runner (29 KB) — contains the movement switch |
| `0x0043e260` | collision (AABB) — already known |
| `0x0043e3b0` | graze-box check |
| `0x62F958` | bullet manager base;  pool = `+0xB8C0`, slot stride `0xD68`, 1024 slots |

## Bullet motion model (Part 10 — solved)

### Bullet `state` (`+0xBFC`)

`0` empty · `1` live · `2/3/4` **spawning ("hang")** · `5` dying.

### The hang / launch (`FUN_00423730` + `FUN_00425a50`)

A bullet's flags word (`+0xBF6`, set from the ECL bullet-type argument) selects a
hang:

| flag bit | spawn state | on spawn |
|---|---|---|
| `0x2` | 2 | `pos -= vel · 4`, play materialise anim (slot `+0x24C`) |
| `0x4` | 3 | `pos -= vel · 4`, anim slot `+0x498` |
| `0x8` | 4 | `pos -= vel · 4`, anim slot `+0x6E4` |
| none | 1 | live immediately |

Per frame, the updater moves a hanging bullet at a **fraction** of its velocity:

```
state 2 → pos += vel · 0.5
state 3 → pos += vel · 0.4
state 4 → pos += vel · 0.333
```

while the materialise animation plays (length is data-driven from the `.anm`,
typically ~8–16 frames). When the anim completes → `state = 1`, full velocity.

**Net effect:** the bullet is placed 4 velocity-steps *behind* its nominal spawn
point and crawls forward at ⅓–½ speed for the anim's duration — covering exactly
those 4 steps — then releases from the nominal point. The `speed` field
(`+0xBB0`) never changes during the hang; the *actual* displacement is
`vel · ratio`.

The visible "launch" that looks like a spike above base is a **separate
effect** — `bullet_effects` flag 1 (see below) — not the hang.

`+0xBF0` is a separate "young" countdown: while `> 0` the bullet skips the
out-of-bounds despawn check (it can start off-screen).

### `bullet_effects` (`FUN_00424290` + per-flag handlers)

Up to 5 effect entries are staged in the bullet at `+0xC14` (0x18 bytes each).
Each frame, for each entry, the engine applies it **only if
`bullet.flags(+0xBF6) & entry.flag`**. `TS` below is the global timescale
(`DAT_00575AC8`, `1.0` in normal play).

- **flag `0x1`** (`FUN_004250d0`) — the **launch kick**. For 17 frames:
  `|vel| = speed + 5.0 · (1 − t/16)` (`t` = frames since arm). Fixed constants.
  Armed when the type-word has bit `0x1` *and* a flag-1 entry is staged (the
  `(−1,−1,−1,−1,1,0)` entry — not a "clear sentinel" as first assumed). This is
  the "hover then launch" and the Table-Turning spike.
- **flag `0x10`** — for `interval` frames: `vel += TS · p1 · (cos,sin)(dir)`,
  where `dir = p2`, or the bullet's **current heading** if `p2 ≤ -990`. `angle`
  is then recomputed as `atan2(vel.y, vel.x)` — so when `p1 < 0` drives the speed
  through zero the heading **flips 180° automatically**. This is Lingering
  Cold's decelerating snow. Modifies `vel` directly; the `speed` field is
  untouched.
- **flag `0x20`** — for `dur` frames: `angle += TS · p2`; `speed += TS · p1`;
  `vel = (cos,sin)(angle) · speed`. Turn + accelerate.
- **flag `0x40`** — decelerate linearly to 0 over `p3` frames, then `angle += p1`,
  `speed = p2` (keep current if `≤ -999`), repeat `p4` times. Pause-and-redirect.
- **flag `0x80`** — same as `0x40` but the new heading is **atan2 toward the
  player** `+ p1`.
- **flag `0xc00`** (`0x400 | 0x800`) — **wall bounce**. Playfield is
  **384 × 448**. `pos.x < 0 || ≥ 384` → `angle = -angle - π`; `pos.y < 0`
  (or `≥ 448` when `0x400` set) → `angle = -angle`. `speed` restored from a
  stored value; effect clears after `+0xD60` bounces. This is Table-Turning.

**`bullet_effects` arg 0 = enable bit** (already found before this session and
confirmed here): an entry with `flag == 0` in `FUN_00424290` falls through
without applying.

### Bullet-pattern angle/speed math (`FUN_00423730`, `mode` = param `+0xC0`)

Confirms the VM's `_emit_bullets`:

- mode 0/1 — fan: `angle = base + step·(i − (n−1)/2)`  (+ aim toward player if mode 0)
- mode 2/3 — ring: `angle = base + i·2π/n`  (+ layer term; + aim if mode 2)
- mode 4/5 — ring offset by `π/n`
- mode 6 — random angle in `[step, base)`
- mode 7 — random speed in `[spd2, spd1)`
- mode 8 — random both

Speed: `layers < 2` → `spd1`; else `spd1 − (spd1 − spd2)·layer/layers`.

## Enemy movement (Part 8)

`FUN_00410520`, `switch(enemyFlags(+0x2E28) >> 2 & 7)` inside movement-type 2/3:

### Orbit (movement type 3) — **confirms `__move_circle_abs`**

```
angle  (+0x2B5C) = norm_angle(angle + TS · angSpeed(+0x2B60))
radius (+0x2B6C) += TS · radiusGrowth(+0x2B70)
target  = centre(+0x2B8C..) + radius · (cos,sin)(angle)
vel     = target − pos           # then pos += vel
```

per frame for `duration(+0x2BA4)` frames, then the movement flag clears. **This
is exactly what `sim/ecl/vm.py`'s circle motion does** — the disassembly
validates the Part 8 orbit implementation.

### `move_point` / `move_dir_time` — arg 1 is an **easing mode** (VM currently ignores it)

Movement-type 2 interpolates `pos` from start to target over `duration` frames
with `progress` shaped by `enemyFlags >> 2 & 7`:

| mode | curve (applied to remaining fraction `x`) |
|---|---|
| 0 | linear |
| 1 | `x²` |
| 2 | `x³` |
| 3 | `x⁴` |
| 4 | `1 − (1−x)²`  (ease-out / decelerate into target) |
| 5 | `1 − (1−x)³` |
| 6 | `1 − (1−x)⁴` |

Letty's script uses mode 4 heavily (`move_point(60, 4, …)`, `move_dir_time(60, 4, …)`).
**The VM uses pure linear** → a real correction. (The boss-track verify still
passed for 127 frames because the pre-RNG moves it checks happen to be short /
mode-0-ish; the easing matters more over longer repositions.)

## Part 4 — the danmaku PRNG (our LCG is the wrong algorithm)

`sim/ecl/rng.py` implements an LCG (`state·0x343FD + 0x269EC3`). That constant
appears **once** in `th07.exe` — in the MSVC CRT `rand()` (`_holdrand`), which the
danmaku does **not** use.

The danmaku / ECL RNG is `FUN_00431870`, a 16-bit generator:

```
next16():                        # all arithmetic is uint16
    u     = (state ^ 0x9630) + 0x9AAD
    state = rotl16(u, 2)         # ((u & 0xC000) >> 14) | (u << 2)
    return state

rand_u32():  return (next16() << 16) | (next16() & 0xFFFF)   # FUN_004318D0
rand_float(): return rand_u32() / 2**32                       # FUN_00431900
```

The ECL random opcodes (`__math_rand`, `set_float_rand_bound*`, random bullet
modes) all funnel through `FUN_00431900`. `sim/ecl/rng.py` should be replaced
with this, then Part 4's KS test re-run.

## Recorder columns added from this (commit ad8401d)

`record_boss_driven.py` used to log only `bullet_effects` staging entry **1**
(`0xC2C/C30/C34`), missing the hang state, the hang bits, and any effect past
entry 1. Now it logs, per bullet per frame:

| col(s) | offset | meaning |
|---|---|---|
| 17 | `+0xBFC` | state — 1 live, 2/3/4 hang, 5 dying |
| 18 | `+0xBF6` | type-word flags (hang bits `0x2/0x4/0x8`) |
| 19 | `+0xBF4` | live active fx-flag word |
| 20 | `+0xBF0` | young countdown |
| 21 | `+0xC10` | staging-entry index processed |
| 22–51 | `+0xC14` | the 5 staging entries — `[p1, p2, interval, repeat, flag, gate]` each |

`bullet_trace.Bullet` exposes `.state` / `.tflag` / `.staging(frame)`;
`bullet_sim.fit_params` reads the hang and effects straight from them. Old
17-column recordings still load (fall back to inference).

Re-record: `python native/record_boss_driven.py letty --which 2 --godmode --dodge-after-boss --n 3`.

## Regenerating

```
tools/ghidra_scripts/th07_dump.java  — the headless dump script
C:\Users\spore\th07_re\              — decomp_all.c, functions.csv, bullet_pool_refs.txt
```

`analyzeHeadless <proj_dir> th07re -import th07.exe -scriptPath tools/ghidra_scripts -postScript th07_dump.java <out_dir> -deleteProject`

## Disassembly audit (2026-09-02) — the ECL opcode dispatch

`FUN_00410520`, `switch(opcode - 1)`, 159 cases. Findings folded into the VM:

- **`set_int_rand*` (op 6/7)** — `dst = rand_u32() % bound` (`+min`), *not*
  `int(rand_float()*bound)`. `set_float_rand*` (8/9) is `rand_float()*scale
  (+offset)` — matched already.
- **`set_*_rand_sign` (op 10/11)** — one 16-bit draw, `& 1`. (Letty unused.)
- **`__math_rand` (op 51)** — `(dst, lo, hi)` → `rand_float()*(hi-lo)+lo`, not
  `(dst, bound)`. (Letty unused.)
- **`__math_rand_rad` (op 52)** — **ignores lo/hi**. Case `0x33`: a ±45° cone
  toward screen centre (right if `SELF_X ≤ 192`, else left), then reflected off
  any wall within 96 px (x) / 48 px (y) of `move_bounds`. Implemented.
- **`move_bounds_set` (62) / `move_bounds_disable` (63)** — now tracked (were
  no-ops); `__math_rand_rad` needs them.
- **Not-actually-no-ops** (Letty doesn't use them, so still stubbed, but for
  Stage 2+): `__move_unknown` (47) sets `vel = (arg0, arg1)` directly;
  `move_at_player` (53) sets the move angle to `atan2(player) + arg0`;
  `__move_change_1/3/2` (59/60/61) switch the enemy into free / orbit /
  interpolate mode for `arg0` frames without re-specifying params.
- **`move_speed` (49) / `move_acceleration` (50) / `move_angular_velocity`
  (48)** — confirmed: set the field + force free-flight mode. **`set_angle`
  (58) takes two floats** (angle, ang-speed) and writes the orbit-angle pair;
  our handler takes one — a gap for a boss that uses it.
- **`move_dir_time` (54)** — `duration < 1` just sets the angle; `≥ 1`
  interpolates. Matches our impl.

**Still open after the audit:** Lingering Cold over-fire — the snow's
*spawn* count is now roughly right, but bullet_sim keeps each snowflake on
screen ~20–25 f too long: after the decel-reversal the recorded steady speed is
~0.67 px/f where the flag-0x10 model gives ~1.5–1.9. The post-reversal regime of
flag `0x10` needs another look.
