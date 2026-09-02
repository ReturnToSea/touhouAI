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

## Disassembly audit 2 (2026-09-01) — the VM execution backbone

Read the remaining inferred handlers straight from `fn410520.c` + the
phase-machine / bullet-update functions in `decomp_all.c`. Everything below was
"inferred, matches recordings"; it is now engine-confirmed. Fixes folded in.

- **`call` (op 41, `caseD_28`)** — pushes the whole context (IP + all locals,
  0x86 dwords) onto a **16-deep stack** (`+0x8fc`, frame stride 0x218; ptr
  `+0x2a7c`), loads sub `arg0 & 0xFFFF`, resets time/IP to 0. `arg0` is the only
  operand. ✓ our VM (the 16-limit is academic for Letty).
- **`ret` (op 42, `caseD_29`)** — pops, restores context; the `+0x8f4` flag
  routes an interrupt-return through `+0x2ee8`→`+0x6fc`. ✓
- **`wait` (op 45)** — sets a frame timer at `+0x76c`; `LAB_0041069a` halts all
  instruction processing while `+0x76c > 0`. ✓
- **`jump` (op 2, `caseD_1`)** — `time = arg0; ip += arg1` **bytes**. Our parser
  resolves the byte delta to an instruction index (`off_to_idx`), so
  `e.ip = arg1` is correct. ✓
- **`jump_dec` (op 3, `case 2`)** — decrements the counter gvar
  **unconditionally** (`*p += -1`) and jumps while the result stays positive.
  **FIX:** our handler only wrote the counter back when it jumped, so a loop
  counter ended on 1 instead of 0. Now unconditional (`vm.py:_jump_dec`).
- **Unknown opcodes** — `caseD_7e` just advances to the next instruction. ✓
  validates stubbing the ~25 cosmetic opcodes (anm/sound/UI) as no-ops.
- **Phase machine** — `FUN_0041fd70` (life thresholds) + `FUN_0041ff80`
  (`boss_timer`): up to **4 HP thresholds** at `+0x2ebc[4]` with callback subs
  at `+0x2ecc[4]`; the timer sub is at `+12000`, the interrupt sub at `+0x2ee4`.
  On any transition the engine: **snaps HP up to the threshold value**, jumps to
  the callback, **zeroes the call stack** (`+0x2a7c = 0`), clears the timer +
  interrupt, and **kills / resets every sub-enemy** (0x1e0-slot loop). The timer
  path additionally clears bullets (`FUN_00424740(10)`). ✓ our `_fire` already
  killed children + `switch_to` zeroed the stack; **added** the HP-snap
  (`vm.py:_service_callbacks`, matters for Part 7).
- **`enemy_create_rel` / `_abs` (op 93 / 92)** → `FUN_0041f430`: **gated by
  `0 < enemy.life` (`+0x2bb8`)** — a boss at 0 HP (the death/transition window)
  spawns nothing. Free-slot scan over the **480-slot shared enemy pool** (stride
  0x4f48); child inherits the parent's 0x1a-dword arg block; **the child's
  frame-0 block runs immediately, this frame**. ✓ our `_spawn_child` matches;
  **added** the `life > 0` gate (only when `max_life > 0`, i.e. the boss).
- **`bullet_effects` (op 79, `case 0x4e`)** — **arg0 is a staging-slot index
  0..4**, written to `enemy+0x2bf4 + slot*0x18`. It is an **overwrite, not an
  append**, and there is no "flag 1 starts a fresh list". Arg order is
  `(slot, flag, gate, interval, repeat, p1, p2)`. **FIX:** our handler appended
  (capped at 5) and used `flag == 1` as a reset heuristic → an orb firing the
  same `bullet_effects(0, flag=0x40)` in a loop stacked 5 identical redirect
  entries, so those bullets stopped-and-redirected 5× and lingered on screen.
  Now a fixed 5-slot template written by index (`vm.py:_bullet_effects`).
  Lingering Cold `danmaku_check` ratio 1.23 → 1.17, curve corr 0.93 → 0.96.
- **Bullet cull** — `FUN_0042d6d8`: a bullet is off-screen once its box fully
  clears `[0,384]×[0,448]` (margin = ½ the type's sprite extent, ~8–32 px). A
  plain bullet is erased that **first** frame; a bullet carrying a bit in mask
  `0xDC0` (pause-redirect / pause-aim / bounce) gets a **128-frame** off-screen
  grace (`+0xbfe` counts up to 0x80). **FIX:** `danmaku_check` took the *last*
  on-screen frame (allowed re-entry); now `bullet_sim.cull_frame` applies the
  engine rule.
- **flag 0x10** (`FUN_004251a0`) — re-read and cross-checked against a recorded
  snow bullet: it is a clean `vel += p1·(cos,sin)(dir)` for `interval` frames
  with `angle = atan2(vel)` each frame, `dir` fixed at arm time (`p2`, or the
  spawn heading if `p2 ≤ -990`). The recording is a textbook `p1 = -0.025`
  parabola — **our model was already exact.** The earlier "snow lingers ~20 f
  too long" note was the effect-*stacking* bug above, not the motion model.

**Still open:** Lingering Cold still births ~+18 % bullets vs the recordings
(VM 3389 vs ~2861 for the phase) — an orb count / fire-cadence gap in
`enemy_create_rel` *timing*, which needs the Part 11 trace↔spawn alignment to
pin down (align is only 39 % tagged). Motion + cull + effects are confirmed
faithful, so the residual is purely spawn scheduling.

### Movement-reflection gvars (from this audit, applied 2026-09-01)

The float-read switch in `FUN_004182d0`-ish (gvar getter) maps, per enemy struct:

| gvar | id | offset | meaning |
|---|---|---|---|
| `CIRCLE_ANGLE` | 10045 | `+0x2b54` | the enemy's **heading** — `atan2` of this frame's displacement, rewritten every frame |
| `CIRCLE_SPEED` | 10046 | `+0x2b58` | the **`angular_velocity`** field (`move_angular_velocity` op 48 target) |
| `DIST_ORIGIN` | 10049 | `+0x2b6c` | the orbit **radius** (shared with `__move_circle_abs`) |
| `ORIGIN_X/Y/Z` | 10050-52 | `+0x2b8c…` | the orbit **centre** |
| — | 10053/54 | `+0x2b5c / +0x2b60` | the orbit's *own* sweep angle / ang-speed (the integrator at `FUN_...+6746` reads these) |

So `CIRCLE_ANGLE` / `CIRCLE_SPEED` are **not** the orbit's internal state —
`__move_circle_abs` keeps its own `angle` / `ang_speed` at `+0x2b5c / +0x2b60`
and never reads `10045` / `10046`. `Sub57`'s `CIRCLE_SPEED /= 1.4` per burst
therefore lands on the unused `angular_velocity` field and leaves the spiral
untouched — matching the recording (constant `0.0262` rad/f for 320 f). The VM
had wired `10045/46` straight to the live `_Motion`, which collapsed the orbit;
fixed. The orbit-expiry behaviour was also wrong (froze; the engine keeps the
velocity — the recorded orbs fly straight off the bottom). `danmaku_check`
1.17 → **1.08**, corr 0.96 → **0.98**.

### `enemy_create_rel` trailing args + off-screen enemy cull (applied 2026-09-01)

- **`enemy_create_rel(sub, x, y, z, hp, item, score)`** — `FUN_0041f430`
  params 3/4/5 → `+0x2bb8` (life, so `hp`), `+0x2e10` (item-drop type, checked
  `< 0`/`== -1` then `FUN_004326f0`), `+0x2bc0` (`/10` into the score counter).
  All cosmetic for the sim — they do **not** affect spawn count or timing.
- **`enemy_flag_oob_immune` (op 137, case 0x88)** sets bit 7 of `+0x2e2a`. The
  enemy update loop: once a sub-enemy has been on screen (`FUN_0042d6d8` box
  test, same as bullets) it is **despawned the frame it leaves** unless that bit
  is set (`&& (-1 < *(char*)(+0x2e2a))` guard before `FUN_004202d0`). Letty's
  orbs rely on this — they spiral off an off-centre boss and stop firing. VM now
  models it (margin ~56 px, estimated); orb lifetimes 2400 f → ~420 f. Doesn't
  move the bullet counts (Letty's orbs finish their shoot loop before drifting
  off) but it's correct and matters for Stages 2-6.

### HP / damage model — Part 7 (2026-09-01)

The shot-damage path is `FUN_00420620` (~13769–13851). A player shot reduces the
boss's `+0x2bb8` (life) **only** when, on `+0x2e29`:

- **bit 0** set (active), **bit 4** set (`enemy_flag_can_take_damage`, op 104,
  default set — orbs clear it on themselves), **bit 2** set (`enemy_flag_
  invulnerable`, op 103).

**`enemy_flag_invulnerable` is a misnomer** — bit 2 is the gate the damage code
*requires*, not one it blocks on. It means "engaged / accepts shot damage".
Letty's subs set it `(1)` when a phase's attack starts and `(0)` in Sub31
(intro) / Sub51 (defeat); the spell subs pair `(1)` with `enemy_flag_armored(N)`
— the timed declaration grace. `+0x4f40 > 0` (armored) zeroes the damage
(`/9` for a boss).

**`enemy_flag_death(mode)`** (op 106, case 0x69 → `+0x2e2a & 7`): what HP ≤ 0
does. The death handler (`FUN_00420620` ~13896: gated `life ≤ 0 && 0x2e29 bit0`)
clears the 4 life-thresholds + timer + interrupt, then `switch(mode)`:
`0/1` remove the enemy, `2` drop items + fire `death_callback` (`+0x2a84`, the
enemy persists), `3` `life = 1` + fire `death_callback`. `death_callback` also
zeroes the call stack. Letty: Sub38 sets mode 2 (LC capture → NS2), Sub39 sets
mode 3 (TT capture → defeat). `enemy_life_set` (op 110) writes `+0x2bb8` and
`+0x2bbc`; `boss_set` (op 99) sets/clears the is-boss bit 6, not the damage
gates.

### ReimuA shot table + per-power damage — Part 12 (2026-09-02)

The blocker ("descriptor table is behind character-select code, never shown
written") is solved: `PLAYER (0x4BDAD8) + 0xb7e70` is a **fixed global**
`0x00575948` holding the unfocused shot-table root ptr (`+0xb7e74` = `0x0057594C`
= focused root; `+0x240b` selects). It can just be read from a running process —
`native/probe_shot_damage.py --dumponly`.

- **`FUN_0043d160`** walks `root + 0x34` as `(u32 table, i32 power_threshold)`
  pairs, advancing while `threshold <= round(power)`. So the pair with the
  *sentinel* threshold `999` is the one used at max power (128).
- Each table is up to 96 **52-byte** (`0x1a` shorts) descriptor entries,
  terminated by a leading `i16 period < 0`. `FUN_0043bdc0`:
  `frame % period == phase` gates a shot; `FUN_0043bbd0` builds it. Layout:
  `i16 period@0, i16 phase@2, f32 xoff@4, f32 yoff@8, f32 dir@0xc/0x10,
  f32 speed@0x18, i16 DAMAGE@0x1c, u8 muzzle@0x1e, u8 type@0x1f,
  i16 sprite@0x20, i16 sfx@0x22`.
- **`FUN_0043d9e0`** (shot↔enemy AABB, `+0x318/+0x31c` shot box): sums
  `shot+0x348` (= descriptor `+0x1c`) over every overlapping shot into the boss's
  per-frame loss; **`/3` (min 1) when `player+0x16a20 != 0`** — appears to be the
  option-deploy startup window, not modelled. `FUN_00420620` then hard-clamps to
  70/frame.

**Unfocused, `power < 999` tier (max power):** 4 forward needles (dmg ~29, every
5f, muzzle 0, x-offset ±8) = **23.2 HP/f lined up** + 8 homing amulets (dmg 5,
every 15f, muzzle 1/2, type 1) = **2.7 HP/f from anywhere** → **~25.9 HP/f peak**.
Lower tiers are weaker and it is a real breakpoint: `power < 128` (i.e. power
96–127) is only 9 shots / **19.5 HP/f**; the 4th needle + the 7th/8th amulet
appear only at full power. Focused `power < 999`: 3 centre needles (~31 ea/5f) +
4 type-2 persuasion needles → 23.9 HP/f, all forward.

`sim/fight_replay.py`: `SHOT_DPS = 25.9`, `HOMING_FRAC = 0.10` (2.7/25.9),
`LANE_HALF = 24`. A 1CC is full-power by the Stage-1 boss, so max power is the
right model; `DPS_LO/HI = 0.55/1.0` randomisation covers real hit-rate < ceiling.

### move_bounds + enemy_set_hitbox (Part 12 prep, 2026-09-01)

- **`move_bounds_set` (op 62)** — the engine clamps scripted boss movement to
  the box (`FUN_004203b0`). Sub38 sets Letty's to `(32, 48, 352, 128)` and it
  persists into the later phases; the VM now clamps `_update_motion` output to
  it (no-op when the box is the full playfield). Letty's `move_position(-32,32)`
  → `move_point(→192,128)` entrance legitimately starts her off-screen.
- **`enemy_set_hitbox (x,y,z)` (op 101)** — body half-extents. A sub-enemy with
  `x > 0.5` is a *lethal* orb (Sub41 NS2, Sub57 TT — `8×8`); the LC shooter orbs
  are `0×0` (harmless). `danmaku_ecl` filters on this for FightSim's `en` array.

### The "Part 10 motion tail" was a method artefact (2026-09-01)

`fit_motion.py`'s median-displacement-profile fit reported ~75 % within 5 px /
90 f with a ~16 % tail. That tail is **averaging genuinely-random `bullet_random`
bullets into one profile** — not a physics gap. Running the engine-faithful
`bullet_sim.simulate` per bullet (the real Part 12 path) on all 28 k
instruction-pure matched traces: **p50 2.16 px, p90 7.09 px, p99 9.9 px, 98 %
within 8 px** — i.e. the recorder's own pos-vs-vel sampling noise floor.
`fit_motion.py` is superseded by `bullet_sim.py`; `align.refit_coverage` now
measures the per-bullet sim.
