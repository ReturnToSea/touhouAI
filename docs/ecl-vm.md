# The ECL VM build plan

Replace [recorded-replay](recording.md) danmaku with a VM that executes the
boss's actual PCB [ECL](ecl.md) script — infinite novel patterns, correct
physics, real HP thresholds. Twelve parts across three stages. Every part ships
only when its verification gate passes.

The downstream pipeline — collision, obs, PPO, the real-game daemon — does not
change. The VM replaces *where the bullet positions come from*, nothing else.

!!! note "What we already have"
    - All six PCB stage ECLs decompiled (`tools/th07_ecl/`), Letty isolated, the
      opcode map `th07.eclm`
    - **`sim/ecl/` — the binary parser (Part 1), verified against thtk on every
      instruction in the game**
    - A dead VM skeleton (`sim/ecl_vm.py` + `ecl_parse/expand/bullet`) from the
      [first attempt](de-ecl-vm.md), kept for reference only
    - PyTouhou's TH06 VM as reference (`pytouhou_ref/eclrunner.py`, ~90% opcode
      overlap)
    - The [DLL hook](hook.md) toolchain (MinHook / inject32)
    - The [zBullet struct](re.md#zbullet-stride-0xd68-1025-slots) RE'd
    - 20 recordings as validation ground truth
    - ~8 GB of freed VRAM

**Autonomy** below is the share of a part that can be built without a human at a
debugger — the rest needs eyes on the game or an external RE reference.

---

## Stage A · The virtual machine

*Pure software. The game is only needed to validate against the recordings.*

Parts 1–8 are Python: port PyTouhou's TH06 VM, adapt it to PCB, drive it with
Letty's decompiled script. The output at the end of Stage A is a full **spawn
schedule** — every bullet's spawn frame, type, position, angle, speed — plus the
boss track and sub-enemy spawns. No bullet motion yet; that is Stage B.

### 1 · ECL binary parser · ✅ done

`sim/ecl/` — parses `ecldataN.ecl` straight from the binary into subs,
timelines, and typed instructions, with no dependency on thtk's text output.

The PCB header does differ from EoSD's: `u16 sub_count`, `u16 timeline_count`,
a **fixed 0x40-byte** timeline-offset area (EoSD packs it), then the sub-offset
table at `0x44`. The instruction header is `i32 time · u16 opcode · u16 size ·
u8 (reserved) · u8 rank_mask · u16 param_mask`; `rank_mask` gates difficulty
(`E N H L` = bits 0–3). Two things PyTouhou's TH06 reader doesn't cover:
`bullet_*` / `laser_*` / `anm_set_poses` pack two `int16`s into their first
slot, and string parameters (spellcard names) are **XOR 0xAA** shift-jis.

- **Verify** (`python -m sim.ecl.verify`): decompile all six stage ECLs +
  extra + phantasm with `thecl -r`, parse the binaries, compare every
  instruction. **19,919 / 19,919 instructions match** — time, opcode, rank,
  every argument, jump-target resolution, and decrypted strings, across all 8
  files, 0 mismatches.

### 2 · VM core — control flow · ✅ done

`sim/ecl/vm.py` — runs a boss's ECL frame by frame. Instruction pointer,
per-sub time gate, call stack (args pass through the shared `ARG_*` / `PARAM_*`
gvars; `I0–I7` / `F0–F9` are snapshotted and restored on `ret`), difficulty
gate on `rank_mask`, and the piece PyTouhou's TH06 runner doesn't hand you: the
**phase machine**. Callbacks (`death_callback_sub`, `timer_callback_*`,
`life_callback_*` / `_ex`, `enemy_interrupt_set`) persist on the enemy across
sub switches; a spellcard timing out with no explicit timer sub falls through to
the death callback — which is how Letty's *Lingering Cold → NS2* transition
works even though Sub42 never names Sub39.

- **Verify** (`python -m sim.ecl.vm_verify`): synthetic subs cover
  jump / conditional jump / call+ret / `jump_dec` loops; then Letty's real ECL
  runs end to end. The phase machine walks **NS1 → Lingering Cold → NS2 →
  Table-Turning → defeat** (Sub 38 → 42 → 39 → 55 → 51), picks the Lunatic
  spellcard variants off the `DIFFICULTY` branch, and lands every transition at
  `[0, 2400, 5400, 7800, 10800]` — within ~1 s of the recorded screen-clears
  (`[2400, 5450, 7820]`; the gap is the repositioning lull the ECL timer
  doesn't count). Easy/Normal correctly route to Flower Wither Away (Sub48),
  Hard to Undulation Ray (Sub52).

### 3 · Arithmetic, comparison & difficulty gate · ✅ done

The mechanical opcodes in `sim/ecl/vm.py`: `math_{int,float}_{add,sub,mul,div,mod}`
(C-truncating int division), `math_inc/dec`, `sin` / `cos` / `atan2`,
`math_norm_angle` (wrap to `[-π, π)`), the six `set_*_rand_*` forms and
`__math_rand_rad`. Int-vs-float is resolved by the destination gvar's slot
(`I0–I7` → int, `F0–F9` → float) plus the parser's literal typing. The
difficulty gate (`rank_mask & (1 << difficulty)`) and the `DIFFICULTY` /
`PLAYER_X/Y` gvars were already in Part 2; `ARG_*` / `PARAM_*` / misc gvars live
in `enemy.extra`.

- **Verify** (`python -m sim.ecl.vm_verify`): a unit case per deterministic
  opcode (add/sub/mul/div/mod, sin/cos/atan2, inc, norm-angle), plus — on
  Letty's real script — the NS1 spawner (Sub40) computing its three sub-enemy
  angles as `0, +2π/3, −2π/3` via `math_float_add` + `math_norm_angle`, and
  every arithmetic opcode reachable in the fight being handled.

### 4 · The PRNG · generator done; validation blocked on Part 5

`sim/ecl/rng.py` — the EoSD-family LCG (`state·0x343FD + 0x269EC3`, bits 16–30),
seeded per VM, wired into every `set_*_rand_*` / `__math_rand*` opcode and
swappable. It is deterministic per seed and uniform (checked in `vm_verify`).

Whether PCB draws from *exactly* this stream — and the KS test of 500 VM
`bullet_random` spreads against the spreads measured from the recordings — needs
`bullet_random` (Part 5) and is deferred to then. For training only the
distribution matters, so this is low-stakes.

### 5 · Spawn-event emission · ✅ done (Lingering Cold count loose, see below)

`bullet_{fan,circle,random}[_aimed]` append `BulletSpawn(frame, kind, btype, x,
y, angle, speed, aimed, effect)` to `vm.bullets` — no motion (Stage B). Parameter
layout decoded from the difficulty-scaled variants in Letty's own script: arg 0
is the graphic group, arg 1 a sub-type, **arg 2 the count, arg 3 the layer
count**, then two speeds, base angle, angular step, and the flags/type word.
`bullet_effects` is recorded on each spawn for Stage B; `enemy_create_rel`
spawns a child runner that inherits the parent's `PARAM_*` and fires its own
bullets (a lighter version of Part 6); `enemy_kill_all` and every phase
transition clear the sub-enemies (the screen-clear).

Two engine details this pinned down: `wait(N)` **freezes** the per-sub frame for
N frames rather than advancing it (otherwise every instruction after a `wait`
gets skipped by the time gate), and `call` maps the caller's `ARG_*` onto the
callee's `PARAM_*` and restores locals on `ret`.

- **Verify** (`python -m sim.ecl.vm_verify`): VM spawn-event count per phase vs
  the mean bullet-birth count over the ten `sim/fights/letty_*` recordings —
  **NS1 +1 %, NS2 +1 %, Table-Turning +6 %, total +7 %**. Lingering Cold runs
  **+18 %** — the `Sub42 → Sub43 → Sub36` orbiting-orb chain over-fires; the
  interpretation is clearly right (the other three phases land inside 6 %) but
  the orb fire-rate needs the frame-level check that comes with Part 10's
  motion models. Also: Sub40 computes its three sub-enemy spawn angles
  `0, ±2π/3`.

### 6 · Sub-enemies — `enemy_create_rel` · ✅ done

The VM is recursive: `enemy_create_rel` (93) spawns a **child `Enemy`** running
its own sub, at the parent's position + offset, inheriting a snapshot of the
parent's `PARAM_*`/`ARG_*` (Sub40's icicles read the `PARAM_R` angle Sub40
just computed). Children run through the same tick loop as the boss — Parts
2–5 didn't have to change.

This part also had to add **motion** — `enemy_create_rel` alone put orbs at a
fixed offset with nowhere to go, and Letty's own repositioning moves needed it
too, so Part 8's opcodes came along with it:

- `move_position` (46) — snap.
- `move_dir_time` (54) / `move_point` (55) — linear interpolation over a given
  frame count, to a computed target or an absolute point.
- `__move_circle_abs` (56) / `set_orbit_distance` (57) — orbit a **fixed**
  center at a **fixed** radius, both evaluated once at the moment the command
  is issued (re-reading `SELF_X` every frame would be self-referential — the
  radius would collapse to zero).

- **Verify** (`python -m sim.ecl.vm_verify`):
    - An orbiting icicle's distance from its orbit center stays constant to
      floating-point precision over 90 frames — the circle math doesn't drift.
    - The boss's own track (`move_position` → `move_point` entrance, then
      `Sub38`'s `move_dir_time` calls), aligned to a recording on the
      first-bullet frame, is **pixel-exact for the first 127 frames** — then
      diverges the instant `Sub38` draws its first `__math_rand_rad` and moves
      on it. That's not drift or error: it's the first frame where the VM's
      RNG stream and the recording's real one disagree about which direction
      to go, and everything downstream of a movement choice is chaotic after
      that regardless of how correct the interpreter is. An exact match up to
      that exact instruction is about as strong a proof of correctness as this
      metric can give.

### 7 · HP, life-callbacks & spellcard phases · ~1 day · autonomy 82%

`enemy_life_set` (110) gives real HP. `life_callback_ex` /
`life_callback_threshold` register "when HP < N, jump to sub M".
`spellcard_start/end` (90/91) bracket the spells. **This replaces the guessed
`KILL_FRAC` with Letty's actual HP thresholds** — damage-phasing becomes exact.

- **Concern:** `enemy_flag_armored` / `enemy_flag_invulnerable` windows — easy to
  miss one when reading the decompile.
- **Verify:** VM phase-transition frames match the recorded screen-clears; HP
  values and timed-out-phase durations match the recordings.

### 8 · Boss movement · ✅ done (came along with Part 6)

Letty's own repositioning runs through the same `move_dir_time` / `move_point`
opcodes as the orbs, so this was one build. See Part 6's verify — the boss
track against the recording, pixel-exact for 127 frames.

---

## Stage B · Measuring the engine

*Needs the game and a DLL hook.*

The ECL says "fire type 42." It does **not** say what type 42 does after it
spawns — delay frames, acceleration, mid-flight speed changes, splitting. That
lives in `th07.exe`, and it is exactly what broke the
[last attempt](de-ecl-vm.md) ("bullets way too fast, delay bullets don't hang"). We get it by **hooking and measuring**, not
static reversing.

### 9 · Bullet-constructor hook · 2 days – 1 week · autonomy 40%

Find `th07.exe`'s function that takes `(type, x, y, angle, speed, …)` and
initialises a fresh zBullet slot. Hook it (MinHook). Log every call's args plus
each spawned bullet's per-frame state (`vel`, `speed`, `angle`, the
effects-state at `+0xC2C`) for its first ~60 frames.

!!! danger "The crux risk to working solo"
    Finding the constructor address needs dynamic analysis. Routes to try:
    (a) pymem memory-diffing — watch for a slot going empty→populated, scan
    nearby code; (b) published PCB reversing — thpatch / priw8 / thcrap have deep
    PCB RE, the address may just be documented; (c) byte-pattern scan for the
    function prologue. **If all three fail, it needs ~30 min of a human in
    x64dbg** — breakpoint on a bullet-slot write, read the return address — then
    the rest is routine. This is the first thing to de-risk once Stage A is
    moving, because [the fallback](#if-stage-b-stalls) needs it too.

- **Verify:** drive to Letty, dump the hook log — its spawn events match the
  VM's emitted events from Part 5 for the same fight. Per-type motion traces
  captured for every type she uses.

### 10 · Per-type motion models · ~3 days · autonomy 85%

For each of Letty's ~10–20 bullet types, fit a motion model from the Part 9
traces: `delay_frames`, `accel`, `speed_final`, `turn_rate`, `split_at/into`.
Express each as a vectorizable `pos(type, spawn, angle, speed, t)`.

- **Concerns:** the multi-slot `bullet_effects` system — a bullet can chain
  several effect stages. A type that doesn't fit a simple
  `delay → accel → cruise` model needs a small piecewise state machine.
  Splitting bullets multiply the object count — the GPU layer has to budget pool
  slots.
- **Verify:** simulate each type from its recorded spawn params; the path stays
  within ~5 px of the recorded path over the first 90 frames, for every type.

### 11 · Difficulty coefficients · ~2 days · autonomy 70%

Lunatic scales counts, speeds, gaps. Some is in the ECL (`!L` params), some is an
engine multiplier keyed off the difficulty global. Record Letty on **Normal**
and **Lunatic**, run the VM for both, diff VM-vs-real per difficulty, add
whatever scaling the VM misses.

- **Concerns:** needs a human to record Normal + Lunatic runs (the recorder needs
  a difficulty-select tweak first). Coefficients could be per-bullet-type or
  nonlinear rather than one scalar.
- **Verify:** VM-Lunatic bullet-density heatmap and bullet-count-over-time curve
  match recorded-Lunatic within ~10%; side-by-side in the viz, the patterns read
  as identical.

---

## Stage C · Integration

*The FightSim / GPU work.*

The re-aiming problem that
[wrecked the replay approach](de-reaim.md) disappears here — an
aimed bullet is *generated* from `(spawn, angle, speed)`, so we set the angle
toward this episode's policy and integrate forward. Nothing to desync from.

### 12 · GPU danmaku layer & train · ~4–5 days · autonomy 90%

`sim/danmaku_ecl.py`: at startup, run the CPU VM ~1–2 k times with different
seeds → a pool of spawn schedules + boss tracks (a few seconds; replaces
"recordings", but now we can make thousands). Each episode picks one. `_now()`
computes bullet positions from the Part 10 motion functions, vectorized. Aimed
bullets resolve toward this episode's policy; RNG spreads reroll per episode.
Collision, obs, damage-phasing — **reuse the existing FightSim code unchanged**.
Then `train_fight.py --sim ecl`.

- **Concerns:** vectorizing split / chained-effect bullets without a per-bullet
  Python loop; schedule-pool size vs diversity (too small → back to memorising
  2000 fights instead of 20); throughput (replay did ~270k steps/s, generated
  motion is more per-frame math — expect 120–200 k).
- **Verify — two gates:**
    1. Viz side-by-side: VM-Letty vs a recording — a human signs off that it
       looks right.
    2. Train a policy, run the real-game daemon: real-transfer median and
       kill-rate **beat** the replay baseline (~100 s median / ~15% real
       kill-rate). If it doesn't beat replay, something in Stages A–B is still
       wrong.

---

## If Stage B stalls

Part 9 is the one place this can get genuinely stuck. If it does and the
constructor address can't be found:

!!! note "Spawn-hook replay — the lighter fallback"
    Skip the VM. Hook `create_bullet` during real fights, record the *spawn
    events* (not final positions), replay them through the Part 10 motion models
    + per-episode re-aim + per-episode RNG perturbation. Bullet paths are
    generated, re-aiming works, RNG varies — but the spawn *schedule* still comes
    from recordings, so it isn't fully generative. ~1 week instead of 3. Enough
    for Letty; the full VM is worth it only because we're committing to all six
    stages. **This fallback still needs Part 9** — the constructor hook is
    load-bearing either way.

## Totals & risks

**~3 weeks** to a validated generative Letty. Beyond Letty, each new boss = point
the VM at that stage's ECL sub + hook/measure its ~10–20 new bullet types +
validate. Lasers (Stages 5–6) are a separate object type — deferred.

| Risk | Mitigation |
|---|---|
| PCB ECL format differs more than expected from TH06 | unlikely — it's the "old" TH06–09 format |
| Multi-slot `bullet_effects` is genuinely complex to model | hook-and-measure per type, not static RE |
| PCB's PRNG isn't TH06's | match the distribution, not the stream |
| Can't find the bullet constructor | ~30 min human in x64dbg |
