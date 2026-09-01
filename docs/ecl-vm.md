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

### 2 · VM core — control flow · ~2 days · autonomy 90%

The interpreter loop from `eclrunner.py`: frame counter, instruction pointer, 32
variable slots, a call stack. Each tick, run every instruction with
`time ≤ frame`, then advance. Implement `wait` / `stop`, `jump`, `jump_if`,
`sub_call`, `sub_ret`, loops.

- **Concerns:** PCB renumbered a few flow opcodes vs EoSD (cross-check against
  `th07.eclm` + priw8's ECL docs); sub-call re-entrancy edge cases.
- **Verify:** run Letty's top-level sub end to end — it terminates at the
  expected frame, and a trace shows it entering the Lingering Cold / NS2 /
  Table-Turning subs at frames that line up with the recorded screen-clears.

### 3 · Arithmetic, comparison & difficulty gate · ~1 day · autonomy 95%

The ~40 mechanical opcodes: `set_int/float`, `add sub mul div mod`, trig
helpers, comparison + conditional jumps, and the `rank_mask` check so only
Lunatic (`rank 3`) lines run. Plus the negative-ID globals: rank, difficulty,
difficulty coefficient, `PLAYER_X/Y` (placeholder).

- **Concern:** ECL is loose about int vs float variable typing; a wrong type
  shifts a spawn angle by a hair.
- **Verify:** unit test per opcode; then on Letty's script, `!L` lines execute
  and `!E`/`!N`/`!H` lines are skipped at rank = Lunatic.

### 4 · The PRNG · ~1 day, or 3–4 · autonomy 60%

`bullet_random*` and spread jitter call the engine RNG. EoSD's is a 16-bit LCG
(`seed*0x343FD + 0x269EC3`, take bits 16–30). Implement it, seed per episode,
check whether PCB uses the same one.

- **Concerns:** if PCB's PRNG differs, finding the real one needs static
  disassembly. Fallback: match the *distribution* (not the exact stream) of a
  recorded `bullet_random` spread empirically — fiddly but doable, just not
  bit-exact. Bit-exactness only matters for reproducing a specific real run;
  training only needs the right distribution, so this is lower-stakes than it
  looks.
- **Verify:** 500 VM `bullet_random` spreads; the angle histogram passes a KS
  test against the spreads measured from the recordings.

### 5 · Spawn-event emission · ~2 days · autonomy 85%

`bullet_fan[_aimed]`, `bullet_circle[_aimed]`, `bullet_random*`, `shoot_*` don't
simulate motion — they append `(frame, type, x, y, angle, speed, aimed_flag)` to
a list. Decode each opcode's parameter layout from `th07.eclm` + priw8.

- **Concerns:** PCB's bullet opcodes have more params than EoSD's (multi-speed,
  extra flags) — a mis-decode = wrong count or wrong fan width. The `aimed` flag
  has to be set for exactly the right opcodes (resolution happens per-episode on
  the GPU later).
- **Verify:** VM spawn-event count and timing vs the recordings' bullet-birth
  count and timing — within ~10%, same phase-boundary frames, overlapping spawn
  heatmap.

### 6 · Sub-enemies — `enemy_create_rel` · ~2 days · autonomy 75%

Every Letty attack spawns orbiting orb sub-enemies (opcode 93) that each run
their *own* ECL sub. The VM goes recursive: the parent runner (Letty) spawns
child runners (orbs), the children fire the bullets. Port from PyTouhou's
`ECLMainRunner`. Track each orb's position — they
[contact-kill](collision.md#enemy-bodies-kill-the-player) the player.

- **Concerns:** the most-likely-to-be-subtly-wrong part — orb spawn offsets,
  orbit motion, lifetime. Child-VM tick order relative to the parent matters for
  aimed sub-shots.
- **Verify:** orb count, spawn frames, per-frame positions from the VM vs the
  recorded `enemies` array — within ~15 px.

### 7 · HP, life-callbacks & spellcard phases · ~1 day · autonomy 82%

`enemy_life_set` (110) gives real HP. `life_callback_ex` /
`life_callback_threshold` register "when HP < N, jump to sub M".
`spellcard_start/end` (90/91) bracket the spells. **This replaces the guessed
`KILL_FRAC` with Letty's actual HP thresholds** — damage-phasing becomes exact.

- **Concern:** `enemy_flag_armored` / `enemy_flag_invulnerable` windows — easy to
  miss one when reading the decompile.
- **Verify:** VM phase-transition frames match the recorded screen-clears; HP
  values and timed-out-phase durations match the recordings.

### 8 · Boss movement · ~1 day · autonomy 85%

The `enemy_move*` opcodes script Letty's position over time. Produces her x/y
track — needed for the firing-lane alignment check and the obs.

- **Concern:** interpolation mode flags (linear / ease / accel).
- **Verify:** VM boss track vs the recorded `boss` x/y array (the one part of the
  recording we can check directly) — within ~10 px.

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
