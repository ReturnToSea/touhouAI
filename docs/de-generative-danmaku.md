# Generative danmaku — the ECL VM

**Verdict:** the most faithful reproduction of Letty the project has ever built,
and the *worst* for transfer. Eight training runs on VM-generated danmaku; real
survival was a flat line the whole time — **~50 s active-fight median, 0 % real
kills**, against ~103 s / 33 % from the [replay baseline](de-letty-replay.md) it
was meant to beat and ~225 s from the [procedural sim](sim.md). The VM's fidelity
metrics kept improving while transfer never moved. A Python reimplementation of a
2003 game engine can be made *close* — it cannot be made *exact*, and close
doesn't transfer.

This page is the postmortem. [The plan as it stood](#the-plan-verbatim) — twelve
verifiable parts, most of them done — is preserved in full below.

## What got built

[The first interpreter](de-ecl-vm.md) died on bullet *motion*. This attempt was
the answer to that: keep the VM for control flow (**Stage A**), and get the
per-type motion the way [recording](recording.md) did — by measuring the engine,
not statically reversing it (**Stage B**).

By the end, `sim/ecl/` did all of this, each behind a passing verification gate:

- **Parsed** every stage's ECL binary — 19,919 / 19,919 instructions match `thtk`.
- **Ran** Letty's real script frame by frame — control flow, the phase machine,
  arithmetic, the real PRNG (`FUN_00431870`, a 16-bit `rotl16` generator, not
  the EoSD LCG), sub-enemy recursion.
- **Moved** the boss and her satellite orbs — the boss track landed
  **pixel-exact against a recording for 127 frames**, until the first RNG-driven
  turn.
- **Emitted** every bullet spawn (frame, type, position, angle, speed) within a
  few percent of the recorded birth counts per phase.
- **Propagated** each bullet with an engine-faithful per-frame model
  (`bullet_sim.py`) — the hang, the launch kick, all five `bullet_effects`
  flags — reproducing recorded trajectories to **p50 2.2 px / p90 7 px**, the
  recorder's own noise floor.
- **Wired** it end to end: a GPU danmaku layer, live re-aiming for aimed shots
  (`aim_pool.py`), real HP thresholds, `train_fight.py --sim ecl`.

`danmaku_check` — VM danmaku vs the recordings, on-screen bullet count per frame
— ended at **ratio 1.03, curve correlation 0.98**.

## What happened when a policy trained on it

Eight runs (`fight_letty_ecl` run 1 – run 8, up to ~800 M steps each). The
transfer daemon played real Letty against every checkpoint.

| | Sim eval | Real game |
|---|---|---|
| survival | ~108 s median, reaching phase **3.58 / 4** (deep Table-Turning) | **~50 s** active-fight median — a flat line across every checkpoint |
| kill-rate | 29–62 % | **0 %** |
| where it dies | — | **Lingering Cold, 62 % of episodes**; NS1 the rest |

The single best moment across all eight runs was run 2 at ~466 M steps: one real
fight reached Table-Turning (127 s). It was never reproducible and the median
never followed.

This is worse than every prior approach. The procedural sim gets a policy to
~225 s real (clears Stage 1, dies in Stage 2). The 20-recording replay pipeline
gets ~103 s and lands real kills a third of the time at its best checkpoint. The
faithful VM — the thing all of that was building toward — got **0 %**.

## Why it didn't transfer

### The tell is the same as every other dead end

`fight_letty_ecl` improved its **sim** score run over run, and its fidelity
metrics improved too — `danmaku_check` went 1.23 → 1.17 → 1.08 → 1.03 over the
fix campaign, birth counts converged, the density curve correlation climbed. Real
transfer never moved off the floor. When the sim eval — and every offline
fidelity check — can't tell a policy that transfers from one that doesn't, the
policy is optimising against structure that only exists in the sim. Here that
structure is **the reimplementation's residual error**.

### It compounds

Individually, the VM's bullets are good — `bullet_sim` matches recorded
trajectories to ~2 px, the recorder's noise floor. But a Lingering Cold screen
has **~500 bullets**, and the errors are not independent noise — they are
systematic, from the same handful of slightly-wrong constants. A ~40 px-wide safe
lane between two bullet streams is the difference between "dodge left" and "die",
and a 3-degree error in a fan's base angle plus a 4 % error in its per-layer
speed moves that lane somewhere the real game doesn't have one. The policy learns
the VM's lanes. They aren't Letty's.

### Two errors we could not close

1. **x87 80-bit floating point.** `th07.exe` (2003) does its bullet and movement
   math in the x87 FPU, with 80-bit intermediates and x87 rounding. `numpy` /
   `torch` are 32- or 64-bit IEEE. The decompiler shows `float10` all through
   `FUN_00423730` (the emitter) and the movement ops. Bit-exact reproduction of
   x87 arithmetic in Python is not practical without an x87 emulator in the hot
   loop. Every bullet is off by a fraction of a pixel *at spawn*, before any
   motion, and it grows.

2. **Runtime-only constants.** Letty's Lingering Cold sweep is driven by
   `move_dir_time [60, 4, __math_rand_rad, 0.7]`, and `__math_rand_rad` pulls the
   boss toward `DAT_004be408`. That address is **zero-initialised BSS** — it
   holds nothing in the executable on disk; the engine writes it during stage
   setup. Static analysis can't give it. The VM's LC boss x-swing came out
   **±42 px** against the real **±80 px** — the boss doesn't range far enough, so
   the whole spell's geometry is narrower than reality. And `DAT_004be408` is one
   of *dozens* of such globals.

### The method hid it

Every offline check was an **aggregate**: total bullet count, per-phase birth
count, a density-map correlation, a KS test on heading distributions. All of them
improved toward the recordings. None of them can see whether the *specific dodge
lane the policy is standing in* exists in the real fight. `danmaku_check` at 1.03
means the VM puts the right *number* of bullets on screen — not in the right
*places*.

## Everything we tried to make it work

The VM side (each a real engine detail, each verified against recordings):

- Rewrote the PRNG from the EoSD-family LCG to the real `rotl16` generator
  (`FUN_00431870`), re-wired every `set_*_rand_*` opcode onto it, added a KS test.
- Fixed the per-layer emitter speed formula: `spd1 − (spd1−spd2)·layer/layers`,
  **÷layers not ÷(layers−1)** — the VM was running every layered fan ~15–25 %
  slow (NS1 bullet speed ×1.31 → ×1.04 of recorded).
- Implemented all **9 emitter modes** from `FUN_00423730` (fan / circle /
  offset-ring / random-angle / random-speed, with layer rotation and count-parity
  fan centring).
- Rebuilt the `bullet_effects` model twice — from "arg0 = enable bit" to
  "arg0 = staging-slot index 0..4, an overwrite" — after reading the VM backbone
  in `fn410520.c`. This was silently stacking 5 redirect entries on looped orbs.
- Added the **spellcard damage divisor** (`/7` while a spell is active,
  `FUN_00420620`), the real HP thresholds (NS1 1700, NS2 2000, spells
  timer-or-capture), the 70-HP/frame damage clamp.
- Added the off-screen sub-enemy cull (`enemy_flag_oob_immune` + box test) —
  orb lifetimes 2400 f → 420 f, engine-correct.
- Fixed the orbit gvars: `CIRCLE_ANGLE` / `CIRCLE_SPEED` are the heading /
  angular-velocity fields, not the `__move_circle_abs` sweep state — `Sub57`'s
  `CIRCLE_SPEED /= 1.4` per burst had been collapsing the Table-Turning orbit.
- Implemented the hang (spawn-state crawl), the launch kick
  (`|vel| = speed + 5·(1−t/16)` for 17 f), the FX pause-redirect / re-aim /
  wall-bounce handlers, and `move_point` / `move_dir_time` easing modes — all
  from a Ghidra decompile session (`docs/th07-re-notes.md`).

Live-game probes, for what static analysis couldn't give:

- **Player hitbox: found it was 2.0 px half-extent in the sim vs 0.825 real** —
  a 2.4× error that had been in the project since the start. Move speed and
  playfield bounds probed and confirmed exact.
- Probed for `DAT_004be408` and the other movement constants — **they're in BSS,
  only readable from a running game**, and even with them the x87 math remains.

Training side:

- **Streaming fresh VM schedules** — a background worker generating new danmaku
  layouts continuously, so the policy couldn't memorise a fixed pool of 48.
- Live re-aiming (`aim_pool.py`) for the ~6 % aimed shots, sized up after it was
  dropping ~12 % of aimed bullets and undersized for Lingering Cold.
- **Per-bullet hitbox fed into the observation** — the danger grid and escape
  scan now use each bullet's real AABB half-extent, C and Python at 0.0 parity.
- Handoff spawn pose (start episodes at the real Stage-1→Letty handoff position),
  a bottom-camp penalty, reward normalisation, LR annealing.

The fidelity metrics responded to all of it. Transfer did not.

## The decision

Stop reimplementing the engine. Train the Letty fight on **the real game**, the
way [Stage 1 fine-tuning](de-realgame.md) already runs: `ST_ROLLOUT` collects a
whole PPO trajectory in C inside each [hooked game](hook.md), 12 instances in
parallel at ~68× real-time each (~5–6 k steps/s aggregate), warm-started from
`ppo_v29`. Every episode plays real Stage 1 into the real Letty fight; the
boss-damage reward term does the rest. There is no fidelity gap because there is
no sim — it *is* Letty's bytecode, running in Letty's engine, with Letty's x87
FPU and Letty's RNG.

[Real-game fine-tuning](de-realgame.md) was previously filed as "a wash on a
solved stage." That verdict holds for *Stage 1 survival*, which the sim policy
already maxes. It does **not** hold for *killing Letty fast*, which the sim policy
can't do and which no simulator has been able to teach.

## What it left us

- The **VM is not deleted.** `sim/ecl/` parses and runs every PCB stage's
  bytecode; it stays the fastest way to get a rough danmaku pattern for
  pre-training, and the parser / phase machine / RE notes are reference for every
  future stage.
- The **RE work is banked** — the emitter modes, the bullet motion model, the
  PRNG, the spellcard damage rules, the HP thresholds are all in
  `docs/th07-re-notes.md` and correct regardless of whether the VM drives
  training.
- The **player-hitbox fix** (2.0 → 0.825) came out of this and applies
  everywhere.
- The lesson, in one line, joins the others in [the table](dead-ends.md):
  **a reimplemented engine is still a fixed artefact** — closer to real than 20
  recordings, but the policy finds the seam between it and reality just the same,
  and here the seam was wide enough to swallow the whole transfer.

---

## The plan, verbatim

*What follows is the former Chapter 13, "Generative danmaku" (`ecl-vm.md`),
exactly as it stood when the approach was shelved. Most of the twelve parts were
done and verified; the two verification gates in Part 12 — a human viz sign-off
and a transfer test that beats the replay baseline — are the ones it never
passed.*

---

# The ECL VM build plan

Replace [recorded-replay](recording.md) danmaku with a VM that executes the
boss's actual PCB [ECL](ecl.md) script — infinite novel patterns, correct
physics, real HP thresholds. Twelve parts across three stages. Every part ships
only when its verification gate passes.

The downstream pipeline — collision, obs, PPO, the real-game daemon — does not
change. The VM replaces *where the bullet positions come from*, nothing else.

!!! note "Where Stage A stands"
    **Parts 1, 2, 3, 5, 6 are done and verified.** `sim/ecl/` parses every
    stage's binary byte-exact, runs Letty's real script frame by frame (control
    flow, the phase machine, arithmetic, a working PRNG), recurses into her
    sub-enemies, and moves both her and them — the boss's own track lands
    **pixel-exact against a recording for 127 frames**, until the first
    RNG-driven choice. Bullet **spawn events** come out the other end (count,
    type, position, angle, speed) within a few percent of the recorded birth
    counts.

    Partial: **Part 4** (the PRNG generator is built; the KS test against
    recorded spreads isn't run yet) and small boss-spell-loop timing residuals
    (Table-Turning fires ~6 % hot; not cleanly validatable against the truncated
    recordings). **Part 7 done** — Letty's real HP thresholds (NS1 15000 → 1700,
    NS2 15000 → 2000, the two spells timer-or-capture) drive the phase graph
    under damage and stay timer-driven with none.

    **Stage B essentially done.** Per-bullet traces reconstructed from the pool
    recordings with no hook (Part 9, bit-exact). The engine-faithful
    `bullet_sim` (Part 10) reproduces recorded trajectories to **p50 2.16 px /
    p90 7.09 px / 98 % within 8 px** — the recorder's noise floor. Part 8's
    sub-enemy movement is **done for Letty** — the orbit op (`__move_circle_abs`)
    plus the `CIRCLE_ANGLE`/`CIRCLE_SPEED` gvar fix and off-screen enemy cull,
    all verified against the recorded orb tracks. The trace-to-spawn matcher
    (Part 11) tags **75 %** of bullets, 100 % instruction-pure. `danmaku_check`
    (VM danmaku vs recordings): **ratio 1.08, curve correlation 0.98**.
    - All six PCB stage ECLs decompiled (`tools/th07_ecl/`), Letty isolated, the
      opcode map `th07.eclm`
    - A dead VM skeleton (`sim/ecl_vm.py` + `ecl_parse/expand/bullet`) from the
      [first attempt](de-ecl-vm.md), kept for reference only
    - PyTouhou's TH06 VM as reference (`pytouhou_ref/eclrunner.py`, ~90% opcode
      overlap)
    - The [zBullet struct](re.md#zbullet-stride-0xd68-1025-slots) fully RE'd,
      including the `bullet_effects` fields — enough that Stage B needs no hook,
      just the existing per-frame recorder
    - 20 recordings as validation ground truth — already exercised, not just
      sitting there: Parts 5/6/8 verify directly against
      `sim/fights/letty_*.npz`

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

### 4 · The PRNG · ✅ real algorithm (RE) + KS test

The [th07.exe RE session](th07-re-notes.md) settled which generator the danmaku
uses. It is **not** the EoSD-family LCG that `sim/ecl/rng.py` originally had —
that constant (`0x343FD`) appears in `th07.exe` only inside the unused MSVC
`rand()`. The real one (`FUN_00431870`) is a 16-bit generator:

```
next16():  u = (state ^ 0x9630) + 0x9AAD;  state = rotl16(u, 2);  return state
rand_u32(): (next16() << 16) | (next16() & 0xFFFF)
rand_float(): rand_u32() / 2**32
```

`sim/ecl/rng.py` now implements this, wired into every `set_*_rand_*` /
`__math_rand*` opcode as before. Deterministic per seed and uniform
(`vm_verify`).

**KS test** (`python -m sim.ecl.ks_test`): VM bullet-*heading* distributions
match the recordings per phase — **KS D 0.014–0.13, all < 0.15** (NS2/TT nearly
exact). The PRNG and its per-opcode consumption are faithful. Bullet *speeds*,
though, run **~×1.2 of recorded in NS1 / NS2 / TT** (LC matches) — the recorded
speeds run past the ECL's literal `!L` `spd1`, so there's a **Lunatic engine
speed multiplier the VM misses**. Isolating it from the `!L` params needs a
Normal-difficulty recording (Part 11's difficulty-coefficient work); it's not a
PRNG bug.

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

Three engine details this pinned down: `wait(N)` **freezes** the per-sub frame
for N frames rather than advancing it (otherwise every instruction after a
`wait` gets skipped by the time gate); `call` maps the caller's `ARG_*` onto the
callee's `PARAM_*` and restores locals on `ret`; and `bullet_effects`'s first
argument selects behaviour.

> **Superseded by disassembly audit 2 (see `th07-re-notes.md`).** `call` / `ret`
> / `wait` / `jump` / `jump_dec` and the phase machine were all read from
> `fn410520.c` + `FUN_0041fd70` / `FUN_0041ff80` and confirmed. Two fixes fell
> out: `jump_dec` decrements its counter **unconditionally**; and
> **`bullet_effects` arg0 is a staging-slot index 0..4, an overwrite** — not an
> enable bit and not an append. The enable-bit reading happened to zero the
> right NS2 bullets (their slot-0 write carries `flag 0`), but an orb looping
> `bullet_effects(0, flag=0x40)` was stacking 5 redirect entries. The slot-write
> fix cut the Lingering Cold `danmaku_check` ratio 1.23 → 1.17 (corr → 0.96).
> The engine also gates `enemy_create_rel` on `boss.life > 0` and snaps HP up to
> the crossed threshold on a phase transition — both now in the VM.

- **Verify** (`python -m sim.ecl.vm_verify`): VM spawn-event count per phase vs
  the mean bullet-birth count over the ten `sim/fights/letty_*` recordings —
  **NS1 +1 %, NS2 +1 %, Table-Turning +6 %, total +7 %**. Lingering Cold runs
  **+18 %** on spawn count — the `Sub42 → Sub43 → Sub36` orbiting-orb chain.
  After disassembly audit 2 (effect-slot fix + engine-faithful cull) and the
  Part 8 orbit-gvar fix, `danmaku_check` is **ratio 1.08, curve corr 0.98, peak
  VM 530 / rec 498**; the birth-count `+18 %` is `Sub47`'s short-lived aimed
  bursts, so *on-screen* density is within ~8 % everywhere. Also: Sub40 computes its
  three sub-enemy spawn angles `0, ±2π/3`.

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

### 7 · HP, life-callbacks & spellcard phases · ✅ done

Letty's real HP structure, from the ECL:

| phase | sub | HP | leaves when |
|---|---|---|---|
| NS1 | 38 | `enemy_life_set(15000)` | HP < **1700** (`life_callback_ex`) **or** t = 2400 → Sub42 |
| Lingering Cold | 42 | inherits (~1700) | HP ≤ 0 (capture) **or** t = 3000 → Sub39 (`death_callback` / spell-timeout) |
| NS2 | 39 | `enemy_life_set(15000)` | HP < **2000** (`life_callback_threshold`, `!L → 55`) **or** t = 2400 → Sub55 |
| Table-Turning | 55 | inherits (~2000) | HP ≤ 0 (capture) **or** t = 3000 → Sub51 |
| defeat | 51 | `enemy_life_set(0)` | — |

- `enemy_life_set` (110), `life_callback_threshold` (112) / `_sub` (113) /
  `_ex` (148), `spellcard_start/end` (90/91) — all wired. On a life-threshold
  cross the engine (`FUN_0041fd70`) snaps HP up to the threshold; on HP ≤ 0 the
  **`enemy_flag_death`** mode decides (`&7`): **2** = drop + fire `death_callback`
  (LC → NS2), **3** = revive to 1 HP + fire `death_callback` (TT → defeat).
- **`enemy_flag_invulnerable` is misnamed.** Bit 2 of `+0x2e29` is the gate the
  shot-damage path *requires* (`FUN_00420620` ~13816) — it means "engaged,
  accepts shot damage". Bosses set it (1) when a phase's attack starts, (0) in
  the intro / defeat. `enemy_flag_armored(N)` is the timed grace layered on top
  (the spellcard-declaration window). `enemy_flag_can_take_damage` (bit 4,
  default set) is the hard gate the orbs clear on themselves.
- **Verify** (`vm_verify._test_hp`): a **dodge-only run stays timer-driven**
  (transitions at 2400 / 5400 / 7800 / 10800 — the recorded screen-clears);
  under damage, **NS1 → LC fires exactly when HP crosses 1700** (frame ==
  13300 / dps, checked at dps 10 & 30), NS2 → TT at 2000, and a heavy-damage run
  **captures both spells** (no timeouts) and ends at Sub51. The `engaged` gate
  is checked directly.

### 8 · Movement · ✅ done for Letty

Letty's repositioning shares `move_dir_time` / `move_point` / `move_position`
with the orbs, and Part 6's verify shows the boss track pixel-exact for 127
frames. The rest of Part 8 was the sub-enemy movement — **every Letty bullet is
fired by a satellite orb**, so the orbs have to be placed right or Parts 10/11
can't close.

- **Free-flight physics.** `move_speed` (49), `move_acceleration` (50),
  `move_angular_velocity` (48), `set_angle` (58) were no-ops; they now drive the
  engine's per-frame integration (`speed += accel; angle += ang_vel;
  pos += speed·[cos,sin]`), with `move_dir_time` (54) reworked onto it. Letty
  barely uses 48/49/50/58 — Stages 2–6 will.
- **`enemy_trace.py`** reconstructs each sub-enemy's track from the recorded
  `enemies` array (slot + position-jump identity, ~100% of rows). Ground truth:
  Letty's **shooter orbs (hb 0×0) sit near-stationary ~48 px off the boss**; the
  **lethal orbs (hb 8×8) spiral outward**.
- **`__move_circle_abs` (56) + `set_orbit_distance` (57) implemented.** The
  semantics came from TH08's documented circle-movement op (same engine
  generation) — no disassembly needed:
  `__move_circle_abs(frames, cx, cy, cz, angle0, ang_speed, radius0,
  radius_growth)` places the enemy at `centre + radius·[cos,sin](angle)` each
  frame, then advances `angle += ang_speed` and `radius += radius_growth`;
  `frames` frames then freeze (`0` = until the sub ends). `set_orbit_distance`
  retargets the live orbit — Letty freezes hers with `(DIST_ORIGIN, 0)` at
  t = 120. The orbit state is exposed back through the `CIRCLE_*` / `DIST_ORIGIN`
  / `ORIGIN_*` gvars (read *and written* mid-orbit).
- **`move_point` / `move_dir_time` arg 1 is an easing mode** (from the
  [th07.exe RE](th07-re-notes.md) — the engine shapes the interpolation
  progress: 0 linear, 1–3 ease-in `x²/³/⁴`, 4–6 ease-out). The VM ignored it;
  now applied. `move_dir_time` reworked back onto the interpolator (travel
  `speed·duration` px along `angle`, eased). Boss track still pixel-exact for
  127 frames.
- **Verify** (`vm_verify`): a synthetic orbit spirals at 0.5 px/f and sweeps its
  angle; **the VM's Sub57 orbs' spiral rate matches the recorded Table-Turning
  orbs**; the RE decompile confirms the orbit integration is exactly this.
  `align` match rate **21 % → 37 %**.

**Still owed:** a dedicated boss-track verify pass (RNG-stream-matched); the
`__move_unknown` (47) / `__move_change_*` (59–61) opcodes (Letty doesn't use
them).

---

## Stage B · Measuring the engine

*Needs the game — but not, it turns out, a hook.*

The ECL says "fire type 42." It does **not** say what type 42 does after it
spawns — delay frames, acceleration, mid-flight speed changes, splitting. That
lives in `th07.exe`, and it is exactly what broke the
[last attempt](de-ecl-vm.md) ("bullets way too fast, delay bullets don't hang").
We get it by **measuring**, not static reversing.

### 9 · Per-bullet trajectory tracking · ✅ done (no hook needed)

The original plan was to find and hook `th07.exe`'s bullet constructor
(40% autonomy, possible x64dbg session). Turns out it isn't necessary:

- The [recorder](recording.md) already polls the whole bullet pool every frame,
  and the [`zBullet` struct](re.md#zbullet-stride-0xd68-1025-slots) — including
  the `bullet_effects` fields at `+0xC2C` — is already RE'd.
- A bullet's identity is its pool slot: born when the slot goes empty→occupied,
  dead when it goes occupied→empty. Measured across ~2 M frame-to-frame
  transitions in one recording: the worst position residual (actual vs
  `pos + vel`) is **6.2 px**, and there are **zero** same-slot swaps on
  consecutive frames — slots sit empty for ~13 frames between reuses. So slot
  presence alone tracks bullets cleanly; no distance heuristic, no hook.

`sim/ecl/bullet_trace.py` re-keys the recorded pool stream into per-bullet
trajectories `(frame, x, y, vx, vy)` + birth class / fx flag.

- **Verify** (`python -m sim.ecl.bullet_trace`): on all ten `letty_*`
  recordings — every one of ~2 M rows re-keyed with **none lost or
  duplicated**, the rebuilt `(frame, x, y)` stream is **bit-identical** to the
  recorded one, frames within a trace are strictly consecutive, and **no trace
  contains a jump** that would betray a mis-tracked slot reuse. `fx=16` bullets
  measurably decelerate (median speed 1.8 → 1.2 px/f over 60 frames); `fx=0`
  bullets hold speed. ~15,200 bullets per fight.

!!! success "Recorder now logs the full motion state"
    `record_boss_driven.py`'s per-bullet row grew from 10 columns to 17:
    added `speed`, `accel`, `angvel`, `angle` (`+0xBB0…0xBBC`) and the
    `bullet_effects` params `p1/p2/interval` (`+0xC2C…0xC34`). Re-recorded three
    Letty fights and checked the new fields:

    - `angle` matches `atan2(vy, vx)` **exactly** (median 0, max 0 over 2 M rows)
    - `fx_p2` is **exactly −999** ("keep speed") and `fx_interval` is **exactly
      `{60, 120}`** — the literal values from `bullet_effects(…, 60/120, …, −999)`
      in Letty's script; `fx_p1` holds `{+0.00833, −0.025, +0.01667}`, each
      traceable to a specific `bullet_effects` call
    - `speed` matches `hypot(vx, vy)` for steady bullets and leads/lags it
      during acceleration — it's the engine's scalar-speed variable; Part 10
      uses `hypot(vx, vy)` as the authoritative per-frame speed
    - `accel` / `angvel` (`+0xBB4/8`) are zero for every Letty bullet — her
      speed changes go through the `fx` mechanism, not those fields. Kept in the
      schema (cost nothing) but unverified until a boss that uses them

    Letty has **no delay bullets** — every bullet moves the frame it spawns.

### 10 · Bullet motion model · ✅ engine-faithful sim, reproduces to the recorder's noise floor

The [th07.exe RE session](th07-re-notes.md) plus a recorder extension (log the
`state` track + all 5 `bullet_effects` staging entries) turned this from a
guessed empirical fit into the engine's actual per-frame update.
`sim/ecl/bullet_sim.py` — the reference Part 12's GPU layer vectorises:

- **The hang** — a bullet whose type-word has bit `0x2`/`0x4`/`0x8` spawns
  **4 velocity-steps behind** its point in state 2/3/4, crawls at
  **0.5 / 0.4 / 0.333×** while its materialise animation plays, then runs the
  live step too (a one-frame crawl+full spike) and goes full. Read straight from
  the recorded `state` column.
- **The launch kick** — `bullet_effects` flag 1 (`FUN_004250d0`): for 17 frames,
  `|vel| = speed + 5.0·(1 − t/16)`. Fixed constants; armed when the type-word
  has bit `0x1` **and** a flag-1 entry is staged. This is the "hover then
  launch" and the Table-Turning spike.
- **All 5 fx flags** — `0x10` directional accel (`−999` = "along heading",
  auto 180° flip on reversal — Lingering Cold), `0x20` turn+accel, `0x40`
  pause-redirect (Letty's "spam phase" — it was this all along, not a mid-flight
  op), `0x80` pause-re-aim-at-player, `0xc00` **wall bounce** (384 × 448) —
  Table-Turning. Each is **gated `(type-word & entry.flag)`**, so btype `0x200`
  ignores the same staged flag-`0x10` entry that `0x211`/`0x215` act on.

**Verify** (`python -m sim.ecl.bullet_sim`): **26/27 Letty bullet groups
reproduce within 8 px / 90 f — overall p50 2.2 px, p90 6.2 px.** The ~6 px
residual is the recorder's own frame-phase noise, not the model. The empirical
`fit_motion.py` (~75 %) stays as a fallback / sanity check.

<details><summary>earlier `fit_motion.py` notes</summary>

`sim/ecl/fit_motion.py` groups the Part 9 traces by an observable "type" —
`(class, fx_flag, fx_p1, fx_interval, base_speed, turns?, ramps?)` — and fits
each group its **median displacement profile**: `mag[t]` and `dheading[t]`
tables in the bullet's own frame. Forward-sim is
`pos0 + cumsum(mag · [cos,sin](heading0 + dheading))` — non-parametric, a plain
GPU lookup, and exact for any behaviour a group actually shares.

What the recordings settled:

- **Position is `cumsum` of `diff(xy)`, not of the pool's `vel` field.** The
  `speed`/`vel` fields (`+0xBB0 / +0xB98`) are read at a different point in the
  engine's frame than position; integrated, they drift **13 px p90 / 22 px p99
  over 90 frames** against the recorded positions. That is the recorder's own
  noise floor — so model the displacement the bullet actually made.
- **Motion shapes:** most bullets hold heading and cruise at constant speed
  after a ~15-frame catch-up transient; ~9–13 % take one sharp scripted heading
  change (a `bullet_effects` redirect), or a 180° flip when a decel ramp
  (`fx_p1 = −0.025` for 120 f) drives speed through zero — the "Lingering Cold"
  snow that drifts out and falls back. `accel` / `angvel` (`+0xBB4/8`) are zero
  for every Letty bullet.

**Result: ~75 % of bullets track within 5 px / 90 f** (p50 **0.00 px**, p75
4.8 px). The two largest populations — ~21 k of ~45 k bullets — fit to **0.00 px
p90**. The remaining ~25 % is a hard tail, and its cause is structural:

!!! note "Resolved — the tail was a method artefact, not a real gap"
    The median-profile fit averages genuinely-random `bullet_random` bullets
    (Sub34/Sub35 — random angle *and* speed per bullet) into one profile, which
    it can't represent. Once `bullet_sim.simulate` runs per bullet from its own
    params (Part 11 links each trace to its spawn), the fit is **p50 2.16 px,
    p90 7.09 px, 98 % within 8 px** — the recorder's noise floor. This module
    (`fit_motion.py`) is superseded by `bullet_sim.py`.

- **Verify** (`python -m sim.ecl.fit_motion`): per-group `p50 / p90` path error
  over 90 frames vs the recorded path, plus the aggregate coverage number. Bar
  is 5 px / 90 f per group; currently met for the groups that map to a single
  attack, flagged `TAIL` for the mixed ones.
- **Fallback if a per-instruction group still won't fit:** a small piecewise
  state machine for that one type, not a project stall.

</details>

### 11 · Trace-to-spawn alignment + difficulty coefficients · 75 % tagged, motion re-fit 84 %

`sim/ecl/align.py` matches each Part 9 bullet trace to the VM spawn that
produced it and tags it with `(source_sub, source_ip)`. Two matchers:

- **1:1** — per phase, cross-correlate the two birth-rate histograms for the
  frame offset, then within each observable `(fx_flag, fx_p1, fx_interval)`
  signature greedily match recorded births to VM spawns on
  `(|Δframe − offset|, Δposition, Δheading)`, fitting a rigid spawn-point shift.
- **ring** — burst patterns (`bullet_circle` firing 5–30 bullets in one frame)
  defeat per-bullet matching, but the *sequence of rings* lines up ~1:1
  (Table-Turning: 257 VM bursts vs 260 recorded, same frames). DP-align the two
  ring lists on `(|Δframe|, ring-size)`, then match traces to spawns inside a
  pair by nearest launch heading.

**Where it stands** (`python -m sim.ecl.align`): **75 % of bullets tagged**
(was 34 %), **100 % instruction-pure** on every big group. The 1:1 matches sit
on their emitter (dpos p50 ~9 px). The ring matches (`Sub57` Table-Turning,
`Sub41` NS2) are frame- and heading-locked (dframe std 4–6 f on `Sub57`, was 25)
but the VM puts their emitter **~100–190 px off** — RNG-driven boss drift (Part
6), *not* a matching failure. Running `bullet_sim.simulate` per bullet on the
~28 k matched traces reproduces them to **p50 2.16 px, p90 7.09 px, 98 % within
8 px** — the recorder's noise floor.

**Part 8 orbit fix (`Sub57` / `Sub41` / LC orbs):** `CIRCLE_ANGLE` (10045) /
`CIRCLE_SPEED` (10046) were wired to the live orbit's sweep angle / speed — but
the [th07.exe RE](th07-re-notes.md) shows they are the *heading* and
*angular-velocity* fields (`+0x2b54 / +0x2b58`), which `__move_circle_abs` never
reads. So `Sub57`'s `CIRCLE_SPEED /= 1.4` per burst was collapsing the orbit's
angular velocity to zero in ~30 steps — the orbs crept out radially instead of
spiralling. Fixed: `10045` → heading (`atan2` of this frame's displacement),
`10046` → the `angular_velocity` field, `10049` (`DIST_ORIGIN`) still → orbit
radius. Also: an expired orbit now hands its **velocity** to free flight instead
of freezing (the recorded orbs fly straight off the bottom). VM orb tracks now
follow the recordings to ~5 px through the whole spiral. **`danmaku_check` ratio
1.17 → 1.08, curve correlation 0.96 → 0.98.**

**Off-screen sub-enemy cull** (`enemy_flag_oob_immune`, op 137): once an orb has
been on screen it's despawned the frame it leaves the play area unless
oob-immune (`FUN_0042d6d8` box test + the `-1 < *(char*)(+0x2e2a)` guard). VM
now models it — orb lifetimes drop from ~2400 f to ~420 f. It doesn't move
Letty's bullet counts (her orbs finish their shoot loop before drifting off) but
it's engine-correct and matters for Stages 2-6. `enemy_create_rel`'s trailing
args decoded too — `(sub, x, y, z, hp, item_drop, score)`, all cosmetic, *not*
count-affecting.

**The "16 % motion tail" was a method artefact.** `fit_motion.py`'s
median-displacement-profile averaged genuinely-random `bullet_random` bullets
into one profile. The engine-faithful `bullet_sim.simulate` per bullet — the
actual Part 12 path — reproduces all ~28 k matched traces to **p50 2.16 px, p90
7.09 px, 98 % within 8 px**, the recorder's noise floor. `fit_motion.py` is
superseded by `bullet_sim.py`.

**Remaining (low priority):**

- **`Sub57` / `Sub41` emitter absolute position** — the ring matcher shows
  ~120 px, but not a bug: those orbs spawn on the boss, whose Table-Turning
  drift is `__math_rand_rad`-driven, so VM and recording diverge as Part 6
  established. Fine for training (random seeds).
- **Table-Turning +~6 % steady-state** — the spell loop fires a touch hot;
  traced to `call(56)`/`call(2)` frame accounting and where the 3000-frame
  timer lands. Small, and the recordings are truncated at frame ~10750 so it
  can't be validated cleanly. On-screen density is fine (`danmaku_check` 1.08).
- **Lingering Cold `Sub36`** — 57 % pure (RNG branch — expected). Birth count
  +18 %, all `Sub47`'s short-lived aimed bursts; on-screen density within ~8 %.

**Difficulty coefficients** — the KS test (Part 4) chased down what looked like a
Lunatic speed multiplier and it turned out to be a **VM bug, not an engine
scale**: `FUN_00423730` sets a layered attack's per-layer speed as
`spd1 − (spd1 − spd2)·layer/layers` (÷`layers`, not ÷`layers−1` — the last layer
never reaches `spd2`). The VM had ÷`layers−1`, so every layered fan/circle ran
~15–25 % slow. Fixed → NS1 bullet speed ×1.31 → ×1.04 of recorded; the NS2/TT
residual is a bimodal-median artefact (per bullet-type the VM matches to
<0.01 px/f — `simulate_batch` gives ring-1 1.80 / ring-2 0.73 at age 30, ==
recorded). So on Lunatic, at least, there is **no separate engine speed
multiplier** — the `!L` ECL params + the (now-correct) layer formula account for
it. A Normal recording would still be the way to confirm nothing else scales
(counts, gaps).

---

## Stage C · Integration

*The FightSim / GPU work.*

The re-aiming problem that
[wrecked the replay approach](de-reaim.md) disappears here — an
aimed bullet is *generated* from `(spawn, angle, speed)`, so we set the angle
toward this episode's policy and integrate forward. Nothing to desync from.

**The CPU pipeline is wired and validated.** `sim/ecl/bullet_sim.from_spawn()`
turns a VM `BulletSpawn` into `BulletParams`; `sim/ecl/danmaku_check.py` runs
the VM → propagates every spawn → compares the on-screen bullet count per frame
against the recordings: **total ratio 1.08, density-curve correlation 0.98,
peak on-screen VM 530 / rec 498** (after the audit-2 fixes + the Part 8
orbit-gvar fix). Every phase window is within ~10 %.

### 12 · GPU danmaku layer & train · pipeline built, training not yet run

- **`sim/ecl/bullet_sim.simulate_batch`** — the scalar `simulate` vectorised
  over `N` bullets (numpy, masked branches for hang / launch / all 5 fx flags).
  Verified **bit-for-bit vs `simulate`** (max abs diff 6e-12 over 8 k bullets;
  `bullet_sim` PASS gate).
- **`sim/danmaku_ecl.py`** — `build_schedules(n)`: run the VM once per seed
  (~2.4 s), propagate every spawn with `simulate_batch`, greedy-assign pool
  slots, screen-clear each bullet at the next phase transition, and emit the
  **same dense `[F, POOL, 2]` dict `_load_dense` produces** (+ `phase_hp` from
  Part 7); aimed bullets go to a separate param table (see below). Verify
  (`python -m sim.danmaku_ecl`): schedules build in ~2.4 s, bullets-on-screen
  **ratio ~1.0** (baked 0.92 + the ~6 % aimed in the runtime pool), two seeds
  differ by **~24 px mean** (fresh RNG), a `FightSim` built on them steps cleanly
  (obs `[B, 236]`).
- **`FightSim(recs=...)`** — injection param; when a rec carries `phase_hp` the
  phase table uses the **real thresholds** (NS1 13300, LC 1700, NS2 13000, TT
  2000) instead of the synthetic `SHOT_DPS·dur·KILL_FRAC`.
- **Per-type hitboxes** — `danmaku_ecl._type_hitboxes()` reads the type-word →
  AABB-half map from the recordings (`{4/5→3.0, 512/516/517/529→2.0,
  530/533/576→3.0}` — pellets 2 px, balls 3 px); covers every Letty bullet.
- **Live re-aiming** (`sim/aim_pool.py`) — aimed shots (`bullet_*_aimed`, ~6 %
  of Letty's danmaku, mostly Lingering Cold's Sub47) can't be baked: they point
  at the player *when they fire*. `danmaku_ecl` emits them as a param table
  carrying the **un-aimed** angle (`angle − atan2(VM-player − emitter)`); at
  runtime `AimPool` spawns each toward the live policy and integrates it with
  the same per-frame update as `bullet_sim.simulate` (torch, vectorised over
  `B × 256` slots, `source_index % 256` slot assignment). Re-aiming is safe here
  — the bullet is *generated*, so there's no recorded track to desync from
  (that's what broke the replay approach). Verified: the aimed ring turns +40°
  between player x=70 and x=314, matching the geometry.
- **`train_fight.py --sim ecl`** — builds `--ecl-schedules` (default 48) fresh
  training schedules + 8 held-out-seed eval schedules; `--fight` forced to
  `letty`. Collision / obs / damage-phasing reuse the existing code unchanged.
  Throughput **~189 k steps/s at B = 4096** (in the expected range). Smoke-tested
  end to end; **training not yet run.**

- **Verify — two gates (pending the training run):**
    1. Viz side-by-side: VM-Letty vs a recording — a human signs off.
    2. Train a policy, run the real-game daemon: real-transfer median +
       kill-rate **beat** the replay baseline (~103 s / ~33 %).

---

## If Stage B stalls

The crux risk was Part 9 — finding and hooking the bullet constructor. That's
**gone**: per-frame polling + slot-identity tracking gets the same data
(`sim/ecl/bullet_trace.py`), verified bit-exact against the recordings, no
dynamic analysis. The next risk was Part 8's sub-enemy movement — also **closed
for Letty**: the orbit op came from TH08's documented equivalent and verifies
against the recorded orb tracks, no disassembly. What's left is Part 11 matcher
refinement (a bounded engineering task with the recordings as ground truth) and
the Part 10 re-fit, which is already at ~75 % / 70 % on the correctly-grouped
subset, with a small piecewise state machine as the fallback for any awkward
type.

## Totals & risks

Beyond Letty, each new boss = point the VM at that stage's ECL sub + trace its
new bullet types + validate. Lasers (Stages 5–6) are a separate object type —
deferred.

| Risk | Status / mitigation |
|---|---|
| PCB ECL format differs more than expected from TH06 | **resolved** — parser matches thtk on 19,919 / 19,919 instructions |
| Multi-slot `bullet_effects` is genuinely complex to model | measure per type from the recordings (Part 10), not static RE |
| PCB's PRNG isn't TH06's | match the distribution, not the stream — Part 4 KS-test still to run |
| ~~Can't find the bullet constructor~~ | **moot** — no hook needed (Part 9) |
