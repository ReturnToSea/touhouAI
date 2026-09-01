# th07 ECL — decompile + the VM's input data

Goal: run each PCB boss's **actual bytecode** so RL trains on real, novel-every-
episode Touhou patterns instead of replayed recordings (which
[hit a ceiling](../../docs/de-letty-replay.md)). This directory holds the
decompiled scripts and the opcode map; the VM that runs them is in
[`sim/ecl/`](../../sim/ecl/). Full plan + per-part status:
[`docs/ecl-vm.md`](../../docs/ecl-vm.md).

## Contents

| file | what |
|---|---|
| `th07.eclm` | opcode / gvar / timeline-opcode names (from zero318 / ExpHP-truth) — **tracked** |
| `annotate.py` | post-processes a `thecl` text dump, naming `ins_NNN` / `[100NN]` from the eclm |
| `ecldataN.ecl` | raw ECL extracted from `th07.dat` — **gitignored** (derived from copyrighted game data) |
| `ecldataN*.tecl`, `_rN.tecl` | thtk decompiles — **gitignored**; `_rN.tecl` are the raw (`-r`) dumps `sim.ecl.verify` compares against |

Extract + decompile:

```
tools/thtk/thtk-bin-12/thdat.exe -x 7 "<game>/th07.dat" ecldataN.ecl   # 1..8 = stages 1-6 + extra + phantasm
tools/thtk/thtk-bin-12/thecl.exe -d 7 -r ecldata1.ecl > _r1.tecl
```

## The format (reverse-engineered — see `sim/ecl/parser.py`)

`th07.exe` uses the "old" TH06–09 ECL. PCB differs from EoSD (PyTouhou's
reference) in a few ways:

- header: `u16 sub_count`, `u16 timeline_count`, a **fixed 0x40-byte**
  timeline-offset area, then `u32 sub_offset[]` at `0x44`
- instruction header: `i32 time · u16 opcode · u16 size · u8 0 · u8 rank_mask ·
  u16 param_mask` — `rank_mask` bits `E N H L` = 0–3; `param_mask` bit *i* =
  logical param *i* is a gvar reference
- `bullet_*` / `laser_*` / `anm_set_poses` pack two `int16` into their first
  4-byte slot; spellcard-name strings are **XOR 0xAA** shift-jis
- `call` maps the caller's `ARG_A..N` (10037+) onto the callee's `PARAM_A..N`
  (10029+); `wait(N)` freezes the per-sub frame counter for N frames

`python -m sim.ecl.verify` round-trips all eight files against `thecl -r`:
**19,919 / 19,919 instructions decode identically**.

## Letty (Stage 1 boss), Lunatic — as the VM executes it

```
timeline: enemy_create(31) → boss_interrupt(0)
Sub31  boss_set, fly in, enemy_interrupt_set(38,0)
Sub38  NS1        — timer 2400 → Sub42;  fires via call(32/33/34)
Sub42  Lingering Cold  (spell 0/5)  — timer 3000, no timer-sub → death_callback → Sub39
Sub39  NS2        — timer 2400 → !L Sub55  (!EN → Sub48 Flower Wither Away, !H → Sub52 Undulation Ray)
Sub55  Table-Turning  (spell 0/9)  — timer 3000 → death_callback → Sub51
Sub51  defeat
```

Sub-enemies: `Sub40→Sub41` (NS1 icicles, `bullet_random`), `Sub43→Sub36`
(Lingering Cold orbs, `bullet_fan`), `Sub56→Sub57` (Table-Turning, `bullet_circle`).

## Status

`sim/ecl/` runs all of the above frame-by-frame — Parts 1–3, 5, 6 of the plan
are done and verified against the `sim/fights/letty_*` recordings (phase timing,
spawn counts within a few %, boss track pixel-exact for 127 frames). Part 8's
movement system is built (that pixel-exact track is it) but hasn't had its own
verify pass; Part 4's PRNG KS-test and Part 7 (HP thresholds — mechanism wired,
needs a damage model) are also outstanding. Then **Stage B** — bullet *motion*,
from hooking `th07.exe` and measuring, not from interpreting `bullet_effects`
statically.

## Refs

- opcode params: priw8 ECL docs (<https://priw8.github.io/#b=ecl-tutorial/>), PyTouhou `pytouhou/formats/ecl.py`
- `pytouhou_ref/eclrunner.py` — the TH06 VM this one is adapted from (gitignored, GPL)
