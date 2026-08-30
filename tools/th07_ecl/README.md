# th07 ECL extraction → sim import

Goal: pull the **real** boss/stage bullet patterns out of th07 and run them in
the GPU danmaku sim, so RL trains on actual Touhou content at ~40k steps/s
instead of made-up danmaku (which tops out ~Stage 1-3 on transfer).

## Status: extraction works, interpreter not built yet

- `thtk` (`tools/thtk/`) `thdat -x 7 th07.dat ecldataN.ecl` → `thecl -d 7` →
  readable script. `annotate.py` names the opcodes/gvars from `th07.eclm`
  (from ExpHP/truth; thtk-bin-12's own eclmap parser rejects the modern format).
- All 6 stage ECLs decompiled → `ecldataN_named.tecl` (gitignored - derived from
  copyrighted game data).
- Stage 1 boss map (`ecldata1_named.tecl`):
  - Cirno midboss: `Sub20` (10000 HP), spell `Sub29` (icicle "First Column")
  - Letty boss: `Sub38`/`Sub39` (15000 HP), non-spell `Sub39`
    - `Sub42` "Lingering Cold" (spell ids 2-5, per difficulty)
    - `Sub48` "Flower Wither Away" (ids 6-7)
    - `Sub52` "Undulation Ray" (id 8)   ← has lasers
    - `Sub55` "Table-Turning" (id 9)

Patterns are fully readable, e.g. Letty's opening (`Sub39`→`Sub40`→`Sub41`):
spawn 3 orbiting "icicle" enemies 120° apart, each runs a 60-iter loop firing
`bullet_random(count, type6, n, 1, speed, speed_var, +pi, -pi, sprite530)` every
other frame, difficulty-scaled (`!E/!N/!H/!L` prefixed lines).

## Remaining work (the ~1.5-2 week build)

1. **bullet_* opcode signatures** - opcodes 64-89 (`bullet_fan[_aimed]`,
   `bullet_circle[_aimed]`, `bullet_offset_circle*`, `bullet_random*`,
   `shoot_*`, `laser_*`). Get exact params from priw8's ECL docs
   (https://priw8.github.io/#b=ecl-tutorial/) + PyTouhou's th06 interpreter
   (same "old" ECL format).
2. **CPU ECL interpreter** (`ecl_interp.py`): VM (jump/call/wait/set/math/rand/
   cmp), `enemy_create_rel` spawns a child VM, `move_*` → position track,
   `bullet_*` → spawn-schedule entries `(frame, x, y, angle, speed, type,
   aimed_flag, ...)`. Run once per boss on CPU → schedule + boss-move track +
   HP phases. RNG: roll our own per-episode (natural variation).
3. **Sim replay** (`sim/danmaku_ecl.py` or a mode in danmaku.py): at each frame
   spawn the scheduled bullets across B episodes, re-aim `aimed` ones toward
   each episode's player. Reuse existing bullet physics.
4. **Validate on Letty**: train a policy on sim-Letty, compare real-Letty
   transfer vs the made-up-danmaku sim. Better → scale to Stages 2-6. Same/worse
   → sim physics is the bottleneck, fix that first.
5. Lasers (Sub52 / Stage 5-6): separate object + physics. Defer past the Letty PoC.

## Refs
- opcode names: `th07.eclm` (ExpHP/truth)
- VM semantics: PyTouhou `pytouhou/formats/ecl.py` + `pytouhou/game/`
- priw8 ECL docs, LiveECL (guy-l.github.io/LiveECL)
- exphp-share/th-re-data (struct layouts)
