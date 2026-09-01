# The first ECL interpreter

**Verdict:** the VM worked; bullet *motion* did not. Shelved in favour of
[recording](recording.md). Being revisited now — see [the plan](ecl-vm.md) — with
a different way of getting the motion.

## The idea

Every PCB boss is a bytecode program in the "old" TH06–09 [ECL format](ecl.md).
`thtk` extracts and decompiles it cleanly. So: parse every boss script, run it in
a CPU interpreter, get a spawn schedule — `(frame, x, y, angle, speed, type)` per
bullet — and feed that to the GPU sim. Fully generative, any difficulty, any
boss, from the game's own script.

`sim/ecl_vm.py` (+ `ecl_parse.py`, `ecl_expand.py`, `ecl_bullet.py`,
`ecl_viz.py`) got built. The parser read the sub-table and instruction lists.
Control flow — waits, jumps, sub-calls, loops, the per-difficulty `rank_mask` —
worked. Phase chaining via `life_callback` worked. `enemy_create_rel` sub-enemy
recursion worked.

## Where it broke

The ECL says `bullet_fan(type=42, count=8, speed=2.5, …)`. It does **not** say
what type 42 *does after it spawns*:

- **Delay bullets** appear small and frozen for N frames, then "pop" and start
  moving.
- **Accelerating** bullets change speed over time.
- Bullets that **change direction** mid-flight, or **split** into more bullets.

That behaviour is a per-type launch script — the multi-slot `bullet_effects`
system — that lives in `th07.exe`'s data, not the ECL. Reversing it from a static
disassembly is undocumented and estimated at weeks. Without it, the VM's output
was visibly wrong:

> "way too fast, they don't pause, the wrong ones rotate"

## The lesson

> Don't reimplement an undocumented engine. The hang / curve / redirect state is
> already sitting in the [zBullet struct](re.md#zbullet-stride-0xd68-1025-slots)
> at `+0xC2C` — read it each frame and you capture the motion with zero
> interpretation.

That lesson produced the [recording pipeline](recording.md), which works but
[hits a ceiling](ceiling.md). The [new ECL VM plan](ecl-vm.md) keeps the VM for
*control flow* (Stage A) and gets the per-type motion the same way recording did
— by [hooking the engine and measuring](ecl-vm.md#stage-b-measuring-the-engine),
not by static RE.
