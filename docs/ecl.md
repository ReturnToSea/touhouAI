# The ECL format

Every boss is a bytecode program in the "old" TH06–09 ECL format. `thtk`
(`thdat` + `thecl`) extracts and decompiles `ecldataN.ecl` from `th07.dat` into
readable script. We read it for structure — phase timing, HP thresholds, which
shots are aimed — and [an interpreter that ran it](de-generative-danmaku.md) was
built and then shelved.

## What the script tells us

- **Phase structure & timing** — each attack, its duration, its `//N` absolute
  frame offset.
- **HP & transitions** — `enemy_life_set(15000)`,
  `life_callback_ex(0, 1700, 42)` (at 1700 life remaining, run sub 42),
  `life_callback_threshold(2000)`.
- **Invulnerability windows** — `enemy_flag_invulnerable(1)` and
  `enemy_flag_armored(240)` around spell intros.
- **Which shots are aimed** — `bullet_fan_aimed` / `bullet_circle_aimed` vs the
  plain variants.

## What Letty actually does

Letty uses **no laser opcodes** on any difficulty. Every attack spawns orbiting
satellite sub-enemies via `enemy_create_rel`, and *those* do the shooting. On
Lunatic:

| Phase | Sub | Fires | Orb hitbox |
|---|---|---|---|
| Non-spell 1 | `Sub38 → 41` | `bullet_random` | `8×8` — lethal |
| Cold Sign "Lingering Cold" | `Sub42 → 43` | `bullet_fan_aimed` | `0×0` — harmless |
| Non-spell 2 | `Sub39 → 40 → 41` | `bullet_random` | `8×8` — lethal |
| "Table-Turning" | `Sub55 → 57` | `bullet_circle`, spd 1.8 | `8×8` — lethal |

The fast `bullet_circle` stream in Table-Turning is what looks like a laser in
gameplay footage — it's ordinary bullets, and we record it fine.

!!! note "Two interpreters, both shelved"
    The [first attempt](de-ecl-vm.md) got control flow right but failed on bullet
    *motion* — the undocumented `bullet_effects` system. The
    [second](de-generative-danmaku.md) ran the scripts for real — control flow,
    phase transitions, arithmetic, sub-enemies, boss/orb movement, and bullet
    motion measured from the engine — reaching ~2 px per-bullet fidelity. A
    policy trained on it still transferred at 0 %, so the project
    [stopped simulating Letty](plan.md). The parser and VM
    (`sim/ecl/`) remain useful for reading structure and roughing out a boss
    that hasn't been recorded yet.
