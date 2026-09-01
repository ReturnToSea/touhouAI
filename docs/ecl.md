# The ECL format

Every boss is a bytecode program in the "old" TH06–09 ECL format. `thtk`
(`thdat` + `thecl`) extracts and decompiles `ecldataN.ecl` from `th07.dat` into
readable script. Today we read it for structure; [the plan](ecl-vm.md) is to
run it.

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

!!! note "Reference then, runtime now"
    The [first interpreter attempt](de-ecl-vm.md) got the control flow right but
    failed on bullet *motion* — the undocumented `bullet_effects` system. The
    [current VM](ecl-vm.md) runs these scripts for real: control flow, phase
    transitions, arithmetic, sub-enemies, and boss/orb movement all execute
    Letty's actual bytecode frame by frame (Parts 1–3, 5, 6, 8 of the plan).
    What it still can't do is bullet *motion* — it gets that by
    [hooking the engine and measuring it](ecl-vm.md#stage-b-measuring-the-engine)
    — the same trick that made [recording](recording.md) work — rather than by
    interpreting `bullet_effects` statically.
