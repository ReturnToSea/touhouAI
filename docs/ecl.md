# 4 · The ECL

Every boss is a bytecode program in the "old" TH06–09 ECL format. `thtk`
(`thdat` + `thecl`) extracts and decompiles `ecldataN.ecl` from `th07.dat` into
readable script. We do not run it — we read it.

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

!!! warning "Reference, not runtime"
    An early plan was to run the ECL in a CPU interpreter and feed its output to
    the sim. It did not work ([ch. 11](dead-ends.md)). The scripts survive as a
    structure map: phase order, HP numbers, which patterns track the player.
