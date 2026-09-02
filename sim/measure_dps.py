"""Measure ReimuA's effective shot DPS against Letty from a shooting recording.

The damage *system* is decompiled (`docs/th07-re-notes.md`): boss HP/frame =
sum of overlapping player-shot `+0x1c` damage values, clamped to 70/frame, no
stage-1 reduction, focused shots x1/3. The per-character descriptor table sits
behind the character-select code and the homing-amulet hit rate would still
need modelling, so the *effective* DPS is measured, not reversed.

    1.  record a fight WITH the player shooting:
          .venv/Scripts/python native/record_boss_driven.py letty_shoot --n 3 --shoot on
    2.  python -m sim.measure_dps
        -> prints the DPS to set as fight_replay.SHOT_DPS

Letty's real phase HP (Part 7): NS1 15000 -> life_callback 1700 (13300 to
phase), NS2 15000 -> 2000 (13000). The spells are timer-or-capture. A phase
that transitions *before* its timer was HP-driven: DPS ~= phase_HP / (frames
from first-attack to the transition).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

_FIGHTS = Path(__file__).resolve().parents[1] / "sim" / "fights"

# (name, HP budget, nonspell-timer frames) for the two damageable phases
_PHASES = [("NS1", 13300.0, 2400), ("NS2", 13000.0, 2400)]
# recorded screen-clear frames (f0-relative) — NS1 ends ~2400, LC ~5450, NS2 ~7820
_CLEARS = [2400, 5450, 7820]
_ARMOR = 60          # boss repositioning / declaration before its first bullet


def _phase_first_bullet(nb: np.ndarray, lo: int, hi: int) -> int:
    seg = nb[lo:hi]
    return lo + int(np.argmax(seg >= 15)) if (seg >= 15).any() else lo


def main() -> int:
    paths = sorted(glob.glob(str(_FIGHTS / "letty_shoot*.npz")))
    if not paths:
        print("no sim/fights/letty_shoot*.npz — record one first:")
        print("  .venv/Scripts/python native/record_boss_driven.py "
              "letty_shoot --n 3 --shoot on")
        return 1

    dps_ns1, dps_ns2 = [], []
    for p in paths:
        d = np.load(p)
        b = d["bullets"]
        f0 = int(b[:, 0].min())
        f = (b[:, 0] - f0).astype(int)
        nb = np.bincount(f, minlength=12000)

        # screen-clear = a >=200-frame stretch where the count collapses then
        # rebuilds; approximate with the known boundaries, snapped to the
        # nearest local minimum in a +-400 window
        clears = []
        for c in _CLEARS:
            w = nb[max(0, c - 400):c + 400]
            clears.append(max(0, c - 400) + int(np.argmin(w)) if len(w) else c)

        for i, (nm, hp, timer) in enumerate(_PHASES):
            p_lo = 0 if i == 0 else clears[1]          # NS1 from 0, NS2 after LC
            p_hi = clears[0] if i == 0 else clears[2]
            fa = _phase_first_bullet(nb, p_lo, p_hi) + _ARMOR
            dur = p_hi - fa
            if dur <= 0:
                continue
            hp_driven = dur < timer - 120                # ended before the timer
            dps = hp / dur
            tag = "HP-driven" if hp_driven else "hit the TIMER (lower bound)"
            print(f"  {Path(p).stem} {nm}: {dur} active frames -> "
                  f"DPS ~= {dps:.1f}   [{tag}]")
            (dps_ns1 if i == 0 else dps_ns2).append(dps if hp_driven else np.nan)

    a = np.array(dps_ns1 + dps_ns2, float)
    a = a[~np.isnan(a)]
    if len(a):
        print(f"\n  effective SHOT_DPS ~= {np.nanmean(a):.1f} "
              f"(n={len(a)} HP-driven phases)")
        print(f"  -> set fight_replay.SHOT_DPS to this; keep the 70/frame clamp")
    else:
        print("\n  every phase hit its timer — the drive policy isn't lined up "
              "under the boss enough. Need a shot-focused recording, or accept "
              "the timer as a lower bound on the real fight's phase durations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
