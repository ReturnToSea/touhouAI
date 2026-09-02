"""Measure ReimuA's effective shot DPS against Letty from a shooting recording.

`record_boss_driven.py` now logs the boss HP (`+0x2BB8`) and the player power,
so this reads the HP-drain rate directly and splits it by whether the player was
lined up under the boss (`|px - boss_x| < LANE_HALF`, the forward-needle lane) or
not (homing amulets only). It also reports drain-rate vs player power.

Feeds `fight_replay`: SHOT_DPS = the lined-up rate, HOMING_FRAC = homing / lined.

    1.  .venv/Scripts/python native/record_boss_driven.py letty_shoot --which 2 \
            --shoot on --godmode --n 3
    2.  python -m sim.measure_dps
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

_FIGHTS = Path(__file__).resolve().parents[1] / "sim" / "fights"
LANE_HALF = 26.0          # forward-needle x-lane half-width (fight_replay)
DMG_CLAMP = 70.0


def _drain(hp: np.ndarray) -> np.ndarray:
    """Per-frame HP loss, ignoring the callback HP *snaps* (jumps UP or huge
    single-frame drops) and non-damage frames."""
    dh = -np.diff(hp.astype(np.float64))
    dh[dh < 0] = 0.0                          # HP snapped up (phase callback)
    dh[dh > DMG_CLAMP + 5] = 0.0              # not a shot (phase reset / capture)
    return dh


def main() -> int:
    paths = sorted(glob.glob(str(_FIGHTS / "letty_shoot*.npz")))
    if not paths:
        print("no sim/fights/letty_shoot*.npz — record first:")
        print("  .venv/Scripts/python native/record_boss_driven.py letty_shoot "
              "--which 2 --shoot on --godmode --n 3")
        return 1

    lined_all, homing_all = [], []
    by_power: dict[int, list] = {}
    for p in paths:
        d = np.load(p)
        boss = d["boss"]
        if boss.shape[1] < 4:
            print(f"  {Path(p).stem}: no HP column — re-record with the updated "
                  f"record_boss_driven.py"); continue
        bstep = boss[:, 0].astype(int)
        f0 = int(d["bullets"][:, 0].min())
        bf = bstep - f0
        m = bf >= 0
        bf, bx, hp = bf[m], boss[m, 1], boss[m, 3]

        pl = d["player"]
        pf = pl[:, 0].astype(int) - f0
        # align player x + power to the boss frames
        px = np.interp(bf, pf, pl[:, 1])
        pw = np.interp(bf, pf, pl[:, 3]) if pl.shape[1] > 3 else np.full(len(bf), np.nan)

        dh = _drain(hp)                                   # [n-1]
        aligned = np.abs(px[:-1] - bx[:-1]) < LANE_HALF
        firing = dh > 0                                   # a shot connected

        lined = dh[firing & aligned]
        homing = dh[firing & ~aligned]
        lined_all.append(lined)
        homing_all.append(homing)
        hp_start = hp[hp > 100]
        print(f"  {Path(p).stem}: HP {hp_start.max():.0f} -> ~{hp[hp>0].min():.0f}, "
              f"{int(firing.sum())} damage frames "
              f"({100 * aligned[firing].mean():.0f}% lined up)  "
              f"lined DPS {np.mean(lined) if len(lined) else 0:.1f}  "
              f"homing DPS {np.mean(homing) if len(homing) else 0:.1f}")

        for w in np.unique(np.round(pw[:-1] / 16) * 16):
            if np.isnan(w):
                continue
            sel = firing & (np.abs(pw[:-1] - w) < 8)
            if sel.sum() > 30:
                by_power.setdefault(int(w), []).extend(dh[sel].tolist())

    L = np.concatenate(lined_all) if lined_all else np.zeros(1)
    H = np.concatenate(homing_all) if homing_all else np.zeros(1)
    print(f"\n  lined-up DPS  {np.mean(L):.1f}  (n={len(L)})")
    print(f"  homing   DPS  {np.mean(H):.1f}  (n={len(H)})")
    if np.mean(L) > 0:
        print(f"\n  -> fight_replay:  SHOT_DPS = {np.mean(L):.0f}   "
              f"HOMING_FRAC = {np.mean(H) / np.mean(L):.2f}")
    if by_power:
        print("\n  drain rate vs player power:")
        for w in sorted(by_power):
            v = np.array(by_power[w])
            print(f"    power ~{w:3d}: {np.mean(v):.1f} HP/frame  (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
