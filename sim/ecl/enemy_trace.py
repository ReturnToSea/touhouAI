"""Reconstruct per-sub-enemy tracks from a recorded fight's `enemies` array.

Letty's danmaku is all fired by orbiting satellite sub-enemies, so Part 8's
movement model has to place *those* right, not just the boss. This re-keys the
recorder's per-frame enemy-pool scan (`enemies`: step, slot, x, y, life,
hb_x, hb_y, hb_z) into individual tracks so a VM run can be checked against them.

Identity is the pool slot, but enemy slots are reused the same frame the old
occupant dies, so a slot alone isn't enough — a track also ends on a position
jump larger than `jump_px` (an orb never moves that fast).

    from sim.ecl.enemy_trace import load_enemy_traces
    tr = load_enemy_traces("sim/fights/letty_0.npz")   # frame-0-relative
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEP, SLOT, X, Y, LIFE, HBX, HBY, HBZ = range(8)


@dataclass
class EnemyTrack:
    slot: int
    birth_frame: int
    hb: tuple[float, float]
    frames: np.ndarray      # int32 [n]
    xy: np.ndarray          # float32 [n, 2]

    @property
    def life(self) -> int:
        return len(self.frames)

    @property
    def step(self) -> np.ndarray:
        """Per-frame displacement magnitude."""
        return np.hypot(*np.diff(self.xy, axis=0).T) if self.life > 1 else np.zeros(0)


def load_enemy_traces(npz_path: str | Path, *, jump_px: float = 18.0,
                      min_life: int = 4) -> list[EnemyTrack]:
    d = np.load(npz_path)
    e = d["enemies"]
    if len(e) == 0:
        return []
    b = d["bullets"]
    f0 = int(b[:, 0].min()) if len(b) else int(e[:, 0].min())

    e = e[np.lexsort((e[:, STEP], e[:, SLOT]))]
    frame = (e[:, STEP] - f0).astype(np.int32)
    slot = e[:, SLOT].astype(np.int64)

    out: list[EnemyTrack] = []
    i = 0
    n = len(e)
    while i < n:
        j = i + 1
        while (j < n and slot[j] == slot[i]
               and frame[j] == frame[j - 1] + 1
               and np.hypot(e[j, X] - e[j - 1, X], e[j, Y] - e[j - 1, Y]) <= jump_px):
            j += 1
        if j - i >= min_life:
            seg = e[i:j]
            out.append(EnemyTrack(
                slot=int(slot[i]), birth_frame=int(frame[i]),
                hb=(float(seg[0, HBX]), float(seg[0, HBY])),
                frames=frame[i:j].copy(),
                xy=seg[:, X:Y + 1].astype(np.float32),
            ))
        i = j
    out.sort(key=lambda t: (t.birth_frame, t.slot))
    return out


def summary(npz_path: str | Path) -> None:
    tr = load_enemy_traces(npz_path)
    print(f"{Path(npz_path).name}: {len(tr)} tracks")
    # bucket by hitbox (0x0 = shooter orb, 8x8 = lethal orb, 24/48 = boss aura)
    from collections import defaultdict
    by_hb: dict[tuple, list[EnemyTrack]] = defaultdict(list)
    for t in tr:
        by_hb[(round(t.hb[0]), round(t.hb[1]))].append(t)
    for hb, ts in sorted(by_hb.items()):
        lifes = np.array([t.life for t in ts])
        sp = np.array([np.median(t.step) if t.life > 1 else 0.0 for t in ts])
        turns = []
        for t in ts:
            if t.life < 6:
                continue
            a = np.unwrap(np.arctan2(*np.diff(t.xy, axis=0).T[::-1]))
            turns.append(abs(np.degrees(a[-1] - a[0])))
        turns = np.array(turns) if turns else np.zeros(1)
        print(f"  hb {hb[0]:2}x{hb[1]:<2}  {len(ts):4} tracks  "
              f"life p50 {np.median(lifes):4.0f}  step/f p50 {np.median(sp):5.2f} "
              f"[{np.percentile(sp, 10):.2f}-{np.percentile(sp, 90):.2f}]  "
              f"|turn| p50 {np.median(turns):5.0f}deg")


if __name__ == "__main__":
    import sys
    import glob
    for p in sys.argv[1:] or sorted(glob.glob("sim/fights/letty_*.npz")):
        summary(p)
