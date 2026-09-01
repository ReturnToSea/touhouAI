"""Reconstruct per-bullet trajectories from a recorded fight (`sim/fights/*.npz`).

The recorder polls the whole bullet pool every frame; this re-keys that stream
by bullet identity so Part 10 can fit per-type motion models (`delay_frames`,
`accel`, `turn_rate`, `speed_final`).

Identity is the pool slot. A bullet is born when its slot goes empty→occupied
and dies when it goes occupied→empty. Slots sit empty for many frames between
reuses — measured across ~2M frame-to-frame transitions in one recording: the
worst position residual (actual vs `pos + vel`) is 6.2 px, and there are *zero*
same-slot swaps on consecutive frames — so slot presence is a reliable identity
and needs no distance heuristic.

    from sim.ecl.bullet_trace import load_traces, verify
    traces = load_traces("sim/fights/letty_0.npz")
    verify("sim/fights/letty_0.npz")        # raises on any inconsistency
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# recorded `bullets` columns — 0-9 always present, 10-16 added for the motion models
STEP, SLOT, X, Y, VX, VY, CLS, FXF, HBX, HBY = range(10)
SPEED, ACCEL, ANGVEL, ANGLE, FXP1, FXP2, FXINT = range(10, 17)


@dataclass
class Bullet:
    id: int
    slot: int
    birth_frame: int
    cls: int
    fxflag: int
    frames: np.ndarray      # int32 [n]
    xy: np.ndarray          # float32 [n, 2]
    vel: np.ndarray         # float32 [n, 2]
    motion: np.ndarray | None = None   # float32 [n, 7]: speed,accel,angvel,angle,fxp1,fxp2,fxint (if recorded)

    @property
    def life(self) -> int:
        return len(self.frames)

    @property
    def speed(self) -> np.ndarray:
        if self.motion is not None:
            return self.motion[:, 0]
        return np.hypot(self.vel[:, 0], self.vel[:, 1])

    @property
    def angle(self) -> np.ndarray:
        if self.motion is not None:
            return self.motion[:, 3]
        return np.arctan2(self.vel[:, 1], self.vel[:, 0])


def load_traces(npz_path: str | Path) -> list[Bullet]:
    d = np.load(npz_path)
    b = d["bullets"]
    if len(b) == 0:
        return []
    f0 = int(b[:, STEP].min())

    order = np.lexsort((b[:, SLOT], b[:, STEP]))
    b = b[order]
    frame = (b[:, STEP] - f0).astype(np.int32)
    slot = b[:, SLOT].astype(np.int64)

    active: dict[int, list[int]] = {}          # slot -> list of row indices, current bullet
    active_prev_frame: dict[int, int] = {}
    out: list[Bullet] = []
    next_id = 0

    # per frame, the set of (slot -> row index)
    fr_uniq = np.unique(frame)
    row_by_frame: dict[int, dict[int, int]] = {}
    for i in range(len(b)):
        row_by_frame.setdefault(int(frame[i]), {})[int(slot[i])] = i

    has_motion = b.shape[1] >= 17

    def _finish(sl: int) -> None:
        rows = active.pop(sl)
        active_prev_frame.pop(sl, None)
        nonlocal next_id
        seg = b[rows]
        out.append(Bullet(
            id=next_id, slot=sl, birth_frame=int(seg[0, STEP] - f0),
            cls=int(seg[0, CLS]), fxflag=int(seg[0, FXF]),
            frames=(seg[:, STEP] - f0).astype(np.int32),
            xy=seg[:, X:Y + 1].astype(np.float32),
            vel=seg[:, VX:VY + 1].astype(np.float32),
            motion=seg[:, SPEED:FXINT + 1].astype(np.float32) if has_motion else None,
        ))
        next_id += 1

    for f in fr_uniq:
        f = int(f)
        present = row_by_frame[f]
        for sl in list(active):
            if sl not in present or active_prev_frame[sl] != f - 1:
                _finish(sl)                     # slot vanished, or a 1-frame gap => reuse
        for sl, ri in present.items():
            if sl in active:
                active[sl].append(ri)
            else:
                active[sl] = [ri]
            active_prev_frame[sl] = f
    for sl in list(active):
        _finish(sl)

    out.sort(key=lambda t: (t.birth_frame, t.slot))
    for i, t in enumerate(out):
        t.id = i
    return out


def verify(npz_path: str | Path, *, jump_px: float = 12.0) -> dict:
    """Round-trip + consistency checks. Raises AssertionError on any failure."""
    d = np.load(npz_path)
    b = d["bullets"]
    traces = load_traces(npz_path)

    # 1. no row lost or duplicated
    n_rows = sum(t.life for t in traces)
    assert n_rows == len(b), f"row count: {n_rows} traced vs {len(b)} recorded"

    # 2. rebuilding the per-frame (frame,x,y) multiset from traces == original
    rebuilt = np.concatenate([
        np.column_stack([t.frames, t.xy]) for t in traces
    ]) if traces else np.zeros((0, 3))
    f0 = int(b[:, STEP].min())
    orig = np.column_stack([b[:, STEP] - f0, b[:, X], b[:, Y]])
    rb = rebuilt[np.lexsort(rebuilt.T[::-1])]
    og = orig[np.lexsort(orig.T[::-1])]
    assert np.array_equal(rb, og), "rebuilt (frame,x,y) stream != recorded stream"

    # 3. within each trace, frames are strictly consecutive
    bad_gaps = sum(1 for t in traces if t.life > 1 and np.any(np.diff(t.frames) != 1))
    assert bad_gaps == 0, f"{bad_gaps} traces have non-consecutive frames"

    # 4. no large internal position jump (would mean a mis-tracked slot reuse)
    resid = []
    jumps = 0
    for t in traces:
        if t.life < 2:
            continue
        pred = t.xy[:-1] + t.vel[:-1]
        e = np.hypot(pred[:, 0] - t.xy[1:, 0], pred[:, 1] - t.xy[1:, 1])
        resid.append(e)
        jumps += int(np.any(e > jump_px))
    resid = np.concatenate(resid) if resid else np.zeros(1)
    assert jumps == 0, f"{jumps} traces contain a >{jump_px}px internal jump"

    stats = {
        "recording": Path(npz_path).name,
        "bullets": len(traces),
        "rows": len(b),
        "frames": int(b[:, STEP].max() - b[:, STEP].min()),
        "life_p50": int(np.median([t.life for t in traces])),
        "life_max": max(t.life for t in traces),
        "singletons": sum(1 for t in traces if t.life == 1),
        "resid_p99_px": round(float(np.percentile(resid, 99)), 3),
        "resid_max_px": round(float(resid.max()), 3),
        "fxflags": dict(zip(*[x.tolist() for x in np.unique(
            [t.fxflag for t in traces], return_counts=True)])),
        "classes": dict(zip(*[x.tolist() for x in np.unique(
            [t.cls for t in traces], return_counts=True)])),
    }
    return stats


if __name__ == "__main__":
    import sys
    import glob

    paths = sys.argv[1:] or sorted(glob.glob("sim/fights/letty_*.npz"))
    for p in paths:
        s = verify(p)
        print(f"{s['recording']:16} {s['bullets']:6} bullets / {s['rows']:8} rows / "
              f"{s['frames']}f   life p50={s['life_p50']} max={s['life_max']} "
              f"singletons={s['singletons']}   resid p99={s['resid_p99_px']} "
              f"max={s['resid_max_px']}px   fx={s['fxflags']}  cls={s['classes']}")
    print("all recordings pass round-trip + consistency checks")
