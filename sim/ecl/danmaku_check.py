"""End-to-end check: VM spawn schedule -> bullet_sim propagation -> danmaku.

The VM's danmaku can't be compared bullet-for-bullet to a recording (the boss
repositions on RNG, so everything downstream diverges after ~frame 190). What
*must* match is the aggregate: how many bullets are on screen over time, and
where. This runs the whole pipeline and checks that against the recordings.

    python -m sim.ecl.danmaku_check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .parser import parse_file
from .vm import VM
from .bullet_sim import from_spawn, simulate, cull_frame
from .bullet_trace import load_traces

_ECL = Path(__file__).resolve().parents[2] / "tools" / "th07_ecl" / "ecldata1.ecl"
_FIGHTS = Path(__file__).resolve().parents[2] / "sim" / "fights"
PF_W, PF_H = 384.0, 448.0
MAXLIFE = 400            # frames to propagate each bullet


def vm_density(seed: int = 0, frames: int = 12000) -> np.ndarray:
    """On-screen bullet count per frame, from VM spawns propagated by bullet_sim."""
    ecl = parse_file(str(_ECL))
    vm = VM(ecl, difficulty=3, seed=seed)
    vm.start_boss(sub=31, interrupt=0)
    vm.run(frames)

    count = np.zeros(frames + MAXLIFE, np.int32)
    for s in vm.bullets:
        p = from_spawn(s)
        xy = simulate(p, MAXLIFE)
        on0 = (-8 <= xy[0, 0] < PF_W + 8) and (-8 <= xy[0, 1] < PF_H + 8)
        if not on0:                       # spawned off-screen -> engine never shows it
            continue
        # the engine erases the bullet the frame it clears the play area (plain)
        # or after a 128-frame grace (redirect/bounce) -- it does NOT come back.
        life = cull_frame(xy, p.fx_flag)
        f0 = int(s.frame)
        count[f0:f0 + life] += 1
    return count[:frames]


def rec_density(npz_path: str | Path) -> np.ndarray:
    d = np.load(npz_path)
    b = d["bullets"]
    f = (b[:, 0] - b[:, 0].min()).astype(int)
    return np.bincount(f)


def main(argv) -> int:
    import glob
    recs = argv[1:] or sorted(glob.glob(str(_FIGHTS / "letty_[0-9]*.npz")))

    vm_c = vm_density()
    # align on the first bullet (VM frame ~60; recordings are f0-relative)
    vm_first = int(np.argmax(vm_c > 0))
    vm_c = vm_c[vm_first:]

    rec_cs = [rec_density(p) for p in recs]
    L = min(len(vm_c), min(len(r) for r in rec_cs), 10600)
    vm_c = vm_c[:L]
    rec_mean = np.mean([r[:L] for r in rec_cs], 0)

    # 1. total bullets over the fight
    ratio = vm_c.sum() / rec_mean.sum()
    # 2. shape: correlation of the smoothed count curves
    k = np.ones(120) / 120
    a = np.convolve(vm_c, k, "same")
    bcv = np.convolve(rec_mean, k, "same")
    corr = float(np.corrcoef(a, bcv)[0, 1])
    # 3. peak on-screen count (playfield saturation)
    vm_peak, rec_peak = int(a.max()), int(bcv.max())

    print(f"  bullets on screen, VM vs recorded ({len(recs)} fights):")
    print(f"    total ratio        {ratio:5.2f}   (want ~1.0)")
    print(f"    curve correlation  {corr:5.2f}   (want > 0.9)")
    print(f"    peak on-screen     VM {vm_peak}  rec {rec_peak}")
    for lo in range(0, L - 600, 1500):
        seg_v = a[lo:lo + 1500].mean()
        seg_r = bcv[lo:lo + 1500].mean()
        print(f"      f{lo:5}-{lo+1500:<5}: VM {seg_v:6.0f}   rec {seg_r:6.0f}")

    ok = 0.80 < ratio < 1.25 and corr > 0.88
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
