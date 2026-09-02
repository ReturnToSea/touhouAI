"""Part 10 (first pass) — per-type median-displacement-profile motion models.

SUPERSEDED by `bullet_sim.py`. Once the th07.exe RE gave us the engine-faithful
per-bullet forward sim (hang states, the flag-1 launch, all fx flags), the
non-parametric median profile stopped being needed: `bullet_sim.simulate` from a
bullet's own params reproduces recorded trajectories to **p50 ~2 px / p90 ~7 px /
98 % within 8 px** (the recorder's pos-vs-vel sampling noise floor) — see
`bullet_sim.verify` and `align.refit_coverage`. The "~75 % / ~16 % tail" this
module reports is an artefact of averaging genuinely-random `bullet_random`
bullets into one profile, not a physics gap. Kept for the grouping analysis and
as the fallback if a future boss needs a profile for a type `bullet_sim` can't
express.

The recorder polls the bullet pool every frame; `bullet_trace` re-keys that into
per-bullet trajectories. This module groups those trajectories by an observable
"type" and fits a motion model each group can be replayed from.

What the recordings tell us (Letty, Stage 1 boss, Lunatic — 3 runs, ~45k bullets):

  * Position is exactly `cumsum` of the per-frame displacement `diff(xy)` — so we
    model displacement directly, not the pool's `speed`/`vel` fields (those are
    read at a different point in the engine frame and integrate to a ~13 px p90
    drift over 90 frames — the recorder's own noise floor).
  * Most bullets: constant heading, a short "catch-up" speed transient in the
    first ~15 frames, then constant speed (optionally a slow `bullet_effects`
    accel ramp — `fx_p1` per frame for `fx_interval` frames).
  * ~13 % turn: a single sharp heading change at one frame (a `bullet_effects`
    redirect), or a 180° flip when a decel ramp drives speed through zero
    ("Lingering Cold" snow that drifts out and falls back).

Model = the **median displacement profile** of the group, in the bullet's own
frame (rotated so displacement[0] points along +x): `mag[t]` and `dheading[t]`
tables. Forward-sim is `pos0 + cumsum(mag · [cos,sin](heading0 + dheading))`.
Non-parametric, trivially a GPU lookup, and exact for any behaviour the group
shares.

Coverage ceiling from the recording alone is ~75 % of bullets within 5 px / 90
frames. The residual tail is groups that share `(class, fx_flag, fx_p1,
fx_interval, base_speed)` but come from *different* ECL instructions with
genuinely different scripted motion — closing it needs each recorded bullet
linked back to its VM spawn (Part 11), so this waits on that.

    python -m sim.ecl.fit_motion            # fit + coverage report on sim/fights/letty_*
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bullet_trace import load_traces, Bullet

HORIZON = 90            # frames the model must track (plan bar)
TOL_PX = 5.0            # per the plan; ~= the recorder's own p50 noise
PROFILE_LEN = 200       # tabulated frames; past this, hold the last value
MIN_GROUP = 30


@dataclass
class MotionModel:
    key: tuple                 # (cls, fx_flag, fx_p1, fx_interval, base_speed, *behaviour)
    mag: np.ndarray            # [PROFILE_LEN] median per-frame displacement, px
    dheading: np.ndarray       # [PROFILE_LEN] median heading offset from frame 0, rad
    n: int

    def simulate(self, spawn_xy, heading0, n_frames):
        """Forward-integrate one bullet from its spawn position and initial
        heading. Returns [n_frames, 2] positions (position 0 == spawn_xy)."""
        idx = np.minimum(np.arange(n_frames - 1), PROFILE_LEN - 1)
        h = heading0 + self.dheading[idx]
        step = np.stack([np.cos(h) * self.mag[idx], np.sin(h) * self.mag[idx]], -1)
        return spawn_xy + np.cumsum(np.vstack([[0.0, 0.0], step]), 0)


def _base_speed(b: Bullet) -> float:
    """The engine's scalar speed field at spawn — constant, and the axis the
    launch transient and the fx accel ramp scale with. Bucketed coarsely: it
    separates sub-populations the fx fields alone don't."""
    if b.motion is not None:
        return float(b.motion[0, 0])
    return float(np.hypot(b.vel[0, 0], b.vel[0, 1]))


def _disp(b: Bullet, upto: int) -> np.ndarray:
    """Per-frame displacement vectors, frames 0 .. upto-1."""
    return np.diff(b.xy[: min(b.life, upto + 1)].astype(np.float64), axis=0)


def _behaviour(b: Bullet) -> tuple:
    """Cheap observable split so a mixed group separates into replayable cohorts:
    does the heading ever swing >8°, and does the speed roughly double?"""
    d = _disp(b, 120)
    if len(d) < 20:
        return (0, 0)
    a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turns = int(np.abs(a - a[0]).max() > np.deg2rad(8))
    m = np.hypot(d[:, 0], d[:, 1])
    ramps = int(np.median(m[-10:]) > 1.3 * np.median(m[:10]) + 0.1)
    return (turns, ramps)


def _type_key(b: Bullet) -> tuple:
    m = b.motion[0] if b.motion is not None else np.zeros(7)
    fx_p1 = round(float(m[4]), 5)
    fx_int = int(m[6])
    base = round(_base_speed(b) * 4) / 4
    return (b.cls, b.fxflag, fx_p1, fx_int, base) + _behaviour(b)


def _fit_group(bullets: list[Bullet]) -> MotionModel:
    T = PROFILE_LEN
    mag = np.full((len(bullets), T), np.nan)
    dh = np.full((len(bullets), T), np.nan)
    for i, b in enumerate(bullets):
        d = _disp(b, T)
        n = len(d)
        if n == 0:
            continue
        mag[i, :n] = np.hypot(d[:, 0], d[:, 1])
        a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dh[i, :n] = a - a[0]
    mp = np.nanmedian(mag, 0)
    dp = np.nanmedian(dh, 0)
    # hold the last non-nan value forward (short-lived bullets leave a nan tail)
    for prof, fill in ((mp, mp), (dp, dp)):
        last = np.where(~np.isnan(prof))[0]
        if len(last):
            prof[last[-1] + 1:] = prof[last[-1]]
        prof[np.isnan(prof)] = 0.0
    return MotionModel(_type_key(bullets[0]), mp, dp, len(bullets))


def fit(npz_paths: list[str]) -> dict[tuple, MotionModel]:
    bullets: list[Bullet] = []
    for p in npz_paths:
        bullets += [b for b in load_traces(p) if b.life >= 30]
    groups: dict[tuple, list[Bullet]] = {}
    for b in bullets:
        groups.setdefault(_type_key(b), []).append(b)
    return {k: _fit_group(v) for k, v in groups.items() if len(v) >= MIN_GROUP}


def verify(npz_paths: list[str], models: dict[tuple, MotionModel],
           horizon: int = HORIZON, tol: float = TOL_PX) -> dict:
    bullets: list[Bullet] = []
    for p in npz_paths:
        bullets += [b for b in load_traces(p) if b.life >= 30]

    errs: list[float] = []
    covered = 0
    per_group: dict[tuple, list[float]] = {}
    for b in bullets:
        k = _type_key(b)
        m = models.get(k)
        n = min(b.life - 1, horizon)
        if n < 10:
            continue
        covered += m is not None
        if m is None:
            continue
        d0 = _disp(b, 1)[0]
        e = np.hypot(*(m.simulate(b.xy[0].astype(np.float64),
                                  np.arctan2(d0[1], d0[0]), n + 1)[: n + 1]
                       - b.xy[: n + 1].astype(np.float64)).T).max()
        errs.append(e)
        per_group.setdefault(k, []).append(e)

    errs = np.array(errs) if errs else np.zeros(1)
    return {
        "bullets": len(errs),
        "modelled_frac": covered / max(1, sum(
            1 for b in bullets if b.life - 1 >= 10)),
        "within_tol": float(np.mean(errs <= tol)),
        "p50": float(np.median(errs)),
        "p75": float(np.percentile(errs, 75)),
        "p90": float(np.percentile(errs, 90)),
        "p99": float(np.percentile(errs, 99)),
        "per_group": {k: (len(v), float(np.median(v)), float(np.percentile(v, 90)))
                      for k, v in sorted(per_group.items())},
    }


def main(argv):
    import glob
    paths = argv[1:] or sorted(glob.glob("sim/fights/letty_*.npz"))
    models = fit(paths)
    print(f"fit {len(models)} bullet-motion models from {len(paths)} recordings\n")
    r = verify(paths, models)
    for k, (n, p50, p90) in r["per_group"].items():
        flag = "ok  " if p90 <= TOL_PX else "TAIL"
        print(f"  {flag} {k!s:34} n={n:5}  p50 {p50:5.2f}  p90 {p90:6.2f} px")
    print(f"\n  bullets modelled:   {r['modelled_frac'] * 100:5.1f}% of >=10-frame bullets")
    print(f"  within {TOL_PX:.0f} px / {HORIZON}f: {r['within_tol'] * 100:5.1f}%")
    print(f"  path err  p50 {r['p50']:.2f}  p75 {r['p75']:.2f}  "
          f"p90 {r['p90']:.2f}  p99 {r['p99']:.1f} px")
    # informational: the recording's own frame-to-frame consistency
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
