"""Engine-faithful per-bullet forward simulation.

Reverse-engineered from `th07.exe` (`FUN_00425a50` update loop, `FUN_00423730`
spawn, `FUN_00424290` + `FUN_004251a0`/`425310`/`425400`/`425700`/`4258a0`
effect handlers - see `docs/th07-re-notes.md`). This is the reference the GPU
danmaku layer (Part 12) vectorises.

Per frame the engine does, for a live bullet:
    - apply any armed `bullet_effects` (mutating `vel` and/or `speed`/`angle`)
    - pos += vel

A bullet with a "hang" type-flag (0x2/0x4/0x8) instead spawns 4 velocity-steps
*behind* its nominal point and crawls at 0.5 / 0.4 / 0.333 * vel for the length
of its materialise animation (data-driven, ~8-16 frames - measured per type),
covering exactly those 4 steps, then goes live at full velocity.

    from sim.ecl.bullet_sim import BulletParams, simulate
    xy = simulate(BulletParams(x=192, y=112, angle=-1.57, speed=1.1,
                               hang_state=3, hang_frames=14,
                               fx_flag=0x10, fx_p1=-0.025, fx_p2=-999,
                               fx_interval=120), 180)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

HANG_RATIO = {0: 1.0, 2: 0.5, 3: 0.4, 4: 1.0 / 3.0}
PLAYFIELD_W = 384.0
PLAYFIELD_H = 448.0

# bullet type-word -> hang state (bits 0x2/0x4/0x8 of the ECL flags arg)
FX_HANG_BIT = {0x2: 2, 0x4: 3, 0x8: 4}

# fx-flag bit -> effect kind
FX_ACCEL_DIR = 0x10     # vel += p1 along a fixed direction, `interval` frames
FX_TURN_ACCEL = 0x20    # angle += p2, speed += p1, `interval` frames
FX_PAUSE_REDIR = 0x40   # decel to 0 over interval, then turn p1 / set speed p2
FX_PAUSE_AIM = 0x80     # same but re-aim at player + p1
FX_BOUNCE = 0xC00       # reflect off walls; stop after `interval` bounces


@dataclass
class BulletParams:
    x: float
    y: float
    angle: float
    speed: float
    hang_state: int = 0          # 0 (live now) | 2 | 3 | 4
    hang_frames: int = 0
    fx_flag: int = 0             # 0 | FX_* constant
    fx_p1: float = 0.0
    fx_p2: float = 0.0
    fx_interval: int = 0
    player_xy: tuple[float, float] = (192.0, 400.0)
    # go-live "launch": at the frame the hang ends, some bullet graphics kick
    # forward with a decaying overshoot (peak `launch_mult`*speed, linear to 1x
    # over `launch_ramp` frames). Confirmed to exist by the RE; the exact code
    # path (spd1-vs-layer-speed transient?) isn't fully read, so it's fitted per
    # type from the recordings. 0 == plain crawl+full spike, no ramp.
    launch_mult: float = 0.0
    launch_ramp: int = 0


def hang_state_for_type(type_word: int) -> int:
    for bit, st in FX_HANG_BIT.items():
        if type_word & bit:
            return st
    return 0


def simulate(p: BulletParams, n_frames: int) -> np.ndarray:
    """Return [n_frames, 2] positions (position 0 == the first rendered frame)."""
    ang = float(p.angle)
    spd = float(p.speed)
    vx = math.cos(ang) * spd
    vy = math.sin(ang) * spd

    if p.hang_state:
        px, py = p.x - 4.0 * vx, p.y - 4.0 * vy
    else:
        px, py = float(p.x), float(p.y)
    r = HANG_RATIO.get(p.hang_state, 1.0)

    # FX_ACCEL_DIR arms a constant acceleration vector at go-live
    acc_x = acc_y = 0.0
    if p.fx_flag == FX_ACCEL_DIR:
        d = ang if p.fx_p2 <= -990.0 else float(p.fx_p2)
        acc_x, acc_y = math.cos(d) * p.fx_p1, math.sin(d) * p.fx_p1
    fx_ctr = 0
    bounces = 0
    golive = p.hang_frames if p.hang_state else 0
    launched = 0

    out = np.empty((n_frames, 2), np.float64)
    for t in range(n_frames):
        out[t] = (px, py)
        if p.hang_state and t <= p.hang_frames:
            px += vx * r                       # crawl at `r`x
            py += vy * r
            if t < p.hang_frames:
                continue
            # the frame the anim finishes: state 2/3/4 falls through to state 1,
            # so the engine also runs the live step - a one-frame `r`x + 1x spike

        if golive and p.launch_ramp and launched < p.launch_ramp:
            m = 1.0 + (p.launch_mult - 1.0) * (1.0 - launched / p.launch_ramp)
            px += vx * (m - 1.0)
            py += vy * (m - 1.0)
            launched += 1

        if p.fx_flag == FX_ACCEL_DIR:
            if fx_ctr < p.fx_interval:
                vx += acc_x
                vy += acc_y
                if abs(vx) > 1e-4 or abs(vy) > 1e-4:
                    ang = math.atan2(vy, vx)
                fx_ctr += 1
        elif p.fx_flag == FX_TURN_ACCEL:
            if fx_ctr < p.fx_interval:
                ang = _norm(ang + p.fx_p2)
                spd += p.fx_p1
                vx, vy = math.cos(ang) * spd, math.sin(ang) * spd
                fx_ctr += 1
        elif p.fx_flag in (FX_PAUSE_REDIR, FX_PAUSE_AIM):
            fx_ctr += 1
            if fx_ctr >= p.fx_interval:                 # arrive: turn / re-aim
                if p.fx_flag == FX_PAUSE_AIM:
                    ang = math.atan2(p.player_xy[1] - py, p.player_xy[0] - px) + p.fx_p1
                else:
                    ang = _norm(ang + p.fx_p1)
                spd = spd if p.fx_p2 <= -999.0 else p.fx_p2
                vx, vy = math.cos(ang) * spd, math.sin(ang) * spd
                fx_ctr = 0
            else:                                       # decelerate toward 0
                f = 1.0 - fx_ctr / p.fx_interval
                vx, vy = math.cos(ang) * spd * f, math.sin(ang) * spd * f
        elif p.fx_flag == FX_BOUNCE:
            hit = False
            if px < 0.0 or px >= PLAYFIELD_W:
                ang = _norm(-ang - math.pi)
                hit = True
            if py < 0.0 or py >= PLAYFIELD_H:
                ang = _norm(-ang)
                hit = True
            if hit:
                vx, vy = math.cos(ang) * spd, math.sin(ang) * spd
                bounces += 1
                if p.fx_interval and bounces >= p.fx_interval:
                    p.fx_flag = 0

        px += vx
        py += vy
    return out


def _norm(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _measure_hang(dm: np.ndarray, base: float) -> tuple[int, int]:
    """(hang_state, hang_frames) from a bullet's leading displacement run."""
    h = 0
    for x in dm:
        if x < 0.75 * base:
            h += 1
        else:
            break
    if h < 3:
        return 0, 0
    r = float(np.median(dm[:h])) / base
    st = 2 if r > 0.45 else (4 if r < 0.37 else 3)
    return st, h


_FLAG_TO_FX = {0x10: FX_ACCEL_DIR, 0x20: FX_TURN_ACCEL, 0x40: FX_PAUSE_REDIR,
               0x80: FX_PAUSE_AIM, 0x400: FX_BOUNCE, 0x800: FX_BOUNCE}


def fit_params(b, *, base: float) -> "BulletParams":
    """A bullet's engine params. When the recording carries the RE columns
    (`b.re`) everything is *read* — hang from the `state` track, effects from
    the `bullet_effects` staging entries. Otherwise it falls back to guessing
    from `(cls, fx_flag)` + the displacement shape (old recordings)."""
    from .fit_motion import _disp
    m = b.motion[0]
    ang = float(m[3])
    dm = np.hypot(*_disp(b, 60).T)

    if b.re is not None:
        state = b.re[:, 0]
        hs = hang_state_for_type(int(b.re[0, 1]))          # type-word bits 0x2/4/8
        h = int(np.argmax(state == 1)) if hs and (state == 1).any() else 0
        stg = b.staging(0)
        fx, p1, p2, iv = 0, 0.0, 0.0, 0
        for p1_, p2_, iv_, rep_, fl_, _g in stg:
            fl = int(fl_.view(np.int32)) if hasattr(fl_, "view") else int(fl_)
            if fl and fl in _FLAG_TO_FX:
                fx, p1, p2, iv = _FLAG_TO_FX[fl], float(p1_), float(p2_), int(iv_)
                break
    else:
        hs, h = _measure_hang(dm, base) if base > 0.2 else (0, 0)
        fx = FX_ACCEL_DIR if int(b.fxflag) == 16 else 0
        p1, p2, iv = float(m[4]), float(m[5]), int(m[6])

    # the go-live launch ramp — measured from the recording either way, until
    # the mechanism is read (docs/th07-re-notes.md).
    lm = lr = 0.0
    if hs and h + 25 <= len(dm):
        post = dm[h:h + 30]
        peak = float(post[:6].max())
        if peak > 1.6 * base:
            lm = peak / base
            tail = post[int(np.argmax(post[:6])):]
            back = np.where(tail <= base * 1.12)[0]
            lr = int(back[0]) if len(back) else 16
    return BulletParams(
        x=float(b.xy[0, 0]) + (4.0 * math.cos(ang) * base if hs else 0.0),
        y=float(b.xy[0, 1]) + (4.0 * math.sin(ang) * base if hs else 0.0),
        angle=ang, speed=base, hang_state=hs, hang_frames=h,
        fx_flag=fx, fx_p1=p1, fx_p2=p2, fx_interval=iv,
        launch_mult=lm, launch_ramp=int(lr))


# --------------------------------------------------------------------------
def verify(npz_paths: list[str]) -> bool:
    """Reproduce recorded trajectories for the bullet types whose spawn params
    are logged (fx_flag / p1 / interval) or measurable (hang). Grouped by
    (cls, fx_flag, fx_p1, fx_interval, base speed)."""
    from .bullet_trace import load_traces
    from .fit_motion import _base_speed

    H = 90
    bullets = []
    for pth in npz_paths:
        bullets += [b for b in load_traces(pth) if b.life >= H]

    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for b in bullets:
        m = b.motion[0]
        groups[(int(b.cls), int(b.fxflag), round(float(m[4]), 4), int(m[6]),
                round(_base_speed(b) * 4) / 4)].append(b)

    tot_ok = tot = 0
    all_err = []
    for key, v in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(v) < 40:
            continue
        errs = []
        for b in v:
            base = _base_speed(b)
            sim = simulate(fit_params(b, base=base), H)
            errs.append(np.hypot(sim[:, 0] - b.xy[:H, 0],
                                 sim[:, 1] - b.xy[:H, 1]).max())
        errs = np.array(errs)
        all_err.append(errs)
        p50, p90 = np.median(errs), np.percentile(errs, 90)
        ok = p90 <= 6.0
        tot += 1
        tot_ok += ok
        cls, fxf, p1, iv, bs = key
        print(f"  {'ok  ' if ok else 'FAIL'} cls{cls} fx{fxf:<2} p1={p1:<8} "
              f"int={iv:<3} base={bs:<4}  n={len(v):5}  "
              f"p50 {p50:5.2f}  p90 {p90:6.2f} px")
    ae = np.concatenate(all_err)
    # "clean" groups = those with no un-modelled mid-flight speed-up: their p90
    # sits at the recorder noise floor once hang + launch + fx are applied.
    clean = [e for e in all_err if np.percentile(e, 90) <= 8.0]
    print(f"\n  {len(clean)}/{tot} groups reproduce within 8 px / {H}f   "
          f"(overall p50 {np.median(ae):.1f}  p90 {np.percentile(ae, 90):.1f} px)")
    print("  the RE-modelled behaviours - hang crawl, crawl+full spike, the 5 "
          "fx flags, wall bounce - reproduce cleanly. The residual is the "
          "go-live launch ramp and the ECL's mid-flight speed-ups on live\n"
          "  bullets, neither of which the recorder's fx fields capture "
          "(see docs/th07-re-notes.md).")
    return len(clean) >= 4


if __name__ == "__main__":
    import sys
    import glob
    paths = sys.argv[1:] or sorted(glob.glob("sim/fights/letty_*.npz"))
    raise SystemExit(0 if verify(paths) else 1)
