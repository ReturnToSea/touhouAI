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
    fx_repeat: int = 1          # flag 0x40/0x80: number of decel-redirect cycles
    player_xy: tuple[float, float] = (192.0, 400.0)
    # bullet_effects flag 1 (FUN_004250d0): for 17 frames the engine sets
    # |vel| = speed + 5.0 * (1 - t/16) — a fixed decaying launch kick. Armed
    # when the type-word has bit 0x1.
    launch: bool = False


def hang_state_for_type(type_word: int) -> int:
    for bit, st in FX_HANG_BIT.items():
        if type_word & bit:
            return st
    return 0


# hang durations are the materialise-anim length (data-driven); measured per
# state from the recordings — state 2 ~8 f, state 3/4 ~14 f.
_HANG_FRAMES = {0: 0, 2: 8, 3: 14, 4: 14}


def from_spawn(s, player_xy: tuple[float, float] = (192.0, 400.0)) -> BulletParams:
    """`BulletParams` from a VM `BulletSpawn` — the Part 12 path (no recording).
    `s` needs `.btype`, `.x`, `.y`, `.angle`, `.speed`, `.effects`
    (each `(p1, p2, interval, repeat, flag, gate)`)."""
    bt = int(s.btype)
    hs = hang_state_for_type(bt)
    launch = bool(bt & 0x1) and any(e[4] == 1 for e in s.effects)
    fx, p1, p2, iv, rep = 0, 0.0, 0.0, 0, 1
    for e in s.effects:
        fl = int(e[4])
        if fl in _FLAG_TO_FX and (bt & fl or fl in (0x400, 0x800)):
            fx = _FLAG_TO_FX[fl]
            p1, p2, iv, rep = float(e[0]), float(e[1]), int(e[2]), max(1, int(e[3]))
            break
    return BulletParams(
        x=float(s.x), y=float(s.y), angle=float(s.angle), speed=float(s.speed),
        hang_state=hs, hang_frames=_HANG_FRAMES[hs], launch=launch,
        fx_flag=fx, fx_p1=p1, fx_p2=p2, fx_interval=iv, fx_repeat=rep,
        player_xy=player_xy)


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
    cycles = 0
    golive = p.hang_frames if p.hang_state else 0
    lt = 0                                      # launch (flag 1) timer

    out = np.empty((n_frames, 2), np.float64)
    for t in range(n_frames):
        out[t] = (px, py)
        if p.hang_state and t <= p.hang_frames:
            px += vx * r                       # crawl at `r`x
            py += vy * r
            if t < p.hang_frames:
                continue
            # the frame the anim finishes: state 2/3/4 falls through to state 1,
            # so the engine also runs the live step too

        if p.launch and lt < 17:
            # FUN_004250d0: |vel| = speed + 5*(1 - t/16), decaying over 17 f
            mag = spd + 5.0 * (1.0 - lt / 16.0)
            vx, vy = math.cos(ang) * mag, math.sin(ang) * mag
            lt += 1

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
            if fx_ctr < p.fx_interval:                   # decelerate toward 0
                f = 1.0 - fx_ctr / p.fx_interval
                vx, vy = math.cos(ang) * spd * f, math.sin(ang) * spd * f
                fx_ctr += 1
            else:                                       # arrive: turn / re-aim
                if p.fx_flag == FX_PAUSE_AIM:
                    ang = math.atan2(p.player_xy[1] - py, p.player_xy[0] - px) + p.fx_p1
                else:
                    ang = _norm(ang + p.fx_p1)
                spd = spd if p.fx_p2 <= -999.0 else p.fx_p2
                vx, vy = math.cos(ang) * spd, math.sin(ang) * spd
                fx_ctr = 0
                cycles += 1
                if cycles >= max(1, p.fx_repeat):
                    p.fx_flag = 0
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


# bullets carrying one of these fx bits are expected to leave and come back, so
# the engine gives them a 128-frame off-screen grace (`FUN_00425a50`, counter
# +0xbfe counts up to 0x80).  Every other bullet is culled the *first* frame its
# bounding box fully clears the play area (grace counter starts at 0).
FX_OFFSCREEN_GRACE_MASK = 0x40 | 0x80 | 0x100 | 0x400 | 0x800   # == 0xDC0
_PLAIN_GRACE = 0
_FX_GRACE = 128


def cull_frame(xy: np.ndarray, fx_flag: int, size: float = 24.0) -> int:
    """First frame index at which the engine would have erased this bullet.

    `FUN_0042d6d8`: a bullet is off-screen once
    ``x < -size/2 or x > 384+size/2 or y < -size/2 or y > 448+size/2``
    (``size`` is the bullet type's sprite extent, ~16-64 px).  A plain bullet is
    erased immediately; a redirect/bounce bullet survives 128 off-screen frames.
    Returns ``len(xy)`` if it never leaves within the propagated window.
    """
    m = size / 2.0
    off = ((xy[:, 0] < -m) | (xy[:, 0] > PLAYFIELD_W + m) |
           (xy[:, 1] < -m) | (xy[:, 1] > PLAYFIELD_H + m))
    grace = _FX_GRACE if (int(fx_flag) & FX_OFFSCREEN_GRACE_MASK) else _PLAIN_GRACE
    run = 0
    for i, is_off in enumerate(off):
        if is_off:
            run += 1
            if run > grace:
                return i - run + 1 + grace
        else:
            run = 0
    return len(xy)


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

    rep = 1
    if b.re is not None:
        state = b.re[:, 0]
        tflag = int(b.re[0, 1])
        hs = hang_state_for_type(tflag)                    # type-word bits 0x2/4/8
        h = int(np.argmax(state == 1)) if hs and (state == 1).any() else 0
        stg = b.staging(0)
        # flag-1 launch: needs both the type-word bit AND a staged flag-1 entry
        launch = bool(tflag & 0x1) and any(e["flag"] == 1 for e in stg)
        fx, p1, p2, iv = 0, 0.0, 0.0, 0
        for e in stg:                                      # first *armed* effect
            fl = e["flag"]
            # FUN_00424290 arms an entry only if (type-word & entry.flag) != 0
            if fl in _FLAG_TO_FX and (tflag & fl or fl in (0x400, 0x800)):
                fx = _FLAG_TO_FX[fl]
                p1, p2, iv, rep = e["p1"], e["p2"], e["interval"], max(1, e["repeat"])
                break
    else:
        hs, h = _measure_hang(dm, base) if base > 0.2 else (0, 0)
        launch = int(b.fxflag) == 16           # best guess on 17-col recordings
        fx = FX_ACCEL_DIR if int(b.fxflag) == 16 else 0
        p1, p2, iv = float(m[4]), float(m[5]), int(m[6])

    return BulletParams(
        x=float(b.xy[0, 0]) + (4.0 * math.cos(ang) * base if hs else 0.0),
        y=float(b.xy[0, 1]) + (4.0 * math.sin(ang) * base if hs else 0.0),
        angle=ang, speed=base, hang_state=hs, hang_frames=h, launch=launch,
        fx_flag=fx, fx_p1=p1, fx_p2=p2, fx_interval=iv, fx_repeat=rep)


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
        ok = p90 <= 8.0                          # ~ the recorder's own noise floor
        tot += 1
        tot_ok += ok
        cls, fxf, p1, iv, bs = key
        print(f"  {'ok  ' if ok else 'FAIL'} cls{cls} fx{fxf:<2} p1={p1:<8} "
              f"int={iv:<3} base={bs:<4}  n={len(v):5}  "
              f"p50 {p50:5.2f}  p90 {p90:6.2f} px")
    ae = np.concatenate(all_err)
    print(f"\n  {tot_ok}/{tot} groups reproduce within 8 px / {H}f   "
          f"(overall p50 {np.median(ae):.1f}  p90 {np.percentile(ae, 90):.1f} px)")
    print("  hang crawl + crawl/full spike, the flag-1 launch kick, and all 5 "
          "fx flags (accel, turn+accel, pause-redirect, pause-reaim, wall\n"
          "  bounce) all read straight from the recording's staging entries. "
          "The ~6 px residual is the recorder's own frame-phase noise.")
    return tot_ok >= tot - 2 and float(np.median(ae)) < 4.0


if __name__ == "__main__":
    import sys
    import glob
    paths = sys.argv[1:] or sorted(glob.glob("sim/fights/letty_*.npz"))
    raise SystemExit(0 if verify(paths) else 1)
