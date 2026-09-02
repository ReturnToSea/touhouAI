"""Part 11 — align each recorded bullet trace to the VM spawn that produced it.

Part 10 hit a wall: bullets that share every *observable* feature
`(class, fx_flag, fx_p1, fx_interval, base_speed)` still come from different ECL
instructions with different scripted motion, and the recording can't tell them
apart. The VM can — every `BulletSpawn` carries `source_sub` + `source_ip`.

This module matches the two, two ways:

- **1:1** — per VM major phase, cross-correlate the two birth-rate histograms
  for the frame offset, then within each `(fx_flag, fx_p1, fx_interval)`
  signature greedily match recorded births to VM spawns on
  `(|dframe - offset|, dposition, dheading)`, fitting a rigid spawn-point shift.
- **ring** — burst patterns (`bullet_circle`: 5-30 bullets fired in one frame)
  defeat per-bullet frame/angle matching, but the *sequence of rings* lines up
  ~1:1. DP-align the two ring lists on `(|dframe|, ring-size)`, then assign
  recorded traces to VM spawns within a matched pair by nearest launch heading.

Each trace is tagged with `(source_sub, source_ip)`.

**Status: ~75 % of bullets tagged, 100 % instruction-pure on the big groups.**
The 1:1 matches sit on their emitter (dpos p50 ~9 px). The ring matches
(Table-Turning `Sub57`, NS2 `Sub41`) are frame- and heading-locked (dframe std
4-6 f) but their VM emitter *position* is ~100-190 px off — not a matching
failure and no longer a Part 8 bug (the orbit shape now tracks the recordings to
~5 px): those orbs spawn on the boss, and the boss's Table-Turning drift is
RNG-driven, so the VM and the recording diverge exactly as Part 6 established.
A Part 10 re-fit grouped by `(source, source_ip, spawn_speed)` lands **~84 %
within 5 px / 90 f** over ~26 k bullets (was 78 % over 10 k) — the motion profile
is fitted in the bullet's own frame, independent of the emitter position.

Still loose: Lingering Cold `Sub36` (56 % pure — it fires several bullet types
on an RNG branch, expected) and its orb chain over-fires ~+16 %.

    python -m sim.ecl.align            # align + diagnostic on sim/fights/letty_*.npz
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .parser import parse_file
from .vm import VM, BulletSpawn
from .bullet_trace import load_traces, Bullet

_ECL = Path(__file__).resolve().parents[2] / "tools" / "th07_ecl" / "ecldata1.ecl"
_FIGHTS = Path(__file__).resolve().parents[2] / "sim" / "fights"

# cost gates — a candidate outside any of these can't be the source
GATE_FRAME = 150      # |dframe - phase offset|
GATE_POS = 20.0       # px between (corrected) spawn point and first recorded position
GATE_ANG = np.deg2rad(30)


@dataclass
class Match:
    bullet_id: int
    source_sub: int
    source_ip: int
    d_frame: float        # recorded_birth - (vm_frame + phase_offset)
    d_pos: float
    d_ang: float
    spawn_speed: float = 0.0
    spawn_angle: float = 0.0
    ring: bool = False     # matched ring-by-ring (burst pattern) rather than 1:1;
    #                        instruction tag + frame + heading are solid, but the
    #                        VM's emitter *position* may be off (a Part 8 gap), so
    #                        d_pos is not a match-quality signal for these


@lru_cache(maxsize=4)
def run_vm(ecl_path: str = str(_ECL), seed: int = 0, frames: int = 13000
           ) -> tuple[BulletSpawn, ...]:
    ecl = parse_file(ecl_path)
    vm = VM(ecl, difficulty=3, seed=seed)
    vm.start_boss(sub=31, interrupt=0)
    vm.run(frames)
    return tuple(vm.bullets)


def _eff_sig(spawn) -> tuple:
    """(fx_flag, fx_p1, fx_interval) of a spawn's first non-launch effect entry
    that its type-word actually gates on — matches the recorder's fx columns."""
    btype = int(spawn.btype)
    for p1, p2, iv, rep, flag, gate in spawn.effects:
        if flag != 1 and (btype & flag):
            return (flag, round(p1, 4), iv)
    return (0, 0.0, 0)


def _bul_sig(b: Bullet) -> tuple:
    """(fx_flag, fx_p1, fx_interval) of the recorded bullet's *armed* effect —
    the staging entry only applies if the type-word gates it (52-col
    recordings); older ones fall back to the raw fx columns."""
    m = b.motion[0]
    if b.re is not None:
        tflag = int(b.re[0, 1])
        for e in b.staging(0):
            if e["flag"] != 1 and (tflag & e["flag"]):
                return (e["flag"], round(e["p1"], 4), e["interval"])
        return (0, 0.0, 0)
    return (int(b.fxflag), round(float(m[4]), 4), int(m[6]))


def _vm_phases() -> list[int]:
    """VM major-phase start frames + a trailing sentinel."""
    ecl = parse_file(str(_ECL))
    vm = VM(ecl, difficulty=3, seed=0)
    vm.start_boss(sub=31, interrupt=0)
    vm.run(13000)
    starts = sorted({0, *(f for f, _s in vm.phase_transitions())})
    return starts + [13000]


def _xcorr_offset(vm_f: np.ndarray, rec_f: np.ndarray, lo=0, hi=320) -> int:
    """Frame shift `o` (rec is `o` frames behind the VM) that best lines up the
    two birth-rate histograms."""
    if len(vm_f) == 0 or len(rec_f) == 0:
        return 60
    bins = np.arange(min(vm_f.min(), rec_f.min()) - 200,
                     max(vm_f.max(), rec_f.max()) + 400, 10.0)
    vh = np.histogram(vm_f, bins)[0].astype(float)
    best, best_o = -1e18, 60
    for o in range(lo, hi + 1, 5):
        rh = np.histogram(rec_f - o, bins)[0].astype(float)
        s = float(np.dot(vh, rh))
        if s > best:
            best, best_o = s, o
    return best_o


def _match_group(trs: list[Bullet], cand: list[tuple[int, BulletSpawn]],
                 off0: int) -> tuple[list[tuple], int, tuple]:
    """Greedy-match one (phase, signature) group. Three passes: pass 0 fixes the
    frame offset, pass 1 fixes a rigid spawn-point shift (absorbs the ~8 px
    firing standoff and a stationary emitter placed at a constant wrong spot —
    not an orbiting one), pass 2 commits. Returns
    (matches, frame_offset, (shift_x, shift_y)); each match is
    (bullet_id, spawn_index, source_sub, source_ip, df, dp, da)."""
    sp_f = np.array([s.frame for _, s in cand], float)
    sp_xy = np.array([(s.x, s.y) for _, s in cand], float)
    sp_a = np.array([s.angle for _, s in cand], float)
    off = off0
    shift = np.zeros(2)
    out: list[tuple] = []
    trs_sorted = sorted(trs, key=lambda z: z.birth_frame)
    for _pass in range(3):
        taken = np.zeros(len(cand), bool)
        out = []
        for t in trs_sorted:
            d0 = t.xy[1] - t.xy[0] if t.life > 1 else t.vel[0]
            ta = float(np.arctan2(d0[1], d0[0]))
            df = np.abs(t.birth_frame - (sp_f + off))
            dp = np.hypot(sp_xy[:, 0] + shift[0] - t.xy[0, 0],
                          sp_xy[:, 1] + shift[1] - t.xy[0, 1])
            da = np.abs(np.arctan2(np.sin(sp_a - ta), np.cos(sp_a - ta)))
            gate_p = GATE_POS if _pass == 2 else GATE_POS * 4   # loose while fitting
            ok = (~taken) & (df <= GATE_FRAME) & (dp <= gate_p) & (da <= GATE_ANG)
            if not ok.any():
                continue
            cost = np.where(ok, df / GATE_FRAME + dp / GATE_POS + da / GATE_ANG, 1e18)
            k = int(np.argmin(cost))
            taken[k] = True
            out.append((t, k, cand[k][0], cand[k][1].source_sub, cand[k][1].source_ip,
                        float(t.birth_frame - (sp_f[k] + off))))
        if not out:
            break
        if _pass == 0:
            off += int(np.median([m[5] for m in out]))
        elif _pass == 1:
            dxy = np.array([[m[0].xy[0, 0] - sp_xy[m[1], 0],
                             m[0].xy[0, 1] - sp_xy[m[1], 1]] for m in out])
            shift = np.median(dxy, 0)
    res = []
    for t, k, j, ssub, sip, dfr in out:
        dp = float(np.hypot(sp_xy[k, 0] + shift[0] - t.xy[0, 0],
                            sp_xy[k, 1] + shift[1] - t.xy[0, 1]))
        d0 = t.xy[1] - t.xy[0] if t.life > 1 else t.vel[0]
        ta = float(np.arctan2(d0[1], d0[0]))
        da = float(abs(np.arctan2(np.sin(sp_a[k] - ta), np.cos(sp_a[k] - ta))))
        res.append((t.id, j, ssub, sip, dfr, dp, da))
    return res, off, (float(shift[0]), float(shift[1]))


def _rings(items: list, frame_of, gap: int = 4) -> list[list]:
    """Split a list into bursts: sort by frame, cut where the frame jumps > gap."""
    if not items:
        return []
    order = sorted(items, key=frame_of)
    out, cur = [], [order[0]]
    for it in order[1:]:
        if frame_of(it) - frame_of(cur[-1]) > gap:
            out.append(cur)
            cur = [it]
        else:
            cur.append(it)
    out.append(cur)
    return out


def _match_group_rings(trs: list[Bullet], cand: list[tuple[int, BulletSpawn]],
                       off0: int) -> list[tuple]:
    """Ring-level match for burst patterns (Table-Turning `bullet_circle`): the
    per-bullet frame/angle is ambiguous inside a 5-30-bullet ring fired in one
    frame, but the *sequence of rings* lines up ~1:1. DP-align the two ring
    lists on (|Δframe - offset|, ring-size difference) allowing skips, then
    inside each matched pair assign recorded traces to VM spawns by nearest
    launch heading. Returns (bullet_id, spawn_index, sub, ip, dframe, dpos, dang)."""
    v_rings = _rings(cand, lambda c: c[1].frame)
    r_rings = _rings(trs, lambda t: t.birth_frame)
    if len(v_rings) < 3 or len(r_rings) < 3:
        return []

    vf = [np.mean([c[1].frame for c in r]) for r in v_rings]
    rf = [np.mean([t.birth_frame for t in r]) for r in r_rings]

    nV, nR = len(v_rings), len(r_rings)
    NEG = 1e9
    dp = np.full((nV + 1, nR + 1), NEG)
    dp[0, 0] = 0.0
    bt = np.zeros((nV + 1, nR + 1), np.int8)          # 0 diag, 1 skip-V, 2 skip-R
    SKIP = 6.0
    for i in range(nV + 1):
        for j in range(nR + 1):
            if i < nV and dp[i, j] - SKIP > dp[i + 1, j]:
                dp[i + 1, j], bt[i + 1, j] = dp[i, j] - SKIP, 1
            if j < nR and dp[i, j] - SKIP > dp[i, j + 1]:
                dp[i, j + 1], bt[i, j + 1] = dp[i, j] - SKIP, 2
            if i < nV and j < nR:
                df = abs((vf[i] + off0) - rf[j])
                dc = abs(len(v_rings[i]) - len(r_rings[j]))
                score = dp[i, j] + 10.0 - df / 3.0 - dc * 0.5
                if score > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1], bt[i + 1, j + 1] = score, 0

    pairs: list[tuple[int, int]] = []
    i, j = nV, nR
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif move == 1 and i > 0:
            i -= 1
        elif move == 2 and j > 0:
            j -= 1
        else:
            break
    pairs.reverse()

    out: list[tuple] = []
    for vi, ri in pairs:
        vr, rr = v_rings[vi], r_rings[ri]
        va = np.array([s.angle for _, s in vr])
        taken = np.zeros(len(vr), bool)
        for t in rr:
            d0 = t.xy[1] - t.xy[0] if t.life > 1 else t.vel[0]
            ta = float(np.arctan2(d0[1], d0[0]))
            da = np.abs(np.arctan2(np.sin(va - ta), np.cos(va - ta)))
            da[taken] = 1e9
            k = int(np.argmin(da))
            if da[k] > np.deg2rad(35):
                continue
            taken[k] = True
            j_sp, s = vr[k]
            dpx = float(np.hypot(s.x - t.xy[0, 0], s.y - t.xy[0, 1]))
            out.append((t.id, j_sp, s.source_sub, s.source_ip,
                        float(t.birth_frame - (s.frame + off0)), dpx, float(da[k])))
    return out


def align(npz_path: str | Path) -> tuple[list[Match], list[int], list[BulletSpawn]]:
    """Returns (matches, unmatched_bullet_ids, unmatched_vm_spawns)."""
    traces = load_traces(npz_path)
    spawns = list(run_vm())
    vph = _vm_phases()
    births = np.array([b.birth_frame for b in traces])

    # per VM phase: its frame offset, then assign each recorded bullet to the
    # phase whose de-shifted window contains its birth
    offs: list[int] = []
    for i in range(len(vph) - 1):
        v0, v1 = vph[i], vph[i + 1]
        sp_f = np.array([s.frame for s in spawns if v0 <= s.frame < v1])
        win = births[(births >= v0 - 50) & (births < v1 + 400)]
        offs.append(_xcorr_offset(sp_f, win))

    phase_of: dict[int, int] = {}
    for t in traces:
        best = None
        for i in range(len(vph) - 1):
            if vph[i] <= t.birth_frame - offs[i] < vph[i + 1]:
                r = abs(t.birth_frame - offs[i]
                        - np.clip(t.birth_frame - offs[i], vph[i], vph[i + 1]))
                if best is None or r < best[1]:
                    best = (i, r)
        if best is None:                       # outside every window — nearest start
            d = [abs(t.birth_frame - offs[i] - vph[i]) for i in range(len(vph) - 1)]
            best = (int(np.argmin(d)), 0)
        phase_of[t.id] = best[0]

    matches: list[Match] = []
    matched_bul: set[int] = set()
    used_spawn: set[int] = set()

    for i in range(len(vph) - 1):
        v0, v1 = vph[i], vph[i + 1]
        ph_tr = [t for t in traces if phase_of[t.id] == i]
        ph_sp = [(j, s) for j, s in enumerate(spawns) if v0 <= s.frame < v1]
        if not ph_tr or not ph_sp:
            continue

        by_sig_sp: dict[tuple, list] = {}
        for j, s in ph_sp:
            by_sig_sp.setdefault(_eff_sig(s), []).append((j, s))
        by_sig_tr: dict[tuple, list] = {}
        for t in ph_tr:
            by_sig_tr.setdefault(_bul_sig(t), []).append(t)

        for sig, trs in by_sig_tr.items():
            cand = by_sig_sp.get(sig, [])
            if not cand:
                continue
            # ring-structured groups (bullet_circle bursts) match ring-by-ring;
            # everything else bullet-by-bullet
            per_frame = np.bincount(
                np.unique([s.frame for _, s in cand], return_inverse=True)[1])
            is_ring = False
            if len(cand) >= 60 and float(np.median(per_frame)) >= 4:
                got = _match_group_rings(trs, cand, offs[i])
                if len(got) < 0.3 * min(len(trs), len(cand)):
                    got, _o, _s = _match_group(trs, cand, offs[i])
                else:
                    is_ring = True
            else:
                got, _off, _shift = _match_group(trs, cand, offs[i])
            for bid, j, ssub, sip, df, dp, da in got:
                if bid in matched_bul or j in used_spawn:
                    continue
                matched_bul.add(bid)
                used_spawn.add(j)
                matches.append(Match(bid, ssub, sip, df, dp, da,
                                     spawns[j].speed, spawns[j].angle, ring=is_ring))

    unmatched_bul = [t.id for t in traces if t.id not in matched_bul]
    unmatched_sp = [s for j, s in enumerate(spawns) if j not in used_spawn]
    return matches, unmatched_bul, unmatched_sp


# --------------------------------------------------------------------------
def _fmt_pct(x):
    return f"{100 * x:5.1f}%"


def verify(npz_paths: list[str]) -> bool:
    from collections import Counter
    ok = True
    p = npz_paths[0]
    traces = {t.id: t for t in load_traces(p)}
    matches, un_b, un_s = align(p)
    n = len(traces)

    by_src: dict[tuple, list] = {}
    for m in matches:
        by_src.setdefault((m.source_sub, m.source_ip), []).append(m)
    total_spawn = Counter((s.source_sub, s.source_ip) for s in run_vm())

    print(f"{Path(p).name}: {len(matches)}/{n} bullets tagged "
          f"({_fmt_pct(len(matches) / n)})\n")
    print("  per source instruction:")
    big_purities = []
    for src in sorted(by_src, key=lambda s: -len(by_src[s])):
        ms = by_src[src]
        bs = [traces[m.bullet_id] for m in ms]
        key = Counter((b.cls, b.fxflag) for b in bs)
        purity = key.most_common(1)[0][1] / len(bs)
        dp = np.array([m.d_pos for m in ms])
        dfr = np.array([m.d_frame for m in ms])
        tag = "ring" if ms[0].ring else "1:1 "
        print(f"    Sub{src[0]}@{src[1]:<3} {len(ms):5}/{total_spawn[src]:<5} {tag}  "
              f"purity {_fmt_pct(purity)}  dpos p50 {np.median(dp):6.1f}  "
              f"dframe std {dfr.std():4.0f}")
        if len(ms) >= 100:
            big_purities.append(purity)

    dfr_all = np.array([m.d_frame for m in matches])
    _dp11 = [m.d_pos for m in matches if not m.ring]
    dp11 = np.array(_dp11) if _dp11 else np.zeros(1)
    print(f"\n  tagged {_fmt_pct(len(matches) / n)} of bullets   "
          f"frame-lock median {np.median(dfr_all):+.0f} std {dfr_all.std():.0f}")
    print(f"  1:1 matches: dpos p50 {np.median(dp11):.1f}px    "
          f"ring matches: instruction+frame+heading locked, emitter position "
          f"follows the RNG-driven boss drift")

    # PASS = we tag a real fraction, the per-instruction tags are pure (Sub36
    #        legitimately fires several bullet types on an RNG branch, a low
    #        outlier), the 1:1 matches sit on their emitter, and frames lock.
    ok = (len(matches) / n >= 0.60
          and float(np.median(dp11)) < 15.0
          and len(big_purities) >= 5
          and float(np.median(big_purities)) >= 0.95
          and float(dfr_all.std()) < 60.0)
    return ok


def refit_coverage(npz_paths: list[str]) -> None:
    """The payoff, on the sources the VM currently places correctly: group
    Part 10 by (source_sub, source_ip) and re-measure the motion-model fit."""
    from .fit_motion import _disp
    H = 90
    errs_src: list[float] = []
    for p in npz_paths:
        traces = {t.id: t for t in load_traces(p)}
        matches, un_b, _ = align(p)
        # a group is usable if its tags are pure — a 1:1 match on its emitter, or
        # a ring match (instruction + heading solid, only the VM emitter position
        # is off, and the profile fit is in the bullet's own frame anyway)
        by_src: dict[tuple, list] = {}
        for m in matches:
            by_src.setdefault((m.source_sub, m.source_ip), []).append(m)
        groups: dict[tuple, list[Bullet]] = {}
        for src, ms in by_src.items():
            bs = [traces[m.bullet_id] for m in ms]
            from collections import Counter as _C
            pur = _C((b.cls, b.fxflag) for b in bs).most_common(1)[0][1] / len(bs)
            if pur < 0.9:
                continue
            for m in ms:
                if m.ring or m.d_pos < 12:
                    groups.setdefault(
                        (m.source_sub, m.source_ip, round(m.spawn_speed, 1)), []
                    ).append(traces[m.bullet_id])
        for src, bs in groups.items():
            bs = [b for b in bs if b.life >= 30]
            if len(bs) < 20:
                continue
            T = 200
            DM = np.full((len(bs), T), np.nan)
            DA = np.full((len(bs), T), np.nan)
            for i, b in enumerate(bs):
                d = _disp(b, T)
                nn = len(d)
                DM[i, :nn] = np.hypot(d[:, 0], d[:, 1])
                a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
                DA[i, :nn] = a - a[0]
            with warnings.catch_warnings():          # empty tail columns -> all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                mp, dp = np.nanmedian(DM, 0), np.nanmedian(DA, 0)
            for prof in (mp, dp):
                last = np.where(~np.isnan(prof))[0]
                if len(last):
                    prof[last[-1] + 1:] = prof[last[-1]]
                prof[np.isnan(prof)] = 0.0
            for b in bs:
                nn = min(b.life - 1, H)
                if nn < 10:
                    continue
                d0 = _disp(b, 1)[0]
                a0 = np.arctan2(d0[1], d0[0])
                idx = np.minimum(np.arange(nn), T - 1)
                h = a0 + dp[idx]
                step = np.stack([np.cos(h) * mp[idx], np.sin(h) * mp[idx]], -1)
                pred = b.xy[0].astype(np.float64) + np.cumsum(
                    np.vstack([[0.0, 0.0], step]), 0)[:nn + 1]
                e = np.hypot(*(pred - b.xy[:nn + 1].astype(np.float64)).T).max()
                errs_src.append(e)
    errs_src = np.array(errs_src) if errs_src else np.zeros(1)
    print(f"\n  Part 10 re-fit on instruction-pure matches, grouped by "
          f"(source_sub, source_ip, spawn_speed):")
    print(f"    {len(errs_src)} bullets   within 5px/90f: "
          f"{_fmt_pct(np.mean(errs_src <= 5.0))}   "
          f"p50 {np.median(errs_src):.2f}  p90 {np.percentile(errs_src, 90):.2f} px")
    print(f"    (per-instruction grouping is right where the VM's geometry is; "
          f"p90 tail = the mid-flight redirects, still to model)")


def main(argv):
    import glob
    paths = argv[1:] or sorted(glob.glob(str(_FIGHTS / "letty_*.npz")))
    ok = verify(paths)
    refit_coverage(paths)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
