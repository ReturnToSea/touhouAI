"""Part 11 — align each recorded bullet trace to the VM spawn that produced it.

Part 10 hit a wall: bullets that share every *observable* feature
`(class, fx_flag, fx_p1, fx_interval, base_speed)` still come from different ECL
instructions with different scripted motion, and the recording can't tell them
apart. The VM can — every `BulletSpawn` carries `source_sub` + `source_ip`.

This module matches the two. Per VM major phase it cross-correlates the two
birth-rate histograms for the frame offset, then within each
`(fx_flag, fx_p1, fx_interval)` signature (which the recorder *does* observe)
greedily matches recorded births to VM spawns on
`(|dframe - offset|, dposition, dheading)` and tags each trace with its
`(source_sub, source_ip)`.

**Status: partial — blocked on Part 8.** Every one of Letty's bullets is fired
by an orbiting sub-enemy, and the VM's movement system freezes those
sub-enemies at their spawn point (`move_speed` / `set_angle` / the orbit-retarget
chain are stubbed). So only ~20 % of bullets — the ones whose emitter the VM
happens to place within ~12 px — get matched. On those the approach is proven:
frames stay locked (dframe std ~20 over a 10 000-frame fight), the tags are
instruction-pure, and a Part 10 re-fit grouped by `(source, spawn_speed)` lands
70 % within 5 px (vs 75 % from the observable key alone, but now *correctly*
grouped and extensible). The rest waits on Part 8 giving the sub-enemies their
real tracks.

    python -m sim.ecl.align            # align + diagnostic on sim/fights/letty_*.npz
"""
from __future__ import annotations

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
            got, _off, _shift = _match_group(trs, cand, offs[i])
            for bid, j, ssub, sip, df, dp, da in got:
                if bid in matched_bul or j in used_spawn:
                    continue
                matched_bul.add(bid)
                used_spawn.add(j)
                matches.append(Match(bid, ssub, sip, df, dp, da,
                                     spawns[j].speed, spawns[j].angle))

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
        print(f"    Sub{src[0]}@{src[1]:<3} {len(ms):5}/{total_spawn[src]:<5} spawns  "
              f"purity {_fmt_pct(purity)}  dpos p50 {np.median(dp):5.1f}  "
              f"dframe std {dfr.std():4.0f}")
        if len(ms) >= 100:
            big_purities.append(purity)

    dfr_all = np.array([m.d_frame for m in matches])
    dp_all = np.array([m.d_pos for m in matches])
    print(f"\n  tagged {_fmt_pct(len(matches) / n)} of bullets   "
          f"dpos p50 {np.median(dp_all):.1f}px   "
          f"frame-lock median {np.median(dfr_all):+.0f} std {dfr_all.std():.0f}")
    print("  (loose frame-lock on Table-Turning is inherent — its ~12f burst "
          "cadence makes bullet-level frame matching ambiguous; the tags are "
          "still instruction-pure, which is what Part 10 needs)")

    # PASS = tags are instruction-pure in the median (Sub36 legitimately fires
    #        several bullet types on an RNG branch, so it's a low outlier),
    #        positions are close, and we tag a real fraction
    ok = (len(matches) / n >= 0.30
          and float(np.median(dp_all)) < 15.0
          and len(big_purities) >= 5
          and float(np.median(big_purities)) >= 0.95)
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
        groups: dict[tuple, list[Bullet]] = {}
        for m in matches:
            if m.d_pos < 12:                       # stationary-emitter matches only
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
    print(f"\n  Part 10 re-fit on stationary-emitter matches, grouped by "
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
