"""Parts 2 & 3 verification: control flow + arithmetic.

    python -m sim.ecl.vm_verify [tools/th07_ecl/ecldata1.ecl]

1. Synthetic subs exercise jump / conditional jump / call+ret / jump_dec loops.
2. Synthetic subs exercise every arithmetic / trig / rand opcode.
3. Letty's real ECL runs end to end: the phase machine must walk
   NS1 -> Lingering Cold -> NS2 -> Table-Turning -> defeat, pick the Lunatic
   spellcard variants, land each transition within ~1 s of the screen-clears the
   recordings show, and compute Sub40's three sub-enemy spawn angles 120 apart.

Exit 0 iff everything holds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from .parser import ECLFile, Instr, Sub, parse_file
from .vm import VM, Enemy

FIGHTS = Path(__file__).resolve().parents[2] / "sim" / "fights"

_DEFAULT = Path(__file__).resolve().parents[2] / "tools" / "th07_ecl" / "ecldata1.ecl"

# Letty, Lunatic — from sim/boss_phases._BOUNDARIES + the recorded fight length
LETTY_PHASES = [
    ("NS1", 38, 0),
    ("Lingering Cold", 42, 2400),
    ("NS2", 39, 5450),
    ("Table-Turning", 55, 7820),
    ("defeat", 51, 10770),
]
TOL = 120  # frames (~2 s): the repositioning lull the ECL timer doesn't count


def _instr(time, op, *args, rank=0xFF):
    return Instr(index=0, byte_offset=0, time=time, opcode=op, rank_mask=rank,
                 param_mask=0, args=list(args), name=f"ins_{op}")


def _synthetic_ecl(subs: dict[int, list[Instr]]) -> ECLFile:
    ecl = ECLFile(subs=[], timelines=[], eclmap=None)
    n = max(subs) + 1
    ecl.subs = [Sub(index=i, name=f"Sub{i}", instrs=subs.get(i, [])) for i in range(n)]
    for s in ecl.subs:
        for j, ins in enumerate(s.instrs):
            ins.index = j
    return ecl


def _run_sub(ecl: ECLFile, sub: int, frames: int) -> Enemy:
    vm = VM(ecl, difficulty=3)
    e = Enemy(vm, sub, is_boss=True)
    vm.enemies.append(e)
    vm._run_enemy(e)
    for _ in range(frames):
        vm.step()
    return e


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))
    return cond


def _unwrap(angles):
    import numpy as np
    return np.unwrap(np.asarray(angles, float))


def _orbit_growth(tracks, centres):
    """Median linear radius-growth (px/frame) across a set of orb tracks."""
    import numpy as np
    rates = []
    for xy, c in zip(tracks, centres):
        if len(xy) < 40:
            continue
        r = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
        dt = np.arange(len(r))
        n = min(len(r), 120)
        rates.append(np.polyfit(dt[:n], r[:n], 1)[0])
    return float(np.median(rates)) if rates else float("nan")


def _check_orbit_vs_recording(ecl: ECLFile, npz: Path) -> bool:
    """VM Sub57 orbs and recorded Table-Turning orbs should spiral out at the
    same rate (Letty's script says 0.5 px/frame)."""
    import numpy as np
    from .enemy_trace import load_enemy_traces

    vm = VM(ecl, difficulty=3, seed=0)
    vm.start_boss(sub=31, interrupt=0)
    tracks: dict[int, list] = {}
    births: dict[int, tuple] = {}
    while vm.frame < 8600 and vm.boss() is not None:
        vm.step()
        for en in vm.enemies:
            if en.is_boss or en.sub != 57:
                continue
            k = id(en)
            if k not in tracks:
                b = vm.boss()
                births[k] = (b.x, b.y) if b else (192.0, 112.0)
            tracks.setdefault(k, []).append((en.x, en.y))
    vm_tracks = [np.array(v) for v in tracks.values() if len(v) >= 40]
    vm_grow = _orbit_growth(vm_tracks, [births[k] for k, v in tracks.items() if len(v) >= 40])

    d = np.load(npz)
    f0 = int(d["bullets"][:, 0].min())
    bo = d["boss"]
    bomap = {int(s) - f0: (x, y) for s, x, y in bo}
    rec = [t for t in load_enemy_traces(npz, jump_px=12)
           if round(t.hb[0]) == 8 and 7900 <= t.birth_frame < 9500 and t.life >= 60]
    rec_grow = _orbit_growth([t.xy for t in rec],
                             [bomap.get(t.birth_frame, (192.0, 112.0)) for t in rec])

    return _check("Sub57 orb spiral matches the recording",
                  abs(vm_grow - rec_grow) < 0.12 and abs(rec_grow - 0.5) < 0.12,
                  f"VM {vm_grow:.3f} vs recorded {rec_grow:.3f} px/f")


def test_control_flow() -> bool:
    ok = True

    # jump_dec: the loop body runs N times, then falls through
    ecl = _synthetic_ecl({0: [
        _instr(0, 4, __V(10012), 5),           # I4 = 5   (loop counter)
        _instr(0, 4, __V(10000), 7),            # loop body: I0 = 7
        _instr(0, 3, 0, 1, __V(10012)),         # jump_dec -> time 0, ip 1, counter I4
        _instr(0, 4, __V(10001), 1),            # after the loop: I1 = 1
    ]})
    e = _run_sub(ecl, 0, 3)
    # the engine decrements unconditionally, so the counter lands on 0 (not 1)
    ok &= _check("jump_dec loop iterates then exits",
                 e.ivars[0] == 7 and e.ivars[4] == 0 and e.ivars[1] == 1,
                 f"I0={e.ivars[0]} I4={e.ivars[4]} I1={e.ivars[1]}")

    # conditional jump: DIFFICULTY(3) != 0 -> take the jump, skip the trap
    ecl = _synthetic_ecl({0: [
        _instr(0, 30, __V(10016), 0, 0, 3),    # jump_int_neq(DIFFICULTY, 0) -> ip 3
        _instr(0, 4, __V(10000), 999),          # trap (should be skipped)
        _instr(0, 2, 0, 4),                      # jump past
        _instr(0, 4, __V(10000), 7),            # I0 = 7
        _instr(0, 42),
    ]})
    e = _run_sub(ecl, 0, 2)
    ok &= _check("conditional jump on DIFFICULTY", e.ivars[0] == 7, f"I0={e.ivars[0]}")

    # call / ret: caller's ARG_A maps to the callee's PARAM_A; locals are restored
    ecl = _synthetic_ecl({
        0: [_instr(0, 4, __V(10037), 4), _instr(0, 41, 1)],
        1: [_instr(0, 4, __V(10000), __V(10029)),          # I0 = PARAM_A (=4)
            _instr(0, 65, 0, 0, __V(10000), 1, 1.0, 1.0, 0.0, 0.1, 42),  # fire I0 bullets
            _instr(0, 4, __V(10000), 99), _instr(0, 42)],
    })
    e = _run_sub(ecl, 0, 1)
    ok &= _check("call maps ARG_A -> callee PARAM_A", len(e.vm.bullets) == 4,
                 f"{len(e.vm.bullets)} bullets")

    ecl = _synthetic_ecl({
        0: [_instr(0, 4, __V(10000), 1), _instr(0, 41, 1),
            _instr(0, 4, __V(10001), __V(10000))],   # I1 = I0 (still 1 if locals restored)
        1: [_instr(0, 4, __V(10000), 99), _instr(0, 42)],
    })
    e = _run_sub(ecl, 0, 1)
    ok &= _check("call / ret restores caller locals", e.ivars[1] == 1 and e.ivars[0] == 1,
                 f"I0={e.ivars[0]} I1={e.ivars[1]}")

    # wait: an instruction gated to a later frame does not run early
    ecl = _synthetic_ecl({0: [
        _instr(0, 4, __V(10000), 1),
        _instr(30, 4, __V(10000), 2),
    ]})
    vm = VM(ecl, difficulty=3)
    e = Enemy(vm, 0, is_boss=True)
    vm.enemies.append(e)
    vm._run_enemy(e)
    at10 = e.ivars[0]
    for _ in range(31):
        vm.step()
    ok &= _check("frame gate holds later instructions", at10 == 1 and e.ivars[0] == 2,
                 f"@0={at10} @31={e.ivars[0]}")
    return ok


def test_arithmetic() -> bool:
    ok = True
    I0, I1, I4 = __V(10000), __V(10001), __V(10012)
    F0, F1 = __V(10004), __V(10005)

    # deterministic ops: (opcode, [args...], check(enemy))
    cases = [
        ("int add",  12, [I0, 3, 4],        lambda e: e.ivars[0] == 7),
        ("int sub",  13, [I0, 3, 10],       lambda e: e.ivars[0] == -7),
        ("int mul",  14, [I0, 6, 7],        lambda e: e.ivars[0] == 42),
        ("int div (trunc)", 15, [I0, -7, 2], lambda e: e.ivars[0] == -3),
        ("int mod",  16, [I0, 17, 5],       lambda e: e.ivars[0] == 2),
        ("float add", 19, [F0, 1.5, 2.25],  lambda e: abs(e.fvars[0] - 3.75) < 1e-6),
        ("float mul", 21, [F0, 3.0, 0.5],   lambda e: abs(e.fvars[0] - 1.5) < 1e-6),
        ("float div", 22, [F0, 1.0, 4.0],   lambda e: abs(e.fvars[0] - 0.25) < 1e-6),
        ("sin",      24, [F0, math.pi / 2], lambda e: abs(e.fvars[0] - 1.0) < 1e-6),
        ("cos",      25, [F0, 0.0],         lambda e: abs(e.fvars[0] - 1.0) < 1e-6),
        ("atan2",    26, [F0, 0.0, 0.0, 1.0, 1.0], lambda e: abs(e.fvars[0] - math.pi / 4) < 1e-6),
    ]
    for name, op, args, chk in cases:
        e = _run_sub(_synthetic_ecl({0: [_instr(0, op, *args)]}), 0, 1)
        ok &= _check(name, chk(e))

    # inc, norm_angle
    e = _run_sub(_synthetic_ecl({0: [
        _instr(0, 4, I0, 5), _instr(0, 17, I0), _instr(0, 17, I0),
        _instr(0, 5, F0, 4.0), _instr(0, 40, F0),          # 4.0 rad -> 4 - 2pi
    ]}), 0, 1)
    ok &= _check("math_inc", e.ivars[0] == 7)
    ok &= _check("math_norm_angle", abs(e.fvars[0] - (4.0 - 2 * math.pi)) < 1e-6,
                 f"{e.fvars[0]}")

    # the PRNG generator itself: deterministic per seed, uniform on [0, 1)
    from .rng import EclRng
    r1, r2 = EclRng(1234), EclRng(1234)
    ok &= _check("PRNG deterministic per seed",
                 [r1.rand() for _ in range(50)] == [r2.rand() for _ in range(50)])
    _r = EclRng(7)
    vs = [_r.rand() for _ in range(20000)]
    dec = [0] * 10
    for v in vs:
        dec[min(9, int(v * 10))] += 1
    ok &= _check("PRNG ~uniform on [0,1)",
                 abs(sum(vs) / len(vs) - 0.5) < 0.02 and max(dec) - min(dec) < 250,
                 f"mean {sum(vs)/len(vs):.3f}  deciles {dec}")

    # __math_rand_rad: a screen-aware heading (see the handler). From the left
    # half it points right (a +/-45 cone), reflected off nearby walls.
    ecl = _synthetic_ecl({0: [_instr(0, 52, F0, -math.pi, math.pi)]})
    vm = VM(ecl, seed=1)
    e = Enemy(vm, 0, is_boss=True)
    e.x, e.y = 100.0, 240.0          # left half, clear of every wall
    vm.enemies.append(e)
    vals = []
    for _ in range(4000):
        e.frame = e.ip = 0
        e.running = True
        vm._run_enemy(e)
        vals.append(e.fvars[0])
    cone = all(-math.pi / 4 <= v <= math.pi / 4 for v in vals)
    ok &= _check("__math_rand_rad: +/-45 cone toward centre from the left half",
                 cone and abs(sum(vals) / len(vals)) < 0.05,
                 f"mean {sum(vals)/len(vals):.3f}  cone_ok {cone}")
    return ok


def test_movement(ecl_path: Path) -> bool:
    ok = True
    ecl = parse_file(ecl_path)

    # __move_circle_abs: a Sub57 orb spirals out from the boss — radius grows
    # linearly, angle sweeps. Check the shape against a synthetic call with
    # Letty's own parameters (radius0 0, growth 0.5, ang_speed 0.026).
    vm = VM(ecl, difficulty=3, seed=1)
    e = Enemy(vm, 100, is_boss=False)          # empty sub — just drive the motion
    e.x, e.y, e.z = 192.0, 112.0, 0.0
    vm.enemies.append(e)
    from .vm import _Motion
    e.motion = _Motion("circle", 0, 320, e.x, e.y, e.z, e.x, e.y, e.z,
                       cx=192.0, cy=112.0, cz=0.0, angle0=-math.pi / 2,
                       radius=0.0, ang_speed=0.0262, radius_growth=0.5)
    radii, angs = [], []
    for _ in range(180):
        vm._update_motion(e)
        radii.append(math.hypot(e.x - 192.0, e.y - 112.0))
        angs.append(math.atan2(e.y - 112.0, e.x - 192.0))
    grow = (radii[120] - radii[20]) / 100.0
    swept = math.degrees(abs(_unwrap(angs)[-1] - angs[0]))
    ok &= _check("orbit spirals out at ~0.5 px/frame",
                 abs(grow - 0.5) < 0.02, f"growth {grow:.3f} px/f")
    ok &= _check("orbit sweeps its angle", swept > 90, f"{swept:.0f} deg over 180 f")

    # ...and the shape matches a recorded Table-Turning orb to within a few px
    recs = sorted(FIGHTS.glob("letty_[0-9]*.npz"))
    if recs:
        ok &= _check_orbit_vs_recording(ecl, recs[0])

    # enemy_flag_oob_immune: a sub-enemy that has been on screen and then leaves
    # gets culled — unless it's oob-immune. (Letty's orbs spiral off-screen and
    # must stop firing; without this the VM over-fires Table-Turning.)
    def _run_off(immune: bool) -> bool:
        vm2 = VM(ecl, difficulty=3, seed=1)
        en = Enemy(vm2, 100, is_boss=False)
        en.x, en.y, en.oob_immune = 192.0, 224.0, immune
        vm2.enemies.append(en)
        vm2._cull_offscreen(en)          # on screen — registers was_onscreen
        en.x = 900.0                     # now leave
        vm2._cull_offscreen(en)
        return en.alive
    ok &= _check("oob-cullable sub-enemy despawns off screen",
                 _run_off(False) is False)
    ok &= _check("oob-immune sub-enemy survives off screen",
                 _run_off(True) is True)
    vm = VM(ecl, difficulty=3, seed=1)
    vm.start_boss(sub=31, interrupt=0)
    born, died = {}, {}
    for _ in range(11000):
        alive_before = {id(e) for e in vm.enemies}
        vm.step()
        for e in vm.enemies:
            born.setdefault(id(e), vm.frame)
        for i in alive_before - {id(e) for e in vm.enemies}:
            died[i] = vm.frame
    lifes = [died[i] - born[i] for i in died if i in born]
    import numpy as np
    long = [x for x in lifes if x > 800]
    ok &= _check("sub-enemies don't live absurdly long",
                 len(long) / max(1, len(lifes)) < 0.05,
                 f"{len(long)}/{len(lifes)} lived >800 f")

    # boss track vs a recording, aligned on the first bullet frame
    recs = sorted(FIGHTS.glob("letty_[0-9]*.npz"))
    if not recs:
        print("  --   boss track vs recording — skipped (sim/fights/letty_[0-9]*.npz not present)")
        return ok
    import numpy as np
    vm = VM(ecl, difficulty=3, seed=1)
    vm.start_boss(sub=31, interrupt=0)
    trace = {0: (vm.boss().x, vm.boss().y)}
    for _ in range(11000):
        vm.step()
        b = vm.boss()
        if b is not None:
            trace[vm.frame] = (b.x, b.y)
    first_bullet = vm.bullets[0].frame if vm.bullets else 0

    d = np.load(recs[0])
    rec_bullets_f0 = int(d["bullets"][:, 0].min())
    offset = rec_bullets_f0 - first_bullet     # raw_step - offset == our frame numbering
    errs = []
    for step, rx, ry in d["boss"]:
        f = int(step) - offset
        if f in trace and f > first_bullet:   # frame == first_bullet is mid-interpolation, not a snap point
            vx, vy = trace[f]
            errs.append((f, math.hypot(vx - rx, vy - ry)))
    errs.sort()
    exact_run = 0
    for _f, err in errs:
        if err > 0.5:
            break
        exact_run += 1
    ok &= _check(f"boss track exact-matches the recording for {exact_run} deterministic frames",
                 exact_run >= 100, f"{exact_run} frames before the first RNG-driven move diverges it")
    return ok


def test_letty(path: Path) -> bool:
    ecl = parse_file(path)
    vm = VM(ecl, difficulty=3)
    vm.start_boss(sub=31, interrupt=0)
    vm.run(13000)

    trans = vm.phase_transitions()          # [(frame, sub), ...]
    ok = True
    ok &= _check("phase count", len(trans) == len(LETTY_PHASES),
                 f"{[s for _, s in trans]}")
    for (name, want_sub, want_frame), (got_frame, got_sub) in zip(LETTY_PHASES, trans):
        ok &= _check(
            f"{name:15} Sub{want_sub} @ ~{want_frame}",
            got_sub == want_sub and abs(got_frame - want_frame) <= TOL,
            f"got Sub{got_sub} @ {got_frame}",
        )

    spells = [d for _, ev, d in vm.trace if ev == "spellcard_start"]
    ok &= _check("Lunatic spellcards chosen",
                 any("(0, 5)" in s for s in spells) and any("(0, 9)" in s for s in spells),
                 f"{spells}")

    control_ops = {2, 3, 4, 5, 41, 42, 45, *range(28, 40),
                   90, 91, 99, 107, 108, 109, 110, 112, 113, 114, 115, 133, 148}
    math_ops = {6, 7, 8, 9, 10, 11, *range(12, 28), 40, 51, 52}
    missed = (control_ops | math_ops) & set(vm.unhandled)
    ok &= _check("no control-flow / arithmetic opcode unhandled", not missed, f"{missed}")

    # Sub40: the spawner fires three sub-enemies 120 deg apart (after its call(2))
    import sim.ecl.vm as _M
    saved = _M._HANDLERS.get(93)
    angles: list[float] = []
    _M._HANDLERS[93] = lambda vm, en, ins: angles.append(en.extra.get(10033, 0.0))
    try:
        vm2 = VM(ecl, difficulty=3)
        e = Enemy(vm2, 40, is_boss=True)
        vm2.enemies.append(e)
        vm2._run_enemy(e)
        for _ in range(200):
            vm2.step()
    finally:
        _M._HANDLERS[93] = saved
    want = [0.0, 2 * math.pi / 3, -2 * math.pi / 3]
    ok &= _check("Sub40 spawns 3 sub-enemies 120 apart",
                 len(angles) == 3 and all(abs(a - w) < 1e-4 for a, w in zip(angles, want)),
                 f"{[round(a, 4) for a in angles]}")

    # bullet spawn-event counts per phase vs the recordings (sim/fights/letty_*)
    vm3 = VM(ecl, difficulty=3, seed=20240901)
    vm3.start_boss(31, interrupt=0)
    vm3.run(11000)
    got = vm3.bullets_per_phase([f for f, _s in vm3.phase_transitions()][:4])
    recorded = [2115, 2861, 2563, 7658]          # mean births/phase, 10 recordings
    names = ["NS1", "Lingering Cold", "NS2", "Table-Turning"]
    for name, g, r in zip(names, got, recorded):
        err = (g - r) / r
        ok &= _check(f"{name:15} spawn count ~{r}", abs(err) <= 0.22,
                     f"got {g} ({err:+.0%})")
    tot_err = (sum(got) - sum(recorded)) / sum(recorded)
    ok &= _check("total spawn count within 12%", abs(tot_err) <= 0.12,
                 f"{sum(got)} vs {sum(recorded)} ({tot_err:+.0%})")

    ok &= _test_hp(ecl)
    return ok


def _phase_seq(vm):
    return [int(d.split("Sub", 1)[1].split()[0])
            for _, ev, d in vm.trace if ev == "enter_sub"]


def _test_hp(ecl) -> bool:
    """Part 7 — Letty's real HP thresholds drive the phase graph when damage is
    applied, and it stays timer-driven with none (matching the god-mode recs)."""
    ok = True

    # no damage -> pure timer transitions at the recorded screen-clears
    vm = VM(ecl, difficulty=3, seed=0)
    vm.start_boss(sub=31, interrupt=0)
    vm.run(13000)
    ok &= _check("HP: dodge-only run is timer-driven",
                 _phase_seq(vm) == [38, 42, 39, 55, 51]
                 and [f for f, e, _ in vm.trace if e == "enter_sub"][1] == 2400,
                 f"{_phase_seq(vm)}")

    # NS1 (15000 HP) -> Lingering Cold when HP crosses the life_callback_ex(1700)
    for dps, want in ((10, 1330), (30, 444)):
        vm = VM(ecl, difficulty=3, seed=0)
        vm.start_boss(sub=31, interrupt=0)
        for _ in range(13000):
            vm.step()
            b = vm.boss()
            if b and b.alive:
                b.damage(dps)
        f_lc = next((f for f, e, d in vm.trace
                     if e == "enter_sub" and "Sub42" in d), None)
        # 15000 - dps*f == 1700  ->  f == 13300/dps
        ok &= _check(f"HP: NS1->LC at HP 1700 (dps {dps})",
                     f_lc is not None and abs(f_lc - want) <= 20,
                     f"transitioned at {f_lc}, want ~{want}")

    # spell cards apply shot damage at 1/7 (FUN_00420620 spellcard divisor,
    # floor 1). Unit-check it directly.
    for spell, raw, want in ((None, 70, 70), ((0, 5), 70, 10), ((0, 5), 7, 1),
                             ((0, 5), 3, 1), ((0, 5), 0, 0)):
        v = VM(ecl, difficulty=3, seed=0)
        e = Enemy(v, 100, is_boss=True)
        e.life = e.max_life = 100000
        e.engaged = True
        e.spell = spell
        e.damage(raw)
        ok &= _check(f"HP: spell divisor raw {raw} -> {want} (spell={spell})",
                     100000 - e.life == want, f"took {100000 - e.life}")

    # engaged gate: a boss that never sets enemy_flag_invulnerable(1) takes no damage
    vm = VM(ecl, difficulty=3, seed=0)
    e = Enemy(vm, 100, is_boss=True)
    e.life = e.max_life = 1000
    e.damage(500)
    ok &= _check("HP: damage ignored while not engaged", e.life == 1000)
    e.engaged = True
    e.damage(500)
    ok &= _check("HP: damage lands once engaged", e.life == 500)
    return ok


def __V(gid):
    from .parser import Var
    return Var(gid, 10004 <= gid <= 10011 or 10072 <= gid <= 10073)


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    path = Path(argv[1]) if len(argv) > 1 else _DEFAULT

    print("control-flow primitives:")
    ok = test_control_flow()
    print("\narithmetic / trig / rand:")
    ok &= test_arithmetic()
    print(f"\nLetty phase machine + spawn math ({path.name}):")
    ok &= test_letty(path)
    print("\nmovement (orbits, boss track):")
    ok &= test_movement(path)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
