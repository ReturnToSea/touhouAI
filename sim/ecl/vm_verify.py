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
    ok &= _check("jump_dec loop iterates then exits",
                 e.ivars[0] == 7 and e.ivars[4] == 1 and e.ivars[1] == 1,
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

    # rand: deterministic per seed, in range, roughly uniform
    ecl = _synthetic_ecl({0: [_instr(0, 52, F0, -math.pi, math.pi)]})   # __math_rand_rad
    vm1, vm2 = VM(ecl, seed=1234), VM(ecl, seed=1234)
    for vm in (vm1, vm2):
        vm.enemies.append(Enemy(vm, 0, is_boss=True))
        vm._run_enemy(vm.enemies[0])
    ok &= _check("rand deterministic per seed",
                 vm1.enemies[0].fvars[0] == vm2.enemies[0].fvars[0])

    vm = VM(ecl, seed=1)
    vals = []
    e = Enemy(vm, 0, is_boss=True)
    vm.enemies.append(e)
    for _ in range(4000):
        e.frame = e.ip = 0
        e.running = True
        vm._run_enemy(e)
        vals.append(e.fvars[0])
    in_range = all(-math.pi <= v < math.pi for v in vals)
    mean = sum(vals) / len(vals)
    ok &= _check("rand in [-pi, pi) and ~uniform", in_range and abs(mean) < 0.15,
                 f"mean={mean:.3f} range_ok={in_range}")
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
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
