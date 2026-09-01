"""Part 2 verification: control flow.

    python -m sim.ecl.vm_verify [tools/th07_ecl/ecldata1.ecl]

1. Synthetic subs exercise jump / conditional jump / call+ret / jump_dec loops.
2. Letty's real ECL runs end to end: the phase machine must walk
   NS1 -> Lingering Cold -> NS2 -> Table-Turning -> defeat, pick the Lunatic
   spellcard variants, and land each transition within ~1 s of the screen-clears
   the recordings show (`sim/boss_phases._BOUNDARIES["letty"]`).

Exit 0 iff everything holds.
"""
from __future__ import annotations

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

    # call / ret: args pass through a shared gvar (10029), I0..I7 are snapshotted
    ecl = _synthetic_ecl({
        0: [
            _instr(0, 4, __V(10029), 5),        # PARAM_A = 5   (shared)
            _instr(0, 41, 1),                    # call Sub1
            _instr(0, 4, __V(10001), __V(10030)),  # I1 = PARAM_B  (Sub1's "return")
        ],
        1: [
            _instr(0, 4, __V(10030), __V(10029)),  # PARAM_B = PARAM_A
            _instr(0, 4, __V(10000), 99),          # I0 = 99  (local; discarded on ret)
            _instr(0, 42),
        ],
    })
    e = _run_sub(ecl, 0, 2)
    ok &= _check("call / ret (shared args, snapshotted locals)",
                 e.ivars[1] == 5 and e.ivars[0] == 0,
                 f"I1={e.ivars[1]} I0={e.ivars[0]}")

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
    missed = control_ops & set(vm.unhandled)
    ok &= _check("no control-flow opcode unhandled", not missed, f"{missed}")
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
    print(f"\nLetty phase machine ({path.name}):")
    ok &= test_letty(path)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
