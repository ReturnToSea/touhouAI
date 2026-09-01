"""Pretty-print an ECL sub or timeline.

    python -m sim.ecl tools/th07_ecl/ecldata1.ecl            # list subs
    python -m sim.ecl tools/th07_ecl/ecldata1.ecl 39         # dump Sub39 (Letty non-spell)
    python -m sim.ecl tools/th07_ecl/ecldata1.ecl 42 --rank L
    python -m sim.ecl tools/th07_ecl/ecldata1.ecl --timeline 0
"""
from __future__ import annotations

import argparse
import sys

from .parser import Var, parse_file
from . import opcodes


def _fmt_arg(a) -> str:
    if isinstance(a, Var):
        return f"[{a.id}{'f' if a.is_float else ''}]"
    if isinstance(a, float):
        return f"{a:g}f"
    if isinstance(a, str):
        return f'"{a}"'
    return str(a)


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="python -m sim.ecl")
    ap.add_argument("ecl")
    ap.add_argument("sub", nargs="?", type=int, help="sub index to dump")
    ap.add_argument("--timeline", type=int, metavar="N", help="dump timeline N instead")
    ap.add_argument("--rank", metavar="E|N|H|L", help="only show lines that run at this difficulty")
    args = ap.parse_args(argv[1:])

    ecl = parse_file(args.ecl)
    rank_bit = {"E": 1, "N": 2, "H": 4, "L": 8}.get((args.rank or "").upper())

    if args.timeline is not None:
        tl = ecl.timelines[args.timeline]
        for ins in tl.instrs:
            a = ", ".join(_fmt_arg(x) for x in ins.args)
            print(f"  t={ins.time:<6} {ins.name}({ins.arg0}{', ' + a if a else ''})")
        return 0

    if args.sub is None:
        print(f"{args.ecl}: {len(ecl.subs)} subs, {len(ecl.timelines)} timelines")
        for s in ecl.subs:
            n = len(s.instrs)
            spell = next((i.args[-1] for i in s.instrs if i.opcode == 90), None)
            tag = f'  {spell}' if isinstance(spell, str) else ""
            print(f"  Sub{s.index:<3} {n:4} instr{tag}")
        return 0

    sub = ecl.subs[args.sub]
    last_t = None
    for ins in sub.instrs:
        if rank_bit is not None and not ins.runs_at(rank_bit):
            continue
        if ins.time != last_t:
            print(f"{ins.time}:")
            last_t = ins.time
        rk = "" if ins.rank == "*" else f"!{ins.rank} "
        a = ", ".join(_fmt_arg(x) for x in ins.args)
        print(f"    {rk}{ins.name}({a})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
