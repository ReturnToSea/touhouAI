"""TH07 (Perfect Cherry Blossom) ECL — the "old" TH06–09 bytecode format.

Part 1 of the ECL VM plan: a faithful binary parser. Reads `ecldataN.ecl` (as
extracted by `thdat -x 7 th07.dat`) into subs, timelines, and typed instructions
— straight from the binary, with no dependency on thtk's text decompile.

    from sim.ecl import parse_file
    ecl = parse_file("tools/th07_ecl/ecldata1.ecl")
    for ins in ecl.subs[39].instrs:          # Letty's non-spell
        print(ins.time, ins.name, ins.args)
"""
from .parser import ECLFile, Sub, Timeline, Instr, TimelineInstr, Var, parse_file, parse_bytes
from .eclmap import EclMap, load_map
from . import opcodes

__all__ = [
    "ECLFile", "Sub", "Timeline", "Instr", "TimelineInstr", "Var",
    "parse_file", "parse_bytes", "EclMap", "load_map", "opcodes",
]
