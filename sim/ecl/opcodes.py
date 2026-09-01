"""TH07 ECL instruction-parameter signatures.

Each signature is a string, one char per *logical* parameter (the unit thtk
prints and the unit `param_mask` bits index):

    i  int32                     f  float32
    h  int16 signed              H  int16 unsigned
       (two int16s pack into one 4-byte slot; a var-ref stores the gvar id there)
    o  int32 byte-offset of a jump target  (resolved to an instr index by the parser)
    s  shift-jis string, consumes the rest of the instruction data
    x  one padding byte, no parameter

Derived by cross-referencing thtk's `-r` decompile of all six stage ECLs against
the binaries (`scratchpad/infer_sigs.py`), then hand-correcting the packed-short
opcodes (thtk shows the two shorts as separate args; the raw data has one slot).
Opcodes absent here are parsed as raw int32 slots.
"""
from __future__ import annotations

# rank_mask bits — difficulty gate (th07.eclm !difficulty_flags)
RANK_EASY = 0x01
RANK_NORMAL = 0x02
RANK_HARD = 0x04
RANK_LUNATIC = 0x08
RANK_ALL = 0xFF
RANK_NAMES = {RANK_EASY: "E", RANK_NORMAL: "N", RANK_HARD: "H", RANK_LUNATIC: "L"}

# time == -1 or opcode == 0xffff terminates a sub / timeline
SUB_END_TIME = -1
END_OPCODE = 0xFFFF

SIGS: dict[int, str] = {
    0: "", 1: "",
    2: "io", 3: "ioi",
    4: "ii", 5: "ff", 6: "ii", 7: "iii", 8: "ff", 9: "fff",
    10: "ii", 11: "ff",
    12: "iii", 13: "iii", 14: "iii", 15: "iii", 16: "iii", 17: "i", 18: "i",
    19: "fff", 20: "fff", 21: "fff", 22: "fff",
    24: "ff", 25: "ff", 26: "fffff", 27: "fiiiffff",
    28: "iiio", 30: "iiio", 31: "ffio", 32: "iiio", 33: "ffio",
    34: "iiio", 35: "ffio", 36: "iiio", 37: "ffio", 38: "iiio", 39: "ffio",
    40: "f", 41: "i", 42: "",
    43: "iii", 44: "ffi", 45: "i", 46: "fff", 47: "ffi", 48: "f", 49: "i",
    50: "f", 52: "fff",
    54: "iiff", 55: "iifff", 56: "ifffffff", 57: "ff", 59: "i",
    62: "ffff", 63: "",
    # bullet_* — two int16 (count, type) packed into slot 0
    64: "hhiiffffi", 65: "hhiiffffi", 66: "hhiiffffi", 67: "hhiiffffi",
    68: "hhiiffffi", 69: "hhiiffffi", 70: "hhiiffffi", 71: "hhiiffffi",
    72: "hhiiffffi",
    73: "i", 74: "i", 75: "", 76: "", 78: "fff", 79: "iiiiiff", 80: "",
    81: "ii",
    # laser_create[_aimed] — two int16 packed into slot 0
    82: "hhffffffiiiiii", 83: "hhffffffiiiiii",
    84: "i", 85: "if", 87: "ifff", 89: "i",
    90: "hhs", 91: "",
    92: "ifffiii", 93: "ifffiii", 94: "",
    95: "i", 96: "hhhhHxx", 97: "ii", 98: "i", 99: "i",
    100: "iffff", 101: "fff", 102: "i", 103: "i", 104: "i", 105: "i",
    106: "i", 107: "i", 108: "ii",
    110: "i", 111: "i", 112: "i", 113: "i", 114: "i", 115: "i", 116: "i",
    117: "iii", 118: "iiifff", 119: "i", 120: "i", 121: "ii", 122: "ii",
    124: "i", 125: "i", 126: "i", 128: "i",
    131: "ffiiii", 132: "i", 133: "", 135: "i", 136: "i", 137: "i",
    138: "iiii", 139: "iiii", 140: "ffff", 141: "i", 142: "i", 143: "f",
    144: "ii", 145: "ii", 146: "", 148: "iii", 149: "iffi", 150: "f",
    151: "ffff", 152: "if", 153: "fff", 154: "i", 155: "f", 156: "ii",
    157: "if", 158: "iff", 159: "ffff", 160: "i", 161: "i",
}

# logical-param index of the jump-offset arg, per opcode (derived from SIGS)
JUMP_PARAM: dict[int, int] = {op: sig.index("o") for op, sig in SIGS.items() if "o" in sig}

# timeline instruction signatures (th07.eclm !timeline_ins_names).
# header already consumes time(h) + arg0(h); these cover the remaining data.
TIMELINE_SIGS: dict[int, str] = {
    0: "fffhhi", 1: "fffhhi", 2: "fffhhi", 3: "fffhhi",
    4: "fffhhi", 5: "fffhhi", 6: "fffhhi", 7: "fffhhi",
    8: "", 9: "i", 10: "iii", 11: "i", 12: "i",
}


def rank_str(mask: int) -> str:
    if mask & RANK_ALL == RANK_ALL:
        return "*"
    return "".join(n for bit, n in RANK_NAMES.items() if mask & bit) or f"0x{mask:02x}"
