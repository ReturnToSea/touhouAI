"""TH07 ECL binary parser.

File layout (little-endian), reverse-engineered from `ecldata1..8.ecl` and
cross-checked against thtk's `-r` decompile:

    0x00  u16 sub_count
    0x02  u16 timeline_count
    0x04  u32 timeline_offset[timeline_count]   (fixed 0x40-byte area; rest zero,
                                                 slot[count] often holds filesize)
    0x44  u32 sub_offset[sub_count]
    ...   subs, then timelines

Sub instruction (12-byte header + data):

    i32 time            u16 opcode         u16 size          (total, incl. header)
    u8  0 (reserved)    u8  rank_mask      u16 param_mask
    u8  data[size - 12]

`param_mask` bit i set => logical parameter i is a variable reference (the slot
holds the global-variable id, as an int, or as float bits for a float param).
A sub ends at an instruction with time == -1 or opcode == 0xffff.

Timeline instruction (8-byte header + data):

    u16 time   i16 arg0   u16 opcode   u16 size   u8 data[size - 8]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import opcodes
from .eclmap import EclMap, load_map

SUB_TABLE_OFFSET = 0x44
STRING_XOR = 0xAA  # ECL string parameters (spellcard names) are XOR'd with this


@dataclass
class Var:
    """A global-variable reference (`param_mask` bit was set)."""
    id: int
    is_float: bool = False

    def __repr__(self) -> str:
        return f"Var({self.id}{'f' if self.is_float else ''})"


@dataclass
class Instr:
    index: int          # position within the sub
    byte_offset: int    # offset from the sub's start (jump arithmetic)
    time: int
    opcode: int
    rank_mask: int
    param_mask: int
    args: list           # int | float | str | Var; jump offsets -> target instr index
    raw: bytes = b""     # raw parameter bytes
    name: str = ""

    @property
    def rank(self) -> str:
        return opcodes.rank_str(self.rank_mask)

    def runs_at(self, rank_bit: int) -> bool:
        return bool(self.rank_mask & rank_bit)


@dataclass
class Sub:
    index: int
    name: str
    instrs: list[Instr]

    def __iter__(self):
        return iter(self.instrs)


@dataclass
class TimelineInstr:
    time: int
    arg0: int
    opcode: int
    args: list
    name: str = ""


@dataclass
class Timeline:
    index: int
    instrs: list[TimelineInstr]


@dataclass
class ECLFile:
    subs: list[Sub]
    timelines: list[Timeline]
    eclmap: EclMap
    path: str | None = None

    def sub(self, i: int) -> Sub:
        return self.subs[i]


# ---------------------------------------------------------------- arg decoding

def _decode_args(sig: str, data: bytes, param_mask: int) -> tuple[list, int]:
    args: list = []
    pos = 0
    pidx = 0  # logical-parameter index (skips 'x' padding), indexes param_mask
    for ch in sig:
        if ch == "x":
            pos += 1
            continue
        need = 2 if ch in "hH" else (0 if ch == "s" else 4)
        if pos + need > len(data):
            break
        is_var = bool(param_mask & (1 << pidx))
        pidx += 1
        if ch == "s":
            blob = bytes(b ^ STRING_XOR for b in data[pos:])
            end = blob.find(b"\0")
            if end >= 0:
                blob = blob[:end]
            args.append(blob.decode("shift_jis", "replace"))
            pos = len(data)
        elif ch in "hH":
            fmt = "<h" if ch == "h" else "<H"
            (v,) = struct.unpack_from(fmt, data, pos)
            pos += 2
            args.append(Var(v & 0xFFFF, False) if is_var else v)
        elif ch == "f":
            (fv,) = struct.unpack_from("<f", data, pos)
            pos += 4
            args.append(Var(int(round(fv)), True) if is_var else fv)
        else:  # 'i' or 'o'
            (iv,) = struct.unpack_from("<i", data, pos)
            pos += 4
            args.append(Var(iv, False) if is_var else iv)
    return args, pos


# --------------------------------------------------------------- sub / timeline

def _parse_sub(data: bytes, start: int, index: int, eclmap: EclMap) -> tuple[Sub, int]:
    instrs: list[Instr] = []
    offsets: list[int] = []
    p = start
    while True:
        time, opcode = struct.unpack_from("<iH", data, p)
        if time == opcodes.SUB_END_TIME or opcode == opcodes.END_OPCODE:
            end = p + 12
            break
        size, _zero, rank_mask, param_mask = struct.unpack_from("<HBBH", data, p + 6)
        body = data[p + 12: p + size]
        sig = opcodes.SIGS.get(opcode)
        if sig is None:
            sig = "i" * (len(body) // 4)
        args, _ = _decode_args(sig, body, param_mask)
        instrs.append(Instr(
            index=len(instrs), byte_offset=p - start, time=time, opcode=opcode,
            rank_mask=rank_mask, param_mask=param_mask, args=args, raw=body,
            name=eclmap.ins_name(opcode),
        ))
        offsets.append(p - start)
        p += size

    off_to_idx = {o: i for i, o in enumerate(offsets)}
    for ins in instrs:
        jp = opcodes.JUMP_PARAM.get(ins.opcode)
        if jp is None or jp >= len(ins.args):
            continue
        tgt = ins.args[jp]
        if isinstance(tgt, int):
            ins.args[jp] = off_to_idx.get(ins.byte_offset + tgt, tgt)

    return Sub(index=index, name=f"Sub{index}", instrs=instrs), end


def _parse_timeline(data: bytes, start: int, index: int, eclmap: EclMap) -> Timeline:
    instrs: list[TimelineInstr] = []
    p = start
    while True:
        time, arg0 = struct.unpack_from("<Hh", data, p)
        if time == 0xFFFF:
            break
        opcode, size = struct.unpack_from("<HH", data, p + 4)
        body = data[p + 8: p + size]
        if 0 <= opcode <= 7 and len(body) >= 12:   # enemy/dummy create: 3 floats, then ints
            sig = "fff" + "i" * ((len(body) - 12) // 4)
        else:
            sig = "i" * (len(body) // 4)
        args, _ = _decode_args(sig, body, 0)
        instrs.append(TimelineInstr(
            time=time, arg0=arg0, opcode=opcode, args=args,
            name=eclmap.timeline_name(opcode),
        ))
        p += size
    return Timeline(index=index, instrs=instrs)


# --------------------------------------------------------------------- entry

def parse_bytes(data: bytes, eclmap: EclMap | None = None) -> ECLFile:
    if eclmap is None:
        eclmap = load_map()
    sub_count, timeline_count = struct.unpack_from("<HH", data, 0)
    timeline_offsets = list(struct.unpack_from(f"<{timeline_count}I", data, 4))
    sub_offsets = list(struct.unpack_from(f"<{sub_count}I", data, SUB_TABLE_OFFSET))

    subs: list[Sub] = []
    for i, so in enumerate(sub_offsets):
        sub, end = _parse_sub(data, so, i, eclmap)
        nxt = sub_offsets[i + 1] if i + 1 < sub_count else (
            timeline_offsets[0] if timeline_offsets else len(data))
        if end != nxt:
            raise ValueError(
                f"Sub{i} parsed to 0x{end:x} but next section is 0x{nxt:x} "
                f"(off-by-{nxt - end}) — instruction layout mismatch")
        subs.append(sub)

    timelines = [_parse_timeline(data, mo, i, eclmap)
                 for i, mo in enumerate(timeline_offsets)]
    return ECLFile(subs=subs, timelines=timelines, eclmap=eclmap)


def parse_file(path: str | Path, eclmap: EclMap | None = None) -> ECLFile:
    ecl = parse_bytes(Path(path).read_bytes(), eclmap)
    ecl.path = str(path)
    return ecl
