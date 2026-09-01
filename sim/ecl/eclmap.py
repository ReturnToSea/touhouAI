"""Parse an eclmap (`th07.eclm`, ExpHP/zero318 format) — opcode names, timeline
opcode names, and global-variable names. Names only; the parser gets its
parameter types from `opcodes.SIGS`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "tools" / "th07_ecl" / "th07.eclm"

_SECTION_RE = re.compile(r"^!(\S+)")
_ROW_RE = re.compile(r"^(-?\d+)\s+(.+?)\s*$")


@dataclass
class EclMap:
    ins: dict[int, str] = field(default_factory=dict)
    timeline_ins: dict[int, str] = field(default_factory=dict)
    gvar: dict[int, str] = field(default_factory=dict)

    def ins_name(self, op: int) -> str:
        return self.ins.get(op, f"ins_{op}")

    def timeline_name(self, op: int) -> str:
        return self.timeline_ins.get(op, f"ins_{op}")

    def gvar_name(self, gid: int) -> str:
        return self.gvar.get(gid, f"[{gid}]")


def load_map(path: str | Path | None = None) -> EclMap:
    p = Path(path) if path is not None else _DEFAULT
    m = EclMap()
    target = None
    routing = {
        "ins_names": m.ins,
        "timeline_ins_names": m.timeline_ins,
        "gvar_names": m.gvar,
    }
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        sec = _SECTION_RE.match(line)
        if sec:
            target = routing.get(sec.group(1))
            continue
        if target is None:
            continue
        row = _ROW_RE.match(line)
        if row:
            target[int(row.group(1))] = row.group(2)
    return m
