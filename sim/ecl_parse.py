"""Parse a thecl text dump (`thecl -d 7 [-r]` + tools/th07_ecl/annotate.py) into
a structured form the ECL VM can execute.

    subs[sub_id] = [Instr(time, diff, op, args, label), ...]

- `time` is absolute (accumulated from `+N:` markers; instructions sharing a time
  run on the same frame, then the VM waits for the next).
- `diff` is one of '*', 'E', 'N', 'H', 'L', 'EN' (from `!X` line prefixes).
- `label` rows carry op=None and just mark a jump target.
- args: int, float, or str (gvar name like 'F2'/'PLAYER_X', or a label like
  'Sub30_448').
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Instr:
    time: int
    diff: str
    op: str | None
    args: list = field(default_factory=list)
    label: str | None = None


_SUB_RE = re.compile(r"^sub (\w+)\(\)")
_LABEL_RE = re.compile(r"^(\w+):\s*$")
_TIME_RE = re.compile(r"^([+-]?\d+):")           # "+6: //6"  or  "12:"
_DIFF_RE = re.compile(r"^!([ENHL*]+)\s+")
_INS_RE = re.compile(r"^([a-zA-Z_]\w*)\((.*)\);\s*$")


def _parse_arg(s: str):
    s = s.strip()
    if not s:
        return None
    if s.endswith("f") and re.match(r"^-?\d", s):
        return float(s[:-1])
    if re.match(r"^-?\d+$", s):
        return int(s)
    if re.match(r"^-?\d*\.\d+$", s):
        return float(s)
    if (s[0] == s[-1]) and s[0] in "\"'":
        return s[1:-1]
    return s          # gvar name or label


def _split_args(s: str) -> list:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            cur += ch
    if cur.strip():
        out.append(cur)
    return [_parse_arg(a) for a in out]


def parse(path: str | Path) -> dict[int, list[Instr]]:
    subs: dict[int, list[Instr]] = {}
    cur_id = None
    t = 0
    pending_diff = "*"
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line in ("{", "}"):
            if line == "}":
                cur_id = None
            continue
        m = _SUB_RE.match(line)
        if m:
            cur_id = int(re.sub(r"\D", "", m.group(1)))
            subs[cur_id] = []
            t = 0
            pending_diff = "*"
            continue
        if cur_id is None:
            continue

        dm = _DIFF_RE.match(line)
        diff = "*"
        if dm:
            diff = dm.group(1)
            line = line[dm.end():].strip()

        tm = _TIME_RE.match(line)
        if tm:
            rest = line[tm.end():].strip()
            cm = re.match(r"^//\s*(-?\d+)", rest)
            if cm:
                t = int(cm.group(1))              # thecl's absolute-time comment
            elif tm.group(1)[0] in "+-":
                t = t + int(tm.group(1))
            else:
                t = int(tm.group(1))
            line = re.sub(r"^//.*$", "", rest).strip()
            if not line:
                continue

        lm = _LABEL_RE.match(line)
        if lm and "(" not in line:
            subs[cur_id].append(Instr(t, diff, None, [], label=lm.group(1)))
            continue

        im = _INS_RE.match(line)
        if im:
            subs[cur_id].append(Instr(t, diff, im.group(1), _split_args(im.group(2))))
        # else: expression-decompiled line (only with non-raw dump) - skip; use -r
    return subs


if __name__ == "__main__":
    import sys
    subs = parse(sys.argv[1])
    print(f"{len(subs)} subs")
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for ins in subs.get(want, []):
        if ins.label:
            print(f"  {ins.label}:")
        else:
            print(f"  t={ins.time:<4} {ins.diff:<3} {ins.op}{tuple(ins.args)}")
