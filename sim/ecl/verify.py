"""Part 1 verification: round-trip the binary parser against thtk's decompile.

    python -m sim.ecl.verify [tools/th07_ecl]

For every `ecldataN.ecl` it:
  * decompiles with `thecl -d 7 -r` (cached as `_rN.tecl`),
  * parses the binary with our parser,
  * checks: same sub count, same instruction count per sub, and every
    instruction's (time, opcode, rank, args) decodes identically.

Exit code 0 iff everything matches.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from .parser import Var, parse_file
from . import opcodes

THECL = Path(__file__).resolve().parents[2] / "tools" / "thtk" / "thtk-bin-12" / "thecl.exe"

_SUB_RE = re.compile(r"^(?:sub|timeline)\s+(\w+)\(\)")
_LABEL_RE = re.compile(r"^(\w+):$")
_TIME_RE = re.compile(r"^([+-]?\d+):(?:\s*//\s*(-?\d+))?")
_DIFF_RE = re.compile(r"^!(\S+)\s+")
_INS_RE = re.compile(r"^ins_(\d+)\((.*)\);$")


def _thtk_raw(ecl_path: Path) -> str:
    out = ecl_path.with_name(f"_r{ecl_path.stem[-1]}.tecl")
    if not out.exists() or out.stat().st_mtime < ecl_path.stat().st_mtime:
        raw = subprocess.run([str(THECL), "-d", "7", "-r", str(ecl_path)],
                             capture_output=True, check=True).stdout
        out.write_bytes(raw)
    # thtk emits spellcard names in the game's locale codepage (shift-jis / cp932)
    return out.read_text(encoding="cp932", errors="replace")


def _split_args(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            depth += ch in "(["
            depth -= ch in ")]"
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _parse_thtk_sub(lines: list[str]) -> list[dict]:
    """Return [{op, time, diff, args:[raw tokens]}] plus a label->index map baked in."""
    instrs: list[dict] = []
    labels: dict[str, int] = {}
    t = 0
    diff = "*"  # persists until the next !X prefix (thtk emits a prefix only on change)
    for raw in lines:
        line = raw.strip()
        if not line or line in "{}":
            continue
        m = _LABEL_RE.match(line)
        if m and "(" not in line:
            labels[m.group(1)] = len(instrs)
            continue
        dm = _DIFF_RE.match(line)
        if dm:
            diff = dm.group(1)
            line = line[dm.end():].strip()
        tm = _TIME_RE.match(line)
        if tm:
            if tm.group(2) is not None:
                t = int(tm.group(2))
            elif tm.group(1)[0] in "+-":
                t += int(tm.group(1))
            else:
                t = int(tm.group(1))
            line = line[tm.end():].strip()
            if not line:
                continue
        im = _INS_RE.match(line)
        if im:
            instrs.append(dict(op=int(im.group(1)), time=t, diff=diff,
                               args=_split_args(im.group(2)) if im.group(2).strip() else []))
    return instrs, labels


def _iter_thtk_subs(text: str):
    cur, buf = None, []
    for line in text.splitlines():
        m = _SUB_RE.match(line)
        if m:
            if cur is not None:
                yield cur, buf
            cur, buf = m.group(1), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        yield cur, buf


def _norm_diff(d: str) -> str:
    if d == "*":
        return "*"
    return "".join(sorted(d))


def _cmp_arg(mine, tok: str, labels: dict[str, int]) -> bool:
    tok = tok.strip()
    var = tok.startswith("[") and tok.endswith("]")
    inner = tok[1:-1].strip() if var else tok
    if var:
        if not isinstance(mine, Var):
            return False
        is_float = inner.endswith("f") or "." in inner
        want = int(float(inner[:-1] if inner.endswith("f") else inner))
        return mine.id == want and mine.is_float == is_float
    if isinstance(mine, Var):
        return False
    if inner in labels:                       # jump target rendered as a label
        return mine == labels[inner]
    if re.match(r"^-?\d+$", inner):
        return isinstance(mine, int) and mine == int(inner)
    if re.match(r"^-?\d*\.?\d+(e-?\d+)?f?$", inner) and (inner.endswith("f") or "." in inner or "e" in inner):
        want = float(inner[:-1] if inner.endswith("f") else inner)
        return isinstance(mine, float) and (abs(mine - want) <= 1e-4 + 1e-4 * abs(want))
    if (inner[:1] in "\"'") and inner[-1:] in "\"'":
        theirs = inner[1:-1].replace('\\"', '"')
        if mine == theirs.replace("\\\\", "\\"):
            return True
        # thtk escapes every 0x5c byte, even one that is a shift-jis trail byte;
        # our cp932 decode merges it correctly. compare at the byte level, letting
        # thtk carry extra 0x5c bytes.
        mb = mine.encode("cp932", "replace")
        tb = theirs.encode("cp932", "replace")
        return mb == tb or mb == tb.replace(b"\\\\", b"\\")
    return str(mine) == inner


def verify_file(ecl_path: Path) -> tuple[int, int, list[str]]:
    text = _thtk_raw(ecl_path)
    ecl = parse_file(ecl_path)
    thtk_subs = {name: _parse_thtk_sub(buf) for name, buf in _iter_thtk_subs(text)
                 if name.startswith("Sub")}

    problems: list[str] = []
    if len(thtk_subs) != len(ecl.subs):
        problems.append(f"sub count: thtk {len(thtk_subs)} vs parser {len(ecl.subs)}")

    checked = 0
    for sub in ecl.subs:
        tk = thtk_subs.get(sub.name)
        if tk is None:
            problems.append(f"{sub.name}: missing from thtk output")
            continue
        tk_instrs, labels = tk
        if len(tk_instrs) != len(sub.instrs):
            problems.append(f"{sub.name}: {len(sub.instrs)} instrs vs thtk {len(tk_instrs)}")
            continue
        for i, (mine, theirs) in enumerate(zip(sub.instrs, tk_instrs)):
            where = f"{sub.name}[{i}] ins_{mine.opcode}"
            if mine.opcode != theirs["op"]:
                problems.append(f"{where}: opcode {mine.opcode} vs {theirs['op']}")
                continue
            if mine.time != theirs["time"]:
                problems.append(f"{where}: time {mine.time} vs {theirs['time']}")
            if _norm_diff(mine.rank) != _norm_diff(theirs["diff"]):
                problems.append(f"{where}: rank {mine.rank!r} vs {theirs['diff']!r}")
            if len(mine.args) != len(theirs["args"]):
                problems.append(f"{where}: {len(mine.args)} args vs {len(theirs['args'])}")
            else:
                for a, (mv, tv) in enumerate(zip(mine.args, theirs["args"])):
                    if not _cmp_arg(mv, tv, labels):
                        problems.append(f"{where} arg{a}: {mv!r} vs {tv!r}")
            checked += 1
    return checked, len(problems), problems


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    root = Path(argv[1]) if len(argv) > 1 else THECL.parents[2] / "th07_ecl"
    files = sorted(root.glob("ecldata?.ecl"))
    if not files:
        print(f"no ecldata?.ecl in {root}", file=sys.stderr)
        return 2

    total_checked = total_problems = 0
    for f in files:
        checked, nprob, problems = verify_file(f)
        total_checked += checked
        total_problems += nprob
        status = "OK  " if nprob == 0 else "FAIL"
        print(f"  {status} {f.name:15} {checked:5} instructions verified"
              + (f"  — {nprob} mismatch(es)" if nprob else ""))
        for p in problems[:25]:
            print(f"         {p}")
        if nprob > 25:
            print(f"         … {nprob - 25} more")

    print(f"\n{total_checked} instructions checked across {len(files)} files, "
          f"{total_problems} mismatches")
    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
