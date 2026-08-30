"""Annotate a raw thecl dump (ins_NNN / [100NN]) with names from th07.eclm.
thtk-bin-12's eclmap parser rejects the modern eclm format, so we post-process.

    python annotate.py ecldata1.tecl [out.tecl]
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
eclm = (HERE / "th07.eclm").read_text(encoding="utf-8", errors="replace").splitlines()

ins, gvar = {}, {}
sec = None
for ln in eclm:
    ln = ln.split("#")[0].strip()
    if not ln:
        continue
    if ln.startswith("!"):
        sec = ln
        continue
    parts = ln.split(None, 1)
    if len(parts) != 2 or not parts[0].lstrip("-").isdigit():
        continue
    n, name = int(parts[0]), parts[1].strip()
    if sec == "!ins_names":
        ins[n] = name
    elif sec == "!gvar_names":
        gvar[n] = name

src = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
src = re.sub(r"ins_(\d+)", lambda m: ins.get(int(m.group(1)), m.group(0)), src)
# [10021] / [10021.0f] -> PLAYER_X etc.
src = re.sub(r"\[(\d+)(?:\.0f)?\]",
             lambda m: gvar.get(int(m.group(1)), m.group(0)), src)
out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].rsplit(".", 1)[0] + "_named.tecl"
Path(out).write_text(src, encoding="utf-8")
print(f"wrote {out}")
