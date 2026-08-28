"""Launch th07.exe with th07hook.dll injected, via the 32-bit inject32.exe helper.

    python native\\inject.py               # launch + inject, print pid
    python native\\inject.py --exe PATH
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EXE = HERE.parent / "Touhou 7 - Perfect Cherry Blossom" / "th07.exe"
DEFAULT_DLL = HERE / "build" / "th07hook.dll"
INJECT32 = HERE / "build" / "inject32.exe"


def inject(exe: Path = DEFAULT_EXE, dll: Path = DEFAULT_DLL) -> int:
    for p in (exe, dll, INJECT32):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} (run native/build.ps1?)")
    r = subprocess.run([str(INJECT32), str(exe), str(dll)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise OSError(f"inject32 failed: {r.stderr.strip() or r.stdout.strip()}")
    return int(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    args = ap.parse_args()
    try:
        print(inject(args.exe, args.dll))
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
