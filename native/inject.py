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
GOOD_CFG = HERE / "th07.windowed.cfg"   # 56-byte windowed config, byte 0x1F == 1


def _heal_cfg(exe: Path) -> None:
    """Keep the game on windowed mode. A half-initialised instance killed
    mid-write can leave th07.cfg zeroed/fullscreen -> the next launch goes
    fullscreen (exclusive D3D, breaks vectorisation). Restore from the backup
    when it looks wrong. The file must stay writable - the game opens it rw at
    startup and hangs if it can't."""
    if not GOOD_CFG.exists():
        return
    cfg = exe.parent / "th07.cfg"
    try:
        b = cfg.read_bytes()
        ok = len(b) == 56 and b[0x1F] == 0x01
    except OSError:
        ok = False
    if not ok:
        cfg.write_bytes(GOOD_CFG.read_bytes())


def inject(exe: Path = DEFAULT_EXE, dll: Path = DEFAULT_DLL) -> int:
    for p in (exe, dll, INJECT32):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} (run native/build.ps1?)")
    _heal_cfg(exe)
    r = subprocess.run([str(INJECT32), str(exe), str(dll)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise OSError(f"inject32 failed: {r.stderr.strip() or r.stdout.strip()}")
    return int(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--dll", type=Path, default=DEFAULT_DLL)
    ap.add_argument("--sound", action="store_true",
                    help="don't mute the game (default: silence it)")
    args = ap.parse_args()

    try:
        pid = inject(args.exe, args.dll)
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    # th07 blasts the title BGM the instant DirectSound inits, before any
    # per-session mute lands - keep the endpoint muted over the launch window,
    # then restore (mirrors native/env.py).
    if not args.sound:
        try:
            from env import _silence_launch
            _silence_launch(pid)
        except Exception:
            pass
    print(pid)


if __name__ == "__main__":
    main()
