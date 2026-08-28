"""Kill every stray th07.exe (and inject32.exe) - use after a crashed run.

    python native\\killall.py
"""
from __future__ import annotations

import subprocess
import sys


def killall() -> int:
    n = 0
    for name in ("th07.exe", "inject32.exe"):
        r = subprocess.run(["taskkill", "/F", "/T", "/IM", name],
                           capture_output=True, text=True)
        n += r.stdout.count("SUCCESS")
    # dismiss any leftover WerFault error dialogs the guard used to spawn
    subprocess.run(["taskkill", "/F", "/IM", "WerFault.exe"],
                   capture_output=True, text=True)
    return n


if __name__ == "__main__":
    killed = killall()
    print(f"killed {killed} process(es)")
    sys.exit(0)
