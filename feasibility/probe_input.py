"""Gate A, step 2: prove we can drive the game with synthetic input.

th07 reads the keyboard through DirectInput device-state polling, so
message-queue tricks do not work. SendInput with KEYEVENTF_SCANCODE does, as
long as (a) the struct is the right size, (b) the game window has focus, and
(c) our process is not blocked by UIPI (game running elevated, us not).

This version is self-checking: it attaches with pymem and reads the game's own
INPUT word (0x4b9e4c) while holding each key, so we know whether the game
actually received the input regardless of what the screen shows. It also prints
the foreground window vs the game window so we can spot a focus problem.

Usage (one line):
    .venv\\Scripts\\python feasibility\\probe_input.py
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

import pymem

import th07_data as D

user32 = ctypes.WinDLL("user32", use_last_error=True)

# ---- SendInput plumbing (correctly sized for 32- and 64-bit) ----
ULONG_PTR = ctypes.c_size_t
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


SC = {"left": 0x4B, "right": 0x4D, "up": 0x48, "down": 0x50,
      "z": 0x2C, "x": 0x2D, "shift": 0x2A}
EXTENDED = {"left", "right", "up", "down"}


def _send(scan: int, extended: bool, up: bool) -> int:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    inp = INPUT(type=INPUT_KEYBOARD,
                u=_INPUTunion(ki=KEYBDINPUT(0, scan, flags, 0, 0)))
    n = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if n != 1:
        print(f"    SendInput FAILED (ret={n}, err={ctypes.get_last_error()})")
    return n


def key_down(name):
    _send(SC[name], name in EXTENDED, up=False)


def key_up(name):
    _send(SC[name], name in EXTENDED, up=True)


def foreground_title() -> str:
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return f"{hwnd:#x} '{buf.value}'"


def main() -> None:
    print(f"sizeof(INPUT) = {ctypes.sizeof(INPUT)} (want 28 on 32-bit py, 40 on 64-bit)")
    try:
        pm = pymem.Pymem("th07.exe")
    except Exception as exc:  # noqa: BLE001
        print(f"could not attach to th07.exe ({exc}); start the game first")
        sys.exit(1)

    game_hwnd = pm.read_uint(D.SUPERVISOR + 0x44)  # Supervisor.hwnd_game_window
    print(f"game window handle: {game_hwnd:#x}")

    def input_word() -> int:
        return pm.read_ushort(D.INPUT_CUR)

    for i in range(4, 0, -1):
        print(f"click the GAME window... {i}", end="\r", flush=True)
        time.sleep(1)
    print()
    print(f"foreground now: {foreground_title()}")
    if user32.GetForegroundWindow() != game_hwnd:
        print("  !! game is NOT the foreground window - it is paused and will "
              "ignore input. Click it and rerun.")

    seq = [("left", 0.8), ("right", 0.8), ("down", 0.6), ("z", 0.4), ("shift", 0.6)]
    for name, dur in seq:
        before = input_word()
        key_down(name)
        time.sleep(dur / 2)
        held = input_word()
        time.sleep(dur / 2)
        key_up(name)
        time.sleep(0.05)
        after = input_word()
        got = "OK" if held != before else "no change"
        print(f"  {name:<6} INPUT_CUR {before:#06x} -> {held:#06x} -> {after:#06x}"
              f"   [{got}]")

    for name in SC:
        key_up(name)
    print("done. 'OK' above means the game received the key.")


if __name__ == "__main__":
    main()
