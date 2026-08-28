"""Launch th07.exe with the 60fps frame limiter and vsync disabled - no DLL.

Static analysis of th07.exe v1.00b (image base 0x400000, no ASLR):

  Window::do_tick (0x4346E0) - the frame limiter. When the game is "early"
  (elapsed < 1/60 s) it skips running a frame unless a debug flag is set:
      0x4348CC  je 0x434924   (QPC timing path)      <- skip-frame branch
      0x434997  je 0x4349E2   (timeGetTime fallback) <- skip-frame branch

  Supervisor::init_d3d_device (0x434BD0) - picks the present interval:
      0x434C8A  je 0x434C96   <- if taken, DON'T force the no-vsync path
  Falling through instead runs `mov [0x575ABC], 1`, and 0x575ABC != 0 selects
  D3DPRESENT_INTERVAL_IMMEDIATE at device creation.

We NOP the three `je`s (2 bytes each) in the suspended process before it runs.
No timing race, no flag that the game can reset.

Restores nothing (it's a fresh child process); close the game to undo.

Usage:
    .venv\\Scripts\\python feasibility\\launch_uncapped.py [--exe PATH]
        [--frameskip N] [--seconds S] [--no-patch]

--seconds S : after launch, attach and print the logic tick rate for S seconds
              (start a stage and keep the game focused during that window).
"""
from __future__ import annotations

import argparse
import ctypes
import struct
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pymem

import th07_data as D

CALC_COUNT = D.SUPERVISOR + 0x150
FRAMESKIP  = 0x00575A8B

# (virtual address, expected original bytes, replacement)
PATCHES = [
    (0x004348CC, b"\x74\x56", b"\x90\x90"),  # limiter, QPC path
    (0x00434997, b"\x74\x49", b"\x90\x90"),  # limiter, timeGetTime path
    (0x00434C8A, b"\x74\x0A", b"\x90\x90"),  # force no-vsync (IMMEDIATE present)
]

DEFAULT_EXE = Path(
    r"C:\Users\spore\Documents\GitHub\touhouAI"
    r"\Touhou 7 - Perfect Cherry Blossom\th07.exe"
)

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
CREATE_SUSPENDED = 0x00000004
PAGE_EXECUTE_READWRITE = 0x40


class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
    ]


def rpm(h, addr, n):
    buf = (ctypes.c_char * n)()
    got = ctypes.c_size_t(0)
    if not k32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(got)):
        raise OSError(f"ReadProcessMemory({addr:#x}) err {ctypes.get_last_error()}")
    return buf.raw


def wpm(h, addr, data):
    old = wintypes.DWORD(0)
    k32.VirtualProtectEx(h, ctypes.c_void_p(addr), len(data),
                         PAGE_EXECUTE_READWRITE, ctypes.byref(old))
    put = ctypes.c_size_t(0)
    ok = k32.WriteProcessMemory(h, ctypes.c_void_p(addr), data, len(data),
                                ctypes.byref(put))
    k32.VirtualProtectEx(h, ctypes.c_void_p(addr), len(data), old, ctypes.byref(old))
    if not ok or put.value != len(data):
        raise OSError(f"WriteProcessMemory({addr:#x}) err {ctypes.get_last_error()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    ap.add_argument("--frameskip", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--no-patch", action="store_true",
                    help="launch normally (control run)")
    args = ap.parse_args()

    if not args.exe.exists():
        print(f"exe not found: {args.exe}")
        sys.exit(1)

    si = STARTUPINFO()
    si.cb = ctypes.sizeof(si)
    pi = PROCESS_INFORMATION()
    if not k32.CreateProcessW(str(args.exe), None, None, None, False,
                              CREATE_SUSPENDED, None, str(args.exe.parent),
                              ctypes.byref(si), ctypes.byref(pi)):
        print(f"CreateProcessW failed: {ctypes.get_last_error()}")
        sys.exit(1)
    print(f"launched suspended: pid {pi.dwProcessId}")

    try:
        if not args.no_patch:
            for va, want, new in PATCHES:
                cur = rpm(pi.hProcess, va, len(want))
                if cur != want:
                    print(f"  {va:#x}: expected {want.hex()} got {cur.hex()} - "
                          f"NOT patching (wrong build?)")
                    continue
                wpm(pi.hProcess, va, new)
                back = rpm(pi.hProcess, va, len(new))
                print(f"  {va:#x}: {want.hex()} -> {back.hex()} "
                      f"{'OK' if back == new else 'FAILED'}")
            if args.frameskip:
                wpm(pi.hProcess, FRAMESKIP, bytes([args.frameskip]))
                print(f"  frameskip 0x575A8B = {args.frameskip}")
        k32.ResumeThread(pi.hThread)
        print("resumed.")
    finally:
        k32.CloseHandle(pi.hThread)
        k32.CloseHandle(pi.hProcess)

    if args.seconds <= 0:
        print("Start a stage; use probe_memory.py / probe_uncap.py to measure.")
        return

    for _ in range(40):
        try:
            pm = pymem.Pymem("th07.exe")
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    else:
        print("could not attach")
        return

    def calc():
        return struct.unpack("<i", pm.read_bytes(CALC_COUNT, 4))[0]

    print(f"attached pid {pm.process_id}. START A STAGE and keep it focused. "
          f"measuring ~{args.seconds:.0f}s:")
    lc, lt = calc(), time.perf_counter()
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        time.sleep(2.0)
        try:
            c, t = calc(), time.perf_counter()
        except pymem.exception.MemoryReadError:
            print("  game exited")
            return
        print(f"  logic rate: {(c - lc) / (t - lt):8.1f} ticks/sec")
        lc, lt = c, t


if __name__ == "__main__":
    main()
