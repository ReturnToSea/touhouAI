"""Python side of the th07hook shared-memory contract (mirror of th07_shm.h)."""
from __future__ import annotations

import ctypes
import mmap
import struct
import time

SHM_MAGIC = 0x37304854
MAX_BULLETS = 2048
MAX_ENEMIES = 64

ST_IDLE, ST_STEP, ST_FREE, ST_RESET = 0, 1, 2, 3


class Bullet(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                ("vx", ctypes.c_float), ("vy", ctypes.c_float)]


class Enemy(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                ("life", ctypes.c_int32), ("maxlife", ctypes.c_int32)]


class Shm(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("done", ctypes.c_uint32),
        ("frame", ctypes.c_uint32),
        ("action", ctypes.c_uint16),
        ("repeat", ctypes.c_uint16),
        ("tick_status", ctypes.c_int32),
        ("alive", ctypes.c_uint32),
        ("player_x", ctypes.c_float),
        ("player_y", ctypes.c_float),
        ("player_vx", ctypes.c_float),
        ("player_vy", ctypes.c_float),
        ("player_state", ctypes.c_uint8),
        ("player_focus", ctypes.c_uint8),
        ("_pad0", ctypes.c_uint8 * 2),
        ("lives", ctypes.c_float),
        ("bombs", ctypes.c_float),
        ("power", ctypes.c_float),
        ("score", ctypes.c_int32),
        ("graze", ctypes.c_int32),
        ("cherry", ctypes.c_int32),
        ("cherry_max", ctypes.c_int32),
        ("stage", ctypes.c_int32),
        ("difficulty", ctypes.c_int32),
        ("gamemode", ctypes.c_int32),
        ("boss_present", ctypes.c_int32),
        ("boss_hp", ctypes.c_float),
        ("boss_hp_max", ctypes.c_float),
        ("bullet_count", ctypes.c_int32),
        ("enemy_count", ctypes.c_int32),
        ("bullets", Bullet * MAX_BULLETS),
        ("enemies", Enemy * MAX_ENEMIES),
    ]


class Hook:
    """Open the mapping th07hook.dll created for a given pid."""

    def __init__(self, pid: int, timeout: float = 15.0):
        name = f"th07hook_{pid}"
        deadline = time.time() + timeout
        self._mm = None
        while time.time() < deadline:
            try:
                mm = mmap.mmap(-1, ctypes.sizeof(Shm), name,
                               access=mmap.ACCESS_WRITE)
            except OSError:
                time.sleep(0.1)
                continue
            shm = Shm.from_buffer(mm)
            if shm.magic == SHM_MAGIC:
                self._mm, self.s = mm, shm
                return
            del shm
            mm.close()
            time.sleep(0.1)
        raise TimeoutError(f"shared mapping {name} never appeared / bad magic")

    # --- control ---
    def set_free(self):
        self.s.state = ST_FREE

    def step(self, action: int, repeat: int = 1, timeout: float = 5.0) -> bool:
        s = self.s
        s.action = action & 0xFFFF
        s.repeat = max(1, repeat)
        s.done = 0
        s.state = ST_STEP
        deadline = time.perf_counter() + timeout
        while not s.done:
            if time.perf_counter() > deadline:
                return False
        return True

    def bullets(self):
        n = min(self.s.bullet_count, MAX_BULLETS)
        return [(self.s.bullets[i].x, self.s.bullets[i].y) for i in range(n)]

    def close(self):
        if self._mm is not None:
            import gc
            self.s = None
            gc.collect()
            try:
                self._mm.close()
            except BufferError:
                pass  # a ctypes view is lingering; GC will free the mapping
            self._mm = None
