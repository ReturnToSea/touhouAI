"""Python side of the th07hook shared-memory contract (mirror of th07_shm.h)."""
from __future__ import annotations

import ctypes
import mmap
import struct
import time

SHM_MAGIC = 0x37304854
MAX_BULLETS = 2048
MAX_ENEMIES = 64
OBS_DIM = 236  # 16 head + 9 escape + 13*13 grid + 6*3 enemies + 8*3 items (mirror th07_shm.h)
N_ACTIONS = 36
MAX_WEIGHTS = 1 << 17

(ST_IDLE, ST_STEP, ST_FREE, ST_RESET, ST_SNAPSHOT, ST_AUTONAV, ST_EVAL,
 ST_HARD_RESET) = range(8)

# input bits (confirmed against the game)
SHOOT, BOMB, SLOW, SKIP = 0x01, 0x02, 0x04, 0x08
UP, DOWN, LEFT, RIGHT = 0x10, 0x20, 0x40, 0x80


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
        ("have_snapshot", ctypes.c_uint32),
        ("nav_frames", ctypes.c_int32),
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
        ("crash_code", ctypes.c_uint32),
        ("crash_eip", ctypes.c_uint32),
        ("crash_addr", ctypes.c_uint32),
        ("crash_rw", ctypes.c_uint32),
        ("eval_frame_skip", ctypes.c_uint32),
        ("eval_max_frames", ctypes.c_uint32),
        ("eval_h1", ctypes.c_uint32),
        ("eval_h2", ctypes.c_uint32),
        ("eval_render", ctypes.c_uint32),
        ("eval_warmup", ctypes.c_uint32),
        ("ep_frames", ctypes.c_uint32),
        ("ep_score", ctypes.c_int32),
        ("ep_graze", ctypes.c_int32),
        ("ep_died", ctypes.c_uint32),
        ("ep_tick_status", ctypes.c_int32),
        ("ep_boss_dmg", ctypes.c_float),
        ("ep_x_dev", ctypes.c_float),
        ("dbg_obs", ctypes.c_float * OBS_DIM),
        ("step_obs", ctypes.c_float * OBS_DIM),   # DLL-built obs, read verbatim by env
        ("step_obs_frame", ctypes.c_uint32),
        ("_pad_obs", ctypes.c_uint32),
        ("bullets", Bullet * MAX_BULLETS),
        ("enemies", Enemy * MAX_ENEMIES),
        ("weights", ctypes.c_float * MAX_WEIGHTS),
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

    def _cmd(self, state: int, timeout: float, poll: float = 0.0,
             on_wait=None) -> bool:
        # step() passes poll=0: spin hot for the first ~1ms (the common case
        # finishes in well under that), then yield the core so that N envs
        # waiting at once don't starve the game do_tick threads of CPU - that
        # was crashing long runs at ST_RESET. reset/snapshot pass poll>0 (a
        # real sleep) since their latency doesn't matter. on_wait (if given) is
        # called each poll iteration - used to pump the HUD during long evals.
        s = self.s
        s.done = 0
        s.state = state
        t_start = time.perf_counter()
        deadline = t_start + timeout
        while not s.done:
            now = time.perf_counter()
            if now > deadline:
                return False
            if on_wait is not None:
                on_wait()
            if poll:
                time.sleep(poll)
            elif now - t_start > 0.001:
                time.sleep(0)   # yield: hand the core to the game thread
        return True

    def step(self, action: int, repeat: int = 1, timeout: float = 10.0) -> bool:
        self.s.action = action & 0xFFFF
        self.s.repeat = max(1, repeat)
        return self._cmd(ST_STEP, timeout)

    def autonav(self, timeout: float = 25.0) -> bool:
        """Tap through the menus into Stage 1. Returns False on failure."""
        if not self._cmd(ST_AUTONAV, timeout, poll=0.001):
            return False
        return self.s.nav_frames >= 0 and self.s.gamemode == 2

    def snapshot(self, timeout: float = 15.0) -> bool:
        """Capture the current game state as the episode-reset point."""
        return self._cmd(ST_SNAPSHOT, timeout, poll=0.001)

    def reset(self, timeout: float = 30.0, tries: int = 3) -> bool:
        """Restore the snapshot (must have called snapshot() first).

        Retries: a timeout here is usually a transient CPU-contention spike
        (many envs resetting while torch has the cores), not a dead game.
        """
        for _ in range(tries):
            if self._cmd(ST_RESET, timeout, poll=0.001):
                return True
        return False

    def hard_reset(self, timeout: float = 30.0) -> bool:
        """Engine-level Stage 1 reload via the supervisor retry word (writes
        SUPERVISOR+0x158 = 10, same as the pause menu's "Give Up and Retry").

        Works from anywhere - including deep in a boss fight, where the
        snapshot restore can't rewind - so one game process can run many
        episodes without a relaunch. Slower than reset(): it ticks through the
        ~40-frame teardown + fade + a 90-frame warmup (~1s wall)."""
        if not self._cmd(ST_HARD_RESET, timeout, poll=0.001):
            return False
        return self.s.nav_frames >= 0

    # --- in-DLL episode eval: blocking + split (for the island orchestrator) ---
    def eval_start(self, weights, h1: int, h2: int, frame_skip: int = 3,
                   max_frames: int = 7200, render: bool = False,
                   warmup: int = 0) -> None:
        """Kick off one full-episode eval with `weights`. Non-blocking."""
        import numpy as np
        w = np.ascontiguousarray(weights, dtype=np.float32)
        ctypes.memmove(self.s.weights, w.ctypes.data, w.size * 4)
        s = self.s
        s.eval_h1, s.eval_h2 = int(h1), int(h2)
        s.eval_frame_skip = int(frame_skip)
        s.eval_max_frames = int(max_frames)
        s.eval_render = 1 if render else 0
        s.eval_warmup = int(warmup)
        s.done = 0
        s.state = ST_EVAL

    def eval_done(self) -> bool:
        return bool(self.s.done)

    def eval_result(self) -> dict:
        s = self.s
        return {
            "frames": int(s.ep_frames),
            "score": int(s.ep_score),
            "graze": int(s.ep_graze),
            "died": bool(s.ep_died),
            "tick_status": int(s.ep_tick_status),
            "boss_dmg": float(s.ep_boss_dmg),
            "x_dev": float(s.ep_x_dev),
        }

    def eval_policy(self, weights, h1: int, h2: int, frame_skip: int = 3,
                    max_frames: int = 7200, render: bool = False,
                    timeout: float = 120.0, on_wait=None, warmup: int = 0):
        """Blocking single-episode eval. Returns stats dict, or None on timeout."""
        self.eval_start(weights, h1, h2, frame_skip, max_frames, render, warmup)
        deadline = time.perf_counter() + timeout
        while not self.s.done:
            if time.perf_counter() > deadline:
                return None
            if on_wait is not None:
                on_wait()
            time.sleep(0.0005 if not render else 0.004)
        return self.eval_result()

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
