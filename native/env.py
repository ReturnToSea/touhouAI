"""Gymnasium environment wrapping one hooked th07.exe instance.

The game process is launched + hooked on construction, auto-navigates into
Stage 1, then snapshot/reset drives episodes. Headless and silent by default
(render skipped, audio session muted); pass render=True to watch it.

  * bomb is not in the action space yet.

    from native.env import Th07Env
    env = Th07Env(frame_skip=3)
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import atexit
import ctypes
import os
import time

import numpy as np


def _mute_pid(pid: int, tries: int = 25, delay: float = 0.1) -> bool:
    """Mute the game process's Windows audio session (best-effort).

    8 instances each blasting the stage BGM is unusable. The session only
    exists once the game has started a sound buffer, so retry briefly.
    """
    try:
        from pycaw.pycaw import AudioUtilities
    except Exception:
        return False
    for _ in range(tries):
        try:
            for sess in AudioUtilities.GetAllSessions():
                if sess.Process and sess.Process.pid == pid:
                    sess.SimpleAudioVolume.SetMute(1, None)
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False


class _BuildLock:
    """Cross-process mutex: serialise env construction (inject + menu nav).

    Concurrent D3D init + menu navigation across instances is racy - one game's
    menu stops advancing while another is bringing its device up (autonav then
    hits its frame cap). Building one env at a time avoids it entirely; once
    constructed they step concurrently without issue.
    """

    _NAME = b"th07env_build_lock"

    def __enter__(self):
        k32 = ctypes.windll.kernel32
        self._h = k32.CreateMutexA(None, False, self._NAME)
        k32.WaitForSingleObject(self._h, 120_000)  # 2 min: N * ~15s worst case
        return self

    def __exit__(self, *exc):
        k32 = ctypes.windll.kernel32
        if self._h:
            k32.ReleaseMutex(self._h)
            k32.CloseHandle(self._h)
            self._h = None


def _kill_pid(pid: int) -> None:
    """TerminateProcess(pid). No-op if it's already gone."""
    if not pid:
        return
    k32 = ctypes.windll.kernel32
    PROCESS_TERMINATE = 0x0001
    h = k32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
    if h:
        k32.TerminateProcess(h, 1)
        k32.CloseHandle(h)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # allow importing without gym for quick checks
    gym = None

import shm as S
from inject import inject

# action index -> (dx, dy) in {-1,0,1}
_DIRS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
NUM_ACTIONS = len(_DIRS) * 2 * 2  # dir x focus x shoot

# observation: the canonical builder is native/obs.py, shared with the sim.
import torch  # noqa: E402
from obs import (build_obs_batch, OBS_DIM, HEAD_DIM, NDIRS, GCELLS,  # noqa: E402,F401
                 GRID, M_ENEMIES, M_ITEMS)


def _decode_action(a: int) -> int:
    d = a % 9
    focus = (a // 9) % 2
    shoot = (a // 18) % 2
    dx, dy = _DIRS[d]
    bits = 0
    if dx < 0:
        bits |= S.LEFT
    if dx > 0:
        bits |= S.RIGHT
    if dy < 0:
        bits |= S.UP
    if dy > 0:
        bits |= S.DOWN
    if focus:
        bits |= S.SLOW
    if shoot:
        bits |= S.SHOOT
    return bits


_Base = gym.Env if gym is not None else object


class Th07Env(_Base):
    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 3, max_seconds: float = 90.0,
                 warmup: int = 90, render: bool = False, mute: bool = True,
                 hard_reset: bool = False):
        super().__init__()
        self.frame_skip = frame_skip
        self.max_steps = int(max_seconds * 60 / frame_skip)
        self.render_mode = "human" if render else None
        # hard_reset=True: every reset() (incl. SB3's auto-reset on episode end)
        # uses the engine-level Stage 1 reload instead of the snapshot restore -
        # the snapshot can't rewind a run that got deep into a boss.
        self._hard_reset_default = hard_reset

        # the DLL reads these at load time from the child's environment
        if render:
            os.environ["TH07_RENDER"] = "1"      # actually draw + present
        else:
            os.environ.pop("TH07_RENDER", None)

        self.pid = 0
        try:
            with _BuildLock():
                self.pid = inject()
                atexit.register(_kill_pid, self.pid)
                self.h = S.Hook(self.pid)
                s = self.h.s
                if mute and not render:
                    _mute_pid(self.pid)
                if not self.h.autonav():
                    raise RuntimeError(
                        f"auto-nav failed (nav_frames={s.nav_frames}, "
                        f"mode={s.gamemode}, stage={s.stage})")
                # a few frames past the stage-intro card, freeze the reset point
                for _ in range(warmup):
                    self.h.step(action=0, repeat=1)
                assert self.h.snapshot(), "snapshot failed"
        except BaseException:
            # never leak a game process on a failed construction - a leaked
            # instance holds a window and poisons later launches
            try:
                self.h.close()
            except Exception:
                pass
            _kill_pid(self.pid)
            raise
        self._pm = None
        try:
            import pymem
            self._pm = pymem.Pymem()
            self._pm.open_process_from_id(self.pid)
        except Exception:
            self._pm = None      # items just read as zeros

        self.start_lives = s.lives
        self.start_stage = s.stage
        print(f"[Th07Env] pid {self.pid}: nav {s.nav_frames}f -> stage {s.stage}, "
              f"lives {s.lives:.0f}, {s.bullet_count} bullets")

        if gym is not None:
            self.action_space = spaces.Discrete(NUM_ACTIONS)
            self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)

        # zero-copy float view of shm.bullets  (MAX_BULLETS x 4: x,y,vx,vy)
        self._bview = np.frombuffer(
            self.h._mm, dtype=np.float32, count=S.MAX_BULLETS * 4,
            offset=S.Shm.bullets.offset,
        ).reshape(S.MAX_BULLETS, 4)
        self._prev_bpos = np.full((S.MAX_BULLETS, 2), -9999.0, np.float32)
        self._prev_ppos = np.full(2, -9999.0, np.float32)
        self._reset_bookkeeping()

    # ------------------------------------------------------------------
    def _reset_bookkeeping(self):
        s = self.h.s
        self._steps = 0
        self._prev_score = s.score
        self._prev_graze = s.graze
        self._prev_boss = s.boss_hp
        self._prev_lives = s.lives
        self._prev_bpos[:] = -9999.0
        self._prev_ppos[:] = -9999.0

    def _bullet_arrays(self):
        xy = self._bview[:, :2].copy()
        live = xy[:, 0] > -9000.0
        prev_live = self._prev_bpos[:, 0] > -9000.0
        both = (live & prev_live)[:, None]
        vel = np.where(both, xy - self._prev_bpos, 0.0).astype(np.float32)
        self._prev_bpos = np.where(live[:, None], xy, -9999.0)
        return xy, vel, live

    # PCB stores the stage-1 MIDBOSS (Cirno) and the BOSS (Letty) as a zEnemy*
    # in EM_BOSSES[0] - NOT in the EM_ENEMIES array - so they're invisible unless
    # we read the pointer directly. (&EM_BOSSES[0] = ENEMY_MANAGER + 0x954598.)
    _EM_BOSSES0 = 0x009A9B00 + 0x00954598
    _E_POS, _E_LIFE, _E_ML = 0x2B0C, 0x2BB8, 0x2BBC

    def _boss(self):
        """(x, y, life, maxlife) of the EM_BOSSES[0] enemy, or None."""
        pm = getattr(self, "_pm", None)
        if pm is None:
            return None
        try:
            import struct
            ptr = struct.unpack_from("<I", pm.read_bytes(self._EM_BOSSES0, 4), 0)[0]
            if not (0x00400000 < ptr < 0x7FFFFFFF):
                return None
            bs = pm.read_bytes(ptr, 0x2C00)
            x, y = struct.unpack_from("<ff", bs, self._E_POS)
            life = struct.unpack_from("<i", bs, self._E_LIFE)[0]
            ml = struct.unpack_from("<i", bs, self._E_ML)[0]
            if not (-64 < x < 448 and -80 < y < 520) or ml < 1 or ml > 1_000_000:
                return None
            return float(x), float(y), int(life), int(ml)
        except Exception:
            return None

    def _big_target(self):
        """(present, hp_fraction) of the stage boss / midboss."""
        s = self.h.s
        if s.boss_present and s.boss_hp_max > 1:
            return True, float(s.boss_hp / s.boss_hp_max)
        b = self._boss()
        if b is not None:
            return True, b[2] / max(b[3], 1)
        best_ml, best_life = 0, 0
        for i in range(min(s.enemy_count, S.MAX_ENEMIES)):
            e = s.enemies[i]
            if -8 <= e.y <= 480 and e.maxlife >= 200 and e.maxlife > best_ml:
                best_ml, best_life = e.maxlife, e.life
        if best_ml:
            return True, best_life / best_ml
        return False, 0.0

    def _obs(self) -> np.ndarray:
        """Build the observation via the shared batched builder (native/obs.py),
        so the real env and the danmaku sim see bit-identical inputs."""
        s = self.h.s
        px, py = s.player_x, s.player_y
        fs = self.frame_skip

        if self._prev_ppos[0] > -9000.0:
            pvx = (px - self._prev_ppos[0]) / fs
            pvy = (py - self._prev_ppos[1]) / fs
        else:
            pvx = pvy = 0.0
        self._prev_ppos[:] = (px, py)

        xy, vel, live = self._bullet_arrays()          # [N,2], per-decision, [N]
        bp = torch.from_numpy(np.ascontiguousarray(xy))[None]
        bv = torch.from_numpy(np.ascontiguousarray(vel / fs))[None]   # per-frame
        ba = torch.from_numpy(live.astype(np.float32))[None]

        pstate = [0.0] * 5
        if 0 <= s.player_state < 5:
            pstate[s.player_state] = 1.0
        bt_present, bt_frac = self._big_target()
        head_aux = torch.tensor([[
            s.lives / 9.0, s.bombs / 9.0, s.power / 128.0,
            float(np.tanh(s.graze / 100.0)), s.stage / 6.0,
            pstate[0], pstate[2],
            1.0 if bt_present else 0.0, float(bt_frac),
        ]], dtype=torch.float32)

        # NEAREST M_ENEMIES (matches the sim, which topk's nearest-6), and the
        # EM_BOSSES[0] midboss/boss is folded in as a normal enemy.
        cand = []
        bt = self._boss()
        if bt is not None:
            bx, by, bl, bml = bt
            cand.append(((bx - px) ** 2 + (by - py) ** 2, bx, by, bl / max(bml, 1)))
        for i in range(min(s.enemy_count, S.MAX_ENEMIES)):
            e = s.enemies[i]
            if e.y < -8 or e.y > 480:
                continue
            cand.append(((e.x - px) ** 2 + (e.y - py) ** 2,
                         e.x, e.y, e.life / max(e.maxlife, 1)))
        cand.sort(key=lambda c: c[0])
        enemies = torch.zeros(1, M_ENEMIES * 3)
        for k, (_, ex, ey, hpf) in enumerate(cand[:M_ENEMIES]):
            enemies[0, k * 3:k * 3 + 3] = torch.tensor(
                [(ex - px) / 128.0, (ey - py) / 128.0, min(max(hpf, 0.0), 1.0)])

        items = self._item_arrays(px, py)      # [1, M_ITEMS*3]

        o = build_obs_batch(
            torch.tensor([[px, py]], dtype=torch.float32),
            torch.tensor([[pvx, pvy]], dtype=torch.float32),
            torch.tensor([float(s.player_focus)]),
            bp, bv, ba, head_aux, enemies, items)
        return o[0].numpy()

    def _item_arrays(self, px: float, py: float) -> "torch.Tensor":
        """Nearest M_ITEMS on-field items from the live ItemManager, as
        (rel_x/128, rel_y/128, type/9). Read straight from process memory
        (pymem); the DLL doesn't mirror items into shm yet. Returns zeros if
        the read isn't available."""
        out = torch.zeros(1, M_ITEMS * 3)
        pm = getattr(self, "_pm", None)
        if pm is None:
            return out
        try:
            import struct
            base = 0x00575C70                    # ITEM_MANAGER (th07_addrs.h)
            # items sit near the cycling `next_index` cursor, not the array front,
            # so the whole 0x44C-slot array must be scanned.
            blob = pm.read_bytes(base, 0x288 * 0x44C)
            cand = []
            for i in range(0x44C):
                b = i * 0x288
                if not blob[b + 0x27D]:           # in_use
                    continue
                x, y = struct.unpack_from("<ff", blob, b + 0x24C)
                if y < -16 or y > 464:
                    continue
                t = blob[b + 0x27C]
                cand.append(((x - px) ** 2 + (y - py) ** 2, x, y, t))
            cand.sort(key=lambda c: c[0])
            for k, (_, x, y, t) in enumerate(cand[:M_ITEMS]):
                out[0, k * 3:k * 3 + 3] = torch.tensor(
                    [(x - px) / 128.0, (y - py) / 128.0, t / 9.0])
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    def rollout_policy(self, weights, hidden, render: bool = False,
                       on_wait=None) -> dict:
        """Evaluate a flat MLP weight vector for one whole episode inside the
        DLL (no per-frame Python round trip). Returns {frames, score, graze,
        died, tick_status}. `hidden` is (h1, h2)."""
        h1, h2 = hidden
        r = self.h.eval_policy(weights, h1, h2, frame_skip=self.frame_skip,
                               max_frames=self.max_steps * self.frame_skip,
                               render=render, on_wait=on_wait)
        if r is None:
            s = self.h.s
            if s.crash_code:
                raise RuntimeError(
                    f"game crashed: exc {s.crash_code:#x} at eip {s.crash_eip:#x}")
            raise RuntimeError("eval_policy timed out (game not responding)")
        return r

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # options={"hard": True} -> engine-level Stage 1 reload (Give Up &
        # Retry). Rewinds from anywhere (deep boss fights included) so one game
        # process serves many episodes; ~1s vs the snapshot restore's instant.
        hard = self._hard_reset_default
        if options and "hard" in options:
            hard = bool(options["hard"])
        ok = self.h.hard_reset() if hard else self.h.reset()
        if not ok:
            s = self.h.s
            if s.crash_code:
                raise RuntimeError(
                    f"game crashed: exc {s.crash_code:#x} at eip {s.crash_eip:#x}, "
                    f"{'wrote' if s.crash_rw else 'read'} {s.crash_addr:#x}")
            raise RuntimeError(
                f"hook {'hard_' if hard else ''}reset timed out (game not responding)")
        self._prev_bpos[:] = -9999.0
        self._prev_ppos[:] = -9999.0
        self._reset_bookkeeping()
        return self._obs(), {}

    def step(self, action: int):
        s = self.h.s
        bits = _decode_action(int(action))
        ok = self.h.step(action=bits, repeat=self.frame_skip)
        self._steps += 1

        reward = 0.02 * self.frame_skip  # alive
        reward += (s.score - self._prev_score) * 1e-4
        # no graze reward - on Lunatic it just trains flying at bullets
        if s.boss_present and s.boss_hp_max > 0:
            reward += max(0.0, self._prev_boss - s.boss_hp) / s.boss_hp_max * 3.0
        died = s.lives < self._prev_lives - 0.5
        if died:
            reward -= 5.0

        self._prev_score, self._prev_graze = s.score, s.graze
        self._prev_boss, self._prev_lives = s.boss_hp, s.lives

        terminated = bool(died or s.tick_status != 0)
        truncated = bool(not ok or self._steps >= self.max_steps)
        info = {"frame": s.frame, "score": s.score, "tick_status": s.tick_status}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.h.set_free()
            self.h.close()
        except Exception:
            pass
        _kill_pid(getattr(self, "pid", 0))
        try:
            atexit.unregister(_kill_pid)
        except Exception:
            pass
