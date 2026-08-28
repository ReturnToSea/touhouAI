"""Gymnasium environment wrapping one hooked th07.exe instance.

v0 limitations:
  * one game process per env; you must navigate into a stage by hand the first
    time (auto menu-nav is a TODO), after which snapshot/reset handles episodes.
  * bomb is not in the action space yet.

    from native.env import Th07Env
    env = Th07Env(frame_skip=3)          # opens the game, asks you to navigate in
    obs, info = env.reset()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import time

import numpy as np

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

K_BULLETS = 32
M_ENEMIES = 6
HEAD_DIM = 12
OBS_DIM = HEAD_DIM + K_BULLETS * 5 + M_ENEMIES * 3 + 2  # = 192


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
                 warmup: int = 90):
        super().__init__()
        self.frame_skip = frame_skip
        self.max_steps = int(max_seconds * 60 / frame_skip)

        self.pid = inject()
        self.h = S.Hook(self.pid)
        s = self.h.s
        if not self.h.autonav():
            raise RuntimeError(f"auto-nav failed (nav_frames={s.nav_frames}, "
                               f"mode={s.gamemode}, stage={s.stage})")
        # a few frames past the stage-intro card, then freeze the reset point
        for _ in range(warmup):
            self.h.step(action=0, repeat=1)
        assert self.h.snapshot(), "snapshot failed"
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

    def _bullet_arrays(self):
        xy = self._bview[:, :2].copy()
        live = xy[:, 0] > -9000.0
        prev_live = self._prev_bpos[:, 0] > -9000.0
        both = (live & prev_live)[:, None]
        vel = np.where(both, xy - self._prev_bpos, 0.0).astype(np.float32)
        self._prev_bpos = np.where(live[:, None], xy, -9999.0)
        return xy, vel, live

    def _obs(self) -> np.ndarray:
        s = self.h.s
        px, py = s.player_x, s.player_y
        W, H = 384.0, 448.0
        pstate = np.zeros(5, np.float32)
        if 0 <= s.player_state < 5:
            pstate[s.player_state] = 1.0

        head = np.array([
            px / W, py / H, s.player_vx / 10.0, s.player_vy / 10.0,
            float(s.player_focus),
            s.lives / 9.0, s.bombs / 9.0, s.power / 128.0,
            np.tanh(s.graze / 100.0),
            s.stage / 6.0,
            pstate[0], pstate[2],  # alive-flag, dead-flag
        ], np.float32)

        xy, vel, live = self._bullet_arrays()
        idx = np.where(live)[0]
        if len(idx):
            rel = xy[idx] - (px, py)
            d = np.hypot(rel[:, 0], rel[:, 1])
            order = np.argsort(d)[:K_BULLETS]
            sel = idx[order]
            rel = (xy[sel] - (px, py)) / 128.0
            v = vel[sel] / 10.0
            dist = (d[order] / 200.0)[:, None]
            b = np.concatenate([rel, v, dist], axis=1)
        else:
            b = np.zeros((0, 5), np.float32)
        if len(b) < K_BULLETS:
            b = np.vstack([b, np.zeros((K_BULLETS - len(b), 5), np.float32)])

        en = np.zeros((M_ENEMIES, 3), np.float32)
        for i in range(min(s.enemy_count, M_ENEMIES)):
            e = s.enemies[i]
            ml = max(e.maxlife, 1)
            en[i] = [(e.x - px) / 128.0, (e.y - py) / 128.0, e.life / ml]

        boss = np.array([float(s.boss_present),
                         (s.boss_hp / s.boss_hp_max) if s.boss_hp_max > 0 else 0.0],
                        np.float32)

        return np.concatenate([head, b.ravel(), en.ravel(), boss]).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        assert self.h.reset(), "hook reset timed out"
        self._reset_bookkeeping()
        return self._obs(), {}

    def step(self, action: int):
        s = self.h.s
        bits = _decode_action(int(action))
        ok = self.h.step(action=bits, repeat=self.frame_skip)
        self._steps += 1

        reward = 0.02 * self.frame_skip  # alive
        reward += (s.score - self._prev_score) * 1e-4
        reward += (s.graze - self._prev_graze) * 0.01
        if s.boss_present and s.boss_hp_max > 0:
            reward += max(0.0, self._prev_boss - s.boss_hp) / s.boss_hp_max * 3.0
        died = s.lives < self._prev_lives - 0.5
        if died:
            reward -= 1.0

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
