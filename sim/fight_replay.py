"""GPU sim that replays recorded real boss fights (sim/fights/*.npz from
native/record_boss_driven.py). Bullets follow their exact recorded per-frame
positions - the real engine's output, hangs / accel / curves and all. B
parallel episodes each pick a random recording + random start offset.

No re-aiming yet (pure replay). With enough recordings (Cirno RNG + varied
player paths) the policy has to learn "dodge Cirno-like clouds", not memorise.

    from fight_replay import FightSim
    sim = FightSim(B=8192, name="cirno", device="cuda")
    obs = sim.reset(); obs, rew, done = sim.step(act)
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import (build_obs_batch, OBS_DIM, PX_LO, PX_HI, PY_LO, PY_HI,  # noqa: E402
                 DIR_SPEED, DIR_SPEED_FOCUS)

FIGHTS = Path(__file__).resolve().parent / "fights"
POOL = 1025                      # th07 bullet pool size
PLAYER_HB = 1.8                  # measured player half-extent
BULLET_HB = 2.5                  # rough stage-1 bullet half-extent
_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)


def _load_dense(npz_path):
    """npz bullets (frame,slot,x,y,vx,vy,cls,fx) -> [F, POOL, 2] float32 positions
    (NaN where the slot is empty that frame) + boss track [F,2]."""
    d = np.load(npz_path)
    b = d["bullets"]
    if len(b) == 0:
        return None
    f = b[:, 0].astype(np.int64)
    f -= f.min()
    F = int(f.max()) + 1
    slot = np.clip(b[:, 1].astype(np.int64), 0, POOL - 1)
    pos = np.full((F, POOL, 2), np.nan, np.float32)
    pos[f, slot, 0] = b[:, 2]
    pos[f, slot, 1] = b[:, 3]
    boss = np.full((F, 2), np.nan, np.float32)
    if len(d["boss"]):
        bf = d["boss"][:, 0].astype(np.int64)
        bf -= bf.min()
        bf = np.clip(bf, 0, F - 1)
        boss[bf] = d["boss"][:, 1:3]
    # forward-fill boss
    for i in range(1, F):
        m = np.isnan(boss[i, 0])
        boss[i] = np.where(m, boss[i - 1], boss[i])
    boss = np.nan_to_num(boss, nan=192.0)
    # trim a quiet lead-in (boss entrance / spellcard intro with ~no bullets)
    nb = (~np.isnan(pos[:, :, 0])).sum(1)
    busy = np.where(nb >= 15)[0]
    if len(busy) and busy[0] > 30:
        s = max(0, busy[0] - 30)
        pos, boss = pos[s:], boss[s:]
    return pos, boss


class FightSim:
    def __init__(self, B=8192, name="cirno", device="cuda", max_frames=11000,
                 min_start=0, warmup_max=200, seed=0):
        self.B, self.d = B, device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self._g = g
        paths = sorted(glob.glob(str(FIGHTS / f"{name}*.npz")))
        recs = [_load_dense(p) for p in paths]
        recs = [r for r in recs if r is not None and r[0].shape[0] > 600]
        assert recs, f"no usable recordings matching {name}*"
        self.n_rec = len(recs)
        self.rec_len = torch.tensor([r[0].shape[0] for r in recs])
        maxF = max(r[0].shape[0] for r in recs)
        self.maxF = min(maxF, max_frames)
        # pack into [n_rec, maxF, POOL, 2] (pad with NaN) + boss [n_rec, maxF, 2]
        self.pos = torch.full((self.n_rec, self.maxF, POOL, 2), float("nan"))
        self.boss = torch.full((self.n_rec, self.maxF, 2), 192.0)
        for i, (p, bo) in enumerate(recs):
            L = min(p.shape[0], self.maxF)
            self.pos[i, :L] = torch.from_numpy(p[:L])
            self.boss[i, :L] = torch.from_numpy(bo[:L])
        self.pos = self.pos.to(device)
        self.boss = self.boss.to(device)
        self.min_start, self.warmup_max = min_start, warmup_max
        print(f"[FightSim] {self.n_rec} recordings, {self.maxF} frames, "
              f"pool {POOL}, B={B}", flush=True)
        self._dirs = _DIRS.to(device)
        self.reset()

    def _sample_starts(self, idx):
        rid = torch.randint(0, self.n_rec, (len(idx),), generator=self._g)
        maxstart = (self.rec_len[rid] - 800).clamp(min=self.min_start + 1)
        off = (torch.rand(len(idx), generator=self._g) *
               (maxstart - self.min_start) + self.min_start).long()
        self.rec_id[idx] = rid.to(self.d)
        self.t0[idx] = off.to(self.d)

    def reset(self, idx=None):
        if idx is None:
            self.rec_id = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.t0 = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.t = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.px = torch.full((self.B,), 192.0, device=self.d)
            self.py = torch.full((self.B,), 384.0, device=self.d)
            self._prev_b = None
            idx = torch.arange(self.B, device=self.d)
        self._sample_starts(idx.cpu())
        self.t[idx] = 0
        self.px[idx] = 192.0
        self.py[idx] = 384.0
        self._prev_b = None
        return self._obs()

    def _bullets_now(self):
        f = (self.t0 + self.t).clamp(max=self.maxF - 1)         # [B]
        # gather [B, POOL, 2]
        bp = self.pos[self.rec_id, f]                            # [B, POOL, 2]
        active = ~torch.isnan(bp[..., 0])
        bp = torch.nan_to_num(bp, nan=-9999.0)
        return bp, active, f

    def _obs(self):
        bp, active, f = self._bullets_now()
        if self._prev_b is None:
            bv = torch.zeros_like(bp)
        else:
            pv, pa = self._prev_b
            both = (active & pa)[..., None]
            bv = torch.where(both, bp - pv, torch.zeros_like(bp))
        self._prev_b = (bp, active)
        bx = self.boss[self.rec_id, f]                           # [B, 2]
        pl = torch.stack([self.px, self.py], -1)
        head = torch.zeros(self.B, 9, device=self.d)
        head[:, 4] = 1.0                                         # stage/6 ~ 1
        head[:, 5] = 1.0                                         # alive
        enemies = torch.zeros(self.B, 18, device=self.d)
        enemies[:, 0] = (bx[:, 0] - self.px) / 128.0
        enemies[:, 1] = (bx[:, 1] - self.py) / 128.0
        enemies[:, 2] = 1.0
        items = torch.zeros(self.B, 24, device=self.d)
        return build_obs_batch(
            pl, torch.zeros(self.B, 2, device=self.d),
            torch.zeros(self.B, device=self.d),
            bp, bv, active.float(), head, enemies, items)

    def step(self, act):
        act = act.long()
        d = act % 9
        focus = (act // 9) % 2
        mv = self._dirs[d]                                       # [B, 2]
        ln = mv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        spd = torch.where(focus[:, None] > 0,
                          torch.tensor(DIR_SPEED_FOCUS, device=self.d),
                          torch.tensor(DIR_SPEED, device=self.d))   # [B, 1]
        moved = (d != 0)[:, None]
        step = torch.where(moved, mv / ln * spd, torch.zeros_like(mv))
        self.px = (self.px + step[:, 0]).clamp(PX_LO, PX_HI)
        self.py = (self.py + step[:, 1]).clamp(PY_LO, PY_HI)
        self.t += 1

        bp, active, f = self._bullets_now()
        rel = (bp - torch.stack([self.px, self.py], -1)[:, None, :]).abs()
        hit_r = PLAYER_HB + BULLET_HB
        hit = (active & (rel[..., 0] < hit_r) & (rel[..., 1] < hit_r)).any(dim=1)
        ended = (self.t0 + self.t) >= (self.rec_len[self.rec_id.cpu()].to(self.d) - 2)
        done = hit | ended

        rew = torch.full((self.B,), 0.02, device=self.d)         # alive
        rew = rew - 5.0 * hit.float()
        obs = self._obs()
        if done.any():
            self.reset(done.nonzero(as_tuple=True)[0])
        return obs, rew, done


if __name__ == "__main__":
    import time
    s = FightSim(B=4096, name="c_", device="cuda" if torch.cuda.is_available() else "cpu")
    o = s.reset()
    print("obs", o.shape, "range", float(o.min()), float(o.max()))
    t0 = time.perf_counter()
    surv = torch.zeros(s.B, device=s.d)
    for i in range(1200):
        a = torch.randint(0, 36, (s.B,), device=s.d)
        o, r, dn = s.step(a)
        surv += (~dn).float()
    dt = time.perf_counter() - t0
    print(f"1200 steps B={s.B}: {dt:.1f}s = {1200*s.B/dt/1e3:.0f}k steps/s")
    print(f"random-policy mean survival: {surv.mean()*3/60:.1f}s")
