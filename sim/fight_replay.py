"""GPU sim that replays recorded real boss fights (sim/fights/*.npz from
native/record_boss_driven.py). Bullets follow their exact recorded per-frame
positions - the real engine's output, hangs / accel / curves and all. B
parallel episodes each pick a random recording + random start offset.

v2: per-bullet hitbox (recorded from zBullet+0xB7C) and satellite sub-enemy
bodies (recorded from EM_ENEMIES) - Letty's orbiting orbs contact-kill the
player, so a policy trained without them flies straight through one on the
real game. Still no re-aiming (pure replay).

    from fight_replay import FightSim
    sim = FightSim(B=8192, name="letty", device="cuda")
    obs = sim.reset(); obs, rew, done = sim.step(act)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import (build_obs_batch, OBS_DIM, PX_LO, PX_HI, PY_LO, PY_HI,  # noqa: E402
                 DIR_SPEED, DIR_SPEED_FOCUS)

FIGHTS = Path(__file__).resolve().parent / "fights"
POOL = 1025                      # th07 bullet pool size
MAX_EN = 48                      # max satellite sub-enemies tracked per frame
PLAYER_HB = 1.8                  # measured player half-extent
BULLET_HB_DEFAULT = 2.5          # fallback when a recording has no hitbox column
ENEMY_BODY_SCALE = 2.0 / 3.0     # player-body vs enemy-body box (pytouhou)
_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)


def _load_dense(npz_path):
    """npz -> (bpos[F,POOL,2], bhalf[F,POOL], boss[F,2], en[F,MAX_EN,3]) where
    en rows are (x, y, body_half); empty slots are NaN / body_half<=0."""
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

    half = np.full((F, POOL), BULLET_HB_DEFAULT, np.float32)
    if b.shape[1] >= 10:                       # cols 8,9 = AABB full size x,y
        hb = np.maximum(b[:, 8], b[:, 9]) * 0.5
        hb[hb <= 0] = BULLET_HB_DEFAULT
        half[f, slot] = hb

    boss = np.full((F, 2), np.nan, np.float32)
    if len(d["boss"]):
        bf = d["boss"][:, 0].astype(np.int64)
        bf -= bf.min()
        bf = np.clip(bf, 0, F - 1)
        boss[bf] = d["boss"][:, 1:3]
    for i in range(1, F):
        m = np.isnan(boss[i, 0])
        boss[i] = np.where(m, boss[i - 1], boss[i])
    boss = np.nan_to_num(boss, nan=192.0)

    # enemies: (frame, slot, x, y, life, hbx, hby, hbz) -> dense [F, MAX_EN, 3]
    # Only satellite sub-enemies (life == 1) with a real hitbox count as lethal
    # bodies - they fly into the dodge space. The boss body is excluded (you can
    # point-blank PCB bosses); revisit if transfer suggests otherwise.
    en = np.full((F, MAX_EN, 3), np.nan, np.float32)
    if "enemies" in d and len(d["enemies"]):
        e = d["enemies"]
        lethal_all = e[(e[:, 4] == 1) & (e[:, 5] > 0.5)]
        ef = lethal_all[:, 0].astype(np.int64)
        ef -= int(e[:, 0].min())
        ef = np.clip(ef, 0, F - 1)
        order = np.argsort(ef, kind="stable")
        lethal_all, ef = lethal_all[order], ef[order]
        idx = np.searchsorted(ef, np.arange(F + 1))
        for fr in range(F):
            rows = lethal_all[idx[fr]:idx[fr + 1]][:MAX_EN]
            n = len(rows)
            en[fr, :n, 0] = rows[:, 2]
            en[fr, :n, 1] = rows[:, 3]
            en[fr, :n, 2] = rows[:, 5] * 0.5 * ENEMY_BODY_SCALE   # (hbx/2)*2/3

    # trim a quiet lead-in (boss entrance / intro with ~no bullets)
    nb = (~np.isnan(pos[:, :, 0])).sum(1)
    busy = np.where(nb >= 15)[0]
    if len(busy) and busy[0] > 30:
        s = max(0, busy[0] - 30)
        pos, half, boss, en = pos[s:], half[s:], boss[s:], en[s:]
    return pos, half, boss, en


class FightSim:
    def __init__(self, B=8192, name="cirno", device="cuda", max_frames=11000,
                 min_start=0, seed=0):
        self.B, self.d = B, device
        self._g = torch.Generator(device="cpu").manual_seed(seed)
        paths = sorted(glob.glob(str(FIGHTS / f"{name}*.npz")))
        recs = [_load_dense(p) for p in paths]
        recs = [r for r in recs if r is not None and r[0].shape[0] > 600]
        assert recs, f"no usable recordings matching {name}*"
        self.n_rec = len(recs)
        self.rec_len = torch.tensor([r[0].shape[0] for r in recs])
        self.maxF = min(max(r[0].shape[0] for r in recs), max_frames)

        self.pos = torch.full((self.n_rec, self.maxF, POOL, 2), float("nan"))
        self.bhalf = torch.full((self.n_rec, self.maxF, POOL), BULLET_HB_DEFAULT)
        self.boss = torch.full((self.n_rec, self.maxF, 2), 192.0)
        self.en = torch.full((self.n_rec, self.maxF, MAX_EN, 3), float("nan"))
        for i, (p, h, bo, e) in enumerate(recs):
            L = min(p.shape[0], self.maxF)
            self.pos[i, :L] = torch.from_numpy(p[:L])
            self.bhalf[i, :L] = torch.from_numpy(h[:L])
            self.boss[i, :L] = torch.from_numpy(bo[:L])
            self.en[i, :L] = torch.from_numpy(e[:L])
        self.pos = self.pos.to(device)
        self.bhalf = self.bhalf.to(device)
        self.boss = self.boss.to(device)
        self.en = self.en.to(device)
        self.min_start = min_start
        n_en = int((~torch.isnan(self.en[..., 0])).any(-1).float().mean() * 100)
        print(f"[FightSim] {self.n_rec} recs, {self.maxF} frames, B={B}, "
              f"enemy bodies on ~{n_en}% of frames", flush=True)
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

    def _now(self):
        f = (self.t0 + self.t).clamp(max=self.maxF - 1)          # [B]
        bp = self.pos[self.rec_id, f]                            # [B, POOL, 2]
        active = ~torch.isnan(bp[..., 0])
        bp = torch.nan_to_num(bp, nan=-9999.0)
        bh = self.bhalf[self.rec_id, f]                          # [B, POOL]
        en = self.en[self.rec_id, f]                             # [B, MAX_EN, 3]
        en_active = ~torch.isnan(en[..., 0])
        en = torch.nan_to_num(en, nan=-9999.0)
        return bp, active, bh, en, en_active, f

    def _obs(self):
        bp, active, bh, en, en_a, f = self._now()
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
        head[:, 4] = 1.0
        head[:, 5] = 1.0
        # nearest lethal enemy body -> the obs "enemies" slot the policy expects
        enemies = torch.zeros(self.B, 18, device=self.d)
        dx = en[..., 0] - self.px[:, None]
        dy = en[..., 1] - self.py[:, None]
        d2 = torch.where(en_a, dx * dx + dy * dy,
                         torch.full_like(dx, 1e12))
        j = d2.argmin(dim=1)
        bi = torch.arange(self.B, device=self.d)
        has_en = en_a.any(dim=1)
        ne_dx = torch.where(has_en, dx[bi, j], bx[:, 0] - self.px)
        ne_dy = torch.where(has_en, dy[bi, j], bx[:, 1] - self.py)
        enemies[:, 0] = ne_dx / 128.0
        enemies[:, 1] = ne_dy / 128.0
        enemies[:, 2] = 1.0
        items = torch.zeros(self.B, 24, device=self.d)
        return build_obs_batch(
            pl, torch.zeros(self.B, 2, device=self.d),
            torch.zeros(self.B, device=self.d),
            bp, bv, active.float(), head, enemies, items)

    def step(self, act):
        act = act.long()
        dd = act % 9
        focus = (act // 9) % 2
        mv = self._dirs[dd]
        ln = mv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        spd = torch.where(focus[:, None] > 0,
                          torch.tensor(DIR_SPEED_FOCUS, device=self.d),
                          torch.tensor(DIR_SPEED, device=self.d))
        moved = (dd != 0)[:, None]
        mstep = torch.where(moved, mv / ln * spd, torch.zeros_like(mv))
        self.px = (self.px + mstep[:, 0]).clamp(PX_LO, PX_HI)
        self.py = (self.py + mstep[:, 1]).clamp(PY_LO, PY_HI)
        self.t += 1

        bp, active, bh, en, en_a, f = self._now()
        pxy = torch.stack([self.px, self.py], -1)[:, None, :]
        # bullets: per-bullet AABB half + player half
        rel = (bp - pxy).abs()
        r = bh + PLAYER_HB                                       # [B, POOL]
        hit_b = (active & (rel[..., 0] < r) & (rel[..., 1] < r)).any(dim=1)
        # enemy bodies: (hb/2)*2/3 + player half, AABB
        erel = (en[..., :2] - pxy).abs()
        er = en[..., 2] + PLAYER_HB                              # [B, MAX_EN]
        hit_e = (en_a & (erel[..., 0] < er) & (erel[..., 1] < er)).any(dim=1)
        hit = hit_b | hit_e

        ended = (self.t0 + self.t) >= (self.rec_len[self.rec_id.cpu()].to(self.d) - 2)
        done = hit | ended
        rew = torch.full((self.B,), 0.02, device=self.d) - 5.0 * hit.float()
        obs = self._obs()
        if done.any():
            self.reset(done.nonzero(as_tuple=True)[0])
        return obs, rew, done


if __name__ == "__main__":
    import time
    name = sys.argv[1] if len(sys.argv) > 1 else "letty"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = FightSim(B=4096, name=name, device=dev)
    o = s.reset()
    print("obs", o.shape, "range", float(o.min()), float(o.max()))
    t0 = time.perf_counter()
    surv = torch.zeros(s.B, device=s.d)
    for i in range(1200):
        a = torch.randint(0, 36, (s.B,), device=s.d)
        o, rw, dn = s.step(a)
        surv += (~dn).float()
    dt = time.perf_counter() - t0
    print(f"1200 steps B={s.B}: {dt:.1f}s = {1200*s.B/dt/1e3:.0f}k steps/s")
    print(f"random-policy mean survival: {surv.mean()/60:.1f}s")
