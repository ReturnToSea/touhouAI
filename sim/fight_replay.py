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

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import (build_obs_batch as _build_obs_eager, OBS_DIM,  # noqa: E402
                 PX_LO, PX_HI, PY_LO, PY_HI, DIR_SPEED, DIR_SPEED_FOCUS)

# torch.compile fuses build_obs_batch's ~40 tiny kernels (topk, the march-step
# loop + scatter_reduce, the 9-dir escape raycast) - it's ~3/4 of a step and
# entirely launch-overhead bound in eager mode. FIGHTSIM_NOCOMPILE=1 to disable.
if os.environ.get("FIGHTSIM_NOCOMPILE"):
    build_obs_batch = _build_obs_eager
else:
    build_obs_batch = torch.compile(_build_obs_eager, dynamic=False,
                                    mode="max-autotune-no-cudagraphs")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boss_phases import total_hp as _boss_total_hp   # noqa: E402

FIGHTS = Path(__file__).resolve().parent / "fights"
POOL = 1025                      # th07 bullet pool size
MAX_EN = 48                      # max satellite sub-enemies tracked per frame
PLAYER_HB = 1.8                  # measured player half-extent
BULLET_HB_DEFAULT = 2.5          # fallback when a recording has no hitbox column
ENEMY_BODY_SCALE = 2.0 / 3.0     # player-body vs enemy-body box (pytouhou)

# synthetic damage-phasing (ReimuA, homing shot). Numbers are approximate -
# tune SHOT_DPS against real damage-phased fight lengths (Letty ~50-70s).
SHOT_DPS = 14.0                  # HP/frame while the shoot bit is held
DMG_REW = 0.003                  # reward per HP dealt
KILL_BONUS = 150.0               # for defeating the boss (drives "kill, don't just dodge")
_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)


AIM_TOL = np.radians(24.0)          # spawn-vel within this of the rec player -> aimed


def _load_dense(npz_path):
    """npz -> dict of dense [F, POOL(...)] arrays: pos, half, spawn (spawn point
    of the bullet in each slot), aimed (bool), rec_ang (angle spawn->rec player
    at spawn), plus boss[F,2] and en[F,MAX_EN,3]."""
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
    vel = np.zeros((F, POOL, 2), np.float32)
    vel[f, slot, 0] = b[:, 4]
    vel[f, slot, 1] = b[:, 5]

    half = np.full((F, POOL), BULLET_HB_DEFAULT, np.float32)
    if b.shape[1] >= 10:                       # cols 8,9 = AABB full size x,y
        hb = np.maximum(b[:, 8], b[:, 9]) * 0.5
        hb[hb <= 0] = BULLET_HB_DEFAULT
        half[f, slot] = hb

    # --- per-bullet birth + aim classification (for re-aiming at replay) ------
    rp = np.full((F, 2), np.nan, np.float32)
    if "player" in d and len(d["player"]):
        p = d["player"]
        pf = np.clip((p[:, 0] - p[:, 0].min()).astype(int), 0, F - 1)
        rp[pf] = p[:, 1:3]
    for i in range(1, F):
        if np.isnan(rp[i, 0]):
            rp[i] = rp[i - 1]
    rp = np.nan_to_num(rp, nan=192.0)

    present = ~np.isnan(pos[:, :, 0])                       # [F, POOL]
    born = present.copy()
    born[1:] &= ~present[:-1]
    fr_idx = np.arange(F, dtype=np.int64)[:, None]
    birth = np.where(present,
                     np.maximum.accumulate(np.where(born, fr_idx, 0), axis=0),
                     0)                                     # [F, POOL] rec-frame
    bi, si = np.where(born)
    aim_map = np.zeros((F, POOL), bool)
    ang_map = np.zeros((F, POOL), np.float32)
    if len(bi):
        p0 = pos[bi, si]                                    # [K, 2]
        v = vel[bi, si]
        va = np.arctan2(v[:, 1], v[:, 0])
        ta = np.arctan2(rp[bi, 1] - p0[:, 1], rp[bi, 0] - p0[:, 0])
        dd = np.abs((va - ta + np.pi) % (2 * np.pi) - np.pi)
        aim_map[bi, si] = (dd < AIM_TOL) & ((v[:, 0] != 0) | (v[:, 1] != 0))
        ang_map[bi, si] = ta
    aimed = np.take_along_axis(aim_map, birth, axis=0) & present
    rec_ang = np.take_along_axis(ang_map, birth, axis=0)
    spawn = np.stack([np.take_along_axis(pos[:, :, 0], birth, axis=0),
                      np.take_along_axis(pos[:, :, 1], birth, axis=0)], -1)

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
    # Lethal = the small satellite orbs: hitbox x in (0.5, 15]. Excludes the
    # 0x0 Lingering-Cold orbs (harmless) and the big 24-64 boss box (Letty's
    # life field reads 1 early in the fight, so filter on hitbox size not life;
    # point-blanking PCB bosses is legit anyway).
    en = np.full((F, MAX_EN, 3), np.nan, np.float32)
    if "enemies" in d and len(d["enemies"]):
        e = d["enemies"]
        lethal_all = e[(e[:, 5] > 0.5) & (e[:, 5] <= 15.0)]
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
    nb = present.sum(1)
    busy = np.where(nb >= 15)[0]
    s0 = 0
    if len(busy) and busy[0] > 30:
        s0 = max(0, busy[0] - 30)
    return dict(pos=pos[s0:], half=half[s0:], boss=boss[s0:], en=en[s0:],
               spawn=(spawn[s0:] - 0), aimed=aimed[s0:],
               rec_ang=rec_ang[s0:], birth=(birth[s0:] - s0))


class FightSim:
    def __init__(self, B=8192, name="cirno", device="cuda", max_frames=11000,
                 min_start=0, seed=0):
        self.B, self.d = B, device
        self._g = torch.Generator(device="cpu").manual_seed(seed)
        paths = sorted(glob.glob(str(FIGHTS / f"{name}*.npz")))
        recs = [_load_dense(p) for p in paths]
        recs = [r for r in recs if r is not None and r["pos"].shape[0] > 600]
        assert recs, f"no usable recordings matching {name}*"
        self.n_rec = len(recs)
        self.rec_len = torch.tensor([r["pos"].shape[0] for r in recs])
        self.maxF = min(max(r["pos"].shape[0] for r in recs), max_frames)

        nF = (self.n_rec, self.maxF, POOL)
        self.pos = torch.full((*nF, 2), float("nan"))
        self.spawn = torch.full((*nF, 2), float("nan"))
        self.bhalf = torch.full(nF, BULLET_HB_DEFAULT)
        self.aimed = torch.zeros(nF, dtype=torch.bool)
        self.rec_ang = torch.zeros(nF)
        self.birth = torch.full(nF, -1, dtype=torch.int32)
        self.boss = torch.full((self.n_rec, self.maxF, 2), 192.0)
        self.en = torch.full((self.n_rec, self.maxF, MAX_EN, 3), float("nan"))
        for i, r in enumerate(recs):
            L = min(r["pos"].shape[0], self.maxF)
            self.pos[i, :L] = torch.from_numpy(r["pos"][:L])
            self.spawn[i, :L] = torch.from_numpy(r["spawn"][:L])
            self.bhalf[i, :L] = torch.from_numpy(r["half"][:L])
            self.aimed[i, :L] = torch.from_numpy(r["aimed"][:L])
            self.rec_ang[i, :L] = torch.from_numpy(r["rec_ang"][:L])
            self.birth[i, :L] = torch.from_numpy(r["birth"][:L].astype(np.int32))
            self.boss[i, :L] = torch.from_numpy(r["boss"][:L])
            self.en[i, :L] = torch.from_numpy(r["en"][:L])
        for k in ("pos", "spawn", "bhalf", "aimed", "rec_ang", "birth",
                  "boss", "en"):
            setattr(self, k, getattr(self, k).to(device))
        self.rec_len_gpu = self.rec_len.to(device)  # step() reads it without a sync
        self.min_start = min_start
        self.HIST = 320             # re-aim window: how far back we keep the sim
        #                            player's path so an aimed bullet can lock
        #                            onto where the player was AT ITS SPAWN
        self.total_hp = _boss_total_hp(name)       # None -> no damage-phasing
        n_en = int((~torch.isnan(self.en[..., 0])).any(-1).float().mean() * 100)
        n_aim = int(self.aimed.float().mean() * 1000) / 10
        print(f"[FightSim] {self.n_rec} recs, {self.maxF} frames, B={B}, "
              f"enemy bodies ~{n_en}% of frames, aimed bullet-frames ~{n_aim}%, "
              f"boss HP {self.total_hp}", flush=True)
        self._dirs = _DIRS.to(device)
        self.reset()

    def _sample_starts(self, idx):
        rid = torch.randint(0, self.n_rec, (len(idx),), generator=self._g)
        if self.total_hp is not None:
            # damage-phasing on -> start from the top so the whole fight + HP
            # pool is available; the agent shortens it by shooting.
            off = torch.full((len(idx),), self.min_start, dtype=torch.long)
        else:
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
            self.boss_hp = torch.zeros(self.B, device=self.d)
            self.px = torch.full((self.B,), 192.0, device=self.d)
            self.py = torch.full((self.B,), 384.0, device=self.d)
            self.pl_ring = torch.zeros(self.B, self.HIST, 2, device=self.d)
            self.pl_ring[..., 0] = 192.0
            self.pl_ring[..., 1] = 384.0
            self._prev_bp = torch.zeros(self.B, POOL, 2, device=self.d)
            self._prev_active = torch.zeros(self.B, POOL, dtype=torch.bool,
                                            device=self.d)
            self._bi = torch.arange(self.B, device=self.d)
            idx = self._bi
        self._sample_starts(idx.cpu())
        self.t[idx] = 0
        self.px[idx] = 192.0
        self.py[idx] = 384.0
        self.pl_ring[idx] = torch.tensor([192.0, 384.0], device=self.d)
        self._prev_active[idx] = False
        if self.total_hp is not None:
            self.boss_hp[idx] = self.total_hp
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

        # --- re-aim: aimed bullets born during this episode lock onto where the
        # sim player was AT THAT SPAWN, then fly the rest of the recorded path
        # rotated by the angle delta. Bullets already in flight at episode start
        # (birth < t0) keep their recorded aim.
        aim = self.aimed[self.rec_id, f] & active               # [B, POOL]
        birth = self.birth[self.rec_id, f].long()               # [B, POOL]
        bstep = birth - self.t0[:, None]                        # sim step at spawn
        age = self.t[:, None] - bstep
        use = aim & (bstep >= 0) & (age >= 0) & (age < self.HIST)   # [B, POOL]
        ring_i = (bstep.clamp(min=0) % self.HIST)               # [B, POOL]
        pxb = torch.gather(self.pl_ring[..., 0], 1, ring_i)
        pyb = torch.gather(self.pl_ring[..., 1], 1, ring_i)
        sp = self.spawn[self.rec_id, f]                         # [B, POOL, 2]
        new_ang = torch.atan2(pyb - sp[..., 1], pxb - sp[..., 0])
        dth = torch.where(use, new_ang - self.rec_ang[self.rec_id, f],
                          torch.zeros_like(new_ang))
        cs, sn = torch.cos(dth), torch.sin(dth)
        rel = bp - sp
        rx = rel[..., 0] * cs - rel[..., 1] * sn
        ry = rel[..., 0] * sn + rel[..., 1] * cs
        bp = torch.where(use[..., None],
                         torch.stack([sp[..., 0] + rx, sp[..., 1] + ry], -1), bp)
        return bp, active, bh, en, en_active, f

    def _obs(self, now=None):
        bp, active, bh, en, en_a, f = now if now is not None else self._now()
        both = (active & self._prev_active)[..., None]
        bv = torch.where(both, bp - self._prev_bp, torch.zeros_like(bp))
        self._prev_bp, self._prev_active = bp, active
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
        has_en = en_a.any(dim=1)
        ne_dx = torch.where(has_en, dx[self._bi, j], bx[:, 0] - self.px)
        ne_dy = torch.where(has_en, dy[self._bi, j], bx[:, 1] - self.py)
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
        self.pl_ring[self._bi, self.t % self.HIST, 0] = self.px
        self.pl_ring[self._bi, self.t % self.HIST, 1] = self.py

        now = self._now()
        bp, active, bh, en, en_a, f = now
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

        ended = (self.t0 + self.t) >= (self.rec_len_gpu[self.rec_id] - 2)
        rew = torch.full((self.B,), 0.02, device=self.d) - 5.0 * hit.float()

        # --- damage-phasing: shooting while not hugging the bottom drains boss
        # HP (ReimuA homing -> no strict firing-lane check). Killing the boss
        # ends the episode early with a bonus, so a shorter fight = higher return.
        killed = torch.zeros(self.B, dtype=torch.bool, device=self.d)
        if self.total_hp is not None:
            shooting = (act >= 18) & (self.py < 400.0)
            dmg = SHOT_DPS * shooting.float()
            dmg = torch.minimum(dmg, self.boss_hp.clamp(min=0))
            self.boss_hp = self.boss_hp - dmg
            rew = rew + DMG_REW * dmg
            killed = self.boss_hp <= 0
            rew = rew + KILL_BONUS * killed.float()

        done = hit | ended | killed
        self.last_killed = killed          # which done-episodes were boss kills
        obs = self._obs(now)
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
