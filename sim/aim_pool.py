"""Runtime pool for aimed bullets (Part 12).

Aimed shots (`bullet_fan_aimed` / `bullet_circle_aimed` — ~6 % of Letty's
danmaku, mostly Lingering Cold's Sub47) point at the player *when they fire*, so
they can't be baked into the schedule's dense array. `sim.danmaku_ecl` emits
them as a param table carrying the **un-aimed** angle; this pool spawns each one
toward the live policy at its frame and integrates it forward with the same
per-frame update as `sim.ecl.bullet_sim.simulate` (torch, vectorised over the B
episodes x MAX_AIM slots).
"""
from __future__ import annotations

import math

import torch

FX_ACCEL_DIR, FX_TURN_ACCEL = 16, 32
FX_PAUSE_REDIR, FX_PAUSE_AIM = 64, 128
_FX_GRACE_MASK = 0x40 | 0x80 | 0x100 | 0x400 | 0x800
PW, PH = 384.0, 448.0
CULL_M = 12.0
# Lingering Cold's slow/paused aimed bullets (fired ~3px/f, decelerate to 0,
# redirect to ~0.5-1.5) take up to ~640 f to leave the small playfield; 420 was
# truncating them mid-screen near the player.
MAXLIFE = 900
_SPAWN_PER_FRAME = 64             # LC calls Sub44-47 in one frame -> ~60-80 spawns


def _pad(recs, key, dtype=torch.float32):
    S = max((len(r["aimed"][key]) if r.get("aimed") else 0) for r in recs)
    out = torch.zeros(len(recs), max(S, 1), dtype=dtype)
    for i, r in enumerate(recs):
        a = r.get("aimed")
        if a is not None and len(a[key]):
            out[i, :len(a[key])] = torch.as_tensor(a[key], dtype=dtype)
    return out


class AimPool:
    def __init__(self, recs, B, device, max_aim=512):
        # Lingering Cold peaks at ~333 concurrent aimed bullets; 256 overflowed,
        # dropping aimed bursts (the "boss isn't shooting at me" bug) and forcing
        # perpetual spawn backlog. 512 has headroom past LC's peak.
        self.B, self.d, self.MA = B, device, max_aim
        self.active_any = any(r.get("aimed") is not None for r in recs)
        if not self.active_any:
            return
        f = {k: _pad(recs, k).to(device) for k in
             ("frame", "x0", "y0", "speed", "unaim", "hb", "hang_state",
              "hang_frames", "fx_flag", "fx_p1", "fx_p2", "fx_interval",
              "fx_repeat", "launch", "end")}
        self.src = f                                    # [n_rec, S] each
        self.src_n = torch.tensor(
            [len(r["aimed"]["frame"]) if r.get("aimed") else 0 for r in recs],
            device=device)
        self.S = f["frame"].shape[1]
        z = lambda: torch.zeros(B, max_aim, device=device)         # noqa: E731
        self.px, self.py = z(), z()
        self.vx, self.vy = z(), z()
        self.ang, self.spd = z(), z()
        self.accx, self.accy = z(), z()
        self.hb = z()
        self.age = z()
        self.alive = torch.zeros(B, max_aim, dtype=torch.bool, device=device)
        self.fxf, self.fxp1, self.fxp2 = z(), z(), z()
        self.fxiv, self.fxrep, self.fxctr, self.cyc = z(), z(), z(), z()
        self.launch, self.lt = z(), z()
        self.hs, self.hf, self.ratio = z(), z(), z()
        self.end = z()
        self.cur = torch.zeros(B, dtype=torch.long, device=device)   # source cursor
        self.rec_id = torch.zeros(B, dtype=torch.long, device=device)
        self.t0 = torch.zeros(B, dtype=torch.long, device=device)
        self._bi = torch.arange(B, device=device)

    # ------------------------------------------------------------------
    def reset(self, idx, rec_id, t0):
        if not self.active_any:
            return
        self.alive[idx] = False
        self.cur[idx] = 0
        self.rec_id[idx] = rec_id
        self.t0[idx] = t0

    def _spawn(self, fnow, px, py, cs, sn):
        """Vectorised, compile-friendly. The K sources due this frame take the
        K lowest-index free pool slots (a proper free-list, no Python loop)."""
        rid = self.rec_id
        k = torch.arange(_SPAWN_PER_FRAME, device=self.d)          # [K]
        s_idx = self.cur[:, None] + k[None, :]                     # [B, K] source ptr
        sc = s_idx.clamp(max=self.S - 1)
        take = (s_idx < self.src_n[rid][:, None]) & \
               (torch.gather(self.src["frame"][rid], 1, sc) <= fnow[:, None])

        def g(key):
            return torch.gather(self.src[key][rid], 1, sc)         # [B, K]

        x0, y0 = g("x0"), g("y0")
        rx = 192.0 + (x0 - 192.0) * cs[:, None] - (y0 - 192.0) * sn[:, None]
        ry = 192.0 + (x0 - 192.0) * sn[:, None] + (y0 - 192.0) * cs[:, None]
        ang = g("unaim") + torch.atan2(py[:, None] - ry, px[:, None] - rx)
        spd, hs = g("speed"), g("hang_state")
        ratio = torch.where(hs == 2, 0.5, torch.where(
            hs == 3, 0.4, torch.where(hs == 4, 1.0 / 3.0, 1.0)))
        vx0, vy0 = torch.cos(ang) * spd, torch.sin(ang) * spd
        bx = torch.where(hs > 0, rx - 4.0 * vx0, rx)
        by = torch.where(hs > 0, ry - 4.0 * vy0, ry)
        fxf, p2 = g("fx_flag"), g("fx_p2")
        ad = torch.where(p2 <= -990.0, ang, p2)
        accx = torch.where(fxf == FX_ACCEL_DIR, torch.cos(ad) * g("fx_p1"), 0.0)
        accy = torch.where(fxf == FX_ACCEL_DIR, torch.sin(ad) * g("fx_p1"), 0.0)

        # free-list: the j-th spawn goes to the (j+1)-th lowest free slot
        free = ~self.alive                                        # [B, MA]
        frank = free.long().cumsum(1) - 1                         # rank among free
        match = (frank.unsqueeze(1) == k.view(1, -1, 1)) & free.unsqueeze(1)
        tgt = match.float().argmax(2)                             # [B, K]
        take = take & match.any(2)                                # a free slot exists
        self.cur = self.cur + take.long().sum(1)
        w = take.float()

        def put(pool, val):
            keep = torch.gather(pool, 1, tgt)
            return pool.scatter(1, tgt, val * w + keep * (1.0 - w))

        self.px, self.py = put(self.px, bx), put(self.py, by)
        self.vx, self.vy = put(self.vx, vx0), put(self.vy, vy0)
        self.ang, self.spd = put(self.ang, ang), put(self.spd, spd)
        self.accx, self.accy = put(self.accx, accx), put(self.accy, accy)
        self.hb = put(self.hb, g("hb"))
        self.hs, self.hf = put(self.hs, hs), put(self.hf, g("hang_frames"))
        self.ratio = put(self.ratio, ratio)
        self.fxf = put(self.fxf, fxf)
        self.fxp1, self.fxp2 = put(self.fxp1, g("fx_p1")), put(self.fxp2, p2)
        self.fxiv = put(self.fxiv, g("fx_interval"))
        self.fxrep = put(self.fxrep, g("fx_repeat"))
        self.launch, self.end = put(self.launch, g("launch")), put(self.end, g("end"))
        self.age = put(self.age, torch.zeros_like(bx))
        self.fxctr = put(self.fxctr, torch.zeros_like(bx))
        self.cyc = put(self.cyc, torch.zeros_like(bx))
        self.lt = put(self.lt, torch.zeros_like(bx))
        self.alive = self.alive.scatter(1, tgt, (take | torch.gather(
            self.alive, 1, tgt)))

    def _advance(self):
        live = self.alive
        hang = self.hs > 0
        t = self.age
        crawl = hang & (t <= self.hf)
        self.px = self.px + torch.where(crawl, self.vx * self.ratio, 0.0)
        self.py = self.py + torch.where(crawl, self.vy * self.ratio, 0.0)
        sl = live & ~(hang & (t < self.hf))

        m = sl & (self.launch > 0.5) & (self.lt < 17)
        mag = self.spd + 5.0 * (1.0 - self.lt / 16.0)
        self.vx = torch.where(m, torch.cos(self.ang) * mag, self.vx)
        self.vy = torch.where(m, torch.sin(self.ang) * mag, self.vy)
        self.lt = self.lt + m.float()

        act = sl & (self.fxctr < self.fxiv)
        a = act & (self.fxf == FX_ACCEL_DIR)
        self.vx = torch.where(a, self.vx + self.accx, self.vx)
        self.vy = torch.where(a, self.vy + self.accy, self.vy)
        upd = a & ((self.vx.abs() > 1e-4) | (self.vy.abs() > 1e-4))
        self.ang = torch.where(upd, torch.atan2(self.vy, self.vx), self.ang)
        tu = act & (self.fxf == FX_TURN_ACCEL)
        self.ang = torch.where(tu, _wrap(self.ang + self.fxp2), self.ang)
        self.spd = torch.where(tu, self.spd + self.fxp1, self.spd)
        self.vx = torch.where(tu, torch.cos(self.ang) * self.spd, self.vx)
        self.vy = torch.where(tu, torch.sin(self.ang) * self.spd, self.vy)
        self.fxctr = self.fxctr + (a | tu).float()

        # FX_PAUSE_REDIR (Letty's Sub47) — decel to 0 over `interval`, then turn
        # by p1 / set speed p2, repeat. (PAUSE_AIM's player re-aim is unused by
        # Letty; falls through to the same turn here.)
        pr = (self.fxf == FX_PAUSE_REDIR) | (self.fxf == FX_PAUSE_AIM)
        decel = sl & pr & (self.fxctr < self.fxiv)
        arrive = sl & pr & (self.fxctr >= self.fxiv)
        f = 1.0 - torch.where(self.fxiv > 0, self.fxctr / self.fxiv.clamp(min=1), 0.0)
        self.vx = torch.where(decel, torch.cos(self.ang) * self.spd * f, self.vx)
        self.vy = torch.where(decel, torch.sin(self.ang) * self.spd * f, self.vy)
        self.fxctr = self.fxctr + decel.float()
        self.ang = torch.where(arrive, _wrap(self.ang + self.fxp1), self.ang)
        self.spd = torch.where(arrive & (self.fxp2 > -999.0), self.fxp2, self.spd)
        self.vx = torch.where(arrive, torch.cos(self.ang) * self.spd, self.vx)
        self.vy = torch.where(arrive, torch.sin(self.ang) * self.spd, self.vy)
        self.cyc = self.cyc + arrive.float()
        self.fxctr = torch.where(arrive, torch.zeros_like(self.fxctr), self.fxctr)
        self.fxf = torch.where(arrive & (self.cyc >= self.fxrep.clamp(min=1)),
                               torch.zeros_like(self.fxf), self.fxf)

        self.px = self.px + torch.where(sl, self.vx, 0.0)
        self.py = self.py + torch.where(sl, self.vy, 0.0)
        self.age = self.age + live.float()

    def step(self, fnow, px, py, cs, sn):
        """Advance one frame. `cs/sn` are the field-rotation cos/sin [B].
        Returns (pos [B, MA, 2], active [B, MA], half [B, MA], vel [B, MA, 2])."""
        if not self.active_any:
            e = torch.zeros(self.B, 0, device=self.d)
            return (torch.zeros(self.B, 0, 2, device=self.d), e.bool(), e,
                    torch.zeros(self.B, 0, 2, device=self.d))
        prev = torch.stack([self.px, self.py], -1)
        prev_alive = self.alive.clone()
        self._spawn(fnow, px, py, cs, sn)
        self._advance()
        grace_ok = (self.fxf.long() & _FX_GRACE_MASK) != 0
        off = ((self.px < -CULL_M) | (self.px > PW + CULL_M) |
               (self.py < -CULL_M) | (self.py > PH + CULL_M))
        self.alive = self.alive & (self.age < MAXLIFE) & \
            (fnow[:, None] < self.end) & ~(off & ~grace_ok)
        pos = torch.stack([self.px, self.py], -1)
        vel = torch.where((self.alive & prev_alive)[..., None], pos - prev,
                          torch.zeros_like(pos))
        pos = torch.where(self.alive[..., None], pos,
                          torch.full_like(pos, -9999.0))
        return pos, self.alive, self.hb, vel


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
