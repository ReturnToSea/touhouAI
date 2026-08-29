"""Phase 1: a fully-vectorised made-up-danmaku environment for GPU training.

B parallel episodes step in lockstep as batched tensors. A FIXED 5-emitter
stage (see ROSTER); only the tunable params (fire rate, bullet speed, sweep
rate, difficulty) jitter per episode. Player physics are the ones measured
from the real game (sim/physics.json). Observations are built by the SHARED
builder (native/obs.py) so a policy trained here sees bit-identical inputs to
the real Th07Env.

Scope: pure dodging. No enemies to shoot, no boss, no items.

    from sim.danmaku import DanmakuSim
    sim = DanmakuSim(B=8192, device="cuda")
    obs = sim.reset()                        # [B, 212]
    obs, rew, done = sim.step(actions)       # actions [B] long, auto-resets
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import build_obs_batch, OBS_DIM, PX_LO, PX_HI, PY_LO, PY_HI  # noqa: E402
from obs import W as PW, H as PH  # noqa: E402

TAU = 2 * math.pi
_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)
SPEED_FAST, SPEED_SLOW = 4.0, 1.6
CX, CY = 192.0, 224.0                     # stage centre

# emitter behaviours
E_OFF, E_CONE, E_LINE, E_BRING, E_SPRAY = 0, 1, 2, 3, 4
#  CONE  : tight fan of bullets aimed at the centre
#  SPRAY : burst of random-angle bullets fanned wide toward the centre
#  LINE  : one bullet per shot, fires FAST, aim sweeps back and forth -> a line
#  BRING : dense ring. last emitter ORBITS the perimeter; the one before it
#          bounces around the interior.

# orbit path for the last BRING emitter (an ellipse hugging the playfield edge)
ORBIT_RX, ORBIT_RY, ORBIT_W = 170.0, 196.0, 0.0105   # ~10 s per lap

# fixed stage: every corner gets a CONE + a SPRAY (placed IN the corner so
# there's no safe pocket behind them); a fast sweeping LINE bottom-right;
# one bouncing dense-ring emitter + one that orbits the perimeter.
_CORNERS = [(20.0, 26.0), (364.0, 26.0), (20.0, 410.0), (364.0, 410.0)]
ROSTER = []
for _cx, _cy in _CORNERS:
    ROSTER += [(E_CONE, _cx, _cy), (E_SPRAY, _cx, _cy)]
ROSTER.append((E_LINE, 350.0, 412.0))
ROSTER += [(E_BRING, CX, CY), (E_BRING, CX, CY)]     # [-2] bounces, [-1] orbits


class DanmakuSim:
    def __init__(self, B=16384, device="cuda", slots_per_emitter=176, spawn_k=24,
                 max_frames=5400, frame_skip=3, alive_rew=0.01, death_rew=-1.0,
                 seed=0, compile=True):
        self.B = B
        self.dev = torch.device(device)
        self.E = len(ROSTER)
        self.SPE = slots_per_emitter
        self.K = spawn_k
        self.N = self.E * slots_per_emitter + 1            # +1 = dump slot
        self.dump = self.N - 1
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.alive_rew = alive_rew
        self.death_rew = death_rew
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        d, E = self.dev, self.E

        self.player = torch.zeros(B, 2, device=d)
        self.frame = torch.zeros(B, 1, device=d)
        self.alive = torch.ones(B, 1, device=d)
        self.diff = torch.zeros(B, 1, device=d)

        self.b_pos = torch.full((B, self.N, 2), 1e4, device=d)
        self.b_vel = torch.zeros(B, self.N, 2, device=d)
        self.b_active = torch.zeros(B, self.N, device=d)
        self.b_rad = torch.zeros(B, self.N, device=d)
        self.b_age = torch.zeros(B, self.N, device=d)      # frames since spawn
        self.cursor = torch.zeros(B, E, device=d)

        self.e_pos = torch.zeros(B, E, 2, device=d)
        self.e_bvel = torch.zeros(B, E, 2, device=d)       # bounce velocity (BRING[-2])
        self.e_oa = torch.zeros(B, E, device=d)            # orbit angle (BRING[-1])
        self.e_speed = torch.zeros(B, E, device=d)
        self.e_ang = torch.zeros(B, E, device=d)           # BRING ring phase
        self.e_dang = torch.zeros(B, E, device=d)          # BRING ring spin
        self.e_period = torch.ones(B, E, device=d) * 30
        self.e_phase = torch.zeros(B, E, device=d)
        self.e_nspawn = torch.ones(B, E, device=d) * 8
        self.e_spread = torch.zeros(B, E, device=d)        # CONE per-bullet fan
        self.e_rad = torch.ones(B, E, device=d) * 3.0
        self.e_swctr = torch.zeros(B, E, device=d)         # LINE sweep centre / amp / rate
        self.e_swamp = torch.zeros(B, E, device=d)
        self.e_swrate = torch.zeros(B, E, device=d)
        self.e_swphase = torch.zeros(B, E, device=d)

        rt = torch.tensor(ROSTER, device=d, dtype=torch.float32)
        self._R_type = rt[:, 0]
        self._R_xy = rt[:, 1:3]
        self._eidx = torch.arange(E, device=d)
        self._is_orbit = (self._eidx == E - 1).float()     # last BRING orbits the edge

        # per-slot bullet lifetime: the moving (BRING) emitters fire very slow
        # bullets, so cap them at 5 s (300 f) or the screen just fills up.
        slot_emit = (torch.arange(self.N, device=d) // self.SPE).clamp(max=E - 1)
        self._slot_life = torch.where(self._R_type[slot_emit] == E_BRING,
                                      torch.full((self.N,), 300.0, device=d),
                                      torch.full((self.N,), 1e9, device=d))

        self._k = torch.arange(self.K, device=d).float()
        self._ebase = self._eidx.float() * self.SPE
        self._z9 = torch.zeros(B, 9, device=d)
        self._z9[:, 5] = 1.0
        self._z18 = torch.zeros(B, 18, device=d)
        self._zeros2 = torch.zeros(B, 2, device=d)
        self._zeros1 = torch.zeros(B, device=d)

        self._obs_fn = build_obs_batch
        self._spawn(torch.ones(B, 1, device=d))
        if compile and self.dev.type == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
                self._advance_c = torch.compile(self._advance, dynamic=False)
                self._obs_fn = torch.compile(build_obs_batch, dynamic=False)
            except Exception as e:
                print(f"[sim] torch.compile off ({e})")
                self._advance_c = self._advance
        else:
            self._advance_c = self._advance

    # ------------------------------------------------------------------ rng
    def _r(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(shape, generator=self.g, device=self.dev) * (hi - lo) + lo

    def _ri(self, lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=self.g, device=self.dev).float()

    # ------------------------------------------------------------------ (re)spawn
    def _spawn(self, m):
        """Re-roll the envs where m[b] > 0.5 (dense: compute for all B, blend)."""
        B, E, d = self.B, self.E, self.dev
        mb = m > 0.5
        mbe = mb.expand(B, E)
        is_line = (self._R_type == E_LINE)[None, :].expand(B, E)     # [B,E]
        is_bring = (self._R_type == E_BRING)[None, :].expand(B, E)
        is_cone = (self._R_type == E_CONE)[None, :].expand(B, E)
        is_spray = (self._R_type == E_SPRAY)[None, :].expand(B, E)
        is_orbit = (self._eidx == E - 1)[None, :].expand(B, E)       # last BRING orbits
        is_bounce = is_bring & ~is_orbit

        # spawn lower-centre (like real Touhou)
        px = self._r(B, 1, lo=PX_LO + 50, hi=PX_HI - 50)
        py = self._r(B, 1, lo=PY_HI - 140, hi=PY_HI - 30)
        self.player = torch.where(mb, torch.cat([px, py], 1), self.player)
        self.frame = torch.where(mb, torch.zeros_like(self.frame), self.frame)
        self.alive = torch.where(mb, torch.ones_like(self.alive), self.alive)
        # per-episode bullet-speed variation (NOT a curriculum - the stage is
        # always the full stage; this just keeps episodes from being identical)
        diff = 0.2 + 0.8 * self._r(B, 1)
        self.diff = torch.where(mb, diff, self.diff)
        dsc = 0.75 + 0.55 * diff                                     # bullet-speed mult ~0.9..1.2

        self.b_active = torch.where(mb, torch.zeros_like(self.b_active), self.b_active)
        self.cursor = torch.where(mb, torch.zeros_like(self.cursor), self.cursor)

        # static emitters: roster positions. bouncing one: random interior start
        # + fixed-speed random heading. orbiting one: random start angle.
        self.e_pos = torch.where(mbe[:, :, None], self._R_xy.expand(B, E, 2), self.e_pos)
        bstart = torch.stack([self._r(B, E, lo=CX - 60, hi=CX + 60),
                              self._r(B, E, lo=CY - 60, hi=CY + 60)], -1)
        self.e_pos = torch.where((mbe & is_bounce)[:, :, None], bstart, self.e_pos)
        bang = self._r(B, E, lo=0, hi=TAU)
        bvel = torch.stack([torch.cos(bang), torch.sin(bang)], -1)   # fixed speed 1.0
        self.e_bvel = torch.where((mbe & is_bounce)[:, :, None], bvel,
                                  torch.where(mb[:, :, None], torch.zeros_like(self.e_bvel),
                                              self.e_bvel))
        self.e_oa = torch.where(mbe & is_orbit, self._r(B, E, lo=0, hi=TAU), self.e_oa)

        # per-type tunables. BRING (the moving emitters) are FULLY FIXED:
        # bullet speed 0.24 (~1/3 of before), period 33 (~half the fire rate),
        # 9 shots, radius 3.2 - no per-episode variation.
        speed = torch.where(is_line, self._r(B, E, lo=3.0, hi=4.0) * dsc,
                torch.where(is_bring, torch.full((B, E), 0.24, device=d),
                torch.where(is_spray, self._r(B, E, lo=1.8, hi=2.8) * dsc,
                            self._r(B, E, lo=2.1, hi=3.0) * dsc)))
        self.e_speed = torch.where(mbe, speed, self.e_speed)

        period = torch.where(is_line, self._ri(5, 9, B, E),
                 torch.where(is_bring, torch.full((B, E), 33.0, device=d),
                 torch.where(is_spray, self._ri(26, 44, B, E),
                             self._ri(28, 48, B, E))))
        self.e_period = torch.where(mbe, period, self.e_period)
        self.e_phase = torch.where(mbe, torch.floor(self._r(B, E) * period), self.e_phase)

        nsp = torch.where(is_line, torch.ones(B, E, device=d),
              torch.where(is_bring, torch.full((B, E), 9.0, device=d),
              torch.where(is_spray, self._ri(4, 8, B, E),
                          self._ri(3, 6, B, E))))
        self.e_nspawn = torch.where(mbe, nsp, self.e_nspawn)

        # e_spread doubles as the CONE per-bullet step and the SPRAY total arc
        spread = torch.where(is_spray, self._r(B, E, lo=1.6, hi=2.5),
                             self._r(B, E, lo=0.09, hi=0.20))
        self.e_spread = torch.where(mbe, spread, self.e_spread)
        rad = torch.where(is_bring, torch.full((B, E), 3.2, device=d),
                          self._r(B, E, lo=2.6, hi=4.4))
        self.e_rad = torch.where(mbe, rad, self.e_rad)
        self.e_ang = torch.where(mbe, self._r(B, E, lo=0, hi=TAU), self.e_ang)
        dang = torch.where(is_bring, torch.full((B, E), 0.06, device=d),
                           self._r(B, E, lo=-0.25, hi=0.25))
        self.e_dang = torch.where(mbe, dang, self.e_dang)

        # LINE sweep: aim oscillates between "up toward top-right" and "left
        # toward bottom-left"  (centre ~ -135deg, amplitude ~ +-50deg)
        self.e_swctr = torch.where(mbe, torch.full((B, E), -2.36, device=d), self.e_swctr)
        self.e_swamp = torch.where(mbe, self._r(B, E, lo=0.75, hi=1.05), self.e_swamp)
        self.e_swrate = torch.where(mbe, self._r(B, E, lo=0.035, hi=0.075) * (0.7 + diff),
                                    self.e_swrate)
        self.e_swphase = torch.where(mbe, self._r(B, E, lo=0, hi=TAU), self.e_swphase)

    def reset(self):
        self._spawn(torch.ones(self.B, 1, device=self.dev))
        return self._obs()

    def _obs(self):
        return self._obs_fn(self.player, self._zeros2, self._zeros1,
                            self.b_pos, self.b_vel, self.b_active,
                            self._z9, self._z18)

    # ------------------------------------------------------------------ hot path
    def _advance(self, mv, focus):
        B, E, K = self.B, self.E, self.K
        # --- player ---
        norm = mv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        spd = torch.where(focus > 0.5, torch.full_like(focus, SPEED_SLOW),
                          torch.full_like(focus, SPEED_FAST))
        moving = (mv.abs().sum(1, keepdim=True) > 0).float()
        self.player = self.player + moving * mv / norm * spd
        self.player = torch.stack([self.player[:, 0].clamp(PX_LO, PX_HI),
                                   self.player[:, 1].clamp(PY_LO, PY_HI)], 1)

        # --- move bullets, cull off-screen + past lifetime ---
        self.b_pos = self.b_pos + self.b_vel
        self.b_age = self.b_age + 1.0
        on = ((self.b_pos[..., 0] > -18) & (self.b_pos[..., 0] < PW + 18) &
              (self.b_pos[..., 1] > -18) & (self.b_pos[..., 1] < PH + 18) &
              (self.b_age < self._slot_life))
        self.b_active = self.b_active * on.float()

        # --- moving emitters: [-2] bounces the interior, [-1] orbits the edge ---
        self.e_pos = self.e_pos + self.e_bvel               # e_bvel is 0 except the bouncer
        ex, ey = self.e_pos[..., 0], self.e_pos[..., 1]
        fx = ((ex < PX_LO + 10) & (self.e_bvel[..., 0] < 0)) | \
             ((ex > PX_HI - 10) & (self.e_bvel[..., 0] > 0))
        fy = ((ey < PY_LO + 10) & (self.e_bvel[..., 1] < 0)) | \
             ((ey > PY_HI - 10) & (self.e_bvel[..., 1] > 0))
        self.e_bvel = self.e_bvel * torch.stack(
            [torch.where(fx, -1.0, 1.0), torch.where(fy, -1.0, 1.0)], -1)
        self.e_pos = torch.stack([ex.clamp(PX_LO + 10, PX_HI - 10),
                                  ey.clamp(PY_LO + 10, PY_HI - 10)], -1)
        self.e_oa = self.e_oa + ORBIT_W * self._is_orbit
        opos = torch.stack([CX + ORBIT_RX * torch.cos(self.e_oa),
                            CY + ORBIT_RY * torch.sin(self.e_oa)], -1)
        self.e_pos = torch.where(self._is_orbit[None, :, None] > 0.5, opos, self.e_pos)

        # --- emitters fire ---
        fr = self.frame                                     # [B,1]
        due = ((fr % self.e_period) == self.e_phase) & (self._R_type[None, :] > 0)
        kk = self._k[None, None, :]                         # [1,1,K]
        nsp = self.e_nspawn[:, :, None]                     # [B,E,1]
        emit = due[:, :, None] & (kk < nsp)                 # [B,E,K]

        epos = self.e_pos
        spd_e = self.e_speed[:, :, None]
        typ = self._R_type[None, :, None]                   # [1,E,1]

        spr = self.e_spread[:, :, None]
        base_aim = torch.atan2(CY - epos[:, :, 1:2], CX - epos[:, :, 0:1])   # [B,E,1]
        cone = base_aim + (kk - (nsp - 1) * 0.5) * spr
        rnd = torch.rand(B, E, K, generator=self.g, device=self.dev)
        spray = base_aim + (rnd - 0.5) * spr                    # spr = wide arc here
        swang = self.e_swctr[:, :, None] + self.e_swamp[:, :, None] * torch.sin(
            fr[:, :, None] * self.e_swrate[:, :, None] + self.e_swphase[:, :, None])
        line = swang + 0.0 * kk
        bring = self.e_ang[:, :, None] + TAU * kk / nsp.clamp(min=1)
        ang = torch.where(typ == E_CONE, cone,
              torch.where(typ == E_SPRAY, spray,
              torch.where(typ == E_LINE, line, bring)))

        cpos = epos[:, :, None, :].expand(B, E, K, 2)
        cvel = torch.stack([spd_e * torch.cos(ang), spd_e * torch.sin(ang)], -1)
        crad = self.e_rad[:, :, None].expand(B, E, K)

        raw = self._ebase[None, :, None] + (self.cursor[:, :, None] + kk) % self.SPE
        idx = torch.where(emit, raw, torch.full_like(raw, float(self.dump))).long()
        idxf = idx.reshape(B, E * K)
        self.b_pos = self.b_pos.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cpos.reshape(B, E * K, 2))
        self.b_vel = self.b_vel.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cvel.reshape(B, E * K, 2))
        self.b_rad = self.b_rad.scatter(1, idxf, crad.reshape(B, E * K))
        self.b_active = self.b_active.scatter(1, idxf, emit.reshape(B, E * K).float())
        self.b_age = self.b_age.scatter(1, idxf, torch.zeros(B, E * K, device=self.dev))
        self.b_active[:, self.dump] = 0.0

        self.cursor = torch.where(due, (self.cursor + self.e_nspawn) % self.SPE, self.cursor)
        self.e_ang = self.e_ang + due.float() * self.e_dang

        # --- collision ---
        dist = (self.b_pos - self.player[:, None, :]).norm(dim=2)
        hit = ((self.b_active > 0.5) & (dist < self.b_rad + 2.0)).any(dim=1, keepdim=True)
        newly_dead = (self.alive > 0.5) & hit
        self.alive = self.alive * (~hit).float()
        self.frame = self.frame + 1.0
        done = newly_dead | (self.frame >= self.max_frames)
        alive_now = self.alive > 0.5
        rew = torch.where(newly_dead, torch.full_like(self.alive, self.death_rew),
              torch.where(alive_now, torch.full_like(self.alive, self.alive_rew),
                          torch.zeros_like(self.alive)))
        return rew.squeeze(1), done.squeeze(1)

    def step(self, actions):
        a = actions.long()
        mv = _DIRS.to(self.dev)[a % 9]
        focus = ((a // 9) % 2).float()[:, None]
        rew_acc = torch.zeros(self.B, device=self.dev)
        done_acc = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        for _ in range(self.frame_skip):
            rew, done = self._advance_c(mv, focus)
            rew_acc = rew_acc + rew * (~done_acc).float()
            done_acc = done_acc | done.bool()
        if done_acc.any():
            self._spawn(done_acc[:, None].float())
        return self._obs(), rew_acc, done_acc


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for B in ([8192, 16384] if dev == "cuda" else [512]):
        sim = DanmakuSim(B=B, device=dev, max_frames=5400)
        obs = sim.reset()
        rng = torch.Generator(device=dev).manual_seed(1)
        for _ in range(15):
            sim.step(torch.randint(0, 36, (B,), generator=rng, device=dev))
        if dev == "cuda":
            torch.cuda.synchronize()
        t0, n, dead = time.perf_counter(), 300, 0
        for _ in range(n):
            _, rew, done = sim.step(torch.randint(0, 36, (B,), generator=rng, device=dev))
            dead += int((rew < 0).sum())
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"B={B:6d}: {n*B/dt:>12,.0f} env-steps/s   deaths={dead}  "
              f"rand-survival ~{n*B/max(dead,1)*3/60:.1f}s  "
              f"obs=[{obs.min():.2f},{obs.max():.2f}]")
