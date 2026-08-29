"""Fully-vectorised made-up-danmaku environment for GPU training.

B parallel episodes step in lockstep as batched tensors. A FIXED 8-emitter stage
(see ROSTER); only per-episode jitter (fire rate, bullet speed) varies. Player
physics are measured from the real game (sim/physics.json). Observations come
from the SHARED builder (native/obs.py) so a policy sees the same inputs as the
real Th07Env.

Scope: dodging + shooting + P-item collection.
- 3 aimed corner turrets (CONE) + 1 SNAKING turret (bottom-left), all placed
  OUTSIDE the playfield so bullets fly IN; 2 spinning-wheel RINGs at the centre
  (alternating spin); 1 dense emitter that ORBITs the edge + 1 that BOUNCEs the
  interior; a half-width "wall" curtain sweeps across every 10 s.
- waves of 9-14 enemies (1 HP) fly in every 12 s, hover, leave; contact kills.
- SHOOT: no auto-aim - only hits an enemy directly above the player. Kills drop
  P items; collecting P raises power, which scales shot damage (0.4x -> 2x,
  capped at 150 items) and widens it (1 -> 3 targets).

    from sim.danmaku import DanmakuSim
    sim = DanmakuSim(B=8192, device="cuda")
    obs = sim.reset()                        # [B, OBS_DIM]
    obs, rew, done = sim.step(actions)       # actions [B] long, auto-resets
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import build_obs_batch, OBS_DIM, PX_LO, PX_HI, PY_LO, PY_HI  # noqa: E402
from obs import W as PW, H as PH, M_ITEMS  # noqa: E402

TAU = 2 * math.pi
_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)
SPEED_FAST, SPEED_SLOW = 4.0, 1.6
CX, CY = 192.0, 224.0                     # stage centre

# emitter behaviours
E_OFF, E_CONE, E_SNAKE, E_RING, E_ORBIT, E_BOUNCE = 0, 1, 2, 3, 4, 5
#  CONE   : tight fan aimed at the centre (one turret per corner, OUTSIDE the
#           playfield so bullets fly IN and the corner isn't a safe pocket)
#  SNAKE  : aimed bullets that weave side-to-side as they travel
#  RING   : concentric spinning "wheels" - bullets on a fixed-radius circle, the
#           whole ring rotating. 2 of them, alternating spin.
#  ORBIT  : a dense-ring emitter that circles the perimeter (anti-camp)
#  BOUNCE : a dense-ring emitter that bounces around the interior (anti-camp)

ORBIT_RX, ORBIT_RY, ORBIT_W = 170.0, 196.0, 0.0105   # ~10 s per lap

ROSTER = [
    (E_CONE,   -14.0,  -14.0),   # 0  top-left      (outside the playfield)
    (E_CONE,   398.0,  -14.0),   # 1  top-right     (keeps the t=1s redirect)
    (E_SNAKE,  -14.0,  462.0),   # 2  bottom-left   (snaking shots)
    (E_CONE,   398.0,  462.0),   # 3  bottom-right
    (E_RING,    CX,     CY),     # 4  inner wheel
    (E_RING,    CX,     CY),     # 5  outer wheel
    (E_ORBIT,   CX,     CY),     # 6  orbits the perimeter
    (E_BOUNCE,  CX,     CY),     # 7  bounces the interior
]
_RING_EIDX = (4, 5)

BULLET_SCALE = 1.0          # v19: reverted the v16 -35% (it hurt transfer)

# SNAKE: lateral heading weave on emitter-2 bullets
SNAKE_AMP = 0.55            # radians of heading swing
SNAKE_FREQ = 0.13           # rad/frame
SNAKE_SPEED = 2.4

# RING: 2 concentric spinning "wheels" - bullets on a fixed-radius circle, the
# whole ring rotating around the centre. Alternating spin direction.
# per-ring: (radius, angular velocity rad/frame signed, spoke count)
RING_RADIUS = (46.0, 120.0)
RING_OMEGA = (0.034, -0.022)
RING_SPOKES = (4, 7)
RING_PERIOD = 140          # refresh the wheel (old spokes expire ~RING_LIFE)
RING_LIFE = 170.0

# ORBIT + BOUNCE ("BRING"): restored to the committed dense-ring params;
# projectile speed bumped 0.24 -> 0.85.
BRING_SPEED = 0.85         # dense ring crawls OUTWARD from the moving emitter
BRING_PERIOD = 33
BRING_NSPAWN = 9
BRING_RAD = 3.2
BRING_DANG = 0.06          # ring pattern rotates each shot -> spiral
BOUNCE_ROAM = 1.0          # emitter roam speed for the bouncer
BRING_LIFE = 300.0         # 5 s cap for ORBIT/BOUNCE + redirecting-cone bullets

# --- enemies ---
MAXE = 36
EN_PER_WAVE = 14
EN_WAVE_LO, EN_WAVE_HI = 9, 15
WAVE_PERIOD = 720           # 12 s
EN_HP = 1.0
EN_RADIUS = 9.0
EN_FLY_SPEED = 2.6
EN_HOVER_FRAMES = 360       # 6 s
EN_DPS = 1.0 / 45.0        # base dmg/frame (before power mult / 30-70 split)
EN_DMG_REW = 0.10

# --- shot model: NO auto-aim - the shot only hits an enemy roughly DIRECTLY
# ABOVE the player. Teaches "get under the target". ---
SHOOT_ALIGN_DX = 26.0

# --- power -> shot: 150 P items to cap; capped bonus is 2x damage ---
POWER_MAX = 150.0
PWR_PER_ITEM = 1.0
PWR_START_HI = 20.0
PWR_DMG_MULT_LO = 0.4
PWR_DMG_MULT_MAX = 2.0

# --- P items ---
IT_MAX = 192
IT_PER_KILL = 4
IT_GRAVITY = 0.10
IT_TERM_VY = 3.0
IT_COLLECT_R = 14.0
IT_REW = 0.30

# --- "wall" attack: a half-width curtain sweeps across (gentler than v16-18) ---
WALL_PERIOD = 600          # 10 s
WALL_FIRST = 100           # first wall at t~1.7 s (player has time to orient)
WALL_SLOTS = 96
WALL_N = 40
WALL_SPEED = 1.0
WALL_RAD = 4.0
WALL_OFF = 48.0            # spawn off-screen (telegraph before it's lethal)

# top-right CONE (ROSTER idx 1): one-shot 50% chance at t=1 s to redirect 360 deg
TR_CONE_EIDX = 1
TR_REDIR_AGE = 60.0


class DanmakuSim:
    def __init__(self, B=16384, device="cuda", slots_per_emitter=200, spawn_k=16,
                 max_frames=5400, frame_skip=3, alive_rew=0.01, death_rew=-1.0,
                 seed=0, compile=True):
        self.B = B
        self.dev = torch.device(device)
        self.E = len(ROSTER)
        self.SPE = slots_per_emitter
        self.K = spawn_k
        self._wall_base = self.E * slots_per_emitter
        self.N = self._wall_base + WALL_SLOTS + 1          # +1 = dump slot
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
        self.b_age = torch.zeros(B, self.N, device=d)
        self.b_ang0 = torch.zeros(B, self.N, device=d)     # heading at spawn (SNAKE)
        self.b_redir = torch.zeros(B, self.N, device=d)    # 1 once the t=1s roll is done
        self.b_omega = torch.zeros(B, self.N, device=d)    # per-frame vel rotation (RING wheels)
        self.cursor = torch.zeros(B, E, device=d)

        self.e_pos = torch.zeros(B, E, 2, device=d)
        self.e_oa = torch.zeros(B, E, device=d)            # ORBIT angle
        self.e_speed = torch.zeros(B, E, device=d)
        self.e_ang = torch.zeros(B, E, device=d)           # RING wheel phase
        self.e_dang = torch.zeros(B, E, device=d)          # RING wheel omega (per frame)
        self.e_period = torch.ones(B, E, device=d) * 30
        self.e_phase = torch.zeros(B, E, device=d)
        self.e_nspawn = torch.ones(B, E, device=d) * 5
        self.e_spread = torch.zeros(B, E, device=d)        # CONE per-bullet step
        self.e_rad = torch.ones(B, E, device=d) * 3.0
        self.e_ringr = torch.zeros(B, E, device=d)         # RING wheel radius (0 else)
        self.e_bvel = torch.zeros(B, E, 2, device=d)       # roam velocity (BOUNCE only)

        # enemies
        self.en_pos = torch.full((B, MAXE, 2), 1e4, device=d)
        self.en_vel = torch.zeros(B, MAXE, 2, device=d)
        self.en_tgt = torch.zeros(B, MAXE, 2, device=d)
        self.en_hp = torch.zeros(B, MAXE, device=d)
        self.en_phase = torch.zeros(B, MAXE, device=d)     # 0 fly-in, 1 hover, 2 leave
        self.en_timer = torch.zeros(B, MAXE, device=d)
        self.en_active = torch.zeros(B, MAXE, device=d)
        self.en_cursor = torch.zeros(B, device=d)
        self._ek = torch.arange(EN_PER_WAVE, device=d).float()

        # P items + power
        self.it_pos = torch.full((B, IT_MAX, 2), 1e4, device=d)
        self.it_vel = torch.zeros(B, IT_MAX, 2, device=d)
        self.it_active = torch.zeros(B, IT_MAX, device=d)
        self.it_cursor = torch.zeros(B, device=d)
        self.power = torch.zeros(B, 1, device=d)

        # wall attack
        self.wall_cursor = torch.zeros(B, device=d)
        self._walln = torch.arange(WALL_N, device=d).float()

        # death-cause diagnostic ("wall" = any bullet, "enemy" = enemy body)
        self.death_wall = torch.zeros(B, device=d)
        self.death_enemy = torch.zeros(B, device=d)
        self.step_death_wall = torch.zeros(B, device=d)
        self.step_death_enemy = torch.zeros(B, device=d)

        rt = torch.tensor(ROSTER, device=d, dtype=torch.float32)
        self._R_type = rt[:, 0]
        self._R_xy = rt[:, 1:3]
        self._eidx = torch.arange(E, device=d)
        self._is_orbit = (self._R_type == E_ORBIT).float()          # [E]

        raw_se = torch.arange(self.N, device=d) // self.SPE
        slot_emit = raw_se.clamp(max=E - 1)
        _emit_t = self._R_type[slot_emit]
        in_e = raw_se < E
        self._slot_life = torch.full((self.N,), 1e9, device=d)
        self._slot_life = torch.where(
            in_e & ((_emit_t == E_ORBIT) | (_emit_t == E_BOUNCE) | (raw_se == TR_CONE_EIDX)),
            torch.full((self.N,), BRING_LIFE, device=d), self._slot_life)
        self._slot_life = torch.where(in_e & (_emit_t == E_RING),
                                      torch.full((self.N,), RING_LIFE, device=d),
                                      self._slot_life)
        self._snake_slot = (raw_se == 2)                            # [N] bool
        self._tr_cone_slot = (raw_se == TR_CONE_EIDX)               # [N] bool
        self._ring_slot = in_e & (_emit_t == E_RING)                # [N] bool
        self._wall_slot = ((torch.arange(self.N, device=d) >= self._wall_base) &
                           (torch.arange(self.N, device=d) < self.dump))

        self._k = torch.arange(self.K, device=d).float()
        self._ebase = self._eidx.float() * self.SPE
        self._head = torch.zeros(B, 9, device=d)
        self._head[:, 5] = 1.0
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
    @torch._dynamo.disable
    def _r(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(shape, generator=self.g, device=self.dev) * (hi - lo) + lo

    @torch._dynamo.disable
    def _ri(self, lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=self.g, device=self.dev).float()

    # ------------------------------------------------------------------ (re)spawn
    def _spawn(self, m):
        B, E, d = self.B, self.E, self.dev
        mb = m > 0.5
        mbe = mb.expand(B, E)
        is_cone = (self._R_type == E_CONE)[None, :].expand(B, E)
        is_snake = (self._R_type == E_SNAKE)[None, :].expand(B, E)
        is_ring = (self._R_type == E_RING)[None, :].expand(B, E)
        is_orbit = (self._R_type == E_ORBIT)[None, :].expand(B, E)
        is_bounce = (self._R_type == E_BOUNCE)[None, :].expand(B, E)
        is_bring = is_orbit | is_bounce

        # per-ring wheel params (radius / omega / spokes), keyed by _RING_EIDX
        r_om = torch.zeros(E, device=d)
        r_nsp = torch.zeros(E, device=d)
        r_rad = torch.zeros(E, device=d)
        for j, ei in enumerate(_RING_EIDX):
            r_om[ei] = RING_OMEGA[j]
            r_nsp[ei] = RING_SPOKES[j]
            r_rad[ei] = RING_RADIUS[j]

        px = self._r(B, 1, lo=PX_LO + 50, hi=PX_HI - 50)
        py = self._r(B, 1, lo=PY_HI - 140, hi=PY_HI - 30)
        self.player = torch.where(mb, torch.cat([px, py], 1), self.player)
        self.frame = torch.where(mb, torch.zeros_like(self.frame), self.frame)
        self.alive = torch.where(mb, torch.ones_like(self.alive), self.alive)
        diff = 0.2 + 0.8 * self._r(B, 1)
        self.diff = torch.where(mb, diff, self.diff)
        dsc = 0.85 + 0.35 * diff                                    # speed mult ~0.9..1.2

        self.b_active = torch.where(mb, torch.zeros_like(self.b_active), self.b_active)
        self.cursor = torch.where(mb, torch.zeros_like(self.cursor), self.cursor)

        self.e_pos = torch.where(mbe[:, :, None], self._R_xy.expand(B, E, 2), self.e_pos)
        # bouncer: random interior start + fixed-speed random heading
        bstart = torch.stack([self._r(B, E, lo=CX - 60, hi=CX + 60),
                              self._r(B, E, lo=CY - 60, hi=CY + 60)], -1)
        self.e_pos = torch.where((mbe & is_bounce)[:, :, None], bstart, self.e_pos)
        bang = self._r(B, E, lo=0, hi=TAU)
        bvel = torch.stack([torch.cos(bang), torch.sin(bang)], -1) * BOUNCE_ROAM
        self.e_bvel = torch.where((mbe & is_bounce)[:, :, None], bvel,
                      torch.where(mbe[:, :, None], torch.zeros_like(self.e_bvel), self.e_bvel))
        self.e_oa = torch.where(mbe & is_orbit, self._r(B, E, lo=0, hi=TAU), self.e_oa)

        speed = torch.where(is_snake, torch.full((B, E), SNAKE_SPEED, device=d) * dsc,
                torch.where(is_bring, torch.full((B, E), BRING_SPEED, device=d),
                            self._r(B, E, lo=2.0, hi=2.9) * dsc))             # CONE
        self.e_speed = torch.where(mbe, speed, self.e_speed)     # rings use e_ringr*|omega|

        # SNAKE fires the same fan/rate as the CONE turrets - it just snakes.
        period = torch.where(is_ring, torch.full((B, E), float(RING_PERIOD), device=d),
                 torch.where(is_bring, torch.full((B, E), float(BRING_PERIOD), device=d),
                             self._ri(28, 46, B, E)))                       # CONE + SNAKE
        self.e_period = torch.where(mbe, period, self.e_period)
        # rings fire from frame 0 (no lead-in); cone/snake/bring stagger
        ph = torch.where(is_ring, torch.zeros(B, E, device=d),
                         torch.floor(self._r(B, E) * period))
        self.e_phase = torch.where(mbe, ph, self.e_phase)

        nsp = torch.where(is_ring, r_nsp[None, :],
              torch.where(is_bring, torch.full((B, E), float(BRING_NSPAWN), device=d),
                          self._ri(4, 7, B, E)))                            # CONE + SNAKE
        self.e_nspawn = torch.where(mbe, nsp, self.e_nspawn)

        spread = self._r(B, E, lo=0.10, hi=0.20)                             # CONE fan step
        self.e_spread = torch.where(mbe, spread, self.e_spread)
        rad = self._r(B, E, lo=2.6, hi=4.2) * BULLET_SCALE
        rad = torch.where(is_ring, self._r(B, E, lo=2.4, hi=3.2) * BULLET_SCALE, rad)
        rad = torch.where(is_bring,
                          torch.full((B, E), BRING_RAD * BULLET_SCALE, device=d), rad)
        self.e_rad = torch.where(mbe, rad, self.e_rad)
        self.e_ang = torch.where(mbe, self._r(B, E, lo=0, hi=TAU), self.e_ang)
        self.e_ringr = torch.where(mbe, r_rad[None, :].expand(B, E), self.e_ringr)
        r_om = torch.where(is_bring[0], torch.full_like(r_om, BRING_DANG), r_om)
        self.e_dang = torch.where(mbe, r_om[None, :].expand(B, E), self.e_dang)

        self.en_active = torch.where(mb, torch.zeros_like(self.en_active), self.en_active)
        mb1 = mb.squeeze(1)
        self.en_cursor = torch.where(mb1, torch.zeros_like(self.en_cursor), self.en_cursor)
        self.it_active = torch.where(mb, torch.zeros_like(self.it_active), self.it_active)
        self.it_cursor = torch.where(mb1, torch.zeros_like(self.it_cursor), self.it_cursor)
        self.wall_cursor = torch.where(mb1, torch.zeros_like(self.wall_cursor), self.wall_cursor)
        self.power = torch.where(mb, self._r(B, 1, lo=0.0, hi=PWR_START_HI), self.power)

    def reset(self):
        self._spawn(torch.ones(self.B, 1, device=self.dev))
        return self._obs()

    def _enemy_obs(self):
        B = self.B
        vis = (self.en_active > 0.5) & (self.en_pos[..., 1] > -8.0) & \
              (self.en_pos[..., 1] < PY_HI + 32.0)
        ed = torch.where(vis, (self.en_pos - self.player[:, None, :]).norm(dim=2),
                         torch.full((B, MAXE), 1e9, device=self.dev))
        dk, ik = ed.topk(6, dim=1, largest=False)
        rel = torch.gather(self.en_pos, 1, ik[:, :, None].expand(-1, -1, 2)) \
            - self.player[:, None, :]
        hpn = torch.gather(self.en_hp, 1, ik) / EN_HP
        v = (dk < 1e8).float()
        o = torch.zeros(B, 18, device=self.dev)
        o[:, 0::3] = rel[..., 0] / 128.0 * v
        o[:, 1::3] = rel[..., 1] / 128.0 * v
        o[:, 2::3] = hpn.clamp(0.0, 1.0) * v
        return o

    def _item_obs(self):
        B = self.B
        ia = self.it_active > 0.5
        idd = torch.where(ia, (self.it_pos - self.player[:, None, :]).norm(dim=2),
                          torch.full((B, IT_MAX), 1e9, device=self.dev))
        dk, ik = idd.topk(M_ITEMS, dim=1, largest=False)
        rel = torch.gather(self.it_pos, 1, ik[:, :, None].expand(-1, -1, 2)) \
            - self.player[:, None, :]
        v = (dk < 1e8).float()
        o = torch.zeros(B, M_ITEMS * 3, device=self.dev)
        o[:, 0::3] = rel[..., 0] / 128.0 * v
        o[:, 1::3] = rel[..., 1] / 128.0 * v
        o[:, 2::3] = 0.0
        return o

    def _obs(self):
        head = self._head.clone()
        head[:, 2] = (self.power[:, 0] / POWER_MAX).clamp(0.0, 1.0)
        return self._obs_fn(self.player, self._zeros2, self._zeros1,
                            self.b_pos, self.b_vel, self.b_active,
                            head, self._enemy_obs(), self._item_obs())

    # ------------------------------------------------------------------ hot path
    def _advance(self, mv, focus, shoot):
        B, E, K = self.B, self.E, self.K
        d = self.dev

        # --- player ---
        norm = mv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        spd = torch.where(focus > 0.5, torch.full_like(focus, SPEED_SLOW),
                          torch.full_like(focus, SPEED_FAST))
        moving = (mv.abs().sum(1, keepdim=True) > 0).float()
        self.player = self.player + moving * mv / norm * spd
        self.player = torch.stack([self.player[:, 0].clamp(PX_LO, PX_HI),
                                   self.player[:, 1].clamp(PY_LO, PY_HI)], 1)

        # --- move bullets, rotate RING-wheel velocities, apply SNAKE weave, cull ---
        self.b_pos = self.b_pos + self.b_vel
        self.b_age = self.b_age + 1.0
        oc, os_ = torch.cos(self.b_omega), torch.sin(self.b_omega)   # per-frame vel spin
        vx, vy = self.b_vel[..., 0], self.b_vel[..., 1]
        self.b_vel = torch.stack([vx * oc - vy * os_, vx * os_ + vy * oc], -1)
        snake_m = self._snake_slot[None, :] & (self.b_active > 0.5)
        # per-bullet phase from b_ang0 so the fan's bullets weave out of sync
        weave = self.b_ang0 + SNAKE_AMP * torch.sin(self.b_age * SNAKE_FREQ
                                                    + self.b_ang0 * 4.0)
        sv = torch.stack([torch.cos(weave), torch.sin(weave)], -1) * SNAKE_SPEED
        self.b_vel = torch.where(snake_m[:, :, None], sv, self.b_vel)
        m = torch.where(self._wall_slot[None, :], 64.0, 18.0)   # wall gets a wide margin
        on = ((self.b_pos[..., 0] > -m) & (self.b_pos[..., 0] < PW + m) &
              (self.b_pos[..., 1] > -m) & (self.b_pos[..., 1] < PH + m) &
              (self.b_age < self._slot_life))
        self.b_active = self.b_active * on.float()

        # --- BOUNCE roams + bounces the interior; ORBIT rides the edge ellipse ---
        is_bounce_col = (self._R_type == E_BOUNCE)[None, :]           # [1,E]
        self.e_pos = self.e_pos + self.e_bvel                         # 0 except the bouncer
        ex, ey = self.e_pos[..., 0], self.e_pos[..., 1]
        fx = (((ex < PX_LO + 10) & (self.e_bvel[..., 0] < 0)) |
              ((ex > PX_HI - 10) & (self.e_bvel[..., 0] > 0)))
        fy = (((ey < PY_LO + 10) & (self.e_bvel[..., 1] < 0)) |
              ((ey > PY_HI - 10) & (self.e_bvel[..., 1] > 0)))
        self.e_bvel = self.e_bvel * torch.stack(
            [torch.where(fx, -1.0, 1.0), torch.where(fy, -1.0, 1.0)], -1)
        clamped = torch.stack([ex.clamp(PX_LO + 10, PX_HI - 10),
                               ey.clamp(PY_LO + 10, PY_HI - 10)], -1)
        self.e_pos = torch.where(is_bounce_col[:, :, None], clamped, self.e_pos)
        self.e_oa = self.e_oa + ORBIT_W * self._is_orbit[None, :]
        opos = torch.stack([CX + ORBIT_RX * torch.cos(self.e_oa),
                            CY + ORBIT_RY * torch.sin(self.e_oa)], -1)
        self.e_pos = torch.where(self._is_orbit[None, :, None] > 0.5, opos, self.e_pos)
        self.e_ang = self.e_ang + self.e_dang           # RING wheels + BRING pattern spin

        # --- emitters fire ---
        fr = self.frame                                          # [B,1]
        due = ((fr % self.e_period) == self.e_phase) & (self._R_type[None, :] > 0)
        kk = self._k[None, None, :]                              # [1,1,K]
        nsp = self.e_nspawn[:, :, None]                          # [B,E,1]
        emit = due[:, :, None] & (kk < nsp)                      # [B,E,K]

        epos = self.e_pos
        spd_e = self.e_speed[:, :, None]
        rr = self.e_ringr[:, :, None]                            # [B,E,1]
        om = self.e_dang[:, :, None]                             # [B,E,1]
        is_ring_e = (self._R_type == E_RING)[None, :, None, None]  # [1,E,1,1]

        is_orbit_e = ((self._R_type == E_ORBIT) |
                      (self._R_type == E_BOUNCE))[None, :, None, None]   # [1,E,1,1]
        spr = self.e_spread[:, :, None]
        base_aim = torch.atan2(CY - epos[:, :, 1:2], CX - epos[:, :, 0:1])   # [B,E,1]
        cone = base_aim + (kk - (nsp - 1) * 0.5) * spr          # CONE / SNAKE aim
        spoke = self.e_ang[:, :, None] + TAU * kk / nsp.clamp(min=1)   # ring/orbit angle

        cone_dir = torch.stack([torch.cos(cone), torch.sin(cone)], -1)   # [B,E,K,2]
        rad_dir = torch.stack([torch.cos(spoke), torch.sin(spoke)], -1)  # radial out
        tan_dir = torch.stack([-torch.sin(spoke), torch.cos(spoke)], -1)  # tangent (wheel)

        cone_pos = epos[:, :, None, :].expand(B, E, K, 2)
        ring_pos = epos[:, :, None, :] + rr[..., None] * rad_dir
        cpos = torch.where(is_ring_e, ring_pos, cone_pos)

        cone_vel = spd_e[..., None] * cone_dir                  # CONE / SNAKE
        orbit_vel = spd_e[..., None] * rad_dir                  # ORBIT: dense ring OUT
        ring_vel = (om[..., None] * rr[..., None]) * tan_dir    # RING wheel: tangential
        cvel = torch.where(is_ring_e, ring_vel,
                           torch.where(is_orbit_e, orbit_vel, cone_vel))
        b_om_fired = torch.where(is_ring_e[..., 0], om.expand(B, E, K),
                                 torch.zeros(B, E, K, device=d))
        crad = self.e_rad[:, :, None].expand(B, E, K)

        raw = self._ebase[None, :, None] + (self.cursor[:, :, None] + kk) % self.SPE
        idx = torch.where(emit, raw, torch.full_like(raw, float(self.dump))).long()
        idxf = idx.reshape(B, E * K)
        self.b_pos = self.b_pos.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cpos.reshape(B, E * K, 2))
        self.b_vel = self.b_vel.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cvel.reshape(B, E * K, 2))
        self.b_rad = self.b_rad.scatter(1, idxf, crad.reshape(B, E * K))
        self.b_ang0 = self.b_ang0.scatter(1, idxf, cone.expand(B, E, K).reshape(B, E * K))
        self.b_omega = self.b_omega.scatter(1, idxf, b_om_fired.reshape(B, E * K))
        self.b_active = self.b_active.scatter(1, idxf, emit.reshape(B, E * K).float())
        self.b_age = self.b_age.scatter(1, idxf, torch.zeros(B, E * K, device=d))
        self.b_redir = self.b_redir.scatter(1, idxf, torch.zeros(B, E * K, device=d))
        self.b_active[:, self.dump] = 0.0
        self.cursor = torch.where(due, (self.cursor + self.e_nspawn) % self.SPE, self.cursor)

        # --- top-right CONE: one-shot 50% redirect at t=1 s (360 deg) ---
        frs = self.frame.squeeze(1)                              # [B]
        tr_due = (self._tr_cone_slot[None, :] & (self.b_age >= TR_REDIR_AGE) &
                  (self.b_redir < 0.5) & (self.b_active > 0.5))
        roll = torch.rand(B, self.N, generator=self.g, device=d) < 0.5
        do_rd = tr_due & roll
        rang = torch.rand(B, self.N, generator=self.g, device=d) * TAU
        rspd = self.b_vel.norm(dim=2, keepdim=True)
        rvel = torch.stack([torch.cos(rang), torch.sin(rang)], -1) * rspd
        self.b_vel = torch.where(do_rd[:, :, None], rvel, self.b_vel)
        self.b_redir = torch.where(tr_due, torch.ones_like(self.b_redir), self.b_redir)

        # --- enemies: waves fly in, hover, leave ---
        wave_due = ((frs % WAVE_PERIOD) < 1.0) & (frs > 1.0)     # [B]
        NW = EN_PER_WAVE
        ek = self._ek[None, :]
        n_wave = self._ri(EN_WAVE_LO, EN_WAVE_HI, B, 1)
        mkw = wave_due[:, None] & (ek < n_wave)
        edge = self._ri(0, 3, B, NW)
        tx = self._r(B, NW, lo=PX_LO + 30, hi=PX_HI - 30)
        sy = self._r(B, NW, lo=PY_LO + 20, hi=PY_LO + 170)
        spx = torch.where(edge == 0, tx, torch.where(edge == 1,
              torch.full_like(tx, -28.0), torch.full_like(tx, PW + 28.0)))
        spy = torch.where(edge == 0, torch.full_like(sy, -28.0), sy)
        spos = torch.stack([spx, spy], -1)
        stgt = torch.stack([self._r(B, NW, lo=PX_LO + 40, hi=PX_HI - 40),
                            self._r(B, NW, lo=PY_LO + 40, hi=CY + 50)], -1)
        v0 = stgt - spos
        v0 = v0 / v0.norm(dim=2, keepdim=True).clamp(min=1e-3) * EN_FLY_SPEED
        slot = ((self.en_cursor[:, None] + ek) % MAXE).long()
        s2 = slot[:, :, None].expand(B, NW, 2)
        wd = wave_due[:, None]
        wd2 = wd[:, :, None]
        self.en_pos = torch.where(wd2, self.en_pos.scatter(1, s2, spos), self.en_pos)
        self.en_tgt = torch.where(wd2, self.en_tgt.scatter(1, s2, stgt), self.en_tgt)
        self.en_vel = torch.where(wd2, self.en_vel.scatter(1, s2, v0), self.en_vel)
        self.en_hp = torch.where(wd, self.en_hp.scatter(
            1, slot, torch.full((B, NW), EN_HP, device=d)), self.en_hp)
        self.en_phase = torch.where(wd, self.en_phase.scatter(
            1, slot, torch.zeros(B, NW, device=d)), self.en_phase)
        self.en_active = torch.where(wd, self.en_active.scatter(
            1, slot, mkw.float()), self.en_active)
        self.en_cursor = torch.where(wave_due, (self.en_cursor + NW) % MAXE, self.en_cursor)

        ea = self.en_active > 0.5
        to_t = self.en_tgt - self.en_pos
        dt = to_t.norm(dim=2)
        arrived = ea & (self.en_phase < 0.5) & (dt < 6.0)
        self.en_phase = torch.where(arrived, torch.ones_like(self.en_phase), self.en_phase)
        self.en_timer = torch.where(arrived, torch.full_like(self.en_timer,
                                    float(EN_HOVER_FRAMES)), self.en_timer)
        hov = ea & (self.en_phase > 0.5) & (self.en_phase < 1.5)
        self.en_timer = torch.where(hov, self.en_timer - 1.0, self.en_timer)
        self.en_phase = torch.where(hov & (self.en_timer <= 0.0),
                                    torch.full_like(self.en_phase, 2.0), self.en_phase)
        v_fly = torch.where(dt[:, :, None] > 1e-3,
                            to_t / dt[:, :, None].clamp(min=1e-3) * EN_FLY_SPEED,
                            torch.zeros_like(to_t))
        aw = self.en_pos - torch.tensor([CX, CY], device=d)
        v_leave = torch.stack([torch.sign(aw[..., 0]) * 0.9,
                               torch.full_like(aw[..., 1], EN_FLY_SPEED)], -1)
        evel = torch.where(self.en_phase[:, :, None] < 0.5, v_fly,
               torch.where(self.en_phase[:, :, None] < 1.5,
                           torch.zeros_like(v_fly), v_leave))
        self.en_pos = self.en_pos + evel * ea[:, :, None].float()
        eoff = ((self.en_pos[..., 0] < -44) | (self.en_pos[..., 0] > PW + 44) |
                (self.en_pos[..., 1] < -44) | (self.en_pos[..., 1] > PH + 44))
        self.en_active = self.en_active * (~eoff).float() * (self.en_hp > 0.0).float()

        # --- SHOOT: no auto-aim - only hits an enemy roughly DIRECTLY ABOVE ---
        ea = self.en_active > 0.5
        on_screen = ((self.en_pos[..., 0] > PX_LO) & (self.en_pos[..., 0] < PX_HI) &
                     (self.en_pos[..., 1] > 0.0) & (self.en_pos[..., 1] < PY_HI))
        rel = self.en_pos - self.player[:, None, :]              # [B,MAXE,2]
        dist = rel.norm(dim=2)
        shootable = ea & on_screen
        pfrac = (self.power[:, 0] / POWER_MAX).clamp(0.0, 1.0)   # [B]
        pmult = PWR_DMG_MULT_LO + (PWR_DMG_MULT_MAX - PWR_DMG_MULT_LO) * pfrac
        n_tgt = 1.0 + (pfrac > 0.45).float() + (pfrac > 0.8).float()   # [B]
        sh = shoot.squeeze(1)                                    # [B]
        NT = 3
        ari = torch.arange(NT, device=d)[None, :]

        # only enemies directly above the player (|rel_x| < ALIGN_DX, above)
        aligned = shootable & (rel[..., 0].abs() < SHOOT_ALIGN_DX) & (rel[..., 1] < 0.0)
        d_al = torch.where(aligned, dist, torch.full_like(dist, 1e9))
        dk_l, ik_l = d_al.topk(NT, dim=1, largest=False)
        ok_l = (dk_l < 1e8) & (ari < n_tgt[:, None])
        per_l = (sh[:, None] * EN_DPS * pmult[:, None]) * ok_l.float()

        hp_l0 = self.en_hp.gather(1, ik_l).clamp(min=0.0)
        self.en_hp = self.en_hp.scatter_add(1, ik_l, -torch.minimum(per_l, hp_l0))
        hp_l1 = self.en_hp.gather(1, ik_l)

        killed = (hp_l0 > 0.0) & (hp_l1 <= 0.0)                 # [B,NT]
        dmg = torch.minimum(per_l, hp_l0).sum(1)               # [B]
        kpos = torch.gather(self.en_pos, 1, ik_l[:, :, None].expand(-1, -1, 2))  # [B,NT,2]

        # --- P items: IT_PER_KILL per killed enemy, varied pop, fall, collect ---
        KM = NT
        NJ = IT_PER_KILL * KM
        jj = torch.arange(NJ, device=d)
        kslot = (jj // IT_PER_KILL)[None, :]                    # [1,NJ]
        islot = ((self.it_cursor[:, None] + jj[None, :]) % IT_MAX).long()
        is2 = islot[:, :, None].expand(B, NJ, 2)
        jpos = torch.gather(kpos, 1, kslot[:, :, None].expand(B, NJ, 2))
        pvy = self._r(B, NJ, lo=-3.6, hi=-1.0)
        pvx = self._r(B, NJ, lo=-1.9, hi=1.9)
        jvel = torch.stack([pvx, pvy], -1)
        jm = torch.gather(killed.float(), 1, kslot.expand(B, NJ)) > 0.5
        cur_ip = torch.gather(self.it_pos, 1, is2)
        cur_iv = torch.gather(self.it_vel, 1, is2)
        cur_ia = torch.gather(self.it_active, 1, islot)
        self.it_pos = self.it_pos.scatter(1, is2, torch.where(jm[:, :, None], jpos, cur_ip))
        self.it_vel = self.it_vel.scatter(1, is2, torch.where(jm[:, :, None], jvel, cur_iv))
        self.it_active = self.it_active.scatter(
            1, islot, torch.where(jm, torch.ones_like(cur_ia), cur_ia))
        self.it_cursor = (self.it_cursor + IT_PER_KILL * killed.float().sum(1)) % IT_MAX

        ia = self.it_active > 0.5
        nv = self.it_vel.clone()
        nv[..., 0] = nv[..., 0] * 0.90                          # sideways pop settles
        nv[..., 1] = (nv[..., 1] + IT_GRAVITY).clamp(max=IT_TERM_VY)
        self.it_vel = torch.where(ia[:, :, None], nv, torch.zeros_like(self.it_vel))
        self.it_pos = self.it_pos + self.it_vel * ia[:, :, None].float()
        idist = (self.it_pos - self.player[:, None, :]).norm(dim=2)
        got = ia & (idist < IT_COLLECT_R)
        n_got = got.sum(dim=1).float()
        it_off = self.it_pos[..., 1] > PY_HI + 24.0
        self.it_active = self.it_active * (~got).float() * (~it_off).float()
        self.power = (self.power + PWR_PER_ITEM * n_got[:, None]).clamp(0.0, POWER_MAX)

        # --- wall attack: a half-width curtain sweeps across every WALL_PERIOD ---
        is_first = (frs == float(WALL_FIRST))[:, None]           # [B,1]
        wall_due = ((((frs % WALL_PERIOD) < 1.0) & (frs > 1.0))[:, None] | is_first)
        wall_due = wall_due.squeeze(1)                           # [B]
        wk = self._walln[None, :]                                # [1,WALL_N]
        wdir = self._ri(0, 4, B, 1)                              # 0 L>R 1 R>L 2 T>B 3 B>T
        wdir = torch.where(is_first, wdir % 2, wdir)             # first: vertical curtain
        half = torch.where(wdir <= 1.5, torch.full_like(wdir, PH * 0.5),
                           torch.full_like(wdir, PW * 0.5))      # [B,1]
        gap_first = (self._ri(0, 2, B, 1) > 0.5) & ~is_first
        seg0 = torch.where(gap_first, half, torch.zeros_like(half))
        perp = seg0 + (wk / (WALL_N - 1)) * half                 # [B,WALL_N]
        woff = WALL_OFF
        wx = torch.where(wdir <= 1.5,
                         torch.where(wdir < 0.5, torch.full_like(perp, -woff),
                                     torch.full_like(perp, PW + woff)), perp)
        wy = torch.where(wdir >= 1.5,
                         torch.where(wdir < 2.5, torch.full_like(perp, -woff),
                                     torch.full_like(perp, PH + woff)), perp)
        z = torch.zeros_like(perp)
        wvx = torch.where(wdir < 0.5, torch.full_like(perp, WALL_SPEED),
              torch.where(wdir < 1.5, torch.full_like(perp, -WALL_SPEED), z))
        wvy = torch.where((wdir > 1.5) & (wdir < 2.5), torch.full_like(perp, WALL_SPEED),
              torch.where(wdir > 2.5, torch.full_like(perp, -WALL_SPEED), z))
        wslot = (self._wall_base + (self.wall_cursor[:, None] + wk) % WALL_SLOTS).long()
        ws2 = wslot[:, :, None].expand(B, WALL_N, 2)
        wm = wall_due[:, None].expand(B, WALL_N)                 # [B,WALL_N]
        wpos = torch.stack([wx, wy], -1)
        wvel = torch.stack([wvx, wvy], -1)
        cur_wp = torch.gather(self.b_pos, 1, ws2)
        cur_wv = torch.gather(self.b_vel, 1, ws2)
        cur_wa = torch.gather(self.b_active, 1, wslot)
        cur_wr = torch.gather(self.b_rad, 1, wslot)
        cur_wg = torch.gather(self.b_age, 1, wslot)
        self.b_pos = self.b_pos.scatter(1, ws2, torch.where(wm[:, :, None], wpos, cur_wp))
        self.b_vel = self.b_vel.scatter(1, ws2, torch.where(wm[:, :, None], wvel, cur_wv))
        self.b_active = self.b_active.scatter(
            1, wslot, torch.where(wm, torch.ones_like(cur_wa), cur_wa))
        self.b_rad = self.b_rad.scatter(
            1, wslot, torch.where(wm, torch.full_like(cur_wr, WALL_RAD), cur_wr))
        self.b_age = self.b_age.scatter(
            1, wslot, torch.where(wm, torch.zeros_like(cur_wg), cur_wg))
        self.wall_cursor = torch.where(wall_due,
                                       (self.wall_cursor + WALL_N) % WALL_SLOTS,
                                       self.wall_cursor)

        # --- collision ---
        bd = (self.b_pos - self.player[:, None, :]).norm(dim=2)
        bhit = ((self.b_active > 0.5) & (bd < self.b_rad + 2.0)).any(dim=1, keepdim=True)
        en_d = (self.en_pos - self.player[:, None, :]).norm(dim=2)
        en_hit = (ea & (en_d < EN_RADIUS + 3.0)).any(dim=1, keepdim=True)
        hit = bhit | en_hit
        newly_dead = (self.alive > 0.5) & hit
        self.death_wall = (newly_dead & bhit & ~en_hit).squeeze(1).float()   # emitter fire
        self.death_enemy = (newly_dead & en_hit).squeeze(1).float()
        self.alive = self.alive * (~hit).float()
        self.frame = self.frame + 1.0
        done = newly_dead | (self.frame >= self.max_frames)
        alive_now = self.alive > 0.5
        rew = torch.where(newly_dead, torch.full_like(self.alive, self.death_rew),
              torch.where(alive_now, torch.full_like(self.alive, self.alive_rew),
                          torch.zeros_like(self.alive)))
        rew = rew + (EN_DMG_REW * dmg + IT_REW * n_got)[:, None] * alive_now.float()
        return rew.squeeze(1), done.squeeze(1)

    def step(self, actions):
        a = actions.long()
        mv = _DIRS.to(self.dev)[a % 9]
        focus = ((a // 9) % 2).float()[:, None]
        shoot = ((a // 18) % 2).float()[:, None]
        rew_acc = torch.zeros(self.B, device=self.dev)
        done_acc = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        dw_acc = torch.zeros(self.B, device=self.dev)
        de_acc = torch.zeros(self.B, device=self.dev)
        for _ in range(self.frame_skip):
            rew, done = self._advance_c(mv, focus, shoot)
            newly = done.bool() & (~done_acc)
            dw_acc = dw_acc + (self.death_wall > 0.5) * newly.float()
            de_acc = de_acc + (self.death_enemy > 0.5) * newly.float()
            rew_acc = rew_acc + rew * (~done_acc).float()
            done_acc = done_acc | done.bool()
        self.step_death_wall = dw_acc
        self.step_death_enemy = de_acc
        if done_acc.any():
            self._spawn(done_acc[:, None].float())
        return self._obs(), rew_acc, done_acc


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for B in ([8192, 16384] if dev == "cuda" else [512]):
        sim = DanmakuSim(B=B, device=dev, max_frames=5400, compile=(dev == "cuda"))
        obs = sim.reset()
        rng = torch.Generator(device=dev).manual_seed(1)
        for _ in range(20):
            sim.step(torch.randint(0, 36, (B,), generator=rng, device=dev))
        if dev == "cuda":
            torch.cuda.synchronize()
        t0, n = time.perf_counter(), 300
        for _ in range(n):
            sim.step(torch.randint(0, 36, (B,), generator=rng, device=dev))
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"B={B:6d}: {n*B*3/dt:>12,.0f} env-steps/s   obs {tuple(obs.shape)}")
