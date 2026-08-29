"""A fully-vectorised made-up-danmaku environment for GPU training.

B parallel episodes step in lockstep as batched tensors. A FIXED stage (see
ROSTER: CONE+SPRAY in every corner, a sweeping LINE, a bouncing + an orbiting
dense-ring emitter); only the tunable params (fire rate, bullet speed, sweep
rate, difficulty) jitter per episode. Bullets use the real th07 hitboxes
(TH07_BULLETS, from native/probe_bullets.py). Player physics are measured from
the real game (sim/physics.json). Observations come from the SHARED builder
(native/obs.py) so a policy sees bit-identical inputs to the real Th07Env.

Scope: dodging + shooting + P-item collection. Waves of 9-15 enemies (1 HP)
fly in every 12 s, hover ~6 s, leave; body contact kills. Holding SHOOT
auto-hits the nearest 1-3 on-screen enemies (count grows with power); kills
drop P items that raise a power meter (-> more shot damage).

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
from obs import W as PW, H as PH, M_ITEMS  # noqa: E402

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

# --- real th07 bullet types (native/probe_bullets.py; memory ref-th07-bullet-hitboxes)
# th07 collision is an AABB overlap, box = pos +- hitbox. We approximate it
# circular: a bullet kills when  dist < hitbox + PLAYER_HB.
#   hitbox = box half-extent (playfield px);  draw = viz radius (bigger, so the
#   rendered bullet looks like the game - the sprite is much larger than the hitbox)
PLAYER_HB = 1.8
TH07_BULLETS = {
    "pellet": dict(hitbox=2.0, draw=4.5),   # small round - stage-1 popcorn / rice
    "ball":   dict(hitbox=3.0, draw=7.5),   # medium round - aimed shots, Letty orbs
}

# fixed stage: every corner gets a CONE + a SPRAY (placed IN the corner so
# there's no safe pocket behind them); a fast sweeping LINE bottom-right;
# one bouncing dense-ring emitter + one that orbits the perimeter.
# each entry: (behaviour, x, y, bullet-type-name)
_CORNERS = [(20.0, 26.0), (364.0, 26.0), (20.0, 410.0), (364.0, 410.0)]
ROSTER = []
for _cx, _cy in _CORNERS:
    ROSTER += [(E_CONE, _cx, _cy, "ball"), (E_SPRAY, _cx, _cy, "ball")]
ROSTER.append((E_LINE, 350.0, 412.0, "ball"))
ROSTER += [(E_BRING, CX, CY, "ball"), (E_BRING, CX, CY, "ball")]  # [-2] bounces, [-1] orbits

# --- enemies (learn to shoot) ---
# v16: big waves fly in from off-screen every 12 s, hover ~6 s, leave. 1 HP each;
# touching one kills the player. Holding SHOOT auto-hits the nearest N active
# enemies (N grows with power) for EN_DPS * power_mult. Enemies drop P items;
# collecting P raises power -> more damage (capped 3x). The damage/power model
# doesn't transfer literally but the "collect P, hold shoot, dodge bodies" habit
# does.
MAXE = 36                    # enemy slots
EN_PER_WAVE = 14             # slots written per wave (9-14 activated)
EN_WAVE_LO, EN_WAVE_HI = 9, 15
WAVE_PERIOD = 720            # 12 s
EN_HP = 1.0
EN_RADIUS = 9.0
EN_FLY_SPEED = 2.6
EN_HOVER_FRAMES = 360        # 6 s
EN_DPS = 1.0 / 45.0          # base dmg/frame at power 0 (1 HP -> 0.75 s to kill)
EN_DMG_REW = 0.10           # reward per HP dealt

# --- power meter -> shot damage (0..POWER_MAX, +PWR_PER_ITEM per P collected) ---
POWER_MAX = 128.0
PWR_PER_ITEM = 2.5
PWR_START_HI = 48.0          # episodes start at power ~U(0, PWR_START_HI)
PWR_DMG_MULT_MAX = 3.0       # damage at full power = 3x

# --- P-item drops ---
IT_MAX = 192                 # item slots (4/kill * big waves)
IT_PER_KILL = 4              # P items dropped per enemy killed
IT_GRAVITY = 0.10            # px/frame^2 downward
IT_TERM_VY = 3.0             # terminal fall speed
IT_COLLECT_R = 14.0
IT_REW = 0.30              # reward per P item collected (v16: 0.15 -> 0.30)

# top-right CONE (ROSTER emitter index 2): one-shot 50% chance at t=1s to
# redirect anywhere in 360 deg, same speed. Those bullets get a 5 s life cap.
TR_CONE_EIDX = 2
TR_REDIR_AGE = 60.0
TR_CONE_LIFE = 300.0


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

        # P-item drops + power meter
        self.it_pos = torch.full((B, IT_MAX, 2), 1e4, device=d)
        self.it_vel = torch.zeros(B, IT_MAX, 2, device=d)
        self.it_active = torch.zeros(B, IT_MAX, device=d)
        self.it_cursor = torch.zeros(B, device=d)
        self._itk = torch.arange(IT_PER_KILL, device=d).float()
        self.power = torch.zeros(B, 1, device=d)              # 0..POWER_MAX

        # death-cause diagnostic (set each frame an env dies): bullet vs enemy body
        self.death_wall = torch.zeros(B, device=d)            # "wall" = any bullet now
        self.death_enemy = torch.zeros(B, device=d)

        self._R_type = torch.tensor([r[0] for r in ROSTER], device=d, dtype=torch.float32)
        self._R_xy = torch.tensor([[r[1], r[2]] for r in ROSTER], device=d, dtype=torch.float32)
        self._R_hitbox = torch.tensor([TH07_BULLETS[r[3]]["hitbox"] for r in ROSTER],
                                      device=d, dtype=torch.float32)          # [E]
        self._eidx = torch.arange(E, device=d)
        self._is_orbit = (self._eidx == E - 1).float()     # last BRING orbits the edge

        # per-slot bullet lifetime: BRING (slow moving emitters) and the
        # redirecting top-right CONE get a 5 s (300 f) cap or the screen fills.
        slot_emit = torch.arange(self.N, device=d) // self.SPE
        _capped = ((slot_emit < E) &
                   ((self._R_type[slot_emit.clamp(max=E - 1)] == E_BRING) |
                    (slot_emit == TR_CONE_EIDX)))
        self._slot_life = torch.where(_capped,
                                      torch.full((self.N,), 300.0, device=d),
                                      torch.full((self.N,), 1e9, device=d))
        self._tr_cone_slot = (slot_emit == TR_CONE_EIDX)
        self.b_redir = torch.zeros(B, self.N, device=d)     # 1 once the t=1s roll is done

        self._k = torch.arange(self.K, device=d).float()
        self._ebase = self._eidx.float() * self.SPE
        # head_aux [B,9] = lives/9, bombs/9, power/128, tanh(graze/100), stage/6,
        # alive, dead, boss_present, boss_frac. Only power + alive vary in the sim.
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
    # NOT traced by torch.compile: dynamo guards on the literal lo/hi values, so
    # the ~30 distinct (lo, hi, shape) call sites would each force a recompile.
    # Run eager, hand back a plain tensor; the rest of _advance still compiles.
    @torch._dynamo.disable
    def _r(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(shape, generator=self.g, device=self.dev) * (hi - lo) + lo

    @torch._dynamo.disable
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
        # bullet radius = the real th07 hitbox for this emitter's bullet type
        self.e_rad = torch.where(mbe, self._R_hitbox[None, :].expand(B, E), self.e_rad)
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

        self.en_active = torch.where(mb, torch.zeros_like(self.en_active), self.en_active)
        self.en_cursor = torch.where(mb.squeeze(1), torch.zeros_like(self.en_cursor),
                                     self.en_cursor)
        self.it_active = torch.where(mb, torch.zeros_like(self.it_active), self.it_active)
        mb1 = mb.squeeze(1)
        self.it_cursor = torch.where(mb1, torch.zeros_like(self.it_cursor), self.it_cursor)
        self.power = torch.where(mb, self._r(B, 1, lo=0.0, hi=PWR_START_HI), self.power)

    def reset(self):
        self._spawn(torch.ones(self.B, 1, device=self.dev))
        return self._obs()

    def _enemy_obs(self):
        """[B,18] = 6 nearest active enemies: (rel_x/128, rel_y/128, hp/2)."""
        B = self.B
        # match env.py: only enemies that have entered the playfield are visible
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
        """[B, M_ITEMS*3] = nearest active P items: (rel_x/128, rel_y/128, 0)."""
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
        o[:, 2::3] = 0.0                         # all items are P in the sim
        return o

    def _obs(self):
        head = self._head.clone()
        head[:, 2] = (self.power[:, 0] / 128.0).clamp(0.0, 1.0)
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
        self.b_redir = self.b_redir.scatter(1, idxf, torch.zeros(B, E * K, device=self.dev))
        self.b_active[:, self.dump] = 0.0

        self.cursor = torch.where(due, (self.cursor + self.e_nspawn) % self.SPE, self.cursor)
        self.e_ang = self.e_ang + due.float() * self.e_dang

        # --- top-right CONE: one-shot 50% redirect at t=1s (360 deg) ---
        tr_due = (self._tr_cone_slot[None, :] & (self.b_age >= TR_REDIR_AGE) &
                  (self.b_redir < 0.5) & (self.b_active > 0.5))          # [B,N]
        roll = torch.rand(B, self.N, generator=self.g, device=d) < 0.5
        do_rd = tr_due & roll
        rang = torch.rand(B, self.N, generator=self.g, device=d) * TAU
        rspd = self.b_vel.norm(dim=2, keepdim=True)
        rvel = torch.stack([torch.cos(rang), torch.sin(rang)], -1) * rspd
        self.b_vel = torch.where(do_rd[:, :, None], rvel, self.b_vel)
        self.b_redir = torch.where(tr_due, torch.ones_like(self.b_redir), self.b_redir)

        # --- enemies: waves fly in, hover, leave ---
        frs = self.frame.squeeze(1)                              # [B]
        wave_due = ((frs % WAVE_PERIOD) < 1.0) & (frs > 1.0)     # [B]
        NW = EN_PER_WAVE
        ek = self._ek[None, :]                                   # [1,NW]
        n_wave = self._ri(EN_WAVE_LO, EN_WAVE_HI, B, 1)          # [B,1]
        mkw = wave_due[:, None] & (ek < n_wave)                  # [B,NW]
        edge = self._ri(0, 3, B, NW)
        tx = self._r(B, NW, lo=PX_LO + 30, hi=PX_HI - 30)
        sy = self._r(B, NW, lo=PY_LO + 20, hi=PY_LO + 170)
        spx = torch.where(edge == 0, tx, torch.where(edge == 1,
              torch.full_like(tx, -28.0), torch.full_like(tx, PW + 28.0)))
        spy = torch.where(edge == 0, torch.full_like(sy, -28.0), sy)
        spos = torch.stack([spx, spy], -1)                       # [B,NW,2]
        stgt = torch.stack([self._r(B, NW, lo=PX_LO + 40, hi=PX_HI - 40),
                            self._r(B, NW, lo=PY_LO + 40, hi=CY + 50)], -1)
        v0 = stgt - spos
        v0 = v0 / v0.norm(dim=2, keepdim=True).clamp(min=1e-3) * EN_FLY_SPEED
        slot = ((self.en_cursor[:, None] + ek) % MAXE).long()    # [B,NW]
        s2 = slot[:, :, None].expand(B, NW, 2)
        wd = wave_due[:, None]                                   # [B,1] gate: only
        wd2 = wd[:, :, None]                                     # touch slots on a wave
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
        dt = to_t.norm(dim=2)                                    # [B,MAXE]
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

        # SHOOT auto-hits the nearest N active, ON-SCREEN enemies (N: 1/2/3 grows
        # with power). dmg = EN_DPS * power_mult (1x at 0 power .. 3x at full).
        ea = self.en_active > 0.5
        on_screen = ((self.en_pos[..., 0] > PX_LO) & (self.en_pos[..., 0] < PX_HI) &
                     (self.en_pos[..., 1] > 0.0) & (self.en_pos[..., 1] < PY_HI))
        shootable = ea & on_screen
        ed = torch.where(shootable, (self.en_pos - self.player[:, None, :]).norm(dim=2),
                         torch.full((B, MAXE), 1e9, device=d))
        pfrac = (self.power[:, 0] / POWER_MAX).clamp(0.0, 1.0)   # [B]
        pmult = 1.0 + (PWR_DMG_MULT_MAX - 1.0) * pfrac
        n_tgt = 1.0 + (pfrac > 0.5).float() + (pfrac > 0.85).float()   # [B]
        NT = 3
        dk, ik = ed.topk(NT, dim=1, largest=False)              # [B,NT]
        tgt_ok = (dk < 1e8) & (torch.arange(NT, device=d)[None, :] < n_tgt[:, None])
        per = (shoot.squeeze(1)[:, None] * EN_DPS * pmult[:, None]) * tgt_ok.float()
        hp_pre = self.en_hp.gather(1, ik).clamp(min=0.0)        # [B,NT]
        per = torch.minimum(per, hp_pre)
        self.en_hp = self.en_hp.scatter_add(1, ik, -per)
        hp_post = self.en_hp.gather(1, ik)
        killed_nt = (hp_pre > 0.0) & (hp_post <= 0.0)           # [B,NT]
        dmg = per.sum(dim=1)                                    # [B] total HP dealt
        kpos_nt = torch.gather(self.en_pos, 1, ik[:, :, None].expand(-1, -1, 2))  # [B,NT,2]

        # --- P items: spawn IT_PER_KILL per killed enemy, varied pop, fall, collect ---
        NJ = IT_PER_KILL * NT
        jj = torch.arange(NJ, device=d)
        kslot = (jj // IT_PER_KILL)[None, :]                    # [1,NJ] which target
        islot = ((self.it_cursor[:, None] + jj[None, :]) % IT_MAX).long()   # [B,NJ]
        is2 = islot[:, :, None].expand(B, NJ, 2)
        jpos = torch.gather(kpos_nt, 1, kslot[:, :, None].expand(B, NJ, 2))
        pvy = self._r(B, NJ, lo=-3.6, hi=-1.0)                  # some pop high, some low
        pvx = self._r(B, NJ, lo=-1.9, hi=1.9)
        jvel = torch.stack([pvx, pvy], -1)
        jm = torch.gather(killed_nt.float(), 1, kslot.expand(B, NJ)) > 0.5   # [B,NJ]
        cur_ip = torch.gather(self.it_pos, 1, is2)
        cur_iv = torch.gather(self.it_vel, 1, is2)
        cur_ia = torch.gather(self.it_active, 1, islot)
        self.it_pos = self.it_pos.scatter(1, is2, torch.where(jm[:, :, None], jpos, cur_ip))
        self.it_vel = self.it_vel.scatter(1, is2, torch.where(jm[:, :, None], jvel, cur_iv))
        self.it_active = self.it_active.scatter(
            1, islot, torch.where(jm, torch.ones_like(cur_ia), cur_ia))
        self.it_cursor = (self.it_cursor +
                          IT_PER_KILL * killed_nt.float().sum(dim=1)) % IT_MAX

        ia = self.it_active > 0.5
        nv = self.it_vel.clone()
        nv[..., 1] = (nv[..., 1] + IT_GRAVITY).clamp(max=IT_TERM_VY)
        self.it_vel = torch.where(ia[:, :, None], nv, torch.zeros_like(self.it_vel))
        self.it_pos = self.it_pos + self.it_vel * ia[:, :, None].float()
        idist = (self.it_pos - self.player[:, None, :]).norm(dim=2)   # [B,IT_MAX]
        got = ia & (idist < IT_COLLECT_R)
        n_got = got.sum(dim=1).float()                          # [B]
        it_off = self.it_pos[..., 1] > PY_HI + 24.0
        self.it_active = self.it_active * (~got).float() * (~it_off).float()
        self.power = (self.power + PWR_PER_ITEM * n_got[:, None]).clamp(0.0, POWER_MAX)

        # --- collision (bullets + enemy bodies) ---
        # th07 uses an AABB overlap; circular approx: dist < bullet_hitbox + PLAYER_HB
        dist = (self.b_pos - self.player[:, None, :]).norm(dim=2)
        bhit = ((self.b_active > 0.5) & (dist < self.b_rad + PLAYER_HB)).any(dim=1, keepdim=True)
        en_d = (self.en_pos - self.player[:, None, :]).norm(dim=2)
        en_hit = (ea & (en_d < EN_RADIUS + 3.0)).any(dim=1, keepdim=True)
        hit = bhit | en_hit
        newly_dead = (self.alive > 0.5) & hit
        self.death_wall = (newly_dead & bhit & ~en_hit).squeeze(1).float()   # died to a bullet
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
