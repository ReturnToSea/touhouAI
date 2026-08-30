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

# --- per-bullet MOTION PROFILES (v27 domain randomisation) --------------------
# Every bullet carries a motion type + one param, rolled per emitter per episode.
# These deliberately BREAK the straight-line assumption baked into the obs
# (native/obs.py marches every bullet as constant-velocity) so the policy has to
# react to real bullet state instead of trusting the marched prediction - the
# fix for ppo_v26's overfit to the old fixed straight-line stage.
M_STRAIGHT, M_ACCEL, M_DECEL, M_SLITHER, M_ARC, M_HOMING, M_PULSE, M_FREEZE = range(8)
#  ACCEL   : speed ramps up   (spd0 -> up to 3x over its life)
#  DECEL   : speed ramps down to a 0.35x floor, then cruises
#  SLITHER : heading oscillates sinusoidally (snaking bullets)
#  ARC     : heading turns at a constant rate (spiralling bullets)
#  HOMING  : steers toward the player for the first ~50 f, then locks
#  PULSE   : speed oscillates (fast-slow-fast) with no heading change
#  FREEZE  : the Perfect-Freeze primitive - cruise, decelerate to a full stop,
#            hold, then re-fire toward the player's position at that moment
_MOTION_POOL = [M_STRAIGHT, M_STRAIGHT, M_ACCEL, M_DECEL, M_SLITHER,
                M_ARC, M_HOMING, M_PULSE, M_FREEZE]        # STRAIGHT weighted 2x
FREEZE_T0 = 48.0        # frames of cruise before the freeze decel starts
FREEZE_STOP = 20.0      # frames to decelerate to 0
FREEZE_HOLD = 22.0      # frames held at rest before the redirect
SLITHER_AMP = 0.55     # rad, peak heading swing
HOMING_FRAMES = 50.0

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

# stage skeleton: home positions for the emitter slots. v27: this is only the
# DEFAULT layout used for slot sizing. Per EPISODE every emitter re-rolls its
# behaviour (CONE / SPRAY / LINE / BRING), its motion profile, a position jitter
# along one axis, and whether it fires at all (a random EMIT_ACTIVE_* of the
# pool). Plus per-episode: an archetype (sparse-fast / dense-slow / mixed) and
# brief "sparse windows". So the policy sees millions of stage variants, not one
# - the fix for ppo_v26's memorisation of the single fixed straight-line stage.
_CORNERS = [(-14.0, -14.0), (398.0, -14.0), (-14.0, 462.0), (398.0, 462.0)]
ROSTER = []
for _cx, _cy in _CORNERS:
    ROSTER += [(E_CONE, _cx, _cy, "ball"), (E_SPRAY, _cx, _cy, "ball")]
ROSTER.append((E_LINE, 350.0, 412.0, "ball"))
ROSTER += [(E_BRING, CX, CY, "ball"), (E_BRING, CX, CY, "ball")]  # [-2] bounces, [-1] orbits
N_CORNER_EMIT = 8                          # ROSTER[0:8] are the corner CONE/SPRAY pairs
EMIT_ACTIVE_LO, EMIT_ACTIVE_HI = 6, 11     # how many of the pool fire per episode
JIT_AXIS = 40.0                            # +-px emitter position jitter (one axis/episode)
# behaviour re-roll pool for the 8 corner slots (LINE and the 2 BRING keep their
# roles; a corner can become any of these). weighted toward CONE/SPRAY.
_EMIT_POOL = [E_CONE, E_CONE, E_CONE, E_SPRAY, E_SPRAY, E_SPRAY, E_LINE, E_BRING]

# sparse windows: per episode, brief spells where MOST emitters go quiet so the
# policy learns the isolated-bullet sidestep and that "safe" is a real state.
SPARSE_PERIOD_LO, SPARSE_PERIOD_HI = 1000, 2400
SPARSE_LEN_LO, SPARSE_LEN_HI = 60, 150

# --- enemies ---
# v16: big waves fly in from off-screen every 12 s, hover ~6 s, leave. 1 HP each;
# touching one kills the player. v23: SHOOT is FRONT-ONLY - it only hits an enemy
# roughly directly above the player (|dx| < SHOOT_ALIGN_DX). Teaches "position
# under the target" (= boss positioning). Enemies also fire 2 aimed bursts at the
# player during their hover (snapshot aim, no tracking) so crowding them hurts.
MAXE = 36                    # enemy slots
EN_PER_WAVE = 14             # slots written per wave (9-14 activated)
EN_WAVE_LO, EN_WAVE_HI = 9, 15
WAVE_PERIOD = 720            # 12 s
EN_HP = 1.0
EN_RADIUS = 13.0            # v26: 9 -> 13, a caution bias (body contact is death)
EN_FLY_SPEED = 2.6
EN_HOVER_FRAMES = 360        # 6 s
EN_DPS = 1.0 / 45.0          # base dmg/frame at power 0 (1 HP -> 0.75 s to kill)
EN_DMG_REW = 0.22          # v27: 0.35 -> 0.22 (v26 locked at ~35% enemy deaths - overshot)
PWR_STAND_REW = 0.0015     # per frame, x power_frac: makes held power lastingly worth it
# v28: nudge off the exact bottom wall (zero escape room = fragile). Only the
# bottom ~12% - the whole lower field stays free (real Touhou is played low).
BOTTOM_Y = 385.0
BOTTOM_PEN = 0.003         # per frame at the wall, ramps to 0 at BOTTOM_Y
SHOOT_ALIGN_DX = 9.0        # v27: 26 -> 15 -> 9. real shot is narrow; the wide
                            # window taught "hit from the edge" -> a real-game miss

# enemy aimed bursts: 2 per hover, snapshot-aimed at the player (no tracking)
EN_BURST_AT = (270.0, 120.0)   # fire when the (down-counting) hover timer crosses these
EN_BURST_N = 4                 # bullets per burst
EN_BURST_ARC = 0.42           # total fan width, radians (~24 deg)
EN_BURST_SPD = 2.4            # px/frame
EN_SLOTS = 400               # shared pool for enemy-burst bullets

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
IT_REW = 0.60              # reward per P item (v16 0.15 -> 0.30 -> v26 0.60)

# --- "spam" phase: every ~45-60 s, N roaming spawners near the top of the screen
# blanket the field with pellets for 10 s, then a 3 s cooldown. ALL other fire +
# enemy waves pause for the whole phase. Each phase adds one more spawner (3, 4,
# 5, ...). The pellets fire outward on a ring around the spawner (evenly spaced +
# jitter so they don't overlap), at a random 0.5x-2x speed, with a 2x burst for
# the first 0.2 s. No life cap - the ones aimed up/sideways die on a wall fast
# (fine), the ones aimed down take a while. The 3 s cooldown just lets MOST clear.
SPAM_PERIOD_LO, SPAM_PERIOD_HI = 2700, 3600   # 45-60 s between phase starts
SPAM_FIRE_FRAMES = 600      # 10 s of firing
SPAM_COOLDOWN = 180         # 3 s before normal fire resumes (most pellets gone by then)
SPAM_N0 = 3                 # spawners in the first phase; +1 each phase
SPAM_MAX = 6               # slot / state cap (a 180 s episode sees ~3 phases: 3, 4, 5)
SPAM_SLOTS = 1600           # pellet pool. free-slot allocation (reuse inactive /
                            # least-threatening slots) + the past-player cull keep
                            # this ~= peak SIMULTANEOUS live count, not a ring lap
SPAM_PAST_DY = 70.0        # cull a spam pellet once it's this far below the player
SPAM_PAST_Y = 300.0       # ...and in the lower field (guards over-cull when high)
SPAM_FIRE_EVERY = 12        # 5 attacks / s
SPAM_PER_ATTACK = 20        # pellets per spawner per attack
SPAM_RING_R = 32.0          # pellets spawn on this ring (fits 20 over a 150 deg fan)
SPAM_ARC = 2.6             # downward fan width, radians (~150 deg) - no up-going pellets
SPAM_BASE_SPD = 2.0         # pellet speed = U(0.5, 2.0) * this  -> range 1.0 .. 4.0
SPAM_BOOST_FRAMES = 12      # first 0.2 s at 2x the chosen speed
SPAM_SPAWNER_VX = 0.75      # spawner drift speed (px/f), x only
SPAM_SEG = 128.0            # reverse direction after drifting ~1/3 of the stage
SPAM_Y = 28.0               # near the TOP of the stage; small +-offset, no y motion
_PELLET_HB = TH07_BULLETS["pellet"]["hitbox"]
_BALL_HB = TH07_BULLETS["ball"]["hitbox"]


class DanmakuSim:
    def __init__(self, B=16384, device="cuda", slots_per_emitter=176, spawn_k=24,
                 max_frames=5400, frame_skip=3, alive_rew=0.01, death_rew=-1.0,
                 seed=0, compile=True):
        self.B = B
        self.dev = torch.device(device)
        self.E = len(ROSTER)
        self.SPE = slots_per_emitter
        self.K = spawn_k
        self._spam_base = self.E * slots_per_emitter
        self._en_base = self._spam_base + SPAM_SLOTS       # enemy-burst bullet pool
        self.N = self._en_base + EN_SLOTS + 1              # +1 = dump slot
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
        # b_rad dropped: bullet hitbox is a per-SLOT constant -> self._slot_rad[N]
        self.b_age = torch.zeros(B, self.N, device=d, dtype=torch.float16)   # frames since spawn
        # v27 per-bullet motion state (emitter slots only): spawn heading + speed,
        # a motion-profile id and one param. b_vel is DERIVED from these each frame
        # for emitter bullets (spam / enemy bursts keep straight b_vel).
        self.b_head0 = torch.zeros(B, self.N, device=d)
        self.b_spd0 = torch.zeros(B, self.N, device=d)
        self.b_mtype = torch.zeros(B, self.N, device=d, dtype=torch.float16)
        self.b_mp = torch.zeros(B, self.N, device=d, dtype=torch.float16)
        self.cursor = torch.zeros(B, E, device=d)

        self.e_pos = torch.zeros(B, E, 2, device=d)
        self.e_type = torch.zeros(B, E, device=d)          # v27: per-episode behaviour
        self.e_on = torch.ones(B, E, device=d)             # v27: per-episode active mask
        self.e_jitxy = torch.zeros(B, E, 2, device=d)      # v27: per-episode position jitter
        self.e_mtype = torch.zeros(B, E, device=d)         # v27: per-episode motion profile
        self.e_mp = torch.zeros(B, E, device=d)            # v27: motion param
        self.e_bvel = torch.zeros(B, E, 2, device=d)       # bounce velocity (BRING[-2])
        self.e_oa = torch.zeros(B, E, device=d)            # orbit angle (BRING[-1])
        self.e_speed = torch.zeros(B, E, device=d)
        self.e_ang = torch.zeros(B, E, device=d)           # BRING ring phase
        self.e_dang = torch.zeros(B, E, device=d)          # BRING ring spin
        self.e_period = torch.ones(B, E, device=d) * 30
        self.e_phase = torch.zeros(B, E, device=d)
        self.e_nspawn = torch.ones(B, E, device=d) * 8
        self.e_spread = torch.zeros(B, E, device=d)        # CONE per-bullet fan
        self.e_swctr = torch.zeros(B, E, device=d)         # LINE sweep centre / amp / rate
        self.e_swamp = torch.zeros(B, E, device=d)
        self.e_swrate = torch.zeros(B, E, device=d)
        self.e_swphase = torch.zeros(B, E, device=d)
        self.arche = torch.zeros(B, 1, device=d)           # v27: episode archetype 0/1/2
        # v27 sparse windows: brief spells where most emitters go quiet
        self.sparse_phase = torch.zeros(B, device=d)       # 0 normal / 1 sparse
        self.sparse_t = torch.zeros(B, device=d)
        self.sparse_next = torch.zeros(B, device=d)
        self.sparse_len = torch.zeros(B, device=d)

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

        # --- spam phase ---
        self.spam_phase = torch.zeros(B, device=d)            # 0 idle / 1 fire / 2 cooldown
        self.spam_t = torch.zeros(B, device=d)                # frames into the current phase
        self.spam_next = torch.zeros(B, device=d)             # frame the next phase starts
        self.spam_n = torch.full((B,), float(SPAM_N0), device=d)   # spawners this episode
        self.en_bcursor = torch.zeros(B, device=d)            # ring into the EN_SLOTS pool
        self._enj = torch.arange(EN_BURST_N, device=d).float()
        self.sp_x = torch.zeros(B, SPAM_MAX, device=d)
        self.sp_y = torch.zeros(B, SPAM_MAX, device=d)
        self.sp_dir = torch.ones(B, SPAM_MAX, device=d)
        self.sp_trav = torch.zeros(B, SPAM_MAX, device=d)
        self._sp_k = torch.arange(SPAM_MAX, device=d).float()
        self._spj = torch.arange(SPAM_PER_ATTACK, device=d).float()

        # death-cause diagnostic (set each frame an env dies): bullet vs enemy body
        self.death_wall = torch.zeros(B, device=d)            # "wall" = any bullet now
        self.death_enemy = torch.zeros(B, device=d)

        self._R_type = torch.tensor([r[0] for r in ROSTER], device=d, dtype=torch.float32)
        self._R_xy = torch.tensor([[r[1], r[2]] for r in ROSTER], device=d, dtype=torch.float32)
        self._R_hitbox = torch.tensor([TH07_BULLETS[r[3]]["hitbox"] for r in ROSTER],
                                      device=d, dtype=torch.float32)          # [E]
        self._eidx = torch.arange(E, device=d)
        self._is_orbit = (self._eidx == E - 1).float()     # last BRING orbits the edge

        # per-slot bullet lifetime: v27 - emitter bullets get a flat 8 s cap
        # (slow / bouncing / arcing / frozen bullets would otherwise pile up).
        slot_emit = torch.arange(self.N, device=d) // self.SPE
        _is_emit_slot = torch.arange(self.N, device=d) < self._spam_base
        self._slot_life = torch.where(_is_emit_slot,
                                      torch.full((self.N,), 480.0, device=d),
                                      torch.full((self.N,), 1e9, device=d))
        self._spam_slot = ((torch.arange(self.N, device=d) >= self._spam_base) &
                           (torch.arange(self.N, device=d) < self._en_base))
        self._en_slot = ((torch.arange(self.N, device=d) >= self._en_base) &
                         (torch.arange(self.N, device=d) < self.dump))
        self._emit_slot = (torch.arange(self.N, device=d) < self._spam_base)  # [N]
        self._slot_emit_ix = slot_emit.clamp(max=E - 1)                       # [N] which emitter
        self.b_redir = torch.zeros(B, self.N, device=d, dtype=torch.float16)  # M_FREEZE: redirected flag

        # per-slot bullet hitbox (radius): emitter slots use their emitter's type,
        # spam slots = pellet, enemy-burst slots = ball. Constant across the batch.
        self._slot_rad = torch.full((self.N,), _BALL_HB, device=d)
        _em = torch.arange(self.N, device=d) < self._spam_base
        self._slot_rad = torch.where(
            _em, self._R_hitbox[slot_emit.clamp(max=E - 1)], self._slot_rad)
        self._slot_rad = torch.where(self._spam_slot,
                                     torch.full((self.N,), _PELLET_HB, device=d),
                                     self._slot_rad)

        self._k = torch.arange(self.K, device=d).float()
        self._ebase = self._eidx.float() * self.SPE
        # head_aux [B,9] = lives/9, bombs/9, power/128, tanh(graze/100), stage/6,
        # alive, dead, boss_present, boss_frac. Only power + alive vary in the sim.
        self._head = torch.zeros(B, 9, device=d)
        self._head[:, 5] = 1.0
        self._zeros2 = torch.zeros(B, 2, device=d)
        self._zeros1 = torch.zeros(B, device=d)
        self.player_focus = torch.zeros(B, device=d)   # v28: real focus -> obs

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

        # ---- v27: re-roll the STAGE per episode ---------------------------------
        is_orbit = (self._eidx == E - 1)[None, :].expand(B, E)       # last slot orbits
        is_line_slot = (self._eidx == E - 3)[None, :].expand(B, E)   # slot E-3 stays LINE
        # corner slots (0 .. E-3 exclusive) re-roll their behaviour; LINE + the 2
        # BRING keep their slot roles.
        pool = torch.tensor(_EMIT_POOL, device=d)
        rolled = pool[self._ri(0, len(_EMIT_POOL), B, E).long()]      # [B,E]
        new_type = torch.where(is_orbit | (self._eidx == E - 2)[None, :].expand(B, E),
                               torch.full((B, E), float(E_BRING), device=d),
                    torch.where(is_line_slot, torch.full((B, E), float(E_LINE), device=d),
                                rolled.float()))
        self.e_type = torch.where(mbe, new_type, self.e_type)
        is_line = new_type == E_LINE
        is_bring = new_type == E_BRING
        is_spray = new_type == E_SPRAY
        is_bounce = is_bring & ~is_orbit

        # per-episode active mask: a random EMIT_ACTIVE_* subset fires
        n_act = self._ri(EMIT_ACTIVE_LO, EMIT_ACTIVE_HI, B, 1)       # [B,1]
        perm_rank = torch.argsort(torch.argsort(self._r(B, E), dim=1), dim=1)
        new_on = (perm_rank < n_act).float()
        self.e_on = torch.where(mbe, new_on, self.e_on)

        # per-episode position jitter along ONE axis (kept so corner emitters stay
        # outside the field: jitter is signed toward "more outside" for corners)
        jamt = self._r(B, E, lo=-JIT_AXIS, hi=JIT_AXIS)
        jx = self._r(B, E) > 0.5
        jit = torch.stack([torch.where(jx, jamt, torch.zeros_like(jamt)),
                           torch.where(jx, torch.zeros_like(jamt), jamt)], -1)
        self.e_jitxy = torch.where(mbe[:, :, None], jit, self.e_jitxy)

        # per-episode motion profile (CONE/SPRAY/LINE roll; BRING stays straight)
        mpool = torch.tensor(_MOTION_POOL, device=d)
        mt = mpool[self._ri(0, len(_MOTION_POOL), B, E).long()].float()
        mt = torch.where(is_bring, torch.zeros_like(mt), mt)         # M_STRAIGHT
        self.e_mtype = torch.where(mbe, mt, self.e_mtype)
        e_mp = torch.where(mt == M_ACCEL, self._r(B, E, lo=0.010, hi=0.030),
               torch.where(mt == M_DECEL, self._r(B, E, lo=0.010, hi=0.024),
               torch.where(mt == M_SLITHER, self._r(B, E, lo=0.06, hi=0.16),
               torch.where(mt == M_ARC, self._r(B, E, lo=-0.028, hi=0.028),
               torch.where(mt == M_HOMING, self._r(B, E, lo=0.006, hi=0.018),
               torch.where(mt == M_PULSE, self._r(B, E, lo=0.05, hi=0.13),
                           self._r(B, E, lo=0.1, hi=0.3)))))))
        self.e_mp = torch.where(mbe, e_mp, self.e_mp)

        # per-episode archetype: 0 fast+sparse, 1 slow+dense, 2 mixed
        arche = self._ri(0, 3, B, 1)
        self.arche = torch.where(mb, arche, self.arche)
        a_spd = torch.where(arche == 0, 1.10, torch.where(arche == 1, 0.90, 1.0))
        a_prd = torch.where(arche == 0, 1.35, torch.where(arche == 1, 0.8, 1.0))   # x fire period

        # spawn lower-centre (like real Touhou)
        px = self._r(B, 1, lo=PX_LO + 50, hi=PX_HI - 50)
        py = self._r(B, 1, lo=PY_HI - 140, hi=PY_HI - 30)
        self.player = torch.where(mb, torch.cat([px, py], 1), self.player)
        self.frame = torch.where(mb, torch.zeros_like(self.frame), self.frame)
        self.alive = torch.where(mb, torch.ones_like(self.alive), self.alive)
        diff = 0.2 + 0.8 * self._r(B, 1)
        self.diff = torch.where(mb, diff, self.diff)
        dsc = (0.75 + 0.55 * diff) * a_spd                           # bullet-speed mult

        self.b_active = torch.where(mb, torch.zeros_like(self.b_active), self.b_active)
        self.cursor = torch.where(mb, torch.zeros_like(self.cursor), self.cursor)

        # emitter positions: home + per-episode jitter; bouncers get a random
        # interior start, the orbiter rides the perimeter ellipse.
        home = self._R_xy.expand(B, E, 2) + self.e_jitxy
        self.e_pos = torch.where(mbe[:, :, None], home, self.e_pos)
        bstart = torch.stack([self._r(B, E, lo=CX - 60, hi=CX + 60),
                              self._r(B, E, lo=CY - 60, hi=CY + 60)], -1)
        self.e_pos = torch.where((mbe & is_bounce)[:, :, None], bstart, self.e_pos)
        bang = self._r(B, E, lo=0, hi=TAU)
        bvel = torch.stack([torch.cos(bang), torch.sin(bang)], -1)   # fixed speed 1.0
        self.e_bvel = torch.where((mbe & is_bounce)[:, :, None], bvel,
                                  torch.where(mb[:, :, None], torch.zeros_like(self.e_bvel),
                                              self.e_bvel))
        self.e_oa = torch.where(mbe & is_orbit, self._r(B, E, lo=0, hi=TAU), self.e_oa)

        speed = torch.where(is_line, self._r(B, E, lo=2.6, hi=3.6) * dsc,
                torch.where(is_bring, self._r(B, E, lo=0.20, hi=0.34),
                torch.where(is_spray, self._r(B, E, lo=1.8, hi=2.8) * dsc,
                            self._r(B, E, lo=2.1, hi=3.0) * dsc)))
        self.e_speed = torch.where(mbe, speed, self.e_speed)

        # NOTE: the fire check is `(frame % e_period) == e_phase` (exact), so
        # e_period MUST stay integer-valued - round after the archetype scale.
        period = (torch.where(is_line, self._ri(5, 9, B, E),
                  torch.where(is_bring, self._ri(28, 40, B, E),
                  torch.where(is_spray, self._ri(24, 46, B, E),
                              self._ri(26, 50, B, E)))) * a_prd).round().clamp(min=3.0)
        self.e_period = torch.where(mbe, period, self.e_period)
        self.e_phase = torch.where(mbe, torch.floor(self._r(B, E) * period), self.e_phase)

        nsp = torch.where(is_line, torch.ones(B, E, device=d),
              torch.where(is_bring, self._ri(7, 12, B, E),
              torch.where(is_spray, self._ri(4, 8, B, E),
                          self._ri(3, 6, B, E))))
        self.e_nspawn = torch.where(mbe, nsp, self.e_nspawn)

        # e_spread doubles as the CONE per-bullet step and the SPRAY total arc
        spread = torch.where(is_spray, self._r(B, E, lo=1.6, hi=2.5),
                             self._r(B, E, lo=0.09, hi=0.20))
        self.e_spread = torch.where(mbe, spread, self.e_spread)
        self.e_ang = torch.where(mbe, self._r(B, E, lo=0, hi=TAU), self.e_ang)
        dang = torch.where(is_bring, self._r(B, E, lo=-0.10, hi=0.10),
                           self._r(B, E, lo=-0.25, hi=0.25))
        self.e_dang = torch.where(mbe, dang, self.e_dang)

        # LINE sweep: aim oscillates between "up toward top-right" and "left
        # toward bottom-left"  (centre ~ -135deg, amplitude ~ +-50deg)
        # LINE sweep centre = aim toward the stage centre from THIS emitter's
        # (per-episode, jittered) position, so a LINE that rolled onto any corner
        # still sweeps across the field instead of firing off the near edge.
        sw_ctr = torch.atan2(CY - self.e_pos[..., 1], CX - self.e_pos[..., 0])
        self.e_swctr = torch.where(mbe, sw_ctr, self.e_swctr)
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

        self.spam_phase = torch.where(mb1, torch.zeros_like(self.spam_phase), self.spam_phase)
        self.spam_t = torch.where(mb1, torch.zeros_like(self.spam_t), self.spam_t)
        self.spam_n = torch.where(mb1, torch.full_like(self.spam_n, float(SPAM_N0)), self.spam_n)
        self.en_bcursor = torch.where(mb1, torch.zeros_like(self.en_bcursor), self.en_bcursor)
        self.spam_next = torch.where(
            mb1, self._r(B, lo=float(SPAM_PERIOD_LO), hi=float(SPAM_PERIOD_HI)), self.spam_next)

        # v27 sparse-window schedule
        self.sparse_phase = torch.where(mb1, torch.zeros_like(self.sparse_phase), self.sparse_phase)
        self.sparse_t = torch.where(mb1, torch.zeros_like(self.sparse_t), self.sparse_t)
        self.sparse_next = torch.where(
            mb1, self._r(B, lo=float(SPARSE_PERIOD_LO), hi=float(SPARSE_PERIOD_HI)),
            self.sparse_next)
        self.sparse_len = torch.where(
            mb1, self._r(B, lo=float(SPARSE_LEN_LO), hi=float(SPARSE_LEN_HI)), self.sparse_len)

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
        return self._obs_fn(self.player, self._zeros2, self.player_focus,
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
        self.player_focus = focus.squeeze(1)      # v28: -> obs escape scalars
        moving = (mv.abs().sum(1, keepdim=True) > 0).float()
        self.player = self.player + moving * mv / norm * spd
        self.player = torch.stack([self.player[:, 0].clamp(PX_LO, PX_HI),
                                   self.player[:, 1].clamp(PY_LO, PY_HI)], 1)

        # --- v27: resolve per-bullet MOTION PROFILE for the emitter slots ------
        # b_vel for emitter slots is DERIVED each frame from (spawn heading +
        # speed, motion type, param, age). Non-linear on purpose - breaks the
        # obs's straight-line march so the policy reacts to real state instead of
        # trusting the prediction. Sliced to [:, :SB] (spam / enemy bursts keep
        # their own straight b_vel) to keep the transient memory down.
        SB = self._spam_base
        age = self.b_age[:, :SB].float()                           # [B,SB]
        mt = self.b_mtype[:, :SB].float()
        mp = self.b_mp[:, :SB].float()
        s0 = self.b_spd0[:, :SB]
        _fz_b = FREEZE_T0 + FREEZE_STOP
        _fz_c = _fz_b + FREEZE_HOLD
        spd_fac = torch.where(mt == M_ACCEL, (1.0 + mp * age).clamp(max=1.8),
                  torch.where(mt == M_DECEL, (1.0 - mp * age).clamp(min=0.35),
                  torch.where(mt == M_PULSE, 1.0 + 0.5 * torch.sin(age * mp),
                  torch.where(mt == M_FREEZE,
                              ((_fz_b - age) / FREEZE_STOP).clamp(0.0, 1.0)
                              + (age >= _fz_c).float(),
                              torch.ones_like(age)))))
        # hard ceiling on the FINAL speed (~1.6x player unfocused) - bounds the
        # base x diff x archetype x motion-profile stack to a realistic range
        # (real th07: stage-1 ~2 px/f, hardest stages ~6). No floor: FREEZE
        # bullets must reach speed 0 during their stop.
        spd = (s0 * spd_fac).clamp(max=6.5)
        # heading: SLITHER oscillates, ARC turns, HOMING steers at the player
        # then locks, FREEZE re-aims at the player once when the hold ends.
        rel = self.player[:, None, :] - self.b_pos[:, :SB]         # [B,SB,2]
        tgt_h = torch.atan2(rel[..., 1], rel[..., 0])
        h0 = self.b_head0[:, :SB]
        fz_redir = ((mt == M_FREEZE) & (age >= _fz_c) & (self.b_redir[:, :SB] < 0.5)
                    & (self.b_active[:, :SB] > 0.5))
        h0 = torch.where(fz_redir, tgt_h, h0)
        self.b_head0 = torch.cat([h0, self.b_head0[:, SB:]], dim=1)
        self.b_redir = torch.cat([torch.where(fz_redir, torch.ones_like(h0),
                                              self.b_redir[:, :SB].float()).half(),
                                  self.b_redir[:, SB:]], dim=1)
        dh = torch.remainder(tgt_h - h0 + math.pi, TAU) - math.pi
        home_h = h0 + dh.clamp(-0.7, 0.7) * (mp * torch.clamp(age, max=HOMING_FRAMES))
        head = torch.where(mt == M_SLITHER, h0 + SLITHER_AMP * torch.sin(age * mp),
               torch.where(mt == M_ARC, h0 + mp * age,
               torch.where(mt == M_HOMING, home_h, h0)))
        mvel = torch.stack([spd * torch.cos(head), spd * torch.sin(head)], -1)
        self.b_vel = torch.cat([mvel, self.b_vel[:, SB:]], dim=1)

        # --- move bullets (spam pellets get a 2x burst for their first 0.2 s),
        #     cull off-screen + past lifetime ---
        boost = torch.where(
            self._spam_slot[None, :] & (self.b_age < float(SPAM_BOOST_FRAMES)), 2.0, 1.0)
        self.b_pos = self.b_pos + self.b_vel * boost[..., None]
        self.b_age = self.b_age + 1.0
        on = ((self.b_pos[..., 0] > -18) & (self.b_pos[..., 0] < PW + 18) &
              (self.b_pos[..., 1] > -18) & (self.b_pos[..., 1] < PH + 18) &
              (self.b_age < self._slot_life))
        # spam pellets that have fallen well past the player (lower field, below
        # by SPAM_PAST_DY) can never threaten it again and aren't in the danger
        # grid -> cull them so their pool slot frees up
        past = (self._spam_slot[None, :] &
                (self.b_pos[..., 1] > self.player[:, 1:2] + SPAM_PAST_DY) &
                (self.b_pos[..., 1] > SPAM_PAST_Y))
        self.b_active = self.b_active * on.float() * (~past).float()

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

        # --- "spam" phase: roaming top-screen spawners blanket the field with
        #     pellets for 10 s every ~45-60 s; all other fire + enemy waves pause
        #     for the phase + a 3 s cooldown; each phase adds one more spawner ---
        frs = self.frame.squeeze(1)                                  # [B]
        trig = (self.spam_phase < 0.5) & (frs >= self.spam_next)     # [B] start a phase
        spk = self._sp_k[None, :]                                    # [1,SPAM_MAX]
        act_k = spk < self.spam_n[:, None]                           # [B,SPAM_MAX] live spawners
        denom = (self.spam_n[:, None] - 1.0).clamp(min=1.0)
        x0 = PW * 0.20 + PW * 0.60 * spk / denom                     # evenly spread in x
        y0 = SPAM_Y + self._r(B, SPAM_MAX, lo=-6.0, hi=10.0)   # stay near the top
        dir0 = torch.where(self._r(B, SPAM_MAX) > 0.5, 1.0, -1.0)
        t3 = trig[:, None]
        self.sp_x = torch.where(t3, x0, self.sp_x)
        self.sp_y = torch.where(t3, y0, self.sp_y)
        self.sp_dir = torch.where(t3, dir0, self.sp_dir)
        self.sp_trav = torch.where(t3, torch.zeros_like(self.sp_trav), self.sp_trav)
        self.spam_phase = torch.where(trig, torch.ones_like(self.spam_phase), self.spam_phase)
        self.spam_t = torch.where(trig, torch.zeros_like(self.spam_t), self.spam_t)

        firing = (self.spam_phase > 0.5) & (self.spam_phase < 1.5)   # [B]
        in_phase = self.spam_phase > 0.5

        # spawners drift on x only; flip after ~1/3 of the stage or at the bounds
        mvx = SPAM_SPAWNER_VX * self.sp_dir * firing[:, None].float()
        nx = self.sp_x + mvx
        self.sp_trav = self.sp_trav + mvx.abs()
        flip = (self.sp_trav >= SPAM_SEG) | (nx < PX_LO + 8.0) | (nx > PX_HI - 8.0)
        self.sp_dir = torch.where(flip, -self.sp_dir, self.sp_dir)
        self.sp_trav = torch.where(flip, torch.zeros_like(self.sp_trav), self.sp_trav)
        self.sp_x = nx.clamp(PX_LO + 8.0, PX_HI - 8.0)
        self.spam_t = torch.where(in_phase, self.spam_t + 1.0, self.spam_t)

        # fire: SPAM_PER_ATTACK pellets/spawner every SPAM_FIRE_EVERY frames, evenly
        # spread over a DOWNWARD ~150 deg fan (no up-going pellets), + small jitter
        # so they don't overlap, each at U(0.5, 2.0) x the base speed
        fire_due = firing & ((self.spam_t % SPAM_FIRE_EVERY) < 1.0) & (self.spam_t > 0.5)
        NB = SPAM_MAX * SPAM_PER_ATTACK
        jj = self._spj[None, None, :]                                # [1,1,SPAM_PER_ATTACK]
        wob = self._r(B, SPAM_MAX, 1, lo=-0.20, hi=0.20)             # small per-attack rotation
        jit = self._r(B, SPAM_MAX, SPAM_PER_ATTACK, lo=-0.05, hi=0.05)
        sang = (math.pi * 0.5) + wob + (jj + jit - (SPAM_PER_ATTACK - 1) * 0.5) * (
            SPAM_ARC / (SPAM_PER_ATTACK - 1))                        # centre = straight down
        sspd = SPAM_BASE_SPD * self._r(B, SPAM_MAX, SPAM_PER_ATTACK, lo=0.5, hi=2.0)
        sdir = torch.stack([torch.cos(sang), torch.sin(sang)], -1)   # [B,SPAM_MAX,SPAM_PER_ATTACK,2]
        sp_pos = torch.stack([self.sp_x, self.sp_y], -1)            # [B,SPAM_MAX,2]
        sbpos = sp_pos[:, :, None, :] + SPAM_RING_R * sdir
        sbvel = sspd[..., None] * sdir                              # cruise speed (2x for 0.2 s)
        semit = (fire_due[:, None, None] & act_k[:, :, None]).expand(
            B, SPAM_MAX, SPAM_PER_ATTACK)                           # [B,SPAM_MAX,SPAM_PER_ATTACK]

        # free-slot allocation: write into the NB lowest-priority spam slots -
        # inactive first, then pellets already below the player, then (only if the
        # pool is genuinely full) the least-threatening live ones. No ring cursor.
        sb, eb = self._spam_base, self._en_base
        harmless = (self.b_pos[:, sb:eb, 1] > self.player[:, 1:2]).float()   # [B,SPAM_SLOTS]
        score = self.b_active[:, sb:eb] - 0.5 * harmless
        free = torch.topk(score, NB, dim=1, largest=False).indices           # [B,NB] local idx
        raw_s = sb + free
        idx_s = torch.where(semit.reshape(B, NB), raw_s,
                            torch.full_like(raw_s, self.dump)).long()
        si2 = idx_s[:, :, None].expand(B, NB, 2)
        self.b_pos = self.b_pos.scatter(1, si2, sbpos.reshape(B, NB, 2))
        self.b_vel = self.b_vel.scatter(1, si2, sbvel.reshape(B, NB, 2))
        self.b_active = self.b_active.scatter(1, idx_s, semit.reshape(B, NB).float())
        self.b_age = self.b_age.scatter(1, idx_s, torch.zeros(B, NB, device=d, dtype=torch.float16))
        self.b_redir = self.b_redir.scatter(1, idx_s, torch.zeros(B, NB, device=d, dtype=torch.float16))
        self.b_active[:, self.dump] = 0.0

        # phase transitions (from this frame's phase, before the change)
        pph = self.spam_phase
        to_cool = (pph > 0.5) & (pph < 1.5) & (self.spam_t >= float(SPAM_FIRE_FRAMES))
        to_idle = (pph > 1.5) & (self.spam_t >= float(SPAM_COOLDOWN))
        self.spam_phase = torch.where(to_cool, torch.full_like(pph, 2.0),
                          torch.where(to_idle, torch.zeros_like(pph), pph))
        self.spam_n = torch.where(to_cool, (self.spam_n + 1.0).clamp(max=float(SPAM_MAX)),
                                  self.spam_n)
        self.spam_t = torch.where(to_cool | to_idle, torch.zeros_like(self.spam_t), self.spam_t)
        self.spam_next = torch.where(
            to_idle, frs + self._r(B, lo=float(SPAM_PERIOD_LO), hi=float(SPAM_PERIOD_HI)),
            self.spam_next)
        spam_gate = (self.spam_phase > 0.5)[:, None]                 # [B,1] pause normal fire

        # --- v27 sparse windows: brief spells where emitters go quiet ----------
        sp_trig = (self.sparse_phase < 0.5) & (frs >= self.sparse_next)
        self.sparse_phase = torch.where(sp_trig, torch.ones_like(self.sparse_phase),
                                        self.sparse_phase)
        self.sparse_t = torch.where(sp_trig, torch.zeros_like(self.sparse_t), self.sparse_t)
        in_sparse = self.sparse_phase > 0.5
        self.sparse_t = torch.where(in_sparse, self.sparse_t + 1.0, self.sparse_t)
        sp_end = in_sparse & (self.sparse_t >= self.sparse_len)
        self.sparse_phase = torch.where(sp_end, torch.zeros_like(self.sparse_phase),
                                        self.sparse_phase)
        self.sparse_next = torch.where(
            sp_end, frs + self._r(B, lo=float(SPARSE_PERIOD_LO), hi=float(SPARSE_PERIOD_HI)),
            self.sparse_next)
        quiet_gate = (spam_gate.squeeze(1) | in_sparse)[:, None]     # [B,1]

        # --- emitters fire ---
        fr = self.frame                                     # [B,1]
        due = (((fr % self.e_period) == self.e_phase) & (self.e_on > 0.5)
               & ~quiet_gate)
        kk = self._k[None, None, :]                         # [1,1,K]
        nsp = self.e_nspawn[:, :, None]                     # [B,E,1]
        emit = due[:, :, None] & (kk < nsp)                 # [B,E,K]

        epos = self.e_pos
        spd_e = self.e_speed[:, :, None]
        typ = self.e_type[:, :, None]                       # [B,E,1] per-episode behaviour

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
        c_mt = self.e_mtype[:, :, None].expand(B, E, K)         # motion profile per bullet
        c_mp = self.e_mp[:, :, None].expand(B, E, K)
        c_sp = self.e_speed[:, :, None].expand(B, E, K)

        raw = self._ebase[None, :, None] + (self.cursor[:, :, None] + kk) % self.SPE
        idx = torch.where(emit, raw, torch.full_like(raw, float(self.dump))).long()
        idxf = idx.reshape(B, E * K)
        self.b_pos = self.b_pos.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cpos.reshape(B, E * K, 2))
        self.b_vel = self.b_vel.scatter(1, idxf[:, :, None].expand(B, E * K, 2),
                                        cvel.reshape(B, E * K, 2))
        self.b_head0 = self.b_head0.scatter(1, idxf, ang.expand(B, E, K).reshape(B, E * K))
        self.b_spd0 = self.b_spd0.scatter(1, idxf, c_sp.reshape(B, E * K))
        self.b_mtype = self.b_mtype.scatter(1, idxf, c_mt.reshape(B, E * K).half())
        self.b_mp = self.b_mp.scatter(1, idxf, c_mp.reshape(B, E * K).half())
        self.b_active = self.b_active.scatter(1, idxf, emit.reshape(B, E * K).float())
        self.b_age = self.b_age.scatter(
            1, idxf, torch.zeros(B, E * K, device=self.dev, dtype=torch.float16))
        self.b_redir = self.b_redir.scatter(
            1, idxf, torch.zeros(B, E * K, device=self.dev, dtype=torch.float16))
        self.b_active[:, self.dump] = 0.0

        self.cursor = torch.where(due, (self.cursor + self.e_nspawn) % self.SPE, self.cursor)
        self.e_ang = self.e_ang + due.float() * self.e_dang

        # --- enemies: waves fly in, hover, leave ---
        wave_due = ((frs % WAVE_PERIOD) < 1.0) & (frs > 1.0) & ~spam_gate.squeeze(1)  # [B]
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
        t_pre = self.en_timer
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

        # --- enemy aimed bursts: 2 per hover, snapshot-aimed at the player (no
        #     tracking); paused during the spam phase ---
        b1 = hov & (t_pre > EN_BURST_AT[0]) & (self.en_timer <= EN_BURST_AT[0])
        b2 = hov & (t_pre > EN_BURST_AT[1]) & (self.en_timer <= EN_BURST_AT[1])
        eburst = (b1 | b2) & ~spam_gate                          # [B,MAXE]
        e_aim = torch.atan2(self.player[:, None, 1] - self.en_pos[..., 1],
                            self.player[:, None, 0] - self.en_pos[..., 0])   # [B,MAXE]
        efan = (self._enj[None, None, :] - (EN_BURST_N - 1) * 0.5) * (
            EN_BURST_ARC / (EN_BURST_N - 1))
        eang = e_aim[..., None] + efan                           # [B,MAXE,EN_BURST_N]
        edir = torch.stack([torch.cos(eang), torch.sin(eang)], -1)
        ebpos = self.en_pos[:, :, None, :].expand(B, MAXE, EN_BURST_N, 2)
        ebvel = EN_BURST_SPD * edir
        ebe = eburst[:, :, None].expand(B, MAXE, EN_BURST_N)
        ENB = MAXE * EN_BURST_N
        ear = torch.arange(ENB, device=d)[None, :]
        eraw = self._en_base + (self.en_bcursor[:, None] + ear) % EN_SLOTS
        eidx = torch.where(ebe.reshape(B, ENB), eraw,
                           torch.full_like(eraw, float(self.dump))).long()
        ei2 = eidx[:, :, None].expand(B, ENB, 2)
        self.b_pos = self.b_pos.scatter(1, ei2, ebpos.reshape(B, ENB, 2))
        self.b_vel = self.b_vel.scatter(1, ei2, ebvel.reshape(B, ENB, 2))
        self.b_active = self.b_active.scatter(1, eidx, ebe.reshape(B, ENB).float())
        self.b_age = self.b_age.scatter(1, eidx, torch.zeros(B, ENB, device=d, dtype=torch.float16))
        self.b_redir = self.b_redir.scatter(1, eidx, torch.zeros(B, ENB, device=d, dtype=torch.float16))
        self.b_active[:, self.dump] = 0.0
        self.en_bcursor = (self.en_bcursor + eburst.float().sum(1) * EN_BURST_N) % EN_SLOTS

        # SHOOT: FRONT-ONLY - only hits an enemy roughly directly above the player
        # (|dx| < SHOOT_ALIGN_DX, above), nearest first. dmg = EN_DPS * power_mult.
        ea = self.en_active > 0.5
        on_screen = ((self.en_pos[..., 0] > PX_LO) & (self.en_pos[..., 0] < PX_HI) &
                     (self.en_pos[..., 1] > 0.0) & (self.en_pos[..., 1] < PY_HI))
        rel_e = self.en_pos - self.player[:, None, :]            # [B,MAXE,2]
        aligned = (ea & on_screen & (rel_e[..., 0].abs() < SHOOT_ALIGN_DX) &
                   (rel_e[..., 1] < 0.0))
        ed = torch.where(aligned, rel_e.norm(dim=2),
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
        # th07 collision is an AABB overlap (box @ zBullet+0xB7C, +-size/2). v28:
        # match it - |dx| AND |dy| within (hitbox + PLAYER_HB). Stage-1 bullets
        # are square so one half-extent per slot. Cheaper than the circle (no sqrt)
        # and slightly harder on the diagonals (the safe direction for transfer).
        rel_b = (self.b_pos - self.player[:, None, :]).abs()
        r_hit = self._slot_rad[None, :] + PLAYER_HB
        bhitmask = ((self.b_active > 0.5) & (rel_b[..., 0] < r_hit) & (rel_b[..., 1] < r_hit))
        bhit = bhitmask.any(dim=1, keepdim=True)
        spam_hit = (bhitmask & self._spam_slot[None, :]).any(dim=1, keepdim=True)
        eshot_hit = (bhitmask & self._en_slot[None, :]).any(dim=1, keepdim=True)
        en_d = (self.en_pos - self.player[:, None, :]).norm(dim=2)
        en_hit = (ea & (en_d < EN_RADIUS + 3.0)).any(dim=1, keepdim=True)
        hit = bhit | en_hit
        newly_dead = (self.alive > 0.5) & hit
        # death cause: "wall" col now = SPAM pellet; "enemy" col = enemy body OR
        # enemy-burst bullet; the remainder (100-spam-enemy) = normal emitter fire
        self.death_wall = (newly_dead & spam_hit).squeeze(1).float()
        self.death_enemy = (newly_dead & (en_hit | eshot_hit) & ~spam_hit).squeeze(1).float()
        self.alive = self.alive * (~hit).float()
        self.frame = self.frame + 1.0
        done = newly_dead | (self.frame >= self.max_frames)
        alive_now = self.alive > 0.5
        rew = torch.where(newly_dead, torch.full_like(self.alive, self.death_rew),
              torch.where(alive_now, torch.full_like(self.alive, self.alive_rew),
                          torch.zeros_like(self.alive)))
        rew = rew + (EN_DMG_REW * dmg + IT_REW * n_got)[:, None] * alive_now.float()
        rew = rew + PWR_STAND_REW * (self.power / POWER_MAX) * alive_now.float()
        # v28: small penalty in the bottom ~12% (linear, 0 at BOTTOM_Y)
        bot = ((self.player[:, 1:2] - BOTTOM_Y) / (PY_HI - BOTTOM_Y)).clamp(min=0.0)
        rew = rew - BOTTOM_PEN * bot * alive_now.float()
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
