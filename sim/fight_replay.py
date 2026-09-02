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
#   dynamic=True: one compile covers every batch size (training B *and* the
#   eval sim's smaller B / the two aim-pool pool sizes) - dynamic=False was
#   doing a fresh ~6-min max-autotune compile per shape, four of them, which
#   made the first ~25 min of a run look like it had stalled.
#   mode default (not max-autotune): the win here is fusing launch overhead,
#   not per-kernel tuning, and default compiles in seconds.
if os.environ.get("FIGHTSIM_NOCOMPILE"):
    build_obs_batch = _build_obs_eager
else:
    build_obs_batch = torch.compile(_build_obs_eager, dynamic=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boss_phases import phase_windows, has_phases, KILL_FRAC   # noqa: E402
MAX_PHASES = 6
MIR_AXIS = 384.0                 # x-mirror reflects about the playfield centre

FIGHTS = Path(__file__).resolve().parent / "fights"
POOL = 1025                      # th07 bullet pool size
PW, PH = 384.0, 448.0           # playfield size (re-aimed bullets despawn off it)
MAX_EN = 48                      # max satellite sub-enemies tracked per frame
PLAYER_HB = 1.8                  # measured player half-extent
BULLET_HB_DEFAULT = 2.5          # fallback when a recording has no hitbox column
ENEMY_BODY_SCALE = 2.0 / 3.0     # player-body vs enemy-body box (pytouhou)

# episode spawn box - randomised so the policy can't memorise one trajectory
SPAWN_X_LO, SPAWN_X_HI = 40.0, 344.0
SPAWN_Y_LO, SPAWN_Y_HI = 160.0, 420.0

# synthetic damage-phasing (ReimuA).  Values are read straight from th07.exe's
# shot descriptor table (`native/probe_shot_damage.py --which 2 --dumponly`),
# which the engine walks by raw power (0-128) to pick a tier.  Per tier:
# forward needles (type 0, every 5f) land only lined up in x; homing amulets
# (type 1, every 15-30f) land from anywhere.  FUN_0043d9e0 sums descriptor+0x1c
# per overlapping shot (no piercing); FUN_00420620 hard-clamps boss loss to
# 70/frame.  (A situational /3 when player+0x16a20 != 0 - option-deploy startup
# window - not modelled.)
#
#   raw power   lined HP/f  homing HP/f   needles     (tier boundary = shot upgrade)
#     0-7          10.5        0.9        1 x48
#     8-15         12.9        0.9        2 x30
#     16-31        13.9        1.9        2 x30
#     32-47        19.1        1.9        3 x~29    <- our runs sit ~here (power ~35)
#     48-63        18.7        1.5        3 x~29
#     64-95        ~19.3       ~2.2       3 x~29
#     96-127       19.5        2.3        3 x~29
#     128 (max)    25.9        2.7        4 x~29    <- top replays reach ~105, never 128
#
# A real Lunatic run reaches Letty at ~power 35 (us) .. ~105 (top replays), so
# training samples raw power uniformly in [POWER_LO, POWER_HI] per episode and
# derives that episode's lined / homing rate from the table.  Cross-check: a
# drive-policy --shoot recording (`sim/measure_dps`) at power ~16-32 drains
# ~11-13 HP/f effective (lands ~55% of frames); it can't cleanly split lined vs
# homing (a moving policy leaves the x-lane in the ~8f a needle takes to land),
# so the table is the authority on the split.
POWER_LO, POWER_HI = 10.0, 50.0   # raw power (0-128) sampled per episode
_PWR_THRESH = (8.0, 16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 128.0)          # shot-upgrade steps
_PWR_LINED  = (10.53, 12.93, 13.87, 19.07, 18.73, 19.00, 19.67, 19.47, 25.87)  # HP/f lined
_PWR_HOMING = (0.93, 0.93, 1.87, 1.87, 1.53, 2.00, 2.47, 2.27, 2.67)           # HP/f homing
SHOT_DPS = 19.1                  # reference (power ~35) - the real value is per-episode
_SYNTH_DPS = 14.0               # only for recs with no phase_hp (the old replay path);
#   ECL schedules carry real Part-7 thresholds so this is unused there
DMG_CLAMP = 70.0                 # th07.exe FUN_00420620: boss HP loss is hard-
#   clamped to 70/frame regardless of shot power (decompiled)
HOMING_FRAC = 0.10              # reference - the real value is per-episode from _PWR_*
LANE_HALF = 24.0               # |px - boss_x| under this -> the forward needles land
#   (needle muzzles are +-8 from player centre; Letty body half ~16)

# reward: "kill the boss, don't get hit". Terms:
#   * SURV_REW / frame alive - a SMALL floor so the 60-70% of episodes that die
#     before they can kill still get a "don't die in Lingering Cold" gradient.
#     Kept ~30x smaller than aligned-shooting damage so it can't cause stalling.
#   * DMG_REW * HP drained   - the dense progress signal, guides it to shoot
#   * KILL_BONUS on defeat   - "finish her"
#   * -HIT_PEN on death      - the big stick
SURV_REW = 0.004             # reward += this every frame alive
DMG_REW = 0.012               # reward += this * boss-HP drained this frame
HIT_PEN = 12.0                # reward -= this on death (bullet or enemy body)
KILL_BONUS = 120.0           # reward += this on final-phase defeat

# --- anti-memorisation randomisation (per episode) ---
# The whole danmaku field is rotated RIGIDLY about screen-centre by a random
# angle each episode. A rigid rotation preserves every trajectory and every gap
# exactly (no bullet's lifetime or shape changes), so it's the safe way to make
# absolute positions unmemorisable - unlike per-bullet re-aim, which desynced
# bullets from the recording's slot lifecycle and made them vanish/flicker.
FIELD_ROT_DEG = 10.0
FIELD_ROT = np.radians(FIELD_ROT_DEG)
CX, CY = 192.0, 192.0         # rotation centre (approx Letty's home position)
DPS_LO, DPS_HI = 0.7, 1.0      # per-episode shot-connection rate on top of the
#                               per-power table: needle travel time (~8f), boss
#                               drift, imperfect lane-holding. A camping policy
#                               lands most needles; 0.7 is the pessimistic floor.
NOPHASE_FRAC = 0.2            # fraction of episodes where shooting deals 0 damage
#                               (pure survival - forces training on the full
#                               phase, not just the part a fast kill skips)

_DIRS = torch.tensor([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)


def _load_dense(npz_path):
    """npz -> dict of dense [F, POOL(...)] arrays: pos, half, spawn (spawn point
    of the bullet in each slot), aimed (bool), rec_ang (angle spawn->rec player
    at spawn), plus boss[F,2] and en[F,MAX_EN,3]."""
    d = np.load(npz_path)
    b = d["bullets"]
    if len(b) == 0:
        return None
    f = b[:, 0].astype(np.int64)
    f0 = int(f.min())                # the bullet origin - EVERYTHING aligns here.
    f -= f0                          # (boss/player/enemy logs can start earlier -
    F = int(f.max()) + 1             #  a bullet-less lead-in - so aligning each to
    slot = np.clip(b[:, 1].astype(np.int64), 0, POOL - 1)   # its own .min() was a
    #                                  40s misalignment that wrecked re-aiming.

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
        bf = d["boss"][:, 0].astype(np.int64) - f0
        m = (bf >= 0) & (bf < F)
        boss[bf[m]] = d["boss"][m, 1:3]
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
        ef = lethal_all[:, 0].astype(np.int64) - f0
        keep = (ef >= 0) & (ef < F)
        lethal_all, ef = lethal_all[keep], ef[keep]
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
    s0 = 0
    if len(busy) and busy[0] > 30:
        s0 = max(0, busy[0] - 30)
    stem = Path(npz_path).stem
    # phase windows from the real screen-clears in THIS recording (trim-relative)
    phw = phase_windows(stem, nb[s0:])
    return dict(pos=pos[s0:], half=half[s0:], boss=boss[s0:], en=en[s0:],
               trim=s0, name=stem, phases=phw)


def _mirror_rec(r):
    """x-flipped copy of a recording (about the playfield centre). Letty's
    patterns are near-symmetric, so this doubles the effective training set for
    free. Timing (phases/birth) is unchanged; only x-coords + angles flip."""
    m = dict(r)
    for k in ("pos", "boss", "en"):
        a = r[k].copy()
        a[..., 0] = MIR_AXIS - a[..., 0]
        m[k] = a
    m["name"] = r["name"] + "_mir"
    return m


class FightSim:
    def __init__(self, B=8192, name="cirno", device="cuda", max_frames=11000,
                 min_start=0, seed=0, phase_start_mix=0.5, mirror=True,
                 randomize=True, recs=None, field_rot_deg=FIELD_ROT_DEG,
                 power_lo=POWER_LO, power_hi=POWER_HI, immortal=False):
        self.B, self.d = B, device
        self.randomize = randomize        # eval instance passes False (clean metric)
        self.power_lo, self.power_hi = float(power_lo), float(power_hi)
        self.immortal = immortal          # viz only: bullets/bodies don't end the run
        self._field_rot = np.radians(field_rot_deg)   # 0 for the ECL sim: fresh
        #   RNG per schedule already unmemorisable, and rotation distorts the
        #   real danmaku geometry + the aim-pool emitter frame
        self._g = torch.Generator(device="cpu").manual_seed(seed)
        if recs is None:                  # load recordings; else use pre-built
            paths = sorted(glob.glob(str(FIGHTS / f"{name}*.npz")))   # dicts (Part 12
            recs = [_load_dense(p) for p in paths]                     # ECL schedules)
        recs = [r for r in recs if r is not None and r["pos"].shape[0] > 600]
        assert recs, f"no usable recordings matching {name}*"
        if mirror:
            recs = recs + [_mirror_rec(r) for r in recs]   # x-flipped copies
        self.n_rec = len(recs)
        self.rec_len = torch.tensor([r["pos"].shape[0] for r in recs])
        self.maxF = min(max(r["pos"].shape[0] for r in recs), max_frames)

        nF = (self.n_rec, self.maxF, POOL)
        self.pos = torch.full((*nF, 2), float("nan"))
        self.bhalf = torch.full(nF, BULLET_HB_DEFAULT)
        self.boss = torch.full((self.n_rec, self.maxF, 2), 192.0)
        self.en = torch.full((self.n_rec, self.maxF, MAX_EN, 3), float("nan"))
        for i, r in enumerate(recs):
            L = min(r["pos"].shape[0], self.maxF)
            self.pos[i, :L] = torch.from_numpy(r["pos"][:L])
            self.bhalf[i, :L] = torch.from_numpy(r["half"][:L])
            self.boss[i, :L] = torch.from_numpy(r["boss"][:L])
            self.en[i, :L] = torch.from_numpy(r["en"][:L])
        for k in ("pos", "bhalf", "boss", "en"):
            setattr(self, k, getattr(self, k).to(device))
        self.rec_len_gpu = self.rec_len.to(device)  # step() reads it without a sync
        self.min_start = min_start
        # per-recording phase table [n_rec, MAX_PHASES, 5]:
        #   (clear_start, first_attack, phase_end, synthetic_hp, dmg_mult)  the
        #   frame indices are trim-relative. clear_start = where t0 jumps to on
        #   advancing INTO this phase; first_attack = when its bullets begin
        #   (armored until then); phase_end = when its attack ends. dmg_mult = 1
        #   for nonspells, 1/7 for Letty's spell cards (the engine's spellcard-
        #   active damage divisor - FUN_00420620 ~13817).
        self.phasing = has_phases(recs[0]["name"]) and recs[0]["phases"] is not None
        self.phase_start_mix = phase_start_mix   # fraction of resets that begin
        #   at a random later phase (curriculum so phases 2+ get trained while
        #   the policy still dies early); eval instance passes 0.0.
        self.ph = torch.zeros(self.n_rec, MAX_PHASES, 5)
        self.n_ph = torch.ones(self.n_rec, dtype=torch.long)
        if self.phasing:
            for i, r in enumerate(recs):
                pw = r["phases"] or [(0, 0, self.maxF)]
                self.n_ph[i] = min(len(pw), MAX_PHASES)
                hp_real = r.get("phase_hp")          # Part 7 thresholds (ECL schedules)
                dm = r.get("phase_dmg_mult")         # 1/7 for spell phases
                for j, (cs, fa, e) in enumerate(pw[:MAX_PHASES]):
                    cs = max(0, min(cs, self.maxF - 4))
                    fa = max(cs, min(fa, self.maxF - 3))
                    e = min(self.maxF - 2, max(fa + 1, e))
                    hp = (hp_real[j] if hp_real and j < len(hp_real)
                          else _SYNTH_DPS * (e - fa) * KILL_FRAC)
                    mult = dm[j] if dm and j < len(dm) else 1.0
                    self.ph[i, j] = torch.tensor([cs, fa, e, hp, mult],
                                                 dtype=torch.float32)
        self.ph_cpu = self.ph.clone()
        self.ph = self.ph.to(device)
        self.n_ph_gpu = self.n_ph.to(device)

        n_en = int((~torch.isnan(self.en[..., 0])).any(-1).float().mean() * 100)
        print(f"[FightSim] {self.n_rec} recs, {self.maxF} frames, B={B}, "
              f"enemy bodies ~{n_en}% of frames, field-rot +-{np.degrees(self._field_rot):.0f}deg, "
              f"phasing {'on' if self.phasing else 'off'} "
              f"({int(self.n_ph.float().mean())} phases)", flush=True)
        self._dirs = _DIRS.to(device)
        from aim_pool import AimPool          # aimed shots re-generated toward the
        self.aim = AimPool(recs, self.B, device)   # live policy each episode
        if self.aim.active_any:
            print(f"[FightSim] aim pool on ({int(self.aim.src_n.float().mean())} "
                  f"aimed shots/schedule)", flush=True)
        self.reset()

    def _sample_starts(self, idx):
        n = len(idx)
        rid = torch.randint(0, self.n_rec, (n,), generator=self._g)
        if self.phasing:
            npj = self.n_ph[rid]                                    # [n]
            rand_later = torch.rand(n, generator=self._g) < self.phase_start_mix
            pj = torch.where(
                rand_later,
                (torch.rand(n, generator=self._g) * npj).long().clamp(max=npj - 1),
                torch.zeros(n, dtype=torch.long))
            row = self.ph_cpu[rid, pj]                              # [n, 4]
            cs, hp = row[:, 0], row[:, 3]
            # a rand_later episode starts at its phase's clear-frame with FULL HP
            # - a genuine "you just cleared the previous phase" state - so phases
            # 2+ get gradient signal without ever training on a state that can't
            # occur. phase_start_mix anneals to 0 (train_fight) so the final
            # policy is tuned on real NS1-start runs.
            off = cs.long()
            self.phase_idx[idx] = pj.to(self.d)
            self.boss_hp[idx] = hp.to(self.d)
        else:
            maxstart = (self.rec_len[rid] - 800).clamp(min=self.min_start + 1)
            off = (torch.rand(n, generator=self._g) *
                   (maxstart - self.min_start) + self.min_start).long()
        self.rec_id[idx] = rid.to(self.d)
        self.t0[idx] = off.to(self.d)

    def reset(self, idx=None):
        if idx is None:
            self.rec_id = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.t0 = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.t = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.boss_hp = torch.zeros(self.B, device=self.d)
            self.phase_idx = torch.zeros(self.B, dtype=torch.long, device=self.d)
            self.armored = torch.zeros(self.B, dtype=torch.bool, device=self.d)
            self.survival = torch.zeros(self.B, dtype=torch.bool, device=self.d)
            self.focus = torch.zeros(self.B, device=self.d)   # last action's focus bit
            self.field_rot = torch.zeros(self.B, device=self.d)
            self._rot_cs = torch.ones(self.B, 1, device=self.d)
            self._rot_sn = torch.zeros(self.B, 1, device=self.d)
            self.dps_mult = torch.ones(self.B, device=self.d)
            self.no_phase = torch.zeros(self.B, dtype=torch.bool, device=self.d)
            # per-episode raw power (0-128) -> that tier's lined / homing rate
            self._pwr_thr = torch.tensor(_PWR_THRESH, device=self.d)
            self._pwr_lin = torch.tensor(_PWR_LINED, device=self.d)
            self._pwr_hom = torch.tensor(_PWR_HOMING, device=self.d)
            pw0 = torch.full((self.B,), (self.power_lo + self.power_hi) * 0.5, device=self.d)
            self.power = pw0
            ti0 = torch.bucketize(pw0, self._pwr_thr, right=True)
            self.shot_dps = self._pwr_lin[ti0]
            self.homing_frac = (self._pwr_hom[ti0] / self.shot_dps).clamp(0, 1)
            self.px = torch.full((self.B,), 192.0, device=self.d)
            self.py = torch.full((self.B,), 384.0, device=self.d)
            self._prev_bp = torch.zeros(self.B, POOL, 2, device=self.d)
            self._prev_active = torch.zeros(self.B, POOL, dtype=torch.bool,
                                            device=self.d)
            self._bi = torch.arange(self.B, device=self.d)
            idx = self._bi
        self._sample_starts(idx.cpu())
        self.t[idx] = 0
        self.aim.reset(idx, self.rec_id[idx], self.t0[idx])
        self._aim_now = None
        # random spawn: the policy must handle being anywhere on the lower field,
        # not memorise a path from one fixed point. Kept out of the top strip
        # (too close to the boss) and off the walls.
        k = len(idx)
        rx = torch.rand(k, device=self.d) * (SPAWN_X_HI - SPAWN_X_LO) + SPAWN_X_LO
        ry = torch.rand(k, device=self.d) * (SPAWN_Y_HI - SPAWN_Y_LO) + SPAWN_Y_LO
        self.px[idx] = rx
        self.py[idx] = ry
        self._prev_active[idx] = False
        self.focus[idx] = 0.0
        if self.randomize:
            self.field_rot[idx] = (torch.rand(k, device=self.d) * 2 - 1) * self._field_rot
            self.dps_mult[idx] = DPS_LO + (DPS_HI - DPS_LO) * torch.rand(k, device=self.d)
            self.no_phase[idx] = torch.rand(k, device=self.d) < NOPHASE_FRAC
            pw = self.power_lo + (self.power_hi - self.power_lo) * torch.rand(k, device=self.d)
            self.power[idx] = pw
            ti = torch.bucketize(pw, self._pwr_thr, right=True)
            self.shot_dps[idx] = self._pwr_lin[ti]
            self.homing_frac[idx] = (self._pwr_hom[ti] / self.shot_dps[idx]).clamp(0, 1)
        self._rot_cs = torch.cos(self.field_rot)[:, None]
        self._rot_sn = torch.sin(self.field_rot)[:, None]
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

        # rigid per-episode rotation of the whole danmaku field about (CX,CY).
        # Preserves every trajectory/gap exactly; just makes absolute positions
        # unmemorisable. Applied to bullets AND the lethal enemy orbs.
        cs, sn = self._rot_cs, self._rot_sn                     # [B,1] each
        bdx, bdy = bp[..., 0] - CX, bp[..., 1] - CY
        bp = torch.where(active[..., None], torch.stack(
            [CX + bdx * cs - bdy * sn, CY + bdx * sn + bdy * cs], -1), bp)
        edx, edy = en[..., 0] - CX, en[..., 1] - CY
        en = torch.cat([torch.where(en_active[..., None], torch.stack(
            [CX + edx * cs - edy * sn, CY + edx * sn + edy * cs], -1),
            en[..., :2]), en[..., 2:3]], -1)
        return bp, active, bh, en, en_active, f

    def _boss_xy(self, f):
        """recorded boss position at frame f, rotated by this episode's field rot."""
        b = self.boss[self.rec_id, f]                            # [B, 2]
        dx, dy = b[:, 0] - CX, b[:, 1] - CY
        cs, sn = self._rot_cs[:, 0], self._rot_sn[:, 0]
        return torch.stack([CX + dx * cs - dy * sn, CY + dx * sn + dy * cs], -1)

    def _obs(self, now=None):
        bp, active, bh, en, en_a, f = now if now is not None else self._now()
        both = (active & self._prev_active)[..., None]
        bv = torch.where(both, bp - self._prev_bp, torch.zeros_like(bp))
        self._prev_bp, self._prev_active = bp, active
        if self._aim_now is not None:                # fold in the runtime aim pool
            ap, aa, ah, av = self._aim_now
            bp = torch.cat([bp, ap], 1)
            active = torch.cat([active, aa], 1)
            bv = torch.cat([bv, av], 1)
            bh = torch.cat([bh, ah], 1)
        bx = self._boss_xy(f)                                    # [B, 2] rotated
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
            self.focus,                       # real focus bit (escape scalars use it)
            bp, bv, active.float(), head, enemies, items)

    def step(self, act):
        act = act.long()
        dd = act % 9
        focus = (act // 9) % 2
        self.focus = focus.to(self.focus.dtype)
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

        # runtime aimed-bullet pool: spawn toward the live player, integrate
        self._aim_now = self.aim.step((self.t0 + self.t).long(), self.px, self.py,
                                      self._rot_cs[:, 0], self._rot_sn[:, 0])

        now = self._now()
        bp, active, bh, en, en_a, f = now
        pxy = torch.stack([self.px, self.py], -1)[:, None, :]
        # bullets: per-bullet AABB half + player half
        rel = (bp - pxy).abs()
        r = bh + PLAYER_HB                                       # [B, POOL]
        hit_b = (active & (rel[..., 0] < r) & (rel[..., 1] < r)).any(dim=1)
        ap, aa, ah, _av = self._aim_now
        if ap.shape[1]:
            arel = (ap - pxy).abs()
            ar = ah + PLAYER_HB
            hit_b = hit_b | (aa & (arel[..., 0] < ar) & (arel[..., 1] < ar)).any(1)
        # enemy bodies: (hb/2)*2/3 + player half, AABB
        erel = (en[..., :2] - pxy).abs()
        er = en[..., 2] + PLAYER_HB                              # [B, MAX_EN]
        hit_e = (en_a & (erel[..., 0] < er) & (erel[..., 1] < er)).any(dim=1)
        hit = hit_b | hit_e
        self.last_hit = hit                # viz reads this for a hit flash
        if self.immortal:
            hit = torch.zeros_like(hit)

        ended = (self.t0 + self.t) >= (self.rec_len_gpu[self.rec_id] - 2)
        rew = SURV_REW - HIT_PEN * hit.float()

        # --- damage-phasing: holding shoot drains the CURRENT phase's synthetic
        # HP - at 20% rate anywhere (homing amulets), full rate only when lined
        # up in x under the boss (forward needles). Draining the phase, or its
        # recorded timer expiring, screen-clears the bullets and jumps the
        # recording to the next phase. Beating the last phase = KILL_BONUS.
        killed = torch.zeros(self.B, dtype=torch.bool, device=self.d)
        advance = torch.zeros(self.B, dtype=torch.bool, device=self.d)
        if self.phasing:
            rid = self.rec_id
            cur = self.ph[rid, self.phase_idx]                   # [B, 4]
            p_first, p_end = cur[:, 1].long(), cur[:, 2].long()
            fnow = self.t0 + self.t
            # armored: from the phase's clear (t0) until its first bullet flies -
            # the boss is repositioning / declaring, deals & takes no damage
            armored = fnow < p_first
            # Letty's spell cards (Lingering Cold / Table-Turning) take shot
            # damage at 1/7 (cur[:, 4]) - capturable, but they usually time out.
            spell = cur[:, 4] < 0.999
            self.survival = spell           # viz flag ("spell - reduced damage")
            shoot_bit = (act >= 18) & ~armored & ~self.no_phase
            bx = self._boss_xy(f)[:, 0]                 # rotated boss x this frame
            aligned = (self.px - bx).abs() < LANE_HALF
            hf = self.homing_frac
            dps = (self.shot_dps * self.dps_mult *
                   (hf + (1.0 - hf) * aligned.float()))
            dmg = (dps * shoot_bit.float()).clamp(max=DMG_CLAMP)   # engine cap
            dmg = dmg * cur[:, 4]                                   # spell divisor
            dmg = torch.minimum(dmg, self.boss_hp.clamp(min=0))
            self.boss_hp = self.boss_hp - dmg
            rew = rew + DMG_REW * dmg
            self.armored = armored          # for the viz / diagnostics
            trans = (self.boss_hp <= 0) | (fnow >= p_end) | ended
            last = self.phase_idx >= (self.n_ph_gpu[rid] - 1)
            killed = trans & last
            advance = trans & ~last
            rew = rew + KILL_BONUS * killed.float()

        done = (hit | killed) if self.phasing else (hit | ended | killed)
        self.last_killed = killed          # which done-episodes were boss kills
        obs = self._obs(now)
        if advance.any():
            ai = advance.nonzero(as_tuple=True)[0]
            self.phase_idx[ai] = self.phase_idx[ai] + 1
            nxt = self.ph[self.rec_id[ai], self.phase_idx[ai]]   # [k, 4]
            self.t0[ai] = nxt[:, 0].long().clamp(max=self.maxF - 2)  # clear_start
            self.t[ai] = 0
            self.boss_hp[ai] = nxt[:, 3]                            # synthetic hp
            self._prev_active[ai] = False                        # screen clear
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
