"""Canonical batched observation builder - the single source of truth for what
the policy sees.

Used by:
  * native/env.py         real game, B = 1
  * sim/danmaku.py        made-up-danmaku trainer, B = thousands (GPU)

Must stay in lockstep with th07hook.cpp `build_obs` (the C version the live
trainer actually runs); the parity smoke test checks env._obs vs the DLL's
dbg_obs byte-for-byte on the grid and ~1e-3 on the escape scalars.

Layout (OBS_DIM = 212):
  [0:16)    head      - px/W, py/H, pvx/6, pvy/6, focus, lives/9, bombs/9,
                        power/128, tanh(graze/100), stage/6, alive, dead,
                        near_d, boss_present, boss_frac, 0
  [16:25)   escape    - for {stay,N,NE,E,SE,S,SW,W,NW}: frames-until-hit if the
                        player holds that move for DIR_HORIZON frames, / DIR_HORIZON
  [25:194)  grid      - 13x13 player-centred danger grid (imminence of a strike
                        per cell from marching every bullet; walls read 0.5)
  [194:212) enemies   - 6 * (rel_x/128, rel_y/128, life/maxlife); zeros in the sim
"""
from __future__ import annotations

import torch

W, H = 384.0, 448.0
# measured (sim/physics.json): the player stops at these coords at each edge
PX_LO, PX_HI, PY_LO, PY_HI = 8.0, 376.0, 16.0, 432.0
GRID = 13
GRID_R = GRID // 2
GCELLS = GRID * GRID
GRID_CELL = 12.0
GRID_HORIZON = 24.0
HEAD_DIM = 16
NDIRS = 9
DIR_SPEED = 4.0          # measured unfocused player move speed (px/frame)
DIR_HORIZON = 20.0
DIR_HIT_R = 7.0          # player_r (~2) + typical stage-1 bullet_r (~4), a touch generous
K_NEAREST = 128          # only the K nearest bullets feed the grid + escape scan
                         # (the rest are too far to touch the +-78px window). DLL
                         # build_obs must apply the same cap for transfer parity.
M_ENEMIES = 6
OBS_DIM = HEAD_DIM + NDIRS + GCELLS + M_ENEMIES * 3   # 212

# index 0 = stay still, then the 8 compass dirs (matches decode_action / _DIRS)
OBS_DIRS = torch.tensor(
    [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]],
    dtype=torch.float32,
)
_MARCH_T = torch.arange(0.0, GRID_HORIZON + 0.5, 0.5)   # 49 steps, t = 0..24 by 0.5


@torch.no_grad()
def build_obs_batch(player_pos, player_vel, player_focus,
                    bullets_pos, bullets_vel, bullets_active,
                    head_aux, enemies):
    """
    player_pos     [B,2]   playfield coords (px, py), origin top-left
    player_vel     [B,2]   per-frame velocity (head only; 0 is fine)
    player_focus   [B]     0/1
    bullets_pos    [B,N,2]
    bullets_vel    [B,N,2]  per-frame
    bullets_active [B,N]    bool / 0-1 mask
    head_aux       [B,9]    lives/9, bombs/9, power/128, tanh(graze/100),
                            stage/6, alive, dead, boss_present, boss_frac
    enemies        [B,18]   6 * (rel_x/128, rel_y/128, life/maxlife); zeros in sim
    returns        [B,212]
    """
    dev = player_pos.device
    dt = player_pos.dtype
    B = player_pos.shape[0]
    px = player_pos[:, 0:1]
    py = player_pos[:, 1:2]
    act_all = bullets_active.to(torch.bool)

    # keep only the K nearest active bullets - everything downstream is O(B*K)
    N = bullets_pos.shape[1]
    d_all = (bullets_pos - player_pos[:, None, :]).norm(dim=2)
    d_all = torch.where(act_all, d_all, torch.full_like(d_all, 1e9))
    if N > K_NEAREST:
        d_k, sel = torch.topk(d_all, K_NEAREST, dim=1, largest=False)
        bpos = torch.gather(bullets_pos, 1, sel[:, :, None].expand(-1, -1, 2))
        bvel = torch.gather(bullets_vel, 1, sel[:, :, None].expand(-1, -1, 2))
        act = d_k < 1e8
    else:
        bpos, bvel, act = bullets_pos, bullets_vel, act_all

    # recycled-slot guard (matches the DLL: if either component blows up, drop both)
    bad = (bvel.abs() > 24.0).any(dim=-1, keepdim=True)
    bv = torch.where(bad, torch.zeros_like(bvel), bvel)
    bullets_pos = bpos

    dirs = OBS_DIRS.to(dev, dt)
    tsteps = _MARCH_T.to(dev, dt)
    o = torch.zeros(B, OBS_DIM, device=dev, dtype=dt)

    # ---------- danger grid ----------
    ci = torch.arange(GRID, device=dev)
    cgy, cgx = torch.meshgrid(ci, ci, indexing="ij")
    cdx = (cgx.reshape(-1).to(dt) - GRID_R) * GRID_CELL      # [GC]
    cdy = (cgy.reshape(-1).to(dt) - GRID_R) * GRID_CELL
    wx = px + cdx                                            # [B,GC]
    wy = py + cdy
    grid = torch.zeros(B, GCELLS, device=dev, dtype=dt)
    wall = (wx < PX_LO) | (wx > PX_HI) | (wy < PY_LO) | (wy > PY_HI)
    grid = torch.where(wall, torch.full_like(grid, 0.5), grid)

    for i in range(tsteps.shape[0]):
        t = tsteps[i]
        bp = bullets_pos + bv * t                            # [B,N,2]
        rel = (bp - player_pos[:, None, :]) / GRID_CELL
        cell = torch.floor(rel + 0.5).long() + GRID_R        # [B,N,2]
        gx = cell[..., 0]
        gy = cell[..., 1]
        valid = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID) & act
        lin = gy.clamp(0, GRID - 1) * GRID + gx.clamp(0, GRID - 1)   # [B,N]
        dval = torch.full_like(lin, float(1.0 - t / GRID_HORIZON), dtype=dt)
        dval = torch.where(valid, dval, torch.zeros_like(dval))
        grid.scatter_reduce_(1, lin, dval, reduce="amax")
    o[:, HEAD_DIM + NDIRS:HEAD_DIM + NDIRS + GCELLS] = grid

    # ---------- escape scalars ----------
    L = dirs.norm(dim=1, keepdim=True)
    unit = torch.where(L > 1e-6, dirs / L.clamp(min=1e-9), torch.zeros_like(dirs))
    pm = unit * DIR_SPEED                                    # [9,2]
    pmx = pm[:, 0][None, :]
    pmy = pm[:, 1][None, :]
    big = torch.full((B, NDIRS), 1e9, device=dev, dtype=dt)

    twx = torch.where(pmx > 1e-6, (PX_HI - px) / pmx.clamp(min=1e-6),
          torch.where(pmx < -1e-6, (PX_LO - px) / pmx.clamp(max=-1e-6), big))
    twy = torch.where(pmy > 1e-6, (PY_HI - py) / pmy.clamp(min=1e-6),
          torch.where(pmy < -1e-6, (PY_LO - py) / pmy.clamp(max=-1e-6), big))
    safe = torch.full((B, NDIRS), DIR_HORIZON, device=dev, dtype=dt)
    safe = torch.minimum(safe, torch.minimum(twx, twy)).clamp(min=0.0)

    rb = bullets_pos - player_pos[:, None, :]                # [B,N,2]
    dist = rb.norm(dim=2)                                    # [B,N]
    near = (dist < 150.0) & act
    r0 = -rb                                                 # player - bullet
    rv = pm[None, None, :, :] - bv[:, :, None, :]            # [B,N,9,2]
    a = (rv * rv).sum(-1)                                    # [B,N,9]
    bdot = (r0[:, :, None, :] * rv).sum(-1)                  # [B,N,9]
    ts = torch.where(a < 1e-6, torch.zeros_like(a), -bdot / a.clamp(min=1e-6))
    ts = ts.clamp(min=0.0)
    cp = r0[:, :, None, :] + rv * ts[..., None]              # [B,N,9,2]
    hit = ((cp * cp).sum(-1) < DIR_HIT_R * DIR_HIT_R) & near[:, :, None]
    cand = torch.where(hit, ts, torch.full_like(ts, 1e9))
    safe = torch.minimum(safe, cand.min(dim=1).values).clamp(min=0.0)
    o[:, HEAD_DIM:HEAD_DIM + NDIRS] = safe / DIR_HORIZON

    # ---------- head ----------
    near_d = torch.where(act, dist, torch.full_like(dist, 1e9)).min(dim=1).values
    near_d = torch.where(near_d < 1e8, (near_d / 80.0).clamp(max=3.0),
                         torch.full_like(near_d, 2.0))
    o[:, 0] = px[:, 0] / W
    o[:, 1] = py[:, 0] / H
    o[:, 2] = player_vel[:, 0] / 6.0
    o[:, 3] = player_vel[:, 1] / 6.0
    o[:, 4] = player_focus.to(dt)
    o[:, 5:12] = head_aux[:, 0:7]        # lives, bombs, power, graze, stage, alive, dead
    o[:, 12] = near_d
    o[:, 13:15] = head_aux[:, 7:9]       # boss_present, boss_frac
    o[:, 15] = 0.0
    o[:, HEAD_DIM + NDIRS + GCELLS:] = enemies
    return o
