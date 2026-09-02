"""Canonical batched observation builder - the single source of truth for what
the policy sees.

Used by:
  * native/env.py         real game, B = 1
  * sim/danmaku.py        made-up-danmaku trainer, B = thousands (GPU)

Must stay in lockstep with th07hook.cpp `build_obs` (the C version the live
trainer actually runs); the parity smoke test checks env._obs vs the DLL's
dbg_obs byte-for-byte on the grid and ~1e-3 on the escape scalars.
NOTE: as of v14 the DLL's build_obs is STALE (no global map / item slots) - it
only matters for the in-DLL live-evo path, not for sim training or transfer.

Layout (OBS_DIM = 236):
  [0:16)     head       - px/W, py/H, pvx/6, pvy/6, focus, lives/9, bombs/9,
                          power/128, tanh(graze/100), stage/6, alive, dead,
                          near_d, boss_present, boss_frac, 0
  [16:25)    escape     - for {stay,N,NE,E,SE,S,SW,W,NW}: frames-until-hit if the
                          player holds that move for DIR_HORIZON frames, / DIR_HORIZON
  [25:194)   local grid - 13x13 player-centred danger grid (imminence of a strike
                          per cell from marching every bullet; walls read 0.5)
  [194:212)  enemies    - 6 * (rel_x/128, rel_y/128, life/maxlife)
  [212:236)  items      - 8 * (rel_x/128, rel_y/128, type/9); P-drops etc.

(v18: the 12x14 absolute-coord "global map" was dropped - the policy's
first-layer weights for it stayed flat at the noise floor across v14-v17, i.e.
it never learned to use it, and it was ~40% of the obs + a chunk of build cost.)
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
DIR_SPEED_FOCUS = 1.6    # focused move speed - the escape scan uses this when
                        # player_focus is set (v28: it assumed 4.0 always, so a
                        # focused policy thought it could out-run bullets it can't)
DIR_HORIZON = 20.0
DIR_HIT_R = 7.0          # fallback strike radius when a bullet has no known half-extent
PLAYER_HALF = 0.825    # ReimuA's exact lethal hitbox half-extent, read from
#   th07.exe (FUN_0043e260 collision box = pos +- (shot_root+0x0C)/2 = 0.825;
#   probe_player.py confirms). matches sim PLAYER_HB. strike = bullet_half + this.
K_NEAREST = 128          # only the K nearest bullets feed the local grid + escape scan

M_ENEMIES = 6
M_ITEMS = 8
ITEM_STRIDE = 3

# slice offsets
_O_ESC = HEAD_DIM                              # 16
_O_GRID = _O_ESC + NDIRS                       # 25
_O_ENE = _O_GRID + GCELLS                      # 194
_O_ITEM = _O_ENE + M_ENEMIES * 3              # 212
OBS_DIM = _O_ITEM + M_ITEMS * ITEM_STRIDE      # 236

# index 0 = stay still, then the 8 compass dirs (matches decode_action / _DIRS)
OBS_DIRS = torch.tensor(
    [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]],
    dtype=torch.float32,
)
_MARCH_T = torch.arange(0.0, GRID_HORIZON + 0.5, 0.5)   # 49 steps, t = 0..24 by 0.5
_MARCH_T_PY = tuple(float(x) for x in _MARCH_T)         # plain floats -> torch.compile
#                                                        doesn't graph-break on .item()


@torch.no_grad()
def build_obs_batch(player_pos, player_vel, player_focus,
                    bullets_pos, bullets_vel, bullets_active,
                    head_aux, enemies, items, bullets_half=None):
    """
    player_pos     [B,2]   playfield coords (px, py), origin top-left
    player_vel     [B,2]   per-frame velocity (head only; 0 is fine)
    player_focus   [B]     0/1
    bullets_pos    [B,N,2]
    bullets_vel    [B,N,2]  per-frame
    bullets_active [B,N]    bool / 0-1 mask
    head_aux       [B,9]    lives/9, bombs/9, power/128, tanh(graze/100),
                            stage/6, alive, dead, boss_present, boss_frac
    enemies        [B,18]   6 * (rel_x/128, rel_y/128, life/maxlife)
    items          [B,24]   8 * (rel_x/128, rel_y/128, type/9)
    returns        [B,236]
    """
    dev = player_pos.device
    dt = player_pos.dtype
    B = player_pos.shape[0]
    px = player_pos[:, 0:1]
    py = player_pos[:, 1:2]
    act_all = bullets_active.to(torch.bool)

    N = bullets_pos.shape[1]
    d_all = (bullets_pos - player_pos[:, None, :]).norm(dim=2)
    d_all = torch.where(act_all, d_all, torch.full_like(d_all, 1e9))

    def _guard(v):
        bad = (v.abs() > 24.0).any(dim=-1, keepdim=True)
        return torch.where(bad, torch.zeros_like(v), v)

    if bullets_half is None:
        bullets_half = torch.full_like(bullets_active, DIR_HIT_R - PLAYER_HALF,
                                      dtype=dt)
    else:
        bullets_half = bullets_half.to(dt)

    # nearest-K for the local grid + escape scan
    if N > K_NEAREST:
        d_k, sel = torch.topk(d_all, K_NEAREST, dim=1, largest=False)
        bpos = torch.gather(bullets_pos, 1, sel[:, :, None].expand(-1, -1, 2))
        bvel = torch.gather(bullets_vel, 1, sel[:, :, None].expand(-1, -1, 2))
        bhalf = torch.gather(bullets_half, 1, sel)
        act = d_k < 1e8
    else:
        bpos, bvel, act = bullets_pos, bullets_vel, act_all
        bhalf = bullets_half
    bv = _guard(bvel)
    strike = (bhalf + PLAYER_HALF).clamp(min=1.0)                  # [B, K]

    dirs = OBS_DIRS.to(dev, dt)
    o = torch.zeros(B, OBS_DIM, device=dev, dtype=dt)

    # ---------- local danger grid ----------
    ci = torch.arange(GRID, device=dev)
    cgy, cgx = torch.meshgrid(ci, ci, indexing="ij")
    grid = torch.zeros(B, GCELLS, device=dev, dtype=dt)
    cdx = (cgx.reshape(-1).to(dt) - GRID_R) * GRID_CELL
    cdy = (cgy.reshape(-1).to(dt) - GRID_R) * GRID_CELL
    wx = px + cdx
    wy = py + cdy
    wall = (wx < PX_LO) | (wx > PX_HI) | (wy < PY_LO) | (wy > PY_HI)
    grid = torch.where(wall, torch.full_like(grid, 0.5), grid)

    # per-bullet danger weight: a big Lingering-Cold crystal (strike ~7) marks its
    # cell harder than a pellet (strike ~4). Single-cell stamp - the 12 px grid is
    # too coarse for a spatial footprint to be worth 5x the march cost.
    sz = (strike * (1.0 / 3.83)).clamp(0.7, 1.6)  # [B,K]  3.83 = a ball's strike -> 1.0
    for t in _MARCH_T_PY:                       # plain float t -> no .item() break
        bp = bpos + bv * t
        rel = (bp - player_pos[:, None, :]) / GRID_CELL
        cell = torch.floor(rel + 0.5).long() + GRID_R
        gx = cell[..., 0]
        gy = cell[..., 1]
        valid = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID) & act
        lin = gy.clamp(0, GRID - 1) * GRID + gx.clamp(0, GRID - 1)
        dval = valid.to(dt) * (1.0 - t / GRID_HORIZON) * sz
        grid.scatter_reduce_(1, lin, dval, reduce="amax")
    o[:, _O_GRID:_O_ENE] = grid.clamp(max=1.6)

    # ---------- escape scalars ----------
    L = dirs.norm(dim=1, keepdim=True)
    unit = torch.where(L > 1e-6, dirs / L.clamp(min=1e-9), torch.zeros_like(dirs))
    # v28: focus-aware move speed - [B,1]
    sp = torch.where(player_focus.reshape(B, 1).to(dt) > 0.5,
                     torch.tensor(DIR_SPEED_FOCUS, device=dev, dtype=dt),
                     torch.tensor(DIR_SPEED, device=dev, dtype=dt))
    pm = unit[None, :, :] * sp[:, :, None]           # [B, NDIRS, 2]
    pmx = pm[:, :, 0]                                 # [B, NDIRS]
    pmy = pm[:, :, 1]
    big = torch.full((B, NDIRS), 1e9, device=dev, dtype=dt)

    twx = torch.where(pmx > 1e-6, (PX_HI - px) / pmx.clamp(min=1e-6),
          torch.where(pmx < -1e-6, (PX_LO - px) / pmx.clamp(max=-1e-6), big))
    twy = torch.where(pmy > 1e-6, (PY_HI - py) / pmy.clamp(min=1e-6),
          torch.where(pmy < -1e-6, (PY_LO - py) / pmy.clamp(max=-1e-6), big))
    safe = torch.full((B, NDIRS), DIR_HORIZON, device=dev, dtype=dt)
    safe = torch.minimum(safe, torch.minimum(twx, twy)).clamp(min=0.0)

    rb = bpos - player_pos[:, None, :]
    dist = rb.norm(dim=2)
    near = (dist < 150.0) & act
    r0 = -rb
    rv = pm[:, None, :, :] - bv[:, :, None, :]        # [B, N, NDIRS, 2]
    a = (rv * rv).sum(-1)
    bdot = (r0[:, :, None, :] * rv).sum(-1)
    ts = torch.where(a < 1e-6, torch.zeros_like(a), -bdot / a.clamp(min=1e-6))
    ts = ts.clamp(min=0.0)
    cp = r0[:, :, None, :] + rv * ts[..., None]
    hit = ((cp * cp).sum(-1) < (strike * strike)[:, :, None]) & near[:, :, None]
    cand = torch.where(hit, ts, torch.full_like(ts, 1e9))
    safe = torch.minimum(safe, cand.min(dim=1).values).clamp(min=0.0)
    o[:, _O_ESC:_O_GRID] = safe / DIR_HORIZON

    # ---------- head ----------
    near_d = torch.where(act, dist, torch.full_like(dist, 1e9)).min(dim=1).values
    near_d = torch.where(near_d < 1e8, (near_d / 80.0).clamp(max=3.0),
                         torch.full_like(near_d, 2.0))
    o[:, 0] = px[:, 0] / W
    o[:, 1] = py[:, 0] / H
    o[:, 2] = player_vel[:, 0] / 6.0
    o[:, 3] = player_vel[:, 1] / 6.0
    o[:, 4] = player_focus.to(dt)
    o[:, 5:12] = head_aux[:, 0:7]
    o[:, 12] = near_d
    o[:, 13:15] = head_aux[:, 7:9]
    o[:, 15] = 0.0
    o[:, _O_ENE:_O_ITEM] = enemies
    o[:, _O_ITEM:] = items
    return o
