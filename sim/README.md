# sim/ — GPU danmaku sim → sim-to-real transfer

Train a dodging + shooting policy in a fast made-up-danmaku environment on the
GPU, then transfer it to the real game. Uses the `.venv-cuda` venv (CUDA PyTorch).

Motivation: after ~10 from-scratch runs on the live game (PPO and evolution),
policies plateaued at "freeze in a safe-ish spot". The live game gives one
scalar of feedback per ~20 s episode at ~1000× real-time — too weak/slow for a
reactive reflex. A vectorised sim runs thousands of episodes in lockstep at
~100k+ env-steps/s and lets PPO learn per-timestep.

## Files

| file | what |
|---|---|
| `physics_probe.py` | measure the real game's move speed, bounds, bullet speeds → `physics.json` |
| `danmaku.py` | the vectorised env. Batched tensors, `torch.compile`d, auto-reset. Fixed procedural stage (below). Observations via the shared `native/obs.py` builder → bit-identical inputs to `Th07Env`. |
| `train.py` | `--algo ppo` (default) / `es`. Actor arch == `MLPPolicy` so `best.pt` loads into the real env. Writes `runs_sim/<name>/{best,last}.pt`, `history.npy`, `meta.json`. Logs greedy survival (median / p90 / >60-120-180 s, capped at the 240 s episode length), sampled survival, entropy, death-cause split (spam / enemy / rest=emitter). `--eager-sim` disables compile. `--hidden 256 256` etc. |
| `hud.py` | live training dashboard (p90+median survival vs steps) — `python sim/hud.py <run>` |
| `watch_sim.py` | watch the sim policy play; **takes a path to `best.pt`**, `--follow` reloads it. Bullets drawn at true hitbox size, coloured by source. `g` local grid, `G` macro tracks |
| `transfer.py` | run a sim-trained `best.pt` on real Lunatic stage 1 (`.venv`, not `.venv-cuda`) |
| `../native/probe_bullets.py` | measure real th07 bullet hitboxes live |
| `../native/probe_cirno.py` | check where the stage-1 midboss/boss lives in memory |

## The observation (`native/obs.py`, `OBS_DIM = 236`)

| part | size | what |
|---|---|---|
| head | 16 | player pos/vel, focus, power, nearest-bullet dist, boss/midboss HP |
| escape scalars | 9 | for {stay, 8 dirs}: frames-until-hit if held 20 frames, `/20` |
| local danger grid | 13×13 | player-centred (±78 px); bullet-strike imminence, marched 24 frames; OOB cells = 0.5 |
| enemies | 6×3 | **nearest** on-screen enemies (rel pos, hp fraction) — includes the midboss/boss |
| items | 8×3 | nearest items (rel pos, type) |

A "global danger map" was tried (obs → 404) and dropped — the policy ignored it.

## The sim stage (`ROSTER` in `danmaku.py`)

- 4 corners × (**CONE** + **SPRAY**), placed **outside** the playfield so fire
  comes *in* from the corner — no safe pocket
- one fast sweeping **LINE** (bottom-right)
- one **BRING** dense-ring emitter that **bounces** the interior + one that
  **orbits** the perimeter (anti-camp); slow bullets, 5 s life cap
- top-right CONE **redirect**: its bullets get one 50 % roll at t=1 s to snap to
  a random heading — breaks the "all bullets go straight" assumption

**Bullet hitboxes are the real th07 values** (`native/probe_bullets.py` +
disasm): collision is an AABB overlap, `dist < hitbox + PLAYER_HB` circular
approx. Per-slot constant (`_slot_rad`): **pellet 2.0**, **ball 3.0**, player
**1.8**. All emitter + enemy-burst bullets are `ball`; spam pellets are `pellet`.

### Enemies (`EN_*`)

Waves of 9–15 fly in every 12 s, hover ~6 s, leave. 1 HP; body contact kills
(`EN_RADIUS 13`, a caution bias). Each hovering enemy fires **2 aimed bursts**
(4 `ball` bullets, ~24° fan, snapshot-aimed at the player — no tracking).

**SHOOT is FRONT-ONLY** — only hits an enemy within ±26 px of the player's x and
above it, nearest first. No auto-aim (that taught "shooting is free" and didn't
transfer). Damage `EN_DPS × power_mult`; `EN_DMG_REW 0.35` per HP.

Kills drop 4 **P items** → `IT_REW 0.60` + raise **power** (`+2.5/128`), which
scales damage 1× → 3× and widens the target count. `PWR_STAND_REW 0.0015`/frame ×
power-fraction makes held power lastingly worth it.

### Spam phase

Every 45–60 s: **N spawners** (3, +1 each phase, cap 6) near the **top** of the
screen drift left/right and rain **20 pellets/attack, 5/s, for 10 s** in a
downward ~150° fan. Free-slot pool (1600) — write into inactive / least-
threatening slots, and a past-player cull frees pellets that fall below the
player. **All other fire + enemy waves pause** for the phase + a 3 s cooldown.

## Status

| run | notes | sim greedy | real Lunatic stage 1 |
|---|---|---|---|
| `ppo_v12` | early layout, obs 212, auto-aim | ~23 s med | **470 s / 1.63 M** (stages 1–3) |
| `ppo_v21` | reverted 8-corner stage, obs 236, 180 s cap | med ~42 / p90 ~139 @ 429M | 105 s / 15 k — camps a corner |
| `ppo_v22` | + spam phase, real hitboxes, corners in | med ~78 / p90 180 @ 82M (too easy) | **368 s / 1.58 M** @ 82M |
| `ppo_v25` | + front-only shot, enemy bursts, top-down spam, free-slot pool | med ~77 / p90 180 @ 1000M (plateaued ~500M) | 159 s / **100 k** — front-only + weak rewards → stopped engaging |
| `ppo_v26` | + `EN_DMG_REW` 0.35, `IT_REW` 0.6, power reward, `EN_RADIUS` 13, `[256,256]`, 240 s cap, death split | *training* | — |

`ppo_v20` (turrets outside + spinning centre wheels + no auto-aim) was abandoned
— sparse stage left a bottom-centre camp spot.

**Known:** v25 transfer confirmed the front-only shot + weak kill/P rewards make
it stop playing the game (just survives). v26 rebalances. Also: the stage-1
midboss (Cirno) and boss (Letty) live in `EM_BOSSES[0]`, **not** `EM_ENEMIES` —
`env.py` now dereferences that pointer so they're visible/targetable (needed
because front-only shot has to position under a target it can see).

## Notes

- `torch.compile` on `_advance` + `build_obs_batch`; **not** the policy net.
- Memory: `b_rad` dropped (per-slot `_slot_rad`), `b_age`/`b_redir` fp16. B=24576
  fits ~11 GB; B≥28672 froze the desktop (VRAM saturation).
- Transfer uses the Python step loop, so the obs matches training regardless of
  the DLL build. `env.reset()` only rewinds stage-1 snapshots — use
  `--episodes 1 --until-death` for deep watches.
- Constants in `native/obs.py` must match `native/th07hook.cpp` for the in-DLL
  fast eval / real-game fine-tuning.
