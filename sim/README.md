# sim/ — GPU danmaku sim → sim-to-real transfer

Train a dodging + shooting policy in a fast made-up-danmaku environment on the
GPU, then transfer it to the real game. Uses the `.venv-cuda` venv (CUDA PyTorch).

> **`sim/ecl/`** is a separate, current effort — a from-scratch VM that runs
> each boss's real PCB bytecode to *generate* danmaku, since both the
> made-up-danmaku sim here and the recorded-replay sim (`fight_replay.py`)
> plateau on transfer. Parser + control-flow VM + spawn emission + movement are
> done and verified; bullet motion is next. See
> [`../docs/ecl-vm.md`](../docs/ecl-vm.md) and `python -m sim.ecl.vm_verify`.
> (The old `ecl_vm.py` / `ecl_parse.py` / `ecl_bullet.py` are the shelved first
> attempt — kept for reference, not used.)

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

## Status (made-up-danmaku sim)

| run | notes | real Lunatic transfer |
|---|---|---|
| `ppo_v12` | early layout, obs 212, auto-aim | past the Chen midboss, died before the Stage 2 boss, ~1.63 M score. "470 s" = its sim survival |
| `ppo_v22` | + spam phase, real hitboxes | **~368 s** @ 82M steps |
| `ppo_v25` | front-only shot, weak rewards | 159 s — stopped engaging |
| `ppo_v26` | rebalanced rewards | overfit the fixed stage; transfer got *worse* as sim rose |
| `ppo_v27` | **per-episode domain randomization** (emitter re-roll, motion profiles, sparse windows) | ~200–320 s, clears Letty; peaked ~30–75 M then mild drift |
| `ppo_v28` | + obs normalization | **killed** — the export folded 1e4× weights → 15–40 s |
| `ppo_v29` | v27 + AABB collision + focus-aware escape + reward-norm + LR anneal (NO obs-norm) | peaked ~320 s @ 46 M, settled ~185–245 s. Best real ckpts are early (`snap_0046M` / `snap_0092M`) |

**Recurring failure mode:** every run overfits the fixed/near-fixed sim after
~50–90 M steps — the sim greedy metric keeps climbing while real transfer
plateaus then dips. `best.pt` (ranked by sim score) is by construction the
*most* overfit checkpoint. Fix is real content, not more sim tuning.

## Real-fight replay (`FightSim`)

`native/record_boss_driven.py` drives to a boss and records every live bullet's
`(x, y, vx, vy, class, fx_flag)` per frame → `sim/fights/<name>_*.npz`. The
`zBullet` struct (RE'd in `native/probe_bullet_motion.py`) holds the exact
velocity and the live `bullet_effects` state, so this captures the real pattern
— hangs, accel, curves — with zero interpretation.

`fight_replay.py FightSim` packs N recordings to `[n_rec, F, 1025, 2]` on the
GPU; B episodes each pick a random recording + start offset and replay the exact
bullet positions (velocity diffed per-slot for the obs). Player physics + AABB
collision + the shared `build_obs_batch`. ~66 k steps/s. No re-aiming yet.

`train_fight.py` — PPO on it. `fight_viz.py <name>` — play a recording back.

**Cirno PoC**: `train_fight` reached ~66 s greedy survival on the replays fast
(then memorized the 10 recordings). Against the *real* Cirno, dodge-only, it
scored the same as the made-up-danmaku policy (~66 s, both clear the fight) —
Cirno is too easy to show a difference. Needs a Stage 2+ boss.

## ECL decompilation (`ecl_*.py`, `../tools/th07_ecl/`)

thtk decompiles th07's stage scripts; `ecl_parse.py` + `ecl_vm.py` reconstruct a
boss as a CPU program → bullet spawn schedule. `ecl_bullet.py` ports PyTouhou's
`Bullet.update()`. Runs Cirno/Letty patterns but isn't faithful — TH07 engine
semantics (bullet-type launch data, difficulty coeffs, multi-slot effects)
aren't documented. Kept as a *structure* reference (phases, timings, which
patterns are aimed); the recorder is the ground truth.

`EM_BOSSES[0]` holds the stage-1 midboss (Cirno) + boss (Letty), **not**
`EM_ENEMIES` — `env.py` dereferences that pointer.

## Notes

- `torch.compile` on `_advance` + `build_obs_batch`; **not** the policy net.
- Memory: `b_rad` dropped (per-slot `_slot_rad`), `b_age`/`b_redir` fp16. B=24576
  fits ~11 GB; B≥28672 froze the desktop (VRAM saturation).
- Transfer uses the Python step loop, so the obs matches training regardless of
  the DLL build. `env.reset()` only rewinds stage-1 snapshots — use
  `--episodes 1 --until-death` for deep watches.
- Constants in `native/obs.py` must match `native/th07hook.cpp` for the in-DLL
  fast eval / real-game fine-tuning.
