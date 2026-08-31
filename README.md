# touhouAI

An AI that plays **Touhou 7 – Perfect Cherry Blossom** (`th07`, v1.00b). Started
as "survive + score on Stage 1 Lunatic"; **Stage 1 is now solved** (a sim-trained
policy clears the stage, reaches + usually beats Letty, ~238 s greedy) and the
goal is a **full Lunatic 1-credit clear**.

## Current direction (2026-08)

The GPU made-up-danmaku sim (track 2 below) gets a policy through Stage 1 but
tops out around Stage 2–3 on transfer — generic dodging isn't enough for the
hand-authored Stage 4–6 spellcards.

- **Real-game PPO fine-tuning** — built (`native/ST_ROLLOUT` runs whole
  trajectories in the DLL at ~68×/env, `train_ppo_dll.py`, `sb3_bridge.py`).
  Result: fine-tuning on Stage 1 was a **wash** (238 s → 223 s greedy) — the sim
  policy is already at the ceiling for content it can handle; PPO only helps
  where the policy is *failing*.
- **Import the real boss patterns** — the live path forward.
  - ECL decompilation (`tools/th07_ecl/`, thtk) gives the boss *scripts*; a CPU
    ECL VM (`sim/ecl_vm.py`) reconstructs them, but TH07's engine semantics
    aren't documented well enough to be faithful without weeks of RE.
  - **Recording the real game instead** (`native/record_boss_driven.py`): the
    `zBullet` struct holds each bullet's exact velocity + effect state, so
    reading it every frame captures the real pattern with no interpretation.
    `sim/fight_replay.py` (`FightSim`) replays those recordings on the GPU at
    ~66k steps/s; `sim/train_fight.py` trains on them.
  - **Cirno PoC**: a policy trained on 10 replayed Cirno recordings and a
    made-up-danmaku policy **both** clear the real Cirno fight dodge-only
    (~66 s) — no measurable gain, because Cirno is easy enough that generic
    dodging already handles her. The real test needs a boss the sim policy
    fails (Stage 2+). Pipeline works; harder target needed.

See `sim/README.md` and the memory notes for the detailed history.

- **Perception:** read game state from process memory — exact player / bullet /
  enemy / boss / score data, no computer vision.
- **Control + speed:** a hand-written 32-bit DLL (`native/th07hook.dll`, MinHook)
  injected into `th07.exe` turns the game into a **headless, step-on-demand**
  environment. Rendering and the 60 Hz limiter are bypassed; the whole episode
  (observation build + policy forward + action) runs *inside the DLL* in C, so
  there's one round-trip to Python per episode, not per frame.
- **Project-based:** nothing depends on absolute paths. The only per-machine
  setting is where the game is installed (gitignored). The game install itself
  is not in this repo (copyrighted; `thbgm.dat` alone exceeds GitHub limits).

## The observation

The Python side (and, when rebuilt, the DLL) computes the *same* 236-value
observation via the shared builder `native/obs.py`:

| part | size | what |
|---|---|---|
| head | 16 | player pos/vel, focus, power, nearest-bullet distance, boss/midboss HP |
| **escape scalars** | 9 | for {stay, N, NE, E, SE, S, SW, W, NW}: frames-until-hit if the player holds that move for 20 frames, `/20` |
| **local danger grid** | 13×13 | player-centred (±78 px); each cell = how imminent a bullet strike there is (bullets marched along their paths); out-of-bounds cells = 0.5 |
| enemies | 6×3 | nearest on-screen enemies (rel pos, hp fraction) |
| items | 8×3 | nearest items (rel pos, type) — P drops etc. |

An absolute-coordinate "global danger map" was tried (obs → 404) and dropped —
ablation showed the policy ignored it (flat first-layer weights).

The policy is a tiny MLP (`236 → h → h → 36`, tanh, argmax). 36 actions =
9 directions × focus × shoot.

## Two training tracks

### 1. Live-game neuroevolution — `train_evo.py`

Island-model Deep GA. N game instances = N islands, each with a sub-population;
one Python process work-queues episode evals across all of them (the episode
runs in-DLL, so Python barely participates). Truncation selection + Gaussian
mutation + elitism + ring migration + periodic island restarts.

```
.venv\Scripts\python train_evo.py --name evo1 --islands 12 --hidden 64 64
```

A tkinter HUD (`native/evohud.py`) shows live stats + a population table;
double-click a row to watch that individual. `watch.py runs\evo1\best.pt --evo`
replays a checkpoint in a visible, sound-on window (`--viz` adds the
`native/viz.py` debug overlay: detection box, danger-grid heatmap, action arrow).

Reaches "survives ~35–65 s, actively dodging" but plateaus — one scalar of
feedback per ~20 s episode is a weak signal for learning a reactive reflex.

### 2. GPU sim → transfer — `sim/`

Train a dodging policy in a **fully-vectorized made-up-danmaku sim** on the GPU,
then drop it into the real game.

- `sim/danmaku.py` — thousands of parallel episodes as batched tensors on an
  RTX 5070 Ti. A fixed stage of procedural emitters (cone + spray in every
  corner, placed *outside* the field — the top-right cone's bullets get one 50 %
  chance to snap to a random heading after 1 s — a sweeping line, a bouncing
  ring emitter, and one that orbits). Waves of 9–15 fly-in enemies (1 HP, lethal
  on contact) that each fire 2 aimed bursts; **FRONT-ONLY shooting** (hits only
  an enemy directly above, no auto-aim) so it must position to deal damage;
  kills drop P items that raise a power meter (→ more damage). Every 45–60 s a
  **spam phase**: roaming top-screen spawners rain pellets for 10 s while all
  else pauses. **Real th07 bullet hitboxes** (pellet 2.0 / ball 3.0 px, player
  1.8 — measured via `native/probe_bullets.py`). 240 s episode cap. Player
  physics measured from the real game (`sim/physics.json`).
- `sim/train.py --algo ppo|es` — PPO (or antithetic ES). The actor's architecture
  is identical to `native/policy.py MLPPolicy`, so the result loads straight into
  the real env.
- `sim/hud.py <run>` — live survival-vs-steps curve + a death-cause split
  (emitter / spam / enemy) so it's clear what's killing the policy.
- `sim/watch_sim.py <best.pt> [--follow]` — watch the sim policy play, with the
  same overlay as `viz.py` plus items and a predicted-bullet-track macro view,
  reloading the checkpoint as it trains.
- `sim/transfer.py <best.pt> [--watch] [--until-death]` — run a sim-trained
  policy on the real game and report survival.

**Result:** sim-trained policies transfer to the real game — best real Stage 1
runs: `ppo_v12` **470 s / 1.63 M** (stages 1–3), `ppo_v27`/`ppo_v29` snapshots
~200–320 s reliably clearing Letty. The sim has been reworked many times
(v12→v29): global map added/dropped, real bullet hitboxes, spam phase,
front-only shot, per-episode domain randomization (v27), then AABB collision +
focus-aware escape (v29). `ppo_v28` was killed — obs-normalization folded 1e4×
weights into the actor and destroyed transfer. Every run overfits the sim after
~50–90 M steps (sim metric keeps rising, real transfer plateaus/dips), which is
why the current focus is real boss content, not more sim tweaks. Full status
table in `sim/README.md`.

`env.py` reads the stage-1 midboss/boss from `EM_BOSSES[0]` (PCB doesn't put
them in the enemy array). `env.reset(hard=True)` does an **engine-level Stage 1
reload** (`ST_HARD_RESET` — writes the supervisor "Give Up and Retry" word), so
episodes reset in ~0.2 s from anywhere without a process relaunch.

Needs a CUDA PyTorch venv (`.venv-cuda`, kept separate so the live-game venv
stays untouched):

```
py -3.12 -m venv .venv-cuda
.venv-cuda\Scripts\pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv-cuda\Scripts\pip install numpy triton-windows
```

## Setup (live game)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r native\requirements.txt
```

Build the DLL (needs MSYS2 mingw32, from **PowerShell** not Git Bash):

```
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

Point the harness at your `th07.exe` install (see `native/README.md`), then:

```
.venv\Scripts\python watch.py --random          # eyeball the env
.venv\Scripts\python train_evo.py --name evo1    # train
```

## Layout

```
native/       the DLL, injector, Gym env, shared-memory contract, obs builder, HUDs
  obs.py            canonical observation (shared by env + sim)
  env.py            Th07Env - Gymnasium wrapper (dll_obs, hard_reset)
  th07hook.cpp      the DLL: hooks, snapshot/reset, hard-reset, ST_ROLLOUT, obs, MLP
  sb3_bridge.py     MLPPolicy <-> SB3 PPO weight bridge
  real_rollout.py   RealRolloutVec - N DLL instances collecting PPO trajectories
  record_boss_driven.py  drive to a boss + record every bullet (x,y,vx,vy,fx) per frame
  eval_cirno.py     measure a policy's real-Cirno-fight survival
  probe_*.py        RE probes (bullet motion fields, hitboxes, the retry trigger, ...)
train_evo.py    island-model neuroevolution on the live game
train_ppo_dll.py  PPO fine-tuning on the real game via ST_ROLLOUT
watch.py        replay a checkpoint in a visible window
sim/          GPU danmaku sim + PPO/ES + sim-to-real transfer
  danmaku.py, train.py    the made-up-danmaku sim + trainer (ppo_v12..v29)
  ecl_parse.py, ecl_vm.py, ecl_bullet.py    CPU ECL decompile -> interpreter
  fight_replay.py (FightSim), train_fight.py, fight_viz.py   replay recorded real fights
tools/th07_ecl/   thtk-decompiled th07 ECL + the annotate/opcode-map helpers
feasibility/  early proof-of-concept notes
```
