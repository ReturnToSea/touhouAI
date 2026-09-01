# touhouAI

An AI that plays **Touhou 7 – Perfect Cherry Blossom** (`th07`, v1.00b). Started
as "survive + score on Stage 1 Lunatic"; **Stage 1 is now solved** (a sim-trained
policy clears the stage, reaches + usually beats Letty, ~238 s greedy) and the
goal is a **full Lunatic 1-credit clear**.

## Current direction (2026-09)

Both simulator approaches — the made-up-danmaku sim (track 2 below) and a
recorded-replay sim — plateau in the same place: they train on a **fixed
artefact** (procedural patterns with no real boss structure, or ~20 recordings
and their symmetries) that the policy learns to exploit in ways the sim eval
can't see. `fight_letty_seg` v9 ran a full billion steps on replayed Letty and
got a **7 % real kill-rate** (bimodal, no trend) — the ceiling.
Full write-up: [`docs/de-letty-replay.md`](docs/de-letty-replay.md),
[`docs/ceiling.md`](docs/ceiling.md).

**The plan now:** *generate* the danmaku — run each boss's actual PCB bytecode
so every episode is a novel, correct pattern. `sim/ecl/` is a from-scratch ECL
VM:

- **binary parser** — decodes every stage's `.ecl`; 19,919 / 19,919
  instructions match thtk (`python -m sim.ecl.verify`)
- **control-flow VM** — runs Letty's real script frame by frame: the phase
  machine (NS1 → Lingering Cold → NS2 → Table-Turning → defeat) lands on the
  recorded screen-clears, arithmetic + a PRNG, recursion into her sub-enemies,
  and movement — the boss's own track is **pixel-exact against a recording for
  127 frames** (`python -m sim.ecl.vm_verify`)
- **bullet spawn events** come out the other end (frame, type, position, angle,
  speed) within a few percent of the recorded birth counts

Parts 1–3, 5, 6 done and verified; movement (Part 8) is built but not yet its
own verified pass; Part 4 (PRNG) and Part 7 (HP thresholds) are partial. Bullet
*motion* is Stage B — hooking `th07.exe` and measuring, not interpreting
`bullet_effects` statically. Full plan + status:
[`docs/ecl-vm.md`](docs/ecl-vm.md).

Everything downstream — collision, the 236-value obs, PPO, the real-game
transfer daemon — is unchanged; the VM only replaces *where the bullet positions
come from*.

<details><summary>Earlier tracks (kept, moved on from)</summary>

- **Real-game PPO fine-tuning** — built (`native/ST_ROLLOUT`, `train_ppo_dll.py`).
  Fine-tuning Stage 1 was a **wash** (238 s → 223 s greedy) — the sim policy is
  already at the ceiling for content it can handle.
- **Recording real boss fights** (`native/record_boss_driven.py` →
  `sim/fight_replay.py` `FightSim`): reading the `zBullet` struct each frame
  captures the real pattern with no interpretation. Landed the first real *kills*
  on a boss, then hit the ceiling above. The recordings are now the VM's
  validation ground truth.

</details>

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

**Result:** sim-trained policies transfer to the real game — `ppo_v12` cleared
Stage 1 + the Chen midboss and died just before the Stage 2 boss (~1.63 M
score; the often-quoted "470 s" is its *sim* survival, not real), and
`ppo_v27`/`ppo_v29`
snapshots transfer to ~225 s real — clearing Stage 1, into Stage 2. The sim has
been reworked many times
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
  fight_replay.py (FightSim), train_fight.py   replay recorded real fights (fight_letty_seg)
  ecl/          the ECL VM: parser.py, vm.py, rng.py, opcodes.py, *_verify.py
  ecl_vm.py, ecl_parse.py, ecl_bullet.py    the shelved first ECL attempt (kept for reference)
tools/th07_ecl/   thtk-decompiled th07 ECL, the opcode map, and its README
docs/             the handbook (mkdocs) — plan, results, and everything tried
feasibility/  early proof-of-concept notes
```
