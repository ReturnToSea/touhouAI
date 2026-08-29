# touhouAI

An AI that plays **Touhou 7 – Perfect Cherry Blossom** (`th07`, v1.00b), aiming
for an agent that survives and maximizes score on **Stage 1 Lunatic**.

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

Both the DLL (`native/th07hook.cpp`) and the Python side compute the *same*
212-value observation via the shared builder `native/obs.py`:

| part | size | what |
|---|---|---|
| head | 16 | player pos/vel, focus, lives/bombs/power/graze, nearest-bullet distance, boss/midboss HP |
| **escape scalars** | 9 | for {stay, N, NE, E, SE, S, SW, W, NW}: frames-until-hit if the player holds that move for 20 frames, `/20` |
| **danger grid** | 13×13 | player-centred; each cell = how imminent a bullet strike there is (bullets marched along their paths); out-of-bounds cells = 0.5 |
| enemies | 6×3 | nearest on-screen enemies (rel pos, hp fraction) |

The policy is a tiny MLP (`212 → h → h → 36`, tanh, argmax). 36 actions =
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

- `sim/danmaku.py` — thousands of parallel episodes as batched tensors, ~100–150k
  env-steps/s on an RTX 5070 Ti (`torch.compile`d). A fixed stage of procedural
  emitters (cones + sprays in every corner, a fast sweeping line, a bouncing
  ring emitter, and one that orbits the perimeter). Player physics are measured
  from the real game (`sim/physics.json`, via `sim/physics_probe.py`).
- `sim/train.py --algo ppo|es` — PPO (or antithetic ES). The actor's architecture
  is identical to `native/policy.py MLPPolicy`, so the result loads straight into
  the real env.
- `sim/hud.py <run>` — live survival-vs-steps curve.
- `sim/watch_sim.py <best.pt> [--follow]` — watch the sim policy play, with the
  same overlay as `viz.py`, reloading the checkpoint as it trains.
- `sim/transfer.py <best.pt> [--watch] [--until-death]` — run a sim-trained
  policy on the real game and report survival.

**Result:** a sim-trained policy (`ppo_v12`, mid-training) transferred to the
real game and survived **470 s / 1.63M score on Lunatic — clearing stages 1–3
and reaching stage 4** — dodging on 95 % of decisions. An earlier checkpoint
did 91 s. This is the from-scratch reactive dodging that neither live-PPO nor
evolution produced across ~10 runs. (`env.reset()` only rewinds stage-1
snapshots, so a long-surviving policy crashes the game on the *next* reset —
use `--episodes 1` for `--until-death`.)

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
  obs.py        canonical observation (shared by env + sim)
  env.py        Th07Env - Gymnasium wrapper
  th07hook.cpp  the DLL: hooks, snapshot/reset, in-DLL episode eval, obs, MLP
  viz.py        live debug overlay for a real-game instance
train_evo.py  island-model neuroevolution on the live game
watch.py      replay a checkpoint in a visible window
sim/          GPU danmaku sim + PPO/ES training + sim-to-real transfer
feasibility/  early proof-of-concept notes
```
