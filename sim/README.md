# sim/ — GPU danmaku sim → sim-to-real transfer

Train a dodging policy in a fast made-up-danmaku environment on the GPU, then
transfer it to the real game. Uses the `.venv-cuda` venv (CUDA PyTorch).

Motivation: after ~10 from-scratch runs on the live game (PPO and evolution),
policies plateaued at "freeze in a safe-ish spot". The live game gives one
scalar of feedback per ~20 s episode at ~1000× real-time — too weak/slow for
learning a reactive reflex. A vectorised sim runs thousands of episodes in
lockstep at ~100k+ env-steps/s and lets PPO learn per-timestep.

## Files

| file | what |
|---|---|
| `physics_probe.py` | measure the real game's move speed, bounds, bullet speeds, collision distance → `physics.json` |
| `danmaku.py` | the vectorised env. Batched tensors, `torch.compile`d, auto-reset. Fixed procedural stage (see below). Observations via the shared `native/obs.py` builder, so a policy sees bit-identical inputs to `Th07Env`. |
| `train.py` | `--algo ppo` (default) or `--algo es`. Actor arch == `MLPPolicy`, so `best.pt` loads into the real env. Writes `runs_sim/<name>/{best,last}.pt`, `history.npy`, `meta.json`. Logs greedy survival (uncapped to 120 s), sampled survival, entropy, and a **death-cause breakdown** (`wall % / enemy %`, rest = emitter fire). Training sim runs eager; `--compile-sim` to try `torch.compile` on it. |
| `hud.py` | live training dashboard (survival-vs-steps curve) — `python sim/hud.py <run>` |
| `watch_sim.py` | watch the sim policy play with the debug overlay; `--follow` reloads `best.pt` every few seconds. `g` = local danger grid, `G` = macro view (predicted bullet tracks) |
| `transfer.py` | run a sim-trained `best.pt` on the real Lunatic stage 1 |

## The sim stage (`ROSTER` in `danmaku.py`)

- 4 corners × (CONE + SPRAY) aimed at centre
- one fast **LINE** emitter (bottom-right) — 1 bullet/shot, aim sweeps back and forth
- one **BRING** (dense ring) emitter that **bounces** the interior
- one **BRING** emitter that **orbits** the perimeter (punishes wall/corner camping)

The moving emitters fire slow bullets with a fixed 5 s lifetime (or the screen
saturates). Only per-episode bullet-speed jitter varies; the layout is fixed.

### Wall attack

Every ~10 s a solid **curtain** of 44 bullets sweeps across the field from a
random edge (L→R / R→L / T→B / B→T). It covers **half** the perpendicular span
(192 px for a vertical sweep, 224 for a horizontal one), leaving a half-width
gap on one side, spawns ~48 px off-screen (visible approaching) and crosses at
~0.95 px/frame. It's a pure *macro* threat — you can't thread it or react at the
last second, you have to already be in the open half — which is what the global
danger map is for.

### Top-right CONE redirect

Bullets from the top-right corner's CONE emitter get one 50 % chance, after
1 s of flight, to snap to a new random heading (full 360°) at the same speed —
then fly straight (5 s life cap). Breaks the "all bullets travel in straight
lines" assumption the sim would otherwise train in, which is closer to the real
game and makes the escape-scalar prediction (linear) usefully imperfect.

### Enemies (learn to shoot)

A wave of 4–6 enemies flies in from off-screen every 12 s, flies to a random
point in the upper-mid playfield, hovers ~6 s, then leaves. 2 HP each; touching
one kills the player like a bullet. Holding **SHOOT** auto-damages the nearest
active on-screen enemy at 1 dmg/s (no projectiles, no aiming) and pays a small
`EN_DMG_REW` (0.10) per HP dealt. On death an enemy drops 2 **P items** that pop
up and fall under gravity; collecting one (within 14 px) pays `IT_REW` (0.15).
The P items have no weapon effect yet — they exist so the policy learns to hold
shoot and to chase drops. None of this damage/item model transfers literally —
the real game does its own shot damage and item logic. Enemies and items fill
the 6×3 and 8×3 observation slots that `Th07Env` also fills (enemies from game
memory, items straight from the live `ItemManager` via pymem).

## Typical run

```
.venv-cuda\Scripts\python sim\train.py --algo ppo --name ppo1 --steps 400e6
.venv-cuda\Scripts\python sim\hud.py ppo1
.venv-cuda\Scripts\python sim\watch_sim.py runs_sim\ppo1\best.pt --follow
# when it's good:
.venv\Scripts\python sim\transfer.py runs_sim\ppo1\best.pt --watch --episodes 3
.venv\Scripts\python sim\transfer.py runs_sim\ppo1\best.pt --watch --until-death --episodes 1
```

## Notes

- `torch.compile` needs `triton-windows`. It's applied to the sim's `_advance`
  + the shared `build_obs_batch` for the **eval** sim; the **training** sim runs
  eager by default (compiling it churns dynamo recompiles — `--compile-sim` to
  retry). It is **not** applied to the policy net — that stalled a run.
- The transfer eval uses the Python step loop (not the in-DLL fast path) so the
  observation matches training exactly regardless of the DLL build.
- `env.reset()` restores a snapshot of *start-of-stage-1*, so it can only rewind
  short stage-1 episodes. A policy that survives deep into later stages will
  crash the game on the next reset - use `--episodes 1` for `--until-death`
  watches (each fresh run still starts clean from stage 1).
- Constants in `native/obs.py` (`DIR_SPEED`, playfield bounds, `K_NEAREST`) must
  match `native/th07hook.cpp` for the fast in-DLL eval / real-game fine-tuning.
