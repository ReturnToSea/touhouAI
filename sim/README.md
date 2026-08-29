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
| `train.py` | `--algo ppo` (default) or `--algo es`. Actor arch == `MLPPolicy`, so `best.pt` loads into the real env. Writes `runs_sim/<name>/{best,last}.pt`, `history.npy`, `meta.json`. Logs greedy survival (distribution: median / p90 / >60-120-180 s, capped at the 180 s episode length), sampled survival, entropy, and a death-cause breakdown (`wall % / enemy %`, rest = emitter fire). Training sim runs compiled; `--eager-sim` to disable. |
| `hud.py` | live training dashboard (survival-vs-steps curve) — `python sim/hud.py <run>` |
| `watch_sim.py` | watch the sim policy play with the debug overlay; **takes a path to `best.pt`**, `--follow` reloads it every few seconds. `g` = local danger grid, `G` = macro view (predicted bullet tracks) |
| `transfer.py` | run a sim-trained `best.pt` on the real Lunatic stage 1 (`.venv`, not `.venv-cuda`) |

## The observation (`native/obs.py`, `OBS_DIM = 236`)

| part | size | what |
|---|---|---|
| head | 16 | player pos/vel, focus, power, nearest-bullet distance, boss/midboss HP |
| escape scalars | 9 | for {stay, N, NE, E, SE, S, SW, W, NW}: frames-until-hit if the player holds that move for 20 frames, `/20` |
| local danger grid | 13×13 | player-centred (±78 px, 12 px cells); each cell = how imminent a bullet strike there is (bullets marched 24 frames along their paths); out-of-bounds cells = 0.5 |
| enemies | 6×3 | nearest on-screen enemies (rel pos, hp fraction) |
| items | 8×3 | nearest items (rel pos, type) |

An absolute-coordinate "global danger map" was tried (obs 236→404) and dropped —
ablation showed flat first-layer weights, the policy ignored it.

## The sim stage (`ROSTER` in `danmaku.py`)

- 4 corners × (**CONE** + **SPRAY**) placed *in* the corners — aimed fan + wide
  random burst, both toward centre. No safe pocket behind them.
- one fast **LINE** emitter (bottom-right) — 1 bullet/shot, aim sweeps back and forth
- one **BRING** (dense ring) emitter that **bounces** the interior
- one **BRING** emitter that **orbits** the perimeter (punishes wall/corner camping)

`BULLET_SCALE = 0.85` — all bullets 15 % smaller than measured (real stage-1
hitboxes are tiny; −35 % was tried in v16-18 and made the policy under-cautious
in transfer). The moving emitters fire slow bullets with a 5 s lifetime (or the
screen saturates). Only per-episode bullet-speed jitter varies; the layout is fixed.

### Wall attack

Every 7 s a solid **curtain** of 56 bullets sweeps across the field from a random
edge (L→R / R→L / T→B / B→T). It covers **half** the perpendicular span (192 px
vertical / 224 px horizontal), leaving a half-width gap on one side, spawns
~48 px off-screen (visible approaching) and crosses at ~1.3 px/frame. A pure
*macro* threat — you have to already be in the open half.

**Known weakness:** the curtain (192–224 px) is wider than the perception window
(±78 px = 156 px), so the policy often can't see the wall *and* its gap at once —
it's been ~40 % of all deaths and unlearned across v15 and v21. v22 plan: shrink
the curtain to ~130–140 px and randomise its position along the full span so the
gap is always in view.

### Top-right CONE redirect

Bullets from the top-right corner's CONE get one 50 % chance, after 1 s of
flight, to snap to a new random heading (full 360°) at the same speed — then fly
straight (5 s life cap). Breaks the "all bullets travel in straight lines"
assumption, closer to the real game, and makes the linear escape-scalar
prediction usefully imperfect.

### Enemies + power

A wave of 9–15 enemies flies in from off-screen every 12 s, flies to a random
point in the upper-mid playfield, hovers ~6 s, then leaves. 1 HP each; touching
one kills the player like a bullet. Holding **SHOOT** auto-damages the nearest
1–3 on-screen enemies (count grows with power) for `EN_DPS × power_mult` and pays
`EN_DMG_REW` (0.10) per HP. On death an enemy drops 4 **P items** that pop up and
fall under gravity; collecting one (within 14 px) pays `IT_REW` (0.30) and raises
**power** (`PWR_PER_ITEM` 2.5 / 128), which scales shot damage 1× → 3× and widens
the target count.

**Known gap:** sim enemies have **no projectiles** — they only kill by body
contact. So the policy learns enemies are safe at range and crowds them for the
shoot reward; real stage-1 fairies fire aimed shots, so crowding them = a
point-blank hit. v22 plan: give hovering enemies a light aimed burst.

## Typical run

```
.venv-cuda\Scripts\python sim\train.py --algo ppo --name ppo1 --steps 400e6
.venv-cuda\Scripts\python sim\hud.py ppo1
.venv-cuda\Scripts\python sim\watch_sim.py runs_sim\ppo1\best.pt --follow
# when it's good:
.venv\Scripts\python sim\transfer.py runs_sim\ppo1\best.pt --watch --episodes 1 --until-death
```

## Status

| run | sim stage | sim greedy | real Lunatic stage 1 |
|---|---|---|---|
| `ppo_v12` | early layout, obs 212 | ~23 s median | **470 s / 1.63M** (stages 1–3, 95 % move) |
| `ppo_v15` | + global map, wall, P items (obs 404) | ~46 s | not transfer-tested |
| `ppo_v21` | reverted to the 8-corner stage, obs 236, `BULLET_SCALE 0.85`, 180 s cap | median ~42 s / p90 ~139 s @ 429M | **105 s / 15k**, 99 % move — a competent dodger that drifts to a corner and stalls; well short of v12 |

`ppo_v20` (a realism redesign — turrets outside, spinning centre wheels, no
auto-aim) was abandoned: the sparser stage left a survivable bottom-centre
pocket and the policy just camped it.

**Next (v22, from scratch):** enemy aimed shots + shorter/position-randomised
wall + lower `ent_coef`. See the two "Known gap/weakness" notes above.

## Notes

- `torch.compile` needs `triton-windows`. Applied to the sim's `_advance` + the
  shared `build_obs_batch`; **not** to the policy net (that stalled a run).
- The transfer eval uses the Python step loop (not the in-DLL fast path) so the
  observation matches training exactly regardless of the DLL build.
- `env.reset()` restores a snapshot of *start-of-stage-1*, so it can only rewind
  short stage-1 episodes. A policy that survives into later stages crashes the
  game on the next reset — use `--episodes 1` for `--until-death` watches (each
  fresh run still starts clean from stage 1).
- Constants in `native/obs.py` (`DIR_SPEED`, playfield bounds, `K_NEAREST`) must
  match `native/th07hook.cpp` for the fast in-DLL eval / real-game fine-tuning.
