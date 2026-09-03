# th07 Agent Handbook

!!! quote ""
    Teaching a machine to clear *Perfect Cherry Blossom* on Lunatic.

A reinforcement-learning agent for Touhou 7 (PCB) on Lunatic, built from
process-memory reads, an injected control DLL, a GPU danmaku simulator, and a
lot of things that did not work.

## How to read this

The handbook is a linear tour — start at Chapter 1 and go through. **Parts 1–3**
build up the system: the game, the goal, and how the agent sees, decides, learns,
and talks to the game. **Part 4** covers training in simulation, the boss's
script format, and the ceiling the simulator hits. **Part 5** is the plan that
came out of that ceiling: train on the real game and accept the slowdown.
**Results** is deliberately thin — that plan is still running. **Extras** holds
every approach that didn't work, the lessons, and the engine-internals reference
— including the [ECL VM](de-generative-danmaku.md), the most complete attempt to
get past the ceiling and the one that failed hardest.

| | |
|---|---|
| obs dimensions | **236** |
| headless speed | **~80×** real-time |
| sim throughput | **273k** env-frames/s |
| bullet pool | **1025** slots |
| stages to clear | **6** |

---

## In one paragraph

The goal is a full Lunatic 1-credit clear of `th07.exe` v1.00b, ideally no-miss
no-bomb ([Chapter 2](ch-goal.md)). Perception is process-memory reads; control is
an injected 32-bit DLL that also runs the game headless at ~80× real-time;
training is a GPU danmaku simulator running ~1000 games in parallel, with the
policy then transferred to the retail game. It works up to a point — and that
point, what's been tried to get past it, and where the project goes next are what
Parts 4 and 5 are about.

### Where it stands

| Milestone | Result | Notes |
|---|---|---|
| Best real playthrough | mid-Stage 2, ~1.63M score | `ppo_v12` (retired 212-d obs) — cleared the Chen midboss, died before the Stage 2 boss |
| [Procedural-sim](sim.md) transfer | `~225 s median` | `ppo_v27` / `ppo_v29` — clears Stage 1, dies in Stage 2 |
| [Recorded-Letty](de-letty-replay.md) transfer | 60 s median / 7% kill-rate (631 fights); best checkpoint ~103 s / 33% | `fight_letty_seg` v9 — first run to *kill* a real boss, not just outlast it; then [plateaued](ceiling.md) |

### What's next

Both simulator approaches — [procedural](sim.md) and
[replayed recordings](recording.md) — plateau in the same place, and the failure
mode says it's a [hard limit of transforming a fixed dataset](ceiling.md), not a
tuning problem. The response was to **generate** the danmaku instead: run Letty's
actual PCB bytecode in a reimplemented VM. That got built almost to completion —
byte-exact parser, control-flow VM, PRNG, sub-enemy recursion, boss and orb
movement, an engine-faithful bullet-motion model — and reproduces Letty's danmaku
to a ~2 px per-bullet noise floor. **It still didn't transfer** — eight training
runs produced a flat ~50 s / 0 %-kill real-game line, *worse* than either
simulator it was meant to replace. A Python reimplementation of a 2003 x87-FPU
engine can be made close but not exact, and the residual error compounds across a
500-bullet screen. The full postmortem is
[Generative danmaku — the ECL VM](de-generative-danmaku.md).

The current direction ([Part 5](plan.md)) is to stop simulating Letty and **train
the fight on the real game** — `ST_ROLLOUT` collecting whole PPO trajectories
inside a dozen [hooked games](hook.md) in parallel, warm-started from the
procedural-sim policy. ~15× slower per step than the GPU sim; no fidelity gap
because there's no sim.

!!! note "Constraints"
    The clear runs on vanilla `th07.exe` with the hook in observe-only mode.
    `thtk` and `thprac` are used only as offline tools — script extraction, and
    (planned) jumping to a boss to record it. No vpatch, no thcrap, no runtime
    patches to the game logic. Character is fixed to **ReimuA** — a fixed shot
    type (~20% homing, ~80% forward), so there's no aim axis to learn, though
    positioning under a boss to land the forward fire is still part of the task.
