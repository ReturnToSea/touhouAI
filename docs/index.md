# th07 Agent Handbook

!!! quote ""
    Teaching a machine to clear *Perfect Cherry Blossom* on Lunatic.

A reinforcement-learning agent for Touhou 7 (PCB) on Lunatic, built from
process-memory reads, an injected control DLL, a GPU danmaku simulator, and a
lot of things that did not work.

## How to read this

The handbook is a linear tour — start at Chapter 1 and go through. **Parts 1–3**
build up the system: the game, the goal, and how the agent sees, decides, learns,
and talks to the game. **Part 4** covers training in simulation and the ceiling
it hits. **Part 5** is the current plan. **Results** is deliberately empty — the
plan hasn't produced any yet. **Extras** holds every approach that didn't work,
the lessons, and the engine-internals reference.

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
point, and the plan to get past it, is what Parts 4 and 5 are about.

### Where it stands

| Milestone | Result | Notes |
|---|---|---|
| Best real playthrough | mid-Stage 2, ~1.63M score | `ppo_v12` (retired 212-d obs) — cleared the Chen midboss, died before the Stage 2 boss |
| [Procedural-sim](sim.md) transfer | `~225 s median` | `ppo_v27` / `ppo_v29` — clears Stage 1, dies in Stage 2 |
| [Recorded-Letty](recording.md) transfer | ~100 s active-fight, lands real kills | `fight_letty_seg` v9 — first run to *kill* a real boss, not just outlast it |

### What's next

Both simulator approaches — [procedural](sim.md) and
[replayed recordings](recording.md) — plateau in the same place, and the failure
mode says it's a [hard limit of transforming a fixed dataset](ceiling.md), not a
tuning problem. The response ([the plan](ecl-vm.md)) is to **generate** the
danmaku instead: run each boss's actual PCB bytecode, getting the bullet motion
by hooking the engine and measuring it. Split into twelve verifiable parts; no
results yet.

!!! note "Constraints"
    The clear runs on vanilla `th07.exe` with the hook in observe-only mode.
    `thtk` and `thprac` are used only as offline tools — script extraction, and
    (planned) jumping to a boss to record it. No vpatch, no thcrap, no runtime
    patches to the game logic. Character is fixed to **ReimuA** — a fixed shot
    type (~20% homing, ~80% forward), so there's no aim axis to learn, though
    positioning under a boss to land the forward fire is still part of the task.
