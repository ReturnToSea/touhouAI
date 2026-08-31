# th07 Agent Handbook

!!! quote ""
    Teaching a machine to dodge *Perfect Cherry Blossom*.

A reinforcement-learning agent for Touhou 7 (PCB) on Lunatic, built from
process-memory reads, an injected control DLL, a GPU danmaku simulator, and a
lot of things that did not work. This is the working reference and the record
of what we tried.

| | |
|---|---|
| obs dimensions | **236** |
| headless speed | **~80×** real-time |
| sim throughput | **273k** env-frames/s |
| bullet pool | **1025** slots |
| stages to clear | **6** |

---

## Overview & thesis

The goal is a full Lunatic 1-credit clear of `th07.exe` v1.00b, ideally no-miss
no-bomb. PCB is the most "perfectable" of the hard Windows Touhou games and its
cherry/border mechanic is a scoring tool, not a survival crutch — so a pure-dodge
policy is a legitimate path to the clear.

### The shape of the system

Perception is process-memory reads — the game has a single fixed layout, so every
struct sits at a known address. Control is a 32-bit DLL injected into a running
`th07.exe`; it hooks the per-frame tick and feeds the agent's action in place of
the keyboard. Training happens in a GPU simulator that runs thousands of danmaku
episodes in parallel, then the policy is transferred to the real game.

### Why a simulator at all

Headless, the real game runs about **80×** real-time per instance. The GPU sim
runs about **273,000 env-frames per second** — roughly a thousand parallel games.
PPO needs hundreds of millions of frames to learn a hard bullet pattern; only the
sim delivers that in minutes instead of days. Real-game RL stays a targeted
touch-up tool, not the main training vehicle.

### Where it stands

| Milestone | Result | Notes |
|---|---|---|
| Best real playthrough | mid-Stage 2, ~1.63M score | `ppo_v12` (retired 212-d obs) — cleared the Chen midboss, died before the Stage 2 boss |
| Current-obs transfer | `~225 s median` | `ppo_v27` / `ppo_v29` — clears Stage 1, dies in Stage 2 |
| Recorded-boss transfer | `150–190 s` | `fight_letty`, trained only on replayed Letty, dodges real Letty |

!!! note "Constraints"
    The clear runs on vanilla `th07.exe` with the hook in observe-only mode.
    `thtk` and `thprac` are used only as offline tools — script extraction, and
    (planned) jumping to a boss to record it. No vpatch, no thcrap, no runtime
    patches to the game logic. Character is fixed to **ReimuA** (homing shot —
    decouples "shoot" from "aim").
