# The goal & the approach

## The goal

A full **Lunatic 1-credit clear** of `th07.exe` v1.00b — six stages, no
continues — ideally **no-miss, no-bomb**. As explained in [The game](ch-game.md),
PCB permits a pure-dodge solution, so the target policy is one that survives, not
one that optimises score.

The clear must run on the **unmodified retail executable** with our tooling in
observe-only mode. `thtk` and `thprac` are used only as offline tools (script
extraction, and — planned — jumping to a boss to record it). No vpatch, no
thcrap, no runtime patches to game logic.

## Why reinforcement learning

Hand-writing a dodging controller means encoding "where is the gap" for every
pattern in the game. RL instead learns a single policy — a function from *what
the agent sees* to *which way to move* — by playing the game hundreds of
millions of times and keeping what survives longer. The cost is that it needs
those hundreds of millions of frames, which shapes the whole system below.

## The shape of the system

```
        ┌─────────────┐     obs      ┌──────────┐   action    ┌──────────┐
        │  th07.exe   │ ───────────▶ │  policy  │ ──────────▶ │  th07.exe │
        │ (+ hook DLL)│              │  (MLP)   │             │ (+ hook)  │
        └─────────────┘              └──────────┘             └──────────┘
```

- **Perception** is process-memory reads. `th07.exe` is a 2003 binary with a
  single fixed memory layout, so every bullet, enemy, and item sits at a known
  address. A [feature extractor](obs.md) turns that raw state into a 236-number
  observation vector.
- **Control** is a 32-bit DLL, [`th07hook.dll`](hook.md), injected into a
  running game. It hooks the per-frame tick and feeds the agent's chosen action
  in place of the keyboard, and it can run the game **headless at ~80×
  real-time**, one logic frame per request.
- **The policy** is a small [multilayer perceptron](ch-policy.md) — 236 inputs,
  two hidden layers, a 36-way action output — trained with
  [PPO](ppo.md).
- **Training** happens in a **GPU danmaku simulator** ([Chapter 9](sim.md)) that
  runs ~1000 games in parallel at hundreds of thousands of frames per second,
  because even 80× real-time is far too slow for PPO. The trained policy is then
  **transferred** to the real game.

## Character: ReimuA

The character is fixed to **ReimuA**. Her shot type homes on the nearest enemy
automatically, which **decouples "shoot" from "aim"** — the agent never has to
point at anything. Combined with PCB's dodge-friendly design, this keeps the
action space down to *move* and *focus* (plus an almost-free *shoot* bit) and
keeps the sim's shot model simple.

## Current scope

The end goal is all six stages, but everything currently in flight targets
**Stage 1 and its boss, Letty**. Stage 1 is effectively solved by the
[procedural-sim](sim.md) policies (~238 s, clears Letty); the
[recorded-Letty](recording.md) work went further and landed the first real
*kills* on a boss. Both approaches then hit
[the same ceiling](ceiling.md), which is what [the plan](ecl-vm.md) responds to.
