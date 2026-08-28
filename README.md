# touhouAI

An AI that plays **Touhou 7 – Perfect Cherry Blossom** (`th07`, v1.00b), aiming
for a highly optimized agent: survive, then maximize score and clear speed.

## Approach

- **Perception:** read game state directly from process memory (`pymem`) — exact
  player/bullet/enemy/boss/score data, no computer vision.
- **Learning:** reinforcement learning. Feasibility work showed the game can be
  run ~24x real-time (and likely much faster) with no DLL — just a launcher that
  NOPs the frame limiter in the suspended process. Parallel instances need only
  a mutex NOP + memory-based input.
- **Project-based:** nothing depends on absolute paths or a specific machine. The
  only per-machine setting is where the game is installed (a gitignored config).

The game install itself is **not** in this repo (copyrighted, and `thbgm.dat`
alone exceeds GitHub's file limit). It's ignored via `.gitignore`.

## Status

Feasibility phase — proving the game can be driven as RL requires. See
[`feasibility/README.md`](feasibility/README.md).

- [x] Confirm version: `th07 v1.00b (original)`
- [x] Pin memory offsets (`feasibility/th07_data.py`, from `exphp-share/th-re-data`)
- [x] 32-bit toolchain for the future hook (`C:\msys64\mingw32`, i686 gcc)
- [x] Gate A: live memory read verified against the screen (player, bullets,
      enemies, boss HP, score, lives, bombs, power, graze, cherry, inputs, stage)
- [x] Gate A: episode-state signal — `state` cycles
      `alive → dead → respawning → invuln`; game over = `lives == 0` +
      `state == respawning` (Continue screen freezes the game)
- [x] Gate A: synthetic input verified — `SendInput` with scancodes reaches the
      game's input word when focused; bitfield mapping confirmed
      (left `0x40`, right `0x80`, down `0x20`, shoot `0x01`, slow `0x04`)
- [ ] Gate A: programmatic restart from the Continue screen (menu automation)
- [x] Gate B: faster-than-realtime — **no DLL needed.** `launch_uncapped.py`
      NOPs 3 branches (frame limiter + vsync path) in the suspended process;
      render frameskip (`0x575A8B`) then gives `(skip+1) x 144` logic ticks/sec
      — measured **24x real-time** at frameskip 9, CPU ceiling likely far higher
- [ ] Harness: parallel instances (mutex NOP + focus-free input), episode reset,
      true headless

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r feasibility\requirements.txt
```

Then follow `feasibility/README.md`.
