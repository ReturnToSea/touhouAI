# The control hook

`th07hook.dll` is a 32-bit DLL injected into a running `th07.exe`. It turns the
game into a step-on-demand environment: the agent asks for one frame, the DLL
runs exactly one frame, and hands back the observation.

## Injection

`inject32.exe` launches the game suspended, flips the single-instance guard
`jnz` at `0x435BFF` to an unconditional jump (so many instances can run under a
vector env), resumes, and calls `LoadLibraryA` via `CreateRemoteThread` before
the game reaches its first tick.

## The hooks

| Address | Function | Purpose |
|---|---|---|
| `0x004346E0` | `Window::do_tick` | the state machine lives here — STEP advances exactly one logic tick per call |
| `0x00430B50` | `read_input` | returns the agent's action word instead of the keyboard |
| `0x004345C0` | `present` | skipped while driving so logic isn't throttled to the monitor's refresh |
| `0x0042FE20` | `run_all_on_draw` | stubbed while driving — state stays bit-identical, ~3× faster |

## The state machine

```
FREE        # menus / demo run normally; the human can play
STEP        # run `repeat` logic ticks, capture obs, park at IDLE
IDLE        # spin — the game does not advance
RESET       # restore the frozen snapshot (RNG included)
SNAPSHOT    # freeze the current state as the episode start
AUTONAV     # mash Shoot: title → difficulty → character → Stage 1
HARD_RESET  # write Supervisor+0x158 = 10 ("Give Up and Retry")
ROLLOUT     # run a whole T-step PPO trajectory in C, ~68×/env
```

## Two kinds of reset

**Snapshot reset** memcpy's a frozen copy of the game state — RNG and all — back
over the live one. It is bit-perfect and instant, which is what you want for
training, but it means every episode replays identically.

**Hard reset** writes `10` to `Supervisor + 0x158`. That is the value the pause
menu's "Give Up and Retry" sets; the engine reloads the current stage from
scratch, with fresh RNG, in about 0.2 s and without a relaunch. This is how the
survey harness gets variety.

## Speed

The 60 fps frame limiter (two skip-branches at `0x004348CC` and `0x00434997`) is
NOP'd once auto-nav starts — held live for the first few seconds so the title BGM
doesn't play back as an 80× screech before the audio session can be muted. With
the limiter gone and rendering stubbed, one instance runs about **80×**
real-time. `ST_ROLLOUT`, which runs the entire PPO rollout loop in C, reaches
~68× *per environment*.

!!! success "Parity"
    The DLL builds the full 236-d observation in C. `test_obs_parity.py` checks
    it against the Python builder byte for byte — current difference: `0.00000`.
