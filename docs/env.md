# The environment

`Th07Env` is a **Gymnasium** wrapper around one hooked `th07.exe`. It turns the
[control DLL](hook.md) into a standard `reset()` / `step()` RL environment:
construct the game, auto-navigate into Stage 1, freeze a snapshot, then drive
episodes from that snapshot.

## Lifecycle

```
__init__   launch th07.exe suspended, inject the DLL, resume
AUTONAV    DLL mashes Shoot through title → difficulty → character → Stage 1
SNAPSHOT   freeze the current game state (RNG included) as the episode start
reset()    restore the snapshot
step(a)    feed action a, run one logic frame, return (obs, reward, term, trunc)
```

## Faithful reset

`reset()` `memcpy`s a frozen copy of the entire game state — **RNG and all** —
back over the live one. It is bit-perfect and instant, which is what training
wants. The cost is that every episode from a given snapshot replays identically;
variety comes from a different mechanism.

`reset(hard=True)` instead writes `10` to `Supervisor + 0x158` — the value the
pause menu's *"Give Up and Retry"* sets — and the engine reloads the stage from
scratch with **fresh RNG** in ~0.2 s, no relaunch. The
[survey harness](de-checkpoints.md) uses this to get run-to-run variety.

## Running many at once

For a vector env, `inject32` flips the single-instance guard so multiple
`th07.exe` processes coexist. D3D initialisation races when several launch
together, so construction takes a **cross-process build lock** — instances
serialise through device creation, then run in parallel.

## Audio

Each session mutes its own audio endpoint. This took a few tries: the DLL holds
the 60 fps limiter live for the first few seconds of the title screen, because
NOP-ing it before the audio session exists plays the title BGM back as an 80×
screech. Once auto-nav starts and the endpoint is muted, the limiter and
rendering are stubbed and the instance runs [~80× real-time](hook.md).

## Two observation paths

- **`dll_obs`** — the DLL builds the 236-float [observation](obs.md) in C.
  Used for headless training and real-game rollouts; fast.
- **Python obs** — the same vector rebuilt from `pymem` reads. Slower, used for
  debugging and the [parity check](obs.md#parity).
