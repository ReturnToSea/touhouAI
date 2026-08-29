# native/ — th07hook.dll + the Gym env

In-process control hook for `th07.exe` v1.00b. Turns the game into a
step-on-demand, headless, memory-observable environment for RL.

## What it does

`inject32.exe` launches `th07.exe` `CREATE_SUSPENDED`, patches the
single-instance-guard `jnz` (so multiple instances can run for parallel
training), resumes, and `LoadLibrary`s `th07hook.dll`. The DLL (MinHook) hooks:

| hook | address | purpose |
|---|---|---|
| `read_input` | `0x430B50` | menu nav vs. the env's `action` bits |
| `Present` | `0x4345C0` | skipped while driving (no DWM vsync cap) |
| `Window::do_tick` | `0x4346E0` | the command dispatcher (see states below) |
| `run_all_on_draw` | `0x42FE20` | stubbed while driving → **~3× faster**, bit-identical state |
| replay recorder | `0x442CD0` | `this` captured so its append buffer can be snapshot/restored (it overran and crashed runs at ~233k frames) |

It also self-mutes the process at the WASAPI level, forces **Lunatic**
(`[0x626280]` pinned during auto-nav), and installs a VEH crash handler that
writes the faulting address to shared memory.

### do_tick states (shared-memory `state`)

- `ST_FREE` — run normally (menus), rendered
- `ST_STEP` — advance `repeat` logic ticks holding `action`, then obs + `done`
- `ST_AUTONAV` — mash Shoot through title → Lunatic → character → Stage 1
- `ST_SNAPSHOT` — memcpy the `.data` section (+ globals + recorder head) as the
  episode reset point
- `ST_RESET` — restore that snapshot (~1.5 ms)
- `ST_EVAL` — **run a whole episode in C**: reset → (build_obs → MLP forward →
  decode action → tick)\* until death or the frame cap. One Python round-trip
  per episode. Optional random phase offset (anti-memorization) and 60 Hz
  pacing (for `watch`).

Measured: ~80× real-time single instance; ~800–1000× aggregate across ~12
island instances.

## The observation

`build_obs()` in `th07hook.cpp` and `build_obs_batch()` in `obs.py` compute the
**same** 212-value vector (verified byte-exact): 16 scalar head + 9 escape
scalars + a 13×13 player-centred danger grid + 6 enemies. `obs.py` is the
canonical version, shared with the GPU sim.

## Shared memory

`th07hook_<pid>` — struct in `th07_shm.h`, mirrored in `shm.py`. Keep them in
sync; `SHM_VERSION` bumps when the layout changes.

## Build

Needs MSYS2 mingw32 (`pacman -S mingw-w64-i686-gcc`). From **PowerShell**
(mingw gcc fails under Git Bash — PATH clash with Git's binutils):

```
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

Produces `native/build/{th07hook.dll,inject32.exe}` (gitignored). Kill any
running `th07.exe` first or the link fails (DLL locked).

## Files

```
th07hook.cpp    the DLL
th07_addrs.h    static addresses / struct offsets (from exphp-share/th-re-data)
th07_shm.h      shared-memory contract (C)
shm.py          shared-memory contract (Python) + Hook helper
inject.py       launch + inject + heal th07.cfg (must stay writable / windowed)
obs.py          canonical batched observation builder
policy.py       MLPPolicy - the flat-param MLP shared by evo + sim + the DLL
env.py          Th07Env - Gymnasium wrapper (reset via snapshot, rollout_policy via ST_EVAL)
evohud.py       tkinter HUD for train_evo.py
viz.py          live debug overlay for one real-game instance
build.ps1       build script
```

## Gotchas

- `th07.cfg` in the game dir **must stay writable** and windowed (byte `0x1F`
  == `0x01`); the game hangs on a read-only cfg. `inject._heal_cfg()` restores
  it from `th07.windowed.cfg`.
- The stray `th07.cfg` in the repo root is **not** the one the game uses.
- Don't `taskkill /IM th07.exe` while `train_evo.py` is running — it kills the
  island instances; the run recovers via relaunch but loses the generation.
