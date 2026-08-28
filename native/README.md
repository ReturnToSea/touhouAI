# native/ — th07hook.dll

In-process control hook for `th07.exe` v1.00b. Turns the game into a
step-on-demand, headless, memory-observable environment for RL.

## What it does

`inject32.exe` launches `th07.exe`, resumes it, and injects `th07hook.dll`.
The DLL (MinHook) installs three hooks and NOPs the 60 Hz frame limiter:

| hook | FREE mode | STEP mode |
|---|---|---|
| `read_input` `0x430B50` | real keyboard (menu nav) | the env's `action` bits |
| `Present` `0x4345C0` | presents (rendered) | skipped (no DWM vsync cap) |
| `Window::do_tick` `0x4346E0` | runs normally + captures obs | 1 logic tick per env step, then obs + `done` |

Rendering (BeginScene→draw→EndScene) still runs each step; only the buffer flip
is skipped. The game **only advances when the env steps it**, so the
Continue-screen free-run corruption seen without the DLL can't happen.

Measured: **~53× real-time** single instance (1 logic tick per step, no crash).

## Shared memory

`th07hook_<pid>` — layout in `th07_shm.h` / `shm.py`. Env writes
`state`/`action`/`repeat`; DLL writes `done`/`frame` + the observation
(player, bullets, enemies, boss, score, lives, ...).

## Build

Needs MSYS2 mingw32 (`pacman -S mingw-w64-i686-gcc`). From PowerShell:

```
powershell -ExecutionPolicy Bypass -File native\build.ps1
```

Produces `native/build/{th07hook.dll,inject32.exe}` (gitignored).
`build.sh` is the bash equivalent but **mingw gcc fails under Git Bash**
(PATH/DLL clash with Git's bundled binutils) — use PowerShell.

## Use

```python
from native.inject import inject
import native.shm as shm
pid = inject()
h = shm.Hook(pid)
# ... navigate into a stage (FREE mode) ...
h.step(action=shm... , repeat=1)   # one headless logic tick
```

## TODO

- episode reset (snapshot/restore of bullet+enemy+player+RNG+ECL state)
- bullet velocity in the obs (currently 0; derive from frame delta or read the
  velocity field)
- try skipping the render block too (crashed pre-DLL; may be safe now that the
  game is fully step-gated)
- auto menu navigation so a fresh instance self-starts a run
