# Feasibility probes

Before writing any RL code, we need to confirm the game can be driven the way an
RL-only approach requires. These scripts test that. Nothing here patches or
modifies the game — it is all read-only except `probe_input.py` (which only sends
keystrokes) and `probe_scale.py --spawn-second` (which you opt into).

## Target configuration (do not change once it works)

| | |
|---|---|
| Executable | `th07.exe`, **v1.00b (original)** — confirmed by thcrap hash |
| Launcher | run `th07.exe` (or `Touhou07.exe` = vpatch) **windowed**. Fullscreen disables vpatch. |
| vpatch / thcrap | fine for now; for the eventual training build we will pin a clean `th07.exe` with no injectors |

Offsets live in `th07_data.py`, taken from
[`exphp-share/th-re-data`](https://github.com/exphp-share/th-re-data)
(`data/th07.v1.00b`) — the same RE data thprac uses.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r feasibility\requirements.txt
```

## Gate A — can we drive the game at 1x?

1. **Read state.** Start the game, begin a run (any character/difficulty), then:
   ```
   .venv\Scripts\python feasibility\probe_memory.py
   ```
   Move around — `player=(x,y)` must track you. Fire at an enemy — `boss` hp %
   must drop. Skim a bullet — `graze` must tick up. If those line up with the
   screen, **memory reading works** and Gate A step 1 passes.

2. **Inject input.** Keep `probe_memory.py` running in one terminal. In another:
   ```
   .venv\Scripts\python feasibility\probe_input.py
   ```
   Click the game window during the countdown. The character should slide
   left then right, shoot, then focus-move down. `keys=` in the memory probe
   should show the buttons. Passes if the character actually responds.

3. **Episode state.** Watch `mode=` and `state=` in `probe_memory.py` across a
   death and a menu transition. We need a reliable "run active / player dead /
   in menu" signal. Note what the values do.

## Gate B — can we run it faster and in parallel? (the RL-only enabler)

```
.venv\Scripts\python feasibility\probe_scale.py
```

Reports the timing fields (`framerate_multiplier`, `replay_fps`, observed
ticks/sec) and the update/draw chain heads our hook would use to skip rendering
and double-tick logic. It also prints where thprac disables the single-instance
mutex (`th07.exe+0x35bff`).

To check the mutex directly, with the game already running:
```
.venv\Scripts\python feasibility\probe_scale.py --spawn-second "C:\...\th07.exe"
```
If the second copy dies immediately, we need the hook to run parallel instances.

`probe_speed.py` additionally tries writing `framerate_multiplier` and
`replay_fps` live and measures whether the tick rate changes.

## Findings (2026-08-28)

### Gate A — PASSED

- **Memory read** works fully. Validated against the screen over multiple runs:
  player pos, `state`, lives, bombs, power (0–128), score, graze, cherry,
  stage, difficulty, `gamemode`, the bullet array (our filtered count matches
  the game's own `bullet_count`), and true boss HP from
  `ENEMY_MANAGER->bosses[0]->life` (`17000 -> 4706` as damage was dealt).
- **Input injection** works: `SendInput` with scancodes reaches the game's
  `INPUT_CUR` word when the window is focused. Bitfield confirmed:
  left `0x40`, right `0x80`, up `0x10`, down `0x20`, shoot `0x01`, slow `0x04`.
- **Episode signal**: `state` cycles `alive -> dead (2-3f) -> respawning (2f,
  snapped to (192,384)) -> invuln (~3s) -> alive`. Game over = a death at
  `lives == 0`, which lands on the Continue screen and **freezes all game
  logic** (`state == respawning`, `lives == 0`) until the player chooses.
- **Focus pause**: the game only ticks while its window is foreground (audio
  keeps playing). Flag at window struct `0x575c20 + 0xC`.

### Gate B — SOLVED, no DLL needed

Static analysis (`scratchpad/disasm.py`, `xref.py` against `th07.exe`) found:

- **Frame limiter** in `Window::do_tick` (`0x4346E0`): when the game is "early"
  it skips the frame unless a debug flag (`byte [0x575C3C]`) is set. That flag
  is a leftover *uncapped/dev mode* toggle — `func @ 0x435BD0` sets it based on
  a `.lnk` / `STARTUPINFO.lpTitle` vs module-name comparison; normal launches
  leave it 0. (vpatch sets it, then re-caps with its own `GameFPS`.)
- **Vsync** in `Supervisor::init_d3d_device` (`0x434BD0`): the same flag forces
  `[0x575ABC]=1` → `PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE`.
- **Render frameskip** `byte [0x575A8B]`: run `N+1` logic ticks per drawn frame.
  Read every frame, so a runtime write sticks.
- Single-instance mutex: `CreateMutexA("Touhou YouYouMu App")` at `0x435BE9`,
  `ERROR_ALREADY_EXISTS` check `jne` at `0x435BFF`.

`launch_uncapped.py` launches th07.exe suspended and NOPs three `je`s
(2 bytes each) before it runs, then resumes:

| VA | `74 xx` → `90 90` | effect |
|---|---|---|
| `0x4348CC` | limiter, QPC path | frame always runs |
| `0x434997` | limiter, timeGetTime path | frame always runs |
| `0x434C8A` | `init_d3d_device` | always forces the no-vsync / IMMEDIATE path |

**Measured** (`probe_uncap.py`, in a stage): `logic_rate = (frameskip+1) x 144`
(144 = the test machine's monitor; `Present()` is DWM-capped in a window):

| `0x575A8B` | ticks/sec | x real-time |
|---|---|---|
| 1 (default) | 288 | 4.8x |
| 3 | 576 | 9.6x |
| 9 | 1440 | **24x** |

Since the agent reads/writes **memory**, not pixels, `run_all_on_tick` running
every logic tick is all that matters — render frameskip costs the agent nothing.
CPU ceiling (higher frameskip / true headless) not yet measured; needs a stable
harness. The game crashed twice during hands-off high-speed poking — expected to
be a non-issue once an agent is actually playing headless with clean resets, but
the harness must relaunch crashed instances.

### Remaining for the training harness (not feasibility)

- Parallel instances: NOP the mutex `jne` at `0x435BFF` (or close the named
  mutex handle) + write input via `INPUT_CUR` so no instance needs focus +
  unpause-on-defocus (window-active check in `do_tick`, `[0x575C20+8]`).
- Programmatic episode reset from the Continue screen (menu automation).
- True headless (no window / skip `Present`) for max speed + stability.
