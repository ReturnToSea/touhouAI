"""Real-game transfer monitor. Runs alongside sim/train.py as a SEPARATE process
(CPU .venv, not .venv-cuda) - it does not touch training.

Loops: read the current train step count -> load the latest checkpoint -> play
one episode to death -> record (wallclock, steps, survival, score) -> append to
runs_sim/<run>/realtransfer.npy -> repeat. One persistent th07 process: each
episode resets via the engine's own "Give Up and Retry" (env.reset(hard=True),
~0.2s) instead of a relaunch. Rebuilt on error or every REBUILD_EVERY episodes.

The hud plots the smoothed survival vs steps next to the sim curve. If the real
line diverges DOWN from the sim curve as steps rise, that's sim overfitting
(the ppo_v26 failure mode) - visible live instead of at a manual test.

    .venv\\Scripts\\python sim\\transfer_daemon.py ppo_v27
    .venv\\Scripts\\python sim\\transfer_daemon.py ppo_v27 --checkpoint best.pt --show

th07 launches without stealing focus (SW_SHOWNOACTIVATE) and its window is then
minimised out of the way; it keeps ticking in the background. --show leaves it
on-screen.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 2 threads: the obs-build march in native/obs.py parallelises a bit, but the
# default ~12 just thrash. The game itself ticks fine while its window is in the
# background - the earlier "freeze" was this being slow, not paused.
torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "native"))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402
import killall as _killall          # noqa: E402

RUNS = HERE.parent / "runs_sim"
_u32 = ctypes.windll.user32


class _Stalled(Exception):
    """The in-game score froze while alive - an unadvanced dialogue. Drop it."""


def _minimise_pid_windows(pid: int) -> None:
    """SW_SHOWMINNOACTIVE th07's windows - it keeps ticking in the background
    (no defocus pause in this config) and stays out of the way."""
    want = ctypes.c_ulong(pid)

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        p = ctypes.c_ulong()
        _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == want.value:
            _u32.ShowWindow(hwnd, 7)   # SW_SHOWMINNOACTIVE
        return True

    _u32.EnumWindows(cb, 0)


def _train_steps(run_dir: Path) -> float:
    hp = run_dir / "history.npy"
    try:
        h = np.load(hp)
        if h.ndim == 2 and len(h):
            return float(h[-1, 1])
    except Exception:
        pass
    return 0.0


def _append_row(path: Path, row) -> None:
    row = np.asarray(row, np.float64).reshape(1, -1)
    if path.exists():
        try:
            old = np.load(path)
            if old.ndim == 2 and old.shape[1] < row.shape[1]:   # pad old (pre-censored col)
                old = np.hstack([old, np.zeros((len(old), row.shape[1] - old.shape[1]))])
            row = np.vstack([old, row])
        except Exception:
            pass
    tmp = path.with_name(path.stem + "_tmp.npy")   # np.save appends .npy to bare names
    with open(tmp, "wb") as f:
        np.save(f, row)
    tmp.replace(path)


def one_episode(env, pol, frame_skip, cap, hb=None, hard=False):
    obs, _ = env.reset(options={"hard": True} if hard else None)
    steps = 0
    done = False
    info = {"score": 0}
    t0 = time.time()
    prev_frame, stall = -1, 0
    prev_score, score_stall = 0, 0
    prev_ppos, ppos_stall = (0.0, 0.0), 0
    force_shoot = 0
    while not done:
        a = int(pol.act(obs))
        # the DLL auto-skips menus but not in-game dialogue (pre-Cirno / post-boss
        # etc). It locks the player and advances one line per shoot PRESS. If the
        # policy stops shooting at that moment -> player + score freeze forever.
        # Detect the freeze and PULSE shoot (edge-triggered, like the menu nav).
        if score_stall > 120 and ppos_stall > 120:
            force_shoot = 240
        if force_shoot > 0:
            a = (a % 18) + (18 if (force_shoot % 12) < 4 else 0)   # ~4 on / 8 off
            force_shoot -= 1
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        done = term or trunc
        s = env.h.s
        ppos = (round(s.player_x, 1), round(s.player_y, 1))
        ppos_stall = ppos_stall + 1 if ppos == prev_ppos else 0
        prev_ppos = ppos
        fr = info.get("frame", -1)
        stall = stall + 1 if fr == prev_frame else 0
        prev_frame = fr
        if stall > 600:
            raise RuntimeError(f"game frame counter stuck at step {steps}")
        s = env.h.s
        sc = int(s.score)
        score_stall = score_stall + 1 if sc == prev_score else 0
        prev_score = sc
        surv_now = steps * frame_skip / 60.0
        # score AND player both frozen for ~45s despite the shoot-mash above = a
        # genuinely stuck game (dialogue that won't advance). Drop it - it's an
        # env bug, not a policy result. The real fix is a DLL dialogue-skip.
        if score_stall > 900 and ppos_stall > 900 and surv_now > 150.0:
            at = (steps - max(score_stall, ppos_stall)) * frame_skip / 60.0
            if hb:
                f0 = int(s.frame)
                dump = []
                for _ in range(4):
                    env.step(0)
                    s = env.h.s
                    dump.append(f"f={s.frame} pst={s.player_state} lives={s.lives:.1f} "
                                f"bombs={s.bombs:.1f} pos=({s.player_x:.0f},{s.player_y:.0f}) "
                                f"gm={s.gamemode} stg={s.stage} tick={s.tick_status} "
                                f"boss={s.boss_present}/{s.boss_hp:.0f}/{s.boss_hp_max:.0f} "
                                f"score={s.score} cherry={s.cherry}")
                hb(f"    STUCK at ~{at:.0f}s - dropping. state over 4 steps:")
                for d in dump:
                    hb("      " + d)
            raise _Stalled(at)
        if hb and steps % 500 == 0:
            hb(f"    ... {steps} st  {fr/60:.0f}s  score {sc}  stage {s.stage}  "
               f"lives {s.lives:.0f}  boss {s.boss_present}({s.boss_hp:.0f}/{s.boss_hp_max:.0f})  "
               f"player ({s.player_x:.0f},{s.player_y:.0f})"
               + ("  [force-shoot]" if force_shoot > 0 else ""))
    surv = steps * frame_skip / 60.0
    return surv, int(info.get("score", 0)), (surv >= cap - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="leave game windows on-screen")
    ap.add_argument("--cap", type=float, default=400.0,
                    help="in-game seconds to cap each episode at (censored past this)")
    ap.add_argument("--settle", type=float, default=3.0, help="pause between episodes (s)")
    args = ap.parse_args()

    run_dir = RUNS / args.run
    out = run_dir / "realtransfer.npy"
    ckpt = run_dir / args.checkpoint
    print(f"transfer daemon: {args.run}  <- {args.checkpoint}   -> {out.name}", flush=True)

    # One persistent game process. Each episode is an engine-level "Give Up and
    # Retry" (env.reset(hard=True)) - no relaunch, no force-tab-out. The process
    # is only rebuilt on an exception or every REBUILD_EVERY episodes (belt-and-
    # braces against any slow state leak across in-place reloads).
    REBUILD_EVERY = 40
    n = n_stall = 0
    env = None
    ep_on_env = 0
    while True:
        if not ckpt.exists():
            print(f"waiting for {ckpt} ...", flush=True)
            time.sleep(20)
            continue
        steps = _train_steps(run_dir)
        try:
            pol = MLPPolicy.load(ckpt)
            if env is not None and ep_on_env >= REBUILD_EVERY:
                try:
                    env.close()
                except Exception:
                    pass
                env = None
            if env is None:
                _killall.killall()
                env = Th07Env(frame_skip=args.frame_skip, max_seconds=args.cap + 5,
                              render=False)
                ep_on_env = 0
                if not args.show:
                    _minimise_pid_windows(env.pid)
            t0 = time.time()
            surv, score, censored = one_episode(
                env, pol, args.frame_skip, args.cap,
                hb=lambda m: print(m, flush=True), hard=(ep_on_env > 0))
            n += 1
            ep_on_env += 1
            _append_row(out, [time.time(), steps, surv, score, float(censored)])
            print(f"[{n:4d}]  {steps/1e6:6.1f}M steps   {surv:6.1f}s{'+' if censored else ' '}  "
                  f"score {score:>9d}   ({time.time() - t0:.0f}s wall)", flush=True)
        except _Stalled:
            n_stall += 1
            ep_on_env += 1
            print(f"    (dropped - {n_stall} stalled of {n + n_stall} total)", flush=True)
        except Exception as e:
            print(f"episode failed: {type(e).__name__}: {e}", flush=True)
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None
            try:
                _killall.killall()
            except Exception:
                pass
            time.sleep(5)
        time.sleep(args.settle)


if __name__ == "__main__":
    main()
