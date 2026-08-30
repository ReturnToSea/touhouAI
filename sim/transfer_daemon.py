"""Real-game transfer monitor. Runs alongside sim/train.py as a SEPARATE process
(CPU .venv, not .venv-cuda) - it does not touch training.

Loops: read the current train step count -> launch a headless th07 -> load the
latest checkpoint -> play one episode to death -> record (wallclock, steps,
survival, score) -> append to runs_sim/<run>/realtransfer.npy -> repeat.

The hud plots the smoothed survival vs steps next to the sim curve. If the real
line diverges DOWN from the sim curve as steps rise, that's sim overfitting
(the ppo_v26 failure mode) - visible live instead of at a manual test.

    .venv\\Scripts\\python sim\\transfer_daemon.py ppo_v27
    .venv\\Scripts\\python sim\\transfer_daemon.py ppo_v27 --checkpoint best.pt --show

th07 windows are minimised + kept from stealing focus. --show leaves them normal.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)   # B=1 policy forward - default 12 OMP threads just thrash

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "native"))
from env import Th07Env             # noqa: E402
from policy import MLPPolicy        # noqa: E402
import killall as _killall          # noqa: E402

RUNS = HERE.parent / "runs_sim"
_u32 = ctypes.windll.user32


def _minimise_pid_windows(pid: int) -> None:
    """SW_SHOWMINNOACTIVE every top-level window owned by pid."""
    SW_SHOWMINNOACTIVE = 7
    want = ctypes.c_ulong(pid)

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        p = ctypes.c_ulong()
        _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == want.value:
            _u32.ShowWindow(hwnd, SW_SHOWMINNOACTIVE)
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
            row = np.vstack([old, row])
        except Exception:
            pass
    tmp = path.with_name(path.stem + "_tmp.npy")   # np.save appends .npy to bare names
    with open(tmp, "wb") as f:
        np.save(f, row)
    tmp.replace(path)


def one_episode(env, pol, frame_skip):
    obs, _ = env.reset()
    steps = 0
    done = False
    info = {"score": 0}
    while not done:
        a = int(pol.act(obs))
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        done = term or trunc
    return steps * frame_skip / 60.0, int(info.get("score", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--show", action="store_true", help="don't minimise the game window")
    ap.add_argument("--max-seconds", type=float, default=36000.0)
    ap.add_argument("--settle", type=float, default=3.0, help="pause between episodes (s)")
    args = ap.parse_args()

    run_dir = RUNS / args.run
    out = run_dir / "realtransfer.npy"
    ckpt = run_dir / args.checkpoint
    print(f"transfer daemon: {args.run}  <- {args.checkpoint}   -> {out.name}", flush=True)

    n = 0
    while True:
        if not ckpt.exists():
            print(f"waiting for {ckpt} ...", flush=True)
            time.sleep(20)
            continue
        steps = _train_steps(run_dir)
        env = None
        try:
            pol = MLPPolicy.load(ckpt)
            env = Th07Env(frame_skip=args.frame_skip, max_seconds=args.max_seconds,
                          render=False)
            if not args.show:
                _minimise_pid_windows(env.pid)
            t0 = time.time()
            surv, score = one_episode(env, pol, args.frame_skip)
            n += 1
            _append_row(out, [time.time(), steps, surv, score])
            print(f"[{n:4d}]  {steps/1e6:6.1f}M steps   {surv:6.1f}s   score {score:>9d}   "
                  f"({time.time() - t0:.0f}s wall)", flush=True)
        except Exception as e:
            print(f"episode failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(5)
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            try:
                _killall.killall()
            except Exception:
                pass
        time.sleep(args.settle)


if __name__ == "__main__":
    main()
