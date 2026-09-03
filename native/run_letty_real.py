"""One command to run the real-game Letty training with live monitoring.

    .venv\\Scripts\\python native\\run_letty_real.py                # start everything
    .venv\\Scripts\\python native\\run_letty_real.py --hud-only      # just the overview
    .venv\\Scripts\\python native\\run_letty_real.py --no-daemon     # trainer + hud, no greedy eval

Starts, in order:
  1. train_ppo_dll.py         - N hooked th07 instances, PPO on the real Letty fight
  2. fight_transfer_daemon    - one more hooked game, greedy-evals each checkpoint
  3. fight_dll_hud.py         - the live overview window (foreground)

Closing the HUD leaves the trainer + daemon running. Ctrl-C here, or
`python native/killall.py`, stops the games; the trainer/daemon are child
processes and are terminated with this script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _wait_for_instances(log: Path, n: int, timeout: float = 240.0):
    print(f"waiting for {n} hooked games to come up...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if log.exists():
            txt = log.read_text(errors="replace")
            if "[RealRolloutVec]" in txt or txt.count("nav ") >= n:
                print("  up.", flush=True)
                return True
            if "Traceback" in txt:
                print("  trainer crashed - see the log:", flush=True)
                print("\n".join(txt.splitlines()[-20:]))
                return False
        time.sleep(2.0)
    print("  timed out waiting - check the log.", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ppo_real_letty")
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600_000_000)
    ap.add_argument("--warmstart", default="runs_sim/ppo_v29/best.pt")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=5, help="greedy eps / checkpoint")
    ap.add_argument("--no-daemon", action="store_true")
    ap.add_argument("--hud-only", action="store_true")
    ap.add_argument("--extra", default="", help="extra args passed to train_ppo_dll")
    args = ap.parse_args()

    run = ROOT / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    log = run / "train.log"
    procs = []

    if not args.hud_only:
        cmd = [str(PY), "-u", "train_ppo_dll.py",
               "--n-envs", str(args.n_envs), "--steps", str(args.steps),
               "--warmstart", args.warmstart, "--name", args.name,
               "--max-ep-seconds", "200", "--anneal-lr",
               "--ent-coef", str(args.ent_coef)]
        if args.extra:
            cmd += args.extra.split()
        print("trainer:", " ".join(cmd), flush=True)
        lf = open(log, "w")
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=lf,
                                      stderr=subprocess.STDOUT))

        if not args.no_daemon:
            if not _wait_for_instances(log, args.n_envs):
                print("not starting the daemon (trainer not healthy).", flush=True)
            else:
                dlog = open(run / "daemon.log", "w")
                dcmd = [str(PY), "-u", "sim/fight_transfer_daemon.py", args.name,
                        "--runsdir", "runs", "--which", "2", "--k", str(args.k)]
                print("daemon: ", " ".join(dcmd), flush=True)
                procs.append(subprocess.Popen(dcmd, cwd=ROOT, stdout=dlog,
                                              stderr=subprocess.STDOUT))
                time.sleep(2.0)

    try:
        subprocess.run([str(PY), "native/fight_dll_hud.py", args.name], cwd=ROOT)
    except KeyboardInterrupt:
        pass

    if procs:
        print("\nHUD closed. trainer + daemon are still running.", flush=True)
        print("  monitor again:  .venv\\Scripts\\python native\\fight_dll_hud.py "
              f"{args.name}", flush=True)
        print("  stop all games: .venv\\Scripts\\python native\\killall.py "
              "(then Ctrl-C this)", flush=True)
        try:
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            subprocess.run([str(PY), "native/killall.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
