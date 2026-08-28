"""PPO on Th07Env (Stage 1, Lunatic). Run from repo root.

    .venv\\Scripts\\python train_ppo.py --n-envs 8 --steps 5_000_000

Checkpoints + tensorboard land in runs/<name>/. The N game processes are the
real CPU cost - the MLP policy is tiny, so torch is capped to a few threads so
it doesn't starve the games' do_tick threads (that was crashing long runs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))

import torch  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402

from hud import TrainHud  # noqa: E402
from vec import make_vec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--name", default="ppo_st1")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--no-hud", action="store_true", help="skip the status window")
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="cap torch CPU threads so the games get scheduler time")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))

    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)

    print(f"launching {args.n_envs} game instances...")
    venv = make_vec(n_envs=args.n_envs, frame_skip=args.frame_skip)
    print("all envs up.")

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(run))
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=512, batch_size=2048, n_epochs=4,
            gamma=0.995, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
            learning_rate=3e-4, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(run), verbose=1,
        )

    cbs = [CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                              save_path=str(run), name_prefix="ppo")]
    if not args.no_hud:
        cbs.append(TrainHud(frame_skip=args.frame_skip, total_steps=args.steps))
    try:
        model.learn(total_timesteps=args.steps, callback=cbs,
                    reset_num_timesteps=args.resume is None, progress_bar=True)
    finally:
        model.save(str(run / "final"))
        venv.close()


if __name__ == "__main__":
    main()
