"""PPO fine-tuning on the REAL game (Th07Env, Stage 1, Lunatic).

    .venv\\Scripts\\python train_ppo.py --n-envs 6 --warmstart runs_sim/ppo_v29/snap_0092M.pt

Warm-starts the policy from a sim checkpoint (sb3_bridge), then continues RL
against N parallel real th07 instances. Each checkpoint is also exported as an
MLPPolicy .pt (runs/<name>/mlp_<steps>.pt) so the transfer daemon / deathcam /
DLL can load it unchanged.

The N game processes are the real cost - the MLP is tiny, so torch is capped to
a few threads so it doesn't starve the games' do_tick threads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))

import torch  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402

from hud import TrainHud  # noqa: E402
from vec import make_vec  # noqa: E402
from sb3_bridge import warmstart_from_mlp, export_mlp, PY_KW  # noqa: E402


class MlpExport(BaseCallback):
    """Every `every` steps, write the policy as an MLPPolicy .pt for the daemon."""

    def __init__(self, run: Path, every: int):
        super().__init__()
        self.run, self.every, self._next = run, every, every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            export_mlp(self.model, self.run / f"mlp_{self.num_timesteps}.pt")
            export_mlp(self.model, self.run / "last_mlp.pt")
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=180.0,
                    help="episode cap - Stage 1 + Letty is ~150s")
    ap.add_argument("--name", default="ppo_real_st1")
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="MLPPolicy .pt to initialise the actor from")
    ap.add_argument("--resume", type=Path, default=None, help="SB3 .zip to resume")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)

    print(f"launching {args.n_envs} game instances (Lunatic, hard-reset)...", flush=True)
    venv = make_vec(n_envs=args.n_envs, frame_skip=args.frame_skip,
                    max_seconds=args.max_seconds, hard_reset=True)
    print("all envs up.", flush=True)

    if args.resume:
        model = PPO.load(args.resume, env=venv, tensorboard_log=str(run))
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=args.n_steps, batch_size=max(64, args.n_steps * args.n_envs // 4),
            n_epochs=4, gamma=0.995, gae_lambda=0.95,
            ent_coef=args.ent_coef, clip_range=0.2, learning_rate=args.lr,
            vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=PY_KW,
            tensorboard_log=str(run), verbose=1, device="cpu",
        )
        if args.warmstart:
            warmstart_from_mlp(model, args.warmstart)
            export_mlp(model, run / "mlp_0.pt")     # sanity: daemon can eval step 0

    cbs = [
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=str(run), name_prefix="ppo"),
        MlpExport(run, every=max(200_000, args.n_steps * args.n_envs * 4)),
    ]
    if not args.no_hud:
        cbs.append(TrainHud(frame_skip=args.frame_skip, total_steps=args.steps))

    try:
        model.learn(total_timesteps=args.steps, callback=cbs,
                    reset_num_timesteps=args.resume is None, progress_bar=True)
    finally:
        model.save(str(run / "final"))
        export_mlp(model, run / "final_mlp.pt")
        venv.close()


if __name__ == "__main__":
    main()
