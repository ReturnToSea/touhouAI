"""Bridge between our MLPPolicy (.pt used by the DLL / transfer daemon / deathcam)
and an SB3 PPO policy, so real-game fine-tuning can warm-start from a sim
checkpoint and its checkpoints stay loadable everywhere else.

Layout match (both are Linear-Tanh-Linear-Tanh-Linear, hidden 256,256):
  MLPPolicy.net:            [0]Linear  [2]Linear  [4]Linear(->36)
  SB3 (net_arch pi=[256,256], activation Tanh):
      policy.mlp_extractor.policy_net: [0]Linear [2]Linear
      policy.action_net:               Linear(256->36)
The SB3 value net is left as-is (the critic re-learns on the real reward).
"""
from __future__ import annotations

from pathlib import Path

import torch

from policy import MLPPolicy, N_ACTIONS
from obs import OBS_DIM

HIDDEN = (256, 256)
# SB3 kwargs that produce the exact pi-path layout above
PY_KW = dict(net_arch=dict(pi=list(HIDDEN), vf=list(HIDDEN)),
             activation_fn=torch.nn.Tanh)


def _pi_linears(model):
    """The three Linear layers on SB3's policy (action) path, in order."""
    pn = model.policy.mlp_extractor.policy_net
    lins = [m for m in pn if isinstance(m, torch.nn.Linear)]
    assert len(lins) == 2, f"expected 2 hidden Linears, got {len(lins)}"
    return [lins[0], lins[1], model.policy.action_net]


def warmstart_from_mlp(model, mlp_path: str | Path) -> None:
    """Copy an MLPPolicy .pt's weights into an SB3 PPO model's policy path."""
    mlp = MLPPolicy.load(mlp_path)
    src = [m for m in mlp.net if isinstance(m, torch.nn.Linear)]
    assert len(src) == 3, f"MLPPolicy has {len(src)} Linears, expected 3"
    assert src[0].in_features == OBS_DIM and src[-1].out_features == N_ACTIONS
    with torch.no_grad():
        for s, d in zip(src, _pi_linears(model)):
            assert s.weight.shape == d.weight.shape, (s.weight.shape, d.weight.shape)
            d.weight.copy_(s.weight)
            d.bias.copy_(s.bias)
    print(f"[sb3_bridge] warm-started PPO policy from {mlp_path}")


def export_mlp(model, out_path: str | Path) -> None:
    """Write an MLPPolicy .pt from an SB3 PPO model (for the DLL / daemon / deathcam)."""
    pol = MLPPolicy(hidden=HIDDEN, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    dst = [m for m in pol.net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for s, d in zip(_pi_linears(model), dst):
            d.weight.copy_(s.weight.cpu())
            d.bias.copy_(s.bias.cpu())
    pol.save(out_path)


if __name__ == "__main__":
    # self-test: round-trip a checkpoint and confirm identical argmax on random obs
    import sys
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces

    src_pt = sys.argv[1] if len(sys.argv) > 1 else "runs_sim/ppo_v29/snap_0092M.pt"

    class _Stub(gym.Env):
        observation_space = spaces.Box(-10, 10, (OBS_DIM,), np.float32)
        action_space = spaces.Discrete(N_ACTIONS)
        def reset(self, **k): return self.observation_space.sample(), {}
        def step(self, a): return self.observation_space.sample(), 0.0, False, True, {}

    m = PPO("MlpPolicy", DummyVecEnv([lambda: _Stub()]), policy_kwargs=PY_KW, device="cpu")
    warmstart_from_mlp(m, src_pt)
    export_mlp(m, "/tmp/_bridge_rt.pt")
    a = MLPPolicy.load(src_pt)
    b = MLPPolicy.load("/tmp/_bridge_rt.pt")
    o = np.random.randn(2000, OBS_DIM).astype(np.float32)
    da = a.act_batch(o); db = b.act_batch(o)
    print(f"round-trip argmax match: {(da == db).mean()*100:.1f}%  (want 100.0)")
