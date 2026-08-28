"""Tiny deterministic MLP policy - shared by the evolution trainer and watch.py.

Kept independent of stable-baselines3: a plain torch module with a flat
parameter vector, argmax action, save/load to a .pt with its arch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 192
N_ACTIONS = 36


class MLPPolicy(nn.Module):
    def __init__(self, hidden=(64, 64), obs_dim=OBS_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden = tuple(hidden)
        layers = []
        d = obs_dim
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32)
        return int(self.net(x).argmax().item())

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        return self.net(x).argmax(dim=-1).cpu().numpy()

    # --- flat parameter vector (for mutation) ---
    @torch.no_grad()
    def get_flat(self) -> np.ndarray:
        return torch.cat([p.reshape(-1) for p in self.parameters()]).cpu().numpy()

    @torch.no_grad()
    def set_flat(self, vec: np.ndarray) -> None:
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.copy_(torch.as_tensor(vec[i:i + n], dtype=torch.float32).reshape(p.shape))
            i += n

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        torch.save({"hidden": self.hidden, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        pol = cls(hidden=blob["hidden"])
        pol.load_state_dict(blob["state_dict"])
        return pol
