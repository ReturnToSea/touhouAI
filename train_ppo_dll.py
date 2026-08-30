"""PPO fine-tuning on the REAL game with DLL-side rollout collection.

    .venv\\Scripts\\python train_ppo_dll.py --n-envs 12 --warmstart runs_sim/ppo_v29/snap_0092M.pt

The DLL runs each T-step trajectory internally (obs -> actor -> sample -> tick ->
env.py reward -> record, hard-reset on death) so the games run at native speed
with no per-step Python. This script only ships weights, computes GAE, and does
the PPO update. Checkpoints export as MLPPolicy .pt (runs/<name>/mlp_*.pt) for
the transfer daemon / deathcam / DLL.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))
from obs import OBS_DIM                       # noqa: E402
from policy import MLPPolicy, N_ACTIONS       # noqa: E402
from real_rollout import RealRolloutVec       # noqa: E402

HID = (256, 256)


def mlp(sizes, gain_last):
    layers = []
    for i in range(len(sizes) - 2):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
    last = nn.Linear(sizes[-2], sizes[-1])
    nn.init.orthogonal_(last.weight, gain_last); nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp([OBS_DIM, *HID, N_ACTIONS], 0.01)
        self.critic = mlp([OBS_DIM, 256, 256, 1], 1.0)

    def flat_actor(self) -> np.ndarray:
        # W0,b0,W1,b1,W2,b2 - matches th07hook.cpp mlp_logits / policy.py get_flat
        return torch.cat([p.reshape(-1) for p in self.actor.parameters()]
                         ).detach().cpu().numpy().astype(np.float32)

    def export_mlp(self, path):
        pol = MLPPolicy(hidden=HID, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        pol.net.load_state_dict(self.actor.state_dict())
        pol.save(path)

    def warmstart(self, mlp_path):
        src = MLPPolicy.load(mlp_path)
        self.actor.load_state_dict(src.net.state_dict())
        print(f"warm-started actor from {mlp_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--rollout", type=int, default=256, help="T (<= ROLL_T_MAX)")
    ap.add_argument("--max-ep-seconds", type=float, default=180.0)
    ap.add_argument("--name", default="ppo_real_dll")
    ap.add_argument("--warmstart", type=Path, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    args = ap.parse_args()

    torch.set_num_threads(4)
    run = Path("runs") / args.name
    run.mkdir(parents=True, exist_ok=True)

    ac = ActorCritic()
    if args.warmstart:
        ac.warmstart(args.warmstart)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)

    vec = RealRolloutVec(n_envs=args.n_envs, frame_skip=args.frame_skip,
                         max_ep_frames=int(args.max_ep_seconds * 60))
    ac.export_mlp(run / "mlp_0.pt")

    T, Nenv = args.rollout, args.n_envs
    total, upd = 0, 0
    t0 = time.perf_counter()
    hist = []
    ep_ret = np.zeros(Nenv, np.float32)
    ep_len = np.zeros(Nenv, np.float32)
    recent_len, recent_ret = [], []

    while total < args.steps:
        # ---- collect (all in the DLLs) ----
        w = ac.flat_actor()
        tc = time.perf_counter()
        obs, act, rew, done, last_obs, ep_ends = vec.collect(w, T, *HID)
        collect_s = time.perf_counter() - tc
        total += T * Nenv
        upd += 1

        # per-env episode return/length bookkeeping (for the survival metric)
        for t in range(T):
            ep_ret += rew[t]; ep_len += 1
            d = done[t] > 0.5
            if d.any():
                recent_ret += ep_ret[d].tolist()
                recent_len += ep_len[d].tolist()
                ep_ret[d] = 0; ep_len[d] = 0

        o = torch.from_numpy(obs).float()          # [T,N,OBS]
        a = torch.from_numpy(act).long()           # [T,N]
        r = torch.from_numpy(rew).float()
        dn = torch.from_numpy(done).float()
        lo = torch.from_numpy(last_obs).float()    # [N,OBS]

        with torch.no_grad():
            logits = ac.actor(o)
            logp_old = torch.log_softmax(logits, -1).gather(-1, a[..., None]).squeeze(-1)
            val = ac.critic(o).squeeze(-1)         # [T,N]
            last_val = ac.critic(lo).squeeze(-1)   # [N]
            adv = torch.zeros(T, Nenv)
            gae = torch.zeros(Nenv)
            for t in reversed(range(T)):
                nextv = last_val if t == T - 1 else val[t + 1]
                nonterm = 1.0 - dn[t]
                delta = r[t] + args.gamma * nextv * nonterm - val[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bo = o.reshape(-1, OBS_DIM); ba = a.reshape(-1)
        blp = logp_old.reshape(-1); badv = adv.reshape(-1)
        bret = ret.reshape(-1); bval = val.reshape(-1)
        n = bo.shape[0]
        for _ in range(args.epochs):
            for s in range(0, n, args.minibatch):
                idx = torch.randint(0, n, (args.minibatch,))
                lg = ac.actor(bo[idx])
                lp = torch.log_softmax(lg, -1).gather(-1, ba[idx][:, None]).squeeze(-1)
                ratio = (lp - blp[idx]).exp()
                pg = -torch.min(ratio * badv[idx],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[idx]).mean()
                v = ac.critic(bo[idx]).squeeze(-1)
                vcl = bval[idx] + (v - bval[idx]).clamp(-args.clip, args.clip)
                vl = 0.5 * torch.max((v - bret[idx]) ** 2, (vcl - bret[idx]) ** 2).mean()
                ent = -(torch.log_softmax(lg, -1) * torch.softmax(lg, -1)).sum(-1).mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

        ml = float(np.mean(recent_len[-200:])) if recent_len else 0.0
        surv_s = ml * args.frame_skip / 60.0
        sps = total / (time.perf_counter() - t0)
        print(f"upd {upd:4d}  {total/1e6:6.2f}M  {sps:6.0f}/s  collect {collect_s:4.1f}s  "
              f"surv {surv_s:6.1f}s  ep_ends {ep_ends:3d}  ent {ent.item():.2f}  "
              f"ret {np.mean(recent_ret[-200:]) if recent_ret else 0:.0f}", flush=True)
        hist.append((time.perf_counter() - t0, total, surv_s, float(ent.item())))
        np.save(run / "history.npy", np.array(hist))

        if upd % 8 == 0:
            ac.export_mlp(run / f"mlp_{total}.pt")
            ac.export_mlp(run / "last_mlp.pt")

    ac.export_mlp(run / "final_mlp.pt")
    vec.close()


if __name__ == "__main__":
    main()
