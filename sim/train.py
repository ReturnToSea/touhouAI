"""Phase 2: train a dodging policy in the danmaku sim (GPU).

    .venv-cuda\\Scripts\\python sim\\train.py --algo ppo  --steps 30e6 --name ppo1
    .venv-cuda\\Scripts\\python sim\\train.py --algo es   --steps 30e6 --name es1

Both algorithms share the same sim and produce an actor whose architecture is
identical to native/policy.py MLPPolicy, so the result loads straight into the
real Th07Env (watch.py --evo, and the eventual transfer / fine-tune).

Checkpoints: runs_sim/<name>/best.pt, last.pt, history.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from danmaku import DanmakuSim          # noqa: E402
from obs import OBS_DIM                  # noqa: E402
from policy import MLPPolicy, N_ACTIONS  # noqa: E402


class RunningMeanStd(nn.Module):
    """Welford running mean/var over the batch axis. Buffers so it saves/loads
    and moves with the module. Standard obs/return normalisation (rl_games / SB3)."""

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x):                    # x: [..., *shape]
        x = x.reshape(-1, *self.mean.shape)
        bn = x.shape[0]
        if bn == 0:
            return
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        n = self.count + bn
        d = bm - self.mean
        self.mean += d * (bn / n)
        self.var.copy_((self.var * self.count + bv * bn + d * d * self.count * bn / n) / n)
        self.count.copy_(n)

    @property
    def std(self):
        return (self.var + 1e-8).sqrt()

    def norm(self, x):
        return (x - self.mean) / self.std


def mlp(sizes, out_gain=1.0):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
    last = nn.Linear(sizes[-2], sizes[-1])
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    layers.append(last)
    for m in layers:
        if isinstance(m, nn.Linear) and m is not last:
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, hidden=(64, 64)):
        super().__init__()
        self.hidden = tuple(hidden)
        self.actor = mlp([OBS_DIM, *hidden, N_ACTIONS], out_gain=0.01)   # == MLPPolicy.net
        self.critic = mlp([OBS_DIM, 128, 128, 1], out_gain=1.0)
        self.obs_rms = RunningMeanStd(OBS_DIM)     # v28: obs normalisation

    def forward(self, o):
        o = self.obs_rms.norm(o)
        return self.actor(o), self.critic(o).squeeze(-1)

    @torch.no_grad()
    def act(self, o, greedy=False):
        logits = self.actor(self.obs_rms.norm(o))
        if greedy:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()

    def export_mlp(self):
        """Fold the current obs normalisation into the actor's first Linear, so
        the exported net is a plain MLP that runs on RAW obs - transfer / watch /
        deathcam load it unchanged."""
        pol = MLPPolicy(hidden=self.hidden)
        sd = {k: v.clone() for k, v in self.actor.state_dict().items()}
        m = self.obs_rms.mean
        inv = 1.0 / self.obs_rms.std                 # [OBS_DIM]
        W0 = sd["0.weight"]                           # [h, OBS_DIM]
        sd["0.bias"] = sd["0.bias"] - (W0 * inv) @ m
        sd["0.weight"] = W0 * inv
        pol.net.load_state_dict(sd)
        return pol


# --------------------------------------------------------------------------- PPO
def train_ppo(args, sim, dev, run):
    ac = ActorCritic(args.hidden).to(dev)
    if args.init:
        src = MLPPolicy.load(args.init)
        ac.actor.load_state_dict(src.net.state_dict())
        print(f"warm-started actor from {args.init}", flush=True)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    # NOTE: not torch.compile'ing the policy - on this tiny [64,64] net the win
    # is marginal and it stalled a run. The sim itself IS compiled (big win).
    B, T = sim.B, args.rollout
    obs = sim.reset()

    # greedy (argmax) eval - this is what transfer uses, and it's much better
    # than the sampled policy PPO's rollout metric measures.
    from danmaku import DanmakuSim
    eval_sim = DanmakuSim(B=2048, device=dev, max_frames=20000,
                          alive_rew=args.alive_rew, seed=args.seed + 777,
                          compile=(dev == "cuda"))

    @torch.no_grad()
    def greedy_eval(n_dec=4800):     # 4800 dec * fs3 = 14400 f = 240 s ceiling
        # sync-free: everything stays on the GPU, single host transfer at the end.
        o = eval_sim.reset()
        B = eval_sim.B
        el = torch.zeros(B, device=dev)
        comp = torch.zeros(n_dec, B, device=dev)   # per-step: length if just died, else 0
        dw = torch.zeros((), device=dev)
        de = torch.zeros((), device=dev)
        nd = torch.zeros((), device=dev)
        for i in range(n_dec):
            a = ac.actor(ac.obs_rms.norm(o)).argmax(-1)
            o, _, done = eval_sim.step(a)
            el = el + 1.0
            df = done.float()
            comp[i] = el * df
            el = el * (1.0 - df)
            nd = nd + df.sum()
            dw = dw + eval_sim.step_death_wall.sum()
            de = de + eval_sim.step_death_enemy.sum()
        fs = 3.0 / 60.0
        lens = torch.cat([comp[comp > 0], el[el > 0]])          # ALL episodes (pooled)
        a = (lens * fs).sort().values.cpu().numpy()             # <-- one sync
        # honest metric: FIRST episode per env, equal weight (the pooled median
        # above is biased low - fast-dying situations auto-reset and cycle more).
        m1 = comp > 0
        died1 = m1.any(dim=0)
        first_len = comp.gather(0, torch.argmax(m1.float(), dim=0, keepdim=True)).squeeze(0)
        fe = (torch.where(died1, first_len, el) * fs).sort().values.cpu().numpy()
        ndeaths = float(nd)
        if a.size == 0:
            a = np.zeros(1)
        if fe.size == 0:
            fe = np.zeros(1)
        return dict(mean=float(a.mean()), med=float(np.median(a)),
                    p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                    f60=float((a > 60).mean()), f120=float((a > 120).mean()),
                    f180=float((a > 180).mean()),
                    med1=float(np.median(fe)), p90_1=float(np.percentile(fe, 90)),
                    f60_1=float((fe > 60).mean()), mean1=float(fe.mean()),
                    wall=float(dw) / max(ndeaths, 1), enemy=float(de) / max(ndeaths, 1))

    ep_ret = torch.zeros(B, device=dev)
    ep_len = torch.zeros(B, device=dev)
    hist, best, last_snap_M = [], -1e9, 0
    total, upd, t0 = 0, 0, time.perf_counter()
    recent_len, recent_ret = [], []
    next_log = 1_000_000            # cheap status line every ~1M steps

    ret_rms = RunningMeanStd(()).to(dev)   # v28: running std of the discounted return
    ret_disc = torch.zeros(B, device=dev)  # per-env discounted-return accumulator
    lr0 = args.lr

    o_buf = torch.zeros(T, B, OBS_DIM, device=dev)
    a_buf = torch.zeros(T, B, dtype=torch.long, device=dev)
    lp_buf = torch.zeros(T, B, device=dev)
    v_buf = torch.zeros(T, B, device=dev)
    r_buf = torch.zeros(T, B, device=dev)
    d_buf = torch.zeros(T, B, device=dev)

    while total < args.steps:
        # v28: linear LR anneal to 10% of lr0
        lr_now = lr0 * max(0.1, 1.0 - total / args.steps)
        for g in opt.param_groups:
            g["lr"] = lr_now
        for t in range(T):
            with torch.no_grad():
                logits, val = ac(obs)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
            o_buf[t], a_buf[t], lp_buf[t], v_buf[t] = obs, act, dist.log_prob(act), val
            obs, rew, done = sim.step(act)
            r_buf[t], d_buf[t] = rew, done.float()
            # v28: reward normalisation (SB3-style) - track the std of the
            # discounted return, divide rewards by it. train-only.
            ret_disc = args.gamma * ret_disc * (1.0 - done.float()) + rew
            ret_rms.update(ret_disc)
            ep_ret += rew
            ep_len += 1
            if done.any():
                recent_ret += ep_ret[done].tolist()
                recent_len += ep_len[done].tolist()
                ep_ret[done] = 0
                ep_len[done] = 0
        total += T * B
        upd += 1

        ac.obs_rms.update(o_buf)                          # v28: obs normalisation stats
        r_use = r_buf / ret_rms.std                       # v28: normalised rewards
        with torch.no_grad():
            _, last_v = ac(obs)
            adv = torch.zeros(T, B, device=dev)
            gae = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                nonterm = 1.0 - d_buf[t]
                nextv = last_v if t == T - 1 else v_buf[t + 1]
                delta = r_use[t] + args.gamma * nextv * nonterm - v_buf[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + v_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bo = o_buf.reshape(-1, OBS_DIM)
        ba = a_buf.reshape(-1)
        blp = lp_buf.reshape(-1)
        badv = adv.reshape(-1)
        bret = ret.reshape(-1)
        bv = v_buf.reshape(-1)
        n = bo.shape[0]
        mb = args.minibatch
        for _ in range(args.epochs):
            for s in range(0, n, mb):
                idx = torch.randint(0, n, (mb,), device=dev)
                logits, val = ac(bo[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(ba[idx])
                ratio = (lp - blp[idx]).exp()
                a1 = ratio * badv[idx]
                a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[idx]
                pg = -torch.min(a1, a2).mean()
                vclip = bv[idx] + (val - bv[idx]).clamp(-args.clip, args.clip)
                vl = 0.5 * torch.max((val - bret[idx]) ** 2, (vclip - bret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()

        fs = sim.frame_skip
        # --- cheap status line every ~1M steps (rollout metrics only) ---
        if total >= next_log:
            next_log = total + 1_000_000
            ml = float(np.mean(recent_len[-2000:])) if recent_len else 0.0
            sps = total / (time.perf_counter() - t0)
            print(f"  {total/1e6:6.1f}M  {sps/1e3:4.0f}k/s  sampled {ml*fs/60:5.1f}s  "
                  f"ent {ent.item():.2f}", flush=True)
            recent_len, recent_ret = recent_len[-4000:], recent_ret[-4000:]

        # --- full greedy eval + checkpoint every args.log_every updates ---
        if upd % args.log_every == 0:
            ml = float(np.mean(recent_len[-2000:])) if recent_len else 0.0
            e = greedy_eval()                        # dict, values already in seconds
            sps = total / (time.perf_counter() - t0)
            print(f"EVAL upd {upd:4d}  {total/1e6:6.1f}M  {sps/1e3:4.0f}k/s  "
                  f"med1 {e['med1']:5.1f}s  p90_1 {e['p90_1']:6.1f}s  >60s {e['f60_1']*100:3.0f}%  "
                  f"(pooled med {e['med']:.0f}/p90 {e['p90']:.0f})  "
                  f"sampled {ml*fs/60:4.1f}s  ent {ent.item():.2f}  lr {lr_now/1e-4:.2f}e-4  "
                  f"deaths: spam {e['wall']*100:2.0f}% enemy {e['enemy']*100:2.0f}% "
                  f"(rest=emitter)", flush=True)
            # history cols 0-11 (v17): wall_s, steps, mean, sampled_dec, ent, med, p90,
            #   f60, f120, f180, wallf, enemyf   -- 12-15 (v28): med1, p90_1, f60_1, mean1
            hist.append((time.perf_counter() - t0, total, e['mean'], ml, float(ent.item()),
                         e['med'], e['p90'], e['f60'], e['f120'], e['f180'], e['wall'], e['enemy'],
                         e['med1'], e['p90_1'], e['f60_1'], e['mean1']))
            score = e['med1'] + 0.5 * e['p90_1']    # v28: rank by the honest metric
            if score > best:
                best = score
                ac.export_mlp().save(run / "best.pt")
            ac.export_mlp().save(run / "last.pt")
            # timestamped snapshots every ~40M steps - sim greedy score does NOT
            # track real-game transfer (ppo_v26 got WORSE on the real game while
            # its sim median rose), so keep a trail to transfer-test and pick from.
            snap_M = int(total / 1e6)
            if snap_M - last_snap_M >= 40:
                last_snap_M = snap_M
                ac.export_mlp().save(run / f"snap_{snap_M:04d}M.pt")
            np.save(run / "history.npy", np.array(hist))


# ---------------------------------------------------------------------------- ES
def train_es(args, sim, dev, run):
    """Antithetic OpenAI-ES on the actor's flat weights. Population = sim.B / 2
    (each env pair evaluates +eps / -eps for a full episode)."""
    base = MLPPolicy(hidden=args.hidden)
    theta = torch.tensor(base.get_flat(), device=dev)
    P = sim.B // 2
    nparams = theta.numel()

    # flat-vector layout of MLPPolicy.net params: W0,b0,W1,b1,W2,b2
    ws, off = [], 0
    for p in base.net.parameters():
        ws.append((off, p.numel(), tuple(p.shape)))
        off += p.numel()

    def fwd(obs, W):
        """obs [B,O], per-env weights W [B,nparams] -> logits [B,36]."""
        def take(k):
            o, n, sh = ws[k]
            return W[:, o:o + n].reshape(-1, *sh)
        h = torch.tanh(torch.einsum("bi,boi->bo", obs, take(0)) + take(1))
        h = torch.tanh(torch.einsum("bi,boi->bo", h, take(2)) + take(3))
        return torch.einsum("bi,boi->bo", h, take(4)) + take(5)

    opt = torch.optim.Adam([theta], lr=args.lr)
    hist, best, total, gen, t0 = [], -1e9, 0, 0, time.perf_counter()

    while total < args.steps:
        eps = torch.randn(P, nparams, device=dev)
        W = torch.cat([theta + args.es_sigma * eps, theta - args.es_sigma * eps], 0)  # [2P,n]
        obs = sim.reset()
        ret = torch.zeros(sim.B, device=dev)
        alive = torch.ones(sim.B, dtype=torch.bool, device=dev)
        steps = 0
        while alive.any() and steps < args.es_horizon:
            with torch.no_grad():
                act = fwd(obs, W).argmax(-1)
            obs, rew, done = sim.step(act)
            ret += rew * alive.float()
            alive &= ~done
            steps += 1
            total += sim.B
        f = ret[:P] - ret[P:]                              # antithetic
        rank = f.argsort().argsort().float()
        rank = (rank / (P - 1) - 0.5)
        grad = (rank[:, None] * eps).mean(0) / args.es_sigma
        opt.zero_grad(set_to_none=True)
        theta.grad = -grad
        opt.step()
        gen += 1

        mean_surv = float((ret[:P] + ret[P:]).mean() / 2 / args.alive_rew) if args.alive_rew else 0
        if gen % args.log_every == 0:
            sps = total / (time.perf_counter() - t0)
            print(f"gen {gen:4d}  {total/1e6:6.1f}M  {sps/1e3:5.0f}k/s  "
                  f"surv~{mean_surv*sim.frame_skip/60:5.1f}s  best_ep_ret {ret.max():.2f}",
                  flush=True)
            hist.append((time.perf_counter() - t0, total, mean_surv, float(ret.max())))
            base.set_flat(theta.cpu().numpy())
            base.save(run / "last.pt")
            if mean_surv > best:
                best = mean_surv
                base.save(run / "best.pt")
            np.save(run / "history.npy", np.array(hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "es"], default="ppo")
    ap.add_argument("--name", default="run")
    ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--B", type=int, default=24576)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--max-frames", type=int, default=14400)   # 240 s training episodes
    ap.add_argument("--seed", type=int, default=0)
    # ppo
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--minibatch", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    # es
    ap.add_argument("--es-sigma", type=float, default=0.02)
    ap.add_argument("--es-horizon", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=26,   # updates between full greedy evals (~20M steps)
                    help="updates per logged point (~0.8M steps each)")
    ap.add_argument("--alive-rew", type=float, default=0.01)
    ap.add_argument("--eager-sim", action="store_true",
                    help="run the training sim eager (default: torch.compile, ~5x)")
    ap.add_argument("--init", type=str, default="",
                    help="warm-start the actor from a checkpoint (.pt) - avoid this "
                         "when the env/obs changed; start fresh instead")
    args = ap.parse_args()
    args.steps = int(args.steps)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    run = Path("runs_sim") / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({
        "algo": args.algo, "hidden": args.hidden, "B": args.B,
        "steps": args.steps, "started": time.time()}))
    # training sim is torch.compile'd by default (~5x; _r/_ri are @dynamo.disable
    # so the varied rand shapes no longer churn recompiles). --eager-sim to skip.
    sim = DanmakuSim(B=args.B, device=dev, max_frames=args.max_frames,
                     alive_rew=args.alive_rew, seed=args.seed,
                     compile=(dev == "cuda" and not args.eager_sim))
    print(f"{args.algo.upper()}  B={args.B}  hidden={args.hidden}  dev={dev}  "
          f"-> {run}", flush=True)
    (train_ppo if args.algo == "ppo" else train_es)(args, sim, dev, run)


if __name__ == "__main__":
    main()
