"""PPO on FightSim - replayed REAL Cirno recordings. The PoC: does training on
real boss content transfer to the real game better than the made-up danmaku sim?

    .venv-cuda/Scripts/python sim/train_fight.py --name fight_cirno --steps 200e6

Exports runs_sim/<name>/mlp_*.pt - load with native/record_boss_driven.py-style
eval against the real Cirno.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fight_replay import FightSim
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native"))
from obs import OBS_DIM
from policy import MLPPolicy, N_ACTIONS

HID = (256, 256)


def mlp(sizes, gain):
    L = []
    for i in range(len(sizes) - 2):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        L += [lin, nn.Tanh()]
    last = nn.Linear(sizes[-2], sizes[-1])
    nn.init.orthogonal_(last.weight, gain); nn.init.zeros_(last.bias)
    return nn.Sequential(*L, last)


class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp([OBS_DIM, *HID, N_ACTIONS], 0.01)
        self.critic = mlp([OBS_DIM, 256, 256, 1], 1.0)

    def export(self, p):
        m = MLPPolicy(hidden=HID, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
        m.net.load_state_dict(self.actor.state_dict())
        m.save(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fight_cirno")
    ap.add_argument("--fight", default="cirno")
    ap.add_argument("--steps", type=float, default=150e6)
    ap.add_argument("--B", type=int, default=12288)
    ap.add_argument("--max-frames", type=int, default=11000,
                    help="cap on recording length loaded into FightSim (Letty's "
                    "full dodge-only fight is ~10750 frames / 179s)")
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=32768)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sim", choices=("replay", "ecl"), default="replay",
                    help="replay = recorded fights; ecl = fresh ECL-VM danmaku "
                         "schedules (sim/danmaku_ecl.py), Letty only")
    ap.add_argument("--ecl-schedules", type=int, default=48)
    ap.add_argument("--power-lo", type=float, default=None,
                    help="raw power (0-128) sampled per episode; default 10-50 "
                         "(a real Lunatic run hits Letty at ~35..105)")
    ap.add_argument("--power-hi", type=float, default=None)
    ap.add_argument("--phase-mix", type=float, default=None,
                    help="initial phase_start_mix (ecl default 0.55), annealed to 0")
    ap.add_argument("--phase-mix-frac", type=float, default=0.6,
                    help="anneal phase_start_mix to 0 over this fraction of steps")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    run = Path("runs_sim") / args.name
    run.mkdir(parents=True, exist_ok=True)
    import json
    (run / "meta.json").write_text(json.dumps({
        "algo": "ppo_fight", "steps": args.steps, "hidden": list(HID),
        "B": args.B, "fight": args.fight}))

    ecl_recs = ecl_eval_recs = None
    if args.sim == "ecl":
        from danmaku_ecl import build_schedules
        print(f"[ecl] building {args.ecl_schedules} + 8 danmaku schedules...", flush=True)
        ecl_recs = build_schedules(args.ecl_schedules, seed0=0)
        ecl_eval_recs = build_schedules(8, seed0=10_000)   # held-out seeds
        args.fight = "letty"

    # ECL sim: no x-mirror, no field rotation (each schedule is already a fresh
    # RNG roll). phase_start_mix STARTS high and anneals to 0 -- Lingering Cold /
    # Table-Turning are real ~50s survival walls, so without mid-fight starts the
    # policy never sees phases 2-4 until it can already clear NS1. It anneals to 0
    # over the first `--phase-mix-frac` of training so the final policy is tuned
    # on honest NS1-start runs.
    ph_mix0 = args.phase_mix if args.phase_mix is not None else (
        0.55 if args.sim == "ecl" else 0.0)
    ecl_kw = (dict(mirror=False, field_rot_deg=0.0, phase_start_mix=ph_mix0)
              if args.sim == "ecl" else {})
    if args.power_lo is not None:
        ecl_kw["power_lo"] = args.power_lo
    if args.power_hi is not None:
        ecl_kw["power_hi"] = args.power_hi
    sim = FightSim(B=args.B, name=args.fight, device=dev, seed=args.seed,
                   max_frames=args.max_frames, recs=ecl_recs, **ecl_kw)
    ev_kw = {**ecl_kw, "phase_start_mix": 0.0}
    ev = FightSim(B=1024, name=args.fight, device=dev, seed=args.seed + 999,
                  max_frames=args.max_frames,
                  randomize=False, recs=ecl_eval_recs, **ev_kw)   # clean phase-0
    if args.sim != "ecl":                        # replay: eval shares the training
        for k in ("pos", "bhalf", "boss", "en"):  # data (identical) to save memory;
            setattr(ev, k, getattr(sim, k))       # ecl eval uses held-out seeds
    ac = AC().to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr, eps=1e-5)
    n_upd_tot = max(1, int(args.steps / (args.B * args.rollout)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_upd_tot, eta_min=args.lr * 0.15)
    B, T = args.B, args.rollout

    # in-training eval is a cheap sanity signal only (same recordings -> prone to
    # memorisation). Real transfer number comes from native/eval_boss.py on the
    # snapshots. FightSim.step == 1 frame; the full fight is ~10750 frames but a
    # ~5000-frame window (83s, into phase 3) is enough to see it learning.
    @torch.no_grad()
    def evaluate(n=11500):                       # n > full 179s fight -> a pure
        o = ev.reset()                           # dodge-survivor is fully visible
        alive = torch.ones(ev.B, dtype=torch.bool, device=dev)
        life = torch.zeros(ev.B, device=dev)
        cleared = torch.zeros(ev.B, dtype=torch.bool, device=dev)
        deepest = torch.zeros(ev.B, dtype=torch.long, device=dev)
        kill_f = torch.zeros(ev.B, device=dev)   # frame the boss was defeated
        for step in range(n):
            a = ac.actor(o).argmax(-1)
            o, r, dn = ev.step(a)
            life += alive.float()
            deepest = torch.maximum(
                deepest, torch.where(alive, ev.phase_idx,
                                     torch.zeros_like(ev.phase_idx)))
            nk = alive & getattr(ev, "last_killed", torch.zeros_like(alive))
            kill_f = torch.where(nk, torch.full_like(kill_f, float(step + 1)), kill_f)
            cleared |= nk
            alive = alive & ~dn
        s = (life / 60.0).cpu().numpy()          # step == frame
        clr = float(cleared.float().mean())
        ph = float(deepest.float().mean()) + 1.0
        kt = kill_f[cleared]
        ktime = float(kt.median().item() / 60.0) if kt.numel() else float("nan")
        return float(np.median(s)), float(np.mean(s)), clr, ph, ktime

    obs = sim.reset()
    ob = torch.zeros(T, B, OBS_DIM, device=dev)
    ab = torch.zeros(T, B, dtype=torch.long, device=dev)
    lb = torch.zeros(T, B, device=dev)
    vb = torch.zeros(T, B, device=dev)
    rb = torch.zeros(T, B, device=dev)
    db = torch.zeros(T, B, device=dev)
    total, upd, t0 = 0, 0, time.perf_counter()
    hist = []
    best_score = -1e9

    while total < args.steps:
        for t in range(T):
            with torch.no_grad():
                lg = ac.actor(obs); v = ac.critic(obs).squeeze(-1)
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample()
            ob[t], ab[t], lb[t], vb[t] = obs, a, dist.log_prob(a), v
            obs, r, dn = sim.step(a)
            rb[t], db[t] = r, dn.float()
        total += T * B
        upd += 1

        if args.sim == "ecl" and ph_mix0 > 0:      # anneal mid-fight starts -> 0
            frac = min(1.0, total / max(1.0, args.phase_mix_frac * args.steps))
            sim.phase_start_mix = ph_mix0 * (1.0 - frac)

        with torch.no_grad():
            lastv = ac.critic(obs).squeeze(-1)
            adv = torch.zeros(T, B, device=dev)
            gae = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                nz = 1.0 - db[t]
                nv = lastv if t == T - 1 else vb[t + 1]
                delta = rb[t] + args.gamma * nv * nz - vb[t]
                gae = delta + args.gamma * args.lam * nz * gae
                adv[t] = gae
            ret = adv + vb
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bo, ba = ob.reshape(-1, OBS_DIM), ab.reshape(-1)
        blp, bad, bre, bva = (lb.reshape(-1), adv.reshape(-1),
                              ret.reshape(-1), vb.reshape(-1))
        n = bo.shape[0]
        for _ in range(args.epochs):
            for s in range(0, n, args.minibatch):
                idx = torch.randint(0, n, (args.minibatch,), device=dev)
                lg = ac.actor(bo[idx])
                d = torch.distributions.Categorical(logits=lg)
                lp = d.log_prob(ba[idx])
                ratio = (lp - blp[idx]).exp()
                pg = -torch.min(ratio * bad[idx],
                                ratio.clamp(1 - args.clip, 1 + args.clip) * bad[idx]).mean()
                v = ac.critic(bo[idx]).squeeze(-1)
                vl = 0.5 * (v - bre[idx]).pow(2).mean()
                loss = pg + 0.5 * vl - args.ent * d.entropy().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
        sched.step()

        if upd % 30 == 0:
            med, mean, f30, ph, ktime = evaluate()   # f30 = kill rate, ktime = median kill-time (s)
            wall = time.perf_counter() - t0
            sps = total / wall
            kts = f"{ktime:4.0f}s" if not np.isnan(ktime) else "  --"
            print(f"upd {upd:4d}  {total/1e6:6.1f}M  {sps/1e3:4.0f}k/s  "
                  f"surv med {med:5.1f}s  kill {f30*100:3.0f}%  "
                  f"kill-time {kts}  phase {ph:.2f}/4  lr {sched.get_last_lr()[0]:.1e}",
                  flush=True)
            hist.append((wall, total, med, mean, f30, ktime))  # 6-col fight schema
            np.save(run / "history.npy", np.array(hist))
            ac.export(run / "last_mlp.pt")
            # keep the best checkpoint by kill-rate (tie-break survival) so a
            # later training thrash can't lose the peak
            score = f30 + med / 1000.0
            if score > best_score:
                best_score = score
                ac.export(run / "best_mlp.pt")
                print(f"    ^ new best (kill {f30*100:.0f}%, surv {med:.0f}s)", flush=True)
            if upd % 60 == 0:
                ac.export(run / f"mlp_{int(total/1e6)}M.pt")

    ac.export(run / "final_mlp.pt")


if __name__ == "__main__":
    main()
