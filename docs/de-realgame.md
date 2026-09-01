# Real-game fine-tuning

**Verdict:** a wash on Stage 1. Real-game PPO can't improve a stage the sim
policy already clears. Its value is on the parts the sim policy *fails*.

## The setup

The infrastructure is real and it works: `ST_ROLLOUT` runs a whole PPO trajectory
in C inside the [hooked game](hook.md) at ~68× per environment, `sb3_bridge`
adapts it, `train_ppo_dll.py` drives it. This is genuine on-policy PPO against the
actual game, not the sim.

## The result

12 M steps of real-game PPO on Stage 1:

| | Before | After |
|---|---|---|
| greedy survival | 238 s | 223 s |
| score | *x* | *~2x* |

Survival flat. Score doubled — the policy learned to shoot more and graze more —
but it wasn't dying on Stage 1 before and it wasn't dying after. Stage 1 was
already at the sim policy's ceiling; there was nothing to fine-tune.

## The lesson

> Real-game RL is slow (~68× vs the sim's thousands-of-parallel-games). Spending
> it on a solved stage buys nothing. Point it at the specific thing the sim
> policy can't do — a Stage 3+ boss, a stage portion with heavy enemy traffic —
> and only after the sim has taken the policy as far as the sim can.

Stage 1 is considered **done** at ~238 s (clears Letty). A 1cc needs Stages 2–6,
and that's a job for a better *sim* — [generative danmaku](ecl-vm.md) — not for
grinding real-game rollouts.
