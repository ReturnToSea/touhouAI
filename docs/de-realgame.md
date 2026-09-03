# Real-game fine-tuning

**Verdict:** a wash on Stage 1 *survival* — the sim policy already maxes it. But
the infrastructure is sound, and after the [ECL VM](de-generative-danmaku.md)
failed to teach a *kill*, real-game PPO became the primary path for the Letty
fight, not a fine-tuning afterthought. See
[The Letty fight, revisited](#the-letty-fight-revisited) below.

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
> policy can't do — a Stage 3+ boss, a stage portion with heavy enemy traffic,
> or a boss the sim can survive but not *kill*.

Stage 1 *survival* is **done** at ~238 s (the sim policy outlasts Letty). What
the sim policy can't do is kill her fast, and — after
[three simulators failed to teach that](de-generative-danmaku.md) — the same
`ST_ROLLOUT` infrastructure is now aimed straight at it.

## The Letty fight, revisited

The [ECL VM postmortem](de-generative-danmaku.md) ends here: stop trying to
simulate Letty faithfully, and train the fight on Letty's actual engine.

`native/run_letty_real.py` starts the trainer, the greedy-eval daemon, and the
live overview (`native/fight_dll_hud.py`) together.

- `train_ppo_dll.py --n-envs 12`, warm-started from `ppo_v29` (the
  procedural-sim policy that already clears Stage 1 and outlasts Letty).
- Each of the 12 [hooked games](hook.md) runs `ST_ROLLOUT` — a whole PPO
  trajectory collected in C, ~68× real-time, no per-step Python. On death or
  timeout the DLL reloads Stage 1; every episode plays the real stage into the
  real Letty fight. ~5–6 k decision-steps/s aggregate.
- Reward mirrors `env.py`: survival floor + score + **boss-HP-drained ×3** −
  death. The boss-damage term is what the sim policies never learned to chase.
- Checkpoints go to the [transfer daemon](de-letty-replay.md#the-transfer-daemon)
  like any other run.

**The bar:** beat the [replay baseline](de-letty-replay.md) — ~103 s
active-fight median, ~33 % real kill-rate — without the bimodal checkpoint swing.
This is on-policy PPO against the retail game, so whatever it learns is already
"transferred." Slower per step than any sim; no fidelity gap to fall through.

Results land in [Results](results.md) and the
[experiment log](experiment-log.md).
