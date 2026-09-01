# How it learns

Training is **PPO — Proximal Policy Optimization** — with a compact hand-rolled
implementation in the sim path (no Stable-Baselines3). PPO collects a batch of
on-policy experience, estimates how much better or worse each action was than the
policy expected, and nudges the policy toward the better ones by a bounded step.

## The loop

1. **Roll out.** Run all ~1000 sim environments for a fixed number of frames,
   sampling actions from the current [policy](ch-policy.md). Store
   `(obs, action, reward, value, log-prob)` per step.
2. **Advantages.** Compute returns and **GAE** advantages from the stored
   rewards and value estimates.
3. **Update.** For **3 epochs** over the batch, in minibatches of **32768**:
   maximise the **clipped surrogate objective** (ratio of new-to-old action
   probability, clipped so no single step moves the policy too far), plus a
   value-regression loss and a small **entropy bonus** (`0.002`) to keep
   exploring.
4. Repeat for hundreds of millions of frames.

## Hyperparameters

| | Value | Note |
|---|---|---|
| optimizer | **Adam**, lr `3e-4`, eps `1e-5`, betas `0.9 / 0.999` | |
| lr schedule | **cosine anneal** to `0.15 × lr` over the run | stabilised the late-training thrash |
| discount γ | `0.995` | lowered from `0.997` — long horizons amplified the reward oscillation |
| GAE λ | `0.95` | |
| epochs / rollout | `3` | |
| minibatch | `32768` | bigger stopped helping once compute-bound |
| grad-norm clip | `0.5` (global) | |

## Reward

The reward is where "dodge" vs "engage" is actually decided, and getting it
wrong was a [long saga](de-shooting.md). The rule that came out of it:

> Decide which objective is the goal and which is instrumental. Make the goal the
> dense signal and the other a thin floor — not a weighted sum.

For the [recorded-Letty](recording.md) work, **killing the boss is the goal**,
survival is instrumental (a dead policy deals no damage):

```
rew =  DMG_REW · boss-HP drained          # dense progress toward the kill
     + KILL_BONUS · killed                 # sparse, on the final phase
     − HIT_PEN  · hit                       # death
     + SURV_REW · alive                     # thin floor, ~30× smaller than aligned shooting
```

The [procedural-sim](sim.md) policies use a survival-only version — survival,
graze, a light bottom-camping penalty, no damage term — which is why they only
learn to dodge. Teaching one policy to both dodge *and* engage is the whole
subject of [the reward saga](de-shooting.md).

## Real-game PPO

`ST_ROLLOUT` runs an entire PPO trajectory **in C inside the
[hooked game](hook.md)** — including a reward function that mirrors `env.py` —
at ~68× real-time per environment, and `sb3_bridge` adapts it to an SB3 training
loop. This is genuine on-policy PPO against the retail executable. It is a
[targeted touch-up tool](de-realgame.md), not the main training vehicle — the sim
is ~1000× faster.

## Throughput

`torch.compile`, plus removing a redundant `_now()` call and fixing a
grid-march graph-break, took the sim from **115 k → 273 k env-frames/s**;
end-to-end training went ~40 k → ~250 k frames/s. Frame-skip is **1** in both
training and deployment so the two match exactly.
