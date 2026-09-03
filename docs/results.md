# Results

**Nothing conclusive yet.**

## Where transfer stands

| Approach | Best real transfer | Status |
|---|---|---|
| [Procedural sim](sim.md) → PPO | **~225 s** active median (`ppo_v27` / `ppo_v29`) — clears Stage 1, dies in Stage 2 | the standing baseline for a full run |
| [Recorded-Letty replay](de-letty-replay.md) → PPO | **~103 s / ~33 % kill** at the best checkpoint; 60 s / 7 % over 631 fights | [plateaued](ceiling.md) — sim score stopped predicting transfer |
| [ECL VM](de-generative-danmaku.md) → PPO | **~50 s / 0 % kill**, flat across every checkpoint | [dead end](de-generative-danmaku.md) — a reimplemented engine is still an artefact |
| [Real-game PPO](de-realgame.md), Stage 1 | 238 → 223 s greedy (flat) | Stage 1 was already at the sim ceiling — [a wash there](de-realgame.md) |

## What's running now

The Letty fight, trained on the [real game](de-realgame.md#the-letty-fight-revisited)
— a dozen [hooked instances](hook.md) collecting whole PPO trajectories in
parallel via `ST_ROLLOUT`, warm-started from `ppo_v29`. The bar it has to clear
is the replay baseline: **beat ~103 s active-fight median and ~33 % real
kill-rate on Letty**, without the bimodal checkpoint swing. When it does, the
numbers land here.

The [experiment log](experiment-log.md) has the run-by-run detail as it happens.

## Earlier results

The project has real results from approaches it has moved on from — the
[procedural-sim](sim.md) policies that reach Stage 2, and the
[recorded-Letty](recording.md) policies that landed the first boss kills. Those
live in the [Extras](dead-ends.md) section and the
[experiment log](experiment-log.md), with the analysis of
[why they weren't enough](ceiling.md).
