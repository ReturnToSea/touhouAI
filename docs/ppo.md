# 10 · PPO

A compact PPO implementation — own GAE, clipped update, no SB3 in the sim path.
The policy is a two-hidden-layer MLP over the 236-d obs; the action space is 9
directions × focus × shoot.

!!! note "Draft"
    **To write.** The reward shaping. `ST_ROLLOUT` and the in-C reward that
    mirrors `env.py`. The `sb3_bridge` round-trip. The `torch.compile` speedup
    (115k → 273k env-frames/s; the redundant `_now()` call; why a bigger batch
    stopped helping). Frame-skip 1 and the train/deploy match.
