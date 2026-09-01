# Observation normalization

**Verdict:** catastrophic. `ppo_v28`. Real transfer collapsed to ~18 s median
(from v27's ~223 s) with a perfectly healthy sim curve. Reverted in `65a4d78`.

## The idea

Standardising each input feature to zero mean / unit variance is textbook. So
`ppo_v28` collected running mean/std for all 236 [obs](obs.md) features from the
sim and divided.

## Why it exploded

Some obs features are **constant in the sim**:

- empty item slots (the sim doesn't model items)
- wall-distance scalars in configurations the sim never produces
- a sim-only zero the real DLL fills with real data

A constant feature has std ≈ 0. Dividing by it folds that feature's input weights
up by `~1e4×`. In the sim it doesn't matter — the feature is always the same
value, so `weight × 1e4 × constant` is just a fixed bias the network absorbs.

On the **real** game those features are *not* constant. `weight × 1e4 ×
(now-varying value)` swamps every other input. The network's output is
noise the moment it sees a real observation.

## The lesson

> Normalisation statistics computed in the sim are a hidden channel for sim-only
> structure to leak into the weights. A feature that never varies in training but
> does in deployment is a landmine.

Removed entirely. Kept **reward** normalisation and LR annealing (those don't
touch the input path). `ppo_v29` — v27 + a batch of other refinements but *no*
obs norm — recovered to ~231 s.
