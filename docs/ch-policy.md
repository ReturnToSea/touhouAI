# The policy network

The policy is a small **actor–critic multilayer perceptron (MLP)**. It maps the
236-number [observation](obs.md) to a probability over 36 actions (the actor)
and a scalar value estimate (the critic). Both are trained together with
[PPO](ppo.md).

## Architecture

```
obs (236) ─▶ Linear 236→256 ─▶ tanh ─▶ Linear 256→256 ─▶ tanh ─┬─▶ Linear 256→36   (action logits)
                                                               └─▶ Linear 256→1    (state value)
```

- **Two hidden layers of 256 units**, `tanh` activations.
- **Orthogonal weight initialisation**, small-gain on the policy head so the
  initial policy is near-uniform.
- Actor and critic are **separate heads on a shared trunk width** but, in
  practice, separate stacks (`[236,256,256,36]` and `[236,256,256,1]`) — cheap
  enough at this size that sharing buys nothing.
- **~200 k parameters.** A forward pass is a handful of small matmuls, which is
  what lets the [simulator](sim.md) run ~1000 environments per step without the
  policy becoming the bottleneck.

## Why an MLP and not a CNN

The agent never sees pixels. The [observation](obs.md) is already a
hand-engineered feature vector — player state, nine "escape" ray-casts, a local
danger grid, nearest-enemy and nearest-item blocks. The spatial reasoning a CNN
would have to learn from a screen buffer is **precomputed** into those features,
so a plain MLP over the vector is both sufficient and far cheaper to train and to
run. It also makes sim-and-real parity tractable: the DLL recomputes the exact
same 236 numbers in C, [checked byte-for-byte](hook.md) against the Python
builder.

## Why small

- **Sample efficiency & generalisation.** A big network fits sim-specific
  quirks faster than it learns transferable dodging — and
  [transfer is the whole problem](ceiling.md). A ~200 k-parameter model has less
  room to memorise.
- **Throughput.** PPO needs hundreds of millions of frames. Rollout speed scales
  with policy cost; keeping the net tiny keeps the GPU busy on the *environment*,
  not the forward pass.
- Widening to `[512,512]` or adding a third layer was tried in the
  [procedural-sim runs](experiment-log.md) and never improved real transfer.

## The action space — 36 discrete actions

The action is a single categorical draw factored as **9 × 2 × 2**:

| Component | Values | Meaning |
|---|---|---|
| direction | 9 | 8 compass directions + "stay" |
| focus | 2 | normal / focused (half-speed, precise) |
| shoot | 2 | hold shot / don't |

Flattened to one 36-way softmax rather than three independent heads — the
choices interact (focused + up-left is a different tactic from unfocused +
up-left) and a single categorical lets the policy represent that jointly.
Because ReimuA's shot [aims itself](ch-goal.md), "shoot" carries no aiming
decision; it is nearly free to leave on, and the reward — not the action space —
decides when engaging matters.

## Training vs deployment

- **Frame-skip 1.** The policy acts every logic frame, in the sim and on the
  real game, so the two match exactly.
- **Stochastic in training** (sample from the softmax, for exploration and for
  PPO's importance ratio), **greedy at deployment** (argmax).
- The critic head is used only during training, to compute
  [advantages](ppo.md); the deployed agent is just the actor.
