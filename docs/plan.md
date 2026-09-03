# Training on the real game

The plan for the last year was: build a fast, faithful simulator, train in it,
transfer the policy to the retail game. The "fast" happened. The "faithful"
never did — not with procedurally generated danmaku, not with 20 recordings, and
not with a near-complete reimplementation of the game's own engine. So the plan
has changed: **train on the real game, one segment at a time — Letty first, then
all of Stage 1, then Stage 2, and on — and accept that it is ~15× slower for the
accuracy of having no simulator at all.** The [roadmap](#the-roadmap) is at the
bottom; this is a work in progress.

## Why there was a simulator

[PPO](ppo.md) is sample-hungry. A policy that clears Stage 1 took on the order of
**200–1000 million environment frames**. The retail game, even
[hooked and stripped of its frame limiter](hook.md), runs one logic frame per
request at ~80× real-time — call it **5,000 frames/s** on one instance. 200 M
frames at that rate is **11 hours**; a billion is over two days, on a single
game that crashes if you look at it wrong.

The [GPU danmaku simulator](sim.md) was the answer: ~1000 bullet-hell episodes
in parallel on the GPU, **~250,000 frames/s**, the whole PPO loop staying on the
device. A run that would take days on the real game finished in
**under an hour**. Everything downstream — the [observation builder](obs.md),
[collision](collision.md), the PPO code, the [transfer daemon](de-letty-replay.md#the-transfer-daemon)
— was built around that simulator being the training environment and the real
game being only the *test*.

## Everything we tried to make the simulator faithful

The simulator was fast from the start. Making its danmaku match a *specific real
boss* — Letty, Stage 1 — was the whole problem, and it got attacked from three
directions:

| Approach | What it was | Where it landed |
|---|---|---|
| [Procedural generation](sim.md) | hand-tuned bullet emitters — fans, rings, spirals — with randomised parameters | policy reaches **~225 s** on the real game: clears Stage 1 by *outlasting* Letty, deals no real damage, dies in Stage 2 |
| [Recorded replay](de-letty-replay.md) | record ~20 real Letty fights frame-by-frame, replay the exact bullet positions on the GPU, invent a boss HP pool | **first real boss kills** — ~103 s / 33 % at the best checkpoint — then [plateaued](ceiling.md); 20 recordings and their mirrors aren't real Letty |
| [The ECL VM](de-generative-danmaku.md) | run Letty's *actual bytecode* frame-by-frame; get bullet motion by measuring the engine | ~2 px per-bullet fidelity, `danmaku_check` ratio 1.03 — and a **flat ~50 s / 0 % real kill-rate** across eight training runs |

Inside those, the sub-efforts:

- **Anti-memorisation for the replay sim** — x-mirroring, rigid ±10° field
  rotation, mid-phase random starts, ±50 % damage randomisation. Each widened
  the training distribution without distorting a trajectory. None of them add
  *new* danmaku.
- **Domain randomisation for the procedural sim** — the `ppo_v27` rewrite
  randomised bullet speeds, spawn timings, enemy density, colours. It reached
  Stage 2 and stalled there.
- **A full reverse-engineering pass on `th07.exe`** for the ECL VM — the real
  PRNG (`rotl16`, not the EoSD LCG), the emitter's 9 layer/rotation modes, the
  bullet-effects staging model, the spellcard `/7` damage divisor, the real HP
  thresholds, the orbit-movement math, the x87 bullet-motion update. All of it
  correct, all of it in [the RE notes](th07-re-notes.md) — and the VM still
  didn't transfer.
- **The re-aim saga** — three attempts to point replayed aimed bullets at the
  live policy instead of the recorded player. [All abandoned](de-reaim.md).
- **Ranking checkpoints by transfer tests, not sim score** — because the
  [sim score stopped predicting real performance](de-checkpoints.md).

## Why it didn't work

Every version of the simulator is a **fixed artefact**, and PPO is an
optimiser that will find any seam between that artefact and reality:

- Procedural danmaku has no real boss *structure* — no phases, no spellcard
  declarations, no satellite-orb choreography. The policy learns to survive
  *plausible* bullets, which is not the same as *Letty's* bullets.
- Replay is 20 trajectories. Every anti-memorisation trick is an affine
  transform of those 20. Real Letty's RNG produces genuinely different spreads,
  counts, and timings every run; no rotation of a fixed set covers that.
- The ECL VM was the strongest possible version of "faithful simulator" — and it
  proved the point in the other direction. A Python reimplementation of a 2003
  game cannot reproduce its **x87 80-bit floating-point** math bit-for-bit, and
  some of its constants live in runtime-initialised memory that static analysis
  can't read. Individually the errors are sub-pixel. Across a 500-bullet
  Lingering Cold screen they compound into safe lanes the real game doesn't
  have — and the policy stands in them.

The diagnostic was the same every time: **the sim eval kept improving while
real transfer stayed flat.** When an evaluation can't tell a policy that
transfers from one that doesn't, the policy is optimising against something that
only exists in the sim.

!!! quote "The ECL VM postmortem, in one line"
    Closer to real than 20 recordings, and it transferred *worse* than them. A
    reimplemented engine is still an artefact. — [the full writeup](de-generative-danmaku.md)

## The new plan: slower, accurate

Stop simulating Letty. Train the fight on **Letty's actual engine**.

- **`ST_ROLLOUT`** — the [hook](hook.md) collects a whole PPO trajectory (obs →
  actor → sampled action → tick → reward → record, hard-reset on death) *in C*,
  inside the running game, with no per-step Python. One instance runs at ~68×
  real-time.
- **A dozen instances in parallel**, each an independent hooked `th07.exe`.
  Aggregate throughput is **~5,000–6,000 decision-steps/s** — versus ~100 k/s in
  the GPU sim. Roughly **15–20× slower per step.**
- **Warm-started** from `ppo_v29`, the procedural-sim policy that already clears
  Stage 1 and outlasts Letty. The real-game run isn't learning to play from
  scratch — it's learning the one thing no simulator could teach it: how to
  **kill Letty fast**. The `boss-HP-drained × 3` reward term drives it.
- Every episode plays real Stage 1 into the real Letty fight; checkpoints go to
  the same [transfer daemon](de-letty-replay.md#the-transfer-daemon) as before —
  except now "sim performance" and "real performance" are the same number.

### The trade, explicitly

| | GPU simulator | Real-game rollout |
|---|---|---|
| throughput | ~100 k steps/s | ~5–6 k steps/s |
| a 200 M-step run | ~30 min | ~10 hours |
| danmaku fidelity | an approximation with a seam | it *is* the game |
| what a good sim score means | often nothing (the [core failure](ceiling.md)) | it's the real score |

We are buying **accuracy with wall-clock time.** A warm-started policy needs far
fewer than 200 M fresh steps to specialise on one fight, and the fight is ~90 s
long, so in practice the runs are hours, not days.

### Running it

```
.venv\Scripts\python native\run_letty_real.py
```

starts the trainer (`train_ppo_dll.py`, 12 hooked games), the greedy-eval daemon
(`fight_transfer_daemon.py --runsdir runs`, one more game), and the live overview
(`native/fight_dll_hud.py`) — two stacked charts:

- **survival** — the training curve (stochastic policy, on the real game) and the
  daemon's greedy median per checkpoint, against the 103 s baseline line;
- **beating Letty** — the daemon's greedy kill-rate and the training run's
  boss-HP-drained %, against the 33 % baseline line.

`history.npy` columns: `wall_s, total_steps, surv_s, entropy, value_expl,
boss_engaged_frac, boss_hp_floor_med, mean_return`.

### The simulator isn't gone

The [GPU sim](sim.md) still earns its place as the **pre-trainer** — it takes a
policy from random to "clears Stage 1, survives Letty" cheaply, and that policy
is the warm-start. The [ECL VM](de-generative-danmaku.md) code
(`sim/ecl/`) still parses and runs every stage's bytecode; it's the fastest way
to get a *rough* pattern for a boss we haven't recorded yet. What changed is
that the **final, transfer-critical training happens on the real game**, not in
any sim.

## The roadmap

!!! note "Work in progress"
    This is the plan being executed right now, not a finished result. Each step
    below is the same loop — hooked games, `ST_ROLLOUT`, PPO, the greedy-eval
    daemon — pointed one segment further into the game. Numbers land in
    [Results](results.md) as each step clears its bar.

The target is a full six-stage Lunatic 1cc. The path there is to extend the
real-game-trained policy one segment at a time, always warm-starting from the
last thing that worked.

| # | Segment | Bar to clear | Status |
|---|---|---|---|
| 1 | **Letty** — the Stage 1 boss, from the fight handoff | beat the [replay baseline](de-letty-replay.md): ~103 s active-fight median, ~33 % kill-rate, no bimodal checkpoint swing | **in progress** — first real-game run training now |
| 2 | **All of Stage 1** — stage portion + Cirno midboss + Letty, one episode | clear Stage 1 end-to-end, greedy, ≥90 % of runs, and *kill* Letty (not time her out) | queued behind #1 |
| 3 | **Stage 2** — full stage + midboss + Chen | reach and clear the Chen fight, greedy, on the real game | the procedural-sim policy already dies here — the first genuinely new ground |
| 4–7 | **Stages 3, 4, 5, 6** — one at a time | clear each stage's boss; Stages 5–6 add lasers (a separate bullet type the obs and collision don't model yet) | not started |
| 8 | **The 1cc** — all six, one credit | six stages, no continue, on vanilla `th07.exe` Lunatic | the goal |

Each step's policy is the next step's warm-start, so the run never restarts from
random — Stage 2 training begins from a policy that already clears Stage 1. The
GPU [procedural sim](sim.md) still does the cheap first pass (random → "survives
the stage") for any segment where it helps; the real-game loop does the part that
has to be exact.

### Known gaps to close along the way

- **Lasers** (Stages 5–6) are a distinct engine object the [observation
  builder](obs.md) and [collision](collision.md) don't handle. That work is
  deferred until Stage 5.
- **Episode length.** Right now every episode replays all of Stage 1 to reach
  Letty (~60 s of "already solved" game per episode). Later segments will want a
  [snapshot](env.md) taken deeper in, so a rollout starts at the stage being
  trained — the [hard-reset](hook.md) already reloads a stage; snapshotting
  mid-stage is the extension.
- **Resource management** (bombs, lives, power routing) isn't in the reward yet.
  A 1cc tolerates losing lives; a *good* 1cc doesn't. That reward shaping comes
  after the policy can physically clear the stages.
