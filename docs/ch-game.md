# The game

*Perfect Cherry Blossom* (PCB, internally `th07.exe`, 2003) is the seventh
*Touhou* game and a **danmaku** — "bullet curtain" — shooter. You fly a small
character up a vertical playfield; enemies and bosses fill the screen with
slow, dense, geometric bullet patterns; you weave through the gaps. There is
almost no time pressure to kill things — the challenge is entirely *not getting
hit*.

## The playfield

The play area is **384 × 448 px** in game units. Your character moves freely
inside it. Enemies enter from the top and sides; bosses sit near the top centre
and stay there for a fixed sequence of attacks.

- **Hitbox.** Only a tiny point at your character's centre is lethal —
  measured [here](collision.md) at about **±1.8 px**. The sprite is ~32 px
  wide; almost all of it passes through bullets harmlessly. "Dodging" is really
  "keep one specific pixel out of the bullets."
- **Focus.** Holding Shift halves your movement speed and shows the hitbox as a
  dot. Focused movement is for threading tight patterns; unfocused is for
  crossing the screen. The agent controls this bit directly.
- **Grazing.** Passing a bullet within ~20 px without dying counts as a graze —
  a scoring mechanic, not a survival one.

## Lives, bombs, and what "clear" means

You start with a few lives and a few bombs. Getting hit costs a life (and some
resources); a bomb clears the screen and gives brief invulnerability but costs
score and a bomb. Run out of lives and you must **continue**, which resets your
score progress.

A **1CC — one-credit clear** — is finishing all six stages without continuing.
On **Lunatic**, the hardest difficulty, this is the benchmark of mastery. The
ideal is **no-miss no-bomb (NMNB)**: the full game without losing a life or
using a bomb.

## Why PCB

PCB is widely considered the most "perfectable" of the hard Windows *Touhou*
games: its patterns are readable and its timing is stable. Its signature
mechanic — the **cherry/border** system — is a *scoring* tool, not a survival
crutch, and nothing in PCB forces you to fight back on a timer.

That shapes the project: the hard part of a clear is almost entirely
**survival**, so dodging is where the effort goes. But a clear still isn't pure
dodging — you have to damage each boss enough to end its phases and manage power
and enemies through the stage sections. PCB's design just means the agent can
prioritise not dying and fit the shooting around it. See
[the goal](ch-goal.md) for how that translates into the objective.

## The bosses

Each stage ends with a boss who runs a scripted sequence of **non-spell**
attacks and named **spell cards**, separated by brief repositioning lulls and
screen-clears. Stage 1's boss, **Letty Whiterock**, has four phases and is the
current target for everything in this handbook. Her patterns and the bytecode
that drives them are covered in [The ECL format](ecl.md).
