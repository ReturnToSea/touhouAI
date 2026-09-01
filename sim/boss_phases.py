"""Per-boss phase windows for FightSim's synthetic damage-phasing.

A "phase" is one HP bar in the real fight (a nonspell, or a spellcard). We can't
read the recorded boss HP (the recorder only logs x/y, and Letty's EM_BOSSES[0]
is null for most of the fight), so per phase we synthesise an HP pool and let
the agent drain it by shooting.

`phase_windows(name, nb)` takes the per-frame active-bullet count of ONE
recording and returns, per phase:

    (clear_start, first_attack, phase_end)   -- all frame indices into that
                                                recording (post lead-in trim)

  clear_start   frame the previous attack's bullets have cleared -> where we
                jump the recording to when the agent damage-phases INTO this
                phase. For phase 0 this is 0.
  first_attack  frame this phase's first bullet appears. Between clear_start and
                first_attack the boss is repositioning / declaring the card:
                she deals no damage and TAKES no damage (armored).
  phase_end     frame this phase's attack ends (the next screen-clear starts).
                Draining the phase HP OR reaching this frame advances.

Boundaries are found from real screen-clears (bullet count collapsing to ~0),
snapped per-recording near the boss's known phase times, so small timer jitter
between recordings is handled. Nonspells have short internal wave-gaps that are
deliberately NOT treated as phase boundaries - the whole nonspell is one HP bar.
"""

import numpy as np

# Kill fraction: continuous fully-lined-up fire clears a phase in this fraction
# of its recorded attack duration. One number for all bosses - tune against real
# damage-phased fight lengths, don't chase exact ECL HP.
KILL_FRAC = 0.45

# approximate phase-boundary frames (60fps). phase_windows snaps each to the
# real screen-clear within +-500 frames of these, per recording.
_BOUNDARIES = {
    "letty": [2400, 5450, 7820],                 # NS1|LingeringCold|NS2|Spell2
    "chen":  [2450, 6050, 8450, 11550],          # rough, not screen-clear-verified
    # cirno / chenmid: single phase, no boundaries
}

_LOW = 15.0        # smoothed bullet count below this = "cleared"
_SMOOTH = 15

# Phases (0-indexed into phase_windows output) that are AIMED attacks in the ECL
# - `bullet_fan_aimed` etc. FightSim re-aims EVERY bullet born in these phases at
# the live policy (not just the ~5% the geometric heuristic catches), so the
# phase genuinely tracks the player and can't be memorised.
#   letty: 1 = "Cold Sign - Lingering Cold" (bullet_fan_aimed from the orbs)
# DISABLED for now: re-aiming a slow, continuously-fired pattern conflicts with
# the replay/slot architecture (re-aimed bullets outlive or underlive their
# recorded slot -> visible despawns). Kept for reference; needs a proper
# generative (ECL VM) sim to do this right.
_AIMED_PHASES = {
    # "letty": {1},
}


def aimed_phases(name):
    """set of phase indices that are fully-aimed attacks for this boss."""
    return _AIMED_PHASES.get(_key(name), set())


def _smooth(nb):
    return np.convolve(nb.astype(float), np.ones(_SMOOTH) / _SMOOTH, mode="same")


def _first_attack(sm, frm=0):
    """first frame at/after `frm` where bullets are actually flying."""
    a = np.argmax(sm[frm:] > 30.0)
    return int(frm + a) if sm[frm:].size and (sm[frm:] > 30.0).any() else frm


def _longest_lull(sm, lo, hi):
    """(start, end) of the longest run of sm < _LOW within [lo, hi), or None."""
    best = None
    i = lo
    while i < hi:
        if sm[i] < _LOW:
            j = i
            while j < hi and sm[j] < _LOW:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def _key(name):
    for k in ("letty", "chenmid", "cirno", "chen"):
        if name.startswith(k):
            return k
    return None


def has_phases(name):
    return _key(name) is not None


def phase_windows(name, nb):
    """-> list of (clear_start, first_attack, phase_end) frame triples, or None."""
    key = _key(name)
    if key is None:
        return None
    sm = _smooth(nb)
    F = len(nb)
    approx = _BOUNDARIES.get(key)
    if not approx:                                   # single-phase boss
        return [(0, _first_attack(sm), F)]

    cuts = []
    for a in approx:
        lull = _longest_lull(sm, max(0, a - 500), min(F, a + 500))
        if lull is not None:
            cuts.append(lull)
    cuts.sort()

    starts = [0] + [c[0] for c in cuts]
    firsts = [_first_attack(sm)] + [c[1] for c in cuts]
    ends = [c[0] for c in cuts] + [F]
    return list(zip(starts, firsts, ends))
