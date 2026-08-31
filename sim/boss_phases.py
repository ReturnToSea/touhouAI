"""Per-boss phase tables for FightSim's synthetic damage-phasing.

Each phase: (start_frame, end_frame, max_hp, kill_threshold, armored_frames)
  start/end  - frame offsets into a dodge-only recording (all recordings of a
               given boss are the same length, phases are timer-based)
  max_hp     - HP the phase starts at (from ECL enemy_life_set)
  threshold  - HP at which it transitions early (ECL life_callback); 0 = only
               the timer / running the bar to empty ends it
  armored    - frames at phase start where damage doesn't register
               (ECL enemy_flag_armored)

When the agent shoots while positioned under the boss, FightSim drains the
phase HP; on hp<=threshold OR frame>=end it clears bullets and jumps t to the
next phase's start_frame. Numbers are approximate - tune SHOT_DPS against real
damage-phased fight lengths.
"""

# spellcard HP where the ECL value wasn't pinned down - a placeholder that
# gives a sane damage-phase length at the default DPS.
_SPELL_HP = 2200

PHASES = {
    # Letty - 10750-frame recording, 4 phases
    "letty": [
        (0,    2500,  15000, 1700, 0),     # NS1
        (2600, 7700,  _SPELL_HP, 0, 300),  # Lingering Cold (aimed fans)
        (7800, 8700,  15000, 2000, 0),     # NS2
        (8700, 10750, _SPELL_HP, 0, 300),  # Table-Turning
    ],
    # Cirno midboss - 63s / 3780-frame recording, one nonspell
    "cirno": [
        (0, 3780, 8000, 0, 0),
    ],
    # Chen midboss - 39s / 2340-frame recording
    "chenmid": [
        (0, 2340, 14500, 2100, 120),
    ],
    # Chen boss - 233s / ~13980-frame recording. ECL: 17000 -> 1500,
    # 13000 -> 2900, plus spellcards. Rough 5-phase split.
    "chen": [
        (0,     2400,  17000, 1500, 0),
        (2500,  6000,  _SPELL_HP, 0, 240),
        (6100,  8400,  13000, 2900, 0),
        (8500,  11500, _SPELL_HP, 0, 240),
        (11600, 13980, _SPELL_HP, 0, 240),
    ],
}


def phases_for(name):
    """recording name (cirno_0, letty_3, ...) -> phase list, or None."""
    for key, ph in PHASES.items():
        if name.startswith(key):
            return ph
    return None


def total_hp(name):
    """total effective HP to drain to defeat this boss = sum over phases of
    (max_hp - kill_threshold). None if the boss isn't in the table."""
    ph = phases_for(name)
    if ph is None:
        return None
    return float(sum(mx - th for (_, _, mx, th, _) in ph))
