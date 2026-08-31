"""Faithful per-frame bullet motion, ported from PyTouhou's `game/bullet.pyx`
`Bullet.update()` (a tested reimplementation of the TH06 engine; TH07 shares the
"old" bullet model). Handles the launch crawl, the initial speed burst,
acceleration (flag 16), accel+rotate (flag 32), and the redirect/hang cycle
(flags 64/128/256 - decelerate to 0 between intervals, then re-aim/turn/snap).

A bullet is a plain list for speed (numpy-friendly later):
    [x, y, angle, speed, frame, flags, attrs, phase, interp]
`attrs` = (interval, count, p1, p2) from bullet_effects; `flags` is the 9th
spawn param OR'd with each bullet_effects flag. `phase` indexes multi-slot
effects (TH07 chains them); `interp` = (f0, f1, s0, s1) speed ramp or None.

    from ecl_bullet import spawn, advance
    b = spawn(x, y, angle, speed, spawn_flags, fx_list)
    advance(b, target_xy)     # one frame; mutates b
"""
from __future__ import annotations

import math

TAU = 2 * math.pi

# bullet-type launch penalty (PyTouhou launch_anim_penalties) - the "crawl"
# multiplier while the launch animation plays, and how many frames it lasts.
LAUNCH_MULT = 0.55
LAUNCH_FRAMES = 12


def spawn(x, y, angle, speed, flags, fx):
    """fx: tuple of (slot, effect_flag, interval, count, p1, p2) from ecl_vm."""
    f = int(flags)
    # merge the effect flags into the bullet's flag word + collect attr slots
    slots = []
    for (slot, eff, interval, count, p1, p2) in fx:
        f |= eff
        slots.append((eff, int(interval), int(count), float(p1), float(p2)))
    launching = LAUNCH_FRAMES if (f & 14) else 0
    interp = None
    if f & 1:                              # initial speed burst -> settle
        interp = (0, 16, speed + 5.0, speed)
    return [float(x), float(y), float(angle), float(speed), 0, f,
            slots, 0, interp, launching]


def _lerp(interp, frame):
    f0, f1, s0, s1 = interp
    if frame >= f1:
        return s1
    return s0 + (s1 - s0) * (frame - f0) / max(1, f1 - f0)


def advance(b, target):
    x, y, angle, speed, frame, flags, slots, phase, interp, launching = b

    if launching > 0:                      # launch crawl: reduced speed, no fx
        dx = math.cos(angle) * speed * LAUNCH_MULT
        dy = math.sin(angle) * speed * LAUNCH_MULT
        b[0], b[1] = x + dx, y + dy
        b[4] += 1
        b[9] -= 1
        return

    eff = slots[phase] if phase < len(slots) else None
    if eff is not None:
        e_flag, interval, count, p1, p2 = eff

        if e_flag == 16 and frame < interval:
            # add a vector to velocity each frame (p2 < -900 -> along current angle)
            va = angle if p2 < -900.0 else p2
            dx = math.cos(angle) * speed + math.cos(va) * p1
            dy = math.sin(angle) * speed + math.sin(va) * p1
            speed = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            if frame + 1 >= interval:
                phase += 1

        elif e_flag == 32 and frame < interval:
            speed += p1
            angle += p2
            if frame + 1 >= interval:
                phase += 1

        elif e_flag in (64, 128, 256):
            # redirect cycle: decelerate to 0 over `interval`, then re-aim
            if interval > 0 and frame % interval == 0:
                if frame != 0:
                    if p2 > -900.0:
                        speed = p2
                    if e_flag == 64:
                        angle += p1
                    elif e_flag == 128:
                        angle = math.atan2(target[1] - y, target[0] - x) + p1
                    else:
                        angle = p1
                    count -= 1
                    b[6][phase] = (e_flag, interval, count, p1, p2)
                if count > 0:
                    interp = (frame, frame + interval - 1, speed, 0.0)  # the hang
                else:
                    phase += 1
                    interp = None

    s = _lerp(interp, frame) if interp else speed
    b[0] = x + math.cos(angle) * s
    b[1] = y + math.sin(angle) * s
    b[2] = angle
    b[3] = speed
    b[4] = frame + 1
    b[7] = phase
    b[8] = interp
