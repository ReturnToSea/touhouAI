"""Expand an ECL Spawn schedule (from ecl_vm) into individual bullets, and
integrate their motion (bullet_effects: accelerate / curve / delay-redirect).

`expand(spawns, aim)` -> list of `B` records:
    B = (birth_frame, x0, y0, angle0, speed0, aimed, sprite, fx)
`step_bullets(recs, age)` -> Nx2 positions at that age (numpy), applying fx.

For `aimed` records angle0 already includes the toward-player term (viz uses
aim_ref; the sim passes its per-episode aim).
"""
from __future__ import annotations

import math

import numpy as np

TAU = 2 * math.pi


def expand(spawns, aim=None):
    out = []
    for s in spawns:
        base = s.base_angle
        if s.aimed:
            base = base + (aim(s) if aim else s.aim_ref)
        n = max(1, s.count)
        rings = max(1, s.shots)
        op = s.opcode
        for r in range(rings):
            spd = (s.speed if rings == 1
                   else s.speed + (s.speed2 - s.speed) * r / (rings - 1))
            ring_rot = base + s.spread * r
            if "circle" in op:
                angs = [ring_rot + TAU * i / n for i in range(n)]
            elif "fan" in op:
                angs = [ring_rot + s.spread * (i - (n - 1) / 2) for i in range(n)]
            elif "random" in op:
                lo, hi = sorted((s.base_angle, s.spread))
                rng = np.random.default_rng(s.frame * 2654435761 % 2**32 ^ (r << 8))
                aim0 = (aim(s) if aim else s.aim_ref) if s.aimed else 0.0
                angs = [aim0 + rng.uniform(lo, hi) for _ in range(n)]
            else:
                angs = [ring_rot]
            for a in angs:
                out.append((s.frame, s.x, s.y, a, spd, s.aimed, s.sprite, s.fx))
    return out


def step_bullets(recs, age):
    """Positions of all `recs` at `age` frames after their own birth (age is a
    scalar 'global frame'; each rec integrates from its own birth_frame)."""
    n = len(recs)
    xs = np.empty(n); ys = np.empty(n)
    for i, (bf, x0, y0, a0, s0, aimed, sprite, fx) in enumerate(recs):
        t = age - bf
        if t <= 0:
            xs[i], ys[i] = x0, y0
            continue
        # analytic-ish integration frame by frame is exact but slow; do a
        # closed-form per effect where possible, else short loop.
        x, y, a, s = x0, y0, a0, s0
        # fast path: no effects -> straight line
        if not fx:
            xs[i] = x0 + s0 * math.cos(a0) * t
            ys[i] = y0 + s0 * math.sin(a0) * t
            continue
        steps = int(t)
        for f in range(steps):
            for (mode, flag, dur, C2, p1, p2) in fx:
                if flag == 16 and (dur <= 0 or f < dur):
                    s += p1
                elif flag == 32 and (dur <= 0 or f < dur):
                    a += p2
                elif flag == 64 and f == dur:
                    a = a + p1
                    if p2 > -900:
                        s = p2
            x += s * math.cos(a)
            y += s * math.sin(a)
        # sub-frame remainder
        frac = t - steps
        xs[i] = x + s * math.cos(a) * frac
        ys[i] = y + s * math.sin(a) * frac
    return np.column_stack([xs, ys])
