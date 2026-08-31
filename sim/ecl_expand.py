"""Expand an ECL Spawn schedule (from ecl_vm) into individual bullet spawn
records. Motion is handled per-frame by ecl_bullet.advance().

record = (birth_frame, x, y, angle0, speed0, aimed, flags, fx)
  angle0 for aimed records already includes the toward-player term.
  flags = the spawn's 9th param (launch-anim / effect-enable bits).
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
                rng = np.random.default_rng((s.frame * 2654435761) % 2**32 ^ (r << 8))
                aim0 = (aim(s) if aim else s.aim_ref) if s.aimed else 0.0
                angs = [aim0 + rng.uniform(lo, hi) for _ in range(n)]
            else:
                angs = [ring_rot]
            for a in angs:
                out.append((s.frame, s.x, s.y, a, spd, s.aimed, s.sprite, s.fx))
    out.sort(key=lambda r: r[0])
    return out
