"""Expand an ECL Spawn schedule (from ecl_vm) into individual bullets.

A `Spawn` describes a shot: `count` bullets per ring, `shots` concentric rings
at speeds lerp(speed -> speed2), each ring rotated by `spread`. This resolves it
to `[(birth_frame, x, y, angle, speed, aimed, sprite)]` - the replay primitive
shared by the visualiser and the sim.

For `aimed` spawns `angle` is an OFFSET from toward-player; the caller adds the
per-episode aim. Here (viz / reference) we add `aim_ref` (boss->ref-player).
"""
from __future__ import annotations

import math

TAU = 2 * math.pi


def expand(spawns, aim=None):
    """aim: fn(spawn)->float toward-player angle, or None to use spawn.aim_ref."""
    out = []
    for s in spawns:
        base = s.base_angle
        if s.aimed:
            base = base + (aim(s) if aim else s.aim_ref)
        n = max(1, s.count)
        rings = max(1, s.shots)
        op = s.opcode

        for r in range(rings):
            spd = s.speed if rings == 1 else s.speed + (s.speed2 - s.speed) * r / (rings - 1)
            ring_rot = base + s.spread * r

            if "circle" in op:
                for i in range(n):
                    a = ring_rot + TAU * i / n
                    out.append((s.frame, s.x, s.y, a, spd, s.aimed, s.sprite))
            elif "fan" in op:
                # n bullets, gap = spread, centred on ring_rot
                for i in range(n):
                    a = ring_rot + s.spread * (i - (n - 1) / 2)
                    out.append((s.frame, s.x, s.y, a, spd, s.aimed, s.sprite))
            elif "random" in op:
                lo, hi = min(s.base_angle, s.spread), max(s.base_angle, s.spread)
                # base_angle/spread here are the two angle-range params
                import random as _r
                rng = _r.Random(s.frame * 2654435761 ^ hash(op))
                for i in range(n):
                    a = rng.uniform(lo, hi)
                    if s.aimed:
                        a += (aim(s) if aim else s.aim_ref)
                    out.append((s.frame, s.x, s.y, a, spd, s.aimed, s.sprite))
            else:
                out.append((s.frame, s.x, s.y, ring_rot, spd, s.aimed, s.sprite))
    return out
