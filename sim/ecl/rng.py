"""The ECL PRNG.

`bullet_random*`, spread jitter and `__math_rand*` all draw from the engine's
linear congruential generator. This is the EoSD-family LCG
(`state = state*0x343FD + 0x269EC3`, take bits 16–30 → a 15-bit value).

Part 4 checks whether PCB uses exactly this stream, and — if not — either finds
the real one or matches the *distribution* empirically. For training only the
distribution matters, so Part 3 uses this generator as-is and treats the exact
constants as swappable.
"""
from __future__ import annotations


class EclRng:
    MUL = 0x343FD
    ADD = 0x269EC3
    _MASK = 0xFFFFFFFF

    def __init__(self, seed: int = 0):
        self.state = seed & self._MASK
        self.calls = 0

    def _next15(self) -> int:
        self.state = (self.state * self.MUL + self.ADD) & self._MASK
        self.calls += 1
        return (self.state >> 16) & 0x7FFF

    def rand_u16(self) -> int:
        return self._next15()

    def rand_u32(self) -> int:
        return (self._next15() << 15) | self._next15()

    def rand(self) -> float:
        """Uniform float in [0, 1)."""
        return self._next15() / 32768.0

    # PyTouhou-compatible alias
    rand_double = rand

    def rand_range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.rand()

    def rand_int(self, n: int) -> int:
        """Uniform int in [0, n)."""
        return int(self.rand() * n) if n else 0
