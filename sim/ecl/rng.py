"""The th07 danmaku PRNG.

`bullet_random*`, fan/ring spread jitter and the `__math_rand*` / `set_*_rand_*`
ECL opcodes all draw from the engine's generator. Reverse-engineered from
`th07.exe` (`FUN_00431870` / `FUN_004318d0` / `FUN_00431900`, see
`docs/th07-re-notes.md`):

    next16():                         # 16-bit state, all arithmetic mod 2**16
        u     = (state ^ 0x9630) + 0x9AAD
        state = rotl16(u, 2)          # ((u & 0xC000) >> 14) | (u << 2)
        return state

    rand_u32():   (next16() << 16) | (next16() & 0xFFFF)
    rand_float(): rand_u32() / 2**32

This is *not* the EoSD-family LCG (`state*0x343FD + 0x269EC3`) — that constant
appears in `th07.exe` only inside the unused MSVC `rand()`.
"""
from __future__ import annotations

_U16 = 0xFFFF
_U32 = 0xFFFFFFFF


class EclRng:
    def __init__(self, seed: int = 0):
        self.state = seed & _U16
        self.calls = 0

    def _next16(self) -> int:
        u = ((self.state ^ 0x9630) + 0x9AAD) & _U16
        self.state = ((u >> 14) | (u << 2)) & _U16     # rotl16(u, 2)
        self.calls += 1
        return self.state

    def rand_u16(self) -> int:
        return self._next16()

    def rand_u32(self) -> int:
        return ((self._next16() << 16) | self._next16()) & _U32

    def rand(self) -> float:
        """Uniform float in [0, 1) — `FUN_00431900`."""
        return self.rand_u32() / 4294967296.0

    # PyTouhou-compatible alias
    rand_double = rand

    def rand_range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.rand()

    def rand_int(self, n: int) -> int:
        """Uniform int in [0, n)."""
        return int(self.rand() * n) if n else 0
