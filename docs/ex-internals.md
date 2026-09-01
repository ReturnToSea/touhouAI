# Engine internals

Reference material for `th07.exe` v1.00b — a 32-bit DirectX 8 binary from 2003
with no meaningful ASLR, so every manager and struct field sits at a fixed
address. Most of it was found by probing: correlating struct floats against
observed per-frame motion, or scanning memory for values already known from the
scripts.

<div class="grid cards" markdown>

- __[Reverse-engineering th07](re.md)__

    ---

    The narrative version — the five managers, the `zBullet` and `zEnemy`
    layouts, why the bullet-effects state sitting *in the struct* is what made
    [recording](recording.md) work, and the probe method behind each finding.

- __[Address reference](reference.md)__

    ---

    The flat table — every static, hooked function, and struct offset RE'd so
    far. Mirrors `native/th07_addrs.h`.

</div>
