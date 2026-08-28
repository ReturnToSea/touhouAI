# Static analysis helpers

Throwaway scripts used to locate the frame limiter / vsync / mutex code in
`th07.exe` v1.00b without a debugger. Kept for reproducibility.

```
set TH07_EXE=C:\path\to\th07.exe        # or edit the default in each file

python disasm.py 0x4346e0:120            # disassemble Window::do_tick
python xref.py  0x575c3c 0x575a8b        # find code refs to a data address
```

Cross-reference names/structs come from
[`exphp-share/th-re-data`](https://github.com/exphp-share/th-re-data)
(`data/th07.v1.00b`: `funcs.json`, `labels.json`, `type-structs-own.json`).

Key addresses found (see `../README.md` "Gate B"):
`Window::do_tick` 0x4346E0, limiter `je`s at 0x4348CC / 0x434997,
`init_d3d_device` vsync `je` at 0x434C8A, uncap flag 0x575C3C,
render frameskip 0x575A8B, mutex at 0x435BE9.
