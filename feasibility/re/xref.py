"""Find code refs to absolute data addresses in th07.exe's .text.

    python xref.py 0x575c3c 0x575a8b        (defaults to a few known flags)

Set TH07_EXE to the game exe path, or edit the default below.
"""
import os, sys, struct, pefile, capstone

EXE = os.environ.get(
    "TH07_EXE",
    r"C:\Users\spore\Documents\GitHub\touhouAI"
    r"\Touhou 7 - Perfect Cherry Blossom\th07.exe",
)
pe = pefile.PE(EXE, fast_load=True)
base = pe.OPTIONAL_HEADER.ImageBase
data = pe.__data__

text = None
for s in pe.sections:
    if b'.text' in s.Name:
        text = s
tv = base + text.VirtualAddress
tdata = data[text.PointerToRawData: text.PointerToRawData + text.SizeOfRawData]

targets = [int(x, 16) for x in sys.argv[1:]] or [0x575c3c, 0x575a8b, 0x575a8a]
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
md.detail = True

for tgt in targets:
    needle = struct.pack("<I", tgt)
    print(f"\n===== refs to {tgt:#x} =====")
    start = 0
    while True:
        i = tdata.find(needle, start)
        if i < 0:
            break
        start = i + 1
        va = tv + i
        # disassemble a window backward-ish: from 16 bytes before, print lines covering va
        b = tv + max(0, i - 24)
        for ins in md.disasm(tdata[max(0, i - 24): i + 12], b):
            if ins.address <= va < ins.address + ins.size:
                ops = ins.op_str
                print(f"  {ins.address:#010x}: {ins.mnemonic:<6} {ops}")
