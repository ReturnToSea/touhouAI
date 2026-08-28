"""Disassemble a VA range of th07.exe, annotating import calls.

    python disasm.py 0x4346e0:120 [0x434020:80 ...]

Set TH07_EXE to the game exe path, or edit the default below.
"""
import os, sys, pefile, capstone

EXE = os.environ.get(
    "TH07_EXE",
    r"C:\Users\spore\Documents\GitHub\touhouAI"
    r"\Touhou 7 - Perfect Cherry Blossom\th07.exe",
)
pe = pefile.PE(EXE, fast_load=True)
base = pe.OPTIONAL_HEADER.ImageBase
data = pe.__data__

def va2off(va):
    rva = va - base
    for s in pe.sections:
        if s.VirtualAddress <= rva < s.VirtualAddress + max(s.Misc_VirtualSize, s.SizeOfRawData):
            return s.PointerToRawData + (rva - s.VirtualAddress)
    return None

# import table: map thunk addr -> name
imports = {}
pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', []):
    dll = entry.dll.decode(errors='replace')
    for imp in entry.imports:
        nm = imp.name.decode(errors='replace') if imp.name else f"ord{imp.ordinal}"
        imports[imp.address] = f"{dll}!{nm}"

md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
md.detail = True

def disasm(va, n=80, note=""):
    off = va2off(va)
    print(f"\n==== {va:#x} ({note})  file@{off:#x} ====")
    code = data[off:off + n*8]
    cnt = 0
    for ins in md.disasm(code, va):
        tail = ""
        # annotate call/jmp targets against imports (via [addr]) and known funcs
        if ins.mnemonic in ("call", "jmp") and ins.op_str.startswith("dword ptr ["):
            try:
                a = int(ins.op_str.split("[")[1].split("]")[0], 16)
                if a in imports: tail = f"   ; {imports[a]}"
            except Exception: pass
        print(f"  {ins.address:#010x}: {ins.mnemonic:<7} {ins.op_str}{tail}")
        cnt += 1
        if cnt >= n: break

for arg in sys.argv[1:]:
    va, _, cnt = arg.partition(":")
    disasm(int(va, 16), int(cnt) if cnt else 80)
