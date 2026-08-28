#!/usr/bin/env bash
# Build th07hook.dll (32-bit) with MSYS2 mingw32.
set -euo pipefail
cd "$(dirname "$0")"

MINGW="${MINGW:-/c/msys64/mingw32/bin}"
CC="$MINGW/gcc.exe"
CXX="$MINGW/g++.exe"
MH=vendor/minhook

[ -x "$CC" ] || { echo "no 32-bit gcc at $CC (pacman -S mingw-w64-i686-gcc)"; exit 1; }

mkdir -p build
echo "== compiling MinHook =="
for f in buffer hook trampoline; do
    "$CC" -c -O2 -m32 -municode -DMINHOOK_STATIC -I"$MH/include" \
        "$MH/src/$f.c" -o "build/$f.o"
done
"$CC" -c -O2 -m32 -I"$MH/include" "$MH/src/hde/hde32.c" -o build/hde32.o

echo "== compiling + linking th07hook.dll =="
"$CXX" -shared -O2 -m32 -std=c++17 -Wall \
    -static -static-libgcc -static-libstdc++ \
    -I"$MH/include" \
    th07hook.cpp build/buffer.o build/hook.o build/trampoline.o build/hde32.o \
    -o build/th07hook.dll \
    -Wl,--kill-at \
    -lkernel32 -luser32

ls -l build/th07hook.dll
file build/th07hook.dll
