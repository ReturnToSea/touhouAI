# Build th07hook.dll (32-bit) with MSYS2 mingw32. Run from PowerShell.
#   powershell -ExecutionPolicy Bypass -File native\build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$mingw = if ($env:MINGW32) { $env:MINGW32 } else { "C:\msys64\mingw32\bin" }
$cc  = Join-Path $mingw "gcc.exe"
$cxx = Join-Path $mingw "g++.exe"
$mh  = "vendor\minhook"

if (-not (Test-Path $cc)) { throw "no 32-bit gcc at $cc" }
# mingw's cc1/as/collect2 load DLLs from their bin dir - it must be on PATH
$env:PATH = "$mingw;$env:PATH"
New-Item -ItemType Directory -Force -Path build | Out-Null

Write-Host "== MinHook =="
foreach ($f in "buffer", "hook", "trampoline") {
    & $cc -c -O2 -m32 -DMINHOOK_STATIC "-I$mh\include" "$mh\src\$f.c" -o "build\$f.o"
    if ($LASTEXITCODE) { throw "compile $f failed" }
}
& $cc -c -O2 -m32 "$mh\src\hde\hde32.c" -o "build\hde32.o"
if ($LASTEXITCODE) { throw "compile hde32 failed" }

Write-Host "== th07hook.dll =="
$cxxArgs = @(
    "-shared", "-O2", "-m32", "-std=c++17", "-Wall",
    "-static", "-static-libgcc", "-static-libstdc++",
    "-I$mh\include",
    "th07hook.cpp", "build\buffer.o", "build\hook.o", "build\trampoline.o", "build\hde32.o",
    "-o", "build\th07hook.dll",
    "-lkernel32", "-luser32", "-lwinmm"
)
& $cxx @cxxArgs
if ($LASTEXITCODE) { throw "link failed" }

Write-Host "== inject32.exe =="
& $cc -O2 -m32 -Wall inject32.c -o build\inject32.exe -lkernel32
if ($LASTEXITCODE) { throw "inject32 build failed" }

Get-Item build\th07hook.dll, build\inject32.exe | Format-List Name, Length, LastWriteTime
