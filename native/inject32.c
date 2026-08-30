// 32-bit injector helper. Same bitness as th07.exe so the remote LoadLibraryA
// address is valid.
//
//   inject32.exe <th07.exe path> <th07hook.dll path>
//
// Launches suspended, patches out the single-instance guard, resumes, then
// injects during the window before the game reaches its first do_tick. Prints
// the pid on stdout on success.

#include <windows.h>
#include <stdio.h>

// th07.exe v1.00b WinMain single-instance guard: a 2nd concurrent instance gets
// a modal error box + abort. Flip the jnz (75) after `cmp eax,ERROR_ALREADY_EXISTS`
// to an unconditional jmp (EB) so N instances can run under SubprocVecEnv.
// Must be applied while suspended, before the main thread reaches it.
// (mirrors th07::INSTANCE_GUARD_JNZ in th07_addrs.h)
#define GUARD_JNZ_VA 0x00435BFF

static int patch_instance_guard(HANDLE proc) {
    void* addr = (void*)GUARD_JNZ_VA;
    unsigned char cur = 0;
    SIZE_T n = 0;
    if (!ReadProcessMemory(proc, addr, &cur, 1, &n) || n != 1) {
        fprintf(stderr, "guard: ReadProcessMemory failed: %lu\n", GetLastError());
        return 1;
    }
    if (cur == 0xEB) return 0;              // already patched
    if (cur != 0x75) {
        fprintf(stderr, "guard: unexpected byte 0x%02x at %p (wrong th07.exe "
                        "version?) - not patching\n", cur, addr);
        return 1;
    }
    DWORD oldprot = 0;
    if (!VirtualProtectEx(proc, addr, 1, PAGE_EXECUTE_READWRITE, &oldprot)) {
        fprintf(stderr, "guard: VirtualProtectEx failed: %lu\n", GetLastError());
        return 1;
    }
    unsigned char jmp = 0xEB;
    int ok = WriteProcessMemory(proc, addr, &jmp, 1, &n) && n == 1;
    DWORD tmp;
    VirtualProtectEx(proc, addr, 1, oldprot, &tmp);
    if (!ok) {
        fprintf(stderr, "guard: WriteProcessMemory failed: %lu\n", GetLastError());
        return 1;
    }
    FlushInstructionCache(proc, addr, 1);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: inject32 <exe> <dll>\n"); return 2; }

    char exe[MAX_PATH], dll[MAX_PATH], dir[MAX_PATH];
    if (!GetFullPathNameA(argv[1], MAX_PATH, exe, NULL) ||
        !GetFullPathNameA(argv[2], MAX_PATH, dll, NULL)) {
        fprintf(stderr, "GetFullPathNameA failed: %lu\n", GetLastError());
        return 1;
    }
    strcpy(dir, exe);
    char* slash = strrchr(dir, '\\');
    if (slash) *slash = 0;

    STARTUPINFOA si = { sizeof(si) };
    // launch WITHOUT stealing focus (TH07_SHOW=1 to override). Not minimised -
    // th07's WinMain skips do_tick when minimised, so the window must be shown;
    // SW_SHOWNOACTIVATE keeps it off the foreground so it doesn't force-tab the
    // user out.
    if (getenv("TH07_SHOW") == NULL) {
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_SHOWNOACTIVATE;    // 4
    }
    PROCESS_INFORMATION pi = { 0 };
    if (!CreateProcessA(exe, NULL, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL,
                        dir, &si, &pi)) {
        fprintf(stderr, "CreateProcessA failed: %lu\n", GetLastError());
        return 1;
    }

    // defeat the single-instance guard while the main thread is still parked
    if (getenv("TH07_NO_GUARD_PATCH") == NULL &&
        patch_instance_guard(pi.hProcess) != 0) {
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }

    SIZE_T n = strlen(dll) + 1;
    void* remote = VirtualAllocEx(pi.hProcess, NULL, n, MEM_COMMIT | MEM_RESERVE,
                                  PAGE_READWRITE);
    if (!remote) { fprintf(stderr, "VirtualAllocEx: %lu\n", GetLastError()); return 1; }
    SIZE_T wrote = 0;
    if (!WriteProcessMemory(pi.hProcess, remote, dll, n, &wrote) || wrote != n) {
        fprintf(stderr, "WriteProcessMemory: %lu\n", GetLastError());
        return 1;
    }

    FARPROC load_lib = GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                      "LoadLibraryA");

    ResumeThread(pi.hThread);

    DWORD code = 0;
    for (int i = 0; i < 80; ++i) {
        HANDLE th = CreateRemoteThread(pi.hProcess, NULL, 0,
                                       (LPTHREAD_START_ROUTINE)load_lib,
                                       remote, 0, NULL);
        if (!th) { fprintf(stderr, "CreateRemoteThread: %lu\n", GetLastError()); return 1; }
        WaitForSingleObject(th, 10000);
        GetExitCodeThread(th, &code);
        CloseHandle(th);
        if (code) {
            printf("%lu\n", pi.dwProcessId);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            return 0;
        }
        Sleep(50);
    }
    fprintf(stderr, "LoadLibraryA kept returning 0\n");
    TerminateProcess(pi.hProcess, 1);
    return 1;
}
