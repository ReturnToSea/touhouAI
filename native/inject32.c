// 32-bit injector helper. Same bitness as th07.exe so the remote LoadLibraryA
// address is valid.
//
//   inject32.exe <th07.exe path> <th07hook.dll path>
//
// Launches suspended, resumes, then injects during the window before the game
// reaches its first do_tick. Prints the pid on stdout on success.

#include <windows.h>
#include <stdio.h>

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
    PROCESS_INFORMATION pi = { 0 };
    if (!CreateProcessA(exe, NULL, NULL, NULL, FALSE, CREATE_SUSPENDED, NULL,
                        dir, &si, &pi)) {
        fprintf(stderr, "CreateProcessA failed: %lu\n", GetLastError());
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
