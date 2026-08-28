// th07hook.dll - in-process control hook for Touhou 7 (PCB) v1.00b.
//
// Injected into a running th07.exe. Hooks:
//   * read_input (0x430B50) - FREE mode returns the real keyboard, STEP mode
//     returns the env's action.
//   * Window::do_tick (0x4346E0) - STEP mode advances exactly one call per
//     env step; IDLE spins; FREE runs normally so menus work.
// The 60fps limiter is NOP'd so do_tick doesn't busy-wait. The game only
// advances when the env asks, so the "Continue screen free-running" corruption
// cannot happen.
//
// Handshake + observation live in a shared mapping "th07hook_<pid>".

#include <windows.h>
#include <stdint.h>
#include <stdio.h>

#include "th07_addrs.h"
#include "th07_shm.h"
#include "vendor/minhook/include/MinHook.h"

using namespace th07;

typedef int(__fastcall* dotick_fn)(void* ecx, void* edx);      // __thiscall(this)
typedef uint16_t(__cdecl* readinput_fn)(void);
typedef int(__cdecl* present_fn)(void);

static dotick_fn    orig_do_tick    = nullptr;
static readinput_fn orig_read_input = nullptr;
static present_fn   orig_present    = nullptr;

static Shm*   g_shm = nullptr;
static HANDLE g_map = nullptr;

constexpr uintptr_t FRAMESKIP_BYTE = 0x00575A8B;

// limiter skip-branches (see feasibility/README.md) - NOP so do_tick never waits
static const struct { uintptr_t va; uint8_t want[2]; } LIMITER[] = {
    {0x004348CC, {0x74, 0x56}},
    {0x00434997, {0x74, 0x49}},
};

template <typename T> static inline T    rd(uintptr_t a)      { return *(T*)a; }
template <typename T> static inline void wr(uintptr_t a, T v) { *(T*)a = v; }

static void patch(uintptr_t a, const uint8_t* b, size_t n) {
    DWORD old;
    VirtualProtect((void*)a, n, PAGE_EXECUTE_READWRITE, &old);
    memcpy((void*)a, b, n);
    VirtualProtect((void*)a, n, old, &old);
    FlushInstructionCache(GetCurrentProcess(), (void*)a, n);
}

// ---- observation --------------------------------------------------------------
static void capture_obs() {
    Shm* s = g_shm;
    s->player_x = rd<float>(PLAYER + PL_POS_X);
    s->player_y = rd<float>(PLAYER + PL_POS_Y);
    s->player_state = rd<uint8_t>(PLAYER + PL_STATE);
    s->player_focus = rd<uint8_t>(PLAYER + PL_IS_FOCUS);
    s->gamemode = rd<int32_t>(SUPERVISOR + SV_GAMEMODE);
    s->stage = rd<int32_t>(GAME_MANAGER + GM_STAGE);
    s->difficulty = rd<int32_t>(GAME_MANAGER + GM_DIFFICULTY);
    s->cherry = rd<int32_t>(GAME_MANAGER + GM_CHERRY);
    s->cherry_max = rd<int32_t>(GAME_MANAGER + GM_CHERRY_MAX);

    uintptr_t g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
    if (g) {
        s->score = rd<int32_t>(g + G_SCORE);
        s->graze = rd<int32_t>(g + G_GRAZE);
        s->lives = rd<float>(g + G_LIFE_COUNT);
        s->bombs = rd<float>(g + G_BOMB_COUNT);
        s->power = rd<float>(g + G_POWER);
    }

    s->boss_present = rd<int32_t>(GUI + GUI_BOSS_PRESENT);
    s->boss_hp = rd<float>(GUI + GUI_BOSS_HP_CUR);
    s->boss_hp_max = rd<float>(GUI + GUI_BOSS_HP_MAX);

    int bn = 0;
    uintptr_t bb = BULLET_MANAGER + BM_BULLETS;
    for (uintptr_t i = 0; i < BM_BULLET_MAX && bn < MAX_BULLETS; ++i) {
        uintptr_t b = bb + i * BM_BULLET_STRIDE;
        uint16_t st = rd<uint16_t>(b + BULLET_STATE);
        if (st == 0 || st == 6) continue;
        float bx = rd<float>(b + BULLET_POS), by = rd<float>(b + BULLET_POS + 4);
        if (bx < -64 || bx > PLAYFIELD_W + 64 || by < -64 || by > PLAYFIELD_H + 64)
            continue;
        s->bullets[bn].x = bx; s->bullets[bn].y = by;
        s->bullets[bn].vx = 0; s->bullets[bn].vy = 0;
        ++bn;
    }
    s->bullet_count = bn;

    int en = 0;
    int32_t ec = rd<int32_t>(ENEMY_MANAGER + EM_ENEMY_COUNT);
    if (ec < 0 || ec > 480) ec = 0;
    for (int i = 0; i < ec && en < MAX_ENEMIES; ++i) {
        uintptr_t e = ENEMY_MANAGER + EM_ENEMIES + (uintptr_t)i * EM_ENEMY_STRIDE;
        s->enemies[en].x = rd<float>(e + ENEMY_POS);
        s->enemies[en].y = rd<float>(e + ENEMY_POS + 4);
        s->enemies[en].life = rd<int32_t>(e + ENEMY_LIFE);
        s->enemies[en].maxlife = rd<int32_t>(e + ENEMY_MAXLIFE);
        ++en;
    }
    s->enemy_count = en;
}

// ---- hooks ------------------------------------------------------------------
static uint16_t __cdecl hooked_read_input(void) {
    if (g_shm && g_shm->state == ST_STEP)
        return g_shm->action;
    return orig_read_input();
}

// Present is DWM-vsync-capped in a window. Skip it while stepping (rendering
// still runs, just no buffer flip) so logic isn't throttled to 144 Hz.
static int __cdecl hooked_present(void) {
    if (g_shm && g_shm->state == ST_STEP)
        return 0;
    return orig_present();
}

static int __fastcall hooked_do_tick(void* self, void* edx) {
    Shm* s = g_shm;
    s->alive = 1;

    switch (s->state) {
        case ST_FREE: {
            int r = orig_do_tick(self, edx);
            capture_obs();
            return r;
        }

        case ST_STEP: {
            uint16_t rep = s->repeat ? s->repeat : 1;
            int r = 0;
            for (uint16_t k = 0; k < rep; ++k) {
                wr<uint8_t>(FRAMESKIP_BYTE, 0);       // 1 logic frame per do_tick call
                r = orig_do_tick(self, edx);
                s->frame++;
                if (r != 0) break;                   // stage clear / quit / restart
            }
            s->tick_status = r;
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_RESET:
            s->done = 1;
            s->state = ST_IDLE;
            return 0;

        case ST_IDLE:
        default:
            Sleep(1);
            return 0;
    }
}

// ---- setup ----------------------------------------------------------------
static DWORD WINAPI init_thread(LPVOID) {
    char name[64];
    sprintf(name, "th07hook_%lu", GetCurrentProcessId());
    g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                               0, sizeof(Shm), name);
    if (!g_map) return 1;
    g_shm = (Shm*)MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shm));
    if (!g_shm) return 1;
    memset(g_shm, 0, sizeof(Shm));
    g_shm->magic = SHM_MAGIC;
    g_shm->version = SHM_VERSION;
    g_shm->state = ST_FREE;
    g_shm->repeat = 1;

    if (MH_Initialize() != MH_OK) return 1;
    if (MH_CreateHook((void*)FN_DO_TICK, (void*)&hooked_do_tick,
                      (void**)&orig_do_tick) != MH_OK) return 1;
    if (MH_CreateHook((void*)0x00430B50, (void*)&hooked_read_input,
                      (void**)&orig_read_input) != MH_OK) return 1;
    if (MH_CreateHook((void*)0x004345C0, (void*)&hooked_present,
                      (void**)&orig_present) != MH_OK) return 1;
    if (MH_EnableHook(MH_ALL_HOOKS) != MH_OK) return 1;

    static const uint8_t nop2[2] = {0x90, 0x90};
    for (auto& L : LIMITER)
        if (memcmp((void*)L.va, L.want, 2) == 0) patch(L.va, nop2, 2);

    return 0;
}

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(h);
        CreateThread(nullptr, 0, init_thread, nullptr, 0, nullptr);
    }
    return TRUE;
}
