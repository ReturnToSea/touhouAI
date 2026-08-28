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
#include <math.h>
#include <mmdeviceapi.h>
#include <audiopolicy.h>

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

// do_tick's render block (0x43471A..0x43478D) can't be skipped wholesale - it
// also runs a Supervisor update that run_all_on_tick needs. But run_all_on_draw
// (0x42FE20, the sprite-draw chain walk) is separable: stubbing it while driving
// gives bit-identical game state (pos/score/deaths frame-for-frame) at ~3x the
// throughput - it's pure rendering. env var TH07_NO_DRAW=0 disables the stub.
static bool g_stub_draw = true;
// watch mode (env var TH07_RENDER=1): actually draw the sprites and flip the
// buffer so a human can see the policy play. Off for training.
static bool g_render = false;
// difficulty to force during menu nav (env var TH07_DIFFICULTY, default Lunatic)
static int32_t g_difficulty = 3;
typedef int(__fastcall* drawall_fn)(void* ecx, void* edx);
static drawall_fn orig_run_all_on_draw = nullptr;
static int __fastcall hooked_run_all_on_draw(void* ecx, void* edx) {
    if (g_stub_draw && g_shm && !(g_shm->state == ST_EVAL && g_shm->eval_render) &&
        !g_render &&
        (g_shm->state == ST_STEP || g_shm->state == ST_RESET ||
         g_shm->state == ST_EVAL))
        return 0;
    return orig_run_all_on_draw(ecx, edx);
}

// --- state snapshot for episode reset ---------------------------------------
// The whole .data/.bss section (0x49C000 .. 0x1365258). It holds every static
// game-state object - ANM_MANAGER/INPUT (0x4B9E44), PLAYER (0x4BDAD8),
// GAME_MANAGER (0x626270), BULLET_MANAGER (0x62F958), ENEMY_MANAGER (0x9A9B00),
// ECL_MANAGER (0x1347938), STAGE_NUM (0x1347FC8) - AND the accumulating counters
// past 0x1350000 (e.g. the frame counter at 0x135E1F8). An earlier, narrower
// span (0x4B0000..0x1350000) left ~86 KB unreset -> a counter there overflowed
// and crashed the game deterministically at ~77.8k steps / ~207 episodes.
constexpr uintptr_t SNAP_LO = 0x0049C000;   // .data section start
constexpr uintptr_t SNAP_HI = 0x01366000;   // .data end (0x1365258), page-rounded
constexpr size_t    SNAP_SZ = SNAP_HI - SNAP_LO;   // ~15.5 MB
constexpr size_t    GLOB_SZ = 0x100;               // zGlobals (heap block)
constexpr size_t    REC_SZ  = 0xE0;   // recorder head: fields seen up to 0xD6

static uint8_t* g_snap_static = nullptr;
static uint8_t  g_snap_glob[GLOB_SZ];
static uintptr_t g_snap_glob_ptr = 0;

// The replay input recorder (0x442CD0, a per-frame update-registry callback)
// also processes live input, so we can't NOP it. But its buffer write pointer
// (this+0x84) advances every frame and never wraps -> after ~233k frames it
// runs off the end and the game AVs. The object is heap-allocated. Hook the fn
// to grab `this`, then snapshot/restore its head each episode like zGlobals so
// the write pointer resets and the buffer never overruns.
static uint8_t  g_snap_rec[REC_SZ];
static void*    g_recorder = nullptr;
typedef int(__fastcall* rec_fn)(void* ecx, void* edx);
static rec_fn   orig_rec = nullptr;
static int __fastcall hooked_rec(void* ecx, void* edx) {
    g_recorder = ecx;
    return orig_rec(ecx, edx);
}


// limiter skip-branches (see feasibility/README.md) - NOP so do_tick never waits
static const struct { uintptr_t va; uint8_t want[2]; } LIMITER[] = {
    {0x004348CC, {0x74, 0x56}},
    {0x00434997, {0x74, 0x49}},
};

// NOTE: skipping do_tick's render block entirely (0x434718 jg -> jmp) breaks the
// game after ~150 frames - the block also runs run_all_on_draw + ANM updates
// that the tick logic depends on. Present-skip (below) is the safe win.

template <typename T> static inline T    rd(uintptr_t a)      { return *(T*)a; }
template <typename T> static inline void wr(uintptr_t a, T v) { *(T*)a = v; }

static void patch(uintptr_t a, const uint8_t* b, size_t n) {
    DWORD old;
    VirtualProtect((void*)a, n, PAGE_EXECUTE_READWRITE, &old);
    memcpy((void*)a, b, n);
    VirtualProtect((void*)a, n, old, &old);
    FlushInstructionCache(GetCurrentProcess(), (void*)a, n);
}

// ---- in-DLL policy: obs builder + MLP (mirror of native/env.py & policy.py) --
constexpr int K_BULLETS = 32;   // nearest bullets in the obs
constexpr int M_ENEMIES = 6;

static float g_prev_bx[MAX_BULLETS];
static float g_prev_by[MAX_BULLETS];
static void reset_bullet_hist() {
    for (int i = 0; i < MAX_BULLETS; ++i) { g_prev_bx[i] = -9999.f; g_prev_by[i] = -9999.f; }
}

// Build the 192-dim observation exactly as env.py `_obs()` does. Bullet
// velocity is the displacement since the previous build_obs() call (i.e. over
// one decision = frame_skip frames), matching the Python `_prev_bpos` diff.
static void build_obs(float* o) {
    const float W = PLAYFIELD_W, H = PLAYFIELD_H;
    float px = rd<float>(PLAYER + PL_POS_X), py = rd<float>(PLAYER + PL_POS_Y);
    uint8_t pst = rd<uint8_t>(PLAYER + PL_STATE);
    uint8_t focus = rd<uint8_t>(PLAYER + PL_IS_FOCUS);
    int32_t stage = rd<int32_t>(GAME_MANAGER + GM_STAGE);
    float lives = 0, bombs = 0, power = 0; int32_t graze = 0;
    uintptr_t g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
    if (g) {
        graze = rd<int32_t>(g + G_GRAZE);
        lives = rd<float>(g + G_LIFE_COUNT);
        bombs = rd<float>(g + G_BOMB_COUNT);
        power = rd<float>(g + G_POWER);
    }
    int boss_present = rd<int32_t>(GUI + GUI_BOSS_PRESENT);
    float bhp = rd<float>(GUI + GUI_BOSS_HP_CUR), bhpm = rd<float>(GUI + GUI_BOSS_HP_MAX);

    // head (12)
    o[0] = px / W; o[1] = py / H; o[2] = 0.f; o[3] = 0.f;   // player_vx/vy unset (== env.py)
    o[4] = (float)focus;
    o[5] = lives / 9.f; o[6] = bombs / 9.f; o[7] = power / 128.f;
    o[8] = tanhf(graze / 100.f);
    o[9] = stage / 6.f;
    o[10] = (pst == 0) ? 1.f : 0.f;
    o[11] = (pst == 2) ? 1.f : 0.f;

    // bullets: gather live, K nearest by distance, [rel/128, vel/10, dist/200]
    float bd[K_BULLETS];                       // sorted ascending distance
    float brx[K_BULLETS], bry[K_BULLETS], bvx[K_BULLETS], bvy[K_BULLETS];
    int nb = 0;
    uintptr_t bb = BULLET_MANAGER + BM_BULLETS;
    const int nslot = (int)BM_BULLET_MAX < MAX_BULLETS ? (int)BM_BULLET_MAX : MAX_BULLETS;
    for (int i = 0; i < nslot; ++i) {
        uintptr_t b = bb + (uintptr_t)i * BM_BULLET_STRIDE;
        uint16_t st = rd<uint16_t>(b + BULLET_STATE);
        float bx = rd<float>(b + BULLET_POS), by = rd<float>(b + BULLET_POS + 4);
        bool live = (st != 0 && st != 6 &&
                     bx > -64 && bx < W + 64 && by > -64 && by < H + 64);
        float vx = 0.f, vy = 0.f;
        if (live && g_prev_bx[i] > -9000.f) { vx = bx - g_prev_bx[i]; vy = by - g_prev_by[i]; }
        g_prev_bx[i] = live ? bx : -9999.f;
        g_prev_by[i] = live ? by : -9999.f;
        if (!live) continue;
        float rx = bx - px, ry = by - py;
        float d = sqrtf(rx * rx + ry * ry);
        if (nb < K_BULLETS) {                  // insertion sort into the K-buffer
            int j = nb++;
            while (j > 0 && bd[j - 1] > d) {
                bd[j] = bd[j - 1]; brx[j] = brx[j - 1]; bry[j] = bry[j - 1];
                bvx[j] = bvx[j - 1]; bvy[j] = bvy[j - 1]; --j;
            }
            bd[j] = d; brx[j] = rx; bry[j] = ry; bvx[j] = vx; bvy[j] = vy;
        } else if (d < bd[K_BULLETS - 1]) {
            int j = K_BULLETS - 1;
            while (j > 0 && bd[j - 1] > d) {
                bd[j] = bd[j - 1]; brx[j] = brx[j - 1]; bry[j] = bry[j - 1];
                bvx[j] = bvx[j - 1]; bvy[j] = bvy[j - 1]; --j;
            }
            bd[j] = d; brx[j] = rx; bry[j] = ry; bvx[j] = vx; bvy[j] = vy;
        }
    }
    float* ob = o + 12;
    for (int i = 0; i < K_BULLETS; ++i) {
        if (i < nb) {
            ob[i * 5 + 0] = brx[i] / 128.f; ob[i * 5 + 1] = bry[i] / 128.f;
            ob[i * 5 + 2] = bvx[i] / 10.f;  ob[i * 5 + 3] = bvy[i] / 10.f;
            ob[i * 5 + 4] = bd[i] / 200.f;
        } else {
            ob[i * 5 + 0] = ob[i * 5 + 1] = ob[i * 5 + 2] = ob[i * 5 + 3] = ob[i * 5 + 4] = 0.f;
        }
    }

    // enemies (6 * 3): [rel/128, life/maxlife]
    float* oe = o + 12 + K_BULLETS * 5;
    for (int i = 0; i < M_ENEMIES; ++i) { oe[i * 3] = oe[i * 3 + 1] = oe[i * 3 + 2] = 0.f; }
    int32_t ec = rd<int32_t>(ENEMY_MANAGER + EM_ENEMY_COUNT);
    if (ec < 0 || ec > 480) ec = 0;
    for (int i = 0; i < ec && i < M_ENEMIES; ++i) {
        uintptr_t e = ENEMY_MANAGER + EM_ENEMIES + (uintptr_t)i * EM_ENEMY_STRIDE;
        float ex = rd<float>(e + ENEMY_POS), ey = rd<float>(e + ENEMY_POS + 4);
        int32_t life = rd<int32_t>(e + ENEMY_LIFE), ml = rd<int32_t>(e + ENEMY_MAXLIFE);
        if (ml < 1) ml = 1;
        oe[i * 3 + 0] = (ex - px) / 128.f;
        oe[i * 3 + 1] = (ey - py) / 128.f;
        oe[i * 3 + 2] = (float)life / (float)ml;
    }

    // boss (2)
    float* obo = o + 12 + K_BULLETS * 5 + M_ENEMIES * 3;
    obo[0] = (float)(boss_present != 0);
    obo[1] = (bhpm > 0.f) ? (bhp / bhpm) : 0.f;
}

// MLP: Linear(192,h1)-Tanh-Linear(h1,h2)-Tanh-Linear(h2,36) -> argmax.
// Flat layout matches policy.py get_flat(): W0,b0,W1,b1,W2,b2 (torch Linear
// weight is [out,in] row-major).
static int mlp_forward(const float* w, const float* in, int h1, int h2) {
    float a[MAX_HIDDEN], b[MAX_HIDDEN];
    const float* p = w;
    for (int j = 0; j < h1; ++j) {
        float acc = 0.f; const float* row = p + (size_t)j * OBS_DIM;
        for (int k = 0; k < OBS_DIM; ++k) acc += row[k] * in[k];
        a[j] = acc;
    }
    p += (size_t)h1 * OBS_DIM;
    for (int j = 0; j < h1; ++j) a[j] = tanhf(a[j] + p[j]);
    p += h1;
    for (int j = 0; j < h2; ++j) {
        float acc = 0.f; const float* row = p + (size_t)j * h1;
        for (int k = 0; k < h1; ++k) acc += row[k] * a[k];
        b[j] = acc;
    }
    p += (size_t)h2 * h1;
    for (int j = 0; j < h2; ++j) b[j] = tanhf(b[j] + p[j]);
    p += h2;
    int best = 0; float bestv = -1e30f;
    const float* bias = p + (size_t)N_ACTIONS * h2;
    for (int j = 0; j < N_ACTIONS; ++j) {
        float acc = 0.f; const float* row = p + (size_t)j * h2;
        for (int k = 0; k < h2; ++k) acc += row[k] * b[k];
        acc += bias[j];
        if (acc > bestv) { bestv = acc; best = j; }
    }
    return best;
}

// action index -> button bitmask (mirror of env.py _decode_action + _DIRS)
static const int8_t A_DIRS[9][2] = {
    {0,0},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}
};
static uint16_t decode_action(int a) {
    int d = a % 9, focus = (a / 9) % 2, shoot = (a / 18) % 2;
    int dx = A_DIRS[d][0], dy = A_DIRS[d][1];
    uint16_t bits = 0;
    if (dx < 0) bits |= BTN_LEFT;
    if (dx > 0) bits |= BTN_RIGHT;
    if (dy < 0) bits |= BTN_UP;
    if (dy > 0) bits |= BTN_DOWN;
    if (focus) bits |= BTN_SLOW;
    if (shoot) bits |= BTN_SHOOT;
    return bits;
}

static void restore_snapshot() {
    memcpy((void*)SNAP_LO, g_snap_static, SNAP_SZ);
    if (g_snap_glob_ptr)
        memcpy((void*)g_snap_glob_ptr, g_snap_glob, GLOB_SZ);
    if (g_recorder)
        memcpy(g_recorder, g_snap_rec, REC_SZ);
    g_shm->frame = 0;
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

    // slot-indexed: bullets[i] is pool slot i (inactive -> x = INACTIVE).
    // Velocity is derived on the Python side by diffing a slot across frames.
    int active = 0;
    uintptr_t bb = BULLET_MANAGER + BM_BULLETS;
    const int nslot = (int)BM_BULLET_MAX < MAX_BULLETS ? (int)BM_BULLET_MAX : MAX_BULLETS;
    for (int i = 0; i < nslot; ++i) {
        uintptr_t b = bb + (uintptr_t)i * BM_BULLET_STRIDE;
        uint16_t st = rd<uint16_t>(b + BULLET_STATE);
        float bx = rd<float>(b + BULLET_POS), by = rd<float>(b + BULLET_POS + 4);
        bool live = (st != 0 && st != 6 &&
                     bx > -64 && bx < PLAYFIELD_W + 64 &&
                     by > -64 && by < PLAYFIELD_H + 64);
        s->bullets[i].x = live ? bx : INACTIVE;
        s->bullets[i].y = live ? by : INACTIVE;
        s->bullets[i].vx = (float)st;   // slot state (debug / classification)
        s->bullets[i].vy = 0;
        active += live;
    }
    s->bullet_count = active;

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
static inline bool driving() {
    return g_shm && (g_shm->state == ST_STEP || g_shm->state == ST_AUTONAV ||
                     g_shm->state == ST_EVAL);
}

static uint16_t __cdecl hooked_read_input(void) {
    if (driving())
        return g_shm->action;
    return orig_read_input();
}

// Present is DWM-vsync-capped in a window. Skip it while we drive the game
// (rendering still runs, just no buffer flip) so logic isn't throttled to 144 Hz.
static int __cdecl hooked_present(void) {
    bool want = g_render || (g_shm && g_shm->state == ST_EVAL && g_shm->eval_render);
    if (driving() && !want)
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
                wr<uint8_t>(FRAMESKIP_BYTE, 0);   // 1 logic tick per do_tick call
                r = orig_do_tick(self, edx);
                s->frame++;
                if (r != 0) break;               // stage clear / quit
            }
            s->tick_status = r;
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_SNAPSHOT: {
            memcpy(g_snap_static, (void*)SNAP_LO, SNAP_SZ);
            g_snap_glob_ptr = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
            if (g_snap_glob_ptr)
                memcpy(g_snap_glob, (void*)g_snap_glob_ptr, GLOB_SZ);
            if (g_recorder)
                memcpy(g_snap_rec, g_recorder, REC_SZ);
            s->have_snapshot = 1;
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_AUTONAV: {
            // Tap Shoot through: title -> Start -> difficulty -> character ->
            // shot type -> Stage 1. The difficulty screen inits its cursor from
            // [DIFFICULTY_SEL] and gameplay reads the same global, so pinning it
            // each frame makes the Shoot-mash confirm our chosen difficulty.
            int nav = 0;
            const int MAXF = 6000;
            for (; nav < MAXF; ++nav) {
                int gm  = rd<int32_t>(SUPERVISOR + SV_GAMEMODE);
                int stg = rd<int32_t>(GAME_MANAGER + GM_STAGE);
                if (gm == 2 && stg >= 1) break;
                wr<int32_t>(DIFFICULTY_SEL, g_difficulty);
                // ~4 frames pressed, ~10 released -> a clean tap the menus accept
                s->action = ((nav % 14) < 4) ? (uint16_t)BTN_SHOOT : (uint16_t)0;
                wr<uint8_t>(FRAMESKIP_BYTE, 0);
                if (orig_do_tick(self, edx) != 0) break;
            }
            s->action = 0;
            s->nav_frames = (nav >= MAXF) ? -1 : nav;
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_RESET: {
            if (s->have_snapshot) restore_snapshot();
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_EVAL: {
            // one whole episode with the in-DLL MLP policy - no per-frame
            // Python round trip. reset -> (obs -> mlp -> action -> tick)* .
            if (!s->have_snapshot) { s->done = 1; s->state = ST_IDLE; return 0; }
            restore_snapshot();
            reset_bullet_hist();
            const int h1 = (int)s->eval_h1, h2 = (int)s->eval_h2;
            const uint32_t fs = s->eval_frame_skip ? s->eval_frame_skip : 3;
            const uint32_t cap = s->eval_max_frames ? s->eval_max_frames : 7200;
            const bool paced = s->eval_render != 0;

            uintptr_t gp = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
            float start_lives = gp ? rd<float>(gp + G_LIFE_COUNT) : 0.f;
            LARGE_INTEGER qpf, t_last; QueryPerformanceFrequency(&qpf);
            QueryPerformanceCounter(&t_last);

            float obs[OBS_DIM];
            uint32_t frames = 0;
            int r = 0;
            bool died = false;
            bool first = true;
            float boss_dmg = 0.f;
            float prev_bhp = 0.f;
            bool prev_boss = false;
            while (frames < cap) {
                build_obs(obs);
                if (first) { memcpy(s->dbg_obs, obs, sizeof(obs)); first = false; }
                s->action = decode_action(mlp_forward(s->weights, obs, h1, h2));
                for (uint32_t k = 0; k < fs && frames < cap; ++k) {
                    wr<uint8_t>(FRAMESKIP_BYTE, 0);
                    r = orig_do_tick(self, edx);
                    s->frame++; frames++;
                    if (r != 0) break;
                }
                // boss damage: sum per-decision HP drops, normalised by hp_max
                // (spell-card refills -> negative delta, ignored). Only count
                // when the boss was present on both this and the previous check.
                bool boss = rd<int32_t>(GUI + GUI_BOSS_PRESENT) != 0;
                float bhp = rd<float>(GUI + GUI_BOSS_HP_CUR);
                float bhpm = rd<float>(GUI + GUI_BOSS_HP_MAX);
                if (boss && prev_boss && bhpm > 1.f && prev_bhp - bhp > 0.f) {
                    float d = (prev_bhp - bhp) / bhpm;
                    boss_dmg += d < 0.25f ? d : 0.25f;   // clamp per-decision
                }
                prev_bhp = bhp; prev_boss = boss;
                gp = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
                float lives = gp ? rd<float>(gp + G_LIFE_COUNT) : start_lives;
                if (r != 0 || lives < start_lives - 0.5f) { died = true; break; }
                if (paced) {
                    LARGE_INTEGER now; QueryPerformanceCounter(&now);
                    double want = (double)fs / 60.0;
                    double el = (double)(now.QuadPart - t_last.QuadPart) / qpf.QuadPart;
                    if (want - el > 0.001) Sleep((DWORD)((want - el) * 1000.0));
                    QueryPerformanceCounter(&t_last);
                }
            }

            uintptr_t g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
            s->ep_frames = frames;
            s->ep_score = g ? rd<int32_t>(g + G_SCORE) : 0;
            s->ep_graze = g ? rd<int32_t>(g + G_GRAZE) : 0;
            s->ep_died = died ? 1 : 0;
            s->ep_tick_status = r;
            s->ep_boss_dmg = boss_dmg;
            capture_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_IDLE:
        default:
            // no Sleep: WinMain re-calls us immediately, so a STEP command from
            // Python is picked up within microseconds (Sleep(1) is 1-15ms).
            return 0;
    }
}

// Mute this whole process at the OS mixer (WASAPI session volume). Sticks even
// if called before the game creates a sound buffer, so 8 training instances
// don't blast 8x stage BGM. Skipped in watch mode.
// (mingw's import libs don't carry these GUIDs - define them here.)
static const GUID kCLSID_MMDeviceEnumerator =
    {0xBCDE0395,0xE52F,0x467C,{0x8E,0x3D,0xC4,0x57,0x92,0x91,0x69,0x2E}};
static const GUID kIID_IMMDeviceEnumerator =
    {0xA95664D2,0x9614,0x4F35,{0xA7,0x46,0xDE,0x8D,0xB6,0x36,0x17,0xE6}};
static const GUID kIID_IAudioSessionManager2 =
    {0x77AA99A0,0x1BD6,0x484F,{0x8B,0xC7,0x2C,0x65,0x4C,0x9A,0x9B,0x6F}};

static void mute_self() {
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return;
    IMMDeviceEnumerator* en = nullptr;
    if (SUCCEEDED(CoCreateInstance(kCLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL,
                                   kIID_IMMDeviceEnumerator, (void**)&en))) {
        IMMDevice* dev = nullptr;
        if (SUCCEEDED(en->GetDefaultAudioEndpoint(eRender, eConsole, &dev))) {
            IAudioSessionManager2* mgr = nullptr;
            if (SUCCEEDED(dev->Activate(kIID_IAudioSessionManager2, CLSCTX_ALL,
                                        nullptr, (void**)&mgr))) {
                ISimpleAudioVolume* vol = nullptr;
                if (SUCCEEDED(mgr->GetSimpleAudioVolume(nullptr, FALSE, &vol))) {
                    vol->SetMute(TRUE, nullptr);
                    vol->Release();
                }
                mgr->Release();
            }
            dev->Release();
        }
        en->Release();
    }
}

// ---- crash diagnostics ---------------------------------------------------
// On the first access violation in the game, record faulting EIP + access
// address + r/w into the shm crash_* fields (then let it crash normally). Lets
// Python report *where* a game crash happened instead of just "reset hung".
static LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION && g_shm && !g_shm->crash_code) {
        g_shm->crash_eip  = (uint32_t)(uintptr_t)ep->ExceptionRecord->ExceptionAddress;
        g_shm->crash_addr = (uint32_t)ep->ExceptionRecord->ExceptionInformation[1];
        g_shm->crash_rw   = (uint32_t)ep->ExceptionRecord->ExceptionInformation[0];
        g_shm->crash_code = (uint32_t)code;         // set last: sentinel
    }
    return EXCEPTION_CONTINUE_SEARCH;   // let it crash normally after recording
}

// ---- setup ----------------------------------------------------------------
static DWORD WINAPI init_thread(LPVOID) {
    AddVectoredExceptionHandler(1, veh);
    char name[64];
    if (char* e = getenv("TH07_NO_DRAW"))
        g_stub_draw = atoi(e) != 0;
    if (char* e = getenv("TH07_RENDER"))
        g_render = atoi(e) != 0;
    if (char* e = getenv("TH07_DIFFICULTY"))
        g_difficulty = atoi(e);
    if (!g_render && !getenv("TH07_NO_MUTE"))
        mute_self();
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
    // capture_obs only fills bullets[0..BM_BULLET_MAX); mark the rest inactive so
    // the Python side doesn't read the memset-0 tail as live bullets at (0,0).
    for (int i = 0; i < MAX_BULLETS; ++i) {
        g_shm->bullets[i].x = INACTIVE;
        g_shm->bullets[i].y = INACTIVE;
    }

    g_snap_static = (uint8_t*)VirtualAlloc(nullptr, SNAP_SZ, MEM_COMMIT | MEM_RESERVE,
                                           PAGE_READWRITE);
    if (!g_snap_static) return 1;

    if (MH_Initialize() != MH_OK) return 1;
    if (MH_CreateHook((void*)FN_DO_TICK, (void*)&hooked_do_tick,
                      (void**)&orig_do_tick) != MH_OK) return 1;
    if (MH_CreateHook((void*)0x00430B50, (void*)&hooked_read_input,
                      (void**)&orig_read_input) != MH_OK) return 1;
    if (MH_CreateHook((void*)0x004345C0, (void*)&hooked_present,
                      (void**)&orig_present) != MH_OK) return 1;
    if (MH_CreateHook((void*)FN_RUN_ALL_ON_DRAW, (void*)&hooked_run_all_on_draw,
                      (void**)&orig_run_all_on_draw) != MH_OK) return 1;
    if (MH_CreateHook((void*)FN_REPLAY_RECORD, (void*)&hooked_rec,
                      (void**)&orig_rec) != MH_OK) return 1;
    if (MH_EnableHook(MH_ALL_HOOKS) != MH_OK) return 1;

    static const uint8_t nop2[2] = {0x90, 0x90};
    for (auto& L : LIMITER)
        if (memcmp((void*)L.va, L.want, 2) == 0) patch(L.va, nop2, 2);

    return 0;
}

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(h);
        timeBeginPeriod(1);   // 1ms scheduler tick (any residual Sleep(1))
        CreateThread(nullptr, 0, init_thread, nullptr, 0, nullptr);
    }
    return TRUE;
}
