// th07hook.dll - in-process control hook for Touhou 7 (PCB) v1.00b.
//
// Injected into a running th07.exe. Hooks:
//   * read_input (0x430B50) - FREE mode returns the real keyboard, STEP mode
//     returns the env's action.
//   * Window::do_tick (0x4346E0) - STEP mode advances exactly one call per
//     env step; IDLE spins; FREE runs normally so menus work.
// The 60fps limiter is NOP'd so do_tick doesn't busy-wait - but only once
// autonav starts (apply_limiter_patch in the ST_AUTONAV case); until then the
// game sits at the title at real-time speed so the launch isn't an 80x screech.
// The game only advances when the env asks, so the "Continue screen
// free-running" corruption cannot happen.
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

static inline int god_tick(void* self, void* edx);   // one tick, death-proofed in god mode

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
// god mode (env var TH07_GODMODE=1): the player cannot die. Used only for
// RECORDING - a weak/any driver can reach a Stage 2+ boss to record its
// patterns. The trained policy and the actual 1cc run never touch this.
static bool g_godmode = false;
typedef int(__fastcall* drawall_fn)(void* ecx, void* edx);
static drawall_fn orig_run_all_on_draw = nullptr;
static int __fastcall hooked_run_all_on_draw(void* ecx, void* edx) {
    if (g_stub_draw && g_shm && !(g_shm->state == ST_EVAL && g_shm->eval_render) &&
        !g_render &&
        (g_shm->state == ST_STEP || g_shm->state == ST_RESET ||
         g_shm->state == ST_EVAL || g_shm->state == ST_ROLLOUT ||
         g_shm->state == ST_HARD_RESET))
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

// The frame limiter is NOP'd so do_tick free-runs at ~80x. But if it's done at
// load time the game boots straight into 80x and the title BGM plays as an
// ear-splitting screech. So the limiter stays LIVE (real-time 60fps) while the
// game sits at the title screen; it's only NOP'd once autonav starts driving
// (apply_limiter_patch, called from the ST_AUTONAV case). Python holds a few
// seconds at the title before autonav so the launch is quiet and normal-speed.
static bool g_limiter_patched = false;

static void apply_limiter_patch() {
    if (g_limiter_patched) return;
    static const uint8_t nop2[2] = {0x90, 0x90};
    for (auto& L : LIMITER)
        if (memcmp((void*)L.va, L.want, 2) == 0) patch(L.va, nop2, 2);
    g_limiter_patched = true;
}

// ---- in-DLL policy: obs builder + MLP (mirror of native/env.py) --------------
// The observation is:
//   * a small scalar head (16)
//   * NDIRS "escape" scalars (9): for each of {stay, N, NE, E, SE, S, SW, W, NW}
//     how many frames until the player is hit if it holds that move for the
//     next DIR_HORIZON frames (1 = safe the whole time, ->0 = hit now). This
//     turns "which way do I dodge" into an argmax, and its stay-still entry
//     is low exactly when freezing is fatal - the thing evolution kept doing.
//   * a player-centred "danger grid" (13x13): each cell = how imminent a
//     bullet strike there is, from marching every live bullet's straight-line
//     path. Cells outside the playfield read 0.5 (wall).
//   * the nearest on-screen enemies (6 x 3).
constexpr int   GRID        = 13;        // danger-grid side (odd; player-centred)
constexpr int   GRID_R      = GRID / 2;  // 6
constexpr int   GCELLS      = GRID * GRID;
constexpr float GRID_CELL   = 12.0f;     // px per cell -> +-78 px window
constexpr float GRID_HORIZON = 24.0f;    // frames of bullet look-ahead
constexpr int   HEAD_DIM    = 16;
constexpr int   NDIRS       = 9;
constexpr int   K_NEAREST   = 128;      // grid + escape use only the K nearest bullets
                                        // (must match native/obs.py K_NEAREST)
constexpr float DIR_SPEED   = 4.0f;      // measured unfocused player move speed (px/frame)
constexpr float DIR_SPEED_FOCUS = 1.6f;  // focused move speed (obs.py v28: escape scan
                                         // uses this when the player is focused)
constexpr float DIR_HORIZON = 20.0f;     // escape look-ahead (frames)
constexpr float DIR_HIT_R2  = 7.0f * 7.0f;   // fallback only (real bullets carry a box)
constexpr float PLAYER_HALF = 2.0f;      // player AABB half-extent (== sim PLAYER_HB);
//   measured ~1.6-1.8; 2.0 is a deliberate safety margin (train conservative).
//   a bullet's real strike radius is its half-extent + PLAYER_HALF. Mirror of
//   native/obs.py: the danger grid stamps a plus of cells its (half+PLAYER_HALF)
//   disc reaches, and the escape scan uses the same per-bullet radius.
static const int KOFF[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
// measured player movement bounds (sim/physics.json)
constexpr float PX_LO = 8.0f, PX_HI = 376.0f, PY_LO = 16.0f, PY_HI = 432.0f;
constexpr int   M_ENEMIES   = 6;
constexpr int   M_ITEMS     = 8;         // nearest on-field items (P-drops etc.)
static_assert(OBS_DIM == HEAD_DIM + NDIRS + GCELLS + M_ENEMIES * 3 + M_ITEMS * 3,
              "OBS_DIM mismatch");

// {dx,dy}: index 0 = stay still, then the 8 compass dirs (matches A_DIRS /
// env.py _DIRS ordering used by decode_action).
static const int8_t OBS_DIRS[NDIRS][2] = {
    {0,0},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}
};

static float g_prev_bx[MAX_BULLETS];
static float g_prev_by[MAX_BULLETS];
static float g_prev_px = -9999.f, g_prev_py = -9999.f;
static void reset_bullet_hist() {
    for (int i = 0; i < MAX_BULLETS; ++i) { g_prev_bx[i] = -9999.f; g_prev_by[i] = -9999.f; }
    g_prev_px = g_prev_py = -9999.f;
}

// The "big target": the stage boss (GUI life bar) or, before it appears, the
// stage-1 midboss - a normal enemy on screen with a large maxlife. Returns the
// present flag and fills hp/hpmax. Shared by the obs head and the eval reward.
static bool big_target(float& hp, float& hpmax) {
    if (rd<int32_t>(GUI + GUI_BOSS_PRESENT) != 0) {
        float m = rd<float>(GUI + GUI_BOSS_HP_MAX);
        if (m > 1.f) { hp = rd<float>(GUI + GUI_BOSS_HP_CUR); hpmax = m; return true; }
    }
    int32_t ec = rd<int32_t>(ENEMY_MANAGER + EM_ENEMY_COUNT);
    if (ec < 0 || ec > 480) ec = 0;
    int best_ml = 0; float best_life = 0.f;
    for (int i = 0; i < ec && i < 64; ++i) {
        uintptr_t e = ENEMY_MANAGER + EM_ENEMIES + (uintptr_t)i * EM_ENEMY_STRIDE;
        float ey = rd<float>(e + ENEMY_POS + 4);
        if (ey < -8.f || ey > PLAYFIELD_H + 32.f) continue;       // offscreen spawner
        int32_t ml = rd<int32_t>(e + ENEMY_MAXLIFE);
        if (ml >= 200 && ml > best_ml) {
            best_ml = ml; best_life = (float)rd<int32_t>(e + ENEMY_LIFE);
        }
    }
    if (best_ml > 0) { hp = best_life; hpmax = (float)best_ml; return true; }
    hp = hpmax = 0.f;
    return false;
}

// Build the danger-grid observation. `frame_skip` scales the per-call position
// deltas into per-frame velocities for the trajectory march.
static void build_obs(float* o, int frame_skip) {
    const float W = PLAYFIELD_W, H = PLAYFIELD_H;
    const float vscale = 1.0f / (float)(frame_skip > 0 ? frame_skip : 3);
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

    // player velocity over the last decision (per-frame)
    float pvx = 0.f, pvy = 0.f;
    if (g_prev_px > -9000.f) {
        pvx = (px - g_prev_px) * vscale; pvy = (py - g_prev_py) * vscale;
    }
    g_prev_px = px; g_prev_py = py;

    float bt_hp = 0.f, bt_max = 0.f;
    bool bt = big_target(bt_hp, bt_max);

    // ---- danger grid -> o[HEAD_DIM + NDIRS ..] ----
    float* grid = o + HEAD_DIM + NDIRS;
    for (int i = 0; i < GCELLS; ++i) grid[i] = 0.f;
    for (int gy = 0; gy < GRID; ++gy)
        for (int gx = 0; gx < GRID; ++gx) {
            float wx = px + (gx - GRID_R) * GRID_CELL;
            float wy = py + (gy - GRID_R) * GRID_CELL;
            if (wx < PX_LO || wx > PX_HI || wy < PY_LO || wy > PY_HI)
                grid[gy * GRID + gx] = 0.5f;     // wall
        }

    // pass 1: gather the K_NEAREST live bullets (insertion-sorted by distance),
    // exactly as native/obs.py does before it builds the grid + escape scalars.
    static float lbx[K_NEAREST], lby[K_NEAREST], lvx[K_NEAREST], lvy[K_NEAREST];
    static float lbd[K_NEAREST], lbh[K_NEAREST];
    int nb = 0;
    float near_d = 1e9f;
    uintptr_t bb = BULLET_MANAGER + BM_BULLETS;
    const int nslot = (int)BM_BULLET_MAX < MAX_BULLETS ? (int)BM_BULLET_MAX : MAX_BULLETS;
    for (int i = 0; i < nslot; ++i) {
        uintptr_t b = bb + (uintptr_t)i * BM_BULLET_STRIDE;
        uint16_t st = rd<uint16_t>(b + BULLET_STATE);
        float bx = rd<float>(b + BULLET_POS), by = rd<float>(b + BULLET_POS + 4);
        bool live = (st != 0 && st != 6 &&
                     bx > -64 && bx < W + 64 && by > -64 && by < H + 64);
        float vx = 0.f, vy = 0.f;
        if (live && g_prev_bx[i] > -9000.f) {
            vx = (bx - g_prev_bx[i]) * vscale; vy = (by - g_prev_by[i]) * vscale;
        }
        g_prev_bx[i] = live ? bx : -9999.f;
        g_prev_by[i] = live ? by : -9999.f;
        if (!live) continue;
        if (vx > 24.f || vx < -24.f || vy > 24.f || vy < -24.f) vx = vy = 0.f;  // recycled slot
        float bh = rd<float>(b + BULLET_HITBOX) * 0.5f;   // AABB half-extent
        float rx = bx - px, ry = by - py;
        float d = sqrtf(rx * rx + ry * ry);
        if (d < near_d) near_d = d;
        if (nb == K_NEAREST && d >= lbd[K_NEAREST - 1]) continue;
        int j = (nb < K_NEAREST) ? nb++ : K_NEAREST - 1;
        for (; j > 0 && lbd[j - 1] > d; --j) {
            lbd[j] = lbd[j - 1]; lbx[j] = lbx[j - 1]; lby[j] = lby[j - 1];
            lvx[j] = lvx[j - 1]; lvy[j] = lvy[j - 1]; lbh[j] = lbh[j - 1];
        }
        lbd[j] = d; lbx[j] = bx; lby[j] = by; lvx[j] = vx; lvy[j] = vy; lbh[j] = bh;
    }

    // pass 2: march those bullets to stamp the danger grid. each bullet stamps a
    // plus of cells its (half + PLAYER_HALF) disc reaches (mirror of obs.py).
    const float inv_h = 1.0f / GRID_HORIZON;
    for (int k = 0; k < nb; ++k) {
        float bx = lbx[k], by = lby[k], vx = lvx[k], vy = lvy[k];
        float strike = lbh[k] + PLAYER_HALF; if (strike < 1.0f) strike = 1.0f;
        float reach2 = strike + GRID_CELL * 0.5f; reach2 *= reach2;
        for (int it = 0; it <= 48; ++it) {                 // 24 frames @ 0.5f
            float t = it * 0.5f;
            float bpx = bx + vx * t, bpy = by + vy * t;
            float basex = floorf((bpx - px) / GRID_CELL + 0.5f);
            float basey = floorf((bpy - py) / GRID_CELL + 0.5f);
            float danger = 1.0f - t * inv_h;
            for (int m = 0; m < 5; ++m) {
                float cxf = basex + KOFF[m][0], cyf = basey + KOFF[m][1];
                float ddx = bpx - (px + cxf * GRID_CELL);
                float ddy = bpy - (py + cyf * GRID_CELL);
                if (ddx * ddx + ddy * ddy >= reach2) continue;
                int gx = (int)cxf + GRID_R, gy = (int)cyf + GRID_R;
                if (gx < 0 || gx >= GRID || gy < 0 || gy >= GRID) continue;
                float* c = &grid[gy * GRID + gx];
                if (danger > *c) *c = danger;
            }
        }
    }

    // ---- escape scalars -> o[HEAD_DIM ..] ----
    // for each candidate move: closest-approach time to any nearby bullet, and
    // the frame the path leaves the playfield. normalised by DIR_HORIZON
    // (1 = safe the whole look-ahead).
    float* od = o + HEAD_DIM;
    const float dir_speed = focus ? DIR_SPEED_FOCUS : DIR_SPEED;   // obs.py v28
    for (int dd = 0; dd < NDIRS; ++dd) {
        float ndx = OBS_DIRS[dd][0], ndy = OBS_DIRS[dd][1];
        float L = sqrtf(ndx * ndx + ndy * ndy);
        float pmx = 0.f, pmy = 0.f;
        if (L > 0.f) { pmx = ndx / L * dir_speed; pmy = ndy / L * dir_speed; }
        float safe_t = DIR_HORIZON;
        if (pmx > 0.f)      { float tw = (PX_HI - px) / pmx; if (tw < safe_t) safe_t = tw; }
        else if (pmx < 0.f) { float tw = (PX_LO - px) / pmx; if (tw < safe_t) safe_t = tw; }
        if (pmy > 0.f)      { float tw = (PY_HI - py) / pmy; if (tw < safe_t) safe_t = tw; }
        else if (pmy < 0.f) { float tw = (PY_LO - py) / pmy; if (tw < safe_t) safe_t = tw; }
        if (safe_t < 0.f) safe_t = 0.f;
        for (int i = 0; i < nb && safe_t > 0.f; ++i) {
            if (lbd[i] > 150.f) break;                  // sorted; matches obs.py `near` gate
            float r0x = px - lbx[i], r0y = py - lby[i];
            float rvx = pmx - lvx[i], rvy = pmy - lvy[i];
            float a = rvx * rvx + rvy * rvy;
            float ts;
            if (a < 1e-6f) ts = 0.f;
            else { ts = -(r0x * rvx + r0y * rvy) / a; if (ts < 0.f) ts = 0.f; }
            if (ts >= safe_t) continue;
            float cx = r0x + rvx * ts, cy = r0y + rvy * ts;
            float si = lbh[i] + PLAYER_HALF; if (si < 1.0f) si = 1.0f;
            if (cx * cx + cy * cy < si * si) safe_t = ts;
        }
        od[dd] = safe_t / DIR_HORIZON;
    }

    // ---- head (16) ----
    o[0] = px / W; o[1] = py / H;
    o[2] = pvx / 6.f; o[3] = pvy / 6.f;
    o[4] = (float)focus;
    o[5] = lives / 9.f; o[6] = bombs / 9.f; o[7] = power / 128.f;
    o[8] = tanhf(graze / 100.f);
    o[9] = stage / 6.f;
    o[10] = (pst == 0) ? 1.f : 0.f;
    o[11] = (pst == 2) ? 1.f : 0.f;
    o[12] = (near_d < 1e8f) ? fminf(near_d / 80.f, 3.f) : 2.f;
    o[13] = bt ? 1.f : 0.f;
    o[14] = (bt && bt_max > 0.f) ? (bt_hp / bt_max) : 0.f;
    o[15] = 0.f;

    // insertion-sort a (dist2, x, y, hpf) candidate into the K nearest (mirror
    // of env.py's cand.sort(key=dist2)[:K]).
    struct Cand { float d2, x, y, v; };
    auto push_near = [](Cand* arr, int& n, int K, float d2, float x, float y, float v) {
        if (n == K && d2 >= arr[K - 1].d2) return;
        int j = (n < K) ? n++ : K - 1;
        for (; j > 0 && arr[j - 1].d2 > d2; --j) arr[j] = arr[j - 1];
        arr[j] = { d2, x, y, v };
    };

    // ---- enemies (6 * 3): NEAREST on-field, [rel/128, life/maxlife] ----
    // matches env.py: EM_BOSSES[0] (Cirno/Letty live there, not in EM_ENEMIES)
    // folded in as a normal enemy, then all sorted by distance.
    float* oe = o + HEAD_DIM + NDIRS + GCELLS;
    for (int i = 0; i < M_ENEMIES * 3; ++i) oe[i] = 0.f;
    Cand en[M_ENEMIES]; int en_n = 0;
    {
        uintptr_t bptr = rd<uintptr_t>(ENEMY_MANAGER + EM_BOSSES);   // &EM_BOSSES[0]
        if (bptr > 0x00400000 && bptr < 0x7FFFFFFF) {
            float bx = rd<float>(bptr + ENEMY_POS), by = rd<float>(bptr + ENEMY_POS + 4);
            int32_t bl = rd<int32_t>(bptr + ENEMY_LIFE), bml = rd<int32_t>(bptr + ENEMY_MAXLIFE);
            if (bx > -64.f && bx < 448.f && by > -80.f && by < 520.f && bml >= 1 && bml <= 1000000)
                push_near(en, en_n, M_ENEMIES, (bx - px) * (bx - px) + (by - py) * (by - py),
                          bx, by, (float)bl / (float)bml);
        }
    }
    int32_t ec = rd<int32_t>(ENEMY_MANAGER + EM_ENEMY_COUNT);
    if (ec < 0 || ec > 480) ec = 0;
    for (int i = 0; i < ec && i < MAX_ENEMIES; ++i) {
        uintptr_t e = ENEMY_MANAGER + EM_ENEMIES + (uintptr_t)i * EM_ENEMY_STRIDE;
        float ex = rd<float>(e + ENEMY_POS), ey = rd<float>(e + ENEMY_POS + 4);
        if (ey < -8.f || ey > 480.f) continue;
        int32_t life = rd<int32_t>(e + ENEMY_LIFE), ml = rd<int32_t>(e + ENEMY_MAXLIFE);
        if (ml < 1) ml = 1;
        push_near(en, en_n, M_ENEMIES, (ex - px) * (ex - px) + (ey - py) * (ey - py),
                  ex, ey, (float)life / (float)ml);
    }
    for (int k = 0; k < en_n; ++k) {
        oe[k * 3 + 0] = (en[k].x - px) / 128.f;
        oe[k * 3 + 1] = (en[k].y - py) / 128.f;
        oe[k * 3 + 2] = fminf(fmaxf(en[k].v, 0.f), 1.f);
    }

    // ---- items (8 * 3): NEAREST on-field, [rel/128, rel/128, type/9] ----
    // matches env.py _item_arrays: scan all IM_ITEM_MAX slots, in_use, y in
    // [-16, 464], sort by distance.
    float* oi = o + HEAD_DIM + NDIRS + GCELLS + M_ENEMIES * 3;
    for (int i = 0; i < M_ITEMS * 3; ++i) oi[i] = 0.f;
    Cand it[M_ITEMS]; int it_n = 0;
    for (int i = 0; i < (int)IM_ITEM_MAX; ++i) {
        uintptr_t s = ITEM_MANAGER + (uintptr_t)i * IM_ITEM_STRIDE;
        if (rd<uint8_t>(s + ITEM_IN_USE) == 0) continue;
        float ix = rd<float>(s + ITEM_POS), iy = rd<float>(s + ITEM_POS + 4);
        if (iy < -16.f || iy > 464.f) continue;
        uint8_t typ = rd<uint8_t>(s + ITEM_TYPE);
        push_near(it, it_n, M_ITEMS, (ix - px) * (ix - px) + (iy - py) * (iy - py),
                  ix, iy, (float)typ);
    }
    for (int k = 0; k < it_n; ++k) {
        oi[k * 3 + 0] = (it[k].x - px) / 128.f;
        oi[k * 3 + 1] = (it[k].y - py) / 128.f;
        oi[k * 3 + 2] = it[k].v / 9.f;
    }
}

// MLP: Linear(OBS_DIM,h1)-Tanh-Linear(h1,h2)-Tanh-Linear(h2,36).
// Flat layout matches policy.py get_flat(): W0,b0,W1,b1,W2,b2 (torch Linear
// weight is [out,in] row-major). Writes the 36 output logits to `out`.
static void mlp_logits(const float* w, const float* in, int h1, int h2, float* out) {
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
    const float* bias = p + (size_t)N_ACTIONS * h2;
    for (int j = 0; j < N_ACTIONS; ++j) {
        float acc = 0.f; const float* row = p + (size_t)j * h2;
        for (int k = 0; k < h2; ++k) acc += row[k] * b[k];
        out[j] = acc + bias[j];
    }
}

static int mlp_forward(const float* w, const float* in, int h1, int h2) {
    float lg[N_ACTIONS];
    mlp_logits(w, in, h1, h2, lg);
    int best = 0; float bestv = -1e30f;
    for (int j = 0; j < N_ACTIONS; ++j) if (lg[j] > bestv) { bestv = lg[j]; best = j; }
    return best;
}

// splitmix64 -> a valid draw from softmax(logits) via the Gumbel-max trick
// (only needs uniforms). PPO recomputes log-prob on the Python side, so the
// sampler just has to draw ~the right distribution, not match a reference bit.
static inline uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline float sm64_uniform(uint64_t* s) {
    return (float)((sm64(s) >> 40) * (1.0 / 16777216.0));   // 24-bit in [0,1)
}
static int gumbel_sample(const float* logits, uint64_t* rng) {
    int best = 0; float bestv = -1e30f;
    for (int i = 0; i < N_ACTIONS; ++i) {
        float u = sm64_uniform(rng);
        float gi = -logf(-logf(u + 1e-20f) + 1e-20f);
        float v = logits[i] + gi;
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}

// Engine-level Stage 1 reload (shared by ST_HARD_RESET and ST_ROLLOUT). Ticks
// through the ~40-frame teardown+fade + a short warmup; returns frames spent.
static int engine_stage1_reload(void* self, void* edx, Shm* s) {
    constexpr int32_t WARM = 90;
    wr<int32_t>(GAME_MANAGER + GM_STAGE, 1);
    wr<int32_t>(SUPERVISOR + SV_RETRY_MODE, SV_RETRY_VAL);
    int nav = 0;
    const int MAXF = 1800;
    bool saw_reload = false;
    int32_t last_tmr = rd<int32_t>(GAME_MANAGER + GM_STAGE_TIMER);
    for (; nav < MAXF; ++nav) {
        s->action = 0;
        wr<uint8_t>(FRAMESKIP_BYTE, 0);
        if (god_tick(self, edx) != 0) break;
        int32_t tmr = rd<int32_t>(GAME_MANAGER + GM_STAGE_TIMER);
        int gm  = rd<int32_t>(SUPERVISOR + SV_GAMEMODE);
        int stg = rd<int32_t>(GAME_MANAGER + GM_STAGE);
        if (!saw_reload && tmr < last_tmr && tmr < 30) saw_reload = true;
        last_tmr = tmr;
        if (saw_reload && gm == 2 && stg == 1 && tmr >= WARM) break;
    }
    reset_bullet_hist();
    return saw_reload ? nav : -1;
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

// Build the full 236-d obs in C and publish it to shm for the Python env to read
// verbatim (skips the per-step Python obs rebuild). frame_skip scales bullet/
// player velocity; post-reset the bullet history is cleared so velocity reads 0
// and the exact value doesn't matter.
static void write_step_obs() {
    build_obs(g_shm->step_obs, g_shm->repeat ? (int)g_shm->repeat : 3);
    g_shm->step_obs_frame = g_shm->frame;
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
                     g_shm->state == ST_EVAL || g_shm->state == ST_HARD_RESET ||
                     g_shm->state == ST_ROLLOUT);
}

static uint16_t __cdecl hooked_read_input(void) {
    if (driving())
        return g_shm->action;
    uint16_t w = orig_read_input();
    if (g_shm && g_shm->state == ST_FREE)   // RE aid: mirror the live input word
        g_shm->action = w;                  // (Python isn't writing it in FREE mode)
    return w;
}

// Present is DWM-vsync-capped in a window. Skip it while we drive the game
// (rendering still runs, just no buffer flip) so logic isn't throttled to 144 Hz.
static int __cdecl hooked_present(void) {
    bool want = g_render || (g_shm && g_shm->state == ST_EVAL && g_shm->eval_render);
    if (driving() && !want)
        return 0;
    return orig_present();
}

// One logic tick, with god mode applied around it when active. Pins the player
// to state 3 (invuln: flashing, still moves & shoots) before the tick so no hit
// registers; if a death slips through anyway, revert state + position + life
// count to their pre-tick values. Pure pass-through when g_godmode is off.
static inline int god_tick(void* self, void* edx) {
    if (!g_godmode) return orig_do_tick(self, edx);
    uint8_t* pst = (uint8_t*)(PLAYER + PL_STATE);
    if (*pst == 0) *pst = 3;
    float px = rd<float>(PLAYER + PL_POS_X);
    float py = rd<float>(PLAYER + PL_POS_Y);
    uintptr_t gp = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
    float lv = gp ? rd<float>(gp + G_LIFE_COUNT) : 0.f;
    int r = orig_do_tick(self, edx);
    if (*pst == 1 || *pst == 2) {              // died anyway -> undo it
        *pst = 3;
        wr<float>(PLAYER + PL_POS_X, px);
        wr<float>(PLAYER + PL_POS_Y, py);
        if (gp) wr<float>(gp + G_LIFE_COUNT, lv);
    }
    return r;
}

static int __fastcall hooked_do_tick(void* self, void* edx) {
    Shm* s = g_shm;
    s->alive = 1;

    // Window::do_tick starts with `if ([this+8] == 0) return 0;` - a "run a
    // logic frame now" gate the WinMain loop normally toggles. Headless it can
    // stick at 0 around boss transitions (Cirno defeat), freezing all game logic
    // while our frame counter keeps ticking. Force it on whenever we're driving.
    if (self && s->state != ST_FREE && s->state != ST_IDLE)
        *(volatile int32_t*)((char*)self + 8) = 1;

    switch (s->state) {
        case ST_FREE: {
            int r = god_tick(self, edx);
            capture_obs();
            return r;
        }

        case ST_STEP: {
            uint16_t rep = s->repeat ? s->repeat : 1;
            int r = 0;
            for (uint16_t k = 0; k < rep; ++k) {
                wr<uint8_t>(FRAMESKIP_BYTE, 0);   // 1 logic tick per do_tick call
                r = god_tick(self, edx);
                s->frame++;
                if (r != 0) break;               // stage clear / quit
            }
            s->tick_status = r;
            capture_obs();
            write_step_obs();
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
            // First real drive command: the title-screen wait is over, NOP the
            // frame limiter now so nav (and everything after) runs at ~80x.
            apply_limiter_patch();
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
                if (god_tick(self, edx) != 0) break;
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
            reset_bullet_hist();          // fresh velocity baseline (matches env.reset())
            capture_obs();
            write_step_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_HARD_RESET: {
            // Engine-level "Give Up and Retry" - no pause menu, no relaunch.
            // (see engine_stage1_reload: forces stage=1, writes the supervisor
            // retry word, ticks the teardown+fade+warmup. Does NOT touch the
            // replay recorder - the retry re-creates it via the engine's own
            // stage init.)
            int nav = engine_stage1_reload(self, edx, s);
            s->action = 0;
            s->frame = 0;
            s->nav_frames = nav;
            capture_obs();
            write_step_obs();
            s->done = 1;
            s->state = ST_IDLE;
            return 0;
        }

        case ST_ROLLOUT: {
            // Collect a T-step PPO trajectory entirely in the DLL: build_obs ->
            // actor logits -> Gumbel sample -> tick fs frames -> env.py reward ->
            // record. On episode end, engine_stage1_reload (teardown frames are
            // NOT counted as steps). Python computes GAE + the PPO update and
            // ships new weights. Games run at native speed - no per-step Python.
            const int T  = (int)(s->roll_T < ROLL_T_MAX ? s->roll_T : ROLL_T_MAX);
            const int fs  = s->roll_frame_skip ? (int)s->roll_frame_skip : 3;
            const int h1  = (int)s->roll_h1, h2 = (int)s->roll_h2;
            const uint32_t ep_cap = s->roll_max_ep_frames ? s->roll_max_ep_frames : 10800;
            uint64_t rng = s->roll_seed ? (uint64_t)s->roll_seed * 0x2545F4914F6CDD1Dull
                                        : 0x9E3779B97F4A7C15ull;
            s->roll_seed++;   // advance so the next rollout draws fresh noise

            uintptr_t g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
            int32_t prev_score = g ? rd<int32_t>(g + G_SCORE) : 0;
            float prev_lives = g ? rd<float>(g + G_LIFE_COUNT) : 0.f;
            float prev_boss = rd<float>(GUI + GUI_BOSS_HP_CUR);
            uint32_t ep_frames = 0, ep_ends = 0;
            float logits[N_ACTIONS];

            for (int t = 0; t < T; ++t) {
                // build_obs maintains a per-bullet velocity history across calls,
                // so call it exactly ONCE per decision (matches ST_STEP). The
                // GAE bootstrap obs is built only on the final step (after that
                // the corrupted history doesn't matter - the rollout is over).
                build_obs(s->roll_obs[t], fs);
                mlp_logits(s->weights, s->roll_obs[t], h1, h2, logits);
                int a = gumbel_sample(logits, &rng);
                s->roll_act[t] = (uint8_t)a;
                s->action = decode_action(a);

                int r = 0;
                for (int k = 0; k < fs; ++k) {
                    wr<uint8_t>(FRAMESKIP_BYTE, 0);
                    r = god_tick(self, edx);
                    s->frame++; ep_frames++;
                    if (r != 0) break;
                }
                if (t == T - 1)               // GAE bootstrap: obs after the last action
                    build_obs(s->roll_last_obs, fs);

                g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
                int32_t score = g ? rd<int32_t>(g + G_SCORE) : prev_score;
                float lives = g ? rd<float>(g + G_LIFE_COUNT) : prev_lives;
                int32_t bp = rd<int32_t>(GUI + GUI_BOSS_PRESENT);
                float bhp = rd<float>(GUI + GUI_BOSS_HP_CUR);
                float bmax = rd<float>(GUI + GUI_BOSS_HP_MAX);

                // reward: mirror env.py Th07Env.step()
                float rew = 0.02f * fs;
                rew += (float)(score - prev_score) * 1e-4f;
                if (bp && bmax > 0.f)
                    rew += fmaxf(0.f, prev_boss - bhp) / bmax * 3.0f;
                bool died = lives < prev_lives - 0.5f;
                if (died) rew -= 5.0f;
                s->roll_rew[t] = rew;

                bool done = died || (r != 0) || (ep_frames >= ep_cap);
                s->roll_done[t] = done ? 1 : 0;
                prev_score = score; prev_lives = lives; prev_boss = bhp;

                if (done) {
                    ep_ends++;
                    engine_stage1_reload(self, edx, s);
                    g = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
                    prev_score = g ? rd<int32_t>(g + G_SCORE) : 0;
                    prev_lives = g ? rd<float>(g + G_LIFE_COUNT) : 0.f;
                    prev_boss = 0.f;
                    ep_frames = 0;
                }
            }

            s->action = 0;
            s->roll_steps_done = T;
            s->roll_ep_ends = ep_ends;
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

            // random phase offset: run the policy for eval_warmup uncounted
            // frames so it can't just memorise one fixed bullet sequence.
            bool warm_dead = false;
            for (uint32_t k = 0; k < s->eval_warmup && !warm_dead; ++k) {
                build_obs(obs, (int)fs);
                s->action = decode_action(mlp_forward(s->weights, obs, h1, h2));
                wr<uint8_t>(FRAMESKIP_BYTE, 0);
                int wr_ = god_tick(self, edx);
                s->frame++;
                uintptr_t wg = rd<uintptr_t>(GAME_MANAGER + GM_GLOBALS_PTR);
                float wl = wg ? rd<float>(wg + G_LIFE_COUNT) : start_lives;
                if (wr_ != 0 || wl < start_lives - 0.5f) warm_dead = true;
            }
            reset_bullet_hist();   // fresh velocity baseline for the counted run

            uint32_t frames = 0;
            uint32_t decisions = 0;
            int r = 0;
            bool died = warm_dead;
            bool first = true;
            float boss_dmg = 0.f;
            float x_dev_sum = 0.f;
            float prev_bt_hp = 0.f, prev_bt_max = 0.f;
            while (!warm_dead && frames < cap) {
                build_obs(obs, (int)fs);
                if (first) { memcpy(s->dbg_obs, obs, sizeof(obs)); first = false; }
                s->action = decode_action(mlp_forward(s->weights, obs, h1, h2));
                for (uint32_t k = 0; k < fs && frames < cap; ++k) {
                    wr<uint8_t>(FRAMESKIP_BYTE, 0);
                    r = god_tick(self, edx);
                    s->frame++; frames++;
                    if (r != 0) break;
                }
                float pxn = rd<float>(PLAYER + PL_POS_X) / PLAYFIELD_W - 0.5f;
                x_dev_sum += pxn * pxn;
                decisions++;
                // big-target damage: sum per-decision HP drops (normalised by
                // hp_max) dealt to the stage boss OR the midboss. Spell-card
                // refills -> negative delta, ignored. Only count when the same
                // target (matched by hp_max) was present on the previous check.
                float bt_hp = 0.f, bt_max = 0.f;
                bool bt = big_target(bt_hp, bt_max);
                if (bt && prev_bt_max > 1.f && fabsf(bt_max - prev_bt_max) < 0.5f
                        && prev_bt_hp - bt_hp > 0.f) {
                    float d = (prev_bt_hp - bt_hp) / bt_max;
                    boss_dmg += d < 0.25f ? d : 0.25f;   // clamp per-decision
                }
                prev_bt_hp = bt ? bt_hp : 0.f;
                prev_bt_max = bt ? bt_max : 0.f;
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
            s->ep_x_dev = decisions ? x_dev_sum / (float)decisions : 0.f;
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

// Mute this process's audio session. Best-effort: the game creates its
// DirectSound session slightly after the DLL loads, so this often doesn't
// stick on its own - env.py also retries a per-session mute from Python, and
// the limiter stays at 60fps (autonav NOP's it) so the user has a few real
// seconds at the title before the game speeds up.
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
    if (char* e = getenv("TH07_GODMODE"))
        g_godmode = atoi(e) != 0;
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
    g_shm->struct_size = (uint32_t)sizeof(Shm);
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

    // Limiter stays LIVE here - the game sits at the title at real-time 60fps
    // (normal audio) until autonav starts, which NOP's it (ST_AUTONAV case).
    // TH07_NO_MUTE / TH07_NO_HOLD: skip the wait, go fast immediately.
    if (getenv("TH07_NO_MUTE") || getenv("TH07_NO_HOLD"))
        apply_limiter_patch();

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
