// Shared-memory contract between th07hook.dll (writer of obs / reader of cmd)
// and the Python env (writer of cmd / reader of obs).
//
// Mapping name: "th07hook_<pid>"  (pid of the game process)
// Single-writer per field; `state`/`done` are the handshake.
//
// Keep this struct in sync with native/shm.py.
#pragma once
#include <stdint.h>

namespace th07 {

constexpr uint32_t SHM_MAGIC   = 0x37304854;  // 'TH07'
constexpr uint32_t SHM_VERSION = 2;

constexpr int   MAX_BULLETS = 2048;   // >= the 1025 pool slots
constexpr int   MAX_ENEMIES = 64;
constexpr float INACTIVE    = -9999.0f;  // sentinel for an empty bullet slot

constexpr int   OBS_DIM     = 192;
constexpr int   N_ACTIONS   = 36;
constexpr int   MAX_HIDDEN  = 256;   // per-layer hidden cap for the in-DLL MLP
constexpr int   MAX_WEIGHTS = 1 << 16;  // 64k float32 = 256 KB flat param buffer

// control.state values
enum ShmState : uint32_t {
    ST_IDLE     = 0,  // hook spins, does not advance logic
    ST_STEP     = 1,  // env wants `repeat` logic ticks with `action` held
    ST_FREE     = 2,  // hook calls the original do_tick (menus, rendered)
    ST_RESET    = 3,  // restore the snapshot, then obs + done
    ST_SNAPSHOT = 4,  // capture the current game state as the reset point
    ST_AUTONAV  = 5,  // tap Shoot through the menus until gamemode==2
    ST_EVAL     = 6,  // reset + run a whole episode with the in-DLL MLP policy
};

#pragma pack(push, 4)
struct Bullet { float x, y, vx, vy; };
struct Enemy  { float x, y; int32_t life, maxlife; };

struct Shm {
    uint32_t magic;
    uint32_t version;

    // --- handshake (env writes state/action/repeat; hook writes done/frame) ---
    volatile uint32_t state;      // ShmState
    volatile uint32_t done;       // hook -> 1 when a STEP/RESET completed
    volatile uint32_t frame;      // logic frames elapsed since inject
    uint16_t action;              // Btn bitmask to hold during the step
    uint16_t repeat;              // logic ticks per step (>=1)
    int32_t  tick_status;         // last do_tick() return (0 = normal)
    uint32_t alive;               // hook -> 1 each pass (heartbeat / crash detect)
    uint32_t have_snapshot;       // hook -> 1 once ST_SNAPSHOT has run
    int32_t  nav_frames;          // hook -> frames ST_AUTONAV took (-1 = failed)

    // --- observation (hook writes) ---
    float    player_x, player_y, player_vx, player_vy;
    uint8_t  player_state;        // 0 alive 1 respawning 2 dead 3 invuln 4 border
    uint8_t  player_focus;
    uint8_t  _pad0[2];
    float    lives, bombs, power;
    int32_t  score, graze;
    int32_t  cherry, cherry_max;
    int32_t  stage, difficulty;
    int32_t  gamemode;

    int32_t  boss_present;
    float    boss_hp, boss_hp_max;

    int32_t  bullet_count;        // number of valid entries in bullets[]
    int32_t  enemy_count;

    // crash diagnostics (VEH fills these on the first access violation)
    uint32_t crash_code;         // 0 = none; else the SEH exception code
    uint32_t crash_eip;          // faulting instruction address
    uint32_t crash_addr;         // address that was read/written
    uint32_t crash_rw;           // 0 = read, 1 = write

    // --- in-DLL policy evaluation (ST_EVAL) ---
    // env writes eval_* + weights, sets state=ST_EVAL; DLL runs a full episode
    // (reset -> obs -> MLP -> action -> tick, until death or the frame cap) and
    // writes ep_*.
    uint32_t eval_frame_skip;    // frames the action is held per decision
    uint32_t eval_max_frames;    // episode cap in game frames
    uint32_t eval_h1, eval_h2;   // MLP hidden layer sizes
    uint32_t eval_render;        // 1 = draw + present + pace to 60 Hz (watch)
    uint32_t ep_frames;          // frames survived
    int32_t  ep_score;           // final score
    int32_t  ep_graze;           // final graze
    uint32_t ep_died;            // 1 = lost a life / terminal, 0 = hit the cap
    int32_t  ep_tick_status;     // last do_tick return
    float    dbg_obs[OBS_DIM];   // DLL writes the episode's first build_obs() here

    Bullet   bullets[MAX_BULLETS];
    Enemy    enemies[MAX_ENEMIES];
    float    weights[MAX_WEIGHTS];  // flat MLP params (env writes)
};
#pragma pack(pop)

}  // namespace th07
