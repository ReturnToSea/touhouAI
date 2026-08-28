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
constexpr uint32_t SHM_VERSION = 1;

constexpr int MAX_BULLETS = 2048;
constexpr int MAX_ENEMIES = 64;

// control.state values
enum ShmState : uint32_t {
    ST_IDLE  = 0,  // hook spins, does not advance logic
    ST_STEP  = 1,  // env wants `repeat` logic ticks with `action` held
    ST_FREE  = 2,  // hook calls the original do_tick (menus / navigation, rendered)
    ST_RESET = 3,  // env wants an episode reset (restore snapshot)
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
    int32_t  tick_status;         // last run_all_on_tick() return (0 = normal)
    uint32_t alive;               // hook -> 1 each pass (heartbeat / crash detect)

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
    Bullet   bullets[MAX_BULLETS];
    Enemy    enemies[MAX_ENEMIES];
};
#pragma pack(pop)

}  // namespace th07
