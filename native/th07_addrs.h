// Static addresses / struct offsets for Touhou 7 (PCB) v1.00b (original).
// Mirrors feasibility/th07_data.py. Source: github.com/exphp-share/th-re-data.
// th07.exe has no ASLR -> image base 0x400000, addresses usable directly.
#pragma once
#include <stdint.h>

namespace th07 {

// --- functions ---
constexpr uintptr_t FN_DO_TICK          = 0x004346E0; // Window::do_tick(this)
constexpr uintptr_t FN_RUN_ALL_ON_TICK  = 0x0042FD60; // UpdateFuncRegistry::run_all_on_tick(this) __thiscall -> int
constexpr uintptr_t FN_RUN_ALL_ON_DRAW  = 0x0042FE20;
constexpr uintptr_t CHAIN               = 0x00626218; // 'this' for run_all_on_tick / run_all_on_draw

// --- input read site to neutralise (Supervisor::on_tick) ---
// 0x437D74: mov ax,[0x4B9E4C] ; 0x437D7A: mov [0x4B9E54],ax  (prev = cur)
// 0x437D80: call 0x430B50     (read keyboard -> ax)
// 0x437D85: mov [0x4B9E4C],ax (cur = keyboard)
// We NOP 0x437D80 (5 bytes) + 0x437D85 (6 bytes) so [0x4B9E4C] keeps our value;
// the prev-input copy at 0x437D7A still runs, which is what we want.
constexpr uintptr_t INPUT_READ_CALL     = 0x00437D80; // 5 bytes: e8 xx xx xx xx
constexpr uintptr_t INPUT_STORE_CUR     = 0x00437D85; // 6 bytes: 66 a3 4c 9e 4b 00

constexpr uintptr_t INPUT_CUR           = 0x004B9E4C; // uint16 buttons this frame
constexpr uintptr_t INPUT_PREV          = 0x004B9E54; // uint16 buttons last frame

// input bits (confirmed): left 0x40 right 0x80 up 0x10 down 0x20 shoot 0x01 slow 0x04 bomb 0x02
enum Btn : uint16_t {
    BTN_SHOOT = 0x01, BTN_BOMB = 0x02, BTN_SLOW = 0x04, BTN_SKIP = 0x08,
    BTN_UP = 0x10, BTN_DOWN = 0x20, BTN_LEFT = 0x40, BTN_RIGHT = 0x80,
};

// --- single-instance guard (WinMain) ---
// 0x435BE9 call CreateMutexA ; 0x435BEF mov [0x135E1F4],eax ; 0x435BF4 call
// GetLastError ; 0x435BFA cmp eax,0xB7 (ERROR_ALREADY_EXISTS) ; 0x435BFF jnz
// 0x435C1B (normal path). A 2nd concurrent th07.exe hits the else branch: a
// modal error MessageBox, then return -1 -> abort. Patch the jnz (75) to an
// unconditional jmp (EB) so extra instances are allowed (needed for SubprocVecEnv).
constexpr uintptr_t INSTANCE_GUARD_JNZ = 0x00435BFF; // expect 0x75, patch -> 0xEB

// --- static objects ---
constexpr uintptr_t GUI            = 0x0049FBF0;
constexpr uintptr_t PLAYER         = 0x004BDAD8;
constexpr uintptr_t SUPERVISOR     = 0x00575950;
constexpr uintptr_t GAME_MANAGER   = 0x00626270;
// difficulty index, read by the menu AND by gameplay/scoring code directly
// (not via the GameManager ptr). 0 Easy 1 Normal 2 Hard 3 Lunatic 4 Extra.
constexpr uintptr_t DIFFICULTY_SEL = 0x00626280;

// Replay input recorder (__thiscall) - also processes live input, so it can't
// be NOP'd. It appends to a heap buffer via this+0x84 which advances every
// frame and never wraps -> AV at 0x442DA8 after ~233k frames. We hook it to
// capture `this` and snapshot/restore the object head each episode so the
// write pointer resets.
constexpr uintptr_t FN_REPLAY_RECORD = 0x00442CD0;
constexpr uintptr_t BULLET_MANAGER = 0x0062F958;
constexpr uintptr_t ENEMY_MANAGER  = 0x009A9B00;
constexpr uintptr_t STAGE_NUM      = 0x01347FC8;

// --- Supervisor ---
constexpr uintptr_t SV_CALC_COUNT  = 0x150; // int32 loop counter
constexpr uintptr_t SV_GAMEMODE    = 0x154; // uint32 (2 == in a run)
// Writing SV_RETRY_VAL here is exactly what the pause-menu "Give Up and Retry ->
// Yes" does: the WinMain loop consumes it, tears the stage down + rebuilds it
// (~24f) then runs a ~15f fade, and clears it back to 2. Verified by
// native/probe_write.py (score resets, stage stays/forces to 1).
constexpr uintptr_t SV_RETRY_MODE  = 0x158; // int32 transition-request word
constexpr int32_t   SV_RETRY_VAL   = 10;

// --- zPlayer ---
constexpr uintptr_t PL_POS_X       = 0x0930; // float (y +4, z +8)
constexpr uintptr_t PL_POS_Y       = 0x0934;
constexpr uintptr_t PL_STATE       = 0x2408; // u8: 0 alive 1 respawning 2 dead 3 invuln 4 border
constexpr uintptr_t PL_IS_FOCUS    = 0x240B; // i8
constexpr uintptr_t PL_VEL_X       = 0x0948; // D3DXVECTOR3 velocity (after hitbox/graze boxes) - verify

// --- zGameManager ---
constexpr uintptr_t GM_GLOBALS_PTR = 0x0008; // zGlobals*
constexpr uintptr_t GM_DIFFICULTY  = 0x0010; // int32
constexpr uintptr_t GM_STAGE_TIMER = 0x95E8; // int32 per-stage frame counter (-> 0 on (re)load)
constexpr uintptr_t GM_STAGE       = 0x95EC; // int32
constexpr uintptr_t GM_CHERRY_MAX  = 0x9618;
constexpr uintptr_t GM_CHERRY      = 0x961C;

// --- zGlobals (*(GAME_MANAGER + 8)) ---
constexpr uintptr_t G_SCORE        = 0x00; // int32 displayed score
constexpr uintptr_t G_GRAZE        = 0x18; // int32
constexpr uintptr_t G_LIFE_COUNT   = 0x5C; // float
constexpr uintptr_t G_BOMB_COUNT   = 0x68; // float
constexpr uintptr_t G_POWER        = 0x7C; // float

// --- zBulletManager ---
constexpr uintptr_t BM_BULLETS       = 0x0000B8C0;  // zBullet[0x401]
constexpr uintptr_t BM_BULLET_STRIDE = 0x00000D68;
constexpr uintptr_t BM_BULLET_MAX    = 0x401;       // 1025
constexpr uintptr_t BM_BULLET_COUNT  = 0x0037A128;  // int32
constexpr uintptr_t BULLET_POS       = 0x0B8C;      // float x, y, z
constexpr uintptr_t BULLET_VEL       = 0x0B98;      // float vx, vy   (verified probe_bullet_motion)
constexpr uintptr_t BULLET_SPEED     = 0x0BB0;      // float
constexpr uintptr_t BULLET_ACCEL     = 0x0BB4;      // float (0 unless a speed effect is active)
constexpr uintptr_t BULLET_ANGVEL    = 0x0BB8;      // float
constexpr uintptr_t BULLET_ANGLE     = 0x0BBC;      // float radians
constexpr uintptr_t BULLET_STATE     = 0x0BFC;      // uint16 (1/2/3/4/5 = live; 0 empty; 6 sentinel)
// live bullet_effects state (matches the ECL bullet_effects params):
constexpr uintptr_t BULLET_FX_P1     = 0x0C2C;      // float - redirect angle / accel
constexpr uintptr_t BULLET_FX_P2     = 0x0C30;      // float - redirect speed (-999 = keep)
constexpr uintptr_t BULLET_FX_INT    = 0x0C34;      // int32 - interval / duration
constexpr uintptr_t BULLET_FX_FLAG   = 0x0C3C;      // int32 - effect flag (16/32/64/128/256)

// --- zEnemyManager ---
constexpr uintptr_t EM_ENEMIES      = 0x00004F50;   // zEnemy[0x1E1]
constexpr uintptr_t EM_ENEMY_STRIDE = 0x00004F48;   // approx sizeof(zEnemy)
constexpr uintptr_t EM_ENEMY_COUNT  = 0x009545BC;   // int32
constexpr uintptr_t EM_BOSSES       = 0x00954598;   // zEnemy*[8]
constexpr uintptr_t ENEMY_POS       = 0x2B0C;       // zFloat3
constexpr uintptr_t ENEMY_LIFE      = 0x2BB8;       // int32
constexpr uintptr_t ENEMY_MAXLIFE   = 0x2BBC;       // int32

// --- zItemManager (static; verified live vs th-re-data) ---
// Items enemies drop on death (P/point/cherry...) + spellcard-capture bonuses.
// ItemManager::on_tick is called by BulletManager::on_tick (0x432990).
constexpr uintptr_t ITEM_MANAGER      = 0x00575C70;  // struct zItemManager
constexpr uintptr_t IM_ITEMS          = 0x00000000;  // zItem items[0x44C]
constexpr uintptr_t IM_ITEM_STRIDE    = 0x00000288;  // sizeof(zItem) = 648
constexpr uintptr_t IM_ITEM_MAX       = 0x44C;       // 1100 slots
constexpr uintptr_t IM_NEXT_INDEX     = 0x000AE2E8;  // int32 spawn cursor (wraps at 0x44C)
constexpr uintptr_t IM_ITEM_COUNT     = 0x000AE2EC;  // int32 live item count
//   within a zItem (stride 0x288):
constexpr uintptr_t ITEM_POS          = 0x24C;       // float x, y, z
constexpr uintptr_t ITEM_VEL          = 0x258;       // float vx, vy, vz
constexpr uintptr_t ITEM_TYPE         = 0x27C;       // uint8: 0 P-small 1 point
                                                     // 2 P-big 3 bomb 4 full-power
                                                     // 5 1up 6 star 7 cherry
                                                     // 8 cherry-petal 9 cherry-bullet
constexpr uintptr_t ITEM_IN_USE       = 0x27D;       // uint8 (slot active)
constexpr uintptr_t ITEM_STATE        = 0x27F;       // uint8 (0 falling; changes on auto-collect)

// --- zGui ---
constexpr uintptr_t GUI_BOSS_PRESENT = 0x24;  // BOOL
constexpr uintptr_t GUI_BOSS_HP_MAX  = 0x28;  // float (life-bar sprite size)
constexpr uintptr_t GUI_BOSS_HP_CUR  = 0x2C;  // float

// playfield extents (game-space units, origin top-left)
constexpr float PLAYFIELD_W = 384.0f;
constexpr float PLAYFIELD_H = 448.0f;

}  // namespace th07
