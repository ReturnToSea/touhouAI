"""Static addresses and struct offsets for Touhou 7 (PCB), version 1.00b (original).

Source: https://github.com/exphp-share/th-re-data  (data/th07.v1.00b)
Same reverse-engineering data that thprac uses. Verified against thprac's
thprac_th07.h struct definitions.

th07.exe is a 2003 32-bit PE with no ASLR, so it always loads at its preferred
image base 0x400000 and these virtual addresses are usable directly. We still
rebase through the module base at runtime in case that ever stops being true.
"""

IMAGE_BASE = 0x00400000
PROCESS_NAMES = ("th07.exe", "th07e.exe")
WINDOW_TITLE_SUBSTRINGS = ("Perfect Cherry Blossom", "th07", "魔郷夢")

# --- Static objects (absolute VAs, i.e. already include IMAGE_BASE) ---
GUI            = 0x0049FBF0   # struct zGui
PLAYER         = 0x004BDAD8   # struct zPlayer
SUPERVISOR     = 0x00575950   # struct zSupervisor
GAME_MANAGER   = 0x00626270   # struct zGameManager
BULLET_MANAGER = 0x0062F958   # struct zBulletManager
ENEMY_MANAGER  = 0x009A9B00   # struct zEnemyManager

INPUT_CUR      = 0x004B9E4C   # WORD  - buttons held this frame
INPUT_PREV     = 0x004B9E54   # WORD  - buttons held last frame
STAGE_NUM      = 0x01347FC8   # int32 - global stage counter

# --- zPlayer (base = PLAYER) ---
PLAYER_POS_X   = 0x0930       # float   (y at +0x934, z at +0x938)
PLAYER_POS_Y   = 0x0934
PLAYER_STATE   = 0x2408       # uint8: 0 alive,1 respawning,2 dead,3 invuln,4 border
PLAYER_ORBSTATE = 0x240A      # int8
PLAYER_IS_FOCUS = 0x240B      # int8  (0/1)

PLAYER_STATE_NAMES = {0: "alive", 1: "respawning", 2: "dead", 3: "invuln", 4: "border"}

# --- zGameManager (base = GAME_MANAGER) ---
GM_GLOBALS_PTR = 0x0008       # zGlobals*
GM_DIFFICULTY  = 0x0010       # int32: 0 Easy .. 4 Extra/Phantasm
GM_STAGE       = 0x95EC       # int32
GM_CHERRY_MAX  = 0x9618       # int32
GM_CHERRY      = 0x961C       # int32
GM_CHERRY_PLUS = 0x9620       # int32

DIFFICULTY_NAMES = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Lunatic", 4: "Extra"}

# --- zGlobals (base = *(GAME_MANAGER + GM_GLOBALS_PTR)) ---
G_DISPLAYED_SCORE = 0x00      # int32  (this is the on-screen score value)
G_TRUE_SCORE      = 0x04      # int32
G_GRAZE           = 0x18      # int32
G_POINT_ITEMS_STAGE = 0x24    # int32
G_POINT_ITEMS_GAME  = 0x28    # int32
G_LIFE_COUNT      = 0x5C      # float  (spare lives)
G_BOMB_COUNT      = 0x68      # float  (spare bombs)
G_POWER           = 0x7C      # float  (0..128)

# --- zBulletManager (base = BULLET_MANAGER) ---
BM_BULLETS       = 0x0000B8C0   # zBullet bullets[0x401]
BM_BULLET_STRIDE = 0x00000D68   # sizeof(zBullet)
BM_BULLET_COUNT_MAX = 0x401     # 1025
BM_BULLET_COUNT  = 0x0037A128   # int32 live bullet count (matches our filtered count)
#   within a zBullet:
BULLET_POS   = 0x0B8C   # zFloat2 (x, y)
BULLET_STATE = 0x0BFC   # uint16  state: 0 empty, 1 active, 2 slow/special,
                        #                3 spawning, 6 sentinel slot at (0,0)
BULLET_STATE_LIVE = (1, 2, 3, 4, 5)   # values that denote a real on-field bullet

# --- zEnemyManager (base = ENEMY_MANAGER) ---
EM_ENEMIES        = 0x00004F50   # zEnemy enemies[0x1E1]
EM_ENEMY_STRIDE   = 0x00004F48   # sizeof(zEnemy) (approx; verify empirically)
EM_ENEMY_COUNT    = 0x009545BC   # int32
EM_BOSSES         = 0x00954598   # zEnemy* bosses[8]  (bosses[0] pointer lives here)
#   within a zEnemy:
ENEMY_POS      = 0x2B0C   # zFloat3
ENEMY_LIFE     = 0x2BB8   # int32  current HP of this enemy
ENEMY_MAXLIFE  = 0x2BBC   # int32  HP at start of the current life segment

# --- zGui (base = GUI) ---
GUI_IMPL_PTR      = 0x08     # zGuiImpl*
GUI_BOSS_PRESENT  = 0x24     # BOOL
GUI_BOSS_HP_MAX   = 0x28     # float  (life bar max size)
GUI_BOSS_HP_CUR   = 0x2C     # float  (life bar current size)

# --- zSupervisor (base = SUPERVISOR) ---
SV_GAMEMODE       = 0x154    # uint32  (thprac treats 2 as "in a run")

# --- input bitfield (INPUT_CUR / INPUT_PREV) ---
# Confirmed against the game's own INPUT_CUR word via SendInput scancodes:
# left=0x40, right=0x80, down=0x20, shoot(Z)=0x01, slow(shift)=0x04.
BTN_SHOOT = 0x01
BTN_BOMB  = 0x02
BTN_SLOW  = 0x04   # focus / shift
BTN_SKIP  = 0x08
BTN_UP    = 0x10
BTN_DOWN  = 0x20
BTN_LEFT  = 0x40
BTN_RIGHT = 0x80

BTN_NAMES = [
    (BTN_SHOOT, "Z"), (BTN_BOMB, "X"), (BTN_SLOW, "shift"), (BTN_SKIP, "skip"),
    (BTN_UP, "up"), (BTN_DOWN, "down"), (BTN_LEFT, "left"), (BTN_RIGHT, "right"),
]

def decode_buttons(bits: int) -> str:
    return "+".join(name for mask, name in BTN_NAMES if bits & mask) or "-"


# Playfield is ~ x in [0, 384], y in [0, 448] in these game-space units,
# origin top-left. Player spawn / respawn point is exactly (192, 384).
PLAYFIELD_W = 384.0
PLAYFIELD_H = 448.0
PLAYER_SPAWN = (192.0, 384.0)

# --- observed episode semantics (from feasibility probes) ---
# PLAYER_STATE goes: alive -> dead (2-3 frames) -> respawning (2 frames, snapped
# to PLAYER_SPAWN) -> invuln (~3 s) -> alive.
# Game over: a death at lives == 0 lands on the Continue screen, which FREEZES
# all game logic with state == respawning and lives == 0 until the player picks
# continue/quit. So: episode_over  <=>  lives == 0 and state == respawning.
# SUPERVISOR gamemode stays 2 during a run AND on the frozen Continue screen;
# the title/other menus have not been probed yet.
