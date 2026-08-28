"""Gate A, step 1: prove we can read live game state out of th07.exe.

Run this while the game is running (in a stage). It attaches with pymem and
prints a state line ~20x/second. Move the character around and watch player_xy
change; fire into an enemy and watch its life bar; graze bullets and watch the
counter. If those numbers track what you see on screen, memory-reading is viable.

Usage:
    .venv\\Scripts\\python feasibility\\probe_memory.py
    .venv\\Scripts\\python feasibility\\probe_memory.py --once      # one snapshot
    .venv\\Scripts\\python feasibility\\probe_memory.py --bullets 5 # show N bullets
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

import pymem
import pymem.process

import th07_data as D


class GameMemory:
    def __init__(self) -> None:
        self.pm = self._attach()
        module = pymem.process.module_from_name(self.pm.process_handle, "th07.exe")
        if module is None:
            # fall back: assume default image base
            self.base = D.IMAGE_BASE
        else:
            self.base = module.lpBaseOfDll
        self.slide = self.base - D.IMAGE_BASE
        if self.slide:
            print(f"[!] module loaded at {self.base:#010x} (slide {self.slide:+#x})")

    @staticmethod
    def _attach() -> pymem.Pymem:
        last_err: Exception | None = None
        for name in D.PROCESS_NAMES:
            try:
                return pymem.Pymem(name)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        print(f"Could not attach to the game ({', '.join(D.PROCESS_NAMES)}).")
        print("Start the game first and get into a stage, then re-run.")
        if last_err:
            print(f"  ({last_err})")
        sys.exit(1)

    # --- typed reads (addresses are absolute VAs, rebased by slide) ---
    def u8(self, va: int) -> int:
        return self.pm.read_bytes(va + self.slide, 1)[0]

    def i16(self, va: int) -> int:
        return struct.unpack("<h", self.pm.read_bytes(va + self.slide, 2))[0]

    def u16(self, va: int) -> int:
        return struct.unpack("<H", self.pm.read_bytes(va + self.slide, 2))[0]

    def i32(self, va: int) -> int:
        return struct.unpack("<i", self.pm.read_bytes(va + self.slide, 4))[0]

    def u32(self, va: int) -> int:
        return struct.unpack("<I", self.pm.read_bytes(va + self.slide, 4))[0]

    def f32(self, va: int) -> float:
        return struct.unpack("<f", self.pm.read_bytes(va + self.slide, 4))[0]

    def block(self, va: int, size: int) -> bytes:
        return self.pm.read_bytes(va + self.slide, size)


def snapshot(g: GameMemory, n_bullets: int) -> dict:
    px = g.f32(D.PLAYER + D.PLAYER_POS_X)
    py = g.f32(D.PLAYER + D.PLAYER_POS_Y)
    state = g.u8(D.PLAYER + D.PLAYER_STATE)
    focus = g.u8(D.PLAYER + D.PLAYER_IS_FOCUS)

    # globals_ptr is a pointer stored in the target; it is already absolute in
    # the target address space, so read through it directly (no slide).
    globals_ptr = g.u32(D.GAME_MANAGER + D.GM_GLOBALS_PTR)
    score = graze = 0
    lives = bombs = power = 0.0
    if globals_ptr:
        gb = g.pm.read_bytes(globals_ptr, 0x100)
        score = struct.unpack_from("<i", gb, D.G_DISPLAYED_SCORE)[0]
        graze = struct.unpack_from("<i", gb, D.G_GRAZE)[0]
        lives = struct.unpack_from("<f", gb, D.G_LIFE_COUNT)[0]
        bombs = struct.unpack_from("<f", gb, D.G_BOMB_COUNT)[0]
        power = struct.unpack_from("<f", gb, D.G_POWER)[0]

    stage = g.i32(D.GAME_MANAGER + D.GM_STAGE)
    difficulty = g.i32(D.GAME_MANAGER + D.GM_DIFFICULTY)
    cherry = g.i32(D.GAME_MANAGER + D.GM_CHERRY)
    cherry_max = g.i32(D.GAME_MANAGER + D.GM_CHERRY_MAX)
    gamemode = g.u32(D.SUPERVISOR + D.SV_GAMEMODE)

    btn_cur = g.u16(D.INPUT_CUR)

    boss_present = g.i32(D.GUI + D.GUI_BOSS_PRESENT)
    # bar sprite value (animates, lags) - kept for comparison
    bar_cur = g.f32(D.GUI + D.GUI_BOSS_HP_CUR)
    bar_max = g.f32(D.GUI + D.GUI_BOSS_HP_MAX)
    bar_frac = (bar_cur / bar_max) if bar_max > 0 else 0.0
    # true HP from the boss enemy struct
    boss_life = boss_maxlife = 0
    boss_ptr = g.u32(D.ENEMY_MANAGER + D.EM_BOSSES)
    if boss_ptr:
        try:
            eb = g.pm.read_bytes(boss_ptr + D.ENEMY_LIFE, 8)
            boss_life, boss_maxlife = struct.unpack("<ii", eb)
        except pymem.exception.MemoryReadError:
            pass
    boss_frac = (boss_life / boss_maxlife) if boss_maxlife > 0 else 0.0

    # bullets: one big strided read of the whole array, parse in python
    raw = g.block(D.BULLET_MANAGER + D.BM_BULLETS,
                  D.BM_BULLET_COUNT_MAX * D.BM_BULLET_STRIDE)
    bm_count = g.i32(D.BULLET_MANAGER + D.BM_BULLET_COUNT)
    active = []
    state_hist: dict[int, int] = {}
    for i in range(D.BM_BULLET_COUNT_MAX):
        off = i * D.BM_BULLET_STRIDE
        st = struct.unpack_from("<H", raw, off + D.BULLET_STATE)[0]
        if st == 0:
            continue
        state_hist[st] = state_hist.get(st, 0) + 1
        if st not in D.BULLET_STATE_LIVE:
            continue
        bx, by = struct.unpack_from("<ff", raw, off + D.BULLET_POS)
        active.append((bx, by, st))
    nearest = min(
        (((bx - px) ** 2 + (by - py) ** 2) ** 0.5 for bx, by, _ in active),
        default=float("nan"),
    )
    active.sort(key=lambda b: (b[0] - px) ** 2 + (b[1] - py) ** 2)

    return {
        "gamemode": gamemode,
        "stage": stage,
        "difficulty": D.DIFFICULTY_NAMES.get(difficulty, difficulty),
        "player_xy": (px, py),
        "state": D.PLAYER_STATE_NAMES.get(state, state),
        "focus": bool(focus),
        "score": score,
        "lives": lives,
        "bombs": bombs,
        "power": power,
        "graze": graze,
        "cherry": f"{cherry}/{cherry_max}",
        "buttons": D.decode_buttons(btn_cur),
        "bullets_active": len(active),
        "bm_count": bm_count,
        "state_hist": state_hist,
        "nearest_bullet_dist": nearest,
        "boss": (f"{boss_life:>5}/{boss_maxlife:<5} bar={bar_frac*100:3.0f}%"
                 if boss_present else "-"),
        "nearest_bullets": active[:n_bullets],
    }


def fmt(s: dict) -> str:
    px, py = s["player_xy"]
    return (
        f"mode={s['gamemode']} st={s['stage']}({s['difficulty']}) "
        f"player=({px:6.1f},{py:6.1f}) {s['state']:<10} "
        f"{'FOCUS' if s['focus'] else '     '} "
        f"score={s['score']:>10} L={s['lives']:.0f} B={s['bombs']:.0f} "
        f"pow={s['power']:5.1f} graze={s['graze']:>4} cherry={s['cherry']:>9} "
        f"bul={s['bullets_active']:>4}(bm={s['bm_count']:>4}) "
        f"near={s['nearest_bullet_dist']:6.1f} "
        f"boss={s['boss']:>22} keys={s['buttons']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop automatically after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--hz", type=float, default=10.0, help="poll rate")
    ap.add_argument("--bullets", type=int, default=0,
                    help="also print the N nearest bullet coordinates")
    args = ap.parse_args()

    g = GameMemory()
    print(f"attached to pid {g.pm.process_id}, image base {g.base:#010x}")
    period = 1.0 / args.hz
    deadline = time.perf_counter() + args.seconds if args.seconds > 0 else None
    try:
        while True:
            t0 = time.perf_counter()
            if deadline is not None and t0 >= deadline:
                return
            try:
                s = snapshot(g, args.bullets)
            except pymem.exception.MemoryReadError:
                print("read failed (game closed or between scenes)...")
                time.sleep(0.5)
                continue
            line = fmt(s)
            if args.bullets:
                hist = ",".join(f"{k}:{v}" for k, v in sorted(s["state_hist"].items()))
                line += f"  states[{hist}]  " + " ".join(
                    f"({bx:.0f},{by:.0f}:{st})" for bx, by, st in s["nearest_bullets"]
                )
            print(line)
            if args.once:
                return
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
