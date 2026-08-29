"""Find the th07 enemy-bullet class + hitbox fields in the live zBullet struct
and tabulate the real collision size per class on Stage 1 Lunatic.

RESULT (see memory ref-th07-bullet-hitboxes): collision is an AABB-overlap test
in fn 0x43e260, using `zBullet + 0xB7C` (float2) as the bullet box, ±size/2.
Stage 1 classes (`+0xB8A`): 4/5 = pellet (size 4.0 -> hitbox half 2.0),
3 = ball (size 6.0 -> half 3.0). Player half-extent ~1.8 (class-3 kills at
4.8 px centre-to-centre head-on). Graze box = hitbox + 20 px.

    .venv\\Scripts\\python native\\probe_bullets.py --explore     # find the offsets
    .venv\\Scripts\\python native\\probe_bullets.py --collect \\
        --kind-off 0xC00 --size-off 0xC2C                         # tabulate per kind

Stage 1 has several distinct bullet types (fairy pellets, rice, the small
red/blue shots, aimed needles, plus Cirno's big blue/ice bullets at the
midboss). The policy (`runs_sim/ppo_v21/best.pt` by default) drives so we
survive far enough to see the midboss; on death the env resets to the stage-1
snapshot and we keep accumulating. Nothing is written to the game.

zBullet: base = BULLET_MANAGER(0x0062F958) + BM_BULLETS(0xB8C0), stride 0xD68.
Known: pos @ +0xB8C (float x,y,z), state @ +0xBFC (uint16, 1..5 = live).
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env  # noqa: E402

IMAGE_BASE = 0x00400000
BULLET_MANAGER = 0x0062F958
BM_BULLETS = 0x0000B8C0
BM_STRIDE = 0x00000D68
BM_MAX = 0x401                       # 1025 slots
B_POS = 0xB8C
B_STATE = 0xBFC
LIVE_STATES = (1, 2, 3, 4, 5)

# window of the struct we scan for the kind / hitbox fields (relative to slot base).
# the bullet "logic" block sits ~0xB40..0xC20; 0..0xB40 is the render AnmVm(s).
SCAN_LO = 0xB40
SCAN_HI = 0xC30


def _load_policy():
    try:
        from policy import MLPPolicy
        p = HERE.parent / "runs_sim" / "ppo_v21" / "best.pt"
        if not p.exists():
            for alt in sorted((HERE.parent / "runs_sim").glob("ppo_v*/best.pt")):
                p = alt
        pol = MLPPolicy.load(p)
        print(f"driving with {p}")
        return pol.act
    except Exception as e:
        print(f"no policy ({e}) - driving with a reactive escape heuristic")
        from obs import HEAD_DIM, NDIRS  # noqa

        def act(o):
            esc = o[HEAD_DIM:HEAD_DIM + NDIRS]
            return int(np.argmax(esc)) + 9      # + focus

        return act


def _read_bullets(pm):
    base = BULLET_MANAGER + BM_BULLETS
    blob = pm.read_bytes(base, BM_STRIDE * BM_MAX)
    live = []
    for i in range(BM_MAX):
        o = i * BM_STRIDE
        st = struct.unpack_from("<H", blob, o + B_STATE)[0]
        if st not in LIVE_STATES:
            continue
        x, y = struct.unpack_from("<ff", blob, o + B_POS)
        if not (-64 < x < 448 and -64 < y < 512):
            continue
        live.append((i, st, x, y, blob[o:o + SCAN_HI]))
    return live


# --------------------------------------------------------------------- explore
def explore(env, act, steps):
    pm = env._pm
    offs = list(range(SCAN_LO, SCAN_HI, 4))
    i32 = {o: Counter() for o in offs}
    f32 = {o: Counter() for o in offs}
    i16 = {o: Counter() for o in range(SCAN_LO, SCAN_HI, 2)}
    n_bullets = 0
    raw_samples = []

    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    for step in range(steps):
        try:
            obs, _, term, trunc, info = env.step(act(obs))
        except Exception:
            break
        try:
            live = _read_bullets(pm)
        except Exception as e:
            print(f"  read fail @{step}: {e}")
            continue
        for (idx, st, x, y, buf) in live:
            n_bullets += 1
            for o in offs:
                v = struct.unpack_from("<i", buf, o)[0]
                fv = struct.unpack_from("<f", buf, o)[0]
                if -1 << 20 < v < 1 << 20:
                    i32[o][v] += 1
                if np.isfinite(fv) and 0.0 <= fv < 200.0:
                    f32[o][round(fv, 2)] += 1
            for o in range(SCAN_LO, SCAN_HI, 2):
                i16[o][struct.unpack_from("<h", buf, o)[0]] += 1
            if len(raw_samples) < 12 and st in (2, 3):
                raw_samples.append((step, idx, st, round(x, 1), round(y, 1), buf))
        if step % 200 == 0:
            print(f"  step {step:4d}  frame {info.get('frame')}  live={len(live)}  "
                  f"total seen={n_bullets}")
        if term:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r

    print(f"\n=== scanned {n_bullets} live-bullet observations ===")
    print("\n-- int32 offsets with a SMALL discrete value set (kind candidates) --")
    for o in offs:
        c = i32[o]
        if 1 < len(c) <= 24 and min(c) >= 0 and max(c) < 4096:
            top = sorted(c.items())
            print(f"  +0x{o:03X}  {len(c)} values  {top[:16]}")
    print("\n-- float32 offsets with a small discrete set in (0.5, 40) (hitbox candidates) --")
    for o in offs:
        c = Counter({k: v for k, v in f32[o].items() if 0.5 <= k <= 40.0})
        if c and len(c) <= 16 and sum(c.values()) > 0.4 * n_bullets:
            print(f"  +0x{o:03X}  {len(c)} values  {sorted(c.items())}")
    print("\n-- int16 offsets with a small discrete set (kind / color / sprite) --")
    for o in range(SCAN_LO, SCAN_HI, 2):
        c = i16[o]
        if 1 < len(c) <= 24 and min(c) >= 0 and max(c) < 4096:
            print(f"  +0x{o:03X}  {len(c)} values  {sorted(c.items())[:16]}")

    print("\n-- raw struct dumps of a few normal (state 2/3) bullets --")
    print("   offset :  int32        float32      int16,int16")
    for (step, idx, st, x, y, buf) in raw_samples:
        print(f"\n  slot {idx} state {st} pos ({x},{y})")
        for o in range(SCAN_LO, SCAN_HI, 4):
            iv = struct.unpack_from("<i", buf, o)[0]
            fv = struct.unpack_from("<f", buf, o)[0]
            a, b = struct.unpack_from("<hh", buf, o)
            fs = f"{fv:12.4f}" if abs(fv) < 1e6 else f"{fv:12.2e}"
            print(f"   +0x{o:03X} : {iv:11d} {fs}   {a:6d} {b:6d}")


# --------------------------------------------------------------------- collect
def collect(env, act, steps, kind_off, size_off, size_is_i16):
    pm = env._pm
    per_kind_bx = defaultdict(Counter)     # hitbox x @ size_off
    per_kind_by = defaultdict(Counter)     # hitbox y @ size_off+4
    per_kind_n = Counter()
    per_kind_state = defaultdict(Counter)
    per_kind_c02 = defaultdict(Counter)    # +0xBF6  (anm sprite / color candidate)
    per_kind_c04 = defaultdict(Counter)    # +0xBF8
    per_kind_pos = defaultdict(list)
    per_kind_dump = {}

    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    for step in range(steps):
        try:
            obs, _, term, trunc, info = env.step(act(obs))
            live = _read_bullets(pm)
        except Exception:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r
            continue
        for (idx, st, x, y, buf) in live:
            kind = struct.unpack_from("<h" if kind_off_i16 else "<i", buf, kind_off)[0]
            bx = struct.unpack_from("<h", buf, size_off)[0] if size_is_i16 else \
                round(struct.unpack_from("<f", buf, size_off)[0], 3)
            by = struct.unpack_from("<h", buf, size_off + 4)[0] if size_is_i16 else \
                round(struct.unpack_from("<f", buf, size_off + 4)[0], 3)
            per_kind_bx[kind][bx] += 1
            per_kind_by[kind][by] += 1
            per_kind_n[kind] += 1
            per_kind_state[kind][st] += 1
            per_kind_c02[kind][struct.unpack_from("<H", buf, 0xBF6)[0]] += 1
            per_kind_c04[kind][struct.unpack_from("<H", buf, 0xBF8)[0]] += 1
            per_kind_pos[kind].append((x, y))
            if (kind, st) not in per_kind_dump and st == 2:
                per_kind_dump[(kind, st)] = (step, idx, round(x, 1), round(y, 1), bytes(buf))
        if step % 200 == 0:
            print(f"  step {step:4d}  frame {info.get('frame')}  kinds={dict(per_kind_n)}")
        if term:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r

    print("\n=== per-kind enemy-bullet hitbox - Stage 1 Lunatic ===")
    print(f"kind @ +0x{kind_off:X} ({'i16' if kind_off_i16 else 'i32'}),  "
          f"box @ +0x{size_off:X}/+0x{size_off+4:X} ({'i16' if size_is_i16 else 'float'})\n")
    for k in sorted(per_kind_n, key=lambda k: -per_kind_n[k]):
        xs = np.array(per_kind_pos[k])
        st = ",".join(f"s{s}:{n}" for s, n in per_kind_state[k].most_common())
        print(f"kind {k:4d}   n={per_kind_n[k]:7d}   {st}")
        print(f"   hitbox x: {dict(per_kind_bx[k].most_common(5))}")
        print(f"   hitbox y: {dict(per_kind_by[k].most_common(5))}")
        print(f"   +0xBF6 : {dict(per_kind_c02[k].most_common(5))}")
        print(f"   +0xBF8 : {dict(per_kind_c04[k].most_common(5))}")
        print(f"   seen x[{xs[:,0].min():.0f},{xs[:,0].max():.0f}] y[{xs[:,1].min():.0f},{xs[:,1].max():.0f}]")

    print("\n=== full struct window per (kind, state=2) ===")
    for (k, st), (step, idx, x, y, buf) in sorted(per_kind_dump.items()):
        print(f"\n  kind {k} state {st}  slot {idx}  pos ({x},{y})  step {step}")
        for o in range(SCAN_LO, SCAN_HI, 4):
            iv = struct.unpack_from("<i", buf, o)[0]
            fv = struct.unpack_from("<f", buf, o)[0]
            a, b = struct.unpack_from("<hh", buf, o)
            fs = f"{fv:11.4f}" if abs(fv) < 1e6 else f"{fv:11.2e}"
            print(f"   +0x{o:03X}: {iv:11d} {fs}  {a:6d} {b:6d}")


PLAYER = 0x004BDAD8
P_POS_X = 0x930          # zPlayer pos (float x @ +0x930, y @ +0x934)
P_STATE = 0x2408         # uint8: 0 alive, 1 respawning, 2 dead, 3 invuln, 4 border


def _nearest(pm, buf_live, px, py):
    best = None
    for (idx, st, x, y, buf) in buf_live:
        d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if best is None or d < best[2]:
            box = struct.unpack_from("<f", buf, 0xB7C)[0]
            kind = struct.unpack_from("<h", buf, 0xB8A)[0]
            best = (kind, box, d)
    return best


# (dx,dy) sign -> sim/env dir index (see _DIRS): 0 none,1 U,2 UR,3 R,4 DR,5 D,6 DL,7 L,8 UL
_DIR_IX = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4,
           (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}


def deathdist(env, act, steps):
    """Dodge with the policy for a random interval, then walk SLOWLY (focused,
    1.6 px/f) straight at the nearest bullet until it kills us. On the death
    frame, log the previous frame's nearest bullet: (0xB8A class, 0xB7C size,
    centre-to-centre distance). min distance over many samples ~= the true
    head-on lethal distance = player_hitbox + bullet_hitbox."""
    import random
    pm = env._pm
    hits = []
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    settle, dodge_for, prev_alive, prev = 0, random.randint(30, 300), True, None
    for step in range(steps):
        try:
            px, py = struct.unpack_from("<ff", pm.read_bytes(PLAYER + P_POS_X, 8), 0)
            pstate = pm.read_bytes(PLAYER + P_STATE, 1)[0]
            live = _read_bullets(pm)
        except Exception:
            live, pstate, px, py = [], 0, 192, 380
        near = _nearest(pm, live, px, py)
        if settle < dodge_for or near is None:
            a = act(obs)
        else:                                    # creep at the nearest bullet, focused
            bx, by = None, None
            for (idx, st, x, y, buf) in live:
                if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 == near[2]:
                    bx, by = x, y
            dx = (bx > px + 0.5) - (bx < px - 0.5)
            dy = (by > py + 0.5) - (by < py - 0.5)
            a = _DIR_IX.get((dx, dy), 0) + 9     # + focus (slow)
        try:
            obs, _, term, trunc, info = env.step(a)
        except Exception:
            break
        settle += 1
        alive = pstate in (0, 3, 4)
        if prev_alive and not alive and prev and settle > dodge_for + 1 and prev[2] < 20:
            hits.append(prev)
        prev_alive, prev = alive, near
        if term:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r
            settle, dodge_for, prev_alive, prev = 0, random.randint(30, 300), True, None
        if step % 500 == 0:
            print(f"  step {step}  frame {info.get('frame')}  deaths={len(hits)}")

    print(f"\n=== point-blank death distance, still player (n={len(hits)}) ===")
    print("  centre-to-centre px on the death frame = player_hitbox + bullet_hitbox\n")
    by = defaultdict(list)
    for cls, spr, dist in hits:
        by[(cls, spr)].append(dist)
    for (cls, spr) in sorted(by):
        ds = np.array(by[(cls, spr)])
        print(f"  class {cls}  sprite-size {spr:.1f}   n={len(ds)}   "
              f"lethal dist: min {ds.min():.2f} / p10 {np.percentile(ds,10):.2f} / "
              f"median {np.median(ds):.2f} / p90 {np.percentile(ds,90):.2f}")
    allds = np.array([d for _, _, d in hits])
    if len(allds):
        print(f"\n  ALL: min {allds.min():.2f}  median {np.median(allds):.2f}  "
              f"(min ~= the true head-on lethal distance)")


kind_off_i16 = True    # set by main from args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--deathdist", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--kind-off", type=lambda s: int(s, 0), default=0xC00)
    ap.add_argument("--kind-i32", action="store_true", help="kind field is int32 not int16")
    ap.add_argument("--size-off", type=lambda s: int(s, 0), default=0xC2C)
    ap.add_argument("--size-i16", action="store_true", help="size field is int16 not float")
    args = ap.parse_args()

    global kind_off_i16
    kind_off_i16 = not args.kind_i32

    env = Th07Env(frame_skip=args.frame_skip, max_seconds=600)
    if getattr(env, "_pm", None) is None:
        import pymem
        env._pm = pymem.Pymem()
        env._pm.open_process_from_id(env.pid)
    act = _load_policy()
    try:
        if args.deathdist:
            deathdist(env, act, args.steps)
        elif args.collect:
            collect(env, act, args.steps, args.kind_off, args.size_off, args.size_i16)
        else:
            explore(env, act, args.steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
