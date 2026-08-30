"""Run a sim-trained policy on the REAL game and, on every death, report which
bullet killed the player: its class (zBullet+0xB8A), collision box (zBullet+0xB7C,
AABB half-extent = box/2), centre-to-centre distance, bearing, and speed - plus
whether the stage midboss/boss (Cirno/Letty) was on screen and where the player
was in the field.

    .venv\\Scripts\\python native\\probe_deathcam.py runs_sim\\ppo_v26\\best.pt
    .venv\\Scripts\\python native\\probe_deathcam.py runs_sim\\ppo_v26\\best.pt --deaths 6 --watch

Uses the Python step loop (env._obs -> shared native/obs.py), same as sim/transfer.py.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env            # noqa: E402
from policy import MLPPolicy       # noqa: E402

# --- live bullet array (from probe_bullets.py) --------------------------------
BULLET_MANAGER = 0x0062F958
BM_BULLETS = 0x0000B8C0
BM_STRIDE = 0x00000D68
BM_MAX = 0x401
B_POS = 0xB8C
B_STATE = 0xBFC
B_BOX = 0xB7C          # float: AABB full size; collision uses +-box/2
B_KIND = 0xB8A         # int16: bullet class (stage 1: 3=ball, 4/5=pellet)
LIVE_STATES = (1, 2, 3, 4, 5)

KIND_NAME = {3: "ball", 4: "pellet", 5: "pellet", 2: "kunai", 6: "star", 7: "big"}


def read_bullets(pm):
    """-> {slot: (x, y, kind, box)} for every live on-field bullet."""
    blob = pm.read_bytes(BULLET_MANAGER + BM_BULLETS, BM_STRIDE * BM_MAX)
    out = {}
    for i in range(BM_MAX):
        o = i * BM_STRIDE
        st = struct.unpack_from("<H", blob, o + B_STATE)[0]
        if st not in LIVE_STATES:
            continue
        x, y = struct.unpack_from("<ff", blob, o + B_POS)
        if not (-64 < x < 448 and -64 < y < 512):
            continue
        box = struct.unpack_from("<f", blob, o + B_BOX)[0]
        kind = struct.unpack_from("<h", blob, o + B_KIND)[0]
        out[i] = (x, y, kind, box)
    return out


def bearing(dx, dy):
    """Compass-ish label for a vector from player to bullet (screen y is down)."""
    ang = math.degrees(math.atan2(-dy, dx))  # 0=right, 90=up
    for lo, hi, name in [(-22.5, 22.5, "right"), (22.5, 67.5, "up-right"),
                         (67.5, 112.5, "above"), (112.5, 157.5, "up-left"),
                         (157.5, 180, "left"), (-180, -157.5, "left"),
                         (-157.5, -112.5, "down-left"), (-112.5, -67.5, "below"),
                         (-67.5, -22.5, "down-right")]:
        if lo <= ang < hi:
            return name
    return "?"


def field_zone(px, py):
    xz = "L" if px < 140 else ("R" if px > 244 else "C")
    yz = "top" if py < 160 else ("bottom" if py > 320 else "mid")
    return f"{yz}-{xz}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--deaths", type=int, default=5, help="stop after this many deaths")
    ap.add_argument("--frame-skip", type=int, default=3)
    ap.add_argument("--watch", action="store_true", help="render + launch viz.py")
    ap.add_argument("--max-seconds", type=float, default=36000.0)
    args = ap.parse_args()

    pol = MLPPolicy.load(args.model)
    print(f"loaded {args.model}  hidden={pol.hidden}  params={pol.n_params()}")
    dt = args.frame_skip / 60.0

    deaths = []                       # list of dicts
    kind_hist = Counter()
    # env.reset() only cleanly rewinds the stage-1 snapshot; once the policy
    # survives deep into the boss fight a reset leaves the game half-restored
    # (Letty frozen, RNG drifted) and episodes 2+ are garbage. So RELAUNCH the
    # game for every death - one clean episode per process lifetime.
    for ep in range(args.deaths):
        env = Th07Env(frame_skip=args.frame_skip, max_seconds=args.max_seconds,
                      render=args.watch)
        pm = env._pm
        viz = None
        if args.watch:
            import subprocess
            viz = subprocess.Popen([sys.executable, str(HERE / "viz.py"), str(env.pid)])
        try:
            obs, _ = env.reset()
            ring = deque(maxlen=8)   # (t_s, px, py, {slot:(x,y,kind,box)}, boss)
            steps = 0
            done = False
            while not done:
                t0 = time.perf_counter()
                # snapshot BEFORE stepping (the killer may despawn on the hit frame)
                s = env.h.s
                px, py = s.player_x, s.player_y
                try:
                    bl = read_bullets(pm)
                except Exception:
                    bl = {}
                boss = env._boss()
                ring.append((steps * dt, px, py, bl, boss))

                a = int(pol.act(obs))
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                done = term or trunc
                if args.watch:
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

            # episode ended - was it a death (lives drop) or stage/game end?
            s = env.h.s
            died = s.tick_status == 0
            t_ep = steps * dt
            if not died:
                print(f"ep {ep}: ended at {t_ep:.1f}s via tick_status={s.tick_status} "
                      f"(stage/game end, not a death) - score {info['score']}")
                continue

            # find the closest bullet approach across the last few pre-step frames
            best = None   # dict
            for k, (ts, bx, by, bl, boss) in enumerate(ring):
                prev_bl = ring[k - 1][3] if k > 0 else {}
                for slot, (x, y, kind, box) in bl.items():
                    d = math.hypot(x - bx, y - by)
                    if best is None or d < best["dist"]:
                        spd = None
                        if slot in prev_bl:
                            spd = math.hypot(x - prev_bl[slot][0],
                                             y - prev_bl[slot][1]) / args.frame_skip
                        best = dict(dist=d, kind=kind, box=box, dx=x - bx,
                                    dy=y - by, t_s=ts, speed=spd)
            last = ring[-1]
            rec = dict(ep=ep, t=t_ep, px=last[1], py=last[2],
                       zone=field_zone(last[1], last[2]),
                       boss=last[4], score=info["score"])
            if best is not None:
                rec.update(dist=best["dist"], kind=best["kind"],
                           kname=KIND_NAME.get(best["kind"], f"k{best['kind']}"),
                           box=best["box"], half=best["box"] / 2,
                           bearing=bearing(best["dx"], best["dy"]),
                           approach_dt=t_ep - best["t_s"], speed=best["speed"])
                kind_hist[rec["kname"]] += 1
            deaths.append(rec)

            b = rec.get("boss")
            bstr = (f"boss@({b[0]:.0f},{b[1]:.0f}) hp {b[2]}/{b[3]}"
                    if b else "no boss on screen")
            if "dist" in rec:
                print(f"ep {ep} DEATH @ {rec['t']:.1f}s  score {rec['score']}\n"
                      f"    player ({rec['px']:.0f},{rec['py']:.0f}) [{rec['zone']}]  {bstr}\n"
                      f"    killed by: class {rec['kind']} ({rec['kname']})  "
                      f"box {rec['box']:.1f} (half {rec['half']:.2f}px)  "
                      f"dist {rec['dist']:.2f}px  from {rec['bearing']}  "
                      + (f"speed ~{rec['speed']:.1f}px/f  " if rec.get('speed') else "")
                      + f"({rec['approach_dt']*1000:.0f}ms before death)")
            else:
                print(f"ep {ep} DEATH @ {rec['t']:.1f}s  score {rec['score']}  "
                      f"(no bullet near the player - body/laser/graze?)  {bstr}")
        finally:
            env.close()
            if viz is not None:
                viz.terminate()

    print("\n=== deaths ===")
    for r in deaths:
        k = r.get("kname", "?")
        print(f"  {r['t']:6.1f}s  {r['zone']:9s}  "
              f"{'boss' if r.get('boss') else 'nobo':4s}  "
              + (f"{k:6s} half {r.get('half',0):.2f}px  d {r.get('dist',0):.2f}  "
                 f"from {r.get('bearing','?')}" if 'dist' in r else "no bullet"))
    print(f"\nkiller-class histogram: {dict(kind_hist)}")
    print("real th07 stage-1: class 3 = ball (box 6.0 -> half 3.0), "
          "class 4/5 = pellet (box 4.0 -> half 2.0); player half ~1.8")
    print("sim uses circular dist < half + 1.8 ; real is an AABB (square) - "
          "up to ~40% bigger on pure diagonals")


if __name__ == "__main__":
    main()
