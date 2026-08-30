"""Check whether the stage-1 midboss (Cirno) shows up in EM_ENEMIES / the enemy
count / EM_BOSSES during her fight (~frame 2450). Drives with a policy to get
there. Nothing is written to the game.

    .venv\\Scripts\\python native\\probe_cirno.py [runs_sim/ppo_vN/best.pt]
"""
from __future__ import annotations
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import Th07Env          # noqa: E402
import shm as S                  # noqa: E402

ENEMY_MANAGER = 0x009A9B00
EM_ENEMIES = 0x00004F50
EM_ENEMY_STRIDE = 0x00004F48
EM_ENEMY_COUNT = 0x009545BC
EM_BOSSES = 0x00954598       # zEnemy* bosses[8]
ENEMY_POS = 0x00              # verify: env reads e+ENEMY_POS ; use th07_addrs
GUI = 0x0049FBF0


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "runs_sim/ppo_v25/best.pt"
    env = Th07Env(frame_skip=3, max_seconds=300)
    pm = env._pm
    try:
        from policy import MLPPolicy
        pol = MLPPolicy.load(model)
        act = pol.act
        print(f"driving with {model}")
    except Exception as e:
        print(f"no policy ({e}); driving with a dodge heuristic")
        from obs import HEAD_DIM, NDIRS
        import numpy as np
        act = lambda o: int(np.argmax(o[HEAD_DIM:HEAD_DIM + NDIRS])) + 9

    # figure out ENEMY_POS / LIFE / MAXLIFE offsets from th07_addrs.h
    hdr = (HERE / "th07_addrs.h").read_text()
    import re
    def off(name, default):
        m = re.search(rf"{name}\s*=\s*(0x[0-9A-Fa-f]+)", hdr)
        return int(m.group(1), 16) if m else default
    E_POS = off("ENEMY_POS", 0x0)
    E_LIFE = off("ENEMY_LIFE", 0x0)
    E_ML = off("ENEMY_MAXLIFE", 0x0)
    print(f"offsets: pos={E_POS:#x} life={E_LIFE:#x} maxlife={E_ML:#x}  stride={EM_ENEMY_STRIDE:#x}")

    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    seen_big_in_arr = 0
    seen_big_frames = []
    boss_ptr_nonzero = 0
    boss_samples = []
    for step in range(6000):
        obs, _, term, trunc, info = env.step(act(obs))
        fr = info.get("frame", 0)
        try:
            ec = struct.unpack_from("<i", pm.read_bytes(ENEMY_MANAGER + EM_ENEMY_COUNT, 4), 0)[0]
            ec = max(0, min(ec, 200))
            blob = pm.read_bytes(ENEMY_MANAGER + EM_ENEMIES, EM_ENEMY_STRIDE * min(ec + 2, 40))
            bosses = pm.read_bytes(ENEMY_MANAGER + EM_BOSSES, 8 * 4)
        except Exception:
            continue
        bptrs = [struct.unpack_from("<I", bosses, k * 4)[0] for k in range(8)]
        if any(bptrs):
            boss_ptr_nonzero += 1
            # dereference EM_BOSSES[0] and read pos/life at the enemy struct offsets
            bp0 = bptrs[0]
            if bp0 and len(boss_samples) < 30:
                try:
                    bs = pm.read_bytes(bp0, 0x2C00)
                    bx, by = struct.unpack_from("<ff", bs, E_POS)
                    bl = struct.unpack_from("<i", bs, E_LIFE)[0]
                    bml = struct.unpack_from("<i", bs, E_ML)[0]
                    boss_samples.append((fr, round(bx), round(by), bl, bml))
                except Exception as ex:
                    boss_samples.append((fr, "deref-fail", str(ex)))
        big_here = []
        for i in range(min(ec + 2, 40)):
            b = i * EM_ENEMY_STRIDE
            try:
                ex, ey = struct.unpack_from("<ff", blob, b + E_POS)
                life = struct.unpack_from("<i", blob, b + E_LIFE)[0]
                ml = struct.unpack_from("<i", blob, b + E_ML)[0]
            except Exception:
                continue
            if 0 < ml < 100000 and ml >= 100 and -50 < ey < 500:
                big_here.append((i, ml, life, round(ex, 0), round(ey, 0)))
        if big_here:
            seen_big_in_arr += 1
            if len(seen_big_frames) < 8:
                seen_big_frames.append((fr, ec, big_here, [f"{p:#x}" for p in bptrs if p]))
        if step % 150 == 0:
            print(f"step {step:4d} frame {fr:5d}  enemy_count={ec:3d}  "
                  f"boss_ptrs={[hex(p) for p in bptrs if p]}  big_in_array={len(big_here)}")
        if term:
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r

    print(f"\n=== result ===")
    print(f"frames with a maxlife>=100 enemy in EM_ENEMIES: {seen_big_in_arr}")
    print(f"frames with a non-null EM_BOSSES pointer:        {boss_ptr_nonzero}")
    print(f"EM_BOSSES[0] deref samples (frame, x, y, life, maxlife):")
    for smp in boss_samples:
        print(f"   {smp}")
    for (fr, ec, big, bp) in seen_big_frames:
        print(f"  frame {fr}: enemy_count={ec}  big=[(idx,maxlife,life,x,y)...] {big}  boss_ptrs={bp}")
    env.close()


if __name__ == "__main__":
    main()
