"""Part 12 — ECL-VM danmaku source for FightSim.

Runs Letty's real Stage-1 boss script (`sim/ecl/vm.py`) with a fresh RNG seed,
propagates every spawn with the engine-faithful `bullet_sim`, and emits the
*same* dense `[F, POOL, 2]` arrays FightSim's recordings use — so a schedule
drops straight into `FightSim(recs=...)`.

Why this beats the recordings: each schedule is a fresh RNG roll of the actual
boss bytecode, not one of ~20 fixed trajectories the policy can memorise. Boss
repositioning, bullet-spread jitter and per-orb RNG branches all reroll; aimed
bullets are *generated* from `(spawn, angle, speed)`.

    from sim.danmaku_ecl import build_schedules
    recs = build_schedules(40)          # ~a few seconds each
    sim  = FightSim(recs=recs, name="letty")

    python -m sim.danmaku_ecl           # verify: schedule sanity + a FightSim smoke test
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> sim.ecl
from sim.ecl.parser import parse_file            # noqa: E402
from sim.ecl.vm import VM                        # noqa: E402
from sim.ecl.bullet_sim import (batch_from_spawns, simulate_batch,   # noqa: E402
                                FX_OFFSCREEN_GRACE_MASK)

_ECL = Path(__file__).resolve().parents[1] / "tools" / "th07_ecl" / "ecldata1.ecl"

POOL = 1025
MAX_EN = 48
MAXLIFE = 420                 # frames to propagate a bullet (LC snow lives ~435)
BULLET_HB = 2.5               # uniform; the recordings carry per-type boxes
ENEMY_BODY_SCALE = 2.0 / 3.0
CULL_MARGIN = 24.0

# Letty's real per-phase HP budget (Part 7): NS1 15000 -> life_callback 1700,
# LC inherits ~1700, NS2 15000 -> 2000, TT inherits ~2000.
LETTY_PHASE_HP = (13300.0, 1700.0, 13000.0, 2000.0)


def _run_vm(seed: int, difficulty: int = 3, frames: int = 13000):
    """VM run capturing boss + lethal-orb positions per frame."""
    ecl = parse_file(str(_ECL))
    vm = VM(ecl, difficulty=difficulty, seed=seed)
    vm.start_boss(sub=31, interrupt=0)
    boss_xy: list[tuple[float, float]] = []
    orbs: list[list[tuple[float, float, float]]] = []
    while vm.frame < frames and any(e.alive for e in vm.enemies):
        vm.step()
        b = vm.boss()
        boss_xy.append((b.x, b.y) if b else (192.0, 112.0))
        fo = [(e.x, e.y, max(e.hitbox) * 0.5 * ENEMY_BODY_SCALE)
              for e in vm.enemies
              if not e.is_boss and e.alive and e.hitbox[0] > 0.5]
        orbs.append(fo[:MAX_EN])
    return vm, np.asarray(boss_xy, np.float32), orbs


def _assign_slots(frames: np.ndarray, lifes: np.ndarray) -> np.ndarray:
    """Give each bullet a pool slot such that no two live bullets share one.
    Greedy free-list, oldest-free-slot first (mimics the engine's pool reuse)."""
    order = np.argsort(frames, kind="stable")
    free: list[int] = list(range(POOL))
    heapq.heapify(free)
    busy: list[tuple[int, int]] = []           # (free_frame, slot), a min-heap
    out = np.full(len(frames), -1, np.int64)
    for i in order:
        f0 = int(frames[i])
        while busy and busy[0][0] <= f0:
            heapq.heappush(free, heapq.heappop(busy)[1])
        if not free:
            continue                            # pool exhausted — drop (rare)
        s = heapq.heappop(free)
        out[i] = s
        heapq.heappush(busy, (f0 + int(lifes[i]), s))
    return out


def build_schedule(seed: int, difficulty: int = 3) -> dict:
    """One ECL danmaku schedule in `_load_dense` format (+ `phase_hp`)."""
    vm, boss, orbs = _run_vm(seed, difficulty)
    F = len(boss)
    spawns = vm.bullets
    N = len(spawns)

    b = batch_from_spawns(spawns)
    xy = simulate_batch(b, MAXLIFE)                     # [N, MAXLIFE, 2]

    fx = b["fx_flag"].astype(int)
    grace = np.where(fx & FX_OFFSCREEN_GRACE_MASK, 128, 0)      # [N]
    m = CULL_MARGIN / 2.0
    off = ((xy[:, :, 0] < -m) | (xy[:, :, 0] > 384.0 + m) |
           (xy[:, :, 1] < -m) | (xy[:, :, 1] > 448.0 + m))      # [N, MAXLIFE]
    run = np.zeros_like(off, np.int32)                          # consecutive-off run
    run[:, 0] = off[:, 0]
    for k in range(1, MAXLIFE):
        run[:, k] = (run[:, k - 1] + 1) * off[:, k]
    exceeded = run > grace[:, None]         # culled the frame the run first exceeds
    life = np.where(exceeded.any(1), exceeded.argmax(1), MAXLIFE).astype(np.int64)

    # a phase transition screen-clears every live bullet — cap each bullet's
    # life at the next transition after it spawned
    trans = [fr for fr, _s in vm.phase_transitions()]
    sp_f = np.array([s.frame for s in spawns], np.int64)
    nxt = np.array([next((t for t in trans if t > f), F) for f in sp_f], np.int64)
    life = np.minimum(life, np.maximum(nxt - sp_f, 0))
    slot = _assign_slots(sp_f, np.minimum(life, MAXLIFE))

    pos = np.full((F, POOL, 2), np.nan, np.float32)
    half = np.full((F, POOL), BULLET_HB, np.float32)
    for i in range(N):
        s = slot[i]
        if s < 0:
            continue
        f0 = int(sp_f[i])
        L = int(min(life[i], MAXLIFE, F - f0))
        if L <= 0:
            continue
        pos[f0:f0 + L, s] = xy[i, :L]

    en = np.full((F, MAX_EN, 3), np.nan, np.float32)
    for f, fo in enumerate(orbs):
        for k, (x, y, h) in enumerate(fo):
            en[f, k] = (x, y, h)

    # phase windows from the VM's exact transitions (first four = the combat
    # phases NS1 / Lingering Cold / NS2 / Table-Turning; Sub51 is the defeat)
    trans = (trans + [F, F, F, F, F])[:5]
    nb = (~np.isnan(pos[:, :, 0])).sum(1)
    phw = []
    for j in range(4):
        cs, e = trans[j], trans[j + 1]
        seg = nb[cs:e]
        fa = cs + int(np.argmax(seg >= 15)) if (seg >= 15).any() else cs
        phw.append((int(cs), int(fa), int(e)))

    # drop the post-defeat tail (Sub51 fires nothing) — keep to the last bullet
    busy = np.where(nb >= 8)[0]
    end = int(busy[-1]) + 60 if len(busy) else F
    end = min(F, max(end, trans[4]))
    pos, half, boss, en = pos[:end], half[:end], boss[:end], en[:end]
    phw = [(cs, fa, min(e, end)) for cs, fa, e in phw]

    return dict(pos=pos, half=half, boss=boss, en=en, trim=0,
               name=f"letty_ecl_{seed}", phases=phw,
               phase_hp=list(LETTY_PHASE_HP))


def build_schedules(n: int, seed0: int = 0, difficulty: int = 3) -> list[dict]:
    return [build_schedule(seed0 + i, difficulty) for i in range(n)]


# --------------------------------------------------------------------------
def _verify() -> bool:
    import time
    from sim.ecl.danmaku_check import rec_density
    import glob

    ok = True
    t0 = time.time()
    recs = build_schedules(4)
    dt = (time.time() - t0) / 4
    print(f"built 4 schedules, {dt:.1f}s each\n")

    r = recs[0]
    F = r["pos"].shape[0]
    on = (~np.isnan(r["pos"][:, :, 0])).sum(1)
    print(f"  frames {F}  peak on-screen {on.max()}  mean {on.mean():.0f}")
    print(f"  boss track: x [{r['boss'][:,0].min():.0f},{r['boss'][:,0].max():.0f}] "
          f"y [{r['boss'][:,1].min():.0f},{r['boss'][:,1].max():.0f}]")
    n_orb = (~np.isnan(r["en"][:, :, 0])).any(-1).sum()
    print(f"  lethal-orb frames: {n_orb}   phases: {r['phases']}")

    # aggregate bullet count vs the recordings (same check as danmaku_check)
    vm_curve = np.mean([(~np.isnan(x["pos"][:, :, 0])).sum(1)[:9000]
                        for x in recs], 0)
    rec_curves = [rec_density(p)[:9000]
                  for p in sorted(glob.glob("sim/fights/letty_*.npz"))[:3]]
    L = min(len(vm_curve), min(len(c) for c in rec_curves))
    rec_mean = np.mean([c[:L] for c in rec_curves], 0)
    # align on first bullet
    v0 = int(np.argmax(vm_curve > 0))
    ratio = vm_curve[v0:v0 + L - v0].sum() / rec_mean[:L - v0].sum()
    print(f"\n  bullets-on-screen ratio ECL/recorded: {ratio:.2f}  (want ~1.0)")
    ok &= 0.75 < ratio < 1.3

    # variety: two seeds must differ (fresh RNG)
    d = np.nanmean(np.abs(np.nan_to_num(recs[0]["pos"]) -
                          np.nan_to_num(recs[1]["pos"])))
    print(f"  seed-0 vs seed-1 mean abs dpos: {d:.1f} px  (fresh RNG -> should be large)")
    ok &= d > 5.0

    # FightSim smoke test
    try:
        import torch
        from sim.fight_replay import FightSim
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fs = FightSim(recs=recs, name="letty", B=256, device=dev,
                      max_frames=F, mirror=False, randomize=False)
        obs = fs.reset()
        rew_sum = 0.0
        for _ in range(400):
            act = torch.randint(0, 18, (fs.B,), device=dev)
            obs, rew, done = fs.step(act)
            rew_sum += float(rew.mean())
        print(f"\n  FightSim(recs=ECL): obs {tuple(obs.shape)}  "
              f"400 steps ok  mean-rew/step {rew_sum/400:+.3f}")
        ok &= obs.shape[0] == fs.B and np.isfinite(rew_sum)
    except Exception as exc:                            # noqa: BLE001
        print(f"\n  FightSim smoke test FAILED: {exc!r}")
        ok = False

    print("\nPASS" if ok else "\nFAIL")
    return ok


if __name__ == "__main__":
    import sys
    raise SystemExit(0 if _verify() else 1)
