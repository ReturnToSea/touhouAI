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
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> sim.ecl
from sim.ecl.parser import parse_file            # noqa: E402
from sim.ecl.vm import VM                        # noqa: E402
from sim.ecl.bullet_sim import (batch_from_spawns, simulate_batch,   # noqa: E402
                                FX_OFFSCREEN_GRACE_MASK)

_ROOT = Path(__file__).resolve().parents[1]
_ECL = _ROOT / "tools" / "th07_ecl" / "ecldata1.ecl"
_FIGHTS = _ROOT / "sim" / "fights"

POOL = 1025
MAX_EN = 48
# frames to propagate a bullet. Measured max real lifetime on the small playfield
# (phase-clamped) is ~640 f (slow LC/TT bullets); 420 was truncating ~1% of
# bullets mid-screen. Peak concurrent is 588 either way, well under POOL.
MAXLIFE = 900
BULLET_HB = 2.5               # fallback for an unseen type-word
ENEMY_BODY_SCALE = 2.0 / 3.0
VM_PLAYER = (192.0, 400.0)    # where the VM aims — undo it, re-aim per episode
CULL_MARGIN = 24.0

# player-frame aim point the VM uses (VM.__init__ default)


def _type_hitboxes() -> dict[int, float]:
    """type-word (`+0xBF6`, the ECL flags arg) -> AABB half-extent, measured
    from the recordings (col 8/9). Covers all of Letty's bullets; ~2 px pellets,
    ~3 px balls, ~5 px the big Lingering-Cold class."""
    import glob
    from collections import defaultdict
    acc: dict[int, list] = defaultdict(list)
    for p in sorted(glob.glob(str(_FIGHTS / "letty_[0-9]*.npz"))):
        b = np.load(p)["bullets"]
        if b.shape[1] < 19:
            continue
        h = np.maximum(b[:, 8], b[:, 9]) * 0.5
        for tf, hb in zip(b[::11, 18].astype(int), h[::11]):
            if hb > 0:
                acc[tf].append(hb)
    return {tf: round(float(np.median(v)), 2) for tf, v in acc.items() if len(v) > 50}


_TYPE_HB = _type_hitboxes()


def _bullet_hb(btype: int, speed: float = 0.0) -> float:
    bt = int(btype) & 0xFFFF
    # Lingering Cold's big snow crystals: the game renders the speed-3.5 "lead"
    # bullet of every Sub44-47 fan (btype 576) as the large class-2 sprite with a
    # 10px box (half 5.0), not the 6px class-3 (half 3.0) the other 96% get.
    # Verified from the recordings: 100% of class-2 LC bullets are btype-576 born
    # at |vel| = 3.5.  _TYPE_HB medians it away to 3.0, so special-case it.
    if bt == 576 and speed >= 3.0:
        return 5.0
    return _TYPE_HB.get(bt, BULLET_HB)

# Letty's real per-phase HP budget + damage multiplier (Part 7, revised twice):
#   NS1 (Sub38): life_set 15000, life_callback_ex -> LC at HP 1700 (13300 drain),
#                full damage.
#   Lingering Cold (Sub42): NO life_callback, but IS capturable - it inherits
#                ~1700 HP and the engine applies shot damage at 1/7 while a
#                spellcard is active (FUN_00420620 ~13817: `DAT_012fe0c8 != 0
#                -> local_1c /= 7`; DAT_012fe0c8 is set by spellcard_start /
#                cleared by spellcard_end). Plus the 480f declaration + 300f
#                armor. So it lasts ~15-50s: captured if the player point-blanks
#                it, else it times out at 3000f.
#   NS2 (Sub39): life_set 15000, life_callback at HP 2000 (13000 drain), full dmg.
#   Table-Turning (Sub55): same as LC, inherits ~2000 HP, 1/7 damage.
SPELL_DMG_MULT = 1.0 / 7.0
LETTY_PHASE_HP = (13300.0, 1700.0, 13000.0, 2000.0)
LETTY_PHASE_DMG_MULT = (1.0, SPELL_DMG_MULT, 1.0, SPELL_DMG_MULT)


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
        # lethal orb = alive sub-enemy with player-collision on (flag bit 1) and
        # a non-degenerate body box. Sub43 (LC emitter) clears the flag.
        fo = [(e.x, e.y, max(e.hitbox) * 0.5 * ENEMY_BODY_SCALE)
              for e in vm.enemies
              if not e.is_boss and e.alive and e.collidable
              and e.hitbox[0] > 0.5 and e.hitbox[1] > 0.5]
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


def _cull_life(xy: np.ndarray, fx_flag: np.ndarray) -> np.ndarray:
    """First frame the engine erases each bullet (`FUN_0042d6d8` box + grace)."""
    grace = np.where(fx_flag.astype(int) & FX_OFFSCREEN_GRACE_MASK, 128, 0)
    m = CULL_MARGIN / 2.0
    off = ((xy[:, :, 0] < -m) | (xy[:, :, 0] > 384.0 + m) |
           (xy[:, :, 1] < -m) | (xy[:, :, 1] > 448.0 + m))
    run = np.zeros_like(off, np.int32)
    run[:, 0] = off[:, 0]
    for k in range(1, off.shape[1]):
        run[:, k] = (run[:, k - 1] + 1) * off[:, k]
    ex = run > grace[:, None]
    return np.where(ex.any(1), ex.argmax(1), off.shape[1]).astype(np.int64)


def build_schedule(seed: int, difficulty: int = 3) -> dict:
    """One ECL danmaku schedule in `_load_dense` format, plus `phase_hp` and an
    `aimed` param table (aimed bullets are re-generated toward the live policy at
    runtime — they carry the *un-aimed* angle here)."""
    vm, boss, orbs = _run_vm(seed, difficulty)
    F = len(boss)
    trans = [fr for fr, _s in vm.phase_transitions()]

    baked = [s for s in vm.bullets if not s.aimed]
    aimed = [s for s in vm.bullets if s.aimed]

    # --- baked (non-aimed) bullets -> dense arrays -------------------------
    pos = np.full((F, POOL, 2), np.nan, np.float32)
    half = np.full((F, POOL), BULLET_HB, np.float32)
    if baked:
        b = batch_from_spawns(baked)
        xy = simulate_batch(b, MAXLIFE)
        life = _cull_life(xy, b["fx_flag"])
        sp_f = np.array([s.frame for s in baked], np.int64)
        nxt = np.array([next((t for t in trans if t > f), F) for f in sp_f], np.int64)
        life = np.minimum(life, np.maximum(nxt - sp_f, 0))     # phase screen-clear
        slot = _assign_slots(sp_f, np.minimum(life, MAXLIFE))
        for i, s in enumerate(baked):
            sl = slot[i]
            if sl < 0:
                continue
            f0 = int(sp_f[i])
            L = int(min(life[i], MAXLIFE, F - f0))
            if L <= 0:
                continue
            pos[f0:f0 + L, sl] = xy[i, :L]
            half[f0:f0 + L, sl] = _bullet_hb(s.btype, s.speed)

    # --- aimed bullets -> param table (re-aimed at runtime) --------------
    vpx, vpy = VM_PLAYER
    if aimed:
        af = np.array([s.frame for s in aimed], np.int64)
        o = np.argsort(af, kind="stable")
        aimed = [aimed[i] for i in o]
        ab = batch_from_spawns(aimed)
        unaim = np.array([s.angle for s in aimed]) - np.arctan2(
            vpy - ab["y"], vpx - ab["x"])
        aim_tbl = dict(
            frame=np.array([s.frame for s in aimed], np.int32),
            x0=ab["x"].astype(np.float32), y0=ab["y"].astype(np.float32),
            speed=ab["speed"].astype(np.float32),
            unaim=unaim.astype(np.float32),
            hb=np.array([_bullet_hb(s.btype, s.speed) for s in aimed], np.float32),
            hang_state=ab["hang_state"].astype(np.float32),
            hang_frames=ab["hang_frames"].astype(np.float32),
            fx_flag=ab["fx_flag"].astype(np.float32),
            fx_p1=ab["fx_p1"].astype(np.float32), fx_p2=ab["fx_p2"].astype(np.float32),
            fx_interval=ab["fx_interval"].astype(np.float32),
            fx_repeat=ab["fx_repeat"].astype(np.float32),
            launch=ab["launch"].astype(np.float32),
            end=np.array([next((t for t in trans if t > s.frame), F)
                          for s in aimed], np.int32),
        )
    else:
        aim_tbl = None

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
               phase_hp=list(LETTY_PHASE_HP),
               phase_dmg_mult=list(LETTY_PHASE_DMG_MULT), aimed=aim_tbl)


def build_schedules(n: int, seed0: int = 0, difficulty: int = 3) -> list[dict]:
    return [build_schedule(seed0 + i, difficulty) for i in range(n)]


# --- streaming: a background process keeps a directory topped up with fresh
# schedules so training never sees the same 48 layouts for 800M steps ----------
_AIM_KEYS = ("frame", "x0", "y0", "speed", "unaim", "hb", "hang_state",
             "hang_frames", "fx_flag", "fx_p1", "fx_p2", "fx_interval",
             "fx_repeat", "launch", "end")


def save_schedule(path, s: dict) -> None:
    """Flatten a schedule dict to a single .npz (dicts/lists don't savez)."""
    out = dict(pos=s["pos"], half=s["half"], boss=s["boss"], en=s["en"],
               phases=np.asarray(s["phases"], np.int64),
               phase_hp=np.asarray(s["phase_hp"], np.float32),
               phase_dmg_mult=np.asarray(s["phase_dmg_mult"], np.float32),
               has_aimed=np.array(s.get("aimed") is not None))
    if s.get("aimed"):
        for k in _AIM_KEYS:
            out[f"aim_{k}"] = s["aimed"][k]
    # stage to a dot-prefixed name so the consumer's "[0-9]*.npz" glob never
    # sees it, then atomic rename. pos is ~40% NaN -> compresses ~3x.
    path = Path(path)
    tmp = path.parent / ("." + path.name)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **out)
    os.replace(tmp, path)


def load_schedule(path) -> dict:
    # np.load returns a lazy NpzFile that holds the archive OPEN; copy everything
    # out and close it, or the caller can't delete the file on Windows.
    with np.load(path, allow_pickle=False) as d:
        g = {k: np.array(d[k]) for k in d.files}
    aimed = ({k: g[f"aim_{k}"] for k in _AIM_KEYS}
             if bool(g["has_aimed"]) else None)
    return dict(pos=g["pos"], half=g["half"], boss=g["boss"], en=g["en"], trim=0,
               name="letty_ecl_stream",
               phases=[tuple(int(x) for x in row) for row in g["phases"]],
               phase_hp=list(map(float, g["phase_hp"])),
               phase_dmg_mult=list(map(float, g["phase_dmg_mult"])), aimed=aimed)


def stream_worker(pooldir, seed0: int = 100_000, difficulty: int = 3,
                  max_pending: int = 60) -> None:
    """Loop forever: build a schedule, drop it in `pooldir` as NNNNNNNNN.npz.
    Throttles when the consumer is behind (dir already has max_pending files).
    `seed0` is offset by the worker's pid so parallel workers don't collide."""
    import time
    from pathlib import Path
    pooldir = Path(pooldir)
    pooldir.mkdir(parents=True, exist_ok=True)
    ctr = seed0 + (os.getpid() % 997) * 100_000
    while True:
        pend = list(pooldir.glob("[0-9]" * 9 + ".npz"))
        if len(pend) >= max_pending:
            time.sleep(2.0)
            continue
        try:
            s = build_schedule(ctr, difficulty)
            save_schedule(pooldir / f"{ctr:09d}.npz", s)
        except Exception as e:               # never let the worker die on one bad seed
            print(f"[stream] seed {ctr} failed: {e}", flush=True)
        ctr += 1


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
    n_aim = len(r["aimed"]["frame"]) if r.get("aimed") else 0
    print(f"  lethal-orb frames: {n_orb}   aimed shots: {n_aim}   "
          f"phases: {r['phases']}")

    # aggregate bullet count vs the recordings (baked + a rough aimed estimate)
    vm_curve = np.mean([(~np.isnan(x["pos"][:, :, 0])).sum(1)[:9000]
                        for x in recs], 0)
    rec_curves = [rec_density(p)[:9000]
                  for p in sorted(glob.glob(str(_FIGHTS / "letty_[0-9]*.npz")))[:3]]
    L = min(len(vm_curve), min(len(c) for c in rec_curves))
    rec_mean = np.mean([c[:L] for c in rec_curves], 0)
    v0 = int(np.argmax(vm_curve > 0))
    baked_ratio = vm_curve[v0:v0 + L - v0].sum() / rec_mean[:L - v0].sum()
    aim_frac = n_aim / max(1, n_aim + int(vm_curve.sum() / 120))   # ~lifetime 120
    print(f"\n  bullets-on-screen ratio (baked only): {baked_ratio:.2f}   "
          f"aimed are ~{aim_frac * 100:.0f}% more, in the runtime pool")
    ok &= 0.75 < baked_ratio + aim_frac < 1.35

    # variety: two seeds must differ (fresh RNG)
    d = np.nanmean(np.abs(np.nan_to_num(recs[0]["pos"]) -
                          np.nan_to_num(recs[1]["pos"])))
    print(f"  seed-0 vs seed-1 mean abs dpos: {d:.1f} px  (fresh RNG -> should be large)")
    ok &= d > 5.0

    # per-type hitboxes actually vary
    hbs = sorted(set(_TYPE_HB.values()))
    print(f"  bullet hitbox halves in use: {hbs}")
    ok &= len(hbs) >= 2

    # aimed bullets rotate with the player  (a full ring aimed left vs right)
    try:
        import torch
        from sim.aim_pool import AimPool
        f0 = int(r["aimed"]["frame"].min())
        rings = []
        for pxv in (70.0, 314.0):
            ap = AimPool([r], B=1, device="cpu")
            z = torch.zeros(1, dtype=torch.long)
            ap.reset(z, z.clone(), z.clone())
            for fr in range(f0 - 3, f0 + 2):
                ap.step(torch.full((1,), fr), torch.full((1,), pxv),
                        torch.full((1,), 400.0), torch.ones(1), torch.zeros(1))
            a = ap.ang[0, ap.alive[0]]
            rings.append(float(a.min()))                # ring's leading edge
        turn = np.degrees(rings[0] - rings[1])
        want = np.degrees(np.arctan2(336.0, 70 - 192) - np.arctan2(336.0, 314 - 192))
        print(f"  aimed ring turns {turn:+.0f} deg between player x=70 and x=314 "
              f"(want ~{want:+.0f})")
        ok &= abs(turn - want) < 12.0

        # AimPool physics must match the scalar bullet_sim.simulate exactly:
        # aim at the VM's own point -> should reproduce simulate_batch(s.angle)
        from sim.ecl.parser import parse_file
        from sim.ecl.vm import VM
        from sim.ecl.bullet_sim import batch_from_spawns, simulate_batch
        vm = VM(parse_file(str(_ECL)), difficulty=3, seed=0)
        vm.start_boss(sub=31, interrupt=0)
        vm.run(13000)
        am = sorted((s for s in vm.bullets if s.aimed), key=lambda s: s.frame)
        ref = simulate_batch(batch_from_spawns(am), 135)
        ap = AimPool([r], B=1, device="cpu")
        z = torch.zeros(1, dtype=torch.long)
        ap.reset(z, z.clone(), z.clone())
        traj, born = {}, {}
        for fr in range(int(am[0].frame), int(am[-1].frame) + 140):
            was = ap.alive[0].clone()
            ap.step(torch.full((1,), fr), torch.full((1,), VM_PLAYER[0]),
                    torch.full((1,), VM_PLAYER[1]), torch.ones(1), torch.zeros(1))
            for s in (ap.alive[0] & ~was).nonzero().flatten().tolist():
                traj[(s, fr)] = []
                born[(s, fr)] = fr
            for kk in list(traj):
                if ap.alive[0, kk[0]] and born[kk] == kk[1]:
                    traj[kk].append((float(ap.px[0, kk[0]]), float(ap.py[0, kk[0]])))
                elif not ap.alive[0, kk[0]]:
                    born[kk] = -1
        de, used = [], set()
        for kk, tj in traj.items():
            tj = np.array(tj)
            if len(tj) < 20:
                continue
            cand = [i for i in range(len(am))
                    if abs(am[i].frame - kk[1]) <= 1 and i not in used]
            if not cand:
                continue
            bi = min(cand, key=lambda i: float(np.hypot(*(tj[0] - ref[i, 1]))))
            used.add(bi)
            L = min(len(tj), 133)
            de.append(float(np.hypot(*(tj[:L] - ref[bi, 1:L + 1]).T).max()))
        de = np.array(de)
        print(f"  AimPool vs bullet_sim.simulate: {len(de)} aimed bullets, "
              f"max {de.max():.2f}px  (must be ~0)")
        ok &= de.max() < 1.0
    except Exception as exc:                            # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"  aim check FAILED: {exc!r}")
        ok = False

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
