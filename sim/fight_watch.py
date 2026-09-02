"""Watch a training run's current policy fight the real boss, live. Runs one
FightSim episode at a time with the newest checkpoint (reloaded every episode,
so it visibly improves while training runs) and renders it.

    .venv-cuda/Scripts/python sim/fight_watch.py                 # fight_letty_seg
    .venv-cuda/Scripts/python sim/fight_watch.py fight_letty_seg
    .venv-cuda/Scripts/python sim/fight_watch.py fight_cirno --fight cirno

Keys: SPACE pause | +/- speed | N next episode now | ESC quit
Colours: white/cyan/orange bullets (plain/curve/redirect), red ring = lethal
enemy body, magenta ring = boss, green = the agent (small dot = focus hitbox).
A yellow "SHOOT" tag shows when the agent is dealing damage.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["FIGHTSIM_NOCOMPILE"] = "1"     # B=1 on CPU - compiling isn't worth it

import numpy as np
import pygame
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from fight_replay import FightSim, LANE_HALF, HOMING_FRAC   # noqa: E402
from policy import MLPPolicy              # noqa: E402

PW, PH, SCALE, MARGIN, PANEL = 384, 448, 1.7, 18, 250


def newest_ckpt(run):
    for cand in ("last_mlp.pt", "final_mlp.pt"):
        p = run / cand
        if p.exists():
            return p
    snaps = sorted(run.glob("mlp_*.pt"))
    return snaps[-1] if snaps else None


def train_status(run):
    hp = run / "history.npy"
    if not hp.exists():
        return "(no history yet)"
    try:
        h = np.load(hp)
        row = h[-1]
        if len(row) >= 6:                       # 6-col: wall,steps,med,mean,kill,ktime
            s, med, kr, kt = row[1], row[2], row[4], row[5]
        else:                                   # 5-col: steps,med,mean,kill,ktime
            s, med, kr = row[0], row[1], row[3]
            kt = row[4] if len(row) > 4 else float("nan")
        ktxt = f"  kill-time {kt:.0f}s" if not np.isnan(kt) else ""
        return (f"{s / 1e6:.0f}M steps   eval: kill {kr * 100:.0f}%{ktxt}"
                f"   survive {med:.0f}s")
    except Exception:
        return "(history unreadable)"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "fight_letty_seg"
    fight = "letty"
    if "--fight" in sys.argv:
        fight = sys.argv[sys.argv.index("--fight") + 1]
    run = HERE.parent / "runs_sim" / name
    assert run.exists(), f"no run dir {run}"

    sim = FightSim(B=1, name=fight, device="cpu", phase_start_mix=0.0, seed=0)
    n_ph = int(sim.n_ph[0])

    pol = None
    ckpt_mtime = 0.0

    def reload_policy():
        nonlocal pol, ckpt_mtime
        c = newest_ckpt(run)
        if c is None:
            return
        m = c.stat().st_mtime
        if m != ckpt_mtime:
            try:
                pol = MLPPolicy.load(c)
                ckpt_mtime = m
            except Exception:
                pass

    reload_policy()
    while pol is None:
        print("waiting for a checkpoint in", run, "...")
        time.sleep(3)
        reload_policy()

    pygame.init()
    W = int(PW * SCALE) + 2 * MARGIN + PANEL
    H = int(PH * SCALE) + 2 * MARGIN
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"watch: {name} vs {fight}")
    big = pygame.font.SysFont("consolas", 15)
    small = pygame.font.SysFont("consolas", 13)
    huge = pygame.font.SysFont("consolas", 30, bold=True)
    clock = pygame.time.Clock()

    def ts(x, y):
        return int(MARGIN + x * SCALE), int(MARGIN + y * SCALE)

    obs = sim.reset()
    ep_frames = 0
    ep_no = 0
    best_reach = 0.0
    last_act = 0
    speed = 1.0
    paused = False
    flash = None            # (text, colour, frames_left)
    results = []            # recent outcomes for the panel

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(8.0, speed * 1.5)
                elif e.key == pygame.K_MINUS:
                    speed = max(0.25, speed / 1.5)
                elif e.key == pygame.K_n:
                    flash = ("SKIPPED", (150, 150, 160), 12)
                    obs = sim.reset(); ep_frames = 0; reload_policy(); ep_no += 1

        stepping = not paused and flash is None
        n_steps = int(speed) if speed >= 1 else (1 if (ep_frames % int(1 / speed) == 0) else 0)
        if stepping:
            for _ in range(max(1, n_steps)):
                a = pol.act_batch(obs.numpy())
                last_act = int(a[0])
                obs, rew, done = sim.step(torch.as_tensor(a))
                ep_frames += 1
                if bool(done[0]):
                    killed = bool(sim.last_killed[0])
                    secs = ep_frames / 60.0
                    if killed:
                        flash = ("LETTY DOWN", (120, 255, 140), 70)
                        results.append(f"ep{ep_no}: KILL @ {secs:.0f}s")
                    else:
                        ph = int(sim.phase_idx[0]) + 1   # already reset? see below
                        flash = (f"DIED @ {secs:.0f}s", (255, 110, 130), 45)
                        results.append(f"ep{ep_no}: died {secs:.0f}s  p{ph}")
                    best_reach = max(best_reach, secs)
                    results[:] = results[-10:]
                    ep_frames = 0
                    ep_no += 1
                    reload_policy()
                    break

        # ---- render ----
        screen.fill((10, 10, 15))
        pygame.draw.rect(screen, (26, 26, 38),
                         (MARGIN, MARGIN, int(PW * SCALE), int(PH * SCALE)))

        bp, active, bh, en, en_a, f = sim._now()
        rid0 = int(sim.rec_id[0])
        bp0, act0 = bp[0].numpy(), active[0].numpy()
        for i in np.nonzero(act0)[0]:
            x, y = bp0[i]
            if -20 < x < PW + 20 and -20 < y < PH + 20:
                pygame.draw.circle(screen, (232, 232, 232), ts(x, y), 3)
        en0, ena0 = en[0].numpy(), en_a[0].numpy()
        for i in np.nonzero(ena0)[0]:
            ex, ey, er = en0[i]
            pygame.draw.circle(screen, (255, 80, 80), ts(ex, ey),
                               max(3, int(er * SCALE)), 1)
        armored = bool(sim.armored[0].item()) if hasattr(sim, "armored") else False
        bx, by = (float(v) for v in sim.boss[rid0, int(f[0])].numpy())
        if not armored:
            # forward-shot lane: stand in this band (x) to land ReimuA's needles
            lane = pygame.Surface((int(2 * LANE_HALF * SCALE), int(PH * SCALE)),
                                  pygame.SRCALPHA)
            lane.fill((90, 240, 140, 26))
            screen.blit(lane, (int(MARGIN + (bx - LANE_HALF) * SCALE), MARGIN))
        boss_col = (110, 110, 130) if armored else (240, 120, 240)
        pygame.draw.circle(screen, boss_col, ts(bx, by), 9 if armored else 7,
                           0 if armored else 2)

        px, py = float(sim.px[0]), float(sim.py[0])
        focus = (last_act // 9) % 2
        shoot_bit = (last_act >= 18) and not armored
        aligned = abs(px - bx) < LANE_HALF
        shooting = shoot_bit and aligned
        pygame.draw.circle(screen, (120, 255, 120), ts(px, py), 6 if not focus else 3)
        if focus:
            pygame.draw.circle(screen, (120, 255, 120), ts(px, py), 8, 1)

        # phase / hp
        pj = int(sim.phase_idx[0])
        p_hp0 = float(sim.ph[rid0, pj, 3])
        hp_frac = max(0.0, min(1.0, float(sim.boss_hp[0]) / max(1.0, p_hp0)))

        px0 = int(PW * SCALE) + 2 * MARGIN
        lines = [
            (name, (235, 235, 245), big),
            (train_status(run), (150, 200, 255), small),
            ("", None, small),
            (f"episode {ep_no}", (235, 235, 245), big),
            (f"  survived {ep_frames / 60:5.1f}s", (210, 210, 220), small),
            (f"  phase {pj + 1}/{n_ph}" + ("   [INVULN - repositioning]" if armored
                                           else ""),
             (150, 150, 200) if armored else (210, 210, 220), small),
            (f"  best this session {best_reach:.0f}s", (170, 170, 180), small),
            ("  boss invulnerable" if armored else
             f"  SHOOT - lined up ({float(sim.shot_dps[0]):.0f} HP/f, pow {float(sim.power[0]):.0f})"
             if shooting else
             f"  shoot - homing only ({float(sim.homing_frac[0])*100:.0f}%)" if shoot_bit else
             "  (not shooting)",
             (120, 120, 150) if armored else
             (255, 230, 90) if shooting else
             (180, 160, 90) if shoot_bit else (110, 110, 120), small),
            ("", None, small),
            (f"  {speed:.2f}x {'PAUSED' if paused else ''}", (170, 170, 180), small),
            ("", None, small),
        ]
        yy = MARGIN
        for txt, col, fnt in lines:
            if txt and col:
                screen.blit(fnt.render(txt, True, col), (px0, yy))
            yy += 20 if fnt is small else 24

        # phase hp bar
        pygame.draw.rect(screen, (60, 60, 72), (px0, yy, PANEL - 20, 12), 1)
        pygame.draw.rect(screen, (240, 120, 240),
                         (px0 + 1, yy + 1, int((PANEL - 22) * hp_frac), 10))
        yy += 26
        for r in results[-10:]:
            screen.blit(small.render(r, True, (150, 155, 165)), (px0, yy))
            yy += 17

        if flash is not None:
            txt, col, left = flash
            surf = huge.render(txt, True, col)
            screen.blit(surf, surf.get_rect(
                center=(MARGIN + PW * SCALE / 2, MARGIN + PH * SCALE / 2)))
            flash = None if left <= 1 else (txt, col, left - 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
