"""Play the ECL Letty fight (the training sim itself) yourself, immortal, to
eyeball that the danmaku / phases / boss motion / shot-damage look right before
training on it. Unlike fight_play.py (which replays a recorded .npz), this drives
the real FightSim + ECL schedules + runtime aim pool the trainer sees.

    .venv-cuda/Scripts/python sim/play_fight.py                 # 6 ECL schedules
    .venv-cuda/Scripts/python sim/play_fight.py --schedules 12
    .venv-cuda/Scripts/python sim/play_fight.py --replay        # recorded danmaku instead
    .venv-cuda/Scripts/python sim/play_fight.py --power 35      # fix raw power (default 10-50 random)
    .venv-cuda/Scripts/python sim/play_fight.py --autoshoot

Move: arrows or WASD    Focus: hold Shift    Shoot: hold Z (or --autoshoot)
SPACE pause | +/- speed | N next schedule | R restart this one | ESC quit

You cannot die - a red border flashes when a bullet / body would have hit you.
Colours: white bullet, amber = runtime aimed shot, red ring = lethal enemy body,
magenta = boss (grey = invulnerable / repositioning), green band = the forward-
needle lane (stand in it to land full damage), green dot = you (small = focus).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["FIGHTSIM_NOCOMPILE"] = "1"          # B=1 on CPU

import numpy as np
import pygame
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "native"))
from fight_replay import FightSim, LANE_HALF     # noqa: E402

PW, PH, SCALE, MARGIN, PANEL = 384, 448, 1.7, 18, 268

# (dx, dy) -> _DIRS index (y+ is down); matches fight_replay._DIRS
_DIR_IDX = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4,
            (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}


def _arg(flag, default=None, cast=str):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    n_sched = int(_arg("--schedules", 6, int))
    use_replay = "--replay" in sys.argv
    autoshoot = "--autoshoot" in sys.argv
    fixed_power = _arg("--power", None, float)

    recs = None
    if not use_replay:
        from danmaku_ecl import build_schedules
        print(f"[ecl] building {n_sched} danmaku schedules (~3s each)...", flush=True)
        recs = build_schedules(n_sched, seed0=0)

    kw = dict(B=1, name="letty", device="cpu", seed=0, immortal=True,
              phase_start_mix=0.0, randomize=True)
    if not use_replay:
        kw.update(mirror=False, field_rot_deg=0.0, recs=recs)
    if fixed_power is not None:
        kw.update(power_lo=fixed_power, power_hi=fixed_power)
    sim = FightSim(**kw)
    n_rec = sim.n_rec

    pygame.init()
    W = int(PW * SCALE) + 2 * MARGIN + PANEL
    H = int(PH * SCALE) + 2 * MARGIN
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("play_fight: ECL Letty (immortal)")
    big = pygame.font.SysFont("consolas", 15)
    small = pygame.font.SysFont("consolas", 13)
    huge = pygame.font.SysFont("consolas", 30, bold=True)
    clock = pygame.time.Clock()

    def ts(x, y):
        return int(MARGIN + x * SCALE), int(MARGIN + y * SCALE)

    sched_i = 0
    ep_frames = 0
    speed = 1.0
    paused = False
    hitflash = 0
    flash = None
    results = []

    def restart(new_schedule):
        nonlocal sched_i, ep_frames
        if new_schedule:
            sched_i = (sched_i + 1) % n_rec
        sim.reset()                       # phase-0 start, per-episode power/hit-mult
        sim.rec_id[0] = sched_i           # ...but cycle schedules deterministically
        sim.no_phase[0] = False           # always let shots deal damage in the viz
        sim.boss_hp[0] = sim.ph[sched_i, 0, 3]
        sim._prev_active[0] = False
        sim.aim.reset(torch.tensor([0]), sim.rec_id[[0]], sim.t0[[0]])
        ep_frames = 0

    restart(False)

    running = True
    while running:
        keys = pygame.key.get_pressed()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(4.0, speed * 1.5)
                elif e.key == pygame.K_MINUS:
                    speed = max(0.15, speed / 1.5)
                elif e.key == pygame.K_n:
                    restart(True)
                    flash = ("SCHEDULE %d" % sched_i, (150, 150, 160), 20)
                elif e.key == pygame.K_r:
                    restart(False)
                    flash = ("RESTART", (150, 150, 160), 20)

        # ---- keyboard -> action int ----
        dx = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = (keys[pygame.K_DOWN] or keys[pygame.K_s]) - (keys[pygame.K_UP] or keys[pygame.K_w])
        focus = 1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0
        shoot = 1 if (autoshoot or keys[pygame.K_z] or keys[pygame.K_j]) else 0
        d = _DIR_IDX[(int(dx), int(dy))]
        act = d + 9 * focus + 18 * shoot

        stepping = not paused and flash is None
        n_steps = int(speed) if speed >= 1 else (1 if ep_frames % max(1, int(1 / speed)) == 0 else 0)
        if stepping:
            for _ in range(max(1, n_steps)):
                _, _, done = sim.step(torch.tensor([act]))
                ep_frames += 1
                if bool(sim.last_hit[0]):
                    hitflash = 6
                if bool(done[0]):
                    killed = bool(sim.last_killed[0])
                    secs = ep_frames / 60.0
                    tag = "LETTY DOWN" if killed else "TIME UP"
                    flash = (f"{tag} @ {secs:.0f}s",
                             (120, 255, 140) if killed else (200, 200, 210), 60)
                    results.append(f"sched {sched_i}: {tag.lower()} {secs:.0f}s")
                    results[:] = results[-10:]
                    restart(False)         # step already auto-reset; re-pin schedule
                    break

        # ---- render ----
        screen.fill((10, 10, 15))
        pygame.draw.rect(screen, (26, 26, 38),
                         (MARGIN, MARGIN, int(PW * SCALE), int(PH * SCALE)))

        bp, active, bh, en, en_a, f = sim._now()
        rid0 = int(sim.rec_id[0])
        fi = int(f[0])
        bp0, act0 = bp[0].numpy(), active[0].numpy()
        n_on = 0
        for i in np.nonzero(act0)[0]:
            x, y = bp0[i]
            if -20 < x < PW + 20 and -20 < y < PH + 20:
                n_on += 1
                pygame.draw.circle(screen, (232, 232, 232), ts(x, y), 3)
        n_aim = 0
        if sim._aim_now is not None:
            ap, aa, ah, _ = sim._aim_now
            ap0, aa0 = ap[0].numpy(), aa[0].numpy()
            for i in np.nonzero(aa0)[0]:
                x, y = ap0[i]
                if -20 < x < PW + 20 and -20 < y < PH + 20:
                    n_aim += 1
                    pygame.draw.circle(screen, (255, 200, 90), ts(x, y), 3)
        en0, ena0 = en[0].numpy(), en_a[0].numpy()
        for i in np.nonzero(ena0)[0]:
            ex, ey, er = en0[i]
            pygame.draw.circle(screen, (255, 80, 80), ts(ex, ey),
                               max(3, int(er * SCALE)), 1)

        armored = bool(sim.armored[0].item()) if hasattr(sim, "armored") else False
        bx, by = (float(v) for v in sim.boss[rid0, fi].numpy())
        if not armored:
            lane = pygame.Surface((int(2 * LANE_HALF * SCALE), int(PH * SCALE)),
                                  pygame.SRCALPHA)
            lane.fill((90, 240, 140, 26))
            screen.blit(lane, (int(MARGIN + (bx - LANE_HALF) * SCALE), MARGIN))
        pygame.draw.circle(screen, (110, 110, 130) if armored else (240, 120, 240),
                           ts(bx, by), 9 if armored else 7, 0 if armored else 2)

        px, py = float(sim.px[0]), float(sim.py[0])
        aligned = abs(px - bx) < LANE_HALF
        pygame.draw.circle(screen, (120, 255, 120), ts(px, py), 6 if not focus else 3)
        if focus:
            pygame.draw.circle(screen, (120, 255, 120), ts(px, py), 8, 1)

        if hitflash > 0:
            pygame.draw.rect(screen, (255, 60, 80),
                             (MARGIN - 3, MARGIN - 3, int(PW * SCALE) + 6,
                              int(PH * SCALE) + 6), 3)
            hitflash -= 1

        pj = int(sim.phase_idx[0])
        p_hp0 = float(sim.ph[rid0, pj, 3])
        hp_frac = max(0.0, min(1.0, float(sim.boss_hp[0]) / max(1.0, p_hp0)))
        n_ph = int(sim.n_ph[rid0])
        shoot_now = shoot and not armored
        mult = float(sim.dps_mult[0])
        dps_line = (
            "  boss invulnerable" if armored else
            f"  SHOOT lined  {float(sim.shot_dps[0]) * mult:.1f} HP/f"
            if (shoot_now and aligned) else
            f"  shoot homing {float(sim.shot_dps[0]) * mult * float(sim.homing_frac[0]):.1f} HP/f"
            if shoot_now else "  (not shooting)")

        x0 = int(PW * SCALE) + 2 * MARGIN
        lines = [
            ("play_fight  (immortal)", (235, 235, 245), big),
            (("ECL schedule %d/%d" % (sched_i, n_rec)) if not use_replay
             else "recorded danmaku", (150, 200, 255), small),
            ("", None, small),
            (f"  t {ep_frames / 60:6.1f}s", (210, 210, 220), small),
            (f"  phase {pj + 1}/{n_ph}" + ("  [INVULN]" if armored else ""),
             (150, 150, 200) if armored else (210, 210, 220), small),
            (f"  raw power {float(sim.power[0]):.0f}   x{mult:.2f} hit-rate",
             (170, 190, 210), small),
            (dps_line,
             (120, 120, 150) if armored else
             (255, 230, 90) if (shoot_now and aligned) else
             (180, 160, 90) if shoot_now else (110, 110, 120), small),
            ("", None, small),
            (f"  bullets on screen {n_on}", (210, 210, 220), small),
            (f"  aimed shots        {n_aim}", (255, 200, 90), small),
            (f"  lethal bodies      {int(ena0.sum())}", (255, 120, 120), small),
            ("", None, small),
            (f"  {speed:.2f}x {'PAUSED' if paused else ''}", (170, 170, 180), small),
            ("", None, small),
        ]
        yy = MARGIN
        for txt, col, fnt in lines:
            if txt and col:
                screen.blit(fnt.render(txt, True, col), (x0, yy))
            yy += 19 if fnt is small else 24

        pygame.draw.rect(screen, (60, 60, 72), (x0, yy, PANEL - 20, 12), 1)
        pygame.draw.rect(screen, (240, 120, 240),
                         (x0 + 1, yy + 1, int((PANEL - 22) * hp_frac), 10))
        yy += 20
        screen.blit(small.render(f"  boss HP {float(sim.boss_hp[0]):.0f}", True,
                                 (200, 160, 200)), (x0, yy))
        yy += 24
        for r in results[-10:]:
            screen.blit(small.render(r, True, (150, 155, 165)), (x0, yy))
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
