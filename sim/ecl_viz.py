"""Visualise an ECL boss pattern: run the VM, expand the spawn schedule, and
play the bullets forward in a pygame window. Eyeball it against the real game.

    .venv/Scripts/python sim/ecl_viz.py tools/th07_ecl/ecldata1_raw_named.tecl 29
    (arg2 = sub id: 29 Cirno First Column, 39 Letty non-spell, 42/48/52/55 Letty spells)

Keys: SPACE pause | . step | LEFT/RIGHT scrub | +/- speed | R restart | ESC quit
Bullets move straight (bullet_effects delay/accel/curve is ignored here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pygame

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ecl_vm import run_boss, REF_PLAYER          # noqa: E402
from ecl_expand import expand                    # noqa: E402
import math                                      # noqa: E402

PW, PH = 384, 448
SCALE = 1.6
MARGIN = 20

_SPRITE_COL = {  # rough th07 palette by sprite id band
    2: (150, 150, 255), 3: (255, 120, 120), 4: (120, 255, 120), 5: (255, 255, 140),
    514: (170, 200, 255), 517: (120, 200, 255), 530: (255, 160, 90), 533: (255, 90, 200),
    576: (200, 230, 255), 512: (150, 150, 255),
}


def col(sprite):
    return _SPRITE_COL.get(sprite, (230, 230, 230))


def main():
    tecl = sys.argv[1]
    sub = int(sys.argv[2]) if len(sys.argv) > 2 else 29
    diff = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 2400

    print(f"running VM: sub {sub} difficulty {diff} for {frames} frames...")
    sched = run_boss(tecl, sub, difficulty=diff, frames=frames)
    bullets = expand(sched)
    bullets.sort()
    print(f"{len(sched)} spawns -> {len(bullets)} bullets")

    pygame.init()
    W = int(PW * SCALE) + 2 * MARGIN + 180
    H = int(PH * SCALE) + 2 * MARGIN
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"ECL viz - sub {sub}")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    def to_screen(x, y):
        return MARGIN + x * SCALE, MARGIN + y * SCALE

    frame = 0.0
    speed = 1.0
    paused = False
    bi = 0                          # index into sorted bullets
    live = []                       # [(x0,y0,vx,vy,birth,sprite)]

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
                elif e.key == pygame.K_PERIOD:
                    frame += 1
                elif e.key == pygame.K_RIGHT:
                    frame += 30
                elif e.key == pygame.K_LEFT:
                    frame = max(0, frame - 30)
                    bi = 0
                    live = []
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(8.0, speed * 1.5)
                elif e.key == pygame.K_MINUS:
                    speed = max(0.1, speed / 1.5)
                elif e.key == pygame.K_r:
                    frame = 0.0
                    bi = 0
                    live = []

        if not paused:
            frame += speed

        # rebuild live set if we scrubbed backward
        if bi > 0 and (not live or live[-1][4] > frame):
            bi = 0
            live = []
        while bi < len(bullets) and bullets[bi][0] <= frame:
            bf, x, y, a, spd, aimed, sprite = bullets[bi]
            live.append((x, y, spd * math.cos(a), spd * math.sin(a), bf, sprite))
            bi += 1

        screen.fill((12, 12, 18))
        pf = pygame.Rect(MARGIN, MARGIN, PW * SCALE, PH * SCALE)
        pygame.draw.rect(screen, (30, 30, 42), pf)
        pygame.draw.rect(screen, (70, 70, 90), pf, 1)

        shown = 0
        for (x0, y0, vx, vy, birth, sprite) in live:
            age = frame - birth
            bx = x0 + vx * age
            by = y0 + vy * age
            if -20 < bx < PW + 20 and -20 < by < PH + 20:
                sx, sy = to_screen(bx, by)
                pygame.draw.circle(screen, col(sprite), (int(sx), int(sy)), 3)
                shown += 1

        # ref player
        px, py = to_screen(*REF_PLAYER)
        pygame.draw.circle(screen, (255, 80, 80), (int(px), int(py)), 4)
        pygame.draw.circle(screen, (255, 80, 80), (int(px), int(py)), 9, 1)

        panel_x = int(PW * SCALE) + 2 * MARGIN
        for i, line in enumerate([
            f"frame {frame:6.0f} / {frames}",
            f"speed {speed:.1f}x  {'PAUSED' if paused else ''}",
            f"bullets on screen: {shown}",
            f"spawned so far: {bi}",
            "",
            "SPACE pause  . step",
            "<- -> scrub   +/- speed",
            "R restart   ESC quit",
        ]):
            screen.blit(font.render(line, True, (200, 200, 210)), (panel_x, MARGIN + i * 20))

        pygame.display.flip()
        clock.tick(60)
        if frame > frames + 300:
            frame = 0.0
            bi = 0
            live = []

    pygame.quit()


if __name__ == "__main__":
    main()
