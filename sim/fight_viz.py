"""Play back a recorded real fight (native/record_boss_driven.py -> fights/*.npz).
This is the exact engine output - it should look identical to the real game.

    .venv-cuda/Scripts/python sim/fight_viz.py cirno
Keys: SPACE pause | . step | <- -> scrub | +/- speed | R restart | ESC quit
Colours: white plain, cyan curve (fx 32), orange redirect (fx 64/128).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pygame

HERE = Path(__file__).resolve().parent
PW, PH, SCALE, MARGIN = 384, 448, 1.6, 20


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cirno"
    d = np.load(HERE / "fights" / f"{name}.npz")
    B = d["bullets"]                     # frame,slot,x,y,vx,vy,cls,fxflag
    boss = d["boss"]; player = d["player"]
    f0 = int(B[:, 0].min())
    B[:, 0] -= f0
    if len(boss):
        boss[:, 0] -= f0
    if len(player):
        player[:, 0] -= f0
    nframes = int(B[:, 0].max()) + 1
    by_frame = [None] * nframes
    order = np.argsort(B[:, 0], kind="stable")
    Bs = B[order]
    idx = np.searchsorted(Bs[:, 0], np.arange(nframes + 1))
    for f in range(nframes):
        by_frame[f] = Bs[idx[f]:idx[f + 1]]
    bpos = {int(r[0]): (r[1], r[2]) for r in boss}
    ppos = {int(r[0]): (r[1], r[2]) for r in player}
    print(f"{name}: {len(B)} rows, {nframes} frames (~{nframes/60:.0f}s)")

    pygame.init()
    W = int(PW * SCALE) + 2 * MARGIN + 170
    H = int(PH * SCALE) + 2 * MARGIN
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"fight: {name}")
    font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    def ts(x, y):
        return int(MARGIN + x * SCALE), int(MARGIN + y * SCALE)

    frame = 0.0
    speed = 1.0
    paused = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_PERIOD:
                    frame += 1
                if e.key == pygame.K_RIGHT:
                    frame = min(nframes - 1, frame + 30)
                if e.key == pygame.K_LEFT:
                    frame = max(0, frame - 30)
                if e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(6, speed * 1.5)
                if e.key == pygame.K_MINUS:
                    speed = max(0.15, speed / 1.5)
                if e.key == pygame.K_r:
                    frame = 0.0
        if not paused:
            frame += speed
        if frame >= nframes:
            frame = 0.0
        cf = int(frame)

        screen.fill((10, 10, 16))
        pygame.draw.rect(screen, (28, 28, 40),
                         (MARGIN, MARGIN, PW * SCALE, PH * SCALE))
        rows = by_frame[cf]
        for r in rows:
            _, _, x, y, vx, vy, cls, fxf = r
            fxf = int(fxf)
            c = ((120, 210, 255) if fxf == 32 else
                 (255, 170, 90) if fxf in (64, 128, 256) else
                 (235, 235, 235))
            pygame.draw.circle(screen, c, ts(x, y), 3)
        if cf in bpos:
            pygame.draw.circle(screen, (255, 120, 255), ts(*bpos[cf]), 6, 2)
        if cf in ppos:
            pygame.draw.circle(screen, (120, 255, 120), ts(*ppos[cf]), 4)

        px = int(PW * SCALE) + 2 * MARGIN
        for i, ln in enumerate([
            f"frame {cf}/{nframes}",
            f"{speed:.1f}x {'PAUSED' if paused else ''}",
            f"bullets {len(rows)}",
            "", "white=plain", "cyan=curve", "orange=redirect",
            "", "SPACE . <-/-> +/- R",
        ]):
            screen.blit(font.render(ln, True, (200, 200, 210)), (px, MARGIN + i * 20))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
