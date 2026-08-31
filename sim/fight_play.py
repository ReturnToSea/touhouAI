"""Play a recorded real fight yourself - control an INVINCIBLE character through
the replayed bullets to eyeball whether the re-aiming is right.

    .venv-cuda/Scripts/python sim/fight_play.py letty_8

Aimed bullets (their spawn velocity points at where the recording's player was)
are rotated about their spawn point to point at YOU instead. Toggle with [A] to
compare against the raw recording.

Keys: arrows move | shift focus | A re-aim on/off | space pause | . step
       , back | +/- speed | R restart | G show recorded-player ghost | esc quit
Colours: white plain, cyan curve(fx32), orange redirect(fx64+), yellow = aimed
         (currently re-aimed at you); red ring = lethal enemy body.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

HERE = Path(__file__).resolve().parent
PW, PH, SCALE, MARGIN = 384, 448, 1.7, 20
MOVE, MOVE_FOCUS = 4.0, 1.6
AIM_TOL = math.radians(24)          # spawn-vel within this of the rec player -> "aimed"


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def load(name):
    d = np.load(HERE / "fights" / f"{name}.npz")
    B = d["bullets"].astype(np.float64)      # frame,slot,x,y,vx,vy,cls,fx[,hbx,hby]
    EN = d["enemies"].astype(np.float64) if "enemies" in d else np.zeros((0, 8))
    P = d["player"].astype(np.float64)       # step,x,y
    f0 = int(B[:, 0].min())
    B[:, 0] -= f0
    if len(EN):
        EN[:, 0] -= f0
    if len(P):
        P[:, 0] -= f0
    F = int(B[:, 0].max()) + 1

    # recorded player position per frame (forward-filled)
    rp = np.full((F, 2), np.nan)
    if len(P):
        pf = np.clip(P[:, 0].astype(int), 0, F - 1)
        rp[pf] = P[:, 1:3]
    for i in range(1, F):
        if np.isnan(rp[i, 0]):
            rp[i] = rp[i - 1]
    rp = np.nan_to_num(rp, nan=192.0)

    # per-frame bullet rows + per-(slot,run) birth info
    order = np.argsort(B[:, 0], kind="stable")
    B = B[order]
    idx = np.searchsorted(B[:, 0], np.arange(F + 1))
    by_frame = [B[idx[f]:idx[f + 1]] for f in range(F)]

    # walk slots: a slot going absent->present starts a new bullet. record, per
    # (spawn_frame, slot): spawn xy, whether it's "aimed" at the rec player, and
    # the rec angle-to-player at spawn (the re-aim reference).
    last_seen = {}
    aimed_key = {}                    # (spawn_frame, slot) -> (aimed, spawn_x, spawn_y, rec_angle)
    for f in range(F):
        for r in by_frame[f]:
            s = int(r[1])
            if last_seen.get(s, -99) != f - 1:       # new bullet in this slot
                x, y, vx, vy = r[2], r[3], r[4], r[5]
                va = math.atan2(vy, vx) if (vx or vy) else 0.0
                ta = math.atan2(rp[f, 1] - y, rp[f, 0] - x)
                aimed_key[(f, s)] = (abs(_angdiff(va, ta)) < AIM_TOL, x, y, ta)
            last_seen[s] = f
    return dict(F=F, by_frame=by_frame, rp=rp, EN=EN, boss=d["boss"],
                aimed_key=aimed_key)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "letty_8"
    R = load(name)
    F, by_frame, rp = R["F"], R["by_frame"], R["rp"]
    aimed_key = R["aimed_key"]
    EN = R["EN"]
    en_by_frame = {}
    for r in EN:
        en_by_frame.setdefault(int(r[0]), []).append((r[2], r[3], r[5]))
    boss = {int(r[0] - R["boss"][:, 0].min()): (r[1], r[2]) for r in R["boss"]} \
        if len(R["boss"]) else {}

    # build, per frame, a list of (x,y,fx, spawn_x,spawn_y, aimed, rec_ang)
    frames = []
    cur_sf = {}                       # slot -> spawn frame it's currently on
    for f in range(F):
        present = {int(r[1]): r for r in by_frame[f]}
        out = []
        for s, r in present.items():
            if (f, s) in aimed_key:                   # born this frame
                cur_sf[s] = f
            sf = cur_sf.get(s, f)
            info = aimed_key.get((sf, s))
            if info is None:                          # started before frame 0
                aimed, sx, sy, rec_ang = False, r[2], r[3], 0.0
            else:
                aimed, sx, sy, rec_ang = info
            out.append((r[2], r[3], int(r[7]), sx, sy, aimed, rec_ang))
        for s in list(cur_sf):
            if s not in present:
                del cur_sf[s]
        frames.append(out)
    naimed = sum(1 for fr in frames for b in fr if b[5])
    print(f"{name}: {F} frames (~{F/60:.0f}s), "
          f"{naimed} aimed bullet-frames of {sum(len(fr) for fr in frames)}")

    pygame.init()
    W = int(PW * SCALE) + 2 * MARGIN + 180
    H = int(PH * SCALE) + 2 * MARGIN
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"fight-play: {name}")
    font = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()

    def ts(x, y):
        return int(MARGIN + x * SCALE), int(MARGIN + y * SCALE)

    px, py = 192.0, 380.0
    fr = 0.0
    speed = 1.0
    paused = False
    reaim = True
    ghost = True
    deaths = 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_a:
                    reaim = not reaim
                if e.key == pygame.K_g:
                    ghost = not ghost
                if e.key == pygame.K_PERIOD:
                    fr = min(F - 1, fr + 1); paused = True
                if e.key == pygame.K_COMMA:
                    fr = max(0, fr - 1); paused = True
                if e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(4, speed * 1.5)
                if e.key == pygame.K_MINUS:
                    speed = max(0.1, speed / 1.5)
                if e.key == pygame.K_r:
                    fr = 0.0; px, py = 192.0, 380.0; deaths = 0

        keys = pygame.key.get_pressed()
        sp = MOVE_FOCUS if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else MOVE
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP])
        if dx or dy:
            n = (dx * dx + dy * dy) ** 0.5
            px = min(PW - 4, max(4, px + dx / n * sp))
            py = min(PH - 4, max(4, py + dy / n * sp))

        if not paused:
            fr += speed
        if fr >= F:
            fr = 0.0
        cf = int(fr)

        screen.fill((10, 10, 16))
        pygame.draw.rect(screen, (26, 26, 38),
                         (MARGIN, MARGIN, PW * SCALE, PH * SCALE))

        touched = False
        for (bx, by, fx, sx, sy, aimed, rec_ang) in frames[cf]:
            if aimed and reaim:
                new_ang = math.atan2(py - sy, px - sx)
                dth = new_ang - rec_ang
                c, s = math.cos(dth), math.sin(dth)
                rx, ry = bx - sx, by - sy
                bx, by = sx + rx * c - ry * s, sy + rx * s + ry * c
            col = ((240, 230, 80) if aimed and reaim else
                   (120, 210, 255) if fx == 32 else
                   (255, 170, 90) if fx in (64, 128, 256) else
                   (225, 225, 225))
            pygame.draw.circle(screen, col, ts(bx, by), 3)
            if abs(bx - px) < 3.8 and abs(by - py) < 3.8:
                touched = True

        for (ex, ey, hbx) in en_by_frame.get(cf, ()):
            if 0.5 < hbx <= 15:
                r = max(3, int(hbx * 0.5 * SCALE))
                pygame.draw.circle(screen, (255, 70, 70), ts(ex, ey), r, 1)
                if abs(ex - px) < hbx * 0.5 * 0.667 + 2 and \
                   abs(ey - py) < hbx * 0.5 * 0.667 + 2:
                    touched = True
        if cf in boss:
            pygame.draw.circle(screen, (255, 120, 255), ts(*boss[cf]), 7, 2)
        if ghost:
            pygame.draw.circle(screen, (90, 120, 90), ts(rp[cf, 0], rp[cf, 1]), 4, 1)

        if touched:
            deaths += 1
        pygame.draw.circle(screen, (120, 255, 120), ts(px, py), 5)
        pygame.draw.circle(screen, (255, 255, 255), ts(px, py), 2)
        if touched:
            pygame.draw.circle(screen, (255, 60, 60), ts(px, py), 9, 2)

        info = [
            f"frame {cf}/{F}   {speed:.1f}x{'  PAUSED' if paused else ''}",
            f"re-aim: {'ON' if reaim else 'OFF (raw)'}   [A]",
            f"ghost: {'on' if ghost else 'off'}   [G]",
            f"bullets {len(frames[cf])}",
            f"touch-frames {deaths}",
            "", "yellow = aimed@you", "green = you (invincible)",
            "grey ring = rec player", "red ring = lethal orb",
            "", "arrows move / shift focus",
        ]
        for i, ln in enumerate(info):
            screen.blit(font.render(ln, True, (200, 200, 210)),
                        (int(PW * SCALE) + 2 * MARGIN, MARGIN + i * 20))
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
