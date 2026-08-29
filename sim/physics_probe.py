"""Phase 0: measure the real game's movement + collision physics so the
danmaku sim can be centred on reality. Writes sim/physics.json.

Measures:
  * player move speed (unfocused / focused / diagonal), and whether it ramps
  * movement bounds (where the player actually stops at each edge)
  * effective collision distance: min dist(player, bullet) at the frame of death,
    over many deaths of a stationary player  (= player_r + bullet_r)
  * bullet speed distribution + how many bullets change speed/heading
"""
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "native"))
import shm as S            # noqa: E402
from inject import inject  # noqa: E402

PW, PH = 384.0, 448.0


def measure_speed(h, buttons, n=40):
    """Hold `buttons` from the snapshot, return per-frame player displacement."""
    h.reset()
    xs, ys = [], []
    for _ in range(n):
        h.step(buttons, 1)
        xs.append(h.s.player_x)
        ys.append(h.s.player_y)
    xs, ys = np.array(xs), np.array(ys)
    dx = np.diff(xs)
    dy = np.diff(ys)
    step = np.hypot(dx, dy)
    return {
        "per_frame_after_settle": float(np.median(step[5:15])),
        "first_5_frames": [round(float(v), 3) for v in step[:5]],
        "dx_settled": round(float(np.median(dx[5:15])), 3),
        "dy_settled": round(float(np.median(dy[5:15])), 3),
    }


def measure_bounds(h):
    b = {}
    for name, btn, comp, sign in [("left", S.LEFT, "x", -1), ("right", S.RIGHT, "x", 1),
                                  ("up", S.UP, "y", -1), ("down", S.DOWN, "y", 1)]:
        h.reset()
        for _ in range(160):
            h.step(btn, 1)
        b[name] = round(h.s.player_x if comp == "x" else h.s.player_y, 1)
    return b


def measure_collision_dist(h, deaths=25):
    """Stationary player; at each death log the nearest bullet distance on the
    death frame and the one before it."""
    proc = ctypes.windll.kernel32.OpenProcess(0x10, False, h_pid)
    got = []
    prev_lives = h.s.lives
    h.reset()
    prev_min = None
    steps_since_reset = 0
    while len(got) < deaths and steps_since_reset < 60000:
        h.step(0, 1)
        steps_since_reset += 1
        s = h.s
        xy = np.array([(s.bullets[i].x, s.bullets[i].y) for i in range(S.MAX_BULLETS)
                       if s.bullets[i].x > -9000.0], np.float32).reshape(-1, 2)
        cur_min = None
        if len(xy):
            cur_min = float(np.hypot(xy[:, 0] - s.player_x, xy[:, 1] - s.player_y).min())
        if s.lives < prev_lives - 0.5:
            if prev_min is not None:
                got.append(round(min(prev_min, cur_min if cur_min else 1e9), 2))
            h.reset()
            steps_since_reset = 0
            prev_lives = h.s.lives
            prev_min = None
            continue
        prev_lives = s.lives
        prev_min = cur_min
    return got


def measure_bullets(h, frames=900):
    """Track bullets across their lifetime -> speed + how many change heading."""
    h.reset()
    prev = {}
    speeds = []
    heading_changes = 0
    speed_changes = 0
    tracked = 0
    for _ in range(frames):
        h.step(0, 1)
        s = h.s
        for i in range(S.MAX_BULLETS):
            bx, by = s.bullets[i].x, s.bullets[i].y
            if bx <= -9000.0:
                prev.pop(i, None)
                continue
            if i in prev:
                pbx, pby, pvx, pvy = prev[i]
                vx, vy = bx - pbx, by - pby
                sp = (vx * vx + vy * vy) ** 0.5
                if sp > 0.01:
                    speeds.append(sp)
                if pvx is not None:
                    pv = (pvx * pvx + pvy * pvy) ** 0.5
                    if pv > 0.5 and sp > 0.5:
                        if abs(sp - pv) / pv > 0.15:
                            speed_changes += 1
                        cross = pvx * vy - pvy * vx
                        if abs(cross) > 0.5:
                            heading_changes += 1
                        tracked += 1
                prev[i] = (bx, by, vx, vy)
            else:
                prev[i] = (bx, by, None, None)
    speeds = np.array(speeds)
    return {
        "count_samples": int(len(speeds)),
        "speed_px_per_frame": {
            "p10": round(float(np.percentile(speeds, 10)), 2),
            "p50": round(float(np.percentile(speeds, 50)), 2),
            "p90": round(float(np.percentile(speeds, 90)), 2),
            "max": round(float(speeds.max()), 2),
        },
        "frac_speed_changing": round(speed_changes / max(tracked, 1), 3),
        "frac_curving": round(heading_changes / max(tracked, 1), 3),
    }


if __name__ == "__main__":
    h_pid = inject()
    h = S.Hook(h_pid)
    assert h.autonav(), "autonav failed"
    for _ in range(90):
        h.step(0, 1)
    assert h.snapshot()
    print(f"pid {h_pid} ready, measuring...")

    out = {}
    out["move_unfocused"] = measure_speed(h, S.LEFT)
    out["move_focused"] = measure_speed(h, S.LEFT | S.SLOW)
    out["move_diag_unfocused"] = measure_speed(h, S.LEFT | S.UP)
    out["move_diag_focused"] = measure_speed(h, S.LEFT | S.UP | S.SLOW)
    print("  speeds done")
    out["bounds"] = measure_bounds(h)
    print("  bounds done")
    out["bullets"] = measure_bullets(h)
    print("  bullets done")
    out["collision_dist_at_death"] = measure_collision_dist(h)
    print("  collision done")

    p = HERE / "physics.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {p}")

    h.set_free()
    h.close()
    ctypes.windll.kernel32.TerminateProcess(
        ctypes.windll.kernel32.OpenProcess(1, False, h_pid), 0)
