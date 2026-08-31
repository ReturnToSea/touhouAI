"""A CPU interpreter for th07 ECL (the "old" TH06-09 format). Runs one boss
script forward and records every bullet spawn as a schedule entry, so the GPU
sim can replay real Touhou patterns.

Scope: enough opcodes for Stage 1 (Cirno + Letty). Not a general/faithful VM -
lasers, some effect opcodes, and player-reactive control flow are stubbed. The
boss's control flow is NOT player-reactive (only aim is), so we run once with a
fixed reference player and mark `*_aimed` spawns for per-episode re-aiming.

    from ecl_vm import run_boss
    sched = run_boss("ecldata1_raw_named.tecl", sub=29, difficulty=3, frames=3600)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ecl_parse import parse, Instr

TAU = 2 * math.pi

# reference player during the one-shot CPU run (playfield 384x448, origin TL)
REF_PLAYER = (192.0, 400.0)

# gvar name -> (kind, slot)   kind: 'i' int reg, 'f' float reg, 'ro' read-only
_GVARS = {}
for i in range(8):
    _GVARS[f"I{i}"] = ("i", i)
    _GVARS[f"F{i}"] = ("f", i)
for k in ("PARAM_A PARAM_B PARAM_C PARAM_D PARAM_R PARAM_S PARAM_M PARAM_N "
          "ARG_A ARG_B ARG_C ARG_D ARG_R ARG_S ARG_M ARG_N").split():
    _GVARS[k] = ("f", k)          # treat all PARAM/ARG as a named float slot
_RO = ("DIFFICULTY RANK SELF_X SELF_Y SELF_Z PLAYER_X PLAYER_Y PLAYER_Z "
       "PLAYER_ANGLE SELF_TIME PLAYER_DISTANCE SELF_LIFE SELF_ANGLE "
       "SELF_ANGLE_VEL CIRCLE_ANGLE CIRCLE_SPEED DIST_ORIGIN "
       "ORIGIN_X ORIGIN_Y RANDF2 RANDF_RANGE").split()
for k in _RO:
    _GVARS[k] = ("ro", k)


@dataclass
class Spawn:
    frame: int
    x: float
    y: float
    opcode: str        # bullet_circle / bullet_fan / bullet_random / *_aimed
    count: int         # bullets per shot (ring size / fan width)
    shots: int         # stacked shots (concentric rings)
    speed: float
    speed2: float
    base_angle: float  # non-aimed: absolute; aimed: OFFSET from toward-player
    spread: float      # fan: gap between bullets; circle: per-shot rotation
    aimed: bool
    sprite: int
    aim_ref: float     # boss->refplayer angle at spawn (for the taint-aimed case)


@dataclass
class VM:
    ins: list[Instr]
    diff: int
    ireg: list = field(default_factory=lambda: [0] * 8)
    freg: dict = field(default_factory=dict)          # 'F0'.. + PARAM_*/ARG_*
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    angle: float = 0.0
    speed: float = 0.0
    accel: float = 0.0
    ang_vel: float = 0.0
    t: int = 0                # sub time (frames)
    pc: int = 0
    stack: list = field(default_factory=list)         # (ins, pc, t) return frames
    dead: bool = False
    move: object = None       # (t0, dur, x0,y0, x1,y1, ease) or dir-move tuple
    shoot_interval: int = 0
    shoot_last: int = -999
    cur_bullet: tuple | None = None      # last bullet_* spec, for shoot_interval
    invuln: bool = True
    _taint: set = field(default_factory=set)          # float slots derived from PLAYER_*


class Boss:
    def __init__(self, subs, difficulty, rng, dps=0.0):
        self.subs = subs
        self.diff = difficulty
        self.rng = rng
        self.dps = dps                     # simulated player damage/sec (0 = timer only)
        self.vms: list[VM] = []
        self.sched: list[Spawn] = []
        self.clears: list[int] = []        # frames where the screen was cleared
        self.frame = 0
        self.life = 999999
        self.root: VM | None = None        # the boss controller VM
        self.phase_start = 0
        self.timer_cb = None               # (threshold_frames, sub_id)
        self.life_cb = None                # (threshold_hp, sub_id)
        self.death_cb = None               # sub_id
        self.phase_life0 = 999999

    # trigger a phase transition: interrupt the controller, jump to `sub`
    def _goto_phase(self, sub):
        r = self.root
        r.ins = self.subs[int(sub)]
        r.pc, r.t, r.stack = 0, 0, []
        r.move = r.cur_bullet = None
        r.speed = r.accel = r.ang_vel = 0.0
        for v in self.vms:
            if v is not r:
                v.dead = True
        self.phase_start = self.frame
        self.phase_life0 = self.life
        self.timer_cb = self.life_cb = None

    # ---- gvar access -------------------------------------------------------
    def _get(self, vm: VM, a):
        if not isinstance(a, str):
            return a
        kind = _GVARS.get(a, ("f", a))
        if kind[0] == "i":
            return vm.ireg[kind[1]]
        if kind[0] == "f":
            return vm.freg.get(kind[1], 0.0)
        # read-only
        px, py = REF_PLAYER
        return {
            "DIFFICULTY": self.diff, "RANK": 8,
            "SELF_X": vm.x, "SELF_Y": vm.y, "SELF_Z": vm.z,
            "SELF_ANGLE": vm.angle, "SELF_TIME": float(vm.t),
            "SELF_LIFE": float(self.life), "SELF_ANGLE_VEL": vm.ang_vel,
            "PLAYER_X": px, "PLAYER_Y": py, "PLAYER_Z": 0.0,
            "PLAYER_ANGLE": math.atan2(py - vm.y, px - vm.x),
            "PLAYER_DISTANCE": math.hypot(px - vm.x, py - vm.y),
            "CIRCLE_ANGLE": 0.0, "CIRCLE_SPEED": 0.0, "DIST_ORIGIN": 0.0,
            "ORIGIN_X": 0.0, "ORIGIN_Y": 0.0, "RANDF2": self.rng.random(),
            "RANDF_RANGE": 0.0,
        }.get(a, 0.0)

    def _set(self, vm: VM, a, val, tainted=False):
        kind = _GVARS.get(a, ("f", a))
        if kind[0] == "i":
            vm.ireg[kind[1]] = int(val)
        else:
            vm.freg[kind[1]] = float(val)
            (vm._taint.add if tainted else vm._taint.discard)(kind[1])

    def _tainted(self, vm, a):
        if isinstance(a, str):
            k = _GVARS.get(a, ("f", a))
            if k[0] == "f":
                return k[1] in vm._taint
            if k[0] == "ro":
                return a.startswith("PLAYER")
        return False

    # ---- spawn a child VM (enemy_create_rel) ------------------------------
    def spawn_enemy(self, parent: VM, sub, dx, dy):
        child = VM(self.subs[sub], self.diff)
        child.x, child.y = parent.x + dx, parent.y + dy
        # PARAM_* of the child inherit the parent's PARAM_* (ECL convention)
        for k in ("PARAM_R", "PARAM_S", "PARAM_M", "PARAM_N",
                  "PARAM_A", "PARAM_B", "PARAM_C", "PARAM_D"):
            child.freg[k] = parent.freg.get(k, 0.0)
        self.vms.append(child)

    # ---- one frame ------------------------------------------------------
    def step(self):
        # simulated boss damage + phase-transition callbacks (whole-fight)
        if self.dps and self.life < 1e8:
            self.life -= self.dps / 60.0
        pt = self.frame - self.phase_start
        if self.root and not self.root.dead:
            if self.life_cb and self.life <= self.life_cb[0]:
                self._goto_phase(self.life_cb[1])
            elif self.timer_cb and pt >= self.timer_cb[0]:
                self._goto_phase(self.timer_cb[1])
        for vm in list(self.vms):
            if vm.dead:
                continue
            self._run_vm_frame(vm)
        # controller finished a phase with a death callback pending -> next phase
        if self.root and self.root.dead and self.death_cb is not None:
            self.root.dead = False
            nxt, self.death_cb = self.death_cb, None
            self._goto_phase(nxt)
        self.vms = [v for v in self.vms if not v.dead]
        self.frame += 1

    def _run_vm_frame(self, vm: VM):
        guard = 0
        while vm.pc < len(vm.ins) and guard < 5000:
            guard += 1
            ins = vm.ins[vm.pc]
            if ins.time > vm.t:
                break
            if ins.label is not None or ins.op is None:
                vm.pc += 1
                continue
            if ins.diff != "*" and _DIFF_CHR[self.diff] not in ins.diff:
                vm.pc += 1
                continue
            jumped = self._exec(vm, ins)
            if vm.dead:
                return
            if not jumped:
                vm.pc += 1
        # movement + rotation
        self._advance_motion(vm)
        # shoot_interval auto-fire
        if vm.shoot_interval > 0 and vm.cur_bullet and \
                self.frame - vm.shoot_last >= vm.shoot_interval:
            vm.shoot_last = self.frame
            self._emit(vm, *vm.cur_bullet)
        vm.t += 1

    # ---- motion --------------------------------------------------------
    def _advance_motion(self, vm: VM):
        vm.speed += vm.accel
        vm.angle += vm.ang_vel
        if vm.move is None:
            if vm.speed:
                vm.x += vm.speed * math.cos(vm.angle)
                vm.y += vm.speed * math.sin(vm.angle)
            return
        # movement uses GLOBAL frame, not sub-time (sub-time resets on loops)
        kind = vm.move[0]
        if kind == "point":
            _, f0, dur, x0, y0, x1, y1, ease = vm.move
            p = min(1.0, (self.frame - f0) / max(1, dur))
            pe = ease(p)
            vm.x = x0 + (x1 - x0) * pe
            vm.y = y0 + (y1 - y0) * pe
            if p >= 1.0:
                vm.move = None
        elif kind == "dir":
            _, f0, dur, ang, spd = vm.move
            vm.x += spd * math.cos(ang)
            vm.y += spd * math.sin(ang)
            if self.frame - f0 >= dur:
                vm.move = None

    # ---- emit a bullet spawn -----------------------------------------
    def _emit(self, vm, opcode, args, aim_tainted=False):
        anim, spr_off, count, shots, spd, spd2, launch, spread, sprite = \
            [self._get(vm, a) for a in args]
        px, py = REF_PLAYER
        aim_ref = math.atan2(py - vm.y, px - vm.x)
        aimed = opcode.endswith("_aimed") or aim_tainted
        base = launch
        self.sched.append(Spawn(
            frame=self.frame, x=vm.x, y=vm.y, opcode=opcode,
            count=int(count), shots=int(shots), speed=float(spd),
            speed2=float(spd2), base_angle=float(base), spread=float(spread),
            aimed=aimed, sprite=int(sprite), aim_ref=aim_ref))

    # ---- execute one instruction; return True if it jumped ------------
    def _exec(self, vm: VM, ins: Instr) -> bool:
        op, A = ins.op, ins.args
        g = lambda i: self._get(vm, A[i])            # noqa: E731

        if op == "nop":
            return False
        if op == "delete":
            vm.dead = True
            return False
        if op == "ret":
            if vm.stack:
                vm.ins, vm.pc, vm.t = vm.stack.pop()
                return True
            vm.dead = True
            return False
        if op == "call":
            sub = A[0]
            vm.stack.append((vm.ins, vm.pc + 1, vm.t))
            # ECL calling convention: caller's ARG_* -> callee's PARAM_*
            for a, p in (("ARG_A", "PARAM_A"), ("ARG_B", "PARAM_B"),
                         ("ARG_C", "PARAM_C"), ("ARG_D", "PARAM_D"),
                         ("ARG_R", "PARAM_R"), ("ARG_S", "PARAM_S"),
                         ("ARG_M", "PARAM_M"), ("ARG_N", "PARAM_N")):
                vm.freg[p] = vm.freg.get(a, 0.0)
                if a in vm._taint:
                    vm._taint.add(p)
            vm.ins = self.subs[int(sub)]
            vm.pc, vm.t = 0, 0
            return True

        # --- jumps ---
        if op in ("jump", "jump_dec") or op.startswith("jump_"):
            return self._jump(vm, op, A)

        # --- set / math ---
        if op == "set_int":
            self._set(vm, A[0], g(1))
            return False
        if op == "set_float":
            self._set(vm, A[0], g(1), self._tainted(vm, A[1]))
            return False
        if op == "set_int_rand_bound":
            self._set(vm, A[0], self._rng().randint(0, max(0, int(g(1)) - 1)))
            return False
        if op in ("set_float_rand_bound", "set_float_rand_bound_min"):
            hi, lo = g(1), (g(2) if len(A) > 2 else 0.0)
            self._set(vm, A[0], self._rng().uniform(min(lo, hi), max(lo, hi)))
            return False
        if op == "set_int_rand_sign":
            self._set(vm, A[0], int(g(1)) * self._rng().choice((-1, 1)))
            return False
        if op == "set_float_rand_sign":
            self._set(vm, A[0], g(1) * self._rng().choice((-1.0, 1.0)))
            return False
        if op in ("__math_rand_rad", "__math_rand"):
            self._set(vm, A[0], self._rng().uniform(g(1), g(2)))
            return False
        if op == "math_inc":
            self._set(vm, A[0], self._get(vm, A[0]) + 1)
            return False
        if op == "math_dec":
            self._set(vm, A[0], self._get(vm, A[0]) - 1)
            return False
        _BIN = {
            "math_int_add": lambda a, b: a + b, "math_float_add": lambda a, b: a + b,
            "math_int_sub": lambda a, b: a - b, "math_float_sub": lambda a, b: a - b,
            "math_int_mul": lambda a, b: a * b, "math_float_mul": lambda a, b: a * b,
            "math_int_div": lambda a, b: a // b if b else 0,
            "math_float_div": lambda a, b: a / b if b else 0.0,
            "math_int_mod": lambda a, b: a % b if b else 0,
            "math_float_mod": lambda a, b: math.fmod(a, b) if b else 0.0,
        }
        if op in _BIN:
            tainted = self._tainted(vm, A[1]) or self._tainted(vm, A[2])
            self._set(vm, A[0], _BIN[op](g(1), g(2)), tainted)
            return False
        if op == "math_sin":
            self._set(vm, A[0], math.sin(g(1)), self._tainted(vm, A[1]))
            return False
        if op == "math_cos":
            self._set(vm, A[0], math.cos(g(1)), self._tainted(vm, A[1]))
            return False
        if op == "math_atan2":
            self._set(vm, A[0], math.atan2(g(1), g(2)), True)
            return False
        if op == "math_norm_angle":
            v = (self._get(vm, A[0]) + math.pi) % TAU - math.pi
            self._set(vm, A[0], v, self._tainted(vm, A[0]))
            return False

        # --- movement ---
        if op == "move_position":
            vm.x, vm.y = g(0), g(1)
            return False
        if op == "set_angle":
            vm.angle = g(0)
            return False
        if op == "move_speed" or op == "set_speed":
            vm.speed, vm.move = g(0), None
            return False
        if op == "move_acceleration":
            vm.accel = g(0)
            return False
        if op == "move_angular_velocity":
            vm.ang_vel = g(0)
            return False
        if op == "move_at_player":
            vm.speed = g(1) if len(A) > 1 else vm.speed
            vm.angle = math.atan2(REF_PLAYER[1] - vm.y, REF_PLAYER[0] - vm.x)
            vm.move = None
            return False
        if op == "move_dir_time":       # (dur, ?, angle, speed)
            vm.move = ("dir", self.frame, int(g(0)), g(2), g(3))
            vm.speed = 0.0
            return False
        if op == "move_bounds_set" or op == "move_bounds_disable":
            return False
        if op in ("move_point", "move_position_time"):
            dur = int(g(0))
            mode = int(A[1]) if isinstance(A[1], (int, float)) else 0
            ease = (lambda p: 2 * p - p * p) if mode in (4, 5) else (lambda p: p)
            vm.move = ("point", self.frame, dur, vm.x, vm.y, g(2), g(3), ease)
            return False
        if op == "__move_circle_abs":
            return False        # orbit - approximate as stationary for the PoC

        # --- bullets ---
        if op.startswith("bullet_") and op not in (
                "bullet_effects", "bullet_sound", "bullet_cancel",
                "bullet_cancel_radius", "bullet_clear", "bullet_rank_influence"):
            tainted = any(self._tainted(vm, a) for a in A[6:8])
            vm.cur_bullet = (op, list(A), tainted)
            self._emit(vm, op, list(A), tainted)
            return False
        if op == "shoot_interval":
            vm.shoot_interval = int(g(0))
            vm.shoot_last = self.frame
            return False
        if op in ("shoot_disable",):
            vm.shoot_interval = 0
            return False
        if op == "shoot_now" and vm.cur_bullet:
            self._emit(vm, vm.cur_bullet[0], vm.cur_bullet[1], vm.cur_bullet[2])
            return False

        # --- enemies / structure ---
        if op == "enemy_create_rel" or op == "enemy_create_abs":
            sub = int(A[0])
            dx, dy = g(1), g(2)
            if op == "enemy_create_abs":
                self.spawn_enemy(VM([], self.diff), sub, dx, dy)
            else:
                self.spawn_enemy(vm, sub, dx, dy)
            return False
        if op == "enemy_life_set":
            self.life = int(g(0))
            self.phase_life0 = self.life
            return False
        if op == "enemy_flag_invulnerable":
            vm.invuln = bool(g(0))
            return False

        # --- phase-transition callbacks (root controller only) ---
        if vm is self.root:
            if op == "timer_callback_threshold":
                self._pend_timer_t = int(g(0))
                return False
            if op == "timer_callback_sub":
                self.timer_cb = (getattr(self, "_pend_timer_t", 99999), int(g(0)))
                return False
            if op == "life_callback_threshold":
                self._pend_life_t = int(g(0))
                return False
            if op == "life_callback_sub":
                self.life_cb = (getattr(self, "_pend_life_t", 0), int(g(0)))
                return False
            if op in ("life_callback_ex", "life_callback"):
                # (mode, threshold, sub)  -> fire when life <= threshold
                self.life_cb = (float(g(1)), int(g(2)))
                return False
            if op == "death_callback_sub":
                self.death_cb = int(g(0))
                return False
        if op in ("timer_callback_threshold", "timer_callback_sub",
                  "life_callback_threshold", "life_callback_sub",
                  "life_callback_ex", "life_callback", "death_callback_sub",
                  "enemy_interrupt", "enemy_interrupt_set", "boss_interrupt"):
            return False
        if op in ("enemy_kill_all", "bullet_cancel", "bullet_clear",
                  "laser_clear_all", "bullet_cancel_radius"):
            if op in ("enemy_kill_all",):
                for v in self.vms:
                    if v is not vm:
                        v.dead = True
            self.clears.append(self.frame)   # replay despawns live bullets here
            return False

        # everything else (anm_*, effect_*, *_sound, *_callback, flags, laser_*,
        # spellcard_*, boss_timer_*, trail_set, ...) - no effect on the schedule
        return False

    def _rng(self):
        return self.rng

    def _jump(self, vm: VM, op, A) -> bool:
        # jump(new_t, label); jump_dec(new_t, label, counter);
        # jump_{int,float}_{cmp}(lhs, rhs, new_t, label)
        if op == "jump":
            new_t, label, do = int(A[0]), A[1], True
        elif op == "jump_dec":
            new_t, label = int(A[0]), A[1]
            self._set(vm, A[2], self._get(vm, A[2]) - 1)
            do = self._get(vm, A[2]) > 0
        else:
            l, r = self._get(vm, A[0]), self._get(vm, A[1])
            new_t, label = int(A[2]), A[3]
            cmp = op.rsplit("_", 1)[-1]
            do = {"equ": l == r, "neq": l != r, "lss": l < r, "leq": l <= r,
                  "gre": l > r, "geq": l >= r}.get(cmp, False)
        if not do:
            return False
        for i, x in enumerate(vm.ins):
            if x.label == label:
                vm.pc, vm.t = i, new_t
                return True
        return False


_DIFF_CHR = {0: "E", 1: "N", 2: "H", 3: "L", 4: "L"}


def run_boss(tecl_path, sub, difficulty=3, frames=3600, seed=0, dps=0.0):
    """`sub` = the boss *controller* sub (Cirno=20, Letty=38). Phase transitions
    (non-spell -> spell -> ...) fire via the timer/life/death callbacks. dps=0 =>
    phases run to their timer threshold (full patterns); dps>0 simulates player
    damage so life-callbacks fire earlier."""
    subs = parse(tecl_path)
    boss = Boss(subs, difficulty, random.Random(seed), dps=dps)
    root = VM(subs[int(sub)], difficulty)
    root.x, root.y = 192.0, 120.0
    root.freg["PARAM_R"] = 192.0
    root.freg["PARAM_S"] = 120.0
    boss.root = root
    boss.vms.append(root)
    for _ in range(frames):
        boss.step()
        if not boss.vms and boss.root.dead:
            break
    boss.sched.sort(key=lambda s: s.frame)
    return boss.sched, sorted(set(boss.clears))


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    sub = int(sys.argv[2]) if len(sys.argv) > 2 else 29
    fr = int(sys.argv[3]) if len(sys.argv) > 3 else 1800
    sched, clears = run_boss(p, sub, difficulty=3, frames=fr)
    print(f"{len(sched)} spawn events over {fr} frames")
    from collections import Counter
    print("by opcode:", Counter(s.opcode for s in sched))
    tot = sum(max(1, s.count) * max(1, s.shots) for s in sched)
    print(f"~{tot} bullets total")
    for s in sched[:12]:
        print(f"  f{s.frame:<4} ({s.x:6.1f},{s.y:6.1f}) {s.opcode:20s} "
              f"n={s.count} shots={s.shots} spd={s.speed:.2f} "
              f"ang={s.base_angle:+.2f} aimed={s.aimed}")
