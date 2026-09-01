"""ECL virtual machine — Part 2: control flow.

Runs a boss's decompiled ECL frame by frame: instruction pointer, per-sub frame
gate, call stack, variable slots, difficulty gate, and the callback / interrupt
machinery that drives phase transitions (`timer_callback_*`, `life_callback_*`,
`death_callback_sub`, `enemy_interrupt_set`).

Not implemented here: arithmetic beyond what control flow needs (Part 3), bullet
emission (Part 5), sub-enemies (Part 6), the damage model (Part 7), boss motion
(Part 8). Every other opcode is dispatched to a handler; unregistered ones are
recorded once in `vm.unhandled` and otherwise ignored, so later parts plug in.

    from sim.ecl import parse_file
    from sim.ecl.vm import VM
    vm = VM(parse_file("tools/th07_ecl/ecldata1.ecl"), difficulty=3)
    vm.start_boss(sub=31, interrupt=0)      # Letty: Sub31 -> interrupt 0 -> Sub38
    vm.run(12000)
    for f, ev, detail in vm.trace:
        print(f, ev, detail)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .parser import ECLFile, Instr, Var
from .rng import EclRng

# gvar id ranges (th07.eclm !gvar_names)
_I_LO, _I_HI = 10000, 10003        # I0..I3  -> slots 0..3
_F_LO, _F_HI = 10004, 10011        # F0..F7  -> slots 0..7
_I2_LO, _I2_HI = 10012, 10015      # I4..I7  -> slots 4..7
_F2_LO, _F2_HI = 10072, 10073      # F8..F9  -> slots 8..9

_CMP = {
    28: lambda a, b: a == b, 29: lambda a, b: a == b,
    30: lambda a, b: a != b, 31: lambda a, b: a != b,
    32: lambda a, b: a < b,  33: lambda a, b: a < b,
    34: lambda a, b: a <= b, 35: lambda a, b: a <= b,
    36: lambda a, b: a > b,  37: lambda a, b: a > b,
    38: lambda a, b: a >= b, 39: lambda a, b: a >= b,
}


_ARG_LO, _PARAM_LO, _ARG_N = 10037, 10029, 8   # ARG_A..ARG_N -> PARAM_A..PARAM_N


@dataclass
class _CallFrame:
    sub: int
    ip: int
    frame: int
    ivars: list
    fvars: list
    extra: dict


@dataclass
class _Motion:
    """A movement command in progress. `kind='linear'` interpolates position
    over `duration` frames from `(x0,y0,z0)` to `(x1,y1,z1)`; `kind='circle'`
    orbits `(cx,cy,cz)` at `radius`, `angle0 + ang_speed*t` — the center and
    radius are fixed at the moment the command was issued (evaluating a
    self-referential `SELF_X` every frame would collapse the radius to 0)."""
    kind: str
    start_frame: int
    duration: int
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    angle0: float = 0.0
    radius: float = 0.0
    ang_speed: float = 0.0


@dataclass
class BulletSpawn:
    """One spawn event — a bullet coming into existence. Motion (delay, accel,
    curve from `effect`) is Stage B; here we record the launch parameters."""
    frame: int
    kind: str                 # "fan" | "circle" | "random"
    btype: int                # bullet graphic/type id (the last opcode arg)
    x: float
    y: float
    angle: float              # launch angle (rad); toward the player if aimed
    speed: float
    aimed: bool
    effect: tuple | None      # the bullet_effects args in force at spawn
    source_sub: int
    source_ip: int = -1       # instruction index within source_sub that fired it


class Enemy:
    """One ECL-scripted entity. Owns variables, position, life, and callbacks."""

    def __init__(self, vm: "VM", sub: int, is_boss: bool):
        self.vm = vm
        self.is_boss = is_boss
        self.alive = True
        self.removed = False

        self.ivars = [0] * 8            # I0..I7
        self.fvars = [0.0] * 10         # F0..F9
        self.extra: dict[int, float] = {}   # ARG_*/PARAM_*/CIRCLE_*/ORIGIN_* etc
        self.x = self.y = self.z = 0.0
        self.life = 0
        self.max_life = 0
        self.time = 0                   # SELF_TIME / timeout counter (gvar 10025)

        # callbacks — persist across sub switches until re-declared
        self.death_sub: int | None = None
        self.timer_thresh: int | None = None
        self.timer_sub: int | None = None
        self.life_thresh: int | None = None
        self.life_sub: int | None = None

        self.spell: tuple[int, int] | None = None   # (group, number) while a spellcard is up
        self.armored_until = 0
        self.invulnerable = False
        self.interrupts: dict[int, int] = {}
        self.pending_interrupt: int | None = None

        self.shoot_offset = (0.0, 0.0, 0.0)
        self.pending_effect: tuple | None = None    # last bullet_effects, attached to the next spawn
        self.motion: _Motion | None = None

        # execution state
        self.sub = sub
        self.ip = 0
        self.frame = 0                  # per-sub instruction-time gate
        self.wait_count = 0             # wait(N): freeze `frame` for N frames
        self.stack: list[_CallFrame] = []
        self.running = True

    # -- variable access ---------------------------------------------------
    def get(self, v):
        return self._gvar(v.id) if isinstance(v, Var) else v

    def _gvar(self, gid: int):
        if _I_LO <= gid <= _I_HI:
            return self.ivars[gid - _I_LO]
        if _F_LO <= gid <= _F_HI:
            return self.fvars[gid - _F_LO]
        if _I2_LO <= gid <= _I2_HI:
            return self.ivars[4 + gid - _I2_LO]
        if _F2_LO <= gid <= _F2_HI:
            return self.fvars[8 + gid - _F2_LO]
        vm = self.vm
        ro = {
            10016: vm.difficulty, 10017: vm.rank,
            10018: self.x, 10019: self.y, 10020: self.z,
            10021: vm.player_x, 10022: vm.player_y, 10023: vm.player_z,
            10025: self.time, 10027: self.life,
        }
        if gid in ro:
            return ro[gid]
        return self.extra.get(gid, 0)

    def set(self, dst, value):
        gid = dst.id if isinstance(dst, Var) else dst
        if _I_LO <= gid <= _I_HI:
            self.ivars[gid - _I_LO] = int(value)
        elif _F_LO <= gid <= _F_HI:
            self.fvars[gid - _F_LO] = float(value)
        elif _I2_LO <= gid <= _I2_HI:
            self.ivars[4 + gid - _I2_LO] = int(value)
        elif _F2_LO <= gid <= _F2_HI:
            self.fvars[8 + gid - _F2_LO] = float(value)
        elif gid == 10018:
            self.x = value
        elif gid == 10019:
            self.y = value
        elif gid == 10020:
            self.z = value
        elif gid == 10025:
            self.time = int(value)
        elif gid == 10027:
            self.life = int(value)
        elif gid in (10016, 10017, 10021, 10022, 10023):
            pass                          # read-only specials
        else:
            self.extra[gid] = value       # ARG_*/PARAM_*/misc

    def switch_to(self, sub: int, *, reason: str):
        self.sub = sub
        self.ip = 0
        self.frame = 0
        self.stack = []
        self.running = True
        self.spell = None
        self.vm._emit("enter_sub", f"Sub{sub}  <{reason}>")

    def damage(self, amount: int):
        """Part 7 will call this; a dodge-only run never does."""
        if not self.alive or self.invulnerable or self.vm.frame < self.armored_until:
            return
        self.life = max(0, self.life - amount)


class VM:
    def __init__(self, ecl: ECLFile, difficulty: int = 3, rank: int | None = None,
                 seed: int = 0, player: tuple[float, float, float] = (192.0, 400.0, 0.0)):
        self.ecl = ecl
        self.difficulty = difficulty
        self.rank = difficulty if rank is None else rank
        self.rank_bit = 1 << difficulty
        self.player_x, self.player_y, self.player_z = player
        self.seed = seed
        self.rng = EclRng(seed)
        self.frame = 0
        self.enemies: list[Enemy] = []
        self.trace: list[tuple[int, str, str]] = []
        self.unhandled: dict[int, str] = {}
        self.bullets: list[BulletSpawn] = []       # the Part 5 spawn schedule
        self._pending_children: list[Enemy] = []
        self.max_children = 4000

    # -- public ----------------------------------------------------------
    def start_boss(self, sub: int, interrupt: int | None = None) -> Enemy:
        e = Enemy(self, sub, is_boss=True)
        self.enemies.append(e)
        self._emit("spawn_boss", f"Sub{sub}")
        if interrupt is not None:
            e.pending_interrupt = interrupt
        self._run_enemy(e)                         # frame-0 block, advances to frame 1
        return e

    def run(self, max_frames: int) -> None:
        while self.frame < max_frames and any(e.alive for e in self.enemies):
            self.step()

    def step(self) -> None:
        for e in list(self.enemies):
            if not e.alive:
                continue
            self._update_motion(e)          # apply any move_* command from a prior frame
            self._service_callbacks(e)      # may switch sub (frame -> 0); does not execute
            if e.alive:
                self._run_enemy(e)          # execute this frame, then e.frame += 1
            e.time += 1
        if self._pending_children:
            self.enemies.extend(self._pending_children)
            self._pending_children = []
        self.enemies = [e for e in self.enemies if e.alive and not e.removed]
        self.frame += 1

    def boss(self) -> Enemy | None:
        return next((e for e in self.enemies if e.is_boss), None)

    def _update_motion(self, e: Enemy) -> None:
        m = e.motion
        if m is None:
            return
        t = self.frame - m.start_frame
        if m.kind == "linear":
            f = 1.0 if m.duration <= 0 else min(1.0, t / m.duration)
            e.x = m.x0 + (m.x1 - m.x0) * f
            e.y = m.y0 + (m.y1 - m.y0) * f
            e.z = m.z0 + (m.z1 - m.z0) * f
            if t >= m.duration:
                e.motion = None
        elif m.kind == "circle":
            ang = m.angle0 + m.ang_speed * t
            e.x = m.cx + m.radius * math.cos(ang)
            e.y = m.cy + m.radius * math.sin(ang)
            e.z = m.cz

    # -- Part 5: bullet spawning & sub-enemies ---------------------------
    def _emit_bullets(self, e: Enemy, kind: str, args: list, source_ip: int = -1) -> None:
        # bullet_{fan,circle,random}[_aimed](t1, t2, count, layers, spd1, spd2,
        #                                    base_angle, span, flags)
        _, _, count, layers, spd1, spd2, base, span, flags = (e.get(a) for a in args)
        count = max(0, int(count))
        layers = max(1, int(layers))
        btype = int(flags)
        aimed = kind.endswith("_aimed")
        k = "fan" if kind.startswith("fan") else ("circle" if kind.startswith("circle") else "random")
        ox, oy, _oz = e.shoot_offset
        sx, sy = e.x + ox, e.y + oy
        if aimed:
            base = base + math.atan2(self.player_y - sy, self.player_x - sx)
        n = count * layers
        lo, hi = min(base, span), max(base, span)     # bullet_random: base/span are hi/lo
        for i in range(n):
            layer = i // max(1, count)
            idx = i % max(1, count)
            if k == "random":
                ang = self.rng.rand_range(lo, hi)      # fresh direction per bullet
                sp = spd1 + spd2 * self.rng.rand()     # spd2 is the speed variance
            elif k == "circle":
                ang = base + (2 * math.pi) * idx / max(1, count)
                sp = spd1 + (spd2 - spd1) * (layer / max(1, layers - 1) if layers > 1 else 0)
            else:  # fan
                ang = base + span * (idx - (count - 1) / 2)
                sp = spd1 + (spd2 - spd1) * (layer / max(1, layers - 1) if layers > 1 else 0)
            self.bullets.append(BulletSpawn(
                frame=self.frame, kind=k, btype=btype, x=sx, y=sy,
                angle=ang, speed=sp, aimed=aimed, effect=e.pending_effect,
                source_sub=e.sub, source_ip=source_ip))

    def _spawn_child(self, parent: Enemy, sub: int, dx: float, dy: float, dz: float) -> None:
        if len(self.enemies) + len(self._pending_children) >= self.max_children:
            return
        c = Enemy(self, sub, is_boss=False)
        c.x, c.y, c.z = parent.x + dx, parent.y + dy, parent.z + dz
        c.extra = dict(parent.extra)          # inherit PARAM_*/ARG_* snapshot
        self._run_enemy(c)                     # child's frame-0 block runs immediately
        self._pending_children.append(c)

    def phase_transitions(self) -> list[tuple[int, int]]:
        """(frame, sub) for every phase-machine sub entry — the Part 2 verify view."""
        return [(f, int(d.split("Sub", 1)[1].split()[0]))
                for f, ev, d in self.trace if ev == "enter_sub"]

    def bullets_per_phase(self, boundaries: list[int]) -> list[int]:
        """Spawn-event counts bucketed by frame into [b0,b1), [b1,b2), ... — the
        Part 5 verify view. `boundaries` is the list of phase-start frames."""
        edges = list(boundaries) + [10**9]
        counts = [0] * (len(edges) - 1)
        for b in self.bullets:
            for i in range(len(counts)):
                if edges[i] <= b.frame < edges[i + 1]:
                    counts[i] += 1
                    break
        return counts

    # -- internals -----------------------------------------------------
    def _emit(self, kind: str, detail: str = "") -> None:
        self.trace.append((self.frame, kind, detail))

    def _fire(self, e: Enemy, target: int | None, reason: str) -> None:
        if target is None:
            return
        e.time = 0
        e.timer_thresh = None
        for other in self.enemies:               # phase transition = screen clear
            if not other.is_boss and other is not e:
                other.alive = False
                other.removed = True
        self._pending_children.clear()
        e.switch_to(target, reason=reason)        # frame -> 0; step() runs the new block

    def _service_callbacks(self, e: Enemy) -> None:
        if e.pending_interrupt is not None:
            iid, e.pending_interrupt = e.pending_interrupt, None
            if iid in e.interrupts:
                self._fire(e, e.interrupts[iid], f"interrupt {iid}")
                return
        if e.timer_thresh is not None and e.time >= e.timer_thresh:
            if e.spell is not None and e.timer_sub is None:
                self._emit("spell_timeout", str(e.spell))
                self._fire(e, e.death_sub, "spell timeout -> death_callback")
            elif e.timer_sub is not None:
                self._fire(e, e.timer_sub, "timer_callback")
            else:
                self._fire(e, e.death_sub, "timeout -> death_callback")
        elif (e.life_thresh is not None and e.life_sub is not None
              and e.life <= e.life_thresh):
            tgt, e.life_thresh = e.life_sub, None
            if e.spell is not None:
                self._emit("spell_captured", str(e.spell))
            self._fire(e, tgt, "life_callback")

    def _run_enemy(self, e: Enemy) -> None:
        if e.wait_count > 0:                            # wait(N): frame frozen
            e.wait_count -= 1
            return
        subs = self.ecl.subs
        guard = 0
        while e.running:
            guard += 1
            if guard > 200000:
                raise RuntimeError(f"Sub{e.sub}@{e.ip}: instruction loop did not settle")
            try:
                ins = subs[e.sub].instrs[e.ip]
            except IndexError:
                e.running = False
                break
            if ins.time > e.frame:
                break
            e.ip += 1
            if ins.time == e.frame and (ins.rank_mask & self.rank_bit or ins.rank_mask == 0):
                self._dispatch(e, ins)
                if e.wait_count > 0:                    # wait just started — hold `frame`
                    return
        e.frame += 1                                   # advance the per-sub time gate

    def _dispatch(self, e: Enemy, ins: Instr) -> None:
        h = _HANDLERS.get(ins.opcode)
        if h is None:
            self.unhandled.setdefault(ins.opcode, ins.name)
            return
        h(self, e, ins)


# ------------------------------------------------------------------- handlers

_HANDLERS: dict[int, "callable"] = {}


def _op(*codes):
    def deco(fn):
        for c in codes:
            _HANDLERS[c] = fn
        return fn
    return deco


@_op(0)
def _nop(vm, e, ins):
    pass


@_op(1)  # delete
def _delete(vm, e, ins):
    e.alive = False
    e.removed = True
    vm._emit("delete", f"Sub{e.sub}")


@_op(2)  # jump(time, target_index)
def _jump(vm, e, ins):
    e.frame, e.ip = ins.args[0], ins.args[1]


@_op(3)  # jump_dec(time, target, counter_var) — loop
def _jump_dec(vm, e, ins):
    t, tgt, var = ins.args
    c = e.get(var) - 1
    if c > 0:
        e.set(var, c)
        e.frame, e.ip = t, tgt


@_op(4, 5)  # set_int / set_float
def _set(vm, e, ins):
    dst, src = ins.args
    e.set(dst, e.get(src))


@_op(*_CMP)  # conditional jumps 28..39
def _cond_jump(vm, e, ins):
    a, b, t, tgt = ins.args
    if _CMP[ins.opcode](e.get(a), e.get(b)):
        e.frame, e.ip = t, tgt


@_op(41)  # call(sub) — snapshot locals + args; map ARG_x -> PARAM_x for the callee
def _call(vm, e, ins):
    e.stack.append(_CallFrame(e.sub, e.ip, e.frame,
                              list(e.ivars), list(e.fvars), dict(e.extra)))
    for k in range(_ARG_N):
        e.extra[_PARAM_LO + k] = e.extra.get(_ARG_LO + k, 0)
    e.sub, e.ip, e.frame = int(e.get(ins.args[0])), 0, 0


@_op(42)  # ret
def _ret(vm, e, ins):
    if not e.stack:
        e.running = False
        return
    fr = e.stack.pop()
    e.sub, e.ip, e.frame = fr.sub, fr.ip, fr.frame
    e.ivars, e.fvars, e.extra = fr.ivars, fr.fvars, fr.extra


@_op(45)  # wait(frames) — freeze `frame` for N frames, then resume at the same time
def _wait(vm, e, ins):
    e.wait_count = max(0, int(e.get(ins.args[0])))


# --- arithmetic (Part 3) -------------------------------------------------

def _c_idiv(a, b):
    return 0 if b == 0 else int(a / b)          # C truncates toward zero


def _c_imod(a, b):
    return 0 if b == 0 else int(a - _c_idiv(a, b) * b)


_BINOP = {
    12: lambda a, b: int(a + b),  13: lambda a, b: int(a - b),
    14: lambda a, b: int(a * b),  15: _c_idiv,  16: _c_imod,
    19: lambda a, b: a + b,  20: lambda a, b: a - b,
    21: lambda a, b: a * b,  22: lambda a, b: a / b if b else 0.0,
    23: lambda a, b: math.fmod(a, b) if b else 0.0,
}


@_op(*_BINOP)  # math_{int,float}_{add,sub,mul,div,mod}  (dst, a, b)
def _binop(vm, e, ins):
    dst, a, b = ins.args
    e.set(dst, _BINOP[ins.opcode](e.get(a), e.get(b)))


@_op(17)  # math_inc(dst)
def _inc(vm, e, ins):
    e.set(ins.args[0], int(e.get(ins.args[0])) + 1)


@_op(18)  # math_dec(dst)
def _dec(vm, e, ins):
    e.set(ins.args[0], int(e.get(ins.args[0])) - 1)


@_op(24)  # math_sin(dst, angle)
def _sin(vm, e, ins):
    e.set(ins.args[0], math.sin(e.get(ins.args[1])))


@_op(25)  # math_cos(dst, angle)
def _cos(vm, e, ins):
    e.set(ins.args[0], math.cos(e.get(ins.args[1])))


@_op(26)  # math_atan2(dst, x1, y1, x2, y2) -> atan2(y2-y1, x2-x1)
def _atan2(vm, e, ins):
    d, x1, y1, x2, y2 = (e.get(a) for a in ins.args)
    e.set(ins.args[0], math.atan2(y2 - y1, x2 - x1))


@_op(40)  # math_norm_angle(dst) — wrap into [-pi, pi)
def _norm_angle(vm, e, ins):
    v = e.get(ins.args[0])
    e.set(ins.args[0], (v + math.pi) % (2 * math.pi) - math.pi)


# --- random (Part 3; the generator itself is Part 4) --------------------

@_op(6)  # set_int_rand_bound(dst, bound) -> [0, bound)
def _rand_i(vm, e, ins):
    e.set(ins.args[0], vm.rng.rand_int(int(e.get(ins.args[1]))))


@_op(7)  # set_int_rand_bound_min(dst, range, min) -> min + [0, range)
def _rand_i_min(vm, e, ins):
    lo = e.get(ins.args[2])
    e.set(ins.args[0], int(lo) + vm.rng.rand_int(int(e.get(ins.args[1]))))


@_op(8)  # set_float_rand_bound(dst, bound) -> [0, bound)
def _rand_f(vm, e, ins):
    e.set(ins.args[0], e.get(ins.args[1]) * vm.rng.rand())


@_op(9)  # set_float_rand_bound_min(dst, range, min) -> min + [0, range)
def _rand_f_min(vm, e, ins):
    e.set(ins.args[0], e.get(ins.args[2]) + e.get(ins.args[1]) * vm.rng.rand())


@_op(10)  # set_int_rand_sign(dst, value) -> +/- value
def _rand_i_sign(vm, e, ins):
    v = int(e.get(ins.args[1]))
    e.set(ins.args[0], v if vm.rng.rand() < 0.5 else -v)


@_op(11)  # set_float_rand_sign(dst, value) -> +/- value
def _rand_f_sign(vm, e, ins):
    v = e.get(ins.args[1])
    e.set(ins.args[0], v if vm.rng.rand() < 0.5 else -v)


@_op(52)  # __math_rand_rad(dst, lo, hi) -> uniform [lo, hi)
def _rand_rad(vm, e, ins):
    e.set(ins.args[0], vm.rng.rand_range(e.get(ins.args[1]), e.get(ins.args[2])))


@_op(51)  # __math_rand(dst, bound) -> [0, bound)  (float)
def _rand(vm, e, ins):
    e.set(ins.args[0], e.get(ins.args[1]) * vm.rng.rand())


@_op(27)  # float_time(dst, mode, duration, ...) — time-interpolation; approximate
def _float_time(vm, e, ins):
    # not used by stage 1; snap to the final target so patterns that do use it
    # don't stall. Revisit if a boss needs the easing curve.
    if len(ins.args) >= 4:
        e.set(ins.args[0], e.get(ins.args[3]))


# --- bullets (Part 5) --------------------------------------------------

_BULLET_KIND = {
    64: "fan_aimed", 65: "fan",
    66: "circle_aimed", 67: "circle",
    68: "circle_aimed", 69: "circle",      # bullet_offset_circle[_aimed]
    70: "random", 71: "random", 72: "random",
}


@_op(*_BULLET_KIND)
def _bullet(vm, e, ins):
    vm._emit_bullets(e, _BULLET_KIND[ins.opcode], ins.args, source_ip=ins.index)


@_op(79)  # bullet_effects(...) — attaches to the next spawn (Stage B models it)
def _bullet_effects(vm, e, ins):
    e.pending_effect = tuple(e.get(a) for a in ins.args)


@_op(78)  # shoot_offset(x, y, z)
def _shoot_offset(vm, e, ins):
    e.shoot_offset = tuple(e.get(a) for a in ins.args[:3])


@_op(73, 74, 75, 76, 80, 81, 143, 146)  # shoot_*, bullet_cancel/clear/radius, bullet_sound
def _shoot_noop(vm, e, ins):
    pass


@_op(94)  # enemy_kill_all — clears every sub-enemy (spell-start screen clear)
def _kill_all(vm, e, ins):
    for other in vm.enemies:
        if not other.is_boss and other is not e:
            other.alive = False
            other.removed = True
    vm._pending_children.clear()


@_op(93)  # enemy_create_rel(sub, x, y, z, ...)
def _create_rel(vm, e, ins):
    sub, dx, dy, dz = (e.get(a) for a in ins.args[:4])
    vm._spawn_child(e, int(sub), dx, dy, dz)


@_op(92)  # enemy_create_abs(sub, x, y, z, ...)
def _create_abs(vm, e, ins):
    sub, x, y, z = (e.get(a) for a in ins.args[:4])
    c = e.__class__(vm, int(sub), is_boss=False)
    c.x, c.y, c.z = x, y, z
    c.extra = dict(e.extra)
    vm._run_enemy(c)
    vm._pending_children.append(c)


# --- phase machine: callbacks & interrupts ---

# --- movement (Part 6/8) -------------------------------------------------

@_op(46)  # move_position(x, y, z) — snap, cancels any motion in progress
def _move_position(vm, e, ins):
    e.x, e.y, e.z = (e.get(a) for a in ins.args)
    e.motion = None


@_op(54)  # move_dir_time(duration, _, angle, speed) — constant heading for `duration` frames
def _move_dir_time(vm, e, ins):
    dur, _u, angle, speed = (e.get(a) for a in ins.args)
    dur = int(dur)
    tx = e.x + math.cos(angle) * speed * dur
    ty = e.y + math.sin(angle) * speed * dur
    e.motion = _Motion("linear", vm.frame, dur, e.x, e.y, e.z, tx, ty, e.z)


@_op(55)  # move_point(duration, _, x, y, z) — linear move to an absolute point
def _move_point(vm, e, ins):
    dur, _u, x, y, z = (e.get(a) for a in ins.args)
    e.motion = _Motion("linear", vm.frame, int(dur), e.x, e.y, e.z, x, y, z)


@_op(56)  # __move_circle_abs(mode, cx, cy, cz, angle, radius, _, ang_speed)
def _move_circle(vm, e, ins):
    _mode, cx, cy, cz, angle, radius, _u, ang_speed = (e.get(a) for a in ins.args)
    e.motion = _Motion("circle", vm.frame, 1 << 30, e.x, e.y, e.z, e.x, e.y, e.z,
                       cx=cx, cy=cy, cz=cz, angle0=angle, radius=radius, ang_speed=ang_speed)


@_op(57)  # set_orbit_distance(new_radius, ...) — retarget an active circle's radius
def _orbit_distance(vm, e, ins):
    if e.motion is not None and e.motion.kind == "circle":
        e.motion.radius = e.get(ins.args[0])


@_op(43, 44, 48, 49, 50, 53, 58, 59, 60, 61, 62, 63)  # boss-relative set, ang-vel/accel/speed
def _move_misc_noop(vm, e, ins):                       # tuning knobs Letty doesn't lean on; no-op
    pass


@_op(107)  # death_callback_sub
def _death_cb(vm, e, ins):
    e.death_sub = int(e.get(ins.args[0]))


@_op(108)  # enemy_interrupt_set(sub, id)
def _int_set(vm, e, ins):
    e.interrupts[int(e.get(ins.args[1]))] = int(e.get(ins.args[0]))


@_op(109)  # enemy_interrupt(id)
def _int_fire(vm, e, ins):
    e.pending_interrupt = int(e.get(ins.args[0]))


@_op(110)  # enemy_life_set
def _life_set(vm, e, ins):
    e.life = e.max_life = int(e.get(ins.args[0]))


@_op(112)  # life_callback_threshold  (-1 disables)
def _life_thresh(vm, e, ins):
    v = int(e.get(ins.args[0]))
    e.life_thresh = None if v < 0 else v
    e.life_sub = None


@_op(113)  # life_callback_sub
def _life_sub(vm, e, ins):
    e.life_sub = int(e.get(ins.args[0]))


@_op(148)  # life_callback_ex(_, threshold, sub)
def _life_ex(vm, e, ins):
    e.life_thresh = int(e.get(ins.args[1]))
    e.life_sub = int(e.get(ins.args[2]))


@_op(114)  # timer_callback_threshold — also resets the timeout counter  (-1 disables)
def _timer_thresh(vm, e, ins):
    v = int(e.get(ins.args[0]))
    e.time = 0
    e.timer_thresh = None if v < 0 else v
    e.timer_sub = None


@_op(115)  # timer_callback_sub
def _timer_sub(vm, e, ins):
    e.timer_sub = int(e.get(ins.args[0]))


@_op(111)  # boss_timer_set(n)
def _boss_timer_set(vm, e, ins):
    e.time = int(e.get(ins.args[0]))


@_op(133)  # boss_timer_clear
def _boss_timer_clear(vm, e, ins):
    pass  # display-only; the following timer_callback_threshold re-arms it


@_op(90)  # spellcard_start(group, number, name)
def _spell_start(vm, e, ins):
    grp = int(e.get(ins.args[0]))
    num = int(e.get(ins.args[1])) if len(ins.args) > 1 else 0
    e.spell = (grp, num)
    name = ins.args[2] if len(ins.args) > 2 and isinstance(ins.args[2], str) else ""
    vm._emit("spellcard_start", f"{e.spell} {name}".rstrip())


@_op(91)  # spellcard_end
def _spell_end(vm, e, ins):
    vm._emit("spellcard_end", str(e.spell))
    e.spell = None


@_op(99)  # boss_set(id) — id < 0 means "no longer the boss"
def _boss_set(vm, e, ins):
    v = int(e.get(ins.args[0]))
    vm._emit("boss_set", str(v))
    if v < 0:
        e.is_boss = False


@_op(103)  # enemy_flag_invulnerable
def _invuln(vm, e, ins):
    e.invulnerable = bool(int(e.get(ins.args[0])))


@_op(142)  # enemy_flag_armored(frames)
def _armored(vm, e, ins):
    e.armored_until = vm.frame + int(e.get(ins.args[0]))
