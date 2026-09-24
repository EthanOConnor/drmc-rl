"""Human-like movement for the named paces: a fitted per-placement model plus an exact generator.

The model (``movement_model_v1.json``, fitted by ``tools/human_movement/fit_movement_model.py``)
holds quantile tables and probabilities per pace profile and situation: reaction, press rhythm,
auto-repeat (DAS) versus tapping, rotation order, tap hold lengths, pauses, descent slack (frames
beyond the fastest drop once steering is complete) and corrections (overshoot-then-back,
wrong-way start, an extra rotation pair, a late sideways move after the drop began). A per-game
style draw shifts all of these through a Gaussian copula fitted from the between-player spread,
so two opponents at one pace do not move identically.

Frame Perfect has no profile: it keeps exact machine movement. For the other paces the planner
still chooses the placement from the named pace's motor-feasible set; only the executed script
changes. ``generate`` realizes the planner's target pose with sampled human timing, replays every
frame with ``simulate_frame``, and returns a script only if it locks exactly at the target. A
failed sample is redrawn with progressively simpler behaviour, then the planner's own route with a
human descent, then the planner's route unchanged: the placement is never lost.

Everything is deterministic for a given game seed and decision key (NumPy PCG64), and cheap enough
to run per decision in the browser's Pyodide backend.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from drmc_rl.execution.pace import BY_ID, Pace
from drmc_rl.planning.fast_reach import FrameState, simulate_frame

MODEL_PATH = Path(__file__).with_name("movement_model_v1.json")
PROFILE_SCHEMA = "drmc-human-movement-profile-v1"
MOVEMENT_MODES = ("exact", "human")
MAX_FRAMES = 1024
ATTEMPTS = 6
DOWN = 3
STYLE_SALT = 0x5717E
DECISION_SALT = 0xDEC1


@lru_cache(maxsize=4)
def load_model(path: str | None = None) -> dict:
    data = Path(path or MODEL_PATH).read_bytes()
    model = json.loads(data)
    if model.get("schema") != "drmc-human-movement-model-v1":
        raise ValueError("unsupported human movement model schema")
    model["sha256"] = hashlib.sha256(data).hexdigest()
    return model


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gravity_cell(threshold: int) -> str:
    return "slow" if threshold >= 13 else "mid" if threshold >= 7 else "fast"


def rows_cell(remaining: int) -> str:
    return ("0-1" if remaining <= 1 else "2-3" if remaining <= 3 else "4-6" if remaining <= 6
            else "7-9" if remaining <= 9 else "10+")


def depth_cell(rows_fallen: int) -> str:
    return "1-4" if rows_fallen <= 4 else "5-8" if rows_fallen <= 8 else "9-12" if rows_fallen <= 12 else "13-16"


@dataclass(frozen=True)
class MovementDecision:
    """Per-decision draws, fixed before the planner chooses a target."""

    pace_id: str
    reaction_frames: int
    latent: tuple[float, ...]
    seed: tuple[int, ...]

    def rng(self, attempt: int = 0) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([*self.seed, 0xA77E, attempt]))


@dataclass(frozen=True)
class MovementAblation:
    """Diagnostic and training switches that keep parts of exact planner execution.

    The default is the complete human generator. ``reaction="pace"`` starts at the named pace's
    reaction with no depth hesitation; ``steering="planner"`` keeps the planner witness's steering
    prefix; ``descent="prompt"`` holds Down as soon as steering ends; ``corrections`` and
    ``pauses`` switch those sampled behaviours off.
    """

    reaction: str = "human"
    steering: str = "human"
    corrections: bool = True
    pauses: bool = True
    descent: str = "human"

    def __post_init__(self):
        if (self.reaction not in ("human", "pace") or self.steering not in ("human", "planner")
                or self.descent not in ("human", "prompt")):
            raise ValueError("unknown human movement ablation")

    @classmethod
    def from_dict(cls, value: dict | None) -> "MovementAblation":
        return cls(**(value or {}))

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v != getattr(FULL_MOVEMENT, k)}


FULL_MOVEMENT = MovementAblation()


class HumanMovement:
    """One pace's fitted human movement profile."""

    def __init__(self, pace_id: str, model: dict | None = None):
        model = model or load_model()
        if pace_id not in model["profiles"]:
            raise ValueError(f"pace {pace_id!r} has no human movement profile")
        self.model = model
        self.pace_id = pace_id
        self.pace = BY_ID[pace_id]
        self.profile = model["profiles"][pace_id]
        self.knots = np.asarray(model["knots"], dtype=np.float64)
        floor = self.profile["floor"]
        # Hard limits for generated scripts. Controller changes on consecutive frames are human
        # (different buttons); the reaction floor is the profile's lowest fitted quantile.
        self.floor = Pace(pace_id, self.pace.label, int(floor["reaction_frames"]),
                          int(floor["edge_interval"]), int(floor["motion_interval"]),
                          max(int(floor["max_buttons"]), self.pace.max_buttons))
        # Placement choice stays within the named pace's motor-feasible set, after the sampled
        # reaction has elapsed; its witness is the generator's last-resort route.
        self.planning = replace(self.pace, reaction_frames=0)
        style = model["style"]
        self.dims = tuple(style["dims"])
        self._between = np.linalg.cholesky(np.asarray(style["between_cov"]) + 1e-9 * np.eye(len(self.dims)))
        self._within = np.linalg.cholesky(np.asarray(style["within_cov"]) + 1e-9 * np.eye(len(self.dims)))

    # ---------------------------------------------------------------- sampling
    def _q(self, table, u: float) -> float:
        return float(np.interp(u, self.knots, table))

    def _index(self, name: str) -> int:
        return self.dims.index(name)

    def style(self, game_seed: int) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([STYLE_SALT, int(game_seed) & 0xFFFFFFFFFFFF]))
        return self._between @ rng.standard_normal(len(self.dims))

    def decide(self, game_seed: int, key: int, *, threshold: int) -> MovementDecision:
        """Sample this decision's latent draws and its planning reaction (frames from spawn).

        Human reaction grows with how far the chosen pill will fall, which is unknown until the
        planner picks a target. The planning reaction is this draw's quantile at the quickest
        depth; ``generate`` adds the rest as hesitation once the target is known, so the total
        first-input time follows the depth-conditioned table exactly.
        """
        seed = (DECISION_SALT, int(game_seed) & 0xFFFFFFFFFFFF, int(key) & 0xFFFFFFFFFFFF)
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        latent = self.style(game_seed) + self._within @ rng.standard_normal(len(self.dims))
        reaction = min(self._reaction(latent, threshold, depth) for depth in self.model["cells"]["depth"])
        return MovementDecision(self.pace_id, reaction, tuple(float(v) for v in latent), seed)

    def _reaction(self, latent, threshold: int, depth: str) -> int:
        table = self.profile["reaction"][f"{gravity_cell(int(threshold))}|{depth}"]
        value = self._q(table, _phi(latent[self._index("reaction")]))
        return int(max(self.floor.reaction_frames, math.floor(value + 0.5)))

    def to_dict(self) -> dict:
        return {"schema": PROFILE_SCHEMA, **asdict(self.pace), "human_calibrated": False,
                "movement": "human", "movement_model": self.model.get("id"),
                "movement_model_sha256": self.model.get("sha256"),
                "generated_script_limits": {k: v for k, v in asdict(self.floor).items() if k not in ("id", "label")},
                "calibration": "corpus-fitted per-placement distributions; not a certified ExecutionProfile"}

    # --------------------------------------------------------------- planning
    def _plan(self, decision: MovementDecision, rng: np.random.Generator, start: FrameState,
              target: tuple[int, int, int], threshold: int, level: int, execution_delay: int,
              ablation: MovementAblation = FULL_MOVEMENT) -> dict:
        p, lat = self.profile, decision.latent
        g = gravity_cell(threshold)
        tx, ty, trot = target
        depth = depth_cell(ty + 1)
        dx = tx - start.x
        adx = min(abs(dx), 4)

        def u_of(name):
            return _phi(lat[self._index(name)])

        def flag(name, prob):
            return u_of(name) > 1.0 - prob
        right_wall = 7 if trot & 1 else 6
        wall = tx in (0, right_wall)
        das = adx >= 1 and flag("das", p["das_p"].get(f"{adx}{'w' if wall else ''}", p["das_p"].get(str(adx), 0.0)))
        if adx == 1 and not wall:
            das = False  # a held single step only reads as auto-repeat when pushed into a wall
        corrected = level < 1 and ablation.corrections and flag("correction", p["correction_p"][g])
        kind = None
        if corrected:
            mix = p["correction_mix"]
            names = list(mix)
            kind = names[int(rng.choice(len(names), p=np.asarray([mix[n] for n in names]) / sum(mix.values())))]
        turns = (trot - start.rot) & 3
        first = int(rng.integers(1, 3))  # A (1) or B (2) for a half turn
        rotations = [] if turns == 0 else [2] if turns == 1 else [1] if turns == 3 else [first, first]
        waypoints = [tx]
        side = 1 if dx > 0 else -1 if dx < 0 else int(rng.choice((-1, 1)))
        if kind == "overshoot":
            waypoints = [tx + side, tx]
        elif kind == "reversal":
            waypoints = [start.x - side, tx]
        elif kind == "extra_rotation":
            pair = [1, 2] if rng.random() < 0.5 else [2, 1]
            at = int(rng.integers(0, len(rotations) + 1))
            rotations = rotations[:at] + pair + rotations[at:]
        late = kind == "late_lateral" and dx != 0
        waypoints = [w for w in waypoints[:-1] if 0 <= w <= right_wall and w != start.x] + [tx]
        gap_table = p["gap"][f"{g}|{depth}"]
        gap_style = float(self.style_shift(decision, "gap"))
        gap_sd = math.sqrt(max(1e-6, 1.0 - float(self.model["style"]["between_sd"][self._index("gap")]) ** 2))
        speed = 0.5 if level >= 3 else 1.0

        def gap():
            u = _phi(gap_style + gap_sd * float(rng.standard_normal()))
            return max(1, int(math.floor(self._q(gap_table, u) * speed + 0.5)))
        steps, x = 0, start.x
        for w in waypoints:
            steps, x = steps + (1 if das else abs(w - x)), w
        presses = len(rotations) + steps
        pause_at = None
        if level < 2 and ablation.pauses and presses >= 2 and flag("pause", p["pause_p"][f"{g}|{depth}"]):
            pause_at = int(rng.integers(1, presses))  # the gap after this press becomes a pause
        # One latent orders the whole descent: the slowest draws never soft drop.
        slack_u = u_of("slack")
        no_down_p = p["no_down_p"]
        return {
            "das": das, "wall": wall, "correction": kind, "rotations": rotations, "waypoints": waypoints,
            "late": late, "late_down": int(rng.integers(2, 9)), "rot_first": rng.random() < p["rot_first_p"],
            "gap": gap, "pause_at": pause_at,
            # Frames of hesitation after the start delay: the depth-conditioned reaction not yet spent.
            "hesitation": 0 if level >= 2 or ablation.reaction == "pace"
            else max(0, self._reaction(lat, threshold, depth) - execution_delay),
            "pause": max(self.model["cells"]["pause_gap"], int(self._q(p["pause"], float(rng.random())) + 0.5)),
            "hold_lat": lambda: max(1, int(self._q(p["hold"]["lateral"], float(rng.random())) * speed + 0.5)),
            "hold_rot": lambda: max(1, int(self._q(p["hold"]["rotation"], float(rng.random())) * speed + 0.5)),
            "slack": (lambda rows: 0 if level >= 3 or ablation.descent == "prompt" else None
                      if slack_u > 1.0 - no_down_p[f"{g}|{rows_cell(rows)}"] else
                      max(0, int(self._q(p["slack"][f"{g}|{rows_cell(rows)}"],
                                         slack_u / max(1e-6, 1.0 - no_down_p[f"{g}|{rows_cell(rows)}"])) + 0.5))),
        }

    def style_shift(self, decision: MovementDecision, name: str) -> float:
        # The per-game component of one latent dimension, recovered from the decision's seed.
        return float(self.style(decision.seed[1])[self._index(name)])

    # -------------------------------------------------------------- realizing
    def _realize(self, cols, start: FrameState, target, threshold: int, plan: dict):
        """Closed-loop steering on the sampled schedule, then a timed descent; None if not exact."""
        tx, ty, trot = target
        state = start
        actions: list[int] = []
        held_dir, held_rot = start.hold_dir.value, start.rot_hold.value
        dir_release = rot_release = None
        das_since = None
        waypoints = list(plan["waypoints"])
        rotations = list(plan["rotations"])
        next_press, presses, first_lateral = plan["hesitation"], 0, False
        lateral_at, lateral_x = None, None
        late_pending, down_left, retries, t = plan["late"], 0, 0, 0

        def step(action):
            nonlocal state, t
            actions.append(action)
            state = simulate_frame(cols, state, action, speed_threshold=threshold)
            t += 1

        while t < MAX_FRAMES:
            if state.locked:
                return None
            while waypoints and state.x == waypoints[0]:
                waypoints.pop(0)
                if das_since is not None and not (plan["wall"] and not waypoints):
                    das_since, held_dir = None, 0  # release on arrival
            if das_since is not None and not waypoints and t - das_since >= 16:
                das_since, held_dir = None, 0  # a wall push held past the auto-repeat delay
            if (len(waypoints) > 1 and lateral_at is not None and t - lateral_at >= 8
                    and state.x == lateral_x and das_since is None):
                waypoints.pop(0)  # a correction step was blocked; give it up
                lateral_at = None
            if not waypoints and state.x != tx:
                waypoints.append(tx)  # a rotation kick moved the pill; steer back
                retries += 1
            if not rotations and state.rot != trot:
                turns = (trot - state.rot) & 3  # a rotation was refused; press again
                rotations = [2] if turns == 1 else [1] if turns == 3 else [1, 1]
                retries += 1
            if retries > 6 or (das_since is not None and t - das_since > 16 + 6 * 8):
                return None
            if dir_release is not None and t >= dir_release:
                held_dir, dir_release = 0, None
            if rot_release is not None and t >= rot_release:
                held_rot, rot_release = 0, None
            if down_left:
                if state.y < ty - 1:
                    down_left -= 1
                    step(DOWN)
                    continue
                down_left, next_press = 0, t
            if not waypoints and not rotations and das_since is None:
                break
            if (late_pending and len(waypoints) == 1 and not rotations and das_since is None
                    and abs(waypoints[0] - state.x) == 1 and t >= next_press):
                # Late sideways move: start the drop, then make the last step.
                late_pending, down_left = False, plan["late_down"]
                held_dir, held_rot, dir_release, rot_release = 0, 0, None, None
                continue
            if t >= next_press:
                can_lat = bool(waypoints) and das_since is None
                can_rot = bool(rotations)
                if can_rot and (plan["rot_first"] or first_lateral or not can_lat):
                    button = rotations[0]
                    if held_rot == button:
                        held_rot, rot_release, next_press = 0, None, t + 1  # release, press next frame
                    else:
                        rotations.pop(0)
                        held_rot, rot_release = button, t + plan["hold_rot"]()
                        presses += 1
                        next_press = t + self._next_gap(plan, presses)
                elif can_lat:
                    direction = 1 if waypoints[0] < state.x else 2
                    if held_dir == direction:
                        held_dir, dir_release, next_press = 0, None, t + 1
                    else:
                        held_dir, first_lateral = direction, True
                        lateral_at, lateral_x = t, state.x
                        presses += 1
                        if plan["das"] and not late_pending:
                            das_since, dir_release = t, None
                        else:
                            dir_release = t + min(15, plan["hold_lat"]())
                        next_press = t + self._next_gap(plan, presses)
            step(held_dir * 6 + held_rot)
        if state.locked or t >= MAX_FRAMES:
            return None
        return self._descend(cols, actions, state, target, threshold, plan)

    def _descend(self, cols, actions, base, target, threshold: int, plan: dict):
        """Neutral wait, then hold Down to the lock, closest to the sampled slack."""
        tx, ty, trot = target
        t = len(actions)
        probe, drop = base, 0
        while not probe.locked and drop < MAX_FRAMES:
            probe = simulate_frame(cols, probe, DOWN, speed_threshold=threshold)
            drop += 1
        if not probe.locked or (probe.x, probe.y, probe.rot) != target:
            return None
        slack = plan["slack"](ty - base.y)
        if slack is None:  # no soft drop: fall by gravity to the lock
            waiting, wait = base, 0
            while not waiting.locked and t + wait < MAX_FRAMES:
                waiting = simulate_frame(cols, waiting, 0, speed_threshold=threshold)
                wait += 1
            if not waiting.locked or (waiting.x, waiting.y, waiting.rot) != target:
                return None
            return np.concatenate((np.asarray(actions, dtype=np.uint8), np.zeros(wait, dtype=np.uint8)))
        wanted = drop + slack
        best, waiting = (0, drop), base
        for wait in range(1, MAX_FRAMES - t):
            waiting = simulate_frame(cols, waiting, 0, speed_threshold=threshold)
            if waiting.locked:
                if (waiting.x, waiting.y, waiting.rot) == target and abs(wait - wanted) < abs(sum(best) - wanted):
                    best = (wait, 0)
                break
            probe, more = waiting, 0
            while not probe.locked and more < MAX_FRAMES:
                probe = simulate_frame(cols, probe, DOWN, speed_threshold=threshold)
                more += 1
            if abs(wait + more - wanted) < abs(sum(best) - wanted):
                best = (wait, more)
            if wait + more > wanted + 2:
                break
        wait, more = best
        return np.concatenate((np.asarray(actions, dtype=np.uint8),
                               np.zeros(wait, dtype=np.uint8), np.full(more, DOWN, dtype=np.uint8)))

    def _planner_steering(self, cols, start: FrameState, target, threshold: int, plan: dict, witness):
        """The witness's exact steering prefix, then this plan's descent; None if not exact."""
        state, final, aligned = start, 0, start
        for index, action in enumerate(witness, 1):
            previous = state
            state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
            if (state.x, state.rot) != (previous.x, previous.rot):
                final, aligned = index, state
            if state.locked:
                break
        if aligned.locked:
            return np.asarray(witness[:final], dtype=np.uint8)
        return self._descend(cols, [int(a) for a in witness[:final]], aligned, target, threshold, plan)

    @staticmethod
    def _next_gap(plan: dict, presses: int) -> int:
        if plan["pause_at"] is not None and presses == plan["pause_at"]:
            return plan["pause"]
        return plan["gap"]()

    # ------------------------------------------------------------ generation
    def generate(self, decision: MovementDecision, cols: np.ndarray, start: FrameState,
                 target: tuple[int, int, int], *, speed_threshold: int, witness,
                 execution_delay: int, ablation: MovementAblation = FULL_MOVEMENT) -> tuple[np.ndarray, dict]:
        """Human-like script from ``start`` (after the full reaction delay) to an exact lock at ``target``.

        ``witness`` is the planner's route for the same target from the same start. The returned
        script always replays to the target and satisfies this profile's generated-script limits.
        """
        columns = np.asarray(cols, dtype=np.uint16).reshape(8)
        target = (int(target[0]), int(target[1]), int(target[2]) & 3)
        threshold = int(speed_threshold)
        info = {"algorithm": "human-movement-v1", "model": self.model.get("id"),
                "reaction_frames": decision.reaction_frames, "start_delay_frames": int(execution_delay)}
        if ablation != FULL_MOVEMENT:
            info["ablation"] = ablation.to_dict()
        base = np.asarray(witness, dtype=np.uint8).reshape(-1)
        for attempt in range(ATTEMPTS):
            level = min(attempt, 3)  # 0 full, 1 no correction, 2 no pause, 3 quick taps + prompt drop
            rng = decision.rng(attempt)
            plan = self._plan(decision, rng, start, target, threshold, level, int(execution_delay), ablation)
            if ablation.steering == "planner":
                script = self._planner_steering(columns, start, target, threshold, plan, base)
            else:
                script = self._realize(columns, start, target, threshold, plan)
            if script is None:
                continue
            audit = self._audit(columns, start, script, threshold, target, execution_delay)
            if audit is not None:
                return script, {**info, "route": "human" if ablation.steering == "human" else "planner_steering",
                                "attempts": attempt + 1,
                                "correction": plan["correction"], "das": bool(plan["das"]),
                                "hesitation_frames": plan["hesitation"],
                                "paused": plan["pause_at"] is not None, **audit}
        from drmc_rl.human.cadence import _retime_planner_route
        rng = decision.rng(ATTEMPTS)
        slack_table = self.profile["slack"]
        requested = len(base) + int(self._q(slack_table[f"{gravity_cell(threshold)}|0-1"], float(rng.random())))
        try:
            retimed = _retime_planner_route(columns, start, base, speed_threshold=threshold,
                                            target=target, requested_frames=requested)
        except ValueError:
            retimed = None
        for route, script in (("planner_retimed", retimed), ("planner", base)):
            if script is None:
                continue
            audit = self._audit(columns, start, script, threshold, target, execution_delay)
            if audit is not None:
                return np.asarray(script, dtype=np.uint8), {**info, "route": route, "attempts": ATTEMPTS,
                                                            "correction": None, **audit}
        raise RuntimeError("planner witness does not reach its target")

    def _audit(self, cols, start, script, threshold, target, execution_delay):
        state = start
        for index, action in enumerate(script, 1):
            state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
            if state.locked:
                if index != len(script) or (state.x, state.y, state.rot) != target:
                    return None
                break
        if not state.locked:
            return None
        try:
            limits = self.floor.validate(cols, start, script, speed_threshold=threshold,
                                         execution_delay=execution_delay)
        except ValueError:
            return None
        return {"validated": True, "unrestricted_fallback": False, "execution_frames": int(len(script)), **limits}


def steering_split(move: dict) -> tuple[int, int]:
    """Frames through the last sideways or rotation change of a move, and its whole script."""
    states = move["controller_states"]
    lock = move["lock_state"]
    poses = [(s["x"], s["rotation"]) for s in states] + [(lock["x"], lock["rotation"])]
    last = max((i for i in range(1, len(poses)) if poses[i] != poses[i - 1]), default=0)
    return last, len(states)


def movement_seed(seed: int, physical: int, variant: str) -> int:
    """One style draw per game, side and entrant; independent of the opponent's variant."""
    digest = hashlib.blake2b(f"{seed}:{physical}:{variant}".encode(), digest_size=6).digest()
    return int.from_bytes(digest, "little")


@lru_cache(maxsize=16)
def movement_for_pace(pace_id: str) -> HumanMovement | None:
    """The pace's human profile, or None for Frame Perfect (exact machine movement)."""
    model = load_model()
    return HumanMovement(pace_id, model) if pace_id in model["profiles"] else None


def human_execution_for_action(candidate, action: int, movement: HumanMovement,
                               decision: MovementDecision, *, delay: int, frame_id: int = 0,
                               ablation: MovementAblation = FULL_MOVEMENT) -> dict:
    """Human-like counterpart of ``anticipation.execution_for_action`` for a planned candidate.

    ``candidate`` must be planned with ``movement.planning`` from the state ``delay`` frames
    after spawn. The returned controller trace is verified frame by frame.
    """
    from drmc_rl.human.backend import ACTION_TO_BUTTONS, ACTION_TO_POSE, _columns, _frame_payload
    from drmc_rl.planning.fast_reach import compute_speed_threshold

    own, _, _, _, speed, ups, start, reach, _, _ = candidate
    pose = int(ACTION_TO_POSE[int(action)])
    x, y, rot = pose % 8, pose // 8 % 16, pose // 128
    witness = reach.script_for_pose(x, y, rot)
    if witness is None:
        raise ValueError("chosen placement has no controller witness")
    columns = _columns(own)
    threshold = compute_speed_threshold(speed, ups)
    script, info = movement.generate(decision, columns, start, (x, y, rot), speed_threshold=threshold,
                                     witness=np.asarray(witness).copy(), execution_delay=delay,
                                     ablation=ablation)
    frame, trace = start, []
    for buttons in script:
        trace.append(_frame_payload(frame))
        frame = simulate_frame(columns, frame, int(buttons), speed_threshold=threshold)
    if not frame.locked or (frame.x, frame.y, frame.rot) != (x, y, rot):
        raise ValueError("human controller script did not reach its placement")
    return {
        "placement": {"action": int(action), "x": x, "y_top": y, "rotation": rot},
        "controller_frames": [ACTION_TO_BUTTONS[int(a)] for a in script],
        "controller_states": trace,
        "execution": {"falling": _frame_payload(start), "start_frame": frame_id + delay, "delay_frames": delay},
        "timing": {"execution_profile": movement.to_dict(), "execution_frames": len(script),
                   "planner_cost_frames": int(len(witness)), "movement": info},
        "lock_state": _frame_payload(frame),
    }


__all__ = ["FULL_MOVEMENT", "HumanMovement", "MOVEMENT_MODES", "MovementAblation", "MovementDecision", "movement_seed", "steering_split", "human_execution_for_action", "load_model",
           "movement_for_pace"]
