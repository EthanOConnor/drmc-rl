"""Rating conditions: the rules under which two entrants' games are comparable.

A condition fixes everything that changes what a game *is* for both players:
rollout backend, native engine revision, level, speed, motor pace (the exact
execution profile, by key), the decision contract (fresh-decision delay and
timing point) and movement model. Both entrants of a pool game always play
under the same condition. Games under different condition keys are never
mixed into one rating; a pooled view is only an explicit average over a named
condition set (``drmc_rl.pool.ratings.pooled``).

Settings that change wall-clock or numerics but not the game (device, threads,
planner workers, memoization, asynchronous planning) are runtime settings and
deliberately not part of the key; replicate audits cover their fidelity.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = "drmc-pool-condition-v1"
BACKENDS = ("events", "frames")
MOVEMENTS = ("exact", "human")
DECISION_POINTS = ("spawn", "settled", "lock", "lock_safe", "commit_safe")
EARLY_PREVIEWS = ("marginal", "repeat", "branches")
# The engine resets at the NES "HI" speed setting; no other speed is executable yet.
SPEEDS = (2,)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# Decision-contract keys a condition may set, with the evaluator's defaults.
DECISION_DEFAULTS = dict(decision_point="spawn", early_preview="marginal", preview_input="visible",
                         compute_input_frames=None, early_delay_input="actual")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def execution_key(profile) -> str:
    """Same digest as ``drmc_rl.arena.experiment.execution_key``."""
    return hashlib.sha256(canonical(profile).encode()).hexdigest()[:16]


def pace_profile(pace: str) -> dict:
    from drmc_rl.execution.pace import resolve_pace
    return resolve_pace(pace).to_dict()


def normalize_decision(decision: dict) -> dict:
    decision = dict(decision)
    unknown = set(decision) - {"delay", *DECISION_DEFAULTS}
    if unknown:
        raise ValueError(f"unknown decision-contract keys {sorted(unknown)}")
    if type(decision.get("delay")) is not int or not 0 <= decision["delay"] <= 120:
        raise ValueError("decision.delay must be an integer frame count")
    point = decision.get("decision_point", "spawn")
    if point not in DECISION_POINTS:
        raise ValueError(f"decision_point must be one of {DECISION_POINTS}")
    if decision.get("early_preview", "marginal") not in EARLY_PREVIEWS:
        raise ValueError(f"early_preview must be one of {EARLY_PREVIEWS}")
    if decision.get("preview_input", "visible") not in ("visible", "marginal"):
        raise ValueError("preview_input must be visible or marginal")
    if decision.get("early_delay_input", "actual") not in ("actual", "contract"):
        raise ValueError("early_delay_input must be actual or contract")
    pinned = decision.get("compute_input_frames")
    if pinned is not None and (type(pinned) is not int or not 0 <= pinned <= 60):
        raise ValueError("compute_input_frames must be an integer frame count")
    if point == "spawn":
        # Early-request knobs are inert for spawn decisions.
        decision.pop("early_preview", None)
        decision.pop("early_delay_input", None)
    return {k: v for k, v in decision.items() if k == "delay" or v != DECISION_DEFAULTS[k]}


def make_condition(*, backend: str, engine: str, level: int, pace: str, decision: dict,
                   movement: str = "exact", speed: int = 2, profile: dict | None = None) -> dict:
    """A validated canonical condition spec (without its name)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if movement not in MOVEMENTS:
        raise ValueError(f"movement must be one of {MOVEMENTS}")
    if speed not in SPEEDS:
        raise ValueError(f"speed {speed} is not executable by the engine (supported: {SPEEDS})")
    if type(level) is not int or not 0 <= level <= 24:
        raise ValueError("level must be an integer 0..24")
    if not isinstance(engine, str) or not engine:
        raise ValueError("engine (native engine commit) is required")
    decision = normalize_decision(decision)
    if backend == "events" and (decision.get("decision_point", "spawn") != "spawn"
                                or decision.get("preview_input", "visible") != "visible"):
        raise ValueError("pre-spawn decision points and preview marginals require the frames backend")
    profile = profile if profile is not None else pace_profile(pace)
    if profile.get("id") != pace:
        raise ValueError("execution profile does not belong to this pace")
    return dict(schema=SCHEMA, backend=backend, engine=engine, level=level, speed=speed, pace=pace,
                execution_key=execution_key(profile), decision=decision, movement=movement)


def condition_key(spec: dict) -> str:
    body = {k: spec[k] for k in ("schema", "backend", "engine", "level", "speed", "pace", "execution_key",
                                 "decision", "movement")}
    return "c" + hashlib.sha256(canonical(body).encode()).hexdigest()[:12]


def default_name(spec: dict) -> str:
    d = spec["decision"]
    point = d.get("decision_point", "spawn")
    contract = f"{point}{d['delay']}" + (f"-{d['early_preview']}" if "early_preview" in d else "")
    extra = "".join(f"-{k}{v}" for k, v in sorted(d.items())
                    if k not in ("delay", "decision_point", "early_preview"))
    movement = "" if spec["movement"] == "exact" else f"-{spec['movement']}"
    return f"{spec['backend'][:2]}-L{spec['level']}-{spec['pace']}-{contract}{extra}{movement}"


def check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"invalid name {name!r}")
    return name


def requirements(spec: dict) -> set[str]:
    """Runtime capabilities a worker build needs to play this condition."""
    needs = {f"backend:{spec['backend']}", f"decision:{spec['decision'].get('decision_point', 'spawn')}",
             f"engine:{spec['engine']}"}
    if spec["movement"] != "exact":
        needs.add(f"movement:{spec['movement']}")
    return needs


def runtime_capabilities(repo) -> set[str]:
    """What this source tree can execute, determined without importing torch."""
    from pathlib import Path
    repo = Path(repo)
    caps = {f"backend:{b}" for b in BACKENDS} | {f"decision:{p}" for p in DECISION_POINTS}
    caps |= {"loader:plain", "loader:pace_adapter"}
    movement = repo / "drmc_rl" / "human" / "movement.py"
    arena = repo / "tools" / "trainer_planning_arena.py"
    if movement.is_file() and arena.is_file() and "movement_for_pace" in arena.read_text():
        caps.add("movement:human")
    return caps


def study_condition(config: dict, match: dict, params_a: dict, params_b: dict) -> tuple[dict | None, str]:
    """The pool condition of one trainer-planning-arena schedule row, or (None, reason).

    Rows whose entrants play different decision contracts or movement models
    are not symmetric games under one condition and are not imported.
    """
    decision_keys = ("delay", *DECISION_DEFAULTS)
    try:
        decisions = [normalize_decision({k: p[k] for k in decision_keys if k in p}) for p in (params_a, params_b)]
    except ValueError as error:
        return None, f"decision contract: {error}"
    if decisions[0] != decisions[1]:
        return None, "the two entrants play different decision contracts"
    movements = [p.get("movement", "exact") for p in (params_a, params_b)]
    if movements[0] != movements[1]:
        return None, "the two entrants use different movement models"
    for params in (params_a, params_b):
        for key in ("anticipation", "own_board_only", "movement_ablation", "context_pace", "planning_pace"):
            if params.get(key):
                return None, f"variant setting {key} is a diagnostic ablation, not a rated player"
    if not config.get("native_commit"):
        return None, "the study did not record its native engine commit"
    profile = match.get("execution_profile") or pace_profile(match.get("pace", "frame_perfect"))
    try:
        spec = make_condition(backend=config.get("rollout_backend", "frames"), engine=str(config["native_commit"])[:7],
                              level=int(match["level"]), pace=match.get("pace", "frame_perfect"),
                              decision=decisions[0], movement=movements[0], profile=profile)
    except ValueError as error:
        return None, str(error)
    if match.get("execution_key") and match["execution_key"] != spec["execution_key"] and movements[0] == "exact":
        return None, "the study's recorded execution key differs from this evaluator's profile"
    return spec, ""
