"""Showiness of rollout games and arm (b)'s bounded big-clear bonus.

Each rollout move row carries the acting side's spawn bottle, its canonical
pill and the chosen macro action, so every placement's clear is resolved
exactly (``drmc_rl.eval.big_clear``) without touching the engine.

The bonus is event based: a learner placement whose clear scores ``s`` earns
``min(event_cap, base + per_point * (s - threshold))`` when ``s >= threshold``
(ordinary clears, including plain 2-garbage combos, earn nothing). A game's
bonuses are capped at ``game_cap`` in placement order. Decision ``t`` receives
the undiscounted return-to-go ``outcome + sum(bonus_k, k >= t)``, matching the
natural-terminal, undiscounted outcome objective: setup placements before a
big clear share its credit, placements after it do not.

With the outcome in {-1, 0, +1} and ``game_cap`` far below the 2-point
win/loss swing, no bonus total can outweigh a win; the largest possible trade
is ``game_cap / 2`` of win probability per game (``bound`` in the spec log).
"""
from __future__ import annotations

from collections import Counter

from drmc_rl.eval import big_clear as bc

SPEC_KEYS = ("threshold", "base", "per_point", "event_cap", "game_cap")


def validate_spec(spec: dict) -> dict:
    if set(spec) - set(SPEC_KEYS) - {"note"}:
        raise ValueError(f"unknown showiness_bonus keys {sorted(set(spec) - set(SPEC_KEYS))}")
    spec = dict(spec)
    for key in ("threshold", "per_point", "event_cap", "game_cap"):
        if key not in spec:
            raise ValueError(f"showiness_bonus needs {key}")
    spec.setdefault("base", 0.0)
    if not (0 <= spec["base"] <= spec["event_cap"] <= spec["game_cap"] < 1.0):
        raise ValueError("showiness_bonus caps must satisfy 0 <= base <= event_cap <= game_cap < 1")
    if spec["per_point"] < 0:
        raise ValueError("showiness_bonus per_point must be non-negative")
    return spec


def move_features(move: dict) -> bc.ClearFeatures:
    return bc.placement_features(bytes(move["board"]), move["pill"], int(move["placement"]["action"]))


def side_summary(moves: list[dict], side: int) -> dict:
    """Clear counts by tier, largest score and feature maxima of one physical side."""
    tiers = Counter()
    best = 0.0
    placements = clears = 0
    maxima = Counter()
    for move in moves:
        if int(move["side"]) != side:
            continue
        placements += 1
        try:
            f = move_features(move)
        except ValueError:
            continue
        if not f.rounds:
            continue
        clears += 1
        score = f.score()
        best = max(best, score)
        tiers[bc.tier(score)] += 1
        for name in ("cells", "rounds", "max_round_lines", "max_line", "viruses"):
            maxima[name] = max(maxima[name], getattr(f, name))
    return dict(placements=placements, clears=clears, best=best,
                **{t: tiers.get(t, 0) for t in ("T1", "T2", "T3")}, max=dict(maxima))


def learner_bonuses(moves: list[dict], spec: dict) -> list[float]:
    """Bonus per learner decision (moves carrying ``learning``), in move order, capped per game."""
    out, total = [], 0.0
    for move in moves:
        if "learning" not in move:
            continue
        try:
            value = bc.showiness_bonus(move_features(move), spec)
        except ValueError:
            value = 0.0
        value = max(0.0, min(value, spec["game_cap"] - total))
        total += value
        out.append(value)
    return out


def apply_bonus(batch, samples: list[dict], spec: dict) -> dict:
    """Add return-to-go bonuses to ``terminal_samples(batch)`` output in place; return fire statistics."""
    index = 0
    stats = Counter()
    for game_id, (row, moves, _) in enumerate(batch):
        if row["reason"] == "timeout":
            continue
        bonuses = learner_bonuses(moves, spec)
        togo = 0.0
        tails = []
        for value in reversed(bonuses):
            togo += value
            tails.append(togo)
        tails.reverse()
        for value in tails:
            sample = samples[index]
            if sample["game_id"] != game_id:
                raise RuntimeError("bonus alignment differs from terminal_samples")
            sample["return"] = float(sample["return"] + value)
            sample["showiness_bonus_to_go"] = value
            index += 1
        stats["games"] += 1
        stats["decisions"] += len(bonuses)
        stats["events"] += sum(v > 0 for v in bonuses)
        stats["games_with_bonus"] += any(v > 0 for v in bonuses)
        stats["bonus_total"] += sum(bonuses)
    if index != len(samples):
        raise RuntimeError("bonus alignment differs from terminal_samples")
    return dict(stats)


__all__ = ["SPEC_KEYS", "apply_bonus", "learner_bonuses", "move_features", "side_summary", "validate_spec"]
