"""Deterministic collection and quota balancing for pair-state banks."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from drmc_rl.teachers.counterfactual_release import canonical_json, source_identity

DEFAULT_LEVELS = (5, 10, 15, 20)
DEFAULT_SPEEDS = (0, 1, 2)
DEFAULT_TACTICAL_STRATA = (
    "midgame",
    "high-pressure",
    "topout-defense",
    "incoming-garbage",
    "race-finish",
)


def _field(payload: Mapping[str, Any], path: str) -> object:
    value: object = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"state is missing quota field {path!r}")
        value = value[component]
    return value


def _score(seed: int, identity: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{int(seed)}\0{identity}".encode()).digest(), "big"
    )


def _validate_belief(row: Mapping[str, Any], identity: str) -> None:
    from drmc_rl.search.pill_belief import PillReserveBelief

    belief = row.get("reserve_belief")
    if not isinstance(belief, Mapping):
        raise ValueError(f"state {identity} lacks public reserve-belief history")
    try:
        parsed = PillReserveBelief.from_dict(belief)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"state {identity} has invalid reserve-belief evidence: {error}"
        ) from error
    if parsed.initial_board is None:
        raise ValueError(
            f"state {identity} reserve belief lacks initial-board conditioning"
        )


@dataclass(frozen=True, slots=True)
class BankBalanceResult:
    selected: tuple[dict[str, Any], ...]
    strata: Mapping[str, int]
    quota: Mapping[str, int]
    shortfall: Mapping[str, int]
    duplicates: int

    @property
    def quota_shortfall(self) -> int:
        return int(sum(self.shortfall.values()))


def cross_product_quota(
    *,
    levels: Sequence[int] = DEFAULT_LEVELS,
    speeds: Sequence[int] = DEFAULT_SPEEDS,
    tactical_strata: Sequence[str] = DEFAULT_TACTICAL_STRATA,
    per_cell: int = 24,
) -> dict[tuple[str, str, str], int]:
    if per_cell < 1 or not levels or not speeds or not tactical_strata:
        raise ValueError("bank quota axes and per-cell target must be nonempty")
    return {
        (str(level), str(speed), str(tactical)): int(per_cell)
        for level, speed, tactical in itertools.product(levels, speeds, tactical_strata)
    }


def select_game_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    global_tactical_counts: MutableMapping[str, int],
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Select throughout a completed game without exhausting the cap early.

    The old collector stopped a game as soon as its row cap was filled, which
    almost entirely excluded race finishes and recovery states. This selector
    receives all candidate decision states from one completed/limited game and
    chooses rows round-robin from the globally least represented tactical
    strata. Within each stratum the choice is a stable content-hash sample.
    ``global_tactical_counts`` is updated for the selected rows.
    """

    if limit < 1:
        raise ValueError("per-game selection limit must be positive")
    grouped: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        identity = source_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        tactical = str(_field(row, "tactical_stratum"))
        grouped.setdefault(tactical, []).append((_score(seed, identity), identity, row))
    for tactical in grouped:
        grouped[tactical].sort(key=lambda item: (item[0], item[1]))

    selected: list[dict[str, Any]] = []
    local_counts: dict[str, int] = {}
    while len(selected) < limit:
        available = [key for key, values in grouped.items() if values]
        if not available:
            break
        tactical = min(
            available,
            key=lambda key: (
                int(global_tactical_counts.get(key, 0)),
                int(local_counts.get(key, 0)),
                key,
            ),
        )
        _score_value, _identity, row = grouped[tactical].pop(0)
        selected.append(row)
        local_counts[tactical] = local_counts.get(tactical, 0) + 1
        global_tactical_counts[tactical] = global_tactical_counts.get(tactical, 0) + 1
    return tuple(selected)


def balance_state_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    quota: Mapping[tuple[str, ...], int],
    fields: Sequence[str] = ("level", "speed", "tactical_stratum"),
    seed: int = 20260816,
    require_reserve_belief: bool = True,
) -> BankBalanceResult:
    if not quota or not fields:
        raise ValueError("quota and fields cannot be empty")
    normalized_quota = {
        tuple(str(item) for item in key): int(value) for key, value in quota.items()
    }
    if any(
        len(key) != len(fields) or value < 1
        for key, value in normalized_quota.items()
    ):
        raise ValueError("quota keys must match fields and values must be positive")
    candidates: dict[tuple[str, ...], list[tuple[int, str, dict[str, Any]]]] = {
        key: [] for key in normalized_quota
    }
    seen: set[str] = set()
    duplicates = 0
    for raw in rows:
        row = dict(raw)
        identity = source_identity(row)
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        if require_reserve_belief:
            _validate_belief(row, identity)
        key = tuple(str(_field(row, field)) for field in fields)
        if key not in candidates:
            continue
        candidates[key].append((_score(seed, identity), identity, row))
    selected: list[dict[str, Any]] = []
    strata: dict[str, int] = {}
    shortfall: dict[str, int] = {}
    quota_flat: dict[str, int] = {}
    for key in sorted(normalized_quota):
        label = "/".join(key)
        target = normalized_quota[key]
        quota_flat[label] = target
        available = sorted(candidates[key], key=lambda item: (item[0], item[1]))
        chosen = available[:target]
        selected.extend(item[2] for item in chosen)
        strata[label] = len(chosen)
        shortfall[label] = max(0, target - len(chosen))
    selected.sort(
        key=lambda row: (
            tuple(str(_field(row, field)) for field in fields),
            _score(seed, source_identity(row)),
            source_identity(row),
        )
    )
    return BankBalanceResult(
        selected=tuple(selected),
        strata=strata,
        quota=quota_flat,
        shortfall=shortfall,
        duplicates=duplicates,
    )


def bank_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    identities = sorted(source_identity(row) for row in rows)
    return hashlib.sha256(canonical_json(identities)).hexdigest()


def quota_from_json(payload: Mapping[str, Any]) -> dict[tuple[str, ...], int]:
    result: dict[tuple[str, ...], int] = {}
    for key, value in payload.items():
        if isinstance(key, str):
            components = tuple(key.split("/"))
        else:
            components = tuple(str(item) for item in key)  # type: ignore[union-attr]
        result[components] = int(value)
    return result


__all__ = [
    "BankBalanceResult",
    "DEFAULT_LEVELS",
    "DEFAULT_SPEEDS",
    "DEFAULT_TACTICAL_STRATA",
    "balance_state_rows",
    "bank_identity",
    "cross_product_quota",
    "quota_from_json",
    "select_game_rows",
]
