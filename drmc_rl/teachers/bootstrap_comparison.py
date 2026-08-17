"""Paired whole-game comparison against an observed-action/V3 bootstrap."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from drmc_rl.eval.wdl_calibration import paired_game_bootstrap, weighted_metrics
from drmc_rl.teachers.release_analysis import ReleaseDataset


@dataclass(frozen=True, slots=True)
class BootstrapRow:
    source_id: str
    game_id: str
    outcome: int
    observed_action: int
    baseline_wdl: tuple[float, float, float]
    stratum: tuple[str, ...] = ()


def _outcome(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"win", "w"}:
            return 0
        if normalized in {"draw", "d"}:
            return 1
        if normalized in {"loss", "l"}:
            return 2
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError(f"invalid W/D/L outcome {value!r}")
    return result


def _candidate_wdl(payload: Mapping[str, Any], action: int) -> tuple[float, float, float]:
    direct = payload.get("baseline_wdl")
    if direct is not None:
        values = tuple(float(item) for item in direct)
    else:
        raw = payload.get("candidates") or ()
        matching = [item for item in raw if int(item["action"]) == int(action)]
        if len(matching) != 1:
            raise ValueError(f"baseline requires exactly one candidate for action {action}")
        item = matching[0]
        values = (float(item["win"]), float(item["draw"]), float(item["loss"]))
    if len(values) != 3 or not np.isfinite(values).all() or min(values) < 0:
        raise ValueError(f"invalid baseline W/D/L {values!r}")
    total = float(sum(values))
    if not np.isclose(total, 1.0, atol=2e-6):
        raise ValueError(f"baseline W/D/L is not normalized: {values!r}")
    return values  # type: ignore[return-value]


def parse_bootstrap_rows(rows: Iterable[Mapping[str, Any]]) -> list[BootstrapRow]:
    result: list[BootstrapRow] = []
    seen: set[str] = set()
    for index, payload in enumerate(rows, 1):
        source_id = str(payload.get("source_id", ""))
        if not source_id or source_id in seen:
            raise ValueError(f"missing or duplicate source_id at bootstrap row {index}")
        seen.add(source_id)
        action = int(payload["observed_action"])
        result.append(
            BootstrapRow(
                source_id=source_id,
                game_id=str(payload["game_id"]),
                outcome=_outcome(payload["outcome"]),
                observed_action=action,
                baseline_wdl=_candidate_wdl(payload, action),
                stratum=tuple(str(item) for item in payload.get("stratum", ())),
            )
        )
    if not result:
        raise ValueError("bootstrap dataset is empty")
    return result


def load_bootstrap_rows(path: str | Path) -> list[BootstrapRow]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8") as handle:
        return parse_bootstrap_rows(json.loads(line) for line in handle if line.strip())


def compare_bootstrap(
    release: ReleaseDataset,
    bootstrap: list[BootstrapRow],
    *,
    seed: int = 20260816,
    bootstrap_samples: int = 4000,
) -> dict[str, Any]:
    cf_probability: list[tuple[float, float, float]] = []
    baseline_probability: list[tuple[float, float, float]] = []
    outcomes: list[int] = []
    groups: list[str] = []
    strata: list[tuple[str, ...]] = []
    missing_sources: list[str] = []
    missing_actions: list[tuple[str, int]] = []
    for row in bootstrap:
        state = release.states.get(row.source_id)
        if state is None:
            missing_sources.append(row.source_id)
            continue
        candidate = state.candidates.get(row.observed_action)
        if candidate is None:
            missing_actions.append((row.source_id, row.observed_action))
            continue
        cf_probability.append(
            (float(candidate["win"]), float(candidate["draw"]), float(candidate["loss"]))
        )
        baseline_probability.append(row.baseline_wdl)
        outcomes.append(row.outcome)
        groups.append(row.game_id)
        strata.append(row.stratum or state.stratum)
    if missing_sources or missing_actions:
        raise ValueError(
            "bootstrap/release coverage mismatch: "
            f"missing_sources={missing_sources[:8]}, missing_actions={missing_actions[:8]}"
        )
    cf = np.asarray(cf_probability, dtype=np.float64)
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    target = np.asarray(outcomes, dtype=np.int64)
    group = np.asarray(groups)
    if len(np.unique(group)) < 2:
        raise ValueError("bootstrap comparison requires at least two games")

    def metrics_for(indices: np.ndarray) -> dict[str, Any]:
        return {
            "rows": int(len(indices)),
            "games": int(len(np.unique(group[indices]))),
            "counterfactual": weighted_metrics(cf[indices], target[indices], group[indices]),
            "v3_bootstrap": weighted_metrics(
                baseline[indices], target[indices], group[indices]
            ),
            "paired_game_bootstrap": paired_game_bootstrap(
                cf[indices],
                baseline[indices],
                target[indices],
                group[indices],
                seed=seed,
                samples=bootstrap_samples,
            ),
        }

    all_indices = np.arange(len(target), dtype=np.int64)
    by_stratum: dict[str, dict[str, Any]] = {}
    for value in sorted(set(strata)):
        indices = np.asarray([index for index, item in enumerate(strata) if item == value])
        if len(indices) and len(np.unique(group[indices])) >= 2:
            by_stratum["/".join(value) or "unspecified"] = metrics_for(indices)
    aggregate = metrics_for(all_indices)
    return {
        "schema": "drmc-counterfactual-v3-bootstrap-comparison-v1",
        "release_sha256": list(release.release_sha256),
        "chance_model": release.chance_model,
        "information_scope": release.information_scope,
        "rows": aggregate["rows"],
        "games": aggregate["games"],
        "counterfactual": aggregate["counterfactual"],
        "v3_bootstrap": aggregate["v3_bootstrap"],
        "paired_game_bootstrap": aggregate["paired_game_bootstrap"],
        "outcomes": {
            "win": int((target == 0).sum()),
            "draw": int((target == 1).sum()),
            "loss": int((target == 2).sum()),
        },
        "by_stratum": by_stratum,
    }


__all__ = [
    "BootstrapRow",
    "compare_bootstrap",
    "load_bootstrap_rows",
    "parse_bootstrap_rows",
]
