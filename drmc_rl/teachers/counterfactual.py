"""Full-candidate counterfactual labels from strict pair-event search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from drmc_rl.search.joint_event import JointEventSearch, PairSearchModel, SearchConfig, WDL

StateT = TypeVar("StateT")


def _logit(probability: float, epsilon: float = 1e-5) -> float:
    value = float(np.clip(probability, epsilon, 1.0 - epsilon))
    return float(np.log(value) - np.log1p(-value))


def win_logit_regret(values: Sequence[WDL]) -> np.ndarray:
    """Return non-negative regret in a stable win-probability logit scale."""

    if not values:
        return np.empty(0, dtype=np.float32)
    logits = np.asarray([_logit(value.win) for value in values], dtype=np.float64)
    return (logits.max() - logits).astype(np.float32)


@dataclass(frozen=True, slots=True)
class CandidateCounterfactual:
    action: int
    win: float
    draw: float
    loss: float
    utility: float
    expected_score: float
    regret_win_logit: float
    uncertainty: float | None
    policy_target: float
    rank: int
    consequences: dict[str, float | int | bool]


@dataclass(frozen=True, slots=True)
class CounterfactualLabel:
    schema: str
    state_key: str
    root_side: int
    best_action: int
    candidates: tuple[CandidateCounterfactual, ...]
    root_win: float
    root_draw: float
    root_loss: float
    nodes: int
    cache_hits: int
    chance_nodes: int
    chance_outcomes: int
    teacher_count: int
    uncertainty_available: bool
    budget_exhausted: bool
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class CounterfactualTeacher(Generic[StateT]):
    """Label every root candidate with ensemble W/D/L and calibrated regret."""

    def __init__(
        self,
        models: PairSearchModel[StateT] | Sequence[PairSearchModel[StateT]],
        *,
        config: SearchConfig | None = None,
    ) -> None:
        if isinstance(models, Sequence):
            self.models = tuple(models)
        else:
            self.models = (models,)
        if not self.models:
            raise ValueError("counterfactual teacher requires at least one model")
        self.config = config or SearchConfig(own_beam=512)

    def label(
        self,
        state: StateT,
        *,
        root_side: int,
        metadata: dict[str, object] | None = None,
    ) -> CounterfactualLabel:
        legal = tuple(int(action) for action in self.models[0].legal_actions(state, root_side))
        if not legal:
            raise ValueError("cannot label a state without legal root actions")
        results = [
            JointEventSearch(model, self.config).search(state, root_side=root_side)
            for model in self.models
        ]
        for result in results:
            if set(result.actions) != set(legal):
                missing = sorted(set(legal) - set(result.actions))
                raise RuntimeError(
                    "counterfactual labeling must cover every legal action; "
                    f"increase own_beam (missing {missing[:8]})"
                )
        by_action: dict[int, list[WDL]] = {action: [] for action in legal}
        policies: dict[int, list[float]] = {action: [] for action in legal}
        for result in results:
            for action, value, target in zip(
                result.actions, result.values, result.policy_target, strict=True
            ):
                by_action[int(action)].append(value)
                policies[int(action)].append(float(target))
        mean_values: list[WDL] = []
        uncertainty: list[float | None] = []
        policy: list[float] = []
        for action in legal:
            values = by_action[action]
            mean_values.append(WDL.mixture(np.ones(len(values)), values))
            uncertainty.append(
                float(np.std([value.utility for value in values], ddof=0))
                if len(values) > 1
                else None
            )
            policy.append(float(np.mean(policies[action])))
        regrets = win_logit_regret(mean_values)
        utilities = np.asarray([value.utility for value in mean_values], dtype=np.float64)
        order = np.argsort(-utilities, kind="stable")
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        candidates = tuple(
            CandidateCounterfactual(
                action=int(action),
                win=float(value.win),
                draw=float(value.draw),
                loss=float(value.loss),
                utility=float(value.utility),
                expected_score=float(value.expected_score),
                regret_win_logit=float(regret),
                uncertainty=None if sigma is None else float(sigma),
                policy_target=float(target),
                rank=int(rank),
                consequences=self._consequences(state, root_side, int(action)),
            )
            for action, value, regret, sigma, target, rank in zip(
                legal,
                mean_values,
                regrets,
                uncertainty,
                policy,
                ranks,
                strict=True,
            )
        )
        best_index = int(np.argmax(utilities))
        root = WDL.mixture(np.asarray(policy), mean_values)
        return CounterfactualLabel(
            schema="drmc-counterfactual-pair-label-v2",
            state_key=str(self.models[0].key(state)),
            root_side=int(root_side),
            best_action=int(legal[best_index]),
            candidates=candidates,
            root_win=root.win,
            root_draw=root.draw,
            root_loss=root.loss,
            nodes=int(sum(result.nodes for result in results)),
            cache_hits=int(sum(result.cache_hits for result in results)),
            chance_nodes=int(sum(result.chance_nodes for result in results)),
            chance_outcomes=int(sum(result.chance_outcomes for result in results)),
            teacher_count=len(results),
            uncertainty_available=len(results) > 1,
            budget_exhausted=any(result.budget_exhausted for result in results),
            metadata=dict(metadata or {}),
        )

    def _consequences(
        self, state: StateT, root_side: int, action: int
    ) -> dict[str, float | int | bool]:
        provider = getattr(self.models[0], "candidate_consequences", None)
        if provider is None:
            return {}
        value = provider(state, root_side, action)
        if not isinstance(value, Mapping):
            raise TypeError("candidate_consequences must return a mapping")
        allowed = (bool, int, float)
        result: dict[str, float | int | bool] = {}
        for key, item in value.items():
            if not isinstance(item, allowed) or not np.isfinite(float(item)):
                raise ValueError(f"invalid candidate consequence {key!r}={item!r}")
            result[str(key)] = item
        return result

    def label_many(
        self,
        states: Iterable[StateT],
        *,
        root_side: int,
    ) -> Iterable[CounterfactualLabel]:
        for state in states:
            yield self.label(state, root_side=root_side)


__all__ = [
    "CandidateCounterfactual",
    "CounterfactualLabel",
    "CounterfactualTeacher",
    "win_logit_regret",
]
