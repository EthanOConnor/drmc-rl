"""Unified decision decoder for quality, human-rate, and trainer products.

Strength, style, cadence, and execution feasibility are deliberately separate:

1. the competitive core supplies rating-independent win probabilities;
2. an execution profile removes mechanically unavailable scripts;
3. trainer strength chooses a calibrated regret envelope;
4. style chooses only inside that envelope;
5. cadence selects an exact replay-valid script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

import numpy as np

from drmc_rl.execution.profile import ExecutionProfile, ScriptMetrics


class ProductMode(str, Enum):
    UNRESTRICTED = "unrestricted"
    HUMAN_RATE = "human_rate"
    TRAINER = "trainer"


class RegretSampler(Protocol):
    def sample(self, rating: float, opportunity: float, rng: np.random.Generator) -> float: ...

    def parameters(self, rating: float, opportunity: float) -> tuple[float, float]: ...


class CadenceSampler(Protocol):
    def sample_slack(self, context: "DecisionContext", rng: np.random.Generator) -> int: ...


@dataclass(frozen=True, slots=True)
class DecisionContext:
    rating: float
    opponent_rating: float | None = None
    game_phase: float = 0.0
    pressure: float = 0.0
    incoming_garbage: float = 0.0
    speed: float = 0.0
    candidate_complexity: float = 0.0
    previous_tau_frames: float = 0.0

    def __post_init__(self) -> None:
        for name in ("game_phase", "pressure", "incoming_garbage", "speed", "candidate_complexity"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")


@dataclass(frozen=True, slots=True)
class CandidateOption:
    action: int
    competitive_win_probability: float
    human_logit: float
    scripts: tuple[np.ndarray, ...]
    style_features: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability = float(self.competitive_win_probability)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("competitive_win_probability must be in [0,1]")
        if not np.isfinite(float(self.human_logit)):
            raise ValueError("human_logit must be finite")
        object.__setattr__(
            self,
            "scripts",
            tuple(np.asarray(script, dtype=np.uint8).reshape(-1) for script in self.scripts),
        )
        object.__setattr__(
            self,
            "style_features",
            np.asarray(self.style_features, dtype=np.float32).reshape(-1),
        )


@dataclass(slots=True)
class HumanFormState:
    """Slowly varying latent form, preserving temporal error correlation."""

    value: float = 0.0
    persistence: float = 0.94
    innovation_std: float = 0.12
    regret_feedback: float = 0.05

    def advance(self, rng: np.random.Generator, *, chosen_regret: float = 0.0) -> float:
        innovation = float(rng.normal(0.0, self.innovation_std))
        feedback = self.regret_feedback * np.tanh(float(chosen_regret))
        self.value = float(np.clip(self.persistence * self.value + innovation + feedback, -1.5, 1.5))
        return self.value


@dataclass(frozen=True, slots=True)
class ContextualRegretAdjustment:
    phase: float = 0.05
    pressure: float = 0.35
    incoming_garbage: float = 0.20
    speed: float = 0.20
    complexity: float = 0.30
    rating_gap: float = -0.05
    form_scale: float = 0.25

    @classmethod
    def fit(
        cls,
        contexts: Sequence[DecisionContext],
        observed_regret: Sequence[float],
        base_regret: Sequence[float],
        *,
        forms: Sequence[float] | None = None,
        ridge: float = 1.0,
    ) -> "ContextualRegretAdjustment":
        """Fit multiplicative context residuals from held-out human choices.

        The base calibration retains its monotone rating-conditioned quantile
        curves. This regression learns only a log-multiplier around that base,
        so it cannot redefine competitive quality or style.
        """

        if len(contexts) != len(observed_regret) or len(contexts) != len(base_regret):
            raise ValueError("contextual regret fit arrays must have equal length")
        form_values = np.zeros(len(contexts)) if forms is None else np.asarray(forms, dtype=float)
        if len(form_values) != len(contexts):
            raise ValueError("forms must match contexts")
        rows = []
        for context, form in zip(contexts, form_values, strict=True):
            opponent = context.rating if context.opponent_rating is None else context.opponent_rating
            rows.append(
                (
                    context.game_phase,
                    context.pressure,
                    context.incoming_garbage,
                    context.speed,
                    context.candidate_complexity,
                    np.clip((context.rating - opponent) / 1000.0, -1.0, 1.0),
                    float(form),
                )
            )
        x = np.asarray(rows, dtype=np.float64)
        observed = np.maximum(np.asarray(observed_regret, dtype=np.float64), 0.0)
        base = np.maximum(np.asarray(base_regret, dtype=np.float64), 0.0)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(observed) & np.isfinite(base)
        if valid.sum() < x.shape[1] + 2:
            raise ValueError("not enough valid rows to fit contextual regret")
        y = np.log1p(observed[valid]) - np.log1p(base[valid])
        design = x[valid]
        normal = design.T @ design + float(ridge) * np.eye(design.shape[1])
        coefficients = np.linalg.solve(normal, design.T @ y)
        return cls(*map(float, coefficients))

    def multiplier(self, context: DecisionContext, form: float) -> float:
        opponent = context.rating if context.opponent_rating is None else context.opponent_rating
        rating_gap = np.clip((context.rating - opponent) / 1000.0, -1.0, 1.0)
        log_multiplier = (
            self.phase * context.game_phase
            + self.pressure * context.pressure
            + self.incoming_garbage * context.incoming_garbage
            + self.speed * context.speed
            + self.complexity * context.candidate_complexity
            + self.rating_gap * rating_gap
            + self.form_scale * float(form)
        )
        return float(np.exp(np.clip(log_multiplier, -1.0, 1.0)))


@dataclass(frozen=True, slots=True)
class DecisionResult:
    action: int
    script: np.ndarray
    chosen_slot: int
    mode: ProductMode
    chosen_win_probability: float
    best_win_probability: float
    regret_win_logit: float
    target_regret: float
    opportunity: float
    style_score: float
    execution_profile: str
    script_metrics: ScriptMetrics
    diagnostics: Mapping[str, object]


class FixedCadence:
    def __init__(self, slack_frames: int = 0):
        self.slack_frames = max(0, int(slack_frames))

    def sample_slack(self, context: DecisionContext, rng: np.random.Generator) -> int:
        return self.slack_frames


def _logit(probability: np.ndarray | float, epsilon: float = 1e-5):
    p = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


class UnifiedDecisionDecoder:
    def __init__(
        self,
        *,
        mode: ProductMode,
        execution_profile: ExecutionProfile | None = None,
        regret: RegretSampler | None = None,
        cadence: CadenceSampler | None = None,
        style_vector: Sequence[float] = (),
        style_temperature: float = 0.0,
        adjustment: ContextualRegretAdjustment | None = None,
        form: HumanFormState | None = None,
        seed: int = 0,
    ) -> None:
        self.mode = ProductMode(mode)
        if self.mode == ProductMode.HUMAN_RATE and execution_profile is None:
            raise ValueError("human-rate mode requires an explicit named execution profile")
        self.profile = execution_profile or ExecutionProfile.unrestricted()
        self.regret = regret
        self.cadence = cadence or FixedCadence(0)
        self.style_vector = np.asarray(style_vector, dtype=np.float64).reshape(-1)
        self.style_temperature = max(0.0, float(style_temperature))
        self.adjustment = adjustment or ContextualRegretAdjustment()
        self.form = form or HumanFormState()
        self.rng = np.random.default_rng(int(seed))
        if self.mode == ProductMode.TRAINER and self.regret is None:
            raise ValueError("trainer mode requires a calibrated regret sampler")

    def choose(
        self,
        candidates: Sequence[CandidateOption],
        *,
        context: DecisionContext,
    ) -> DecisionResult:
        if not candidates:
            raise ValueError("cannot choose without candidates")
        feasible: list[tuple[int, CandidateOption, list[tuple[np.ndarray, ScriptMetrics]]]] = []
        rejected: dict[int, list[str]] = {}
        for slot, candidate in enumerate(candidates):
            scripts: list[tuple[np.ndarray, ScriptMetrics]] = []
            reasons: list[str] = []
            for script in candidate.scripts:
                validation = self.profile.validate(script)
                if validation.valid:
                    scripts.append((script, validation.metrics))
                else:
                    reasons.extend(validation.violations)
            if scripts:
                feasible.append((slot, candidate, scripts))
            else:
                rejected[slot] = sorted(set(reasons or ["no_script"]))
        if not feasible:
            raise RuntimeError(
                f"no candidate has a script valid under execution profile {self.profile.id!r}"
            )

        probabilities = np.asarray(
            [candidate.competitive_win_probability for _, candidate, _ in feasible],
            dtype=np.float64,
        )
        logits = _logit(probabilities)
        best_logit = float(logits.max())
        regrets = np.maximum(best_logit - logits, 0.0)
        opportunity = float(np.std(logits))
        best_probability = float(probabilities[int(np.argmax(logits))])

        if self.mode in {ProductMode.UNRESTRICTED, ProductMode.HUMAN_RATE}:
            local = int(np.argmax(logits))
            target_regret = 0.0
            style_score = float(feasible[local][1].human_logit)
        else:
            assert self.regret is not None
            form_value = self.form.value
            sampled = float(self.regret.sample(context.rating, opportunity, self.rng))
            _median, tolerance = self.regret.parameters(context.rating, opportunity)
            multiplier = self.adjustment.multiplier(context, form_value)
            target_regret = max(0.0, sampled * multiplier)
            tolerance = max(float(tolerance), 0.02)
            distance = np.abs(np.log1p(regrets) - np.log1p(target_regret))
            closest = float(distance.min())
            plausible = distance <= closest + tolerance
            style_scores = np.full(len(feasible), -np.inf, dtype=np.float64)
            for index, (_slot, candidate, _scripts) in enumerate(feasible):
                if not plausible[index]:
                    continue
                features = candidate.style_features.astype(np.float64)
                if self.style_vector.size and features.size != self.style_vector.size:
                    raise ValueError(
                        f"style feature dimension {features.size} does not match control "
                        f"dimension {self.style_vector.size}"
                    )
                style_scores[index] = float(candidate.human_logit) + (
                    float(features @ self.style_vector) if self.style_vector.size else 0.0
                )
            if not np.isfinite(style_scores).any():
                local = int(np.argmin(distance))
            elif self.style_temperature <= 0:
                local = int(np.argmax(style_scores))
            else:
                finite = np.isfinite(style_scores)
                scaled = np.where(
                    finite,
                    (style_scores - np.max(style_scores[finite])) / self.style_temperature,
                    -np.inf,
                )
                weights = np.where(finite, np.exp(np.clip(scaled, -60.0, 0.0)), 0.0)
                weights /= weights.sum()
                local = int(self.rng.choice(len(feasible), p=weights))
            style_score = float(style_scores[local])
            self.form.advance(self.rng, chosen_regret=float(regrets[local]))

        slot, candidate, scripts = feasible[local]
        requested_slack = max(0, int(self.cadence.sample_slack(context, self.rng)))
        fastest = min(metrics.frames for _script, metrics in scripts)
        desired_frames = fastest + requested_slack
        # Cadence changes script choice, never candidate quality. Ties prefer
        # lower complexity and then fewer edges.
        script, metrics = min(
            scripts,
            key=lambda item: (
                abs(item[1].frames - desired_frames),
                item[1].complexity,
                item[1].total_edges,
                item[1].frames,
            ),
        )
        return DecisionResult(
            action=int(candidate.action),
            script=script.copy(),
            chosen_slot=int(slot),
            mode=self.mode,
            chosen_win_probability=float(candidate.competitive_win_probability),
            best_win_probability=best_probability,
            regret_win_logit=float(regrets[local]),
            target_regret=float(target_regret),
            opportunity=opportunity,
            style_score=style_score,
            execution_profile=self.profile.id,
            script_metrics=metrics,
            diagnostics={
                "requested_slack_frames": requested_slack,
                "realized_slack_frames": metrics.frames - fastest,
                "feasible_candidates": len(feasible),
                "rejected_candidates": rejected,
                "form": self.form.value,
                "strength_mechanism": (
                    "quality_argmax"
                    if self.mode != ProductMode.TRAINER
                    else "calibrated_win_logit_regret"
                ),
            },
        )


__all__ = [
    "CadenceSampler",
    "CandidateOption",
    "ContextualRegretAdjustment",
    "DecisionContext",
    "DecisionResult",
    "FixedCadence",
    "HumanFormState",
    "ProductMode",
    "RegretSampler",
    "UnifiedDecisionDecoder",
]
