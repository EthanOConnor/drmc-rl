"""Hierarchical Bayesian ratings for immutable arena checkpoints.

The arena is a fixed-skill paired-comparison problem, not an online player
rating problem.  This module therefore fits the complete match history with a
Davidson W/D/L likelihood and a checkpoint-lineage hierarchy.  It deliberately
contains no P1/P2 term: physical sides are balanced by the scheduler and any
measured side asymmetry is an arena diagnostic, not skill.

Posterior inference uses adaptive Hamiltonian Monte Carlo with multiple chains.
The implementation is NumPy-only so rating remains a lightweight arena service
and does not inherit the training stack.
"""

from __future__ import annotations

import math
from io import BytesIO
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np


MODEL_VERSION = "hierarchical-davidson-hmc-v1"
ELO_SCALE = 400.0 / math.log(10.0)


@dataclass(frozen=True)
class PairCounts:
    """Aggregated outcomes for one canonical pair (``i < j``)."""

    i: int
    j: int
    wins_i: int
    draws: int
    wins_j: int

    @property
    def games(self) -> int:
        return self.wins_i + self.draws + self.wins_j


@dataclass(frozen=True)
class RatingConfig:
    chains: int = 4
    warmup: int = 800
    samples: int = 1_200
    leapfrog_min: int = 20
    leapfrog_max: int = 50
    target_accept: float = 0.95
    root_skill_sd: float = 3.5
    lineage_scale: float = 0.9
    draw_scale: float = 1.0
    seed: int = 0xD0C7_0A11
    max_rhat: float = 1.01
    min_ess: float = 400.0
    require_convergence: bool = True

    def __post_init__(self) -> None:
        if self.chains < 2:
            raise ValueError("at least two HMC chains are required")
        if self.warmup < 20 or self.samples < 20:
            raise ValueError("warmup and samples must each be at least 20")
        if not (1 <= self.leapfrog_min <= self.leapfrog_max):
            raise ValueError("invalid leapfrog range")
        if not 0.5 < self.target_accept < 1.0:
            raise ValueError("target_accept must be between 0.5 and 1")
        if min(self.root_skill_sd, self.lineage_scale, self.draw_scale) <= 0:
            raise ValueError("prior scales must be positive")


@dataclass(frozen=True)
class AgentRating:
    mean: float
    sd: float
    low: float
    high: float
    probability_best: float
    rank_median: float
    rank_low: float
    rank_high: float
    draw_propensity: float
    probability_better_parent: float | None


@dataclass(frozen=True)
class RatingFit:
    agents: tuple[AgentRating, ...]
    diagnostics: dict[str, float | int | str]
    hyperparameters: dict[str, float]
    samples: "PosteriorSamples"


@dataclass(frozen=True)
class PosteriorSamples:
    skills: np.ndarray
    draw_tendencies: np.ndarray
    draw_intercepts: np.ndarray
    lineage_scales: np.ndarray
    draw_scales: np.ndarray
    log_weights: np.ndarray
    parents: np.ndarray

    def encode(self) -> bytes:
        buffer = BytesIO()
        np.savez_compressed(
            buffer,
            skills=self.skills,
            draw_tendencies=self.draw_tendencies,
            draw_intercepts=self.draw_intercepts,
            lineage_scales=self.lineage_scales,
            draw_scales=self.draw_scales,
            log_weights=self.log_weights,
            parents=self.parents,
        )
        return buffer.getvalue()

    @classmethod
    def decode(cls, payload: bytes) -> "PosteriorSamples":
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            names = (
                "skills", "draw_tendencies", "draw_intercepts",
                "lineage_scales", "draw_scales", "log_weights",
            )
            values = [np.asarray(arrays[name]) for name in names]
            parents = (
                np.asarray(arrays["parents"], dtype=np.int32)
                if "parents" in arrays
                else np.full(values[0].shape[1], -1, dtype=np.int32)
            )
            return cls(*values, parents)


class RatingConvergenceError(RuntimeError):
    pass


class DavidsonPosterior:
    """Differentiable posterior in a non-centered lineage parameterization."""

    def __init__(
        self,
        num_agents: int,
        pairs: Sequence[PairCounts],
        parents: Sequence[int | None],
        *,
        root_skill_sd: float = 3.5,
        lineage_scale: float = 0.9,
        draw_scale: float = 1.0,
    ) -> None:
        self.n = int(num_agents)
        if self.n <= 0 or len(parents) != self.n:
            raise ValueError("parents must contain one entry per agent")
        self.parents = tuple(None if p is None else int(p) for p in parents)
        self.topological = self._topological_order()
        self.roots = tuple(i for i, p in enumerate(self.parents) if p is None)
        self.children = tuple(i for i, p in enumerate(self.parents) if p is not None)
        self.root_skill_sd = float(root_skill_sd)
        self.lineage_scale = float(lineage_scale)
        self.draw_scale = float(draw_scale)
        self.pairs = tuple(pairs)
        for pair in self.pairs:
            if not (0 <= pair.i < pair.j < self.n) or min(
                pair.wins_i, pair.draws, pair.wins_j
            ) < 0:
                raise ValueError(f"invalid pair counts: {pair}")
        self.pair_i = np.asarray([pair.i for pair in self.pairs], dtype=np.intp)
        self.pair_j = np.asarray([pair.j for pair in self.pairs], dtype=np.intp)
        self.counts = np.asarray(
            [(pair.wins_i, pair.draws, pair.wins_j) for pair in self.pairs],
            dtype=np.float64,
        )
        self.game_counts = self.counts.sum(axis=1)
        agent_games = np.zeros(self.n, dtype=np.float64)
        np.add.at(agent_games, self.pair_i, self.game_counts)
        np.add.at(agent_games, self.pair_j, self.game_counts)
        # Centered lineage parameters mix much better when a checkpoint has
        # substantial direct evidence; a fresh child stays non-centered to
        # avoid the classic hierarchical funnel before its first games.
        self.noncentered = tuple(
            self.parents[i] is not None and agent_games[i] < 50 for i in range(self.n)
        )

    @property
    def dimension(self) -> int:
        # non-centered skill n, draw tendency n, intercept, log(tau), log(sigma_draw)
        return 2 * self.n + 3

    def _topological_order(self) -> tuple[int, ...]:
        state = np.zeros(self.n, dtype=np.int8)
        order: list[int] = []

        def visit(i: int) -> None:
            if state[i] == 1:
                raise ValueError("checkpoint lineage contains a cycle")
            if state[i] == 2:
                return
            state[i] = 1
            parent = self.parents[i]
            if parent is not None:
                if not 0 <= parent < self.n or parent == i:
                    raise ValueError(f"invalid parent {parent} for agent {i}")
                visit(parent)
            state[i] = 2
            order.append(i)

        for agent in range(self.n):
            visit(agent)
        return tuple(order)

    def initial_position(self) -> np.ndarray:
        position = np.zeros(self.dimension, dtype=np.float64)
        total = sum(pair.games for pair in self.pairs)
        draws = sum(pair.draws for pair in self.pairs)
        draw_rate = (draws + 0.5) / (total + 1.0)
        # At equal skill Davidson gives p(draw)=exp(delta)/(2+exp(delta)).
        position[2 * self.n] = math.log(2.0 * draw_rate / max(1.0 - draw_rate, 1e-6))
        position[2 * self.n + 1] = math.log(0.6)
        position[2 * self.n + 2] = math.log(0.5)
        return position

    def unpack(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        x = position[: self.n]
        draw_z = position[self.n : 2 * self.n]
        delta = float(position[2 * self.n])
        tau = math.exp(float(position[2 * self.n + 1]))
        sigma_draw = math.exp(float(position[2 * self.n + 2]))
        skills = np.empty(self.n, dtype=np.float64)
        for i in self.topological:
            parent = self.parents[i]
            skills[i] = (
                skills[parent] + tau * x[i]
                if parent is not None and self.noncentered[i]
                else x[i]
            )
        draw_tendency = sigma_draw * (draw_z - draw_z.mean())
        return skills, draw_tendency, delta, tau, sigma_draw

    def log_density_and_grad(self, position: np.ndarray) -> tuple[float, np.ndarray]:
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (self.dimension,):
            raise ValueError(f"expected position shape {(self.dimension,)}, got {position.shape}")
        if (
            not np.all(np.isfinite(position))
            or abs(float(position[2 * self.n + 1])) > 30.0
            or abs(float(position[2 * self.n + 2])) > 30.0
        ):
            return -math.inf, np.zeros_like(position)
        skills, draw_tendency, delta, tau, sigma_draw = self.unpack(position)
        x = position[: self.n]
        draw_z = position[self.n : 2 * self.n]

        logp = 0.0
        grad_skill = np.zeros(self.n, dtype=np.float64)
        grad_draw = np.zeros(self.n, dtype=np.float64)
        grad_delta = 0.0
        if len(self.pairs):
            skill_gap = skills[self.pair_i] - skills[self.pair_j]
            draw_logit = delta + 0.5 * (
                draw_tendency[self.pair_i] + draw_tendency[self.pair_j]
            )
            logits = np.column_stack((0.5 * skill_gap, draw_logit, -0.5 * skill_gap))
            maximum = logits.max(axis=1)
            exp_logits = np.exp(logits - maximum[:, None])
            normalizers = exp_logits.sum(axis=1)
            log_norm = maximum + np.log(normalizers)
            logp += float(np.sum(self.counts * logits) - self.game_counts @ log_norm)
            residual = self.counts - self.game_counts[:, None] * exp_logits / normalizers[:, None]
            skill_gradient = 0.5 * (residual[:, 0] - residual[:, 2])
            draw_gradient = residual[:, 1]
            np.add.at(grad_skill, self.pair_i, skill_gradient)
            np.add.at(grad_skill, self.pair_j, -skill_gradient)
            np.add.at(grad_draw, self.pair_i, 0.5 * draw_gradient)
            np.add.at(grad_draw, self.pair_j, 0.5 * draw_gradient)
            grad_delta += float(draw_gradient.sum())

        # Center well-observed lineage edges directly. Fresh children remain
        # non-centered and are transformed in the reverse tree pass below.
        grad_log_tau = 0.0
        for i in self.children:
            if self.noncentered[i]:
                continue
            parent = self.parents[i]
            assert parent is not None
            residual = skills[i] - skills[parent]
            standardized = residual / tau
            logp -= 0.5 * standardized * standardized + math.log(tau)
            edge_gradient = residual / (tau * tau)
            grad_skill[i] -= edge_gradient
            grad_skill[parent] += edge_gradient
            grad_log_tau += standardized * standardized - 1.0

        # Reverse the remaining non-centered skill edges. Every descendant
        # contributes to its transformed ancestors and to the scale tau.
        accumulated = grad_skill.copy()
        grad_x = np.zeros(self.n, dtype=np.float64)
        for i in reversed(self.topological):
            parent = self.parents[i]
            if parent is None or not self.noncentered[i]:
                grad_x[i] += accumulated[i]
            else:
                grad_x[i] += tau * accumulated[i]
                grad_log_tau += tau * x[i] * accumulated[i]
                accumulated[parent] += accumulated[i]

        root_precision = 1.0 / (self.root_skill_sd * self.root_skill_sd)
        for i in self.roots:
            logp -= 0.5 * root_precision * x[i] * x[i]
            grad_x[i] -= root_precision * x[i]
        for i in self.children:
            if not self.noncentered[i]:
                continue
            logp -= 0.5 * x[i] * x[i]
            grad_x[i] -= x[i]

        centered_draw_grad = grad_draw - grad_draw.mean()
        grad_draw_z = sigma_draw * centered_draw_grad - draw_z
        grad_log_sigma = float(grad_draw @ draw_tendency)
        logp -= 0.5 * float(draw_z @ draw_z)

        # Weakly informative Davidson intercept prior.
        delta_mean, delta_sd = -0.5, 1.5
        delta_residual = delta - delta_mean
        logp -= 0.5 * (delta_residual / delta_sd) ** 2
        grad_delta -= delta_residual / (delta_sd * delta_sd)

        # Half-normal hyperpriors, transformed through log(scale).  The +log
        # terms are the Jacobians and are required for correct HMC density.
        logp += float(position[2 * self.n + 1]) - 0.5 * (tau / self.lineage_scale) ** 2
        grad_log_tau += 1.0 - (tau / self.lineage_scale) ** 2
        logp += float(position[2 * self.n + 2]) - 0.5 * (sigma_draw / self.draw_scale) ** 2
        grad_log_sigma += 1.0 - (sigma_draw / self.draw_scale) ** 2

        gradient = np.concatenate(
            (grad_x, grad_draw_z, np.array((grad_delta, grad_log_tau, grad_log_sigma)))
        )
        return logp, gradient


def _posterior_mode(model: DavidsonPosterior, *, iterations: int = 2_000) -> np.ndarray:
    """Find a stable common chain center with Adam; inference remains HMC."""

    position = model.initial_position()
    first = np.zeros_like(position)
    second = np.zeros_like(position)
    best = position.copy()
    best_logp = -math.inf
    for iteration in range(1, iterations + 1):
        logp, gradient = model.log_density_and_grad(position)
        if not math.isfinite(logp) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("non-finite posterior during mode initialization")
        if logp > best_logp:
            best_logp, best = logp, position.copy()
        norm = float(np.linalg.norm(gradient))
        if norm > 2_000.0:
            gradient *= 2_000.0 / norm
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        corrected_first = first / (1.0 - 0.9**iteration)
        corrected_second = second / (1.0 - 0.999**iteration)
        learning_rate = 0.025 / math.sqrt(1.0 + iteration / 800.0)
        position += learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    position = best
    for _ in range(20):
        logp, gradient = model.log_density_and_grad(position)
        if float(np.max(np.abs(gradient))) < 1e-5:
            break
        hessian = _negative_hessian(model, position)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        inverse = (eigenvectors * (1.0 / np.clip(eigenvalues, 1e-4, 1e8))) @ eigenvectors.T
        step = inverse @ gradient
        step_norm = float(np.linalg.norm(step))
        if step_norm > 5.0:
            step *= 5.0 / step_norm
        scale = 1.0
        while scale >= 1e-4:
            proposal = position + scale * step
            proposal_logp = model.log_density_and_grad(proposal)[0]
            if math.isfinite(proposal_logp) and proposal_logp > logp:
                position = proposal
                break
            scale *= 0.5
        else:
            break
    return position


def _negative_hessian(model: DavidsonPosterior, position: np.ndarray) -> np.ndarray:
    dimension = len(position)
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    for i in range(dimension):
        step = 2e-4 * max(1.0, abs(float(position[i])))
        plus = position.copy()
        minus = position.copy()
        plus[i] += step
        minus[i] -= step
        grad_plus = model.log_density_and_grad(plus)[1]
        grad_minus = model.log_density_and_grad(minus)[1]
        hessian[:, i] = -(grad_plus - grad_minus) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


@dataclass(frozen=True)
class _HmcMetric:
    mass: np.ndarray
    inverse: np.ndarray
    cholesky: np.ndarray
    covariance_cholesky: np.ndarray

    def momentum(self, rng: np.random.Generator) -> np.ndarray:
        return self.cholesky @ rng.normal(size=len(self.mass))

    def velocity(self, momentum: np.ndarray) -> np.ndarray:
        return self.inverse @ momentum

    def kinetic_energy(self, momentum: np.ndarray) -> float:
        return 0.5 * float(momentum @ self.velocity(momentum))


def _curvature_metric(model: DavidsonPosterior, mode: np.ndarray) -> _HmcMetric:
    """Full local posterior geometry for efficient correlated HMC proposals."""

    hessian = _negative_hessian(model, mode)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.clip(eigenvalues, 1e-3, 1e6)
    mass = (eigenvectors * eigenvalues) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    cholesky = np.linalg.cholesky(mass)
    covariance_cholesky = np.linalg.cholesky(inverse)
    return _HmcMetric(mass, inverse, cholesky, covariance_cholesky)


def _leapfrog(
    model: DavidsonPosterior,
    position: np.ndarray,
    momentum: np.ndarray,
    gradient: np.ndarray,
    *,
    step_size: float,
    steps: int,
    metric: _HmcMetric,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    q = position.copy()
    p = momentum.copy()
    grad = gradient
    p += 0.5 * step_size * grad
    logp = -math.inf
    for step in range(steps):
        q += step_size * metric.velocity(p)
        logp, grad = model.log_density_and_grad(q)
        if not math.isfinite(logp) or not np.all(np.isfinite(grad)):
            return q, p, -math.inf, grad
        if step + 1 < steps:
            p += step_size * grad
    p += 0.5 * step_size * grad
    return q, p, logp, grad


def _hmc_transition(
    model: DavidsonPosterior,
    position: np.ndarray,
    logp: float,
    gradient: np.ndarray,
    *,
    step_size: float,
    steps: int,
    metric: _HmcMetric,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray, float, bool]:
    momentum = metric.momentum(rng)
    proposal, proposal_momentum, proposal_logp, proposal_grad = _leapfrog(
        model, position, momentum, gradient,
        step_size=step_size, steps=steps, metric=metric,
    )
    initial_energy = -logp + metric.kinetic_energy(momentum)
    proposal_energy = -proposal_logp + metric.kinetic_energy(proposal_momentum)
    energy_error = initial_energy - proposal_energy
    divergent = not math.isfinite(energy_error) or abs(energy_error) > 100.0
    acceptance = 0.0 if not math.isfinite(energy_error) else min(1.0, math.exp(min(0.0, energy_error)))
    if not divergent and rng.random() < acceptance:
        return proposal, proposal_logp, proposal_grad, acceptance, False
    return position, logp, gradient, acceptance, divergent


def _find_step_size(
    model: DavidsonPosterior,
    position: np.ndarray,
    metric: _HmcMetric,
    rng: np.random.Generator,
) -> float:
    logp, gradient = model.log_density_and_grad(position)
    step_size = 0.1
    for _ in range(24):
        _, _, _, acceptance, divergent = _hmc_transition(
            model, position, logp, gradient,
            step_size=step_size, steps=1, metric=metric, rng=rng
        )
        if divergent or acceptance < 0.5:
            step_size *= 0.5
        elif acceptance > 0.9:
            step_size *= 2.0
        else:
            break
    return float(np.clip(step_size, 1e-5, 1.0))


def _sample_chain(
    model: DavidsonPosterior,
    start: np.ndarray,
    metric: _HmcMetric,
    config: RatingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | int]]:
    position = start.copy()
    logp, gradient = model.log_density_and_grad(position)
    step_size = _find_step_size(model, position, metric, rng)

    # Hoffman-Gelman dual averaging for the integration step size.
    mu = math.log(10.0 * step_size)
    log_step = math.log(step_size)
    log_step_bar = log_step
    error_sum = 0.0
    warmup_divergences = 0
    accepts: list[float] = []
    adaptation_iteration = 0
    for iteration in range(1, config.warmup + 1):
        adaptation_iteration += 1
        steps = int(rng.integers(config.leapfrog_min, config.leapfrog_max + 1))
        position, logp, gradient, acceptance, divergent = _hmc_transition(
            model, position, logp, gradient,
            step_size=math.exp(float(np.clip(log_step, -12.0, 1.0))),
            steps=steps, metric=metric, rng=rng,
        )
        warmup_divergences += int(divergent)
        eta = 1.0 / (adaptation_iteration + 10.0)
        error_sum = (1.0 - eta) * error_sum + eta * (config.target_accept - acceptance)
        log_step = float(np.clip(
            mu - math.sqrt(adaptation_iteration) / 0.05 * error_sum, -12.0, 1.0
        ))
        weight = adaptation_iteration ** -0.75
        log_step_bar = weight * log_step + (1.0 - weight) * log_step_bar
    step_size = math.exp(float(np.clip(log_step_bar, -12.0, 1.0)))
    draws = np.empty((config.samples, model.dimension), dtype=np.float64)
    sampling_divergences = 0
    for sample in range(config.samples):
        steps = int(rng.integers(config.leapfrog_min, config.leapfrog_max + 1))
        position, logp, gradient, acceptance, divergent = _hmc_transition(
            model, position, logp, gradient,
            step_size=step_size, steps=steps, metric=metric, rng=rng,
        )
        accepts.append(acceptance)
        sampling_divergences += int(divergent)
        draws[sample] = position
    return draws, {
        "acceptance": float(np.mean(accepts)),
        "step_size": step_size,
        "warmup_divergences": warmup_divergences,
        "divergences": sampling_divergences,
    }


def _split_rhat_basic(chains: np.ndarray) -> np.ndarray:
    chain_count, sample_count, dimension = chains.shape
    half = sample_count // 2
    if half < 2:
        return np.full(dimension, math.inf)
    split = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = np.mean(np.var(split, axis=1, ddof=1), axis=0)
    between = half * np.var(np.mean(split, axis=1), axis=0, ddof=1)
    variance = (half - 1.0) / half * within + between / half
    return np.sqrt(np.divide(variance, within, out=np.ones_like(variance), where=within > 0))


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    flat = chains.reshape(-1, chains.shape[-1])
    normalized = np.empty_like(flat)
    normal = NormalDist()
    count = len(flat)
    for dimension in range(flat.shape[1]):
        order = np.argsort(flat[:, dimension], kind="mergesort")
        ranks = np.empty(count, dtype=np.float64)
        ranks[order] = np.arange(1, count + 1, dtype=np.float64)
        probabilities = (ranks - 0.375) / (count + 0.25)
        normalized[:, dimension] = np.fromiter(
            (normal.inv_cdf(float(value)) for value in probabilities),
            dtype=np.float64,
            count=count,
        )
    return normalized.reshape(chains.shape)


def _rank_normalized_rhat(chains: np.ndarray) -> np.ndarray:
    bulk = _split_rhat_basic(_rank_normalize(chains))
    folded = np.abs(chains - np.median(chains, axis=(0, 1), keepdims=True))
    return np.maximum(bulk, _split_rhat_basic(_rank_normalize(folded)))


def _effective_sample_size(chains: np.ndarray) -> np.ndarray:
    chain_count, sample_count, dimension = chains.shape
    centered = chains - chains.mean(axis=1, keepdims=True)
    size = 1 << (2 * sample_count - 1).bit_length()
    transform = np.fft.rfft(centered, n=size, axis=1)
    autocov = np.fft.irfft(transform * np.conjugate(transform), n=size, axis=1)
    autocov = autocov[:, :sample_count] / np.arange(sample_count, 0, -1)[None, :, None]
    variance = np.mean(autocov[:, 0, :], axis=0)
    rho = np.mean(autocov, axis=0) / np.maximum(variance, 1e-15)
    totals = np.ones(dimension)
    for lag in range(1, sample_count - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        positive = pair > 0
        totals += np.where(positive, 2.0 * pair, 0.0)
        rho[:, ~positive] = 0.0
    return np.minimum(chain_count * sample_count / np.maximum(totals, 1.0), chain_count * sample_count)


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(np.asarray(quantiles), cumulative, ordered_values)


def _summarize_samples(
    samples: PosteriorSamples,
) -> tuple[tuple[AgentRating, ...], dict[str, float]]:
    weights = _normalized_weights(samples.log_weights)
    elo = samples.skills * ELO_SCALE
    ranks = np.argsort(np.argsort(-elo, axis=1), axis=1) + 1
    best = np.argmax(elo, axis=1)
    summaries = []
    for i in range(elo.shape[1]):
        mean = float(weights @ elo[:, i])
        variance = float(weights @ ((elo[:, i] - mean) ** 2))
        low, high = _weighted_quantile(elo[:, i], weights, (0.025, 0.975))
        rank_low, rank_median, rank_high = _weighted_quantile(
            ranks[:, i].astype(np.float64), weights, (0.025, 0.5, 0.975)
        )
        summaries.append(AgentRating(
            mean=mean,
            sd=math.sqrt(max(variance, 0.0)),
            low=float(low),
            high=float(high),
            probability_best=float(weights[best == i].sum()),
            rank_median=float(rank_median),
            rank_low=float(rank_low),
            rank_high=float(rank_high),
            draw_propensity=float(weights @ samples.draw_tendencies[:, i]),
            probability_better_parent=(
                None
                if int(samples.parents[i]) < 0
                else float(weights[elo[:, i] > elo[:, int(samples.parents[i])]].sum())
            ),
        ))
    hyperparameters = {
        "lineage_sd_elo": float(weights @ samples.lineage_scales) * ELO_SCALE,
        "lineage_sd_elo_p95": float(_weighted_quantile(
            samples.lineage_scales, weights, (0.95,)
        )[0] * ELO_SCALE),
        "draw_intercept": float(weights @ samples.draw_intercepts),
        "draw_tendency_sd": float(weights @ samples.draw_scales),
    }
    return tuple(summaries), hyperparameters


def sequential_update(
    samples: PosteriorSamples,
    pairs: Sequence[PairCounts],
    *,
    base_diagnostics: dict[str, float | int | str],
) -> RatingFit:
    """Exactly reweight posterior draws by a new independent match batch."""

    log_weights = samples.log_weights.copy()
    for pair in pairs:
        gap = samples.skills[:, pair.i] - samples.skills[:, pair.j]
        draw_logit = samples.draw_intercepts + 0.5 * (
            samples.draw_tendencies[:, pair.i] + samples.draw_tendencies[:, pair.j]
        )
        logits = np.column_stack((0.5 * gap, draw_logit, -0.5 * gap))
        maximum = logits.max(axis=1)
        log_norm = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        log_weights += (
            pair.wins_i * logits[:, 0]
            + pair.draws * logits[:, 1]
            + pair.wins_j * logits[:, 2]
            - pair.games * log_norm
        )
    log_weights -= float(np.max(log_weights))
    updated = PosteriorSamples(
        samples.skills, samples.draw_tendencies, samples.draw_intercepts,
        samples.lineage_scales, samples.draw_scales, log_weights, samples.parents,
    )
    weights = _normalized_weights(log_weights)
    importance_ess = float(1.0 / np.sum(weights * weights))
    summaries, hyperparameters = _summarize_samples(updated)
    diagnostics = {
        **base_diagnostics,
        "method": "sequential_importance",
        "importance_ess": importance_ess,
        "importance_ess_fraction": importance_ess / len(weights),
        "side_effect": "fixed_zero",
    }
    return RatingFit(summaries, diagnostics, hyperparameters, updated)


def superiority_matrix(samples: PosteriorSamples) -> np.ndarray:
    """Joint posterior P(skill_i > skill_j) for every ordered agent pair."""

    weights = _normalized_weights(samples.log_weights)
    count = samples.skills.shape[1]
    matrix = np.full((count, count), 0.5, dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            probability = float(weights[samples.skills[:, i] > samples.skills[:, j]].sum())
            matrix[i, j] = probability
            matrix[j, i] = 1.0 - probability
    return matrix


def matchup_information_matrix(samples: PosteriorSamples) -> np.ndarray:
    """Expected information gain, in nats, from one additional W/D/L result."""

    weights = _normalized_weights(samples.log_weights)
    count = samples.skills.shape[1]
    information = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            gap = samples.skills[:, i] - samples.skills[:, j]
            draw_logit = samples.draw_intercepts + 0.5 * (
                samples.draw_tendencies[:, i] + samples.draw_tendencies[:, j]
            )
            logits = np.column_stack((0.5 * gap, draw_logit, -0.5 * gap))
            maximum = logits.max(axis=1)
            probabilities = np.exp(logits - maximum[:, None])
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            predictive = weights @ probabilities
            predictive_entropy = -float(
                np.sum(predictive * np.log(np.maximum(predictive, 1e-300)))
            )
            conditional_entropy = -float(
                weights @ np.sum(
                    probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1
                )
            )
            information[i, j] = information[j, i] = max(
                predictive_entropy - conditional_entropy, 0.0
            )
    return information


def fit_bayesian_ratings(
    num_agents: int,
    pairs: Sequence[PairCounts],
    parents: Sequence[int | None],
    *,
    config: RatingConfig | None = None,
) -> RatingFit:
    """Sample and summarize the fixed-skill arena posterior."""

    config = config or RatingConfig()
    model = DavidsonPosterior(
        num_agents, pairs, parents,
        root_skill_sd=config.root_skill_sd,
        lineage_scale=config.lineage_scale,
        draw_scale=config.draw_scale,
    )
    mode = _posterior_mode(model)
    metric = _curvature_metric(model, mode)
    seed_sequence = np.random.SeedSequence(config.seed)
    chain_seeds = seed_sequence.spawn(config.chains)
    chain_draws = []
    chain_stats = []
    for chain_seed in chain_seeds:
        rng = np.random.default_rng(chain_seed)
        start = mode + 0.25 * metric.covariance_cholesky @ rng.normal(size=model.dimension)
        draws, stats = _sample_chain(model, start, metric, config, rng)
        chain_draws.append(draws)
        chain_stats.append(stats)
    chains = np.asarray(chain_draws)
    rank_chains = _rank_normalize(chains)
    rhat = _rank_normalized_rhat(chains)
    bulk_ess = _effective_sample_size(rank_chains)
    tail_ess = np.full(model.dimension, math.inf)
    for quantile in (0.05, 0.95):
        threshold = np.quantile(chains, quantile, axis=(0, 1), keepdims=True)
        indicator = (chains <= threshold).astype(np.float64)
        tail_ess = np.minimum(tail_ess, _effective_sample_size(indicator))
    ess = np.minimum(bulk_ess, tail_ess)
    max_rhat = float(np.max(rhat))
    min_ess = float(np.min(ess))
    divergences = int(sum(int(stats["divergences"]) for stats in chain_stats))
    if config.require_convergence and (
        max_rhat > config.max_rhat or min_ess < config.min_ess or divergences > 0
    ):
        worst_rhat = int(np.argmax(rhat))
        worst_ess = int(np.argmin(ess))
        chain_detail = ", ".join(
            f"a={float(stats['acceptance']):.2f}/eps={float(stats['step_size']):.4g}"
            for stats in chain_stats
        )
        raise RatingConvergenceError(
            f"rating posterior did not converge: max R-hat={max_rhat:.3f}, "
            f"min ESS={min_ess:.0f}, divergences={divergences}; "
            f"worst dimensions rhat={worst_rhat}, ess={worst_ess}; chains {chain_detail}"
        )

    flat = chains.reshape(-1, model.dimension)
    skill_draws = np.empty((len(flat), num_agents), dtype=np.float64)
    draw_tendencies = np.empty_like(skill_draws)
    taus = np.empty(len(flat), dtype=np.float64)
    draw_sigmas = np.empty(len(flat), dtype=np.float64)
    deltas = np.empty(len(flat), dtype=np.float64)
    for sample, position in enumerate(flat):
        skill, draw_tendency, delta, tau, draw_sigma = model.unpack(position)
        skill_draws[sample] = skill - skill.mean()
        draw_tendencies[sample] = draw_tendency
        taus[sample] = tau
        draw_sigmas[sample] = draw_sigma
        deltas[sample] = delta
    samples = PosteriorSamples(
        skill_draws,
        draw_tendencies,
        deltas,
        taus,
        draw_sigmas,
        np.zeros(len(flat), dtype=np.float64),
        np.asarray([-1 if parent is None else parent for parent in parents], dtype=np.int32),
    )
    summaries, hyperparameters = _summarize_samples(samples)
    diagnostics: dict[str, float | int | str] = {
        "model": MODEL_VERSION,
        "chains": config.chains,
        "samples_per_chain": config.samples,
        "warmup_per_chain": config.warmup,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "min_bulk_ess": float(np.min(bulk_ess)),
        "min_tail_ess": float(np.min(tail_ess)),
        "divergences": divergences,
        "mean_acceptance": float(np.mean([stats["acceptance"] for stats in chain_stats])),
        "method": "hmc",
        "side_effect": "fixed_zero",
    }
    return RatingFit(summaries, diagnostics, hyperparameters, samples)
