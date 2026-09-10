"""Bounded adaptive joint-action allocation over the existing full-pair tree.

Unknown root matrix entries retain the full utility interval [-1,1]. Lower
and upper games produce security strategies; only a directly checked worst
best-response gap can stop expansion. This bounds the configured search game,
not the accuracy of its learned leaves or real match outcomes.
"""

from dataclasses import dataclass

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import SearchConfig
from drmc_rl.search.queued_event import QueuedJointEventSearch


@dataclass(frozen=True)
class AdaptiveSearchResult:
    actions: tuple[int, ...]
    opponent_actions: tuple[int, ...]
    policy_target: np.ndarray
    opponent_policy: np.ndarray
    utility_lower: np.ndarray
    utility_upper: np.ndarray
    evaluated: np.ndarray
    joint_wdl: tuple
    security_lower: float
    security_upper: float
    gap_upper: float
    certified: bool
    stop_reason: str
    nodes: int
    cache_hits: int
    chance_nodes: int
    chance_outcomes: int
    inference_batches: int
    matrix_solves: int
    matrix_solve_ms: float
    allocation_rounds: int
    allocated_joint_actions: int
    discarded_joint_actions: int
    matrix_failures: tuple[str, ...]

    def select_action(self, rng):
        if not self.certified:
            raise ValueError("adaptive search has no valid response-gap certificate")
        return int(rng.choice(self.actions, p=self.policy_target / self.policy_target.sum()))

    def to_dict(self):
        # Nulls preserve unexamined WDL. Midpoints are never training targets.
        return dict(
            schema="drmc-adaptive-joint-search-v1", actions=list(self.actions),
            opponent_actions=list(self.opponent_actions),
            policy_target=self.policy_target.tolist(), opponent_policy=self.opponent_policy.tolist(),
            utility_lower=self.utility_lower.tolist(), utility_upper=self.utility_upper.tolist(),
            evaluated=self.evaluated.tolist(),
            joint_wdl=[[(None if v is None else [v.win, v.draw, v.loss]) for v in row]
                       for row in self.joint_wdl],
            security_lower=self.security_lower, security_upper=self.security_upper,
            gap_upper=self.gap_upper, certified=self.certified, stop_reason=self.stop_reason,
            nodes=self.nodes, cache_hits=self.cache_hits, chance_nodes=self.chance_nodes,
            chance_outcomes=self.chance_outcomes, inference_batches=self.inference_batches,
            matrix_solves=self.matrix_solves, matrix_solve_ms=self.matrix_solve_ms,
            allocation_rounds=self.allocation_rounds,
            total_joint_actions=int(self.evaluated.size),
            evaluated_joint_actions=int(self.evaluated.sum()),
            allocated_joint_actions=self.allocated_joint_actions,
            discarded_joint_actions=self.discarded_joint_actions,
            candidate_truncation=0, matrix_failures=list(self.matrix_failures),
            calibrated=False, usable_for_quality_training=False,
            scope="Response bound for the configured finite-depth critic game only.")


class AdaptiveJointEventSearch(QueuedJointEventSearch):
    """Allocate root joint actions by their unresolved best-response influence.

    Interior decisions, mixed backups, forced events and correlated reveals
    use the existing queued PairSearchModel traversal without approximation
    beyond the explicitly configured depth and learned leaf evaluator.
    """

    def __init__(self, model, config=None, *, batch_size=32, allocation_batch=16,
                 max_joint_actions=262144, response_gap=.02, evaluation_tolerance=0.):
        config = config or SearchConfig(opponent_mode="mixed")
        if config.opponent_mode != "mixed" or config.matrix_solver != "linear_program":
            raise ValueError("adaptive matrix allocation requires mixed linear-program search")
        if (type(allocation_batch) is not int or allocation_batch < 1
                or type(max_joint_actions) is not int or max_joint_actions < 1):
            raise ValueError("adaptive joint-action budgets must be positive integers")
        if not np.isfinite(response_gap) or response_gap <= 0:
            raise ValueError("response_gap must be finite and positive")
        if not np.isfinite(evaluation_tolerance) or not 0 <= evaluation_tolerance <= 1:
            raise ValueError("evaluation_tolerance must be finite and in [0,1]")
        super().__init__(model, config, batch_size=batch_size)
        self.allocation_batch = allocation_batch
        self.max_joint_actions = max_joint_actions
        self.response_gap = float(response_gap)
        self.evaluation_tolerance = float(evaluation_tolerance)

    def search(self, state, *, root_side, root_actions=None):
        if root_side not in (0, 1):
            raise ValueError("root_side must be zero or one")
        if (self.model.boundary(state) != DecisionBoundary.BOTH
                or self.model.terminal_value(state, root_side) is not None):
            raise ValueError("adaptive matrix allocation requires a live simultaneous root")
        self._cache.clear()
        self._nodes = self._cache_hits = self._chance_nodes = self._chance_outcomes = 0
        self._budget_exhausted = False
        self.inference_batches = 0
        self._reset_matrices()
        actions = (self._ranked_actions(state, root_side, 512, maximize=True)
                   if root_actions is None else list(root_actions))
        if (not actions or len(set(actions)) != len(actions)
                or set(actions) != set(self.model.legal_actions(state, root_side))):
            raise ValueError("adaptive search requires the complete unique root inventory")
        opponent = self._ranked_actions(state, 1 - root_side, 512, maximize=False)
        if not opponent:
            raise ValueError("simultaneous root has no opponent action")
        shape = len(actions), len(opponent)
        lower, upper = np.full(shape, -1.), np.ones(shape)
        evaluated = np.zeros(shape, bool)
        values = [[None for _ in opponent] for _ in actions]
        tie_prior = np.outer(self._prior_weights(state, root_side, actions),
                             self._prior_weights(state, 1 - root_side, opponent))
        allocated = discarded = rounds = 0
        certified, reason = False, "joint_budget"
        p, q = np.full(shape[0], 1 / shape[0]), np.full(shape[1], 1 / shape[1])
        security_lower, security_upper = -1., 1.
        while True:
            p, _ = self._solve_payoffs(lower)
            _, q = self._solve_payoffs(upper)
            security_lower = float((p @ lower).min())
            security_upper = float((upper @ q).max())
            if not self._equilibrium_converged:
                reason = "solver_failure"
                break
            if security_upper - security_lower <= self.response_gap:
                certified, reason = True, "response_bound"
                break
            if evaluated.all():
                # Covers a numerical solver that passed its own tolerance but
                # not the separately requested adaptive response tolerance.
                reason = "response_tolerance"
                break
            remaining = self.max_joint_actions - allocated
            if remaining <= 0:
                break
            if self._nodes >= self.config.max_nodes:
                reason = "node_budget"
                break
            # Resolve uncertainty along the currently dangerous row/column.
            # Policy priors only break exact allocation ties; they cannot
            # narrow intervals, remove actions or certify a stopping decision.
            row = int(np.argmax(upper @ q))
            column = int(np.argmin(p @ lower))
            influence = np.zeros(shape)
            width = upper - lower
            influence[row, :] += q * width[row, :]
            influence[:, column] += p * width[:, column]
            unknown = np.flatnonzero(~evaluated)
            order = np.lexsort((unknown, -tie_prior.ravel()[unknown],
                                -influence.ravel()[unknown]))
            chosen = unknown[order[:min(self.allocation_batch, remaining, len(unknown))]]
            indices = [tuple(map(int, np.unravel_index(i, shape))) for i in chosen]
            batch = self._resolve([self._joint_child(
                state, root_side, actions[i], opponent[j], self.config.depth_events)
                for i, j in indices])
            allocated += len(indices)
            rounds += 1
            if self._budget_exhausted or not self._equilibrium_converged:
                # A shared traversal budget can expire partway through the
                # batch. Conservatively retain its entire old interval rather
                # than accepting a fabricated neutral fallback as a label.
                discarded += len(indices)
                reason = "node_budget" if self._budget_exhausted else "solver_failure"
                break
            for (i, j), value in zip(indices, batch, strict=True):
                values[i][j] = value
                lower[i, j] = max(-1., value.utility - self.evaluation_tolerance)
                upper[i, j] = min(1., value.utility + self.evaluation_tolerance)
                evaluated[i, j] = True
        return AdaptiveSearchResult(
            tuple(actions), tuple(opponent), p.copy(), q.copy(), lower, upper, evaluated,
            tuple(tuple(row) for row in values), security_lower, security_upper,
            max(0., security_upper - security_lower), certified, reason,
            self._nodes, self._cache_hits, self._chance_nodes, self._chance_outcomes,
            self.inference_batches, self._matrix_games, self._matrix_solve_ms,
            rounds, allocated, discarded, tuple(self._matrix_failures))
