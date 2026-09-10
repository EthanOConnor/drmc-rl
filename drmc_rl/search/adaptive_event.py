"""Bounded adaptive joint-action allocation over the existing full-pair tree.

Unknown root matrix entries retain the full utility interval [-1,1]. Lower
and upper games produce security strategies; only a directly checked worst
best-response gap can stop expansion. This bounds the configured search game,
not the accuracy of its learned leaves or real match outcomes.
"""

from dataclasses import dataclass

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import SearchConfig, WDL
from drmc_rl.search.queued_event import QueuedJointEventSearch


@dataclass(frozen=True)
class _UtilityInterval:
    lower: float = -1.
    upper: float = 1.
    wdl: WDL | None = None

    def __post_init__(self):
        if (not np.isfinite([self.lower, self.upper]).all()
                or not -1. <= self.lower <= self.upper <= 1.):
            raise ValueError("invalid finite-game utility interval")


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
    nested: bool = False
    total_allocated_joint_actions: int = 0
    nested_certificates: tuple = ()
    node_budget_exhausted: bool = False
    tactical_extensions: int = 0
    tactical_reasons: tuple = ()

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
            nested=self.nested, total_allocated_joint_actions=self.total_allocated_joint_actions,
            nested_certificates=list(self.nested_certificates),
            node_budget_exhausted=self.node_budget_exhausted,
            tactical_extensions=self.tactical_extensions, tactical_reasons=dict(self.tactical_reasons),
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
                 max_joint_actions=262144, response_gap=.02, evaluation_tolerance=0., nested=False):
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
        if type(nested) is not bool:
            raise ValueError("nested allocation must be an explicit boolean")
        self.nested = nested

    def search(self, state, *, root_side, root_actions=None):
        if root_side not in (0, 1):
            raise ValueError("root_side must be zero or one")
        if (self.model.boundary(state) != DecisionBoundary.BOTH
                or self.model.terminal_value(state, root_side) is not None):
            raise ValueError("adaptive matrix allocation requires a live simultaneous root")
        if self.nested:
            return self._search_nested(state, root_side, root_actions)
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
            rounds, allocated, discarded, tuple(self._matrix_failures),
            total_allocated_joint_actions=allocated, node_budget_exhausted=self._budget_exhausted,
            tactical_extensions=self._tactical_extensions,
            tactical_reasons=tuple(sorted(self._tactical_reasons.items())))

    def _search_nested(self, state, root_side, root_actions):
        self._cache.clear()
        self._nodes = self._cache_hits = self._chance_nodes = self._chance_outcomes = 0
        self._budget_exhausted = False
        self.inference_batches = self._total_joint_actions = 0
        self._nested_certificates = []
        self._reset_matrices()
        self._prepare([(state, root_side), (state, 1-root_side)])
        actions = (self._ranked_actions(state, root_side, 512, maximize=True)
                   if root_actions is None else list(root_actions))
        if (not actions or len(set(actions)) != len(actions)
                or set(actions) != set(self.model.legal_actions(state, root_side))):
            raise ValueError("adaptive search requires the complete unique root inventory")
        result, = self._resolve([self._interval_matrix(
            state, root_side, actions, self.config.depth_events, self.response_gap, root=True)])
        return AdaptiveSearchResult(
            tuple(actions), tuple(result['opponent']), result['p'], result['q'],
            result['lower'], result['upper'], result['evaluated'], result['wdl'],
            result['security_lower'], result['security_upper'], result['gap'],
            result['certified'], result['reason'], self._nodes, self._cache_hits,
            self._chance_nodes, self._chance_outcomes, self.inference_batches,
            self._matrix_games, self._matrix_solve_ms, result['rounds'],
            result['allocated'], result['discarded'], tuple(self._matrix_failures),
            nested=True, total_allocated_joint_actions=self._total_joint_actions,
            nested_certificates=tuple(self._nested_certificates),
            node_budget_exhausted=self._budget_exhausted,
            tactical_extensions=self._tactical_extensions,
            tactical_reasons=tuple(sorted(self._tactical_reasons.items())))

    def _interval_joint_child(self, state, root_side, own, other, depth, gap):
        if self._total_joint_actions >= self.max_joint_actions:
            return _UtilityInterval(), False
        if self._nodes >= self.config.max_nodes:
            self._budget_exhausted = True
            return _UtilityInterval(), False
        self._total_joint_actions += 1
        child = self.model.apply_actions(state, own if root_side == 0 else other,
                                         other if root_side == 0 else own)
        return (yield from self._interval_visit(child, depth-1, root_side, gap)), True

    def _interval_matrix(self, state, root_side, actions, depth, gap, *, root=False):
        yield state, 1-root_side
        opponent = self._ranked_actions(state, 1-root_side, 512, maximize=False)
        if not actions or not opponent:
            raise ValueError("simultaneous boundary requires both complete inventories")
        shape = len(actions), len(opponent)
        lower, upper = np.full(shape, -1.), np.ones(shape)
        attempted, evaluated = np.zeros(shape, bool), np.zeros(shape, bool)
        values = [[None for _ in opponent] for _ in actions]
        tie_prior = np.outer(self._prior_weights(state, root_side, actions),
                             self._prior_weights(state, 1-root_side, opponent))
        rounds = allocated = discarded = 0
        reason, certified = 'joint_budget', False
        while True:
            p, _ = self._solve_payoffs(lower)
            _, q = self._solve_payoffs(upper)
            # Outward arithmetic guard is separate from learned-leaf error.
            security_lower = max(-1., float((p @ lower).min())-1e-12)
            security_upper = min(1., float((upper @ q).max())+1e-12)
            bound = max(0., security_upper-security_lower)
            if not self._equilibrium_converged:
                reason = 'solver_failure'
                break
            if bound <= gap:
                certified, reason = True, 'response_bound'
                break
            if self._nodes >= self.config.max_nodes:
                self._budget_exhausted = True
                reason = 'node_budget'
                break
            remaining = self.max_joint_actions-self._total_joint_actions
            if remaining <= 0:
                break
            if attempted.all():
                # Descendant intervals, including numerical evaluator error,
                # remain intervals. Never substitute their midpoint or a WDL.
                reason = 'interval_tolerance'
                break
            row, column = int(np.argmax(upper @ q)), int(np.argmin(p @ lower))
            width = upper-lower
            influence = np.zeros(shape)
            influence[row, :] += q*width[row, :]
            influence[:, column] += p*width[:, column]
            unknown = np.flatnonzero(~attempted)
            order = np.lexsort((unknown, -tie_prior.ravel()[unknown], -influence.ravel()[unknown]))
            chosen = unknown[order[:min(self.allocation_batch, remaining, len(unknown))]]
            indices = [tuple(map(int, np.unravel_index(i, shape))) for i in chosen]
            # A stricter child gap leaves room for unresolved parent actions.
            # This is an allocation target, not an assumed error bound.
            children = yield from self._cooperate([self._interval_joint_child(
                state, root_side, actions[i], opponent[j], depth, gap/4.) for i, j in indices])
            rounds += 1
            allocated += len(indices)
            for (i, j), (value, applied) in zip(indices, children, strict=True):
                attempted[i, j] = True
                evaluated[i, j] = applied
                discarded += int(not applied)
                lower[i, j], upper[i, j] = value.lower, value.upper
                values[i][j] = value.wdl
        self._nested_certificates.append(dict(depth=depth, root=root,
            actions=len(actions), opponent_actions=len(opponent), evaluated=int(evaluated.sum()),
            allocated=allocated, requested_gap=gap, security_lower=security_lower,
            security_upper=security_upper, gap_upper=bound, certified=certified, stop_reason=reason))
        return dict(opponent=opponent, lower=lower, upper=upper, evaluated=evaluated,
            wdl=tuple(tuple(row) for row in values), p=p.copy(), q=q.copy(),
            security_lower=security_lower, security_upper=security_upper, gap=bound,
            certified=certified, reason=reason, rounds=rounds, allocated=allocated, discarded=discarded)

    def _interval_visit(self, state, depth, root_side, gap):
        if self._nodes >= self.config.max_nodes:
            self._budget_exhausted = True
            return _UtilityInterval()
        self._nodes += 1
        terminal = self.model.terminal_value(state, root_side)
        if terminal is not None:
            return _UtilityInterval(terminal.utility, terminal.utility, terminal)
        boundary = self.model.boundary(state)
        if boundary == DecisionBoundary.TERMINAL:
            raise RuntimeError("terminal boundary has no authoritative outcome")
        key = (self.model.key(state), int(depth), int(root_side), float(gap))
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        if boundary == DecisionBoundary.ADVANCE:
            outcomes = tuple(self.model.chance_outcomes(state))
            if outcomes:
                outcomes = self._ordered_chance(outcomes)
                self._chance_nodes += 1
                self._chance_outcomes += len(outcomes)
                children = yield from self._cooperate([self._interval_visit(
                    item.state, self._forced_depth(depth), root_side, gap) for item in outcomes])
                weights = np.asarray([o.probability for o in outcomes], float)
                weights /= weights.sum()
                lo = float(weights @ [v.lower for v in children])
                hi = float(weights @ [v.upper for v in children])
                wdl = (WDL.mixture(weights, [v.wdl for v in children])
                       if all(v.wdl is not None for v in children) else None)
                value = _UtilityInterval(max(-1., lo-1e-12), min(1., hi+1e-12), wdl)
            else:
                child = self.model.advance(state)
                if self.model.key(child) == key[0]:
                    raise RuntimeError("deterministic advance made no progress")
                value = yield from self._interval_visit(child, self._forced_depth(depth), root_side, gap)
        else:
            acting = root_side if boundary == DecisionBoundary.BOTH else (
                0 if boundary == DecisionBoundary.P1 else 1)
            yield state, acting
            if not self._expand_decision(state, depth):
                wdl = self.model.evaluate(state, root_side)
                value = _UtilityInterval(max(-1., wdl.utility-self.evaluation_tolerance),
                    min(1., wdl.utility+self.evaluation_tolerance), wdl)
            elif boundary == DecisionBoundary.BOTH:
                actions = self._ranked_actions(state, root_side, 512, maximize=True)
                result = yield from self._interval_matrix(state, root_side, actions, depth, gap)
                value = _UtilityInterval(max(-1., result['security_lower']),
                    min(1., result['security_upper']))
            else:
                actions = self._ranked_actions(state, acting, 512, maximize=acting == root_side)
                if not actions:
                    raise ValueError("active decision has no legal actions")

                def child(action):
                    if self._nodes >= self.config.max_nodes:
                        self._budget_exhausted = True
                        return _UtilityInterval()
                    next_state = self.model.apply_actions(state, action if acting == 0 else None,
                                                          action if acting == 1 else None)
                    return (yield from self._interval_visit(next_state, depth-1, root_side, gap))

                children = yield from self._cooperate([child(a) for a in actions])
                backup = max if acting == root_side else min
                value = _UtilityInterval(backup(v.lower for v in children), backup(v.upper for v in children))
        self._cache[key] = value
        return value
