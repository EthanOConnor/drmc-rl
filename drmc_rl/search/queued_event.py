"""Cooperative root-frontier batching for the existing exact event search.

Each root continuation yields before policy/value inference. The coordinator
prefetches those independent public states together, then resumes the same
backups. Native transitions remain serial and restore their exact source state.
"""

from __future__ import annotations

import numpy as np

from drmc_rl.game.pair_state import DecisionBoundary
from drmc_rl.search.joint_event import JointEventSearch, SearchResult, WDL


class QueuedJointEventSearch(JointEventSearch):
    def __init__(self, model, config=None, *, batch_size=64):
        super().__init__(model, config)
        if batch_size < 1:
            raise ValueError("frontier batch size must be positive")
        self.batch_size = int(batch_size)
        self.inference_batches = 0

    def search(self, state, *, root_side, root_actions=None):
        if root_side not in (0, 1):
            raise ValueError("root_side must be 0 or 1")
        self._cache.clear()
        self._nodes = 0
        self._cache_hits = 0
        self._budget_exhausted = False
        self._chance_nodes = 0
        self._chance_outcomes = 0
        self.inference_batches = 0
        boundary = self.model.boundary(state)
        expected = DecisionBoundary.P1 if root_side == 0 else DecisionBoundary.P2
        if self.model.terminal_value(state, root_side) is not None or boundary not in (
            expected,
            DecisionBoundary.BOTH,
        ):
            raise ValueError("root must be an active nonterminal decision")
        actions = (
            self._ranked_actions(state, root_side, self.config.own_beam, maximize=True)
            if root_actions is None
            else list(root_actions)
        )
        if (
            not actions
            or len(set(actions)) != len(actions)
            or not set(actions) <= set(self.model.legal_actions(state, root_side))
        ):
            raise ValueError("root actions must be nonempty, unique and legal")

        def root(action):
            if boundary == DecisionBoundary.BOTH:
                return (yield from self._given(state, root_side, action, self.config.depth_events))
            child = self.model.apply_actions(
                state, action if root_side == 0 else None, action if root_side == 1 else None
            )
            return (yield from self._visit(child, self.config.depth_events - 1, root_side))

        pending = [(i, root(action)) for i, action in enumerate(actions)]
        values = [None] * len(actions)
        try:
            while pending:
                waiting = []
                requests = []
                for i, generator in pending:
                    try:
                        request = next(generator)
                    except StopIteration as completed:
                        values[i] = completed.value
                    else:
                        requests.append(request)
                        waiting.append((i, generator))
                self._prepare(requests)
                pending = waiting
        finally:
            for _, generator in pending:
                generator.close()
        utilities = np.asarray([v.utility for v in values], np.float64)
        policy = self._policy_target(utilities)
        best = int(np.argmax(utilities))
        return SearchResult(
            tuple(actions),
            tuple(values),
            utilities.astype(np.float32),
            policy.astype(np.float32),
            int(actions[best]),
            WDL.mixture(policy, values),
            self._nodes,
            self._cache_hits,
            self.config.depth_events,
            self._budget_exhausted,
            self._chance_nodes,
            self._chance_outcomes,
        )

    def _prepare(self, requests):
        prepare = getattr(self.model, "prepare_requests", None)
        fallback = getattr(self.model, "prepare_batch", None)
        if prepare is None and fallback is None:
            return
        # The model's complete key includes public history and execution costs;
        # it may also contain the reserve posterior for offline chance search.
        unique = list(
            {(self.model.key(state), side): (state, side) for state, side in requests}.values()
        )
        for start in range(0, len(unique), self.batch_size):
            chunk = unique[start : start + self.batch_size]
            if prepare is not None:
                prepare(chunk)
            else:
                fallback([state for state, _ in chunk])
            self.inference_batches += 1

    def _visit(self, state, depth, root_side):
        self._nodes += 1
        terminal = self.model.terminal_value(state, root_side)
        if self._nodes > self.config.max_nodes:
            self._budget_exhausted = True
        if terminal is not None:
            return terminal
        if self._nodes > self.config.max_nodes:
            return WDL(0.5, 0.0, 0.5)
        boundary = self.model.boundary(state)
        if boundary == DecisionBoundary.TERMINAL:
            raise RuntimeError("terminal boundary has no authoritative outcome")
        key = (self.model.key(state), int(depth), int(root_side))
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        if boundary == DecisionBoundary.ADVANCE:
            outcomes = tuple(self.model.chance_outcomes(state))
            if outcomes:
                self._chance_nodes += 1
                self._chance_outcomes += len(outcomes)
                outcomes = sorted(outcomes, key=lambda o: o.probability, reverse=True)[
                    : self.config.chance_beam
                ]
                children = []
                for item in outcomes:
                    children.append(
                        (yield from self._visit(item.state, max(0, depth - 1), root_side))
                    )
                value = WDL.mixture([o.probability for o in outcomes], children)
            else:
                child = self.model.advance(state)
                if self.model.key(child) == key[0]:
                    raise RuntimeError("deterministic advance made no progress")
                value = yield from self._visit(child, max(0, depth - 1), root_side)
        else:
            acting = (
                root_side
                if boundary == DecisionBoundary.BOTH
                else 0
                if boundary == DecisionBoundary.P1
                else 1
            )
            yield state, acting
            if depth <= 0:
                value = self.model.evaluate(state, root_side)
            elif boundary == DecisionBoundary.BOTH:
                actions = self._ranked_actions(
                    state, root_side, self.config.own_beam, maximize=True
                )
                children = []
                for action in actions:
                    children.append((yield from self._given(state, root_side, action, depth)))
                value = (
                    max(children, key=lambda v: v.utility)
                    if children
                    else self.model.evaluate(state, root_side)
                )
            else:
                acting = 0 if boundary == DecisionBoundary.P1 else 1
                own = acting == root_side
                actions = self._ranked_actions(
                    state,
                    acting,
                    self.config.own_beam if own else self.config.opponent_beam,
                    maximize=own,
                )
                children = []
                for action in actions:
                    child = self.model.apply_actions(
                        state, action if acting == 0 else None, action if acting == 1 else None
                    )
                    children.append((yield from self._visit(child, depth - 1, root_side)))
                value = (
                    self.model.evaluate(state, root_side)
                    if not children
                    else max(children, key=lambda v: v.utility)
                    if own
                    else self._opponent_backup(state, acting, actions, children)
                )
        self._cache[key] = value
        return value

    def _given(self, state, root_side, action, depth):
        yield state, 1 - root_side
        opponent = 1 - root_side
        actions = self._ranked_actions(state, opponent, self.config.opponent_beam, maximize=False)
        if not actions:
            child = self.model.apply_actions(
                state, action if root_side == 0 else None, action if root_side == 1 else None
            )
            return (yield from self._visit(child, depth - 1, root_side))
        values = []
        for other in actions:
            child = self.model.apply_actions(
                state, action if root_side == 0 else other, other if root_side == 0 else action
            )
            values.append((yield from self._visit(child, depth - 1, root_side)))
        return self._opponent_backup(state, opponent, actions, values)
