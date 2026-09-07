"""Bounded full-pair outcome rollouts with public-posterior reserve overrides.

Only native transitions inspect opaque checkpoints. Every future preview is
overridden at its reveal boundary with one complete public-posterior reserve
hypothesis; the continuation actor never receives that hypothesis. There is no
critic, heuristic, or horizon-as-draw fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.search.native_pair import NativePairSearchState, capture_native_state
from drmc_rl.search.pill_belief import (
    PillReserveBelief, _matching_seed_indices, pill_id_to_raw_pair, reserve_table,
)


def reserve_hypotheses(belief: PillReserveBelief) -> tuple[tuple[bytes, float], ...]:
    """Enumerate distinct complete reserves, retaining their exact prior mass."""
    if belief.initial_board is None:
        raise ValueError("terminal quality requires initial public bottle conditioning")
    seeds = _matching_seed_indices(belief.observations, belief.level, belief.initial_board)
    reserves, counts = np.unique(reserve_table()[seeds], axis=0, return_counts=True)
    return tuple((row.tobytes(), float(count/len(seeds)))
                 for row, count in zip(reserves, counts, strict=True))


@dataclass(frozen=True, slots=True)
class RolloutTask:
    id: int
    state: NativePairSearchState
    root_side: int
    action: int
    reserve: bytes
    weight: float

    def __post_init__(self):
        if self.root_side not in (0, 1) or not self.state.privileged.need_action[self.root_side]:
            raise ValueError("rollout root must be an acting side")
        if self.action not in self.state.legal_actions_by_side[self.root_side]:
            raise ValueError("rollout root action must be legal")
        if len(self.reserve) != 128 or any(value > 8 for value in self.reserve):
            raise ValueError("rollout requires one complete valid reserve hypothesis")
        if not np.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise ValueError("rollout hypothesis weight must be in (0,1]")


def rollout_tasks(tasks, continuation, *, batch_size=32, max_events=2048, progress=None):
    """Return natural outcomes for all tasks; incomplete results retain None.

    Slots refill immediately after termination so short candidate failures do
    not leave most of a GPU batch idle while one long game finishes.
    """
    if batch_size < 1 or max_events < 1:
        raise ValueError("batch size and maximum events must be positive")
    tasks = iter(tasks)
    slots: list[dict[str, Any]] = []
    results = []
    exhausted = False

    def fill(runner):
        nonlocal exhausted
        try:
            task = next(tasks)
        except StopIteration:
            exhausted = True
            return None
        runner.restore(0, task.state.privileged.engine_checkpoint)
        if runner.snapshot(0) != task.state.privileged.engine_checkpoint:
            raise RuntimeError("native restore changed the rollout checkpoint")
        return {"runner": runner, "task": task, "state": task.state,
                "events": 0, "reveals": 0, "root_forced": False}

    runners = []
    try:
        for _ in range(batch_size):
            runner = DrMarioVsPoolRunner(num_pairs=1)
            runners.append(runner)
            slot = fill(runner)
            if slot is None:
                break
            slots.append(slot)
        while slots:
            requests, destinations = [], []
            for index, slot in enumerate(slots):
                state = slot["state"]
                action = np.full(2, -2, dtype=np.int32)
                slot["action"] = action
                # Reveal stops expose no new player action until the selected
                # posterior preview has been injected into the native engine.
                reveal = slot["runner"].search_reveal_info(0)
                slot["reveal"] = reveal
                if reveal is not None:
                    continue
                for side, need in enumerate(state.privileged.need_action):
                    if not need:
                        continue
                    if not slot["root_forced"] and side == slot["task"].root_side:
                        action[side] = slot["task"].action
                        slot["root_forced"] = True
                    elif state.legal_actions_by_side[side]:
                        requests.append((state, side))
                        destinations.append((index, side))
                    else:
                        action[side] = -1
            predictions = continuation.infer_batch(requests)
            for (index, side), (probability, _unused_value) in zip(destinations, predictions, strict=True):
                legal = slots[index]["state"].legal_actions_by_side[side]
                slots[index]["action"][side] = max(
                    legal, key=lambda action: probability.get(action, 1e-8))
            active = []
            for slot in slots:
                runner, task = slot["runner"], slot["task"]
                if slot["reveal"] is None:
                    runner.step_search(slot["action"])
                    if np.any(runner.buffers.invalid_action >= 0):
                        raise RuntimeError("terminal rollout rejected a complete-frontier action")
                else:
                    side, reserve_index = slot["reveal"]
                    runner.search_reveal(0, side, pill_id_to_raw_pair(task.reserve[reserve_index % 128]))
                    slot["reveals"] += 1
                slot["events"] += 1
                state = capture_native_state(
                    runner, level=task.state.level, speed_setting=task.state.speed_setting,
                    viruses_initial=task.state.viruses_initial)
                slot["state"] = state
                outcome = state.privileged.terminal_outcome[task.root_side]
                terminal = outcome in (1, 2, 3)
                if terminal or runner.buffers.truncated[0] or slot["events"] >= max_events:
                    if not slot["root_forced"]:
                        raise RuntimeError("rollout terminated before forcing its root action")
                    results.append({"id": task.id, "outcome": outcome if terminal else None,
                                    "events": slot["events"], "reveals": slot["reveals"],
                                    "weight": task.weight})
                    if progress is not None and len(results) % 128 == 0:
                        progress(len(results))
                    replacement = None if exhausted else fill(runner)
                    if replacement is not None:
                        active.append(replacement)
                else:
                    active.append(slot)
            slots = active
    finally:
        for runner in runners:
            runner.close()
    return results


def aggregate_outcomes(tasks, results):
    """Aggregate exact mass only when every posterior continuation completed."""
    expected = {task.id: task for task in tasks}
    actual = {result["id"]: result for result in results}
    if len(expected) != len(tasks) or len(actual) != len(results) or expected.keys() != actual.keys():
        raise ValueError("terminal outcome coverage does not match rollout tasks")
    mass = sum(task.weight for task in tasks)
    if not np.isclose(mass, 1., atol=1e-12, rtol=0):
        raise ValueError("candidate posterior mass must sum to one")
    if any(result["outcome"] is None for result in results):
        return None
    # Native outcome codes are win=1, loss=2, draw=3; public rows use W/D/L.
    positions = {1: 0, 3: 1, 2: 2}
    probability = np.zeros(3, dtype=np.float64)
    for task in tasks:
        result = actual[task.id]
        if result["weight"] != task.weight:
            raise ValueError("rollout result changed its hypothesis weight")
        probability[positions[result["outcome"]]] += task.weight
    return probability.tolist()
