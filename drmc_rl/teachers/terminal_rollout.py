"""Bounded full-pair outcome rollouts with public-posterior reserve overrides.

Only native transitions inspect opaque checkpoints. Every future preview is
overridden at its reveal boundary with one complete public-posterior reserve
hypothesis; the continuation actor never receives that hypothesis. There is no
critic, heuristic, or horizon-as-draw fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Any
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
import time

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
    continuation_id: str = ""
    opponent_id: str = ""

    def __post_init__(self):
        if self.root_side not in (0, 1) or not self.state.privileged.need_action[self.root_side]:
            raise ValueError("rollout root must be an acting side")
        if self.action not in self.state.legal_actions_by_side[self.root_side]:
            raise ValueError("rollout root action must be legal")
        if len(self.reserve) != 128 or any(value > 8 for value in self.reserve):
            raise ValueError("rollout requires one complete valid reserve hypothesis")
        if not np.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise ValueError("rollout hypothesis weight must be in (0,1]")


@lru_cache(maxsize=None)
def _native_executor(workers):
    # Reuse threads across roots, including the native thread-local reachability
    # workspace. Neural inference remains on the calling thread and batched.
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="terminal-native")


def _advance_native(slot):
    runner, task = slot["runner"], slot["task"]
    if slot["reveal"] is None:
        runner.step_search(slot["action"])
        if np.any(runner.buffers.invalid_action >= 0):
            raise RuntimeError("terminal rollout rejected a complete-frontier action")
    else:
        side, reserve_index = slot["reveal"]
        runner.search_reveal(0, side, pill_id_to_raw_pair(task.reserve[reserve_index % 128]))
        slot["reveals"] += 1


def rollout_tasks(
    tasks, continuation, *, batch_size=32, max_events=2048, progress=None, native_workers=1,
    on_result=None, metrics=None,
):
    """Return natural outcomes for all tasks; incomplete results retain None.

    Slots refill immediately after termination so short candidate failures do
    not leave most of a GPU batch idle while one long game finishes.
    """
    if batch_size < 1 or max_events < 1 or native_workers < 1:
        raise ValueError("batch size, native workers and maximum events must be positive")
    executor = _native_executor(min(native_workers, batch_size, 32)) if native_workers > 1 else None
    tasks = iter(tasks)
    slots: list[dict[str, Any]] = []
    results = []
    measured = Counter()
    inference_rows = Counter()
    started = time.perf_counter()
    exhausted = False
    last_progress = time.monotonic()

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
            measured['scheduler_iterations'] += 1
            measured['live_slot_iterations'] += len(slots)
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
            inference_started = time.perf_counter()
            if isinstance(continuation, Mapping):
                groups = {}
                for i, (index, side) in enumerate(destinations):
                    task = slots[index]["task"]
                    member = task.continuation_id if side == task.root_side else task.opponent_id
                    groups.setdefault(member, []).append(i)
                predictions = [None] * len(requests)
                for member, indices in groups.items():
                    answers = continuation[member].infer_batch([requests[i] for i in indices])
                    inference_rows[len(indices)] += 1
                    for i, answer in zip(indices, answers, strict=True):
                        predictions[i] = answer
            else:
                predictions = continuation.infer_batch(requests)
                if requests:
                    inference_rows[len(requests)] += 1
            measured['inference_seconds'] += time.perf_counter() - inference_started
            measured['policy_decisions'] += len(requests)
            for (index, side), (probability, _unused_value) in zip(destinations, predictions, strict=True):
                legal = slots[index]["state"].legal_actions_by_side[side]
                slots[index]["action"][side] = max(
                    legal, key=lambda action: probability.get(action, 1e-8))
            active = []
            # Distinct runners have independent physics and native workspaces.
            # Preserve request and completion order while releasing the GIL
            # inside each ctypes native call. The serial path is the reference.
            native_started = time.perf_counter()
            if executor is None:
                for slot in slots:
                    _advance_native(slot)
            else:
                pending = [executor.submit(_advance_native, slot) for slot in slots]
                try:
                    for future in pending:
                        future.result()
                finally:
                    # A failed slot must not close other native runners while
                    # their worker calls are still using those handles.
                    wait(pending)
            measured['native_seconds'] += time.perf_counter() - native_started
            for slot in slots:
                runner, task = slot["runner"], slot["task"]
                slot["events"] += 1
                state = capture_native_state(
                    runner,
                    level=task.state.level,
                    speed_setting=task.state.speed_setting,
                    viruses_initial=task.state.viruses_initial,
                    previous=slot["state"],
                )
                slot["state"] = state
                outcome = state.privileged.terminal_outcome[task.root_side]
                terminal = outcome in (1, 2, 3)
                if terminal or runner.buffers.truncated[0] or slot["events"] >= max_events:
                    if not slot["root_forced"]:
                        raise RuntimeError("rollout terminated before forcing its root action")
                    results.append(
                        {
                            "id": task.id,
                            "outcome": outcome if terminal else None,
                            "events": slot["events"],
                            "reveals": slot["reveals"],
                            "weight": task.weight,
                            "continuation_id": task.continuation_id,
                            "opponent_id": task.opponent_id,
                        }
                    )
                    if on_result is not None:
                        # Synchronous delivery lets a multi-root teacher commit
                        # one complete root without draining unrelated slots.
                        on_result(dict(results[-1]))
                    replacement = None if exhausted else fill(runner)
                    if replacement is not None:
                        active.append(replacement)
                else:
                    active.append(slot)
            slots = active
            if progress is not None and (not slots or time.monotonic() - last_progress >= 5.0):
                progress(len(results))
                last_progress = time.monotonic()
    finally:
        for runner in runners:
            runner.close()
        if metrics is not None:
            metrics.update(measured)
            metrics.update(wall_seconds=time.perf_counter()-started,
                           completed_rollouts=len(results),
                           inference_batch_rows=dict(sorted(inference_rows.items())))
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
