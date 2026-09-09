"""Exact conditional next-pill opportunities, before policy fitting.

Every current motor-feasible placement is retained. Each branch resolves our
known pill on our visible bottle, then plans the visible preview with the same
motor limits, the next gravity period and carried horizontal DAS. Incoming
garbage is explicitly excluded. Both possible spawn parities are separate
conditions, never two independent samples or an assumed probability mixture.

Finite costs describe validated planner witnesses, not proofs of globally
minimum execution time. Natural terminal success is distinct from having no
reachable next placement. These are geometry/effect targets, not match values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drmc_rl.execution.pace import Pace
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
from drmc_rl.human.anticipation import CANON_TO_RAW, execution_for_action
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.human.search import semantic_planes_to_nes_board

OPPORTUNITY_SCHEMA = "drmc-conditional-next-pill-opportunity-v1"
OPPORTUNITY_CONDITION = "no-intervening-incoming-garbage-neutral-input-between-pills"
UNREACHABLE = np.uint16(65535)


def action_cells(action):
    orientation, cell = divmod(int(action), 128)
    row, column = divmod(cell, 8)
    if not 0 <= orientation < 4:
        raise ValueError("invalid placement action")
    dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[orientation]
    if not (0 <= row + dy < 16 and 0 <= column + dx < 8):
        raise ValueError("placement half is outside the bottle")
    return cell, cell + 8 * dy + dx


@dataclass(frozen=True)
class MotorOpportunity:
    actions: np.ndarray
    root_costs: np.ndarray
    after_fields: np.ndarray
    root_terminal: np.ndarray
    root_viruses_cleared: np.ndarray
    root_nonviruses_cleared: np.ndarray
    root_clear_events: np.ndarray
    next_costs: np.ndarray
    next_clear_events: np.ndarray
    next_viruses_cleared: np.ndarray
    next_terminal: np.ndarray
    reachable_cells: np.ndarray
    clearable_cells: np.ndarray
    next_speed_ups: int
    delay: int

    def arrays(self):
        """Packed root order; axes are [root action, spawn parity, ...]."""
        return {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}

    def summary(self):
        valid = self.root_terminal == 0
        counts = (self.next_costs != UNREACHABLE).sum(-1)
        clear = (self.next_clear_events > 0).any(-1)
        return dict(
            schema=OPPORTUNITY_SCHEMA, condition=OPPORTUNITY_CONDITION,
            candidates=len(self.actions), next_candidates=int(counts.sum()),
            terminal_clear=int((self.root_terminal == 1).sum()),
            terminal_topout=int((self.root_terminal == 2).sum()),
            no_next_choice_both_parities=int((valid & (counts == 0).all(-1)).sum()),
            clear_available_both_parities=int((valid & clear.all(-1)).sum()),
            parity_changes_frontier=int((valid & np.any(
                (self.next_costs[:, 0] != UNREACHABLE) !=
                (self.next_costs[:, 1] != UNREACHABLE), axis=-1)).sum()),
            next_speed_ups=self.next_speed_ups, next_delay_frames=self.delay,
        )


class MotorOpportunityLabeler:
    def __init__(self, planner, *, lib_path=None, batch_size=128):
        self.planner = planner
        self.simulator = NativeAfterstateSimulator(num_envs=batch_size, lib_path=lib_path)

    def close(self):
        self.simulator.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _effects(self, fields, pills, previews, actions, costs, speed, speed_ups):
        counts = np.asarray([len(row) for row in actions], np.int32)
        width = int(counts.max())
        aa = np.zeros((len(actions), width), np.int32)
        cc = np.zeros_like(aa, dtype=np.uint16)
        for i, n in enumerate(counts):
            aa[i, :n], cc[i, :n] = actions[i], costs[i]
        result = self.simulator.simulate_packed(
            fields=np.asarray(fields), pills=np.asarray(pills), previews=np.asarray(previews),
            candidate_actions=aa, candidate_costs=cc, candidate_count=counts,
            speed=np.full(len(actions), speed), speed_ups=np.full(len(actions), speed_ups),
        )
        if result.invalid.any():
            raise RuntimeError("a complete motor-feasible frontier failed native afterstate execution")
        if not np.isin(result.terminal_reason, (0, 1, 2)).all():
            raise RuntimeError("conditional afterstate returned an unsupported terminal reason")
        return result

    def label(self, state, pace: Pace, *, compute_frames=4, validate_next_scripts=False):
        if type(compute_frames) is not int or compute_frames < 0:
            raise ValueError("computation charge must be a nonnegative integer")
        if "pill_counter_total" not in state:
            raise ValueError("next gravity requires the observed packed-BCD pill counter")
        counter = int(state["pill_counter_total"])
        if not 0 <= counter <= 65535 or any((counter >> shift) & 15 > 9 for shift in (0, 4, 8, 12)):
            raise ValueError("invalid packed-BCD pill counter")
        # Same-color rotations can carry distinct controller state. Do not use
        # the frozen actor's historical two-orientation deduplication here.
        root = {**state, "public_context_schema": PUBLIC_CONTEXT_SCHEMA,
                "opponent_board_planes": np.zeros((8, 16, 8), np.uint8)}
        delay = max(compute_frames, pace.reaction_frames)
        candidate = plan_candidates(self.planner, root, delay, pace)
        actions = np.flatnonzero(candidate[-1] != UNREACHABLE).astype(np.int16)
        root_costs = candidate[-1][actions].copy()
        moves = [execution_for_action(candidate, int(a), pace, delay=delay) for a in actions]
        field = semantic_planes_to_nes_board(state["board_planes"])
        raw = np.asarray(CANON_TO_RAW, np.uint8)
        pill = raw[np.asarray(state["pill"], dtype=np.int64)]
        preview = raw[np.asarray(state["preview"], dtype=np.int64)]
        effects = self._effects([field], [pill], [preview], [actions], [root_costs],
                                state["speed"], state["speed_ups"])
        n = len(actions)
        costs = np.full((n, 2, 512), UNREACHABLE, np.uint16)
        clears = np.zeros((n, 2, 512), np.uint16)
        viruses = np.zeros((n, 2, 512), np.uint16)
        terminal = np.zeros((n, 2, 512), np.uint8)
        cells = np.full((n, 2, 128), UNREACHABLE, np.uint16)
        clear_cells = np.full_like(cells, UNREACHABLE)
        ups = min(49, int(state["speed_ups"]) + int((counter & 15) == 9))
        branches = []
        for i, move in enumerate(moves):
            if effects.terminal_reason[i]:
                continue
            for parity in (0, 1):
                predicted = {
                    "board_planes": board_bytes_to_semantic_planes(effects.fields[i]),
                    "opponent_board_planes": np.zeros((8, 16, 8), np.uint8),
                    "pill": state["preview"],
                    # The new preview cannot affect this pill's geometry or
                    # deterministic clears. It is never scored by a policy.
                    "preview": [0, 0], "speed": state["speed"], "speed_ups": ups,
                    "public_context_schema": PUBLIC_CONTEXT_SCHEMA,
                    "falling": dict(x=3, y=0, rotation=0, speed_counter=0,
                                    horizontal_velocity=move["lock_state"]["horizontal_velocity"],
                                    hold_dir=0, rotation_hold=0, frame_parity=parity),
                }
                try:
                    next_candidate = plan_candidates(self.planner, predicted, delay, pace)
                except NoReachablePlacement:
                    continue
                aa = np.flatnonzero(next_candidate[-1] != UNREACHABLE)
                cc = next_candidate[-1][aa]
                if int(cc.max()) + delay >= int(UNREACHABLE):
                    raise OverflowError("opportunity witness exceeds the uint16 cost contract")
                costs[i, parity, aa] = cc + delay
                if validate_next_scripts:
                    for a in aa:
                        execution_for_action(next_candidate, int(a), pace, delay=delay)
                for action, cost in zip(aa, cc, strict=True):
                    indices = list(action_cells(action))
                    cells[i, parity, indices] = np.minimum(cells[i, parity, indices], int(cost) + delay)
                branches.append((i, parity, aa, cc))
        if branches:
            future = self._effects(
                [effects.fields[i] for i, _, _, _ in branches],
                [preview] * len(branches), [raw[[0, 0]]] * len(branches),
                [a for _, _, a, _ in branches], [c for _, _, _, c in branches],
                state["speed"], ups,
            )
            offset = 0
            for i, parity, aa, cc in branches:
                count = len(aa)
                slots = slice(offset, offset + count)
                clears[i, parity, aa] = future.clear_events[slots]
                viruses[i, parity, aa] = future.viruses_cleared[slots]
                terminal[i, parity, aa] = future.terminal_reason[slots]
                for action, cost, events in zip(aa, cc, future.clear_events[slots], strict=True):
                    if events:
                        indices = list(action_cells(action))
                        clear_cells[i, parity, indices] = np.minimum(
                            clear_cells[i, parity, indices], int(cost) + delay)
                offset += count
        return MotorOpportunity(
            actions, root_costs, effects.fields, effects.terminal_reason,
            effects.viruses_cleared, effects.nonviruses_cleared, effects.clear_events,
            costs, clears, viruses, terminal, cells, clear_cells, ups, delay,
        )


def state_from_controller_replay(replay, index):
    """Recover observed geometry without inventing missing old replay fields."""
    import json
    from drmc_rl.models.policy.controller_core import CONTROLLER_GEOMETRY_FIELDS

    metadata = json.loads(str(replay["metadata"]))
    if (metadata["schema"] != "drmc-public-controller-replay-v2" or
            tuple(metadata["controller_geometry_fields"]) != CONTROLLER_GEOMETRY_FIELDS):
        raise ValueError("motor opportunity labels require exact v2 controller replay geometry")
    geometry = dict(zip(CONTROLLER_GEOMETRY_FIELDS, map(int, replay["controller_geometry"][index]), strict=True))
    return {
        "board_planes": replay["observation"][index, :8],
        "opponent_board_planes": replay["observation"][index, 8:16],
        "pill": replay["pill"][index].tolist(), "preview": replay["preview"][index].tolist(),
        "falling": {key: geometry[key] for key in CONTROLLER_GEOMETRY_FIELDS[5:]},
        **{key: geometry[key] for key in CONTROLLER_GEOMETRY_FIELDS[:5]},
        "public_context_schema": PUBLIC_CONTEXT_SCHEMA,
    }
