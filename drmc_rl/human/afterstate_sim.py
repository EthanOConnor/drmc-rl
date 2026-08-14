"""Batched exact one-placement simulation for corpus and search training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drmc_rl.envs.backends.drmario_pool import (
    MACRO_ACTIONS,
    DrMarioPoolRunner,
    build_reset_spec,
)


@dataclass(slots=True)
class AfterstateBatch:
    fields: np.ndarray
    terminal_reason: np.ndarray
    invalid: np.ndarray
    tau_frames: np.ndarray
    viruses_remaining: np.ndarray
    viruses_cleared: np.ndarray
    nonviruses_cleared: np.ndarray
    clear_events: np.ndarray


def encode_sparse_deltas(
    root_fields: np.ndarray, candidate_count: np.ndarray, afterstate_fields: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode candidate fields as exact cell changes from their root bottle."""

    roots = np.repeat(
        np.asarray(root_fields, dtype=np.uint8).reshape(-1, 128),
        np.asarray(candidate_count, dtype=np.int64),
        axis=0,
    )
    afterstates = np.asarray(afterstate_fields, dtype=np.uint8).reshape(-1, 128)
    if roots.shape != afterstates.shape:
        raise ValueError(f"root/afterstate shape mismatch: {roots.shape} != {afterstates.shape}")
    changed = afterstates != roots
    counts = changed.sum(axis=1, dtype=np.uint32)
    offsets = np.empty(len(afterstates) + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    flat_changed = np.flatnonzero(changed.reshape(-1))
    cells = (flat_changed % 128).astype(np.uint8)
    values = afterstates.reshape(-1)[flat_changed]
    return offsets, cells, values


def decode_sparse_deltas(
    root_field: np.ndarray,
    candidate_start: int,
    candidate_stop: int,
    delta_offsets: np.ndarray,
    delta_cells: np.ndarray,
    delta_values: np.ndarray,
) -> np.ndarray:
    """Reconstruct a contiguous candidate range from its exact sparse deltas."""

    start, stop = int(candidate_start), int(candidate_stop)
    count = stop - start
    fields = np.broadcast_to(np.asarray(root_field, dtype=np.uint8), (count, 128)).copy()
    candidate_index = np.arange(start, stop, dtype=np.int64)
    change_counts = (
        np.asarray(delta_offsets)[candidate_index + 1] - np.asarray(delta_offsets)[candidate_index]
    ).astype(np.int64)
    delta_start, delta_stop = int(delta_offsets[start]), int(delta_offsets[stop])
    if delta_stop > delta_start:
        slots = np.repeat(np.arange(count, dtype=np.int64), change_counts)
        cells = np.asarray(delta_cells[delta_start:delta_stop], dtype=np.uint8)
        fields[slots, cells] = delta_values[delta_start:delta_stop]
    return fields


class NativeAfterstateSimulator:
    """Fan legal candidates through the native engine in packed order."""

    def __init__(self, *, num_envs: int = 1024, lib_path: str | None = None) -> None:
        self.num_envs = int(max(1, num_envs))
        self.runner = DrMarioPoolRunner(
            num_envs=self.num_envs,
            # Annotation needs the settled board and event counters, not the
            # next spawn's observation or reachability plan. Lazy decision
            # output avoids an otherwise dominant BFS for every alternative.
            obs_spec=0,
            obs_channels=0,
            emit_board=True,
            lazy_decision_outputs=True,
            lib_path=lib_path,
        )

    def close(self) -> None:
        self.runner.close()

    def __enter__(self) -> "NativeAfterstateSimulator":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @staticmethod
    def _root_spec(
        board: np.ndarray,
        pill: np.ndarray,
        preview: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        count: int,
        speed: int,
        speed_ups: int,
    ):
        feasible = np.zeros(MACRO_ACTIONS, dtype=np.uint8)
        costs512 = np.full(MACRO_ACTIONS, 0xFFFF, dtype=np.uint16)
        valid_actions = np.asarray(actions[:count], dtype=np.int64)
        feasible[valid_actions] = 1
        costs512[valid_actions] = np.asarray(costs[:count], dtype=np.uint16)
        return build_reset_spec(
            level=14,
            speed_setting=int(speed),
            speed_ups=int(speed_ups),
            checkpoint_enabled=True,
            checkpoint_board=board,
            checkpoint_falling_colors=(int(pill[0]), int(pill[1])),
            checkpoint_preview_colors=(int(preview[0]), int(preview[1])),
            checkpoint_speed_ups=int(speed_ups),
            inject_plan=True,
            inject_feasible=feasible,
            inject_costs=costs512,
        )

    def simulate_packed(
        self,
        *,
        fields: np.ndarray,
        pills: np.ndarray,
        previews: np.ndarray,
        candidate_actions: np.ndarray,
        candidate_costs: np.ndarray,
        candidate_count: np.ndarray,
        speed: np.ndarray,
        speed_ups: np.ndarray,
    ) -> AfterstateBatch:
        fields = np.asarray(fields, dtype=np.uint8).reshape(-1, 128)
        counts = np.asarray(candidate_count, dtype=np.int64).reshape(-1)
        total = int(counts.sum())
        result_fields = np.empty((total, 128), dtype=np.uint8)
        terminal = np.empty(total, dtype=np.uint8)
        invalid = np.empty(total, dtype=np.bool_)
        tau = np.empty(total, dtype=np.uint32)
        remaining = np.empty(total, dtype=np.uint16)
        viruses = np.empty(total, dtype=np.uint16)
        nonviruses = np.empty(total, dtype=np.uint16)
        events = np.empty(total, dtype=np.uint16)

        roots = [
            self._root_spec(
                fields[i],
                pills[i],
                previews[i],
                candidate_actions[i],
                candidate_costs[i],
                int(counts[i]),
                int(speed[i]),
                int(speed_ups[i]),
            )
            for i in range(len(fields))
        ]
        row_index = np.repeat(np.arange(len(fields), dtype=np.int64), counts)
        slot_index = (
            np.concatenate([np.arange(n, dtype=np.int64) for n in counts])
            if total
            else np.empty(0, dtype=np.int64)
        )
        actions_flat = candidate_actions[row_index, slot_index].astype(np.int32, copy=False)
        buf = self.runner.buffers
        for start in range(0, total, self.num_envs):
            stop = min(start + self.num_envs, total)
            size = stop - start
            chunk_rows = row_index[start:stop]
            specs = [roots[int(row)] for row in chunk_rows]
            actions = np.full(self.num_envs, -1, dtype=np.int32)
            actions[:size] = actions_flat[start:stop]
            if size < self.num_envs:
                specs.extend([roots[int(chunk_rows[0])]] * (self.num_envs - size))
            self.runner.reset(None, specs)
            self.runner.step(actions, None, None)
            result_fields[start:stop] = buf.board_bytes[:size]
            terminal[start:stop] = buf.terminal_reason[:size]
            invalid[start:stop] = buf.invalid_action[:size] != -1
            tau[start:stop] = buf.tau_frames[:size]
            remaining[start:stop] = buf.viruses_rem[:size]
            viruses[start:stop] = buf.tiles_cleared_virus[:size]
            nonviruses[start:stop] = buf.tiles_cleared_nonvirus[:size]
            events[start:stop] = buf.match_events[:size]
        return AfterstateBatch(
            result_fields, terminal, invalid, tau, remaining, viruses, nonviruses, events
        )


__all__ = [
    "AfterstateBatch",
    "NativeAfterstateSimulator",
    "decode_sparse_deltas",
    "encode_sparse_deltas",
]
