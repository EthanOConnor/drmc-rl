"""Corpus-derived human execution profiles for controller scripts.

A human-rate claim is made against one named profile.  Validation uses hard
limits over reaction, button edges, short-window bursts, chords, corrections,
and total script complexity.  Average APM alone is intentionally insufficient.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

BUTTON_RIGHT = 0x01
BUTTON_LEFT = 0x02
BUTTON_DOWN = 0x04
BUTTON_UP = 0x08
BUTTON_START = 0x10
BUTTON_SELECT = 0x20
BUTTON_B = 0x40
BUTTON_A = 0x80
GAMEPLAY_MASK = BUTTON_RIGHT | BUTTON_LEFT | BUTTON_DOWN | BUTTON_B | BUTTON_A
DIRECTION_MASK = BUTTON_RIGHT | BUTTON_LEFT | BUTTON_DOWN | BUTTON_UP
MENU_MASK = BUTTON_START | BUTTON_SELECT


def _bit_count(value: int) -> int:
    return int(value & 0xFF).bit_count()


@dataclass(frozen=True, slots=True)
class ScriptMetrics:
    frames: int
    reaction_frames: int
    rising_edges: int
    falling_edges: int
    total_edges: int
    min_inter_edge_frames: int | None
    peak_edges_250ms: int
    peak_edges_1s: int
    peak_edges_10s: int
    max_simultaneous_buttons: int
    chord_frames: int
    forbidden_direction_frames: int
    menu_button_frames: int
    direction_reversals: int
    correction_bursts: int
    rotation_presses: int
    soft_drop_frames: int
    active_frames: int
    complexity: float

    @property
    def edges_per_second(self) -> float:
        return 0.0 if self.frames <= 0 else self.total_edges * 60.1 / self.frames


@dataclass(frozen=True, slots=True)
class ExecutionValidation:
    valid: bool
    violations: tuple[str, ...]
    metrics: ScriptMetrics


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Hard operation envelope plus descriptive distribution targets."""

    id: str
    description: str
    fps: float = 60.1
    min_reaction_frames: int = 0
    max_reaction_frames: int = 300
    min_inter_edge_frames: int = 0
    max_edges_250ms: int = 64
    max_edges_1s: int = 256
    max_edges_10s: int = 2048
    max_simultaneous_buttons: int = 5
    max_direction_reversals: int = 128
    max_correction_bursts: int = 128
    max_complexity: float = 1e9
    allow_left_right_chord: bool = False
    allow_up_down_chord: bool = False
    allow_menu_buttons: bool = False
    distribution_targets: Mapping[str, tuple[float, float]] | None = None

    def validate(self, script: Sequence[int] | np.ndarray) -> ExecutionValidation:
        metrics = script_metrics(script, fps=self.fps)
        violations: list[str] = []
        checks = (
            (
                metrics.reaction_frames < self.min_reaction_frames,
                f"reaction<{self.min_reaction_frames}",
            ),
            (
                metrics.reaction_frames > self.max_reaction_frames,
                f"reaction>{self.max_reaction_frames}",
            ),
            (
                metrics.min_inter_edge_frames is not None
                and metrics.min_inter_edge_frames < self.min_inter_edge_frames,
                f"inter_edge<{self.min_inter_edge_frames}",
            ),
            (metrics.peak_edges_250ms > self.max_edges_250ms, "burst_250ms"),
            (metrics.peak_edges_1s > self.max_edges_1s, "burst_1s"),
            (metrics.peak_edges_10s > self.max_edges_10s, "burst_10s"),
            (
                metrics.max_simultaneous_buttons > self.max_simultaneous_buttons,
                "simultaneous_buttons",
            ),
            (
                metrics.direction_reversals > self.max_direction_reversals,
                "direction_reversals",
            ),
            (metrics.correction_bursts > self.max_correction_bursts, "correction_bursts"),
            (metrics.complexity > self.max_complexity, "complexity"),
            (
                metrics.forbidden_direction_frames > 0 and not self.allow_left_right_chord,
                "left_right_chord",
            ),
            (metrics.menu_button_frames > 0 and not self.allow_menu_buttons, "menu_buttons"),
        )
        violations.extend(label for failed, label in checks if failed)
        if not self.allow_up_down_chord:
            masks = np.asarray(script, dtype=np.uint8).reshape(-1)
            if bool(((masks & BUTTON_UP != 0) & (masks & BUTTON_DOWN != 0)).any()):
                violations.append("up_down_chord")
        return ExecutionValidation(not violations, tuple(violations), metrics)

    @classmethod
    def unrestricted(cls) -> "ExecutionProfile":
        return cls(id="unrestricted", description="Exact planner scripts without human limits")

    @classmethod
    def elite_p99(cls) -> "ExecutionProfile":
        """Conservative placeholder until corpus extraction records a signed profile.

        The identifier deliberately says ``provisional``: release claims require
        replacing these limits with a versioned corpus-derived profile.
        """

        return cls(
            id="elite-p99-provisional",
            description="Provisional elite envelope; replace with corpus release evidence",
            min_reaction_frames=0,
            max_reaction_frames=90,
            min_inter_edge_frames=1,
            max_edges_250ms=8,
            max_edges_1s=24,
            max_edges_10s=160,
            max_simultaneous_buttons=3,
            max_direction_reversals=8,
            max_correction_bursts=4,
            max_complexity=120.0,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema"] = "drmc-execution-profile-v1"
        return payload

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExecutionProfile":
        if value.get("schema", "drmc-execution-profile-v1") != "drmc-execution-profile-v1":
            raise ValueError("unsupported execution profile schema")
        targets_raw = value.get("distribution_targets")
        targets = None
        if targets_raw is not None:
            if not isinstance(targets_raw, Mapping):
                raise ValueError("distribution_targets must be a mapping")
            targets = {
                str(key): (float(pair[0]), float(pair[1]))  # type: ignore[index]
                for key, pair in targets_raw.items()
            }
        keys = {
            "id",
            "description",
            "fps",
            "min_reaction_frames",
            "max_reaction_frames",
            "min_inter_edge_frames",
            "max_edges_250ms",
            "max_edges_1s",
            "max_edges_10s",
            "max_simultaneous_buttons",
            "max_direction_reversals",
            "max_correction_bursts",
            "max_complexity",
            "allow_left_right_chord",
            "allow_up_down_chord",
            "allow_menu_buttons",
        }
        kwargs = {key: value[key] for key in keys if key in value}
        kwargs["distribution_targets"] = targets
        return cls(**kwargs)  # type: ignore[arg-type]


def profile_from_json(path: str | Path) -> ExecutionProfile:
    return ExecutionProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def script_metrics(script: Sequence[int] | np.ndarray, *, fps: float = 60.1) -> ScriptMetrics:
    masks = np.asarray(script, dtype=np.uint8).reshape(-1)
    frames = int(masks.size)
    if frames == 0:
        return ScriptMetrics(0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    previous = np.concatenate((np.zeros(1, dtype=np.uint8), masks[:-1]))
    rising = masks & ~previous
    falling = previous & ~masks
    rising_counts = np.fromiter((_bit_count(int(value)) for value in rising), dtype=np.int16)
    falling_counts = np.fromiter((_bit_count(int(value)) for value in falling), dtype=np.int16)
    edge_counts = rising_counts + falling_counts
    edge_frames = np.flatnonzero(edge_counts)
    reaction_candidates = np.flatnonzero(masks & GAMEPLAY_MASK)
    reaction = int(reaction_candidates[0]) if reaction_candidates.size else frames

    edge_times: list[int] = []
    for frame, count in enumerate(edge_counts):
        edge_times.extend([frame] * int(count))
    if len(edge_times) < 2:
        min_interval = None
    else:
        min_interval = int(np.diff(np.asarray(edge_times, dtype=np.int32)).min())

    def peak(window_seconds: float) -> int:
        window = max(1, int(round(float(fps) * window_seconds)))
        cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(edge_counts)))
        starts = np.arange(frames, dtype=np.int64)
        stops = np.minimum(starts + window, frames)
        return int(np.max(cumulative[stops] - cumulative[starts], initial=0))

    active_counts = np.fromiter((_bit_count(int(value & GAMEPLAY_MASK)) for value in masks), dtype=np.int16)
    chord_frames = int((active_counts > 1).sum())
    forbidden_lr = int(((masks & BUTTON_LEFT != 0) & (masks & BUTTON_RIGHT != 0)).sum())
    menu_frames = int((masks & MENU_MASK != 0).sum())

    horizontal = np.zeros(frames, dtype=np.int8)
    horizontal[masks & BUTTON_LEFT != 0] = -1
    horizontal[masks & BUTTON_RIGHT != 0] = 1
    nonzero = horizontal[horizontal != 0]
    reversals = int((nonzero[1:] != nonzero[:-1]).sum()) if len(nonzero) >= 2 else 0
    correction_bursts = 0
    last_dir = 0
    last_change = -1000
    for frame, direction in enumerate(horizontal):
        if direction == 0 or direction == last_dir:
            continue
        if last_dir and frame - last_change <= max(2, int(round(fps * 0.15))):
            correction_bursts += 1
        last_dir = int(direction)
        last_change = frame

    rotation_presses = int(
        np.fromiter(
            (_bit_count(int(value & (BUTTON_A | BUTTON_B))) for value in rising),
            dtype=np.int16,
        ).sum()
    )
    soft_drop = int((masks & BUTTON_DOWN != 0).sum())
    active_frames = int((masks & GAMEPLAY_MASK != 0).sum())
    total_edges = int(edge_counts.sum())
    complexity = float(
        total_edges
        + 2.0 * reversals
        + 2.5 * correction_bursts
        + 0.5 * chord_frames
        + 0.1 * active_frames
    )
    return ScriptMetrics(
        frames=frames,
        reaction_frames=reaction,
        rising_edges=int(rising_counts.sum()),
        falling_edges=int(falling_counts.sum()),
        total_edges=total_edges,
        min_inter_edge_frames=min_interval,
        peak_edges_250ms=peak(0.25),
        peak_edges_1s=peak(1.0),
        peak_edges_10s=peak(10.0),
        max_simultaneous_buttons=int(active_counts.max(initial=0)),
        chord_frames=chord_frames,
        forbidden_direction_frames=forbidden_lr,
        menu_button_frames=menu_frames,
        direction_reversals=reversals,
        correction_bursts=correction_bursts,
        rotation_presses=rotation_presses,
        soft_drop_frames=soft_drop,
        active_frames=active_frames,
        complexity=complexity,
    )


def pareto_frontier(
    scripts: Iterable[Sequence[int] | np.ndarray],
    *,
    profile: ExecutionProfile | None = None,
) -> list[tuple[np.ndarray, ScriptMetrics]]:
    """Return non-dominated scripts over lock time, edges, bursts, and complexity."""

    candidates: list[tuple[np.ndarray, ScriptMetrics]] = []
    for script in scripts:
        array = np.asarray(script, dtype=np.uint8).reshape(-1)
        validation = (profile or ExecutionProfile.unrestricted()).validate(array)
        if validation.valid:
            candidates.append((array.copy(), validation.metrics))

    def vector(metrics: ScriptMetrics) -> tuple[float, ...]:
        return (
            float(metrics.frames),
            float(metrics.total_edges),
            float(metrics.peak_edges_250ms),
            float(metrics.direction_reversals),
            float(metrics.complexity),
        )

    frontier: list[tuple[np.ndarray, ScriptMetrics]] = []
    for index, candidate in enumerate(candidates):
        current = vector(candidate[1])
        dominated = False
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            alternative = vector(other[1])
            if all(a <= b for a, b in zip(alternative, current, strict=True)) and any(
                a < b for a, b in zip(alternative, current, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda item: vector(item[1]))
    return frontier


__all__ = [
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_DOWN",
    "BUTTON_LEFT",
    "BUTTON_RIGHT",
    "ExecutionProfile",
    "ExecutionValidation",
    "ScriptMetrics",
    "pareto_frontier",
    "profile_from_json",
    "script_metrics",
]
