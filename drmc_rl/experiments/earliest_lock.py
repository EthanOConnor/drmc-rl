"""Earliest-lock dominance gate for the placement-only action abstraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.envs.backends.vs_forced import ForcedLock, ForcedLockDriver


@dataclass(frozen=True, slots=True)
class TimingProbe:
    id: str
    reset_spec: Mapping[str, object]
    target_side: int
    column: int
    row_bottom: int
    rotation: int
    lock_frames: tuple[int, ...]
    opponent_lock: ForcedLock | None = None
    stratum: str = "unspecified"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.target_side not in (0, 1):
            raise ValueError("target_side must be 0 or 1")
        if not self.lock_frames or any(frame < 0 for frame in self.lock_frames):
            raise ValueError("lock_frames must contain non-negative pair-clock frames")
        if tuple(sorted(set(self.lock_frames))) != self.lock_frames:
            raise ValueError("lock_frames must be strictly increasing and unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TimingProbe":
        opponent_raw = value.get("opponent_lock")
        opponent = None
        if opponent_raw is not None:
            opponent_map = dict(opponent_raw)  # type: ignore[arg-type]
            opponent = ForcedLock(
                column=int(opponent_map.get("column", opponent_map.get("col", 0))),
                row_bottom=int(opponent_map.get("row_bottom", opponent_map.get("row", 0))),
                rotation=int(opponent_map.get("rotation", opponent_map.get("rot", 0))),
                lock_frame=int(opponent_map["lock_frame"]),
            )
        return cls(
            id=str(value["id"]),
            reset_spec=dict(value["reset_spec"]),  # type: ignore[arg-type]
            target_side=int(value["target_side"]),
            column=int(value["column"]),
            row_bottom=int(value["row_bottom"]),
            rotation=int(value["rotation"]),
            lock_frames=tuple(int(item) for item in value["lock_frames"]),  # type: ignore[arg-type]
            opponent_lock=opponent,
            stratum=str(value.get("stratum", "unspecified")),
            metadata=dict(value.get("metadata", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TimingOutcome:
    probe_id: str
    stratum: str
    lock_frame: int
    earliest_frame: int
    full_state_hash: str
    structural_state_hash: str
    transition_changed: bool
    clock_changed: bool
    value: float | None
    value_delta: float | None
    terminal: bool
    truncated: bool
    side_frames: tuple[int, int]
    need_action: tuple[bool, bool]
    outcome: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TimingExperimentReport:
    schema: str
    probes: int
    outcomes: int
    changed_probes: int
    changed_fraction: float
    clock_divergent_probes: int
    clock_divergent_fraction: float
    beneficial_delays: int
    beneficial_fraction: float | None
    wilson_low: float
    wilson_high: float
    by_stratum: Mapping[str, Mapping[str, float | int]]
    records: tuple[TimingOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class TimingBackend(Protocol):
    def evaluate(self, probe: TimingProbe, lock_frame: int) -> tuple[Mapping[str, object], float | None]: ...


class NativeForcedTimingBackend:
    """Strict native transition backend.  Optional values can be added later."""

    def __init__(
        self,
        runner: DrMarioVsPoolRunner,
        *,
        value_fn: Callable[[Mapping[str, object], TimingProbe], float] | None = None,
    ):
        if runner.num_pairs != 1:
            raise ValueError("timing experiment backend requires a one-pair runner")
        self.runner = runner
        self.driver = ForcedLockDriver(runner)
        self.value_fn = value_fn

    def evaluate(self, probe: TimingProbe, lock_frame: int):
        from drmc_rl.envs.backends.drmario_vs_pool import build_vs_reset_spec

        reset_spec = dict(probe.reset_spec)
        for key in (
            "checkpoint_board",
            "checkpoint_falling_colors",
            "checkpoint_preview_colors",
        ):
            if key in reset_spec and reset_spec[key] is not None:
                reset_spec[key] = np.asarray(reset_spec[key], dtype=np.uint8)
        spec = build_vs_reset_spec(**reset_spec)
        self.runner.reset(np.ones(1, dtype=np.uint8), [spec])
        locks: list[ForcedLock | None] = [None, None]
        locks[probe.target_side] = ForcedLock(
            column=probe.column,
            row_bottom=probe.row_bottom,
            rotation=probe.rotation,
            lock_frame=int(lock_frame),
        )
        opponent = 1 - probe.target_side
        locks[opponent] = probe.opponent_lock or ForcedLock.spectator()
        self.driver.step(locks)
        snapshot = self.driver.snapshot()
        normalized = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in snapshot.items()
        }
        value = None if self.value_fn is None else float(self.value_fn(normalized, probe))
        return normalized, value


def _state_hash(snapshot: Mapping[str, object], *, include_clocks: bool) -> str:
    payload_source = dict(snapshot)
    if not include_clocks:
        payload_source.pop("side_frames", None)
    payload = json.dumps(
        payload_source, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


class EarliestLockExperiment:
    def __init__(self, backend: TimingBackend, *, value_epsilon: float = 1e-3):
        self.backend = backend
        self.value_epsilon = float(value_epsilon)

    def run(self, probes: Sequence[TimingProbe]) -> TimingExperimentReport:
        records: list[TimingOutcome] = []
        changed_probes = 0
        clock_divergent_probes = 0
        beneficial = 0
        valued_delays = 0
        strata: dict[str, dict[str, int]] = {}
        for probe in probes:
            baseline_frame = probe.lock_frames[0]
            baseline_snapshot, baseline_value = self.backend.evaluate(probe, baseline_frame)
            baseline_full_hash = _state_hash(baseline_snapshot, include_clocks=True)
            baseline_structural_hash = _state_hash(baseline_snapshot, include_clocks=False)
            baseline_clocks = tuple(int(item) for item in baseline_snapshot.get("side_frames", (0, 0)))
            probe_changed = False
            probe_clock_changed = False
            stratum = strata.setdefault(
                probe.stratum,
                {"probes": 0, "changed": 0, "clock_divergent": 0, "beneficial": 0},
            )
            stratum["probes"] += 1
            for frame in probe.lock_frames:
                if frame == baseline_frame:
                    snapshot, value = baseline_snapshot, baseline_value
                else:
                    snapshot, value = self.backend.evaluate(probe, frame)
                full_state_hash = _state_hash(snapshot, include_clocks=True)
                structural_state_hash = _state_hash(snapshot, include_clocks=False)
                changed = structural_state_hash != baseline_structural_hash
                clocks = tuple(int(item) for item in snapshot.get("side_frames", (0, 0)))
                clock_changed = clocks != baseline_clocks
                probe_changed |= changed
                probe_clock_changed |= clock_changed
                value_delta = (
                    None
                    if value is None or baseline_value is None
                    else float(value - baseline_value)
                )
                if frame != baseline_frame and value_delta is not None:
                    valued_delays += 1
                    if value_delta > self.value_epsilon:
                        beneficial += 1
                        stratum["beneficial"] += 1
                side_frames = tuple(int(item) for item in snapshot.get("side_frames", (0, 0)))
                need_action = tuple(bool(item) for item in snapshot.get("need_action", (False, False)))
                outcome = tuple(int(item) for item in snapshot.get("outcome", (0, 0)))
                terminal_raw = snapshot.get("terminated", (False,))
                truncated_raw = snapshot.get("truncated", (False,))
                records.append(
                    TimingOutcome(
                        probe_id=probe.id,
                        stratum=probe.stratum,
                        lock_frame=int(frame),
                        earliest_frame=int(baseline_frame),
                        full_state_hash=full_state_hash,
                        structural_state_hash=structural_state_hash,
                        transition_changed=changed,
                        clock_changed=clock_changed,
                        value=None if value is None else float(value),
                        value_delta=value_delta,
                        terminal=bool(tuple(terminal_raw)[0]),  # type: ignore[arg-type]
                        truncated=bool(tuple(truncated_raw)[0]),  # type: ignore[arg-type]
                        side_frames=(side_frames[0], side_frames[1]),
                        need_action=(need_action[0], need_action[1]),
                        outcome=(outcome[0], outcome[1]),
                    )
                )
            if probe_changed:
                changed_probes += 1
                stratum["changed"] += 1
            if probe_clock_changed:
                clock_divergent_probes += 1
                stratum["clock_divergent"] += 1
        low, high = _wilson(changed_probes, len(probes))
        by_stratum = {
            key: {
                **value,
                "changed_fraction": value["changed"] / max(value["probes"], 1),
                "clock_divergent_fraction": value["clock_divergent"]
                / max(value["probes"], 1),
            }
            for key, value in strata.items()
        }
        return TimingExperimentReport(
            schema="drmc-earliest-lock-dominance-v1",
            probes=len(probes),
            outcomes=len(records),
            changed_probes=changed_probes,
            changed_fraction=changed_probes / max(len(probes), 1),
            clock_divergent_probes=clock_divergent_probes,
            clock_divergent_fraction=clock_divergent_probes / max(len(probes), 1),
            beneficial_delays=beneficial,
            beneficial_fraction=None if valued_delays == 0 else beneficial / valued_delays,
            wilson_low=low,
            wilson_high=high,
            by_stratum=by_stratum,
            records=tuple(records),
        )


def load_probes(path: str | Path) -> list[TimingProbe]:
    probes: list[TimingProbe] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                probes.append(TimingProbe.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid timing probe at line {line_number}: {exc}") from exc
    return probes


__all__ = [
    "EarliestLockExperiment",
    "NativeForcedTimingBackend",
    "TimingBackend",
    "TimingExperimentReport",
    "TimingOutcome",
    "TimingProbe",
    "load_probes",
]
