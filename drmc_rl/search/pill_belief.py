"""Public-belief model for the hidden 128-pill reserve.

The native engine generates the entire reserve from a two-byte RNG before play.
The next reserve entry is therefore not an independent uniform ordered color
pair. This module enumerates the experiment's uniform 16-bit reset-seed prior,
conditions it on the public initial virus bottle and every publicly observed
reserve entry, and returns exact posterior-predictive reveal probabilities
under that declared prior.

The real console advances one fixed LFSR orbit from its boot state.  The
uniform-reset prior used by the native training environment is deliberately not
described as the retail console prior.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

RESERVE_LENGTH = 128
PILL_COMBINATIONS = 9
SEED_COUNT = 1 << 16
CHANCE_MODEL_ID = "nes-reserve-public-seed-belief-v2"
BELIEF_SCHEMA = "drmc-pill-reserve-belief-v2"
PRIOR_ID = "uniform-two-byte-reset-seed-v1"
_CANONICAL_TO_RAW = (1, 0, 2)
_RAW_TO_CANONICAL = (1, 0, 2)


def canonical_pair_to_pill_id(colors: Sequence[int]) -> int:
    pair = tuple(int(value) for value in colors)
    if len(pair) != 2 or any(value not in (0, 1, 2) for value in pair):
        raise ValueError(f"expected two canonical colors in [0,2], got {pair!r}")
    return 3 * _CANONICAL_TO_RAW[pair[0]] + _CANONICAL_TO_RAW[pair[1]]


def pill_id_to_canonical_pair(pill_id: int) -> tuple[int, int]:
    value = int(pill_id)
    if not 0 <= value < PILL_COMBINATIONS:
        raise ValueError(f"pill id must be in [0,{PILL_COMBINATIONS - 1}]")
    return _RAW_TO_CANONICAL[value // 3], _RAW_TO_CANONICAL[value % 3]


def pill_id_to_raw_pair(pill_id: int) -> tuple[int, int]:
    """Return the native/raw NES ordered colors for a reserve pill id."""

    value = int(pill_id)
    if not 0 <= value < PILL_COMBINATIONS:
        raise ValueError(f"pill id must be in [0,{PILL_COMBINATIONS - 1}]")
    return value // 3, value % 3


def reserve_for_seed(seed0: int, seed1: int) -> np.ndarray:
    """Generate the retail rev0 reserve for one two-byte RNG seed."""

    first = int(seed0)
    second = int(seed1)
    if not 0 <= first <= 0xFF or not 0 <= second <= 0xFF:
        raise ValueError("seed bytes must be in [0,255]")
    reserve = np.empty(RESERVE_LENGTH, dtype=np.uint8)
    pill_id = 0
    for index in range(RESERVE_LENGTH - 1, -1, -1):
        carry = 1 if ((first ^ second) & 0x02) else 0
        old_first = first
        first = ((first >> 1) | (carry << 7)) & 0xFF
        carry = old_first & 0x01
        second = ((second >> 1) | (carry << 7)) & 0xFF
        pill_id = (pill_id + (first & 0x0F)) % PILL_COMBINATIONS
        reserve[index] = pill_id
    return reserve


@functools.lru_cache(maxsize=1)
def reserve_table() -> np.ndarray:
    """Return [65536,128] pill ids for the uniform two-byte seed prior."""

    first = np.repeat(np.arange(256, dtype=np.uint16), 256)
    second = np.tile(np.arange(256, dtype=np.uint16), 256)
    pill = np.zeros(SEED_COUNT, dtype=np.uint16)
    table = np.empty((SEED_COUNT, RESERVE_LENGTH), dtype=np.uint8)
    for index in range(RESERVE_LENGTH - 1, -1, -1):
        carry = ((first ^ second) & 0x02) != 0
        old_first = first
        first = (first >> 1) | (carry.astype(np.uint16) << 7)
        carry_second = old_first & 0x01
        second = (second >> 1) | (carry_second << 7)
        pill = (pill + (first & 0x0F)) % PILL_COMBINATIONS
        table[:, index] = pill.astype(np.uint8)
    table.setflags(write=False)
    return table


@functools.lru_cache(maxsize=8)
def initial_board_table(level: int) -> np.ndarray:
    """Return the public initial virus bottle for every reset-seed hypothesis."""

    from drmc_rl.seedlab.rng import generate_game

    normalized_level = int(level)
    if not 0 <= normalized_level <= 20:
        raise ValueError("level must be in [0,20]")
    table = np.empty((SEED_COUNT, 128), dtype=np.uint8)
    for seed in range(SEED_COUNT):
        table[seed] = np.frombuffer(
            generate_game(normalized_level, seed).board,
            dtype=np.uint8,
        )
    table.setflags(write=False)
    return table


def _normalize_observations(
    observations: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    by_index: dict[int, int] = {}
    for raw_index, raw_pill in observations:
        index = int(raw_index) % RESERVE_LENGTH
        pill = int(raw_pill)
        if not 0 <= pill < PILL_COMBINATIONS:
            raise ValueError("observed pill id must be in [0,8]")
        previous = by_index.get(index)
        if previous is not None and previous != pill:
            raise ValueError(f"contradictory reserve observations at index {index}")
        by_index[index] = pill
    return tuple(sorted(by_index.items()))


# A path-dependent search can create many distinct reveal histories. Each
# cached result may hold tens of thousands of seed indices, so allowing one
# entry per possible seed can consume gigabytes. A bounded 4k cache preserves
# repeated-node speed while keeping a practical worker memory ceiling.
@functools.lru_cache(maxsize=4096)
def _matching_seed_indices(
    observations: tuple[tuple[int, int], ...],
    level: int | None,
    initial_board: bytes | None,
) -> np.ndarray:
    table = reserve_table()
    mask = np.ones(SEED_COUNT, dtype=bool)
    if (level is None) != (initial_board is None):
        raise ValueError("initial board conditioning requires both level and board")
    if initial_board is not None:
        board = np.frombuffer(initial_board, dtype=np.uint8)
        if board.shape != (128,):
            raise ValueError("initial public bottle must contain exactly 128 bytes")
        mask &= np.all(initial_board_table(int(level)) == board, axis=1)
    for index, pill in observations:
        mask &= table[:, index] == pill
    matches = np.flatnonzero(mask).astype(np.int32)
    matches.setflags(write=False)
    return matches


@dataclass(frozen=True, slots=True)
class PillReserveBelief:
    """Posterior over reset seeds consistent with public board and reveals."""

    observations: tuple[tuple[int, int], ...] = ()
    prior_id: str = PRIOR_ID
    level: int | None = None
    initial_board: bytes | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_observations(self.observations)
        object.__setattr__(self, "observations", normalized)
        if self.prior_id != PRIOR_ID:
            raise ValueError(f"unsupported reserve prior {self.prior_id!r}")
        if (self.level is None) != (self.initial_board is None):
            raise ValueError("initial board conditioning requires both level and board")
        if self.level is not None and not 0 <= int(self.level) <= 20:
            raise ValueError("level must be in [0,20]")
        if self.initial_board is not None and len(self.initial_board) != 128:
            raise ValueError("initial public bottle must contain exactly 128 bytes")
        if self.seed_count == 0:
            raise ValueError(
                "reserve observations are impossible under the configured prior"
            )

    @property
    def seed_count(self) -> int:
        return int(
            _matching_seed_indices(
                self.observations,
                self.level,
                self.initial_board,
            ).size
        )

    @property
    def entropy_bits(self) -> float:
        return float(np.log2(self.seed_count))

    def probabilities(self, reserve_index: int) -> np.ndarray:
        index = int(reserve_index) % RESERVE_LENGTH
        seeds = _matching_seed_indices(
            self.observations,
            self.level,
            self.initial_board,
        )
        counts = np.bincount(
            reserve_table()[seeds, index], minlength=PILL_COMBINATIONS
        ).astype(np.float64)
        total = float(counts.sum())
        if total <= 0:
            raise RuntimeError("empty reserve posterior")
        probability = counts / total
        probability.setflags(write=False)
        return probability

    def condition(self, reserve_index: int, pill_id: int) -> "PillReserveBelief":
        return PillReserveBelief(
            self.observations
            + ((int(reserve_index) % RESERVE_LENGTH, int(pill_id)),),
            self.prior_id,
            self.level,
            self.initial_board,
        )

    @classmethod
    def from_initial_board(
        cls,
        *,
        level: int,
        board: bytes | bytearray | memoryview | np.ndarray,
    ) -> "PillReserveBelief":
        if isinstance(board, (bytes, bytearray, memoryview)):
            raw = bytes(board)
        else:
            raw = np.asarray(board, dtype=np.uint8).reshape(-1).tobytes()
        return cls(level=int(level), initial_board=raw)

    def condition_visible(
        self,
        *,
        reserve_counter: int,
        falling_colors: Sequence[int],
        preview_colors: Sequence[int],
    ) -> "PillReserveBelief":
        """Condition on visible falling/preview pills for one side.

        At a normal decision boundary the reserve counter points one past the
        preview entry: falling=counter-2, preview=counter-1 (mod 128).
        """

        counter = int(reserve_counter) % RESERVE_LENGTH
        result = self.condition(
            counter - 2, canonical_pair_to_pill_id(falling_colors)
        )
        return result.condition(
            counter - 1, canonical_pair_to_pill_id(preview_colors)
        )

    def stable_hash(self) -> str:
        payload = json.dumps(
            {
                "prior_id": self.prior_id,
                "level": self.level,
                "initial_board": (
                    None if self.initial_board is None else self.initial_board.hex()
                ),
                "observations": self.observations,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": BELIEF_SCHEMA,
            "chance_model": CHANCE_MODEL_ID,
            "prior_id": self.prior_id,
            "level": self.level,
            "initial_board": (
                None if self.initial_board is None else self.initial_board.hex()
            ),
            "initial_board_conditioned": self.initial_board is not None,
            "observations": [list(item) for item in self.observations],
            "seed_count": self.seed_count,
            "stable_hash": self.stable_hash(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PillReserveBelief":
        if payload.get("schema", BELIEF_SCHEMA) != BELIEF_SCHEMA:
            raise ValueError("unsupported pill reserve belief schema")
        if payload.get("chance_model", CHANCE_MODEL_ID) != CHANCE_MODEL_ID:
            raise ValueError("pill reserve belief uses an incompatible chance model")
        raw = payload.get("observations", ())
        if not isinstance(raw, Sequence) or isinstance(
            raw, (str, bytes, bytearray)
        ):
            raise ValueError("belief observations must be a sequence")
        raw_board = payload.get("initial_board")
        board = None if raw_board is None else bytes.fromhex(str(raw_board))
        belief = cls(
            observations=tuple(
                (int(item[0]), int(item[1])) for item in raw  # type: ignore[index]
            ),
            prior_id=str(payload.get("prior_id", PRIOR_ID)),
            level=None if payload.get("level") is None else int(payload["level"]),
            initial_board=board,
        )
        expected_conditioned = payload.get("initial_board_conditioned")
        if expected_conditioned is not None and bool(expected_conditioned) != (
            belief.initial_board is not None
        ):
            raise ValueError("initial-board conditioning flag is inconsistent")
        expected_count = payload.get("seed_count")
        if expected_count is not None and int(expected_count) != belief.seed_count:
            raise ValueError(
                "pill reserve belief seed_count does not match observations"
            )
        expected_hash = payload.get("stable_hash")
        if expected_hash is not None and str(expected_hash) != belief.stable_hash():
            raise ValueError("pill reserve belief hash does not match observations")
        return belief


__all__ = [
    "BELIEF_SCHEMA",
    "CHANCE_MODEL_ID",
    "PILL_COMBINATIONS",
    "PRIOR_ID",
    "PillReserveBelief",
    "RESERVE_LENGTH",
    "canonical_pair_to_pill_id",
    "initial_board_table",
    "pill_id_to_canonical_pair",
    "pill_id_to_raw_pair",
    "reserve_for_seed",
    "reserve_table",
]
