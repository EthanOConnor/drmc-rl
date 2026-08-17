"""Exact public-belief model for the hidden 128-pill reserve.

The native engine generates the entire reserve from a two-byte RNG before play.
The next reserve entry is therefore not an independent uniform ordered color
pair. This module enumerates the 16-bit seed prior once, conditions it only on
publicly observed reserve entries, and returns exact posterior-predictive reveal
probabilities.
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
CHANCE_MODEL_ID = "nes-reserve-seed-belief-v1"
BELIEF_SCHEMA = "drmc-pill-reserve-belief-v1"
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
    observations: tuple[tuple[int, int], ...]
) -> np.ndarray:
    table = reserve_table()
    mask = np.ones(SEED_COUNT, dtype=bool)
    for index, pill in observations:
        mask &= table[:, index] == pill
    matches = np.flatnonzero(mask).astype(np.int32)
    matches.setflags(write=False)
    return matches


@dataclass(frozen=True, slots=True)
class PillReserveBelief:
    """Uniform posterior over seed hypotheses consistent with public reveals."""

    observations: tuple[tuple[int, int], ...] = ()
    prior_id: str = "uniform-two-byte-seed-v1"

    def __post_init__(self) -> None:
        normalized = _normalize_observations(self.observations)
        object.__setattr__(self, "observations", normalized)
        if self.prior_id != "uniform-two-byte-seed-v1":
            raise ValueError(f"unsupported reserve prior {self.prior_id!r}")
        if self.seed_count == 0:
            raise ValueError(
                "reserve observations are impossible under the configured prior"
            )

    @property
    def seed_count(self) -> int:
        return int(_matching_seed_indices(self.observations).size)

    @property
    def entropy_bits(self) -> float:
        return float(np.log2(self.seed_count))

    def probabilities(self, reserve_index: int) -> np.ndarray:
        index = int(reserve_index) % RESERVE_LENGTH
        seeds = _matching_seed_indices(self.observations)
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
        )

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
        belief = cls(
            tuple((int(item[0]), int(item[1])) for item in raw),  # type: ignore[index]
            str(payload.get("prior_id", "uniform-two-byte-seed-v1")),
        )
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
    "PillReserveBelief",
    "RESERVE_LENGTH",
    "canonical_pair_to_pill_id",
    "pill_id_to_canonical_pair",
    "reserve_for_seed",
    "reserve_table",
]
