"""Pure-Python mirror of the engine's seed-determined content generation.

Ports `rng_step`, `generatePillsReserve`, and `addVirus`/`setupLevel` from
`vendor/drmario_native/GameLogic.cpp` (themselves rules-exact ports of the rev0 ROM).
Used by the seed catalog for census/dedup (game hashes) without rollouts.
Parity vs the engine is asserted by `tests/test_seedlab_rng.py`.

Seed convention: a seed int packs the engine bytes as
`(rng_state[0] << 8) | rng_state[1]`, so the engine default `0x89, 0x88`
reads as `0x8988`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

MAIN_ORBIT_START = 0x8988
ORBIT_PERIOD = 32767

BOARD_WIDTH = 8
BOARD_HEIGHT = 16
BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT
LAST_ROW = 15

TILE_EMPTY = 0xFF
TILE_VIRUS = 0xD0
MASK_COLOR = 0x03

# Raw NES colors: 0=Yellow, 1=Red, 2=Blue.
COLOR_COMBINATION_LEFT = (0, 0, 0, 1, 1, 1, 2, 2, 2)
COLOR_COMBINATION_RIGHT = (0, 1, 2, 0, 1, 2, 0, 1, 2)

VIRUS_COLOR_RANDOM = (0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2, 1)

ADDVIRUS_MAX_HEIGHT_BY_LEVEL = (
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x09, 0x09, 0x09, 0x0A, 0x0A, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C,
)

# Matches `virusPlacement_colorBits`: Y->1, R->2, B->4, masked-empty->0.
_COLOR_BITS = (0x01, 0x02, 0x04, 0x00)


def seed_to_bytes(seed: int) -> Tuple[int, int]:
    return (seed >> 8) & 0xFF, seed & 0xFF


def bytes_to_seed(r0: int, r1: int) -> int:
    return ((int(r0) & 0xFF) << 8) | (int(r1) & 0xFF)


def rng_step(r0: int, r1: int) -> Tuple[int, int]:
    """One step of the NES shift register (GameLogic.cpp:rng_step)."""

    carry = 1 if ((r0 ^ r1) & 0x02) else 0
    n0 = ((r0 >> 1) | (carry << 7)) & 0xFF
    carry2 = r0 & 0x01
    n1 = ((r1 >> 1) | (carry2 << 7)) & 0xFF
    return n0, n1


@lru_cache(maxsize=1)
def orbit() -> Tuple[int, ...]:
    """The 32,767 console-reachable seeds, in phase order from 0x8988."""

    seeds: List[int] = []
    r0, r1 = seed_to_bytes(MAIN_ORBIT_START)
    for _ in range(ORBIT_PERIOD):
        seeds.append(bytes_to_seed(r0, r1))
        r0, r1 = rng_step(r0, r1)
    return tuple(seeds)


@lru_cache(maxsize=1)
def _orbit_pos_map() -> dict:
    return {s: i for i, s in enumerate(orbit())}


def orbit_pos(seed: int) -> int:
    """Phase index of `seed` on the main orbit; -1 if unreachable on console."""

    return _orbit_pos_map().get(int(seed) & 0xFFFF, -1)


def generate_pill_reserve(r0: int, r1: int) -> Tuple[bytes, int, int]:
    """Port of `generatePillsReserve` ($82A0): 128 pill ids, stored backward.

    Returns (reserve[0..127], r0, r1) with the RNG state after generation.
    The first pill played is `reserve[0]` (the last id generated).
    """

    reserve = bytearray(128)
    pill_id = 0
    for pills_to_generate in range(128, 0, -1):
        r0, r1 = rng_step(r0, r1)
        pill_id += r0 & 0x0F
        while pill_id >= 9:
            pill_id -= 9
        reserve[pills_to_generate - 1] = pill_id
    return bytes(reserve), r0, r1


def _color_bits_at(board: bytearray, pos: int) -> int:
    if pos < 0 or pos >= BOARD_SIZE:
        return 0
    return _COLOR_BITS[board[pos] & MASK_COLOR]


def place_viruses(level: int, r0: int, r1: int) -> Tuple[bytearray, int, int]:
    """Port of `setupLevel` + `addVirus` ($850D) onto an empty board.

    Returns (board[128] in engine layout: index 0 = top-left, row-major),
    plus the RNG state after placement.
    """

    lvl = min(int(level), 20)
    viruses_to_place = (lvl + 1) * 4
    viruses_added = 0
    board = bytearray([TILE_EMPTY] * BOARD_SIZE)

    max_attempts = max(viruses_to_place * 64, 256)
    attempts = 0
    while viruses_added < viruses_to_place and attempts < max_attempts:
        attempts += 1

        # generateRandNum: one advance yields rng0 (height) and rng1 (column).
        r0, r1 = rng_step(r0, r1)
        virus_height = r0 & 0x0F
        if virus_height > ADDVIRUS_MAX_HEIGHT_BY_LEVEL[lvl]:
            continue  # RNG consumed, no placement (RETRY_HEIGHT)

        col = r1 & 0x07
        virus_pos = (LAST_ROW - virus_height) * BOARD_WIDTH + col

        virus_to_add = viruses_to_place - viruses_added
        virus_color = virus_to_add & 0x03
        if virus_color == 3:
            r0, r1 = rng_step(r0, r1)
            virus_color = VIRUS_COLOR_RANDOM[r1 & 0x0F]
        virus_color &= MASK_COLOR

        # Scan forward to the next empty cell (no RNG consumed).
        while virus_pos < BOARD_SIZE and board[virus_pos] != TILE_EMPTY:
            virus_pos += 1
        if virus_pos >= BOARD_SIZE:
            continue  # GIVE_UP for this attempt

        placed = False
        while not placed:
            pos_col = virus_pos & 0x07
            verif = 0
            verif |= _color_bits_at(board, virus_pos - 16)
            verif |= _color_bits_at(board, virus_pos + 16)
            if pos_col >= 2:
                verif |= _color_bits_at(board, virus_pos - 2)
            if pos_col <= 5:
                verif |= _color_bits_at(board, virus_pos + 2)

            if verif == 0x07:
                virus_pos += 1
                while virus_pos < BOARD_SIZE and board[virus_pos] != TILE_EMPTY:
                    virus_pos += 1
                if virus_pos >= BOARD_SIZE:
                    break  # GIVE_UP for this attempt
                continue

            for _ in range(4):
                if ((1 << (virus_color & MASK_COLOR)) & verif) == 0:
                    break
                virus_color = 2 if virus_color == 0 else virus_color - 1

            board[virus_pos] = TILE_VIRUS | (virus_color & MASK_COLOR)
            viruses_added += 1
            placed = True

    return board, r0, r1


@dataclass(frozen=True, slots=True)
class GameContent:
    """Everything the seed determines: virus layout + full pill sequence."""

    level: int
    seed: int
    board: bytes  # 128 bytes, engine layout (index 0 = top-left)
    pills: bytes  # 128 pill ids (0..8); pills[0] is the first pill played
    virus_count: int


def generate_game(level: int, seed: int) -> GameContent:
    """Generate the seed-determined game content for (level, seed).

    Ordering matters and matches the engine reset path: the pill reserve is
    generated first, then viruses, both from one RNG stream with no warmup.
    """

    r0, r1 = seed_to_bytes(seed)
    pills, r0, r1 = generate_pill_reserve(r0, r1)
    board, r0, r1 = place_viruses(level, r0, r1)
    virus_count = sum(1 for t in board if t != TILE_EMPTY)
    return GameContent(
        level=int(level), seed=int(seed) & 0xFFFF, board=bytes(board),
        pills=pills, virus_count=virus_count,
    )


def game_hash(content: GameContent) -> bytes:
    """8-byte identity of the game's seed-determined content."""

    return hashlib.sha1(content.board + content.pills).digest()[:8]


def pill_colors_raw(pill_id: int) -> Tuple[int, int]:
    """(left, right) raw NES colors (0=Y, 1=R, 2=B) for a pill id 0..8."""

    return COLOR_COMBINATION_LEFT[pill_id], COLOR_COMBINATION_RIGHT[pill_id]
