"""Big, showy clears: exact per-placement clear features and a showiness score.

One definition serves the corpus miner (``tools.mine_big_clears``), the start
bank (``tools.build_big_clear_bank``), the benchmark (``tools.eval_big_clear``),
the pool-game metrics and arm (b)'s shaped bonus, so "big" means the same thing
everywhere.

Mechanics (ROM, ``drmc_rl.game.afterstate``): after a lock the bottle settles
in *rounds*; each round marks every horizontal then vertical run of four or
more same-colored tiles (a cell may belong to a horizontal and a vertical run
at once), clears them, drops what is unsupported and scans again. The ROM
combo counter counts matched lines over all rounds; two or more lines store
``min(lines, 4)`` garbage pieces for the opponent.

What counts as big (the experiment is about size and spectacle, not ordinary
attacks): an ordinary 2-line, 2-garbage combo scores low. The score is a sum of
points, each zero for a plain single 4-line clear:

* ``cells``: tiles cleared in the whole resolution, beyond 4;
* ``rounds``: cascade rounds beyond the first (tall chains);
* ``simultaneous``: lines in the richest single round beyond one;
* ``long``: tiles beyond four in each line (5+ in a row);
* ``viruses``: viruses cleared beyond two;
* ``span``: rows spanned by the cleared cells beyond six (clears reaching
  across much of the bottle);
* ``cross``: a round where a horizontal and a vertical line share a cell;
* ``rainbow``: all three colors cleared in one resolution;
* ``garbage``: a small bonus for 3 or 4 pieces only (2 pieces earn nothing).

Level and speed are context for sampling, never part of the score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from drmc_rl.game import cascade as _cascade

SCHEMA = "drmc-big-clear-v1"
# Point weights (pre-registered in runs/review-20260909/big-clear-v1/preregistration.json).
WEIGHTS = dict(cells=1.0, rounds=3.0, simultaneous=2.0, long=1.5, viruses=1.0,
               span=0.5, cross=2.0, rainbow=2.0, garbage3=1.0, garbage4=2.0)
# Showiness tiers, fixed from the human score distribution (scores are
# multiples of 0.5). See ``tier``; T0 is any clear below T1.
TIERS = (("T1", 20.0), ("T2", 30.0), ("T3", 42.0))
_ATTACK_MIN, _ATTACK_MAX = 2, 4
_CANONICAL_TO_NES = (1, 0, 2)
_SECOND = ((0, 1), (1, 0), (0, -1), (-1, 0))
_TILE_TYPES = ((0x60, 0x70), (0x40, 0x50), (0x70, 0x60), (0x50, 0x40))


@dataclass(frozen=True)
class ClearFeatures:
    cells: int = 0
    viruses: int = 0
    rounds: int = 0
    lines: int = 0
    max_round_lines: int = 0
    long_tiles: int = 0  # sum over lines of (length - 4)
    max_line: int = 0
    span_rows: int = 0
    span_cols: int = 0
    cross: int = 0
    colors: int = 0
    garbage: int = 0

    @property
    def cleared(self) -> bool:
        return self.rounds > 0

    def points(self) -> dict[str, float]:
        if not self.rounds:
            return {k: 0.0 for k in ("cells", "rounds", "simultaneous", "long", "viruses", "span",
                                     "cross", "rainbow", "garbage")}
        w = WEIGHTS
        return dict(
            cells=w["cells"] * max(0, self.cells - 4),
            rounds=w["rounds"] * max(0, self.rounds - 1),
            simultaneous=w["simultaneous"] * max(0, self.max_round_lines - 1),
            long=w["long"] * self.long_tiles,
            viruses=w["viruses"] * max(0, self.viruses - 2),
            span=w["span"] * max(0, self.span_rows - 6),
            cross=w["cross"] * (self.cross > 0),
            rainbow=w["rainbow"] * (self.colors >= 3),
            garbage=w["garbage4"] if self.garbage >= 4 else w["garbage3"] if self.garbage == 3 else 0.0,
        )

    def score(self) -> float:
        return float(sum(self.points().values()))

    def to_dict(self) -> dict:
        return dict(asdict(self), score=self.score(), tier=tier(self.score()))


def tier(score: float) -> str:
    name = "T0"
    for label, bar in TIERS:
        if score >= bar:
            name = label
    return name


def tier_index(score: float) -> int:
    return sum(score >= bar for _, bar in TIERS)


def _mark_lines(board: bytearray) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """ROM horizontal-then-vertical scan; returns (orientation, length, color, cells) per line."""
    width, height = _cascade.BOARD_WIDTH, _cascade.BOARD_HEIGHT
    lines = []
    for row in range(height):
        col = 0
        while col <= width - _cascade.MIN_CHAIN:
            tile = board[row * width + col]
            if tile >= _cascade.TILE_JUST_EMPTIED:
                col += 1
                continue
            color = tile & _cascade.MASK_COLOR
            chain = 1
            while col + chain < width and (board[row * width + col + chain] & _cascade.MASK_COLOR) == color:
                chain += 1
            if chain >= _cascade.MIN_CHAIN:
                cells = tuple(row * width + col + k for k in range(chain))
                lines.append((0, chain, color & 3, cells))
                for index in cells:
                    board[index] = _cascade.TILE_CLEARED | (board[index] & _cascade.MASK_COLOR)
                col += chain
            else:
                col += 1
    for col in range(width):
        row = 0
        while row <= height - _cascade.MIN_CHAIN:
            tile = board[row * width + col]
            if tile >= _cascade.TILE_JUST_EMPTIED:
                row += 1
                continue
            color = tile & _cascade.MASK_COLOR
            chain = 1
            while row + chain < height and (board[(row + chain) * width + col] & _cascade.MASK_COLOR) == color:
                chain += 1
            if chain >= _cascade.MIN_CHAIN:
                cells = tuple((row + k) * width + col for k in range(chain))
                lines.append((1, chain, color & 3, cells))
                for index in cells:
                    board[index] = _cascade.TILE_CLEARED | (board[index] & _cascade.MASK_COLOR)
                row += chain
            else:
                row += 1
    return lines


def resolve(placed) -> tuple[bytes, ClearFeatures, list]:
    """Settle a bottle whose locked pill is already written in.

    Returns the settled 128-byte field, the clear features and the per-round
    detail ``[(lines, cleared cell indices, virus cell indices), ...]``.
    """
    board = bytearray(bytes(placed))
    if len(board) != 128:
        raise ValueError("a bottle is 128 bytes")
    detail = []
    cells = viruses = long_tiles = max_line = max_round = cross = 0
    total_lines = 0
    colors: set[int] = set()
    rows_hit: set[int] = set()
    cols_hit: set[int] = set()
    while True:
        while _cascade._drop_pass(board):
            pass
        pre = bytes(board)
        found = _mark_lines(board)
        if not found:
            break
        cleared = [i for i in range(128) if board[i] & 0xF0 == _cascade.TILE_CLEARED
                   and pre[i] & 0xF0 != _cascade.TILE_CLEARED]
        virus_cells = [i for i in cleared if pre[i] & 0xF0 == 0xD0]
        horizontal = {c for o, _, _, cs in found if o == 0 for c in cs}
        vertical = {c for o, _, _, cs in found if o == 1 for c in cs}
        cross += bool(horizontal & vertical)
        cells += len(cleared)
        viruses += len(virus_cells)
        total_lines += len(found)
        max_round = max(max_round, len(found))
        for _, length, color, _ in found:
            long_tiles += length - 4
            max_line = max(max_line, length)
            colors.add(color)
        rows_hit.update(i // 8 for i in cleared)
        cols_hit.update(i % 8 for i in cleared)
        detail.append((found, cleared, virus_cells))
        _cascade._update_field(board)
    for index in range(128):
        if board[index] >= _cascade.TILE_JUST_EMPTIED:
            board[index] = 0xFF
    garbage = min(total_lines, _ATTACK_MAX) if total_lines >= _ATTACK_MIN else 0
    features = ClearFeatures(
        cells=cells, viruses=viruses, rounds=len(detail), lines=total_lines, max_round_lines=max_round,
        long_tiles=long_tiles, max_line=max_line,
        span_rows=(max(rows_hit) - min(rows_hit) + 1) if rows_hit else 0,
        span_cols=(max(cols_hit) - min(cols_hit) + 1) if cols_hit else 0,
        cross=cross, colors=len(colors), garbage=garbage)
    return bytes(board), features, detail


def place(field, pill_canonical, action: int) -> bytes:
    """Write a canonical-color pill at a macro action (``drmc_rl.game.afterstate`` geometry)."""
    orientation, cell = divmod(int(action), 128)
    if not 0 <= orientation < 4:
        raise ValueError("invalid placement action")
    row, col = divmod(cell, 8)
    dr, dc = _SECOND[orientation]
    board = bytearray(bytes(field))
    for (r, c), tile, color in zip(((row, col), (row + dr, col + dc)), _TILE_TYPES[orientation],
                                   pill_canonical, strict=True):
        if not (0 <= r < 16 and 0 <= c < 8) or board[r * 8 + c] != 0xFF:
            raise ValueError("placement outside the empty bottle cells")
        board[r * 8 + c] = tile | _CANONICAL_TO_NES[int(color)]
    return bytes(board)


def placement_features(field, pill_canonical, action: int) -> ClearFeatures:
    """Features of one placement from its spawn bottle, canonical pill and macro action."""
    return resolve(place(field, pill_canonical, action))[1]


def forms_line(placed: bytes, cells: tuple[int, int]) -> bool:
    """Cheap test: does either new pill cell complete a run of four (a clear starts)?"""
    for index in cells:
        color = placed[index] & 0x0F
        row, col = divmod(index, 8)
        run = 1
        c = col - 1
        while c >= 0 and placed[row * 8 + c] != 0xFF and placed[row * 8 + c] & 0x0F == color:
            run += 1; c -= 1
        c = col + 1
        while c < 8 and placed[row * 8 + c] != 0xFF and placed[row * 8 + c] & 0x0F == color:
            run += 1; c += 1
        if run >= 4:
            return True
        run = 1
        r = row - 1
        while r >= 0 and placed[r * 8 + col] != 0xFF and placed[r * 8 + col] & 0x0F == color:
            run += 1; r -= 1
        r = row + 1
        while r < 16 and placed[r * 8 + col] != 0xFF and placed[r * 8 + col] & 0x0F == color:
            run += 1; r += 1
        if run >= 4:
            return True
    return False


def ascii_board(field, highlight=()) -> list[str]:
    """Rows of a bottle: viruses R/Y/B, pill tiles r/y/b, '.' empty; highlighted cells in brackets."""
    names = "YRB?"
    marks = set(highlight)
    out = []
    for row in range(16):
        line = []
        for col in range(8):
            index = row * 8 + col
            tile = field[index]
            glyph = "." if tile == 0xFF else names[tile & 3] if tile & 0xF0 == 0xD0 else names[tile & 3].lower()
            line.append(f"[{glyph}]" if index in marks else f" {glyph} ")
        out.append("|" + "".join(line) + "|")
    return out


def showiness_bonus(features: ClearFeatures, spec: dict) -> float:
    """Arm (b) event bonus for one learner placement (see ``ShowinessBonus`` in the trainer).

    ``spec``: ``per_point`` reward per showiness point above ``threshold``
    (clears below it earn nothing), capped at ``event_cap`` per placement.
    """
    if not features.rounds:
        return 0.0
    excess = features.score() - float(spec.get("threshold", TIERS[0][1]))
    if excess < 0:
        return 0.0
    value = float(spec.get("base", 0.0)) + float(spec["per_point"]) * excess
    return float(min(value, float(spec["event_cap"])))


__all__ = ["SCHEMA", "TIERS", "WEIGHTS", "ClearFeatures", "ascii_board", "forms_line", "place",
           "placement_features", "resolve", "showiness_bonus", "tier", "tier_index"]
