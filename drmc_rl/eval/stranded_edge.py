"""Stranded edge-virus endgames: detector, resolution tracking and ASCII boards.

The scenario: late in a round, with few viruses left, a virus sits in an edge
column (x=0 or x=7) with an empty shaft beneath it. Viruses never fall, so the
shaft is not a threat to the virus; it is a construction problem. Every pill
that reaches the virus row beside it, or fills the shaft under it, has to stand
on a stack built up from the floor in the edge column and its inner neighbour,
and a clear inside that stack drops everything resting on it.

Everything here is a pure function of settled 128-byte bottles (row 0 at the
top, ``index = row * 8 + col``), so arena move traces, human-corpus decision
rows and start banks share one definition.

Definitions (``Definition`` defaults are the pre-registered benchmark values):

- ``gap``: empty cells directly below the virus in its own column, down to the
  first occupied cell or the floor.
- ``open``: empty cells in the inner neighbour column (x=1 or x=6) from the
  virus row downward. ``open >= 1`` means nothing stands beside the virus.
- ``stranded`` kind: ``gap >= min_gap``.
- ``pillar`` kind (thin support): ``gap <= 1`` and ``open >= min_gap`` with
  only pills below the virus in its column, i.e. the virus sits (almost) on a
  one-column tower with nothing beside it.
- ``above``: occupied cells above the virus in its column (a "buried" virus has
  junk stacked on top, which also closes the approach from above).
- A board qualifies when it has ``1..max_viruses`` viruses and at least one
  edge virus of either kind.

Resolution (``track``): the episode starts at the first qualifying decision for
a given virus and ends at the first later decision where that virus is gone.
A *support-destroying clear* is a placement, made while the virus is still on
the board, after which the build zone below the virus got lower: the stack top
in the edge column under the virus or in the neighbour column at/below the
virus row moved down (heights only fall through clears).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

SCHEMA = "drmc-stranded-edge-v1"
EMPTY = 0xFF
VIRUS = 0xD0
ROWS, COLS = 16, 8
EDGE_COLUMNS = (0, 7)
# NES low nibble: 0 yellow, 1 red, 2 blue.
_COLOR = "YRB?"


@dataclass(frozen=True)
class Definition:
    max_viruses: int = 3
    min_gap: int = 4
    # "Rest of the board mostly clear": non-virus occupied cells outside the
    # edge column and its neighbour. None disables the check.
    max_other_cells: int | None = 24
    include_pillar: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def grid(board) -> np.ndarray:
    if isinstance(board, (bytes, bytearray, memoryview)):
        array = np.frombuffer(bytes(board), dtype=np.uint8)
    else:
        array = np.asarray(board, dtype=np.uint8)
    if array.size != ROWS * COLS:
        raise ValueError(f"expected a 128-cell bottle, got {array.size}")
    return array.reshape(ROWS, COLS)


def _column_run(occupied: np.ndarray, start: int, col: int) -> int:
    """Empty cells in ``col`` from row ``start`` downward to the first occupied cell or the floor."""
    run = 0
    for row in range(start, ROWS):
        if occupied[row, col]:
            break
        run += 1
    return run


def inner(col: int) -> int:
    return 1 if col == 0 else COLS - 2


def edge_geometry(g: np.ndarray, row: int, col: int) -> dict:
    occupied = g != EMPTY
    gap = _column_run(occupied, row + 1, col)
    neighbour = inner(col)
    open_ = _column_run(occupied, row, neighbour)
    return dict(gap=gap, open=open_, support_row=row + 1 + gap, neighbour_top=row + open_,
                above=int(occupied[:row, col].sum()))


def other_cells(g: np.ndarray, col: int) -> int:
    keep = np.ones((ROWS, COLS), dtype=bool)
    keep[:, [col, inner(col)]] = False
    return int(((g != EMPTY) & ((g & 0xF0) != VIRUS) & keep).sum())


def stranded(board, definition: Definition = Definition()) -> list[dict]:
    """Qualifying edge viruses on one bottle, most stranded first."""
    g = grid(board)
    viruses = (g & 0xF0) == VIRUS
    count = int(viruses.sum())
    if not 1 <= count <= definition.max_viruses:
        return []
    found = []
    for col in EDGE_COLUMNS:
        for row in np.flatnonzero(viruses[:, col]):
            row = int(row)
            geometry = edge_geometry(g, row, col)
            if geometry["gap"] >= definition.min_gap:
                kind = "stranded"
            elif (definition.include_pillar and geometry["gap"] <= 1 and geometry["open"] >= definition.min_gap
                  and not viruses[row + 1:, col].any()):
                kind = "pillar"  # a pill-only tower under the virus, nothing beside it
            else:
                continue
            others = other_cells(g, col)
            if definition.max_other_cells is not None and others > definition.max_other_cells:
                continue
            found.append(dict(row=row, col=col, color=int(g[row, col] & 3), kind=kind,
                              viruses=count, other_cells=others, **geometry))
    found.sort(key=lambda v: (v["kind"] != "stranded", -v["gap"], -v["open"]))
    return found


def build_zone(g: np.ndarray, row: int, col: int) -> tuple[int, int]:
    """Stack tops (row indices, 16 = floor) under the virus and beside it."""
    geometry = edge_geometry(g, row, col)
    return geometry["support_row"], geometry["neighbour_top"]


def virus_present(g: np.ndarray, row: int, col: int, color: int) -> bool:
    tile = int(g[row, col])
    return (tile & 0xF0) == VIRUS and (tile & 3) == color


@dataclass
class Episode:
    onset: int
    row: int
    col: int
    color: int
    kind: str
    gap: int
    open: int
    above: int
    viruses: int
    other_cells: int
    decisions_left: int
    cleared: bool = False
    pills: int | None = None  # placements from onset up to and including the clearing one
    frames: int | None = None
    support_destroying: int = 0
    clears: int = 0  # placements during the episode that removed any tile
    max_gap: int = 0
    gap_at_clear: int | None = None
    open_at_clear: int | None = None
    censored: bool = False  # the game ended with the virus still on the board
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(out.pop("extra"))
        return out


def _occupied_count(g: np.ndarray) -> int:
    return int((g != EMPTY).sum())


def track(boards: Sequence, frames: Sequence[int] | None = None, definition: Definition = Definition(), *,
          cleared_out: bool = False, end_frame: int | None = None, placed: Sequence[int] | None = None
          ) -> list[Episode]:
    """Stranded-virus episodes over one side's decision-time bottles.

    ``boards[k]`` is the settled bottle when decision ``k`` is taken.
    ``cleared_out`` says the side cleared its last virus with its final
    placement (the trace has no bottle after it); ``end_frame`` is then that
    moment. ``placed[k]`` (optional) is the number of tiles placement ``k``
    added, used to tell clears from garbage; it defaults to 2.
    """
    grids = [grid(b) for b in boards]
    n = len(grids)
    frames = list(frames) if frames is not None else list(range(n))
    active: dict[tuple[int, int, int], Episode] = {}
    episodes: list[Episode] = []
    seen: set[tuple[int, int, int]] = set()
    for k, g in enumerate(grids):
        for key, episode in list(active.items()):
            if virus_present(g, *key):
                continue
            episode.cleared, episode.pills = True, k - episode.onset
            episode.frames = int(frames[k]) - int(frames[episode.onset])
            del active[key]
        for v in stranded(g, definition):
            key = (v["row"], v["col"], v["color"])
            if key in seen:
                continue
            seen.add(key)
            episode = Episode(onset=k, row=v["row"], col=v["col"], color=v["color"], kind=v["kind"],
                              gap=v["gap"], open=v["open"], above=v["above"], viruses=v["viruses"],
                              other_cells=v["other_cells"], decisions_left=n - k, max_gap=v["gap"],
                              gap_at_clear=v["gap"], open_at_clear=v["open"])
            active[key] = episode
            episodes.append(episode)
        if k + 1 >= n:
            continue
        nxt = grids[k + 1]
        added = 2 if placed is None else int(placed[k])
        removed = _occupied_count(g) + added - _occupied_count(nxt)
        for key, episode in active.items():
            if not virus_present(nxt, *key):
                continue  # cleared by this placement; closed at k + 1
            row, col, _ = key
            before, after = build_zone(g, row, col), build_zone(nxt, row, col)
            if removed > 0:
                episode.clears += 1
            if after[0] > before[0] or after[1] > before[1]:
                episode.support_destroying += 1
            geometry = edge_geometry(nxt, row, col)
            episode.max_gap = max(episode.max_gap, geometry["gap"])
            episode.gap_at_clear, episode.open_at_clear = geometry["gap"], geometry["open"]
    for episode in active.values():
        if cleared_out:
            episode.cleared, episode.pills = True, n - episode.onset
            if end_frame is not None:
                episode.frames = int(end_frame) - int(frames[episode.onset])
        else:
            episode.censored = True
    return episodes


def follow(boards: Sequence, frames: Sequence[int], target: tuple[int, int, int], *, cleared_out: bool = False,
           end_frame: int | None = None) -> Episode:
    """Resolution of one known target virus ``(row, col, color)`` from decision 0 (benchmark starts)."""
    grids = [grid(b) for b in boards]
    row, col, color = map(int, target)
    if not grids or not virus_present(grids[0], row, col, color):
        raise ValueError("the target virus is not on the starting bottle")
    g0 = grids[0]
    geometry = edge_geometry(g0, row, col)
    count = int(((g0 & 0xF0) == VIRUS).sum())
    kind = "stranded" if geometry["gap"] >= Definition.min_gap else "pillar" if geometry["gap"] <= 1 \
        and geometry["open"] >= Definition.min_gap else "grounded"
    episode = Episode(onset=0, row=row, col=col, color=color, kind=kind, gap=geometry["gap"],
                      open=geometry["open"], above=geometry["above"], viruses=count,
                      other_cells=other_cells(g0, col),
                      decisions_left=len(grids), max_gap=geometry["gap"], gap_at_clear=geometry["gap"],
                      open_at_clear=geometry["open"])
    for k in range(1, len(grids) + 1):
        if k == len(grids):
            if cleared_out:
                episode.cleared, episode.pills = True, k
                if end_frame is not None:
                    episode.frames = int(end_frame) - int(frames[0])
            else:
                episode.censored = True
            break
        g, prev = grids[k], grids[k - 1]
        if not virus_present(g, row, col, color):
            episode.cleared, episode.pills = True, k
            episode.frames = int(frames[k]) - int(frames[0])
            break
        if _occupied_count(prev) + 2 - _occupied_count(g) > 0:
            episode.clears += 1
        before, after = build_zone(prev, row, col), build_zone(g, row, col)
        if after[0] > before[0] or after[1] > before[1]:
            episode.support_destroying += 1
        geometry = edge_geometry(g, row, col)
        episode.max_gap = max(episode.max_gap, geometry["gap"])
        episode.gap_at_clear, episode.open_at_clear = geometry["gap"], geometry["open"]
    return episode


def mirror_board(board) -> np.ndarray:
    """Left-right mirror of a settled bottle; horizontal pill halves swap left/right tiles."""
    g = grid(board)[:, ::-1].copy()
    kind = g & 0xF0
    left, right = kind == 0x60, kind == 0x70
    g[left] = 0x70 | (g[left] & 0x0F)
    g[right] = 0x60 | (g[right] & 0x0F)
    return g


def ascii_board(board, *, mark: Iterable[tuple[int, int]] = ()) -> str:
    """Viruses as upper-case R/Y/B, pill halves lower-case, empty as '.'; ``mark`` cells in brackets."""
    g = grid(board)
    marked = set(mark)
    lines = []
    for row in range(ROWS):
        cells = []
        for col in range(COLS):
            tile = int(g[row, col])
            if tile == EMPTY:
                ch = "."
            else:
                ch = _COLOR[tile & 3]
                ch = ch if (tile & 0xF0) == VIRUS else ch.lower()
            cells.append(f"[{ch}]" if (row, col) in marked else f" {ch} ")
        lines.append(f"{row:2d} |" + "".join(cells) + "|")
    lines.append("   +" + "---" * COLS + "+")
    lines.append("    " + "".join(f" {c} " for c in range(COLS)))
    return "\n".join(lines)
