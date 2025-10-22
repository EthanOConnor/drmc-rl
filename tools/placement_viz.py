#!/usr/bin/env python3
"""Interactive CLI visualiser for placement planner outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from envs.retro.placement_actions import PLACEMENT_EDGES
from envs.retro.placement_planner import BoardState, PillSnapshot, PlacementPlanner

ARROW_CHARS = {"U": "^", "D": "v", "L": "<", "R": ">"}


def parse_board_file(board_path: Optional[str]) -> List[str]:
    if board_path is None:
        return ["........" for _ in range(16)]
    path = Path(board_path)
    rows = [line.rstrip("\n") for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 16:
        raise ValueError("Board description must contain exactly 16 rows")
    for row in rows:
        if len(row) != 8:
            raise ValueError("Each board row must contain exactly 8 characters")
    return rows


def board_from_rows(rows: List[str]) -> BoardState:
    columns = np.zeros(8, dtype=np.uint16)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                columns[c] |= 1 << r
    return BoardState(columns=columns)


def parse_pill(spec: str, gravity_period: Optional[int], lock_counter: Optional[int]) -> PillSnapshot:
    parts = [int(x.strip()) for x in spec.split(",")]
    if len(parts) != 6:
        raise ValueError("Pill spec must be row,col,orient,left_color,right_color,gravity")
    row, col, orient, left_color, right_color, gravity = parts
    period = gravity_period if gravity_period is not None else gravity
    lock_val = lock_counter if lock_counter is not None else 0
    return PillSnapshot(
        row=row,
        col=col,
        orient=orient,
        colors=(left_color, right_color),
        gravity_counter=gravity,
        gravity_period=period,
        lock_counter=lock_val,
        spawn_id=1,
    )


def render_board(rows: List[str]) -> str:
    return "\n".join("".join(row) for row in rows)


def overlay_arrows(rows: List[str], mask: np.ndarray) -> List[str]:
    grid = [list(row) for row in rows]
    for edge in PLACEMENT_EDGES:
        if not mask[edge.index]:
            continue
        r, c = edge.origin
        arrow = ARROW_CHARS.get(edge.direction, "*")
        grid[r][c] = arrow
    return ["".join(row) for row in grid]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualise placement planner output")
    parser.add_argument("--board", type=str, default=None, help="Path to ASCII board description (16x8)")
    parser.add_argument(
        "--pill",
        type=str,
        default="0,3,0,1,0,6",
        help="row,col,orient,left_color,right_color,gravity counter",
    )
    parser.add_argument("--gravity-period", type=int, default=None, help="Override gravity period")
    parser.add_argument("--lock-counter", type=int, default=None, help="Override lock counter")
    args = parser.parse_args(argv)

    rows = parse_board_file(args.board)
    board = board_from_rows(rows)
    pill = parse_pill(args.pill, args.gravity_period, args.lock_counter)

    planner = PlacementPlanner()
    out = planner.plan_all(board, pill)

    legal_count = int(out.legal_mask.sum())
    feasible_count = int(out.feasible_mask.sum())

    print('Board ("#" = occupied):')
    print(render_board(rows))
    print()
    print(f"Legal placements: {legal_count}")
    print(f"Feasible placements: {feasible_count}")
    print()
    print("Feasible placement directions (arrows mark half-0 destination):")
    overlay = overlay_arrows(rows, out.feasible_mask)
    print(render_board(overlay))
    print()

    if out.plan_count:
        ranked = sorted(out.plans, key=lambda plan: plan.cost)
        print("Top 5 placements by frame cost:")
        for plan in ranked[:5]:
            edge = PLACEMENT_EDGES[plan.action]
            print(
                f"  idx={plan.action:3d} origin={edge.origin} dir={edge.direction} "
                f"cost={plan.cost} frames={len(plan.controller)}"
            )
    else:
        print("No feasible placements found for the provided configuration.")


if __name__ == "__main__":
    main()
