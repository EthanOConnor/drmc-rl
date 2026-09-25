"""Search references for the stranded edge-virus benchmark (clairvoyant beam and public two-ply).

For each bank row, search the row's actual pill stream (falling, preview, then
``reserve[(pill_counter + k) % 128]`` of its allocated seed) for the fewest
placements that clear the target virus. Placements are the exact planner
frontier at the declared pace and delay; every lock is settled with the
ROM cascade (``drmc_rl.game.afterstate``). There is no opponent, no garbage and
no hidden information: the result is an achievable upper bound on the true
minimum, a yardstick for how many pills a position needs, not a policy.

Beam ranking: the target's cheapest completable line (empty cells in a 4-window
through it plus the empty cells beneath each, blocked by other colors), then
fewer occupied cells, then lower stack height.

``--mode preview`` is the public-information counterpart: a receding two-ply
search over the falling pill and the preview only, played one pill at a time.

  python -m tools.oracle_stranded_edge --bank BANK --study STUDY --out ORACLE.jsonl [--beam 48 --horizon 24]
  python -m tools.oracle_stranded_edge --mode preview --horizon 40 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.eval.stranded_edge import EMPTY, VIRUS, grid
from drmc_rl.game.afterstate import resolve_placement
from drmc_rl.game.observation import board_bytes_to_semantic_planes

_CANON = (1, 0, 2)  # NES low nibble (Y, R, B) <-> canonical (R, Y, B); self-inverse


def pill_stream(bank, i: int, seed: int, horizon: int) -> list[tuple[int, int]]:
    from drmc_rl.search.pill_belief import pill_id_to_raw_pair
    from drmc_rl.seedlab.rng import generate_pill_reserve

    reserve, _, _ = generate_pill_reserve(seed & 0xFF, seed >> 8)
    counter = int(bank["pill_counter"][i][0])
    raw = [tuple(bank["falling"][i][0]), tuple(bank["preview"][i][0])]
    raw += [pill_id_to_raw_pair(reserve[(counter + k) % 128]) for k in range(2, horizon + 1)]
    return [tuple(_CANON[int(c) & 3] for c in pair) for pair in raw]


def line_cost(g: np.ndarray, row: int, col: int) -> float:
    color = int(g[row, col]) & 3
    best = 99.0
    windows = [[(row, c) for c in range(s, s + 4)] for s in range(col - 3, col + 1) if 0 <= s and s + 3 < 8]
    windows += [[(r, col) for r in range(s, s + 4)] for s in range(row - 3, row + 1) if 0 <= s and s + 3 < 16]
    for window in windows:
        cost = 0.0
        for r, c in window:
            tile = int(g[r, c])
            if tile == EMPTY:
                below = 0
                while r + 1 + below < 16 and g[r + 1 + below, c] == EMPTY:
                    below += 1
                cost += 1 + 0.5 * below
            elif (tile & 3) != color:
                cost = 99.0
                break
        best = min(best, cost)
    return best


def search(board, pills, target, planner, pace, delay, beam: int, speed_ups: int = 0) -> dict:
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates

    row, col, color = target
    frontier = [bytes(np.asarray(board, np.uint8).reshape(128))]
    nodes = 0
    for depth in range(len(pills) - 1):
        children = {}
        for b in frontier:
            planes = board_bytes_to_semantic_planes(b)
            state = dict(board_planes=planes, opponent_board_planes=planes, pill=list(pills[depth]),
                         preview=list(pills[depth + 1]), speed=2, speed_ups=speed_ups)
            try:
                costs = plan_candidates(planner, state, delay, pace)[-1]
            except NoReachablePlacement:
                continue
            for action in np.flatnonzero(costs != 0xFFFF):
                after, facts = resolve_placement(np.frombuffer(b, np.uint8), pills[depth], int(action))
                nodes += 1
                if facts[8]:  # spawn blocked: topped out
                    continue
                g = grid(after)
                tile = int(g[row, col])
                if not ((tile & 0xF0) == VIRUS and (tile & 3) == color):
                    return dict(pills=depth + 1, nodes=nodes)
                if after not in children:
                    occupied = int((g != EMPTY).sum())
                    height = 16 - int(np.argmax((g != EMPTY).any(axis=1))) if occupied else 0
                    children[after] = (line_cost(g, row, col), occupied, height)
        if not children:
            return dict(pills=None, nodes=nodes, dead=True)
        frontier = [b for b, _ in sorted(children.items(), key=lambda kv: kv[1])[:beam]]
    return dict(pills=None, nodes=nodes)


def _children(b: bytes, pill, preview, target, planner, pace, delay, speed_ups, whole_round: bool = False):
    """Settled afterstates of every feasible placement: (action, after, done, key, target_gone).

    ``done`` is the target gone, or with ``whole_round`` every virus gone, and
    the key then sums the line cost of every remaining virus.
    """
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates

    row, col, color = target
    planes = board_bytes_to_semantic_planes(b)
    state = dict(board_planes=planes, opponent_board_planes=planes, pill=list(pill), preview=list(preview),
                 speed=2, speed_ups=speed_ups)
    try:
        costs = plan_candidates(planner, state, delay, pace)[-1]
    except NoReachablePlacement:
        return []
    out = []
    for action in np.flatnonzero(costs != 0xFFFF):
        after, facts = resolve_placement(np.frombuffer(b, np.uint8), pill, int(action))
        if facts[8]:
            continue
        g = grid(after)
        tile = int(g[row, col])
        gone = not ((tile & 0xF0) == VIRUS and (tile & 3) == color)
        occupied = int((g != EMPTY).sum())
        height = 16 - int(np.argmax((g != EMPTY).any(axis=1))) if occupied else 0
        if whole_round:
            viruses = [tuple(v) for v in np.argwhere((g & 0xF0) == VIRUS)]
            done, cost = not viruses, sum(line_cost(g, r, c) for r, c in viruses)
        else:
            done, cost = gone, 0.0 if gone else line_cost(g, row, col)
        out.append((int(action), after, done, (cost, occupied, height), gone))
    return out


def receding(board, pills, target, planner, pace, delay, speed_ups: int = 0, whole_round: bool = False) -> dict:
    """Public-information reference: two-ply search over the falling pill and preview only.

    Each step picks the placement whose best preview follow-up clears the
    target soonest (else has the cheapest target line), plays it, and moves on
    with the next actual pill. No future reserve pills are used.
    """
    b = bytes(np.asarray(board, np.uint8).reshape(128))
    target_pills = None
    for depth in range(len(pills) - 1):
        first = _children(b, pills[depth], pills[depth + 1], target, planner, pace, delay, speed_ups, whole_round)
        if not first:
            return dict(pills=None, target_pills=target_pills, dead=True)
        done = [c for c in first if c[2]]
        if done:
            return dict(pills=depth + 1, target_pills=target_pills or depth + 1)
        best = None
        for action, after, _, key, gone in first:
            nxt = pills[depth + 2] if depth + 2 < len(pills) else pills[depth + 1]
            follow = _children(after, pills[depth + 1], nxt, target, planner, pace, delay, speed_ups, whole_round)
            score = min([(0 if c[2] else 1, c[3]) for c in follow], default=(2, key))
            score = (score, key)
            if best is None or score < best[0]:
                best = (score, after, gone)
        b = best[1]
        if best[2] and target_pills is None:
            target_pills = depth + 1
    return dict(pills=None, target_pills=target_pills)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beam", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pace", default="frame_perfect")
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--mode", choices=("clairvoyant", "preview", "preview-round"), default="clairvoyant",
                        help="clairvoyant beam over the known reserve, or public two-ply receding search "
                             "for the target (preview) or for the whole bottle (preview-round)")
    args = parser.parse_args()
    from drmc_rl.execution.pace import resolve_pace
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from drmc_rl.program.seed_reserve import allocated_seeds

    data = np.load(args.bank, allow_pickle=False)
    bank = {k: data[k] for k in data.files}
    seeds = allocated_seeds(args.study)
    pace = resolve_pace(args.pace)
    planner = NativeReachabilityRunner()
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["row"] for l in out.read_text().splitlines()}
    started = time.time()
    with out.open("a") as sink:
        for i in range(len(bank["boards"])):
            if i in done:
                continue
            seed = int(seeds[int(bank["group"][i])])
            pills = pill_stream(bank, i, seed, args.horizon)
            target = tuple(int(v) for v in bank["target"][i])
            delay = max(args.delay, pace.reaction_frames)
            if args.mode == "clairvoyant":
                result = search(bank["boards"][i][0].reshape(128), pills, target, planner, pace, delay, args.beam,
                                int(bank["speed_ups"][i][0]))
            else:
                result = receding(bank["boards"][i][0].reshape(128), pills, target, planner, pace, delay,
                                  int(bank["speed_ups"][i][0]), whole_round=args.mode == "preview-round")
            sink.write(json.dumps(dict(row=i, seed=seed, stratum=int(bank["stratum"][i]), mode=args.mode, beam=args.beam,
                                       horizon=args.horizon, pace=args.pace, **result)) + "\n")
            sink.flush()
            if i % 20 == 0:
                print(f"row {i}: {result} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
