"""Start banks of stranded edge-virus endgames (StartBank npz schema).

Positions come from ``tools.mine_stranded_edge`` episode files: the onset
bottle of each stranded-virus episode, from strong human games and from arena
games. Both sides of a row get the same bottle and the same falling/preview
pill, so a game from the row is a race from identical positions; the reset
seed supplies the pill stream after the preview.

``benchmark``: a fixed, deterministic selection (per source and kind, each
position also mirrored left-right so both edges are covered). Evaluation
seeds are *not* stored here; ``tools.eval_stranded_edge`` takes them from a
recorded evaluation-reserve allocation.

``training``: every eligible position plus generated variants (left-right
mirror, global color permutation, the stranded virus moved within its shaft,
the neighbour stack trimmed), settled and checked for standing lines. Training
seeds are drawn from ``training_seed_pool`` at reset time, never here.

  python -m tools.build_stranded_edge_bank benchmark --episodes A.jsonl.gz B.jsonl.gz --out bank.npz
  python -m tools.build_stranded_edge_bank training --episodes ... --out train.npz --variants 8
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from drmc_rl.eval.stranded_edge import (
    COLS, EMPTY, ROWS, SCHEMA, VIRUS, Definition, edge_geometry, grid, mirror_board, other_cells, stranded,
)
from drmc_rl.game import cascade

BANK_SCHEMA = "drmc-stranded-edge-bank-v1"
KINDS = ("stranded", "pillar", "grounded")  # grounded: benchmark control twin
_CANONICAL_TO_NES = (1, 0, 2)
# Level 14 viruses occupy rows 6..15 (ROM virus height table); variants stay inside it.
TOP_VIRUS_ROW = 6


def _load(paths, *, level: int, min_rating: float, arena_variants: set[str] | None):
    rows = []
    for path in paths:
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if r.get("level") != level or r["kind"] not in KINDS[:2]:
                continue
            if r["source"] == "corpus":
                if r.get("rating") is None or r["rating"] < min_rating:
                    continue
                pill, preview = r["pill"], r["preview"]  # raw NES colors
                ordinal = None
            else:
                if arena_variants and r["variant"] not in arena_variants:
                    continue
                pill = [_CANONICAL_TO_NES[c] for c in r["pill"]]
                preview = [_CANONICAL_TO_NES[c] for c in r["preview"]]
                ordinal = int(r["onset"])
            rows.append(dict(board=np.asarray(r["board"], np.uint8).reshape(ROWS, COLS),
                             pill=[int(c) & 3 for c in pill], preview=[int(c) & 3 for c in preview],
                             speed_ups=int(r.get("speed_ups", 0)), ordinal=ordinal,
                             target=(int(r["row"]), int(r["col"]), int(r["color"])), kind=r["kind"],
                             source=r["source"], origin=_origin(r), seed=r.get("seed")))
    return rows


def _origin(r) -> str:
    if r["source"] == "corpus":
        return f"corpus:{r['game_id']}:{r['slot']}:{r['frame']}"
    return f"arena:{r['run']}:{r['comparison']}:{r['index']}:{r['frame']}"


def _game(origin: str) -> str:
    """Source game of a position origin (drops the frame and any mirror/variant suffix)."""
    parts = origin.split(":")
    return ":".join(parts[:3] if parts[0] == "corpus" else parts[:4])


def spawn_safe(g: np.ndarray) -> bool:
    return bool((g[:2, 2:6] == EMPTY).all())


def standing_lines(g: np.ndarray) -> bool:
    colors = np.where(g == EMPTY, -1, (g & 3).astype(np.int16))
    for r in range(ROWS):
        for c in range(COLS - 3):
            if colors[r, c] >= 0 and (colors[r, c:c + 4] == colors[r, c]).all():
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if colors[r, c] >= 0 and (colors[r:r + 4, c] == colors[r, c]).all():
                return True
    return False


def settle(g: np.ndarray) -> np.ndarray | None:
    board = bytearray(g.reshape(-1).tobytes())
    while cascade._drop_pass(board):
        pass
    out = np.frombuffer(bytes(board), np.uint8).reshape(ROWS, COLS).copy()
    return None if standing_lines(out) else out


def eligible(g: np.ndarray, target) -> bool:
    row, col, color = target
    tile = int(g[row, col])
    if (tile & 0xF0) != VIRUS or (tile & 3) != color or not spawn_safe(g) or standing_lines(g):
        return False
    return any((v["row"], v["col"]) == (row, col) for v in stranded(g))


def recolor(g: np.ndarray, perm) -> np.ndarray:
    out = g.copy()
    filled = out != EMPTY
    out[filled] = (out[filled] & 0xFC) | np.asarray(perm, np.uint8)[out[filled] & 3]
    return out


def _orphan(g: np.ndarray, r: int, c: int) -> None:
    """Turn the partner of a removed half into a single tile."""
    kind = int(g[r, c]) & 0xF0
    partner = {0x40: (r + 1, c), 0x50: (r - 1, c), 0x60: (r, c + 1), 0x70: (r, c - 1)}.get(kind)
    g[r, c] = EMPTY
    if partner and 0 <= partner[0] < ROWS and 0 <= partner[1] < COLS:
        pr, pc = partner
        if g[pr, pc] != EMPTY and (int(g[pr, pc]) & 0xF0) in (0x40, 0x50, 0x60, 0x70):
            g[pr, pc] = 0x80 | (g[pr, pc] & 3)


def grounded_twin(position: dict) -> dict | None:
    """The matched control: the virus and everything stacked on it lowered onto its support.

    The edge column from the top down to the virus drops by the shaft height,
    so the virus rests on its support and any junk above it keeps resting on
    the virus. Horizontal pills that straddled the edge column and its
    neighbour are split into single halves first; the bottle is then settled
    and rejected if it holds a standing line.
    """
    g = position["board"].copy()
    row, col, color = position["target"]
    gap = edge_geometry(g, row, col)["gap"]
    neighbour = 1 if col == 0 else COLS - 2
    for r in range(row + 1):
        if (int(g[r, col]) & 0xF0) in (0x60, 0x70):
            g[r, col] = 0x80 | (g[r, col] & 3)
            if (int(g[r, neighbour]) & 0xF0) in (0x60, 0x70):
                g[r, neighbour] = 0x80 | (g[r, neighbour] & 3)
    segment = g[:row + 1, col].copy()
    g[:row + 1, col] = EMPTY
    g[gap:row + 1 + gap, col] = segment
    g = settle(g)
    if g is None or not spawn_safe(g):
        return None
    target = (row + gap, col, color)
    if not virus_present_at(g, target):
        return None
    return dict(position, board=g, target=target, kind="grounded", origin=position["origin"] + ":grounded")


def virus_present_at(g: np.ndarray, target) -> bool:
    row, col, color = target
    tile = int(g[row, col])
    return (tile & 0xF0) == VIRUS and (tile & 3) == color


def variants(position: dict, rng: np.random.Generator, count: int) -> list[dict]:
    """Perturbed copies: mirror, color permutation, virus moved in its shaft, neighbour stack trimmed."""
    out = []
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for _ in range(count * 4):
        if len(out) >= count:
            break
        g = position["board"].copy()
        row, col, color = position["target"]
        pill, preview = list(position["pill"]), list(position["preview"])
        if rng.random() < 0.5:
            g, col = mirror_board(g), COLS - 1 - col
        perm = perms[int(rng.integers(len(perms)))]
        g, color = recolor(g, perm), perm[color]
        pill, preview = [perm[c] for c in pill], [perm[c] for c in preview]
        if position["kind"] == "stranded" and rng.random() < 0.6:
            geometry = edge_geometry(g, row, col)
            low = max(TOP_VIRUS_ROW, row - 2)
            high = geometry["support_row"] - 1 - Definition.min_gap
            options = [r for r in range(low, high + 1) if r != row
                       and (g[min(r, row + 1):max(r + 1, row), col] == EMPTY).all()]
            if options:
                new_row = int(rng.choice(options))
                g[new_row, col], g[row, col] = g[row, col], EMPTY
                row = new_row
        if rng.random() < 0.4:
            neighbour = 1 if col == 0 else COLS - 2
            filled = np.flatnonzero((g[:, neighbour] != EMPTY) & ((g[:, neighbour] & 0xF0) != VIRUS))
            if filled.size:
                top = int(filled.min())
                _orphan(g, top, neighbour)
        g = settle(g)
        if g is None or not eligible(g, (row, col, color)):
            continue
        out.append(dict(position, board=g, target=(row, col, color), pill=pill, preview=preview,
                        origin=position["origin"] + ":variant"))
    return out


def to_arrays(positions: list[dict], *, level: int, speed: int) -> dict[str, np.ndarray]:
    n = len(positions)
    boards = np.stack([np.stack([p["board"], p["board"]]) for p in positions]).astype(np.uint8)
    pill = np.asarray([[p["pill"], p["pill"]] for p in positions], np.uint8)
    preview = np.asarray([[p["preview"], p["preview"]] for p in positions], np.uint8)
    ordinal = np.asarray([min(127, p["ordinal"]) if p["ordinal"] is not None else 64 for p in positions], np.uint8)
    speed_ups = np.asarray([p["speed_ups"] for p in positions], np.uint8)
    targets = np.asarray([p["target"] for p in positions], np.int16).reshape(n, 3)
    kind = np.asarray([KINDS.index(p["kind"]) for p in positions], np.uint8)
    return dict(
        boards=boards, falling=pill, preview=preview, pill_counter=np.stack([ordinal, ordinal], 1),
        speed_ups=np.stack([speed_ups, speed_ups], 1),
        # 0/1 stranded left/right, 2/3 pillar, 4/5 grounded control twin
        stratum=(kind * 2 + (targets[:, 1] == COLS - 1)).astype(np.uint8),
        group=np.asarray([p.get("group", -1) for p in positions], np.int32),
        levels=np.full((n, 2), level, np.uint8), speeds=np.full((n, 2), speed, np.uint8),
        target=targets, kind=kind, mirror=np.asarray([p.get("mirror", 0) for p in positions], np.uint8),
        source=np.asarray([p["source"] for p in positions]), origin=np.asarray([p["origin"] for p in positions]),
        gap=np.asarray([edge_geometry(p["board"], *p["target"][:2])["gap"] for p in positions], np.uint8),
        open=np.asarray([edge_geometry(p["board"], *p["target"][:2])["open"] for p in positions], np.uint8),
        viruses=np.asarray([int(((p["board"] & 0xF0) == VIRUS).sum()) for p in positions], np.uint8),
        other=np.asarray([other_cells(p["board"], p["target"][1]) for p in positions], np.uint8),
    )


def pool_check(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Rows that load exactly into the native VS pool with both sides live and able to move."""
    from tools.build_clear_endgame_bank import pool_load_mask

    return pool_load_mask(dict(arrays, spawn_f=np.zeros(len(arrays["boards"]), np.int64)))


def _unique(positions):
    seen, out = set(), []
    for p in positions:
        key = p["board"].tobytes() + bytes(p["pill"]) + bytes(p["preview"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("benchmark", "training"))
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", type=int, default=14)
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--min-rating", type=float, default=2000.0)
    parser.add_argument("--arena-variants", nargs="*", default=None)
    parser.add_argument("--per-source", type=int, nargs=2, default=(40, 8), metavar=("STRANDED", "PILLAR"),
                        help="benchmark: positions per source (corpus, arena) for each kind, before mirroring")
    parser.add_argument("--variants", type=int, default=8, help="training: generated variants per position")
    parser.add_argument("--exclude-bank", nargs="*", default=(), help="training: drop positions in these banks")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--skip-pool-check", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    raw = _load(args.episodes, level=args.level, min_rating=args.min_rating,
                arena_variants=set(args.arena_variants) if args.arena_variants else None)
    positions = _unique([p for p in raw if eligible(p["board"], p["target"])])
    print(f"{len(raw)} episodes -> {len(positions)} eligible unique positions")
    if args.mode == "benchmark":
        groups = defaultdict(list)
        for p in positions:
            groups[(p["source"], p["kind"])].append(p)
        chosen = []
        for (source, kind), items in sorted(groups.items()):
            want = args.per_source[KINDS.index(kind)]
            # One position per source game, so no game dominates the bank.
            by_game = {}
            for p in items:
                by_game.setdefault(_game(p["origin"]), p)
            pool = sorted(by_game.values(), key=lambda p: hashlib.sha256(p["origin"].encode()).hexdigest())
            picked = pool[:want]
            print(f"{source}/{kind}: {len(items)} positions, {len(by_game)} games, picked {len(picked)}")
            chosen.extend(picked)
        # One group per source position: its mirror and grounded twins share the group's seed.
        final = []
        for group, p in enumerate(chosen):
            row, col, color = p["target"]
            pair = [dict(p, mirror=0, group=group),
                    dict(p, board=mirror_board(p["board"]), target=(row, COLS - 1 - col, color),
                         mirror=1, origin=p["origin"] + ":mirror", group=group)]
            final.extend(pair)
            if p["kind"] == "stranded":
                final.extend(t for t in map(grounded_twin, pair) if t is not None)
    else:
        boards, games = set(), set()
        for bank in args.exclude_bank:
            data = np.load(bank, allow_pickle=False)
            boards |= {b[0].tobytes() for b in data["boards"]}
            games |= {_game(o) for o in data["origin"]}
        from drmc_rl.program.seed_reserve import load_reserve

        blocked = load_reserve().blocked  # arena positions from reserve games stay out of training
        base = [p for p in positions if p["board"].tobytes() not in boards and _game(p["origin"]) not in games
                and p["seed"] not in blocked]
        print(f"training: {len(base)} base positions after excluding benchmark banks")
        final = []
        for p in base:
            final.append(dict(p, mirror=0))
            final.extend(dict(v, mirror=0) for v in variants(p, rng, args.variants))
        final = _unique(final)
    arrays = to_arrays(final, level=args.level, speed=args.speed)
    if not args.skip_pool_check:
        ok = pool_check(arrays)
        if args.mode == "benchmark":  # keep each group whole
            bad = set(arrays["group"][~ok].tolist())
            ok = ~np.isin(arrays["group"], list(bad))
        print(f"pool check dropped {int((~ok).sum())} rows")
        arrays = {k: v[ok] for k, v in arrays.items()}
    if args.mode == "benchmark":
        arrays["group"] = np.unique(arrays["group"], return_inverse=True)[1].astype(np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = dict(schema=BANK_SCHEMA, detector=SCHEMA, definition=Definition().to_dict(), mode=args.mode,
                rows=int(len(arrays["boards"])), level=args.level, speed=args.speed, min_rating=args.min_rating,
                arena_variants=args.arena_variants, episodes=[str(p) for p in args.episodes], seed=args.seed,
                strata={s: int((arrays["stratum"] == i).sum()) for i, s in enumerate(
                    ("stranded_left", "stranded_right", "pillar_left", "pillar_right",
                     "grounded_left", "grounded_right"))},
                groups=int(len(set(arrays["group"].tolist()))),
                sources={s: int((arrays["source"] == s).sum()) for s in ("corpus", "arena")},
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
