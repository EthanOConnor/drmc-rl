"""Clear-practice start bank: real boards, viruses thinned to a handful.

The Go-Exploit start bank (tools/build_start_bank.py) samples real human
mid-game positions, but those almost never sit near a clear (median 33-47
viruses even in the "late"/"crisis" strata; only 2.7% have a side with <=4
viruses). So a clear_win_bonus is never earned from them — the agent never
practices closing out a clear.

This bank fixes that: take real bank boards (realistic, often tall stacks)
and randomly THIN each side's viruses down to a small count, so a full clear
is genuinely a few good moves away. The board structure (pill debris, height)
is preserved — closing out a near-clear on a tall board is exactly the
endgame skill the attrition metagame never teaches. Same npz schema as the
Go-Exploit bank, so drmc_rl/training/envs/start_bank.StartBank loads it unchanged.

Usage:
  python -m tools.build_clear_practice_bank \
      [--src runs/start_bank/start_bank_v1.npz] \
      [--out runs/start_bank/clear_practice_v1.npz] \
      [--n 20000] [--kmin 2] [--kmax 8] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EMPTY = 0xFF
VIRUS_HI = 0xD0  # (board & 0xF0) == 0xD0 catches virus colors 0xD0/D1/D2


def virus_count(board_side: np.ndarray) -> np.ndarray:
    """Per-position virus count for a (..., 16, 8) tile array."""
    return ((board_side & 0xF0) == VIRUS_HI).sum(axis=(-2, -1))


def thin_viruses(board: np.ndarray, keep: int, rng: np.random.Generator) -> int:
    """In-place: keep `keep` random virus cells on a (16,8) board, empty the rest.

    Returns the resulting virus count (== min(keep, original count)).
    """
    vy, vx = np.where((board & 0xF0) == VIRUS_HI)
    n = len(vy)
    if n <= keep:
        return n
    drop = rng.permutation(n)[: n - keep]
    board[vy[drop], vx[drop]] = EMPTY
    return keep


def spawn_safe(board_side: np.ndarray) -> np.ndarray:
    """True where the top 2 rows of a (...,16,8) board are empty (spawn clear)."""
    top = board_side[..., :2, :]
    return (top == EMPTY).all(axis=(-2, -1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="runs/start_bank/start_bank_v1.npz")
    ap.add_argument("--out", default="runs/start_bank/clear_practice_v1.npz")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = np.load(args.src, allow_pickle=True)
    boards = src["boards"].copy()  # (N,2,16,8)
    N = boards.shape[0]

    # Keep only spawn-safe positions on BOTH sides (a thinned board is never
    # taller than the original, so this stays safe after thinning).
    safe = spawn_safe(boards[:, 0]) & spawn_safe(boards[:, 1])
    cand = np.flatnonzero(safe)
    rng.shuffle(cand)
    cand = cand[: args.n]

    out = {k: src[k][cand].copy() for k in src.files if src[k].shape[:1] == (N,)}
    out_boards = boards[cand]
    for row in range(len(cand)):
        for side in (0, 1):
            keep = int(rng.integers(args.kmin, args.kmax + 1))
            thin_viruses(out_boards[row, side], keep, rng)
    out["boards"] = out_boards
    # Relabel stratum to a sentinel so downstream can tell the source apart.
    if "stratum" in out:
        out["stratum"] = np.full(len(cand), 9, dtype=out["stratum"].dtype)
    for k in src.files:  # carry non-row-indexed metadata (e.g. quark_names)
        if k not in out:
            out[k] = src[k]

    vc = virus_count(out_boards)  # (n,2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(cand)} positions "
          f"(from {len(np.flatnonzero(safe))} spawn-safe of {N})")
    print(f"virus count per side: median {np.median(vc):.0f}, "
          f"range {vc.min()}-{vc.max()}, mean {vc.mean():.1f}")
    print(f"positions with a side at 0 viruses already: {(vc == 0).any(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
