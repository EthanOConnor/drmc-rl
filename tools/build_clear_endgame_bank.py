"""Build a start bank from real human clear-endgame states.

The Go-Exploit bank (tools/build_start_bank.py) samples mid-game positions
whose strata stop well before end-of-game clearing sequences, and
tools/build_clear_practice_bank.py fakes near-clear boards by thinning
viruses. This tool extracts the REAL thing from the fightcadeRatings corpus:
spawn events where one side is genuinely 1..8 viruses from curing out while
the opponent is still alive (>= 1 virus). Same npz schema as the Go-Exploit
bank, so drmc_rl/training/envs/start_bank.StartBank loads it unchanged.

Selection (per game, per side, capped at --per-game-side positions):
  - sampled side's spawn snapshot has 1..8 virus tiles ((tile & 0xF0) == 0xD0)
  - opponent still has >= 1 virus (game live both ways)
  - partner board = its latest spawn snapshot at f' <= f (as build_start_bank)
  - both sides spawn-safe (top 2 rows empty) and load cleanly into the pool

Side balance: the learner is always side 0 in opponent-pool training, so each
accepted position is emitted twice — as-is AND side-swapped (boards, colors,
counters swapped) — teaching both closing-out and defending. The swapped rows
are pool-validated too (a row is dropped with its mirror if either fails).

Strata (tagged per row; closer side = min virus count of the two):
  0 v1_2   closer side has 1-2 viruses
  1 v3_5   closer side has 3-5 viruses
  2 v6_8   closer side has 6-8 viruses

Usage:
  nice -n 19 .venv/bin/python -m tools.build_clear_endgame_bank \
      [--out runs/start_bank/clear_endgame_v1.npz] [--max-rows 40000] \
      [--max-quarks N] [--validate 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.annotate_replay_events import decode_field, parse_quark_events
from tools.build_clear_practice_bank import spawn_safe, virus_count
from tools.build_start_bank import (
    FCR_ROOT,
    GRID_H,
    GRID_W,
    REPO_ROOT,
    positions_to_arrays,
    validate_bank,
)

CLEAR_STRATA = ("v1_2", "v3_5", "v6_8")

_ROW_KEYS = ("boards", "falling", "preview", "pill_counter", "speed_ups", "levels", "speeds")


def _stratum_of(closer_viruses: int) -> int:
    if closer_viruses <= 2:
        return 0
    if closer_viruses <= 5:
        return 1
    return 2


def mirror_position(pos: Dict[str, Any]) -> Dict[str, Any]:
    """Side-swapped copy of a position dict: boards, colors and counters swap
    sides; scalar fields (stratum, spawn_f, game_idx) are shared."""

    out = dict(pos)
    for key in _ROW_KEYS + ("virus_rem",):
        out[key] = np.ascontiguousarray(pos[key][::-1])
    return out


def extract_clear_endgame_positions(
    raw: bytes,
    *,
    per_game_side: int = 4,
    min_virus: int = 1,
    max_virus: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Near-clear two-board positions from one quark's v2 event blob.

    Returns pre-mirror dicts with the build_start_bank row keys plus
    ``virus_rem`` (2,). Deterministic for a fixed blob + rng seed. Scans each
    side's spawns backwards and stops once its virus count exceeds
    ``max_virus`` (virus counts only ever decrease within a game).
    """

    if rng is None:
        rng = np.random.default_rng(0)
    events = parse_quark_events(raw)
    if not events["spawn"]:
        return []

    timeline: List[tuple] = []
    for ev in events["init"]:
        timeline.append((int(ev["f"]), 0, ev))
    for ev in events["spawn"]:
        timeline.append((int(ev["f"]), 1, ev))
    timeline.sort(key=lambda t: (t[0], t[1]))

    out: List[Dict[str, Any]] = []
    game_idx = -1
    game: Optional[Dict[str, Any]] = None

    def flush(game: Optional[Dict[str, Any]]) -> None:
        if game is None:
            return
        spawns = game["spawns"]
        for p in (1, 2):
            own_idx = [k for k, sp in enumerate(spawns) if int(sp["p"]) == p]
            candidates: List[tuple] = []  # (k, stratum, own_field, par_field, vc, par_vc)
            for k in reversed(own_idx):
                own_field = decode_field(spawns[k]["field"])
                vc = int(virus_count(own_field))
                if vc > max_virus:
                    break  # earlier spawns of this side have >= vc viruses
                if vc < min_virus:
                    continue
                partner = game["partner_at"][k]
                if partner is None:
                    continue
                par_field = decode_field(partner["field"])
                par_vc = int(virus_count(par_field))
                if par_vc < 1:
                    continue
                if not (spawn_safe(own_field) and spawn_safe(par_field)):
                    continue
                stratum = _stratum_of(min(vc, par_vc))
                candidates.append((k, stratum, own_field, par_field, vc, par_vc))
            if not candidates:
                continue
            # Pick up to per_game_side samples, preferring stratum coverage.
            chosen: List[tuple] = []
            by_stratum: Dict[int, List[tuple]] = {}
            for item in candidates:
                by_stratum.setdefault(item[1], []).append(item)
            for s in range(len(CLEAR_STRATA)):
                if s in by_stratum and len(chosen) < per_game_side:
                    items = by_stratum[s]
                    chosen.append(items[int(rng.integers(0, len(items)))])
            while len(chosen) < per_game_side and candidates:
                item = candidates[int(rng.integers(0, len(candidates)))]
                if item[0] not in [c[0] for c in chosen]:
                    chosen.append(item)
                else:
                    break
            for k, stratum, own_field, par_field, vc, par_vc in chosen:
                sp = spawns[k]
                partner = game["partner_at"][k]
                boards = np.full((2, GRID_H, GRID_W), 0xFF, dtype=np.uint8)
                falling = np.zeros((2, 2), dtype=np.uint8)
                preview = np.zeros((2, 2), dtype=np.uint8)
                counters = np.zeros((2,), dtype=np.uint8)
                speedups = np.zeros((2,), dtype=np.uint8)
                virus_rem = np.zeros((2,), dtype=np.uint8)
                for side, ev, field in ((p - 1, sp, own_field), (2 - p, partner, par_field)):
                    boards[side] = field
                    falling[side] = [int(ev["pill"][0]) & 0x03, int(ev["pill"][1]) & 0x03]
                    preview[side] = [int(ev["prev"][0]) & 0x03, int(ev["prev"][1]) & 0x03]
                    speedups[side] = int(ev["spdups"]) & 0xFF
                counters[p - 1] = min(127, game["ordinal_at"][k] - 1)
                counters[2 - p] = min(127, game["partner_ordinal_at"][k] - 1)
                virus_rem[p - 1] = vc
                virus_rem[2 - p] = par_vc
                out.append(
                    {
                        "boards": boards,
                        "falling": falling,
                        "preview": preview,
                        "pill_counter": counters,
                        "speed_ups": speedups,
                        "levels": np.asarray(game["levels"], dtype=np.uint8),
                        "speeds": np.asarray(game["speeds"], dtype=np.uint8),
                        "stratum": stratum,
                        "spawn_f": int(sp["f"]),
                        "game_idx": game["idx"],
                        "virus_rem": virus_rem,
                    }
                )

    for _f, kind, ev in timeline:
        if kind == 0:  # init -> new game
            flush(game)
            game_idx += 1
            game = {
                "idx": game_idx,
                "levels": [int(ev["lvl"][0]), int(ev["lvl"][1])],
                "speeds": [int(ev["spd"][0]), int(ev["spd"][1])],
                "spawns": [],
                "ordinal_at": [],
                "partner_at": [],
                "partner_ordinal_at": [],
                "counts": {1: 0, 2: 0},
                "last_spawn": {1: None, 2: None},
            }
        elif game is None:
            continue
        else:  # spawn
            p = int(ev["p"])
            game["counts"][p] += 1
            game["spawns"].append(ev)
            game["ordinal_at"].append(game["counts"][p])
            game["partner_at"].append(game["last_spawn"][3 - p])
            game["partner_ordinal_at"].append(game["counts"][3 - p])
            game["last_spawn"][p] = ev
    flush(game)
    return out


def mirror_expand(
    positions: List[Dict[str, Any]], quark_idx: List[int]
) -> tuple[List[Dict[str, Any]], Dict[str, list]]:
    """Interleave each position with its side-swapped mirror: rows 2i / 2i+1
    are the same position in both orientations. Returns (rows, extra cols)."""

    rows: List[Dict[str, Any]] = []
    extra: Dict[str, list] = {"quark_idx": [], "mirror": [], "virus_rem": []}
    for qidx, pos in zip(quark_idx, positions):
        for m, r in ((0, pos), (1, mirror_position(pos))):
            rows.append(r)
            extra["quark_idx"].append(qidx)
            extra["mirror"].append(m)
            extra["virus_rem"].append(r["virus_rem"])
    return rows, extra


def _spec_for_row(arrays: Dict[str, np.ndarray], i: int, rng: np.random.Generator):
    from drmc_rl.envs.backends.drmario_vs_pool import build_vs_reset_spec

    return build_vs_reset_spec(
        level=(int(arrays["levels"][i, 0]), int(arrays["levels"][i, 1])),
        speed_setting=(int(arrays["speeds"][i, 0]), int(arrays["speeds"][i, 1])),
        rng_state=(int(rng.integers(0, 256)), int(rng.integers(0, 256))),
        rng_override=True,
        frame_counter_base=int(arrays["spawn_f"][i]),
        checkpoint_enabled=True,
        checkpoint_board=arrays["boards"][i].reshape(2, 128),
        checkpoint_falling_colors=arrays["falling"][i],
        checkpoint_preview_colors=arrays["preview"][i],
        checkpoint_pill_counter=tuple(int(v) for v in arrays["pill_counter"][i]),
        checkpoint_speed_ups=tuple(int(v) for v in arrays["speed_ups"][i]),
    )


def pool_load_mask(
    arrays: Dict[str, np.ndarray], *, batch_pairs: int = 64, seed: int = 0
) -> np.ndarray:
    """Per-row bool mask: the row loads cleanly into the native VS pool
    (board round-trips exactly, both sides parked at a live decision with
    feasible placements)."""

    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner

    n = arrays["boards"].shape[0]
    ok = np.zeros((n,), dtype=bool)
    rng = np.random.default_rng(seed)
    runner = DrMarioVsPoolRunner(num_pairs=batch_pairs)
    try:
        for lo in range(0, n, batch_pairs):
            idx = list(range(lo, min(lo + batch_pairs, n)))
            padded = idx + [idx[-1]] * (batch_pairs - len(idx))
            runner.reset(None, [_spec_for_row(arrays, i, rng) for i in padded])
            buf = runner.buffers
            boards = buf.board_bytes.reshape(batch_pairs, 2, 128)
            need = buf.need_action.reshape(batch_pairs, 2)
            feas = buf.feasible_mask.reshape(batch_pairs, 2, -1)
            for j, i in enumerate(idx):
                ok[i] = bool(
                    np.array_equal(boards[j], arrays["boards"][i].reshape(2, 128))
                    and need[j].all()
                    and (feas[j].sum(axis=1) > 0).all()
                )
    finally:
        runner.close()
    return ok


def _slice_rows(arrays: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = mask.shape[0]
    return {k: (v[mask] if v.shape[:1] == (n,) else v) for k, v in arrays.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(FCR_ROOT / "data" / "drmario.sqlite"))
    ap.add_argument("--fcr-root", default=str(FCR_ROOT))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "start_bank" / "clear_endgame_v1.npz"))
    ap.add_argument("--per-game-side", type=int, default=4)
    ap.add_argument("--min-virus", type=int, default=1)
    ap.add_argument("--max-virus", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=40000, help="cap on total rows (post-mirror)")
    ap.add_argument("--max-quarks", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=9000.0, help="scan budget in seconds (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", type=int, default=64, help="rows to play to completion (0 = skip)")
    args = ap.parse_args()

    import sqlite3

    sys.path.insert(0, str(Path(args.fcr_root)))
    import store  # fightcadeRatings/store.py

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    quarks = con.execute("SELECT quarkid, sha256 FROM processed_replay ORDER BY quarkid").fetchall()
    if args.max_quarks:
        quarks = quarks[: int(args.max_quarks)]

    rng = np.random.default_rng(int(args.seed))
    positions: List[Dict[str, Any]] = []
    pos_quark_idx: List[int] = []
    quark_names: List[str] = []
    t0 = time.time()
    deadline = t0 + float(args.time_budget) if args.time_budget else None
    for quarkid, sha in quarks:
        if deadline is not None and time.time() > deadline:
            print(f"time budget hit after {len(quark_names)} quarks; capping scan here")
            break
        try:
            raw = store.get_blob(sha)
        except FileNotFoundError:
            continue
        got = extract_clear_endgame_positions(
            raw,
            per_game_side=int(args.per_game_side),
            min_virus=int(args.min_virus),
            max_virus=int(args.max_virus),
            rng=rng,
        )
        if not got:
            continue
        quark_names.append(str(quarkid))
        qidx = len(quark_names) - 1
        positions.extend(got)
        pos_quark_idx.extend([qidx] * len(got))
        if len(quark_names) % 100 == 0:
            print(f"{len(quark_names)} quarks -> {len(positions)} positions ({time.time()-t0:.0f}s)")

    # Cap to --max-rows post-mirror rows (deterministic subsample of positions).
    max_positions = int(args.max_rows) // 2
    if len(positions) > max_positions:
        keep = np.sort(rng.choice(len(positions), size=max_positions, replace=False))
        positions = [positions[i] for i in keep]
        pos_quark_idx = [pos_quark_idx[i] for i in keep]
        print(f"subsampled to {len(positions)} positions (--max-rows {args.max_rows})")

    rows, extra = mirror_expand(positions, pos_quark_idx)
    arrays = positions_to_arrays(rows, extra)

    # Drop position pairs where either orientation fails to load in the pool.
    if rows:
        ok = pool_load_mask(arrays, seed=int(args.seed))
        pair_ok = ok.reshape(-1, 2).all(axis=1)
        row_keep = np.repeat(pair_ok, 2)
        dropped = int((~row_keep).sum())
        if dropped:
            print(f"pool load check dropped {dropped} rows ({int((~pair_ok).sum())} position pairs)")
        arrays = _slice_rows(arrays, row_keep)

    arrays["quark_names"] = np.asarray(quark_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    n_rows = int(arrays["boards"].shape[0])
    strata = {CLEAR_STRATA[s]: int((arrays["stratum"] == s).sum()) for s in range(len(CLEAR_STRATA))}
    print(f"wrote {out_path}: {n_rows} rows ({n_rows // 2} positions x2 mirror) "
          f"from {len(quark_names)} quarks, strata={strata}")

    if int(args.validate) > 0 and n_rows:
        half = max(1, int(args.validate) // 2)
        stats_orig = validate_bank(_slice_rows(arrays, np.asarray(arrays["mirror"] == 0)), n=half, seed=int(args.seed))
        stats_mirr = validate_bank(_slice_rows(arrays, np.asarray(arrays["mirror"] == 1)), n=half, seed=int(args.seed))
        print(f"validation (original rows): {stats_orig}")
        print(f"validation (mirrored rows): {stats_mirr}")
        meta = {
            "rows": n_rows,
            "quarks": len(quark_names),
            "strata": strata,
            "validation_original": stats_orig,
            "validation_mirrored": stats_mirr,
            "built_at": time.time(),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
