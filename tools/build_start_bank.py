"""Go-Exploit start-state bank from real human VS positions (fightcadeRatings).

Samples mid-game two-board positions from the v2 replay events in
``../fightcadeRatings/data/drmario.sqlite`` (read-only) and stores them as
native VS-pool checkpoint reset specs (see ``DrmVsResetSpec`` checkpoint
fields / ``build_vs_reset_spec``): per side, the decision-time board, the
falling + preview pill colors, the pill spawn ordinal, and the speed-up
counter.

What is and is not reconstructible from the corpus:
  - Boards / pills / counters: exact, from the spawn records' field snapshots.
  - The two sides' snapshots are NOT from the same frame: the sampled side's
    board is at its spawn frame f; the partner board is its most recent spawn
    snapshot at f' <= f (boards only change between moves; the partner's
    in-flight pill respawns as its new falling pill). Skew is one move at most.
  - Mid-game RNG state is NOT reconstructible (the console LFSR advances every
    frame and is not in the event stream), so the bank does not store RNG;
    the loader randomizes it. Go-Exploit needs realistic boards, not exact
    continuations.
  - Stored-attack counters in flight are dropped (volleys release only
    between moves, so the boards already contain all released garbage).

Strata (tagged per row):
  0 early    spawn ordinal  8..20
  1 mid      spawn ordinal 21..45
  2 late     spawn ordinal 46+
  3 crisis   first spawn of the receiving side within `--crisis-window`
             frames after a garbage volley hit its board

Filters: both sides >= 1 virus remaining (game still live both ways) and the
sampled boards load cleanly into the native pool.

Usage:
  nice -n 19 .venv/bin/python -m tools.build_start_bank \
      [--out runs/start_bank/start_bank_v1.npz] [--per-game 4] [--validate 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FCR_ROOT = REPO_ROOT.parent / "fightcadeRatings"

from tools.annotate_replay_events import decode_field, parse_quark_events

GRID_H, GRID_W = 16, 8
STRATA = ("early", "mid", "late", "crisis")

_TILE_VIRUS = 0xD0
_MASK_TYPE = 0xF0


def _virus_count(field: np.ndarray) -> int:
    return int(((field & _MASK_TYPE) == _TILE_VIRUS).sum())


def _spawn_cells_free(field: np.ndarray) -> bool:
    """Pill spawn cells (top row, cols 3..4) must be empty or the side
    tops out immediately on reset."""

    return bool((field[0, 3] == 0xFF) and (field[0, 4] == 0xFF))


def _stratum_of(ordinal: int) -> int:
    if ordinal <= 20:
        return 0
    if ordinal <= 45:
        return 1
    return 2


def extract_positions_from_events(
    raw: bytes,
    *,
    per_game: int = 4,
    crisis_window: int = 90,
    min_ordinal: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Two-board mid-game positions from one quark's v2 event blob.

    Returns dicts with keys: boards (2,16,8), falling (2,2), preview (2,2),
    pill_counter (2,), speed_ups (2,), levels (2,), speeds (2,), stratum,
    spawn_f, game_idx. Deterministic for a fixed blob + rng seed.
    """

    if rng is None:
        rng = np.random.default_rng(0)
    events = parse_quark_events(raw)
    if not events["spawn"]:
        return []

    # Split into games on init records; keep per-game ordered spawn lists.
    timeline: List[tuple] = []
    for ev in events["init"]:
        timeline.append((int(ev["f"]), 0, ev))
    for ev in events["spawn"]:
        timeline.append((int(ev["f"]), 1, ev))
    for ev in events["grb"]:
        timeline.append((int(ev["f"]), 2, ev))
    timeline.sort(key=lambda t: (t[0], t[1]))

    out: List[Dict[str, Any]] = []
    game_idx = -1
    game: Optional[Dict[str, Any]] = None

    def flush(game: Optional[Dict[str, Any]]) -> None:
        if game is None:
            return
        candidates: List[tuple] = []  # (sample_key, spawn_idx_in_game, stratum)
        spawns = game["spawns"]
        # Crisis samples: receiver's first spawn after each volley.
        crisis_idx = set()
        for gf, receiver in game["volleys"]:
            for k, sp in enumerate(spawns):
                f = int(sp["f"])
                if f > gf and int(sp["p"]) == receiver:
                    if f - gf <= crisis_window:
                        crisis_idx.add(k)
                    break
        for k, sp in enumerate(spawns):
            p = int(sp["p"])
            ordinal = game["ordinal_at"][k]
            if ordinal < min_ordinal:
                continue
            # Partner must have a snapshot already.
            if game["partner_at"][k] is None:
                continue
            stratum = 3 if k in crisis_idx else _stratum_of(ordinal)
            candidates.append((k, stratum))
        if not candidates:
            return
        # Pick up to per_game samples, preferring stratum coverage.
        chosen: List[tuple] = []
        by_stratum: Dict[int, List[tuple]] = {}
        for item in candidates:
            by_stratum.setdefault(item[1], []).append(item)
        for s in (3, 0, 1, 2):  # crisis first, then early/mid/late
            if s in by_stratum and len(chosen) < per_game:
                items = by_stratum[s]
                chosen.append(items[int(rng.integers(0, len(items)))])
        while len(chosen) < per_game and candidates:
            item = candidates[int(rng.integers(0, len(candidates)))]
            if item not in chosen:
                chosen.append(item)
            else:
                break
        for k, stratum in chosen:
            sp = spawns[k]
            p = int(sp["p"])
            partner = game["partner_at"][k]
            own_field = decode_field(sp["field"])
            par_field = decode_field(partner["field"])
            if _virus_count(own_field) < 1 or _virus_count(par_field) < 1:
                continue
            if not (_spawn_cells_free(own_field) and _spawn_cells_free(par_field)):
                continue
            boards = np.full((2, GRID_H, GRID_W), 0xFF, dtype=np.uint8)
            falling = np.zeros((2, 2), dtype=np.uint8)
            preview = np.zeros((2, 2), dtype=np.uint8)
            counters = np.zeros((2,), dtype=np.uint8)
            speedups = np.zeros((2,), dtype=np.uint8)
            for side, ev in ((p - 1, sp), (2 - p, partner)):
                boards[side] = decode_field(ev["field"])
                falling[side] = [int(ev["pill"][0]) & 0x03, int(ev["pill"][1]) & 0x03]
                preview[side] = [int(ev["prev"][0]) & 0x03, int(ev["prev"][1]) & 0x03]
                speedups[side] = int(ev["spdups"]) & 0xFF
            counters[p - 1] = min(127, game["ordinal_at"][k] - 1)
            counters[2 - p] = min(127, game["partner_ordinal_at"][k] - 1)
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
                "volleys": [],
            }
        elif game is None:
            continue
        elif kind == 1:  # spawn
            p = int(ev["p"])
            game["counts"][p] += 1
            game["spawns"].append(ev)
            game["ordinal_at"].append(game["counts"][p])
            game["partner_at"].append(game["last_spawn"][3 - p])
            game["partner_ordinal_at"].append(game["counts"][3 - p])
            game["last_spawn"][p] = ev
        else:  # grb volley: {"f":N,"grb":player,...}
            game["volleys"].append((int(ev["f"]), int(ev["grb"])))
    flush(game)
    return out


def positions_to_arrays(rows: List[Dict[str, Any]], extra: Optional[Dict[str, list]] = None) -> Dict[str, np.ndarray]:
    n = len(rows)
    arrays = {
        "boards": np.zeros((n, 2, GRID_H, GRID_W), dtype=np.uint8),
        "falling": np.zeros((n, 2, 2), dtype=np.uint8),
        "preview": np.zeros((n, 2, 2), dtype=np.uint8),
        "pill_counter": np.zeros((n, 2), dtype=np.uint8),
        "speed_ups": np.zeros((n, 2), dtype=np.uint8),
        "levels": np.zeros((n, 2), dtype=np.uint8),
        "speeds": np.zeros((n, 2), dtype=np.uint8),
        "stratum": np.zeros((n,), dtype=np.uint8),
        "spawn_f": np.zeros((n,), dtype=np.uint32),
    }
    for i, r in enumerate(rows):
        for key in ("boards", "falling", "preview", "pill_counter", "speed_ups", "levels", "speeds", "stratum", "spawn_f"):
            arrays[key][i] = r[key]
    for key, values in (extra or {}).items():
        arrays[key] = np.asarray(values)
    return arrays


def validate_bank(arrays: Dict[str, np.ndarray], *, n: int = 64, seed: int = 0) -> Dict[str, int]:
    """Load `n` random bank rows into the native VS pool and play each pair to
    completion with random feasible actions. Raises on pool errors."""

    from envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec

    rng = np.random.default_rng(seed)
    total = arrays["boards"].shape[0]
    idx = rng.choice(total, size=min(n, total), replace=False)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    stats = {"loaded": 0, "board_mismatch": 0, "terminal": 0, "timeout": 0}
    for i in idx:
        spec = build_vs_reset_spec(
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
        runner.reset(None, [spec])
        buf = runner.buffers
        got = buf.board_bytes.reshape(2, 128)
        if not np.array_equal(got, arrays["boards"][i].reshape(2, 128)):
            stats["board_mismatch"] += 1
        stats["loaded"] += 1
        done = False
        for _step in range(3000):
            acts = np.full(2, -2, dtype=np.int32)
            for s in range(2):
                if buf.need_action[s]:
                    feas = np.flatnonzero(buf.feasible_mask[s])
                    acts[s] = int(rng.choice(feas)) if feas.size else -1
            runner.step(acts, None, None)
            if buf.terminated[0] or buf.truncated[0]:
                done = True
                break
        stats["terminal" if done else "timeout"] += 1
    runner.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(FCR_ROOT / "data" / "drmario.sqlite"))
    ap.add_argument("--fcr-root", default=str(FCR_ROOT))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "start_bank" / "start_bank_v1.npz"))
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--crisis-window", type=int, default=90)
    ap.add_argument("--max-quarks", type=int, default=None)
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
    rows: List[Dict[str, Any]] = []
    quark_names: List[str] = []
    extra: Dict[str, list] = {"quark_idx": []}
    t0 = time.time()
    for quarkid, sha in quarks:
        try:
            raw = store.get_blob(sha)
        except FileNotFoundError:
            continue
        got = extract_positions_from_events(
            raw, per_game=int(args.per_game), crisis_window=int(args.crisis_window), rng=rng
        )
        if not got:
            continue
        quark_names.append(str(quarkid))
        qidx = len(quark_names) - 1
        rows.extend(got)
        extra["quark_idx"].extend([qidx] * len(got))
        if len(quark_names) % 100 == 0:
            print(f"{len(quark_names)} quarks -> {len(rows)} positions ({time.time()-t0:.0f}s)")

    arrays = positions_to_arrays(rows, extra)
    arrays["quark_names"] = np.asarray(quark_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    strata = {STRATA[s]: int((arrays["stratum"] == s).sum()) for s in range(4)}
    print(f"wrote {out_path}: {len(rows)} positions from {len(quark_names)} quarks, strata={strata}")

    if int(args.validate) > 0 and rows:
        stats = validate_bank(arrays, n=int(args.validate), seed=int(args.seed))
        print(f"validation: {stats}")
        meta = {
            "rows": len(rows),
            "quarks": len(quark_names),
            "strata": strata,
            "validation": stats,
            "built_at": time.time(),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
