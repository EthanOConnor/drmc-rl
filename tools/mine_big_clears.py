"""Mine big, showy clears from the human corpus (CPU only, one month file at a time).

Every placement with a valid lock pose is written into its spawn bottle and
settled exactly (``drmc_rl.eval.big_clear``). Per month this writes

* ``clears-<year>-<month>.parquet``: every clear that scores at least
  ``--keep-score`` or reaches a drmariostats ``big_clear`` bar (cells >= 18,
  rounds >= 4 or viruses >= 6), with its features, context and path facts;
* ``summary-<year>-<month>.json``: counts over *every* placement and clear
  (score histogram by speed and level band, prefilter agreement, seed and
  path checks), so rarity is measured against all human play.

Path facts per placement ``k`` of a side-game:

* ``ordinal``: ``k`` (spawn order); its falling pill is ``reserve[k + 1]`` of
  the game's pill reserve (the games table ``seed`` is byte-swapped relative
  to the arena convention: ``arena = (seed & 0xFF) << 8 | seed >> 8``);
* ``seed_ok``: every pill and preview of the side-game matches that reserve;
* ``clean_run``: how many consecutive preceding placements settled exactly
  into the next spawn bottle (no garbage arrived, pose valid). A lookback of
  ``L`` placements is on the human's own garbage-free path iff
  ``clean_run >= L``.

Sources are streamed from a host (``--remote HOST:RELEASE``) into a scratch
directory and deleted after use, or read from a local release.

  python -m tools.mine_big_clears mine --remote mombox:fightcadeRatings/data/corpus/releases/human-v2-20260924T170224Z \
      --out DATA/mining --workers 6
  python -m tools.mine_big_clears summarize --out DATA/mining
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

SCHEMA = "drmc-big-clear-mining-v1"
BOARD_BARS = dict(cells=18, rounds=4, viruses=6)  # drmariostats big_clear (clears.py NOTABLE_*)
COLUMNS = ["game_id", "quarkid", "set_idx", "crown_idx", "day", "player_slot", "player", "opponent", "won",
           "spawn_frame", "lock_frame", "field", "pill_left", "pill_right", "preview_left", "preview_right",
           "speed", "speed_ups", "lock_x", "lock_y_top", "lock_rotation", "lock_repaired"]


def arena_seed(stored: int) -> int:
    """Corpus games ``seed`` -> arena reset seed (``rng_state = (s & 0xFF, s >> 8)``)."""
    if stored is None:
        return -1
    stored = int(stored)
    return ((stored & 0xFF) << 8) | (stored >> 8)


def reserves() -> np.ndarray:
    """``[65536, 128]`` pill ids of every arena seed (ROM generatePillsReserve)."""
    from drmc_rl.program.seed_reserve import step_seeds

    s = np.arange(65536, dtype=np.int64)
    pid = np.zeros(65536, dtype=np.int64)
    out = np.zeros((65536, 128), dtype=np.int8)
    for x in range(127, -1, -1):
        s = step_seeds(s)
        pid = ((s & 0xF) + pid) % 9
        out[:, x] = pid
    return out


def band(level: int) -> str:
    return "20+" if level >= 20 else "14-19" if level >= 14 else "10-13" if level >= 10 else "<10"


def _files(args) -> list[str]:
    if args.remote:
        host, root = args.remote.split(":", 1)
        out = subprocess.run(["ssh", host, f"cd {root} && find decisions -name '*.parquet' | sort"],
                             check=True, capture_output=True, text=True).stdout.split()
        return out
    root = Path(args.release)
    return sorted(str(p.relative_to(root)) for p in root.glob("decisions/**/*.parquet"))


def _fetch(args, relative: str, scratch: Path) -> Path:
    if not args.remote:
        return Path(args.release) / relative
    host, root = args.remote.split(":", 1)
    target = scratch / relative.replace("/", "_")
    subprocess.run(["rsync", "-a", f"{host}:{root}/{relative}", str(target)], check=True)
    return target


def _games(args, scratch: Path) -> dict:
    import pyarrow.parquet as pq

    if args.remote:
        host, root = args.remote.split(":", 1)
        local = scratch / "games"
        if not local.exists():
            subprocess.run(["rsync", "-a", f"{host}:{root}/games/", str(local) + "/"], check=True)
    else:
        local = Path(args.release) / "games"
    table = pq.ParquetDataset(str(local)).read(columns=["game_id", "seed", "level_p1", "level_p2",
                                                         "speed_p1", "speed_p2", "ending"]).to_pydict()
    return {g: (arena_seed(s), (int(l1 or 0), int(l2 or 0)), (int(s1 or 0), int(s2 or 0)), e) for g, s, l1, l2, s1, s2, e in
            zip(table["game_id"], table["seed"], table["level_p1"], table["level_p2"],
                table["speed_p1"], table["speed_p2"], table["ending"])}


def mine_file(relative: str, args, scratch: Path, games: dict, reserve_path: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from drmc_rl.eval import big_clear as bc
    from drmc_rl.human.corpus_public_state import pose_action, raw_pair

    started = time.time()
    table_reserves = np.load(reserve_path, mmap_mode="r")
    path = _fetch(args, relative, scratch)
    table = pq.read_table(path, columns=COLUMNS)
    if args.remote:
        path.unlink()
    n = table.num_rows
    field = np.frombuffer(table.column("field").combine_chunks().buffers()[1], dtype=np.uint8).reshape(n, 128)
    col = {k: table.column(k).to_pylist() for k in COLUMNS if k != "field"}
    ids = table.column("game_id").dictionary_encode().combine_chunks().indices.to_numpy()
    order = np.lexsort((np.asarray(col["spawn_frame"]), np.asarray(col["player_slot"]), ids))
    counts = Counter()
    hist = defaultdict(Counter)  # "speed|band" -> score*2 -> clears
    feature_hist = defaultdict(Counter)
    kept = []
    month = relative.split("year=")[1].split("/")[0] + "-" + relative.split("month=")[1].split("/")[0]
    start = 0
    while start < n:
        head = order[start]
        stop = start
        while stop < n and ids[order[stop]] == ids[head] and col["player_slot"][order[stop]] == col["player_slot"][head]:
            stop += 1
        rows = order[start:stop]
        start = stop
        gid = col["game_id"][head]
        slot = int(col["player_slot"][head])
        meta = games.get(gid)
        counts["side_games"] += 1
        if meta is None:
            counts["side_games_without_game_row"] += 1
            continue
        seed, levels, speeds, ending = meta
        level, speed = levels[slot - 1], speeds[slot - 1]
        reserve = table_reserves[max(seed, 0)]
        seed_ok = seed > 0 and all(
            reserve[(k + 1) & 127] == (col["pill_left"][i] & 3) * 3 + (col["pill_right"][i] & 3)
            and reserve[(k + 2) & 127] == (col["preview_left"][i] & 3) * 3 + (col["preview_right"][i] & 3)
            for k, i in enumerate(rows))
        counts["side_games_seed_ok"] += seed_ok
        clean_run = 0
        key = f"{speed}|{band(level)}"
        for k, i in enumerate(rows):
            counts["placements"] += 1
            this_run = clean_run
            clean_run = 0
            x, y, rot = col["lock_x"][i], col["lock_y_top"][i], col["lock_rotation"][i]
            action = -1 if x is None or y is None else pose_action(x, y, int(rot) & 3)
            if action < 0:
                counts["no_pose"] += 1
                continue
            board = field[i].tobytes()
            pill = raw_pair(col["pill_left"][i], col["pill_right"][i])
            try:
                placed = bc.place(board, pill, action)
            except ValueError:
                counts["invalid_pose"] += 1
                continue
            o, cell = divmod(action, 128)
            r, c = divmod(cell, 8)
            dr, dc = bc._SECOND[o]
            quick = bc.forms_line(placed, (cell, (r + dr) * 8 + c + dc))
            if quick:
                settled, features, detail = bc.resolve(placed)
            else:
                settled, features, detail = placed, bc.ClearFeatures(), []
                if args.verify_prefilter and counts["placements"] % args.verify_prefilter == 0:
                    counts["prefilter_checked"] += 1
                    counts["prefilter_missed"] += bc.resolve(placed)[1].rounds > 0
            nxt = rows[k + 1] if k + 1 < len(rows) else None
            clean = nxt is not None and field[nxt].tobytes() == settled
            clean_run = this_run + 1 if clean else 0
            if not features.rounds:
                continue
            counts["clears"] += 1
            counts["horizontal_clears"] += features.horizontal_lines > 0
            counts["horizontal_combo_clears"] += features.horizontal_lines > 0 and features.lines >= 2
            score = features.score()
            hist[key][int(round(score * 2))] += 1
            for name in ("cells", "rounds", "max_round_lines", "max_line", "viruses", "garbage"):
                feature_hist[name][getattr(features, name)] += 1
            feature_hist["cross"][features.cross > 0] += 1
            board_bar = (features.cells >= BOARD_BARS["cells"] or features.rounds >= BOARD_BARS["rounds"]
                         or features.viruses >= BOARD_BARS["viruses"])
            if score < args.keep_score and not board_bar:
                continue
            viruses_before = int(((field[i] & 0xF0) == 0xD0).sum())
            occupied = np.flatnonzero((field[i] != 0xFF).reshape(16, 8).any(1))
            kept.append(dict(
                month=month, game_id=gid, quarkid=col["quarkid"][i], set_idx=col["set_idx"][i],
                crown_idx=col["crown_idx"][i], day=col["day"][i], slot=slot, player=col["player"][i],
                opponent=col["opponent"][i], won=bool(col["won"][i]), ending=ending,
                spawn_frame=col["spawn_frame"][i], lock_frame=col["lock_frame"][i], ordinal=k, placements=len(rows),
                level=level, opponent_level=levels[2 - slot], speed=speed, opponent_speed=speeds[2 - slot],
                speed_ups=col["speed_ups"][i], seed=seed, seed_ok=seed_ok, clean_run=this_run,
                clean_after=clean, action=action, viruses_before=viruses_before,
                height_before=int(16 - occupied[0]) if occupied.size else 0,
                lock_repaired=bool(col["lock_repaired"][i]), board_bar=board_bar,
                **{f"f_{k2}": v for k2, v in features.to_dict().items() if k2 != "tier"},
                tier=bc.tier(score)))
    out = Path(args.out)
    if kept:
        pq.write_table(pa.Table.from_pylist(kept), out / f"clears-{month}.parquet", compression="zstd")
    summary = dict(schema=SCHEMA, file=relative, month=month, counts=dict(counts),
                   score_hist={k: {str(s): c for s, c in sorted(v.items())} for k, v in hist.items()},
                   feature_hist={k: {str(s): c for s, c in sorted(v.items())} for k, v in feature_hist.items()},
                   kept=len(kept), seconds=round(time.time() - started, 1))
    (out / f"summary-{month}.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def cmd_mine(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch or out / "scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    reserve_path = out / "reserves.npy"
    if not reserve_path.exists():
        np.save(reserve_path, reserves())
    games = _games(args, scratch)
    files = [f for f in _files(args) if not (out / f"summary-{_month(f)}.json").exists() or args.force]
    if args.months:
        files = [f for f in files if _month(f) in set(args.months)]
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} files, {len(games)} games", flush=True)
    def report(s):
        print(json.dumps(dict(month=s["month"], seconds=s["seconds"], kept=s["kept"],
                                  placements=s["counts"].get("placements"), clears=s["counts"].get("clears"))),
              flush=True)

    if args.workers <= 1:
        for f in files:
            report(mine_file(f, args, scratch, games, str(reserve_path)))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(mine_file, f, args, scratch, games, str(reserve_path)) for f in files]
            for future in as_completed(futures):
                report(future.result())
    if args.remote:
        shutil.rmtree(scratch, ignore_errors=True)


def _month(relative: str) -> str:
    return relative.split("year=")[1].split("/")[0] + "-" + relative.split("month=")[1].split("/")[0]


def cmd_summarize(args) -> None:
    from drmc_rl.eval.big_clear import TIERS

    out = Path(args.out)
    counts, hist, feats = Counter(), defaultdict(Counter), defaultdict(Counter)
    for path in sorted(out.glob("summary-*.json")):
        s = json.loads(path.read_text())
        counts.update(s["counts"])
        for k, v in s["score_hist"].items():
            hist[k].update({int(a): b for a, b in v.items()})
        for k, v in s["feature_hist"].items():
            feats[k].update(v)
    total = Counter()
    for v in hist.values():
        total.update(v)
    clears = sum(total.values())
    scores = np.array(sorted(total))
    freq = np.array([total[s] for s in scores], dtype=np.float64)
    tail = freq[::-1].cumsum()[::-1]
    def at_least(bar):
        return int(tail[scores >= bar * 2][0]) if (scores >= bar * 2).any() else 0
    quantiles = {}
    cum = freq.cumsum() / clears
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999):
        quantiles[str(q)] = float(scores[np.searchsorted(cum, q)] / 2)
    tiers = {name: dict(bar=bar, clears=at_least(bar), per_clear=at_least(bar) / clears,
                        per_placement=at_least(bar) / counts["placements"]) for name, bar in TIERS}
    by_context = {}
    for key, v in sorted(hist.items()):
        n = sum(v.values())
        by_context[key] = dict(clears=n, **{name: sum(c for s, c in v.items() if s >= bar * 2) / max(1, n)
                                             for name, bar in TIERS})
    ladder = {str(bar): at_least(bar) for bar in range(0, 61, 2)}
    result = dict(schema=SCHEMA, counts=dict(counts), clears=clears, quantiles=quantiles, tiers=tiers,
                  by_speed_level=by_context, at_least=ladder,
                  features={k: dict(sorted(v.items(), key=lambda kv: float({"False": 0, "True": 1}.get(kv[0], kv[0])))) for k, v in feats.items()})
    (out / "mining-summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "clears", "quantiles", "tiers")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--remote", help="HOST:RELEASE_DIR streamed with rsync")
    m.add_argument("--release", help="local release directory")
    m.add_argument("--out", required=True)
    m.add_argument("--scratch")
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--keep-score", type=float, default=10.0)
    m.add_argument("--verify-prefilter", type=int, default=997, help="also resolve every Nth non-line placement")
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--months", nargs="*", help="only these YYYY-MM months")
    m.add_argument("--force", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = parser.parse_args()
    (cmd_mine if args.command == "mine" else cmd_summarize)(args)


if __name__ == "__main__":
    main()
