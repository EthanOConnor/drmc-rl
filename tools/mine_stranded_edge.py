"""Mine stranded edge-virus endgames from arena journals or the human corpus.

``arena``: every game of the named arena output directories (``games.jsonl``
plus ``moves/<comparison>-<index>.json.gz``), both sides, labelled by variant.
``corpus``: the human corpus decisions table (CPU only; filters by the
player's interpolated rating at the game day).

Both write ``episodes.jsonl.gz`` (one row per stranded-virus episode, with the
onset position needed to rebuild a start state) and ``summary.json`` with the
scan totals, so frequency is measured against every scanned side-game.

  python -m tools.mine_stranded_edge arena --out DIR RUN_DIR [RUN_DIR ...]
  python -m tools.mine_stranded_edge corpus --out DIR [--min-rating 2000]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.eval.stranded_edge import SCHEMA, Definition, stranded, track


def _definition(args) -> Definition:
    return Definition(max_viruses=args.max_viruses, min_gap=args.min_gap,
                      max_other_cells=None if args.max_other_cells < 0 else args.max_other_cells)


def _variant_names(run: Path) -> dict[str, tuple[str, str]]:
    config = run.parent / f"{run.name}.json"
    if not config.exists():
        return {}
    data = json.loads(config.read_text())
    return {s["id"]: (s["a"], s["b"]) for s in data.get("schedule", [])}


def mine_arena(args) -> None:
    definition = _definition(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    side_games, tail_side_games = Counter(), Counter()
    started = time.time()
    with gzip.open(out / "episodes.jsonl.gz", "wt") as sink:
        for run in map(Path, args.runs):
            names = _variant_names(run)
            for line in (run / "games.jsonl").open():
                game = json.loads(line)
                comparison = game["comparison"]
                a, b = names.get(comparison, ("a", "b"))
                trace = run / "moves" / f"{comparison}-{int(game['index']):04d}.json.gz"
                if not trace.exists():
                    totals["missing_traces"] += 1
                    continue
                moves = json.load(gzip.open(trace))["moves"]
                by_side = defaultdict(list)
                for move in moves:
                    by_side[int(move["side"])].append(move)
                for physical, rows in by_side.items():
                    variant = a if physical == int(game["side"]) else b
                    won = game["winner"] == ("a" if variant == a else "b")
                    if a == b:
                        won = game["winner"] == ("a" if physical == int(game["side"]) else "b")
                    level = int(game.get("level", 14))
                    side_games[(variant, level)] += 1
                    if any(1 <= sum(1 for t in m["board"] if t & 0xF0 == 0xD0) <= definition.max_viruses
                           for m in rows[-1:]) or (won and game["reason"] == "clear"):
                        tail_side_games[(variant, level)] += 1
                    episodes = track([m["board"] for m in rows], [m["frame"] for m in rows], definition,
                                     cleared_out=won and game["reason"] == "clear", end_frame=game["frames"])
                    for episode in episodes:
                        onset = rows[episode.onset]
                        row = episode.to_dict()
                        row.update(source="arena", run=run.name, comparison=comparison, index=game["index"],
                                   seed=game["seed"], variant=variant, level=level, pace=game.get("pace"),
                                   won=bool(won), draw=game["winner"] == "draw", reason=game["reason"],
                                   board=onset["board"], opponent=onset["opponent"], pill=onset["pill"],
                                   preview=onset["preview"], speed_ups=onset.get("speed_ups", 0),
                                   pill_ordinal=episode.onset, frame=onset["frame"])
                        sink.write(json.dumps(row) + "\n")
                        totals["episodes"] += 1
                totals["games"] += 1
                if totals["games"] % 5000 == 0:
                    print(f"{totals['games']} games, {totals['episodes']} episodes, {time.time()-started:.0f}s",
                          flush=True)
    summary = dict(schema=SCHEMA, source="arena", runs=[str(r) for r in args.runs],
                   definition=definition.to_dict(), totals=dict(totals),
                   side_games={f"{v}@{l}": n for (v, l), n in sorted(side_games.items())},
                   tail_side_games={f"{v}@{l}": n for (v, l), n in sorted(tail_side_games.items())},
                   seconds=time.time() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["totals"]))


def mine_corpus(args) -> None:
    """Vectorised scan: only each side-game's tail with <= max_viruses viruses is tracked.

    Virus counts never rise within a game (garbage is pill tiles), so the rows
    with ``1..max_viruses`` viruses are exactly the suffix an episode can start
    in; earlier rows are counted for the denominators and dropped.
    """
    import pyarrow.parquet as pq

    from drmc_rl.data.human_corpus import HumanCorpus

    definition = _definition(args)
    corpus = HumanCorpus(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = {}
    for batch in corpus.batches("games", columns=["game_id", "level_p1", "level_p2"]):
        for g, l1, l2 in zip(batch["game_id"].to_pylist(), batch["level_p1"].to_pylist(),
                             batch["level_p2"].to_pylist()):
            levels[g] = (int(l1), int(l2))
    ratings: dict[tuple[str, int], float | None] = {}

    def band_of(player: str, day: int):
        key = (player, int(day))
        if key not in ratings:
            ratings[key] = corpus.rating_at(player, int(day))[0]
        r = ratings[key]
        return r, ("unrated" if r is None else "2000+" if r >= args.min_rating else
                   "1500-2000" if r >= 1500 else "<1500")

    light = ["game_id", "player_slot", "player", "day"]
    heavy = ["game_id", "player_slot", "player", "opponent", "day", "won", "ending", "spawn_frame",
             "lock_frame", "field", "opp_field", "pill_left", "pill_right", "preview_left", "preview_right",
             "speed", "speed_ups", "garbage_mask"]
    totals, side_games, levels_seen = Counter(), Counter(), Counter()
    started = time.time()
    with gzip.open(out / "episodes.jsonl.gz", "wt") as sink:
        for entry in corpus.files("decisions", months=args.months):
            parquet = pq.ParquetFile(corpus.path(entry))
            seen_sides: dict[tuple[str, int], tuple[str, int]] = {}
            tails: dict[tuple[str, int], list] = defaultdict(list)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=heavy)
                field = np.frombuffer(table.column("field").combine_chunks().buffers()[1],
                                      dtype=np.uint8).reshape(-1, 128)
                viruses = ((field & 0xF0) == 0xD0).sum(axis=1)
                ids = table.column("game_id").to_pylist()
                slots = table.column("player_slot").to_numpy()
                players = table.column("player").to_pylist()
                days = table.column("day").to_numpy()
                for i in range(len(ids)):
                    seen_sides.setdefault((ids[i], int(slots[i])), (players[i], int(days[i])))
                keep = np.flatnonzero((viruses >= 1) & (viruses <= definition.max_viruses))
                if keep.size:
                    sub = table.take(keep)
                    cols = {c: sub.column(c).to_pylist() for c in heavy}
                    for j in range(sub.num_rows):
                        tails[(cols["game_id"][j], int(cols["player_slot"][j]))].append(
                            {c: cols[c][j] for c in heavy})
                del table, field
            for key, (player, day) in seen_sides.items():
                side_games[band_of(player, day)[1]] += 1
            for (game_id, slot), rows in tails.items():
                rows.sort(key=lambda r: r["spawn_frame"])
                first = rows[0]
                rating, band = band_of(first["player"], first["day"])
                if args.strong_only and band != "2000+":
                    continue
                boards = [r["field"] for r in rows]
                # garbage_mask marks the garbage tiles standing in the field, so arrivals are its growth.
                garbage = [sum(bin(x).count("1") for x in r["garbage_mask"]) for r in rows]
                placed = [2 + max(0, garbage[k + 1] - garbage[k]) if k + 1 < len(rows) else 2
                          for k in range(len(rows))]
                won = bool(first["won"])
                level = levels.get(game_id, (None, None))[slot - 1]
                levels_seen[(band, level)] += 1
                episodes = track(boards, [r["spawn_frame"] for r in rows], definition,
                                 cleared_out=won and first["ending"] in ("clear", "allclear"),
                                 end_frame=rows[-1]["lock_frame"], placed=placed)
                for episode in episodes:
                    onset = rows[episode.onset]
                    row = episode.to_dict()
                    row.update(source="corpus", game_id=game_id, slot=slot, player=first["player"],
                               opponent=first["opponent"], rating=rating, band=band, level=level,
                               speed=int(onset["speed"]), won=won, reason=first["ending"],
                               board=list(onset["field"]), opponent_board=list(onset["opp_field"]),
                               pill=[int(onset["pill_left"]), int(onset["pill_right"])],
                               preview=[int(onset["preview_left"]), int(onset["preview_right"])],
                               speed_ups=int(onset["speed_ups"]), frame=int(onset["spawn_frame"]),
                               tail_decisions=len(rows))
                    sink.write(json.dumps(row) + "\n")
                    totals[f"episodes_{band}"] += 1
            totals["files"] += 1
            print(f"{entry.path}: side-games {dict(side_games)} {dict(totals)} {time.time()-started:.0f}s",
                  flush=True)
            del tails, seen_sides
    summary = dict(schema=SCHEMA, source="corpus", release=corpus.release_id, definition=definition.to_dict(),
                   min_rating=args.min_rating, totals=dict(totals), side_games=dict(side_games),
                   tail_side_games={f"{b}@{l}": n for (b, l), n in sorted(levels_seen.items(), key=str)},
                   seconds=time.time() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["totals"]))


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return dict(episodes=0)
    cleared = [r for r in rows if r["cleared"]]
    pills = np.asarray([r["pills"] for r in cleared] or [np.nan], float)
    frames = np.asarray([r["frames"] for r in cleared if r["frames"] is not None] or [np.nan], float)
    sd = np.asarray([r["support_destroying"] for r in rows], float)
    clears = sum(r["clears"] for r in rows)
    decided = [r for r in rows if not r.get("draw")]
    return dict(
        episodes=n, target_cleared=round(len(cleared) / n, 3),
        pills_median=float(np.nanmedian(pills)), pills_mean=round(float(np.nanmean(pills)), 2),
        pills_p90=float(np.nanpercentile(pills, 90)), frames_median=float(np.nanmedian(frames)),
        support_destroying_per_episode=round(float(sd.mean()), 3),
        support_destroying_any=round(float((sd > 0).mean()), 3),
        support_destroying_per_clear=round(float(sd.sum() / max(1, clears)), 3),
        clears_per_episode=round(clears / n, 2),
        built_under=round(float(np.mean([(r["gap_at_clear"] or 0) == 0 for r in cleared])) if cleared else 0, 3),
        round_lost=round(float(np.mean([not r["won"] for r in decided])) if decided else 0, 3),
        round_lost_target_standing=round(float(np.mean([not r["won"] for r in rows if not r["cleared"]]))
                                         if n > len(cleared) else 0, 3),
    )


def report(args) -> None:
    """Frequency and resolution tables: champion (and other arena variants) against human bands."""
    def load(path):
        return [json.loads(l) for l in gzip.open(Path(path) / "episodes.jsonl.gz", "rt")]

    def summary(path):
        return json.loads((Path(path) / "summary.json").read_text())

    out = {}
    arena, corpus = load(args.arena), load(args.corpus)
    arena_control, corpus_control = load(args.arena_control), load(args.corpus_control)
    arena_tail, corpus_tail = summary(args.arena)["tail_side_games"], summary(args.corpus)["tail_side_games"]
    groups = [(f"{v}@{args.level}", lambda r, v=v: r["source"] == "arena" and r["variant"] == v,
               arena_tail.get(f"{v}@{args.level}")) for v in args.variants]
    groups += [(f"human {b}@{args.level}", lambda r, b=b: r["source"] == "corpus" and r["band"] == b,
                corpus_tail.get(f"{b}@{args.level}")) for b in args.bands]
    for name, pick, tail in groups:
        rows = [r for r in arena + corpus if r["level"] == args.level and pick(r)]
        control = [r for r in arena_control + corpus_control if r["level"] == args.level and pick(r)
                   and r["gap"] == 0 and r["kind"] == "stranded"]
        entry = dict(endgame_side_games=tail)
        kinds = dict(stranded=lambda r: r["kind"] == "stranded",
                     stranded_open_above=lambda r: r["kind"] == "stranded" and r["above"] == 0,
                     stranded_buried=lambda r: r["kind"] == "stranded" and r["above"] > 0,
                     pillar=lambda r: r["kind"] == "pillar")
        for kind, keep in kinds.items():
            chosen = [r for r in rows if keep(r)]
            games = {(r.get("run"), r.get("comparison"), r.get("index"), r.get("game_id"), r.get("slot"),
                      r.get("variant")) for r in chosen}
            entry[kind] = dict(_stats(chosen), side_games_with_episode=len(games),
                               frequency=round(len(games) / tail, 4) if tail else None)
        entry["control_gap0"] = _stats(control)
        out[name] = entry
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("arena", "corpus"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--max-viruses", type=int, default=Definition.max_viruses)
        p.add_argument("--min-gap", type=int, default=Definition.min_gap)
        p.add_argument("--max-other-cells", type=int, default=Definition.max_other_cells,
                       help="negative disables the mostly-clear check")
        if name == "arena":
            p.add_argument("runs", nargs="+")
        else:
            p.add_argument("--root", default=None)
            p.add_argument("--min-rating", type=float, default=2000.0)
            p.add_argument("--strong-only", action="store_true")
            p.add_argument("--months", nargs="*", default=None, help="YYYY-MM subset (testing)")
    p = sub.add_parser("report")
    p.add_argument("--arena", required=True)
    p.add_argument("--arena-control", required=True, help="the same arena mining at --min-gap 0")
    p.add_argument("--corpus", required=True)
    p.add_argument("--corpus-control", required=True)
    p.add_argument("--level", type=int, default=14)
    p.add_argument("--variants", nargs="+", default=["retention_mixed", "public_core_300m", "parent"])
    p.add_argument("--bands", nargs="+", default=["2000+", "1500-2000"])
    p.add_argument("--out")
    args = parser.parse_args()
    if args.command == "report":
        report(args)
    elif args.command == "arena":
        mine_arena(args)
    else:
        if args.root is None:
            from drmc_rl.data.human_corpus import DEFAULT_ROOT

            args.root = DEFAULT_ROOT
        mine_corpus(args)


if __name__ == "__main__":
    main()
