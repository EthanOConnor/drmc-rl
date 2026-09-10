"""Import a closed remote arena snapshot into the running local dashboard."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from collections import Counter
from contextlib import closing

from drmc_rl.arena.store import ArenaStore
from drmc_rl.arena.experiment import dump, relative_ratings, outcome_summary


def completed_journal_matches(report, games):
    """The append-only journal retains games lost by old unscoped SQLite keys."""
    matches = {match["id"]: match for match in report["tournaments"]}
    unique, pairs = {}, {}
    for row in games:
        key = f"{row['comparison']}-{row['index']}"
        if key in unique and unique[key] != row:
            raise ValueError("conflicting controller journal records")
        unique[key] = row
        if row["comparison"] not in matches or row["side"] not in (0, 1):
            raise ValueError("controller journal identity does not match its schedule")
        pair = pairs.setdefault((row["comparison"], row["seed"]), {})
        if row["side"] in pair and pair[row["side"]] != key:
            raise ValueError("controller journal repeats a side/seed within one condition")
        pair[row["side"]] = key
    complete = {}
    for sides in pairs.values():
        if set(sides) != {0, 1}:
            continue
        rows = [unique[key] for key in sides.values()]
        if any(row["reason"] == "timeout" or row.get("score") is None for row in rows):
            continue
        for key in sides.values():
            row = unique[key]
            match = matches[row["comparison"]]
            if row["level"] != match["level"] or row["pace"] != match.get("pace", "frame_perfect"):
                raise ValueError("controller journal condition differs from its schedule")
            expected_winner = {0.0: "b", 0.5: "draw", 1.0: "a"}.get(row["score"])
            if expected_winner is None or row["winner"] != expected_winner:
                raise ValueError("controller journal winner and score disagree")
            complete[key] = (match, row)
    return complete


def sync(source: Path, target: Path, feed: str = "screen", *, refresh=True):
    if not feed or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in feed):
        raise ValueError("feed must be a simple lowercase identifier")
    report = json.loads((source / "results.json").read_text())
    games = [json.loads(line) for line in (source / "games.jsonl").read_text().splitlines()]
    journal = completed_journal_matches(report, games)
    # The source is a closed snapshot. Close readers explicitly before the next
    # atomic replacement, avoiding leaked SSHFS .fuse_hidden snapshot copies.
    with closing(sqlite3.connect(f"file:{source / 'arena.sqlite'}?mode=ro&immutable=1", uri=True)) as remote:
        remote.row_factory = sqlite3.Row
        store = ArenaStore(target / "arena.sqlite")
        try:
            for row in remote.execute("SELECT * FROM agents"):
                values = dict(row)
                columns = list(values)
                store.conn.execute(f"INSERT OR IGNORE INTO agents ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                   tuple(values.values()))
            existing = {r[0] for r in store.conn.execute("SELECT match_key FROM matches")}
            for key, (match, _) in journal.items():
                if key in existing:
                    store.conn.execute(
                        "UPDATE matches SET condition_key=? WHERE match_key=? AND condition_key=''",
                        (match["id"], key),
                    )
            count = 0
            for row in remote.execute("SELECT * FROM matches ORDER BY id"):
                if row["match_key"] in existing or row["match_key"] not in journal:
                    continue
                values = dict(row)
                values.pop("id")
                values["condition_key"] = journal[row["match_key"]][0]["id"]
                if values.get("replay_ref"):
                    replay = Path(values["replay_ref"])
                    if replay.is_absolute() or ".." in replay.parts:
                        raise ValueError("invalid replay path")
                    destination = target / "replays" / replay
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / "replays" / replay, destination)
                columns = list(values)
                inserted = store.conn.execute(f"INSERT OR IGNORE INTO matches ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                              tuple(values.values())).rowcount
                count += inserted
                if inserted:
                    existing.add(row["match_key"])
            for key, (match, row) in journal.items():
                if key in existing:
                    continue
                inserted = store.record(
                    match["a"], match["b"], seed=row["seed"], side=row["side"],
                    winner=row["winner"], match_len_sec=row["frames"] / 60.0988,
                    decisions=row["a_stats"].get("decisions", 0) + row["b_stats"].get("decisions", 0),
                    terminal_reason=row["reason"], match_key=key, condition_key=match["id"],
                    level=row["level"], speed_setting=2,
                    provenance={"controller_frames": True, "pace": row["pace"],
                                "move_trace": f"{match['id']}-{row['index']:04d}.json.gz",
                                "recovered_from_game_journal": True},
                    commit=False,
                )
                if not inserted:
                    raise ValueError("controller journal conflicts with a stored match identity")
                count += 1
            # Replace only this feed's disposable worker telemetry.
            workers = list(remote.execute("SELECT * FROM worker_samples"))
            for worker_id in {row["worker_id"] for row in workers}:
                store.conn.execute("DELETE FROM worker_samples WHERE worker_id=?", (worker_id,))
            for row in workers:
                values = dict(row)
                values.pop("id")
                columns = list(values)
                store.conn.execute(f"INSERT INTO worker_samples ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                   tuple(values.values()))
            store.conn.commit()
        finally:
            store.close()
    feeds = target / "feeds"
    feeds.mkdir(exist_ok=True)
    dump(feeds / f"{feed}.json", {"report": report, "games": games})
    if refresh:
        refresh_results(target)
    return count


def refresh_results(target: Path):
    """Publish once after importing all feeds in a refresh cycle."""
    feeds = target / "feeds"
    comparisons, records, updated, workers, entrants = {}, {}, [], [], {}
    for path in sorted(feeds.glob("*.json")):
        component = json.loads(path.read_text())
        updated.append(component["report"]["updated_at"])
        if component["report"].get("worker"):
            workers.append({**component["report"]["worker"],"feed":path.stem,
                            "updated_at":component["report"]["updated_at"]})
        for entrant in component["report"].get("entrants",[]):
            previous = entrants.get(entrant["id"],{})
            entrants[entrant["id"]] = {**entrant,"ready":entrant["ready"] or previous.get("ready",False)}
        for match in component["report"]["tournaments"]:
            comparisons[match["id"]] = {**match, "compute_model": component["report"].get("compute_model")}
        for game in component["games"]:
            records[(game["comparison"], game["index"])] = game
    totals = {}
    by_comparison = {}
    for (comparison, _), row in records.items():
        by_comparison.setdefault(comparison, []).append(row)
    for id, match in comparisons.items():
        rows = by_comparison.get(id, [])
        match.update(
            played=len(rows),
            **outcome_summary(rows),
            status="Complete" if len(rows) >= match["target"] else match.get("status", "Queued"),
        )
        for row in rows:
            for side in ("a", "b"):
                totals.setdefault(match[side], Counter()).update(row[side+"_stats"])
    metrics = [{"label": f"{id} · spawn wait", "value": f"{stats['spawn_wait_frames']/max(stats['decisions'],1):.2f} frames",
        "detail": f"{stats['decisions']:,} decisions · {stats['cache_hit']:,} exact hits · {stats['cache_stale_opponent']:,} older opponent contexts"}
        for id, stats in totals.items()]
    plan_path = target / "experiment.json"
    anchor = json.loads(plan_path.read_text()).get("rating_anchor", "baseline8") if plan_path.is_file() else "baseline8"
    dump(target / "results.json", {"updated_at": max(updated), "tournaments": list(comparisons.values()),
        "metrics": metrics, "execution_totals": totals,
        "workers":workers,"entrants":list(entrants.values()),
        "rating_groups": relative_ratings(comparisons, records, anchor=anchor),
        "unified_rating_groups":relative_ratings(comparisons,records,anchor=anchor,unified=True)})


def watch(path):
    errors = {}
    while True:
        config = json.loads(path.read_text())
        target = Path(config["target"])
        for mirror in config.get("checkpoint_mirrors",[]):
            destination = Path(mirror["target"])
            destination.mkdir(parents=True,exist_ok=True)
            names = mirror["files"]
            if any(Path(name).name != name or name in (".","..") for name in names):
                raise ValueError("checkpoint mirrors require explicit filenames")
            key = f"checkpoints:{destination}"
            try:
                command = ["rsync","-az"]
                for name in names:
                    command.extend(["--include","/"+name])
                command.extend(["--exclude","*",mirror["source"].rstrip("/")+"/",str(destination)+"/"])
                subprocess.run(command,check=True,capture_output=True,timeout=30)
                errors.pop(key,None)
            except (OSError,subprocess.SubprocessError) as error:
                if errors.get(key) != str(error):
                    print(f"{key}: {error}",flush=True)
                errors[key] = str(error)
        refresh = False
        for feed, remote in config["feeds"].items():
            source = target / f"incoming-{feed}"
            source.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["rsync", "-az", "--exclude", "moves", "--exclude", "working/", "--exclude", "*.pt*",
                                "--exclude", ".fuse_hidden*", "--exclude", "*-wal", "--exclude", "*-shm",
                                remote.rstrip("/")+"/", str(source)+"/"],
                    check=True, capture_output=True, timeout=30)
                if not (source / "games.jsonl").is_file():
                    continue
                count = sync(source, target, feed, refresh=False)
                refresh = True
                if count or feed in errors:
                    print(f"{feed}: imported {count} games", flush=True)
                errors.pop(feed, None)
            except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
                message = str(error)
                if errors.get(feed) != message:
                    print(f"{feed}: waiting for a complete remote snapshot: {message}", flush=True)
                errors[feed] = message
        if refresh:
            try:
                refresh_results(target)
                errors.pop("standings", None)
            except (OSError, ValueError, sqlite3.Error) as error:
                message = str(error)
                if errors.get("standings") != message:
                    print(f"standings: waiting for a consistent refresh: {message}", flush=True)
                errors["standings"] = message
        time.sleep(max(10, float(config.get("interval_seconds", 20))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--feed", default="screen")
    parser.add_argument("--watch-config", type=Path)
    args = parser.parse_args()
    if args.watch_config:
        watch(args.watch_config)
        return
    if args.source is None or args.target is None:
        parser.error("provide --source and --target, or --watch-config")
    print(f"Imported {sync(args.source, args.target, args.feed)} new games")


if __name__ == "__main__":
    main()
