"""Import a closed remote arena snapshot into the running local dashboard."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from collections import Counter

from drmc_rl.arena.store import ArenaStore
from drmc_rl.arena.experiment import dump, relative_ratings, score_interval


def sync(source: Path, target: Path, feed: str = "screen"):
    if not feed or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in feed):
        raise ValueError("feed must be a simple lowercase identifier")
    # Never replace the served SQLite file while its readers may have WAL open.
    with sqlite3.connect(f"file:{source / 'arena.sqlite'}?mode=ro", uri=True) as remote:
        remote.row_factory = sqlite3.Row
        store = ArenaStore(target / "arena.sqlite")
        try:
            for row in remote.execute("SELECT * FROM agents"):
                values = dict(row)
                columns = list(values)
                store.conn.execute(f"INSERT OR IGNORE INTO agents ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                   tuple(values.values()))
            existing = {r[0] for r in store.conn.execute("SELECT match_key FROM matches")}
            count = 0
            for row in remote.execute("SELECT * FROM matches ORDER BY id"):
                if row["match_key"] in existing:
                    continue
                values = dict(row)
                values.pop("id")
                if values.get("replay_ref"):
                    replay = Path(values["replay_ref"])
                    if replay.is_absolute() or ".." in replay.parts:
                        raise ValueError("invalid replay path")
                    destination = target / "replays" / replay
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / "replays" / replay, destination)
                columns = list(values)
                store.conn.execute(f"INSERT OR IGNORE INTO matches ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                   tuple(values.values()))
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
    report = json.loads((source / "results.json").read_text())
    games = [json.loads(line) for line in (source / "games.jsonl").read_text().splitlines()]
    feeds = target / "feeds"
    feeds.mkdir(exist_ok=True)
    dump(feeds / f"{feed}.json", {"report": report, "games": games})
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
    for id, match in comparisons.items():
        rows = [r for (comparison, _), r in records.items() if comparison == id]
        match.update(played=len(rows), wins=sum(r["score"] == 1 for r in rows),
            losses=sum(r["score"] == 0 for r in rows), draws=sum(r["score"] == .5 for r in rows),
            score_ci=score_interval(rows), status="Complete" if len(rows) >= match["target"] else match.get("status","Queued"))
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
    return count


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
        for feed, remote in config["feeds"].items():
            source = target / f"incoming-{feed}"
            source.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(["rsync", "-az", "--exclude", "moves", "--exclude", "working/", "--exclude", "*.pt*",
                                remote.rstrip("/")+"/", str(source)+"/"],
                    check=True, capture_output=True, timeout=30)
                if not (source / "games.jsonl").is_file():
                    continue
                count = sync(source, target, feed)
                if count or feed in errors:
                    print(f"{feed}: imported {count} games", flush=True)
                errors.pop(feed, None)
            except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
                message = str(error)
                if errors.get(feed) != message:
                    print(f"{feed}: waiting for a complete remote snapshot: {message}", flush=True)
                errors[feed] = message
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
