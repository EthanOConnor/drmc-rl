from __future__ import annotations

import json
import gzip
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.tournament import elo_mle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  family TEXT NOT NULL,
  generation INTEGER NOT NULL,
  parent_id TEXT REFERENCES agents(id),
  checkpoint TEXT NOT NULL,
  mode TEXT NOT NULL DEFAULT 'plain',
  params TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'candidate',
  created TEXT NOT NULL,
  promoted TEXT,
  metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS matches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_a TEXT NOT NULL REFERENCES agents(id),
  agent_b TEXT NOT NULL REFERENCES agents(id),
  seed INTEGER NOT NULL,
  side_assignment INTEGER NOT NULL,
  winner TEXT NOT NULL,
  match_len_sec REAL,
  decisions INTEGER,
  replay TEXT,
  created TEXT NOT NULL,
  UNIQUE(agent_a, agent_b, seed, side_assignment)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  agent_id TEXT REFERENCES agents(id),
  detail TEXT NOT NULL DEFAULT '{}',
  created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS matches_pair ON matches(agent_a, agent_b);
CREATE INDEX IF NOT EXISTS matches_created ON matches(created);
"""


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    family: str
    generation: int
    parent_id: str | None
    checkpoint: str
    mode: str
    params: dict[str, Any]
    status: str
    created: str
    promoted: str | None
    metadata: dict[str, Any]

    def entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "mode": self.mode,
            "params": self.params,
        }


class ArenaStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(matches)")}
        if "replay" not in columns:
            self.conn.execute("ALTER TABLE matches ADD COLUMN replay TEXT")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def register(
        self,
        *,
        agent_id: str,
        name: str,
        family: str,
        generation: int,
        checkpoint: str,
        parent_id: str | None = None,
        mode: str = "plain",
        params: dict[str, Any] | None = None,
        status: str = "candidate",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        self.conn.execute(
            """INSERT INTO agents
               (id,name,family,generation,parent_id,checkpoint,mode,params,status,created,metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET checkpoint=excluded.checkpoint,
                 params=excluded.params, metadata=excluded.metadata""",
            (
                agent_id,
                name,
                family,
                generation,
                parent_id,
                checkpoint,
                mode,
                json.dumps(params or {}, sort_keys=True),
                status,
                now,
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        self.conn.execute(
            "INSERT INTO events(kind,agent_id,detail,created) VALUES('registered',?,?,?)",
            (agent_id, "{}", now),
        )
        self.conn.commit()

    def agents(self, statuses: Iterable[str] | None = None) -> list[Agent]:
        args: list[Any] = []
        where = ""
        if statuses:
            values = list(statuses)
            where = f" WHERE status IN ({','.join('?' for _ in values)})"
            args.extend(values)
        rows = self.conn.execute("SELECT * FROM agents" + where + " ORDER BY created", args)
        return [self._agent(row) for row in rows]

    def agent(self, agent_id: str) -> Agent:
        row = self.conn.execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
        if row is None:
            raise KeyError(agent_id)
        return self._agent(row)

    @staticmethod
    def _agent(row: sqlite3.Row) -> Agent:
        return Agent(
            id=row["id"], name=row["name"], family=row["family"],
            generation=row["generation"], parent_id=row["parent_id"],
            checkpoint=row["checkpoint"], mode=row["mode"],
            params=json.loads(row["params"]), status=row["status"],
            created=row["created"], promoted=row["promoted"],
            metadata=json.loads(row["metadata"]),
        )

    def record(self, a: str, b: str, *, seed: int, side: int, winner: str,
               match_len_sec: float, decisions: int,
               replay: list[dict[str, Any]] | None = None) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO matches
               (agent_a,agent_b,seed,side_assignment,winner,match_len_sec,decisions,replay,created)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (a, b, seed, side, winner, match_len_sec, decisions,
             json.dumps(replay, separators=(",", ":")) if replay else None, utc_now()),
        )
        self.conn.commit()

    def promote(self, agent_id: str, *, detail: dict[str, Any]) -> None:
        now = utc_now()
        agent = self.agent(agent_id)
        self.conn.execute(
            "UPDATE agents SET status='champion', promoted=? WHERE id=?", (now, agent_id)
        )
        # Historical champions stay active and immutable in the arena.
        self.conn.execute(
            "UPDATE agents SET status='lineage' WHERE family=? AND status='champion' AND id<>?",
            (agent.family, agent_id),
        )
        self.conn.execute(
            "INSERT INTO events(kind,agent_id,detail,created) VALUES('promoted',?,?,?)",
            (agent_id, json.dumps(detail, sort_keys=True), now),
        )
        self.conn.commit()

    def snapshot(self) -> dict[str, Any]:
        agents = self.agents()
        idx = {a.id: i for i, a in enumerate(agents)}
        games: list[tuple[int, int, float]] = []
        pair: dict[tuple[str, str], list[int]] = {}
        for row in self.conn.execute("SELECT * FROM matches ORDER BY id"):
            score = 1.0 if row["winner"] == "a" else 0.0 if row["winner"] == "b" else 0.5
            games.append((idx[row["agent_a"]], idx[row["agent_b"]], score))
            key = tuple(sorted((row["agent_a"], row["agent_b"])))
            rec = pair.setdefault(key, [0, 0, 0])
            if score == 0.5:
                rec[1] += 1
            elif (row["agent_a"] == key[0] and score == 1) or (row["agent_b"] == key[0] and score == 0):
                rec[0] += 1
            else:
                rec[2] += 1
        ratings, errors = elo_mle(len(agents), games) if games else (
            np.zeros(len(agents)), np.zeros(len(agents))
        )
        counts = [0] * len(agents)
        for i, j, _ in games:
            counts[i] += 1
            counts[j] += 1
        board = []
        for i, agent in enumerate(agents):
            board.append({
                **agent.__dict__,
                "rating": round(float(ratings[i]), 1),
                "rating95": round(float(errors[i] * 1.96), 1),
                "games": counts[i],
            })
        board.sort(key=lambda item: (-item["rating"], item["name"]))
        events = [dict(row) for row in self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 30"
        )]
        for event in events:
            event["detail"] = json.loads(event["detail"])
        recent = [dict(row) for row in self.conn.execute(
            "SELECT id,agent_a,agent_b,winner,match_len_sec,decisions,created,replay IS NOT NULL AS has_replay FROM matches ORDER BY id DESC LIMIT 50"
        )]
        training = self._training_snapshot()
        return {
            "generated": utc_now(), "agents": board, "games": len(games),
            "pairs": [{"a": k[0], "b": k[1], "wins_a": v[0], "draws": v[1], "wins_b": v[2]}
                      for k, v in pair.items()],
            "events": events, "recent": recent, "training": training,
        }

    def replay(self, match_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT agent_a,agent_b,winner,match_len_sec,replay FROM matches WHERE id=?",
            (match_id,),
        ).fetchone()
        if row is None or row["replay"] is None:
            return None
        return {**dict(row), "replay": json.loads(row["replay"])}

    def _training_snapshot(self) -> dict[str, Any]:
        telemetry = self.path.parent / "training.json"
        if telemetry.is_file():
            try:
                return json.loads(telemetry.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        runs_root = self.path.parent.parent
        streams = sorted(runs_root.glob("*/**/metrics.jsonl.gz"),
                         key=lambda path: path.stat().st_mtime, reverse=True)
        if not streams:
            return {}
        path = streams[0]
        latest: dict[str, Any] = {}
        history: dict[str, list[list[float]]] = {}
        try:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        break
                    if row.get("type") != "scalar":
                        continue
                    name = str(row["name"])
                    latest[name] = row["value"]
                    if name in {"perf/sps", "perf/dps", "train/return_mean",
                                "search_distill/searched_fraction_actual"}:
                        history.setdefault(name, []).append([row["step"], row["value"]])
        except (EOFError, OSError):
            pass
        return {
            "run": str(path.parent.relative_to(runs_root)),
            "updated": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "latest": latest,
            "history": {name: points[-60:] for name, points in history.items()},
        }

    def matchup_games(self, a: str, b: str) -> list[float]:
        out = []
        for row in self.conn.execute(
            "SELECT * FROM matches WHERE (agent_a=? AND agent_b=?) OR (agent_a=? AND agent_b=?)",
            (a, b, b, a),
        ):
            if row["winner"] == "draw":
                out.append(0.5)
            else:
                winner = row["agent_a"] if row["winner"] == "a" else row["agent_b"]
                out.append(1.0 if winner == a else 0.0)
        return out
