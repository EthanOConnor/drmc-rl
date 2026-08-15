from __future__ import annotations

import json
import gzip
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from drmc_rl.arena.ratings import (
    MODEL_VERSION,
    PairCounts,
    RatingConfig,
    RatingFit,
    PosteriorSamples,
    fit_bayesian_ratings,
    matchup_information_matrix,
    sequential_update,
    superiority_matrix,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


DEFAULT_SCHEDULER_BOOST = {
    "multiplier": 2.0,
    "max_games": 256,
    "los_target": 0.95,
}


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
  terminal_reason TEXT NOT NULL DEFAULT 'unknown',
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
CREATE TABLE IF NOT EXISTS rating_fits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_version TEXT NOT NULL,
  match_count INTEGER NOT NULL,
  agent_count INTEGER NOT NULL,
  config TEXT NOT NULL,
  diagnostics TEXT NOT NULL,
  hyperparameters TEXT NOT NULL,
  method TEXT NOT NULL DEFAULT 'hmc',
  last_match_id INTEGER NOT NULL DEFAULT 0,
  hmc_match_count INTEGER NOT NULL DEFAULT 0,
  sample_state BLOB,
  created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rating_estimates (
  fit_id INTEGER NOT NULL REFERENCES rating_fits(id) ON DELETE CASCADE,
  agent_id TEXT NOT NULL REFERENCES agents(id),
  mean REAL NOT NULL,
  sd REAL NOT NULL,
  low REAL NOT NULL,
  high REAL NOT NULL,
  probability_best REAL NOT NULL,
  rank_median REAL NOT NULL,
  rank_low REAL NOT NULL,
  rank_high REAL NOT NULL,
  draw_propensity REAL NOT NULL,
  probability_better_parent REAL,
  PRIMARY KEY(fit_id, agent_id)
);
CREATE TABLE IF NOT EXISTS rating_superiority (
  fit_id INTEGER NOT NULL REFERENCES rating_fits(id) ON DELETE CASCADE,
  agent_id TEXT NOT NULL REFERENCES agents(id),
  opponent_id TEXT NOT NULL REFERENCES agents(id),
  probability REAL NOT NULL,
  PRIMARY KEY(fit_id, agent_id, opponent_id)
);
CREATE TABLE IF NOT EXISTS rating_matchup_information (
  fit_id INTEGER NOT NULL REFERENCES rating_fits(id) ON DELETE CASCADE,
  agent_a TEXT NOT NULL REFERENCES agents(id),
  agent_b TEXT NOT NULL REFERENCES agents(id),
  information_gain REAL NOT NULL,
  PRIMARY KEY(fit_id, agent_a, agent_b)
);
CREATE INDEX IF NOT EXISTS matches_pair ON matches(agent_a, agent_b);
CREATE INDEX IF NOT EXISTS matches_created ON matches(created);
CREATE INDEX IF NOT EXISTS rating_estimates_agent ON rating_estimates(agent_id, fit_id);
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
        if "terminal_reason" not in columns:
            self.conn.execute(
                "ALTER TABLE matches ADD COLUMN terminal_reason TEXT NOT NULL DEFAULT 'unknown'"
            )
        rating_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(rating_fits)")
        }
        if "method" not in rating_columns:
            self.conn.execute(
                "ALTER TABLE rating_fits ADD COLUMN method TEXT NOT NULL DEFAULT 'hmc'"
            )
        if "last_match_id" not in rating_columns:
            self.conn.execute(
                "ALTER TABLE rating_fits ADD COLUMN last_match_id INTEGER NOT NULL DEFAULT 0"
            )
        if "sample_state" not in rating_columns:
            self.conn.execute("ALTER TABLE rating_fits ADD COLUMN sample_state BLOB")
        if "hmc_match_count" not in rating_columns:
            self.conn.execute(
                "ALTER TABLE rating_fits ADD COLUMN hmc_match_count INTEGER NOT NULL DEFAULT 0"
            )
        estimate_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(rating_estimates)")
        }
        if "probability_better_parent" not in estimate_columns:
            self.conn.execute(
                "ALTER TABLE rating_estimates ADD COLUMN probability_better_parent REAL"
            )
        # Milestone checkpoints were registered before scheduler policy became
        # explicit. Give still-active milestone entrants the same bounded
        # initial boost that future candidate registrations receive.
        for row in self.conn.execute(
            "SELECT id,metadata FROM agents WHERE status IN ('candidate','provisional')"
        ).fetchall():
            row_metadata = json.loads(row["metadata"])
            if "scheduler_boost" not in row_metadata and "milestone" in row_metadata:
                row_metadata["scheduler_boost"] = dict(DEFAULT_SCHEDULER_BOOST)
                self.conn.execute(
                    "UPDATE agents SET metadata=? WHERE id=?",
                    (json.dumps(row_metadata, sort_keys=True), row["id"]),
                )
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
        existing = self.conn.execute(
            "SELECT metadata FROM agents WHERE id=?", (agent_id,)
        ).fetchone()
        supplied_metadata = dict(metadata or {})
        if existing is not None:
            # Discovery is idempotent. Preserve scheduler state and other
            # runtime metadata when a later poll refreshes a checkpoint path.
            current_metadata = json.loads(existing["metadata"])
            current_metadata.update(supplied_metadata)
            supplied_metadata = current_metadata
        elif status in ("candidate", "provisional"):
            supplied_metadata.setdefault(
                "scheduler_boost", dict(DEFAULT_SCHEDULER_BOOST)
            )
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
                json.dumps(supplied_metadata, sort_keys=True),
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
               match_len_sec: float, decisions: int, terminal_reason: str = "unknown",
               replay: list[dict[str, Any]] | None = None) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO matches
               (agent_a,agent_b,seed,side_assignment,winner,match_len_sec,decisions,
                terminal_reason,replay,created)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (a, b, seed, side, winner, match_len_sec, decisions, terminal_reason,
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

    def ratings_need_refresh(self, *, min_new_matches: int = 64) -> bool:
        latest = self.conn.execute(
            """SELECT match_count,agent_count,hmc_match_count,sample_state IS NOT NULL AS has_samples
               FROM rating_fits ORDER BY id DESC LIMIT 1"""
        ).fetchone()
        match_count = int(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        agent_count = int(self.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0])
        return (
            latest is None
            or int(latest["agent_count"]) != agent_count
            or not bool(latest["has_samples"])
            or int(latest["hmc_match_count"]) <= 0
            or match_count - int(latest["match_count"]) >= min_new_matches
        )

    @staticmethod
    def _rating_pairs(
        rows: list[sqlite3.Row], idx: dict[str, int]
    ) -> list[PairCounts]:
        aggregate: dict[tuple[int, int], list[int]] = {}
        for row in rows:
            a, b = idx[row["agent_a"]], idx[row["agent_b"]]
            i, j = sorted((a, b))
            counts = aggregate.setdefault((i, j), [0, 0, 0])
            if row["winner"] == "draw":
                counts[1] += 1
            else:
                winner = a if row["winner"] == "a" else b
                counts[0 if winner == i else 2] += 1
        return [PairCounts(i, j, *counts) for (i, j), counts in sorted(aggregate.items())]

    def _publish_rating_fit(
        self,
        fit: RatingFit,
        agents: list[Agent],
        *,
        match_count: int,
        last_match_id: int,
        hmc_match_count: int,
        method: str,
        config: dict[str, Any],
        replace_fit_id: int | None = None,
    ) -> dict[str, Any]:
        created = utc_now()
        with self.conn:
            values = (
                MODEL_VERSION, match_count, len(agents),
                json.dumps(config, sort_keys=True),
                json.dumps(fit.diagnostics, sort_keys=True),
                json.dumps(fit.hyperparameters, sort_keys=True),
                method, last_match_id, hmc_match_count, fit.samples.encode(), created,
            )
            if replace_fit_id is None:
                # Posterior arrays are useful only for the current sequential
                # updater. Keep historical summaries, not duplicate megabyte
                # sample blobs from every calibration era.
                self.conn.execute("UPDATE rating_fits SET sample_state=NULL")
                cursor = self.conn.execute(
                    """INSERT INTO rating_fits
                       (model_version,match_count,agent_count,config,diagnostics,hyperparameters,
                        method,last_match_id,hmc_match_count,sample_state,created)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    values,
                )
                fit_id = int(cursor.lastrowid)
            else:
                fit_id = int(replace_fit_id)
                self.conn.execute(
                    """UPDATE rating_fits SET
                       model_version=?,match_count=?,agent_count=?,config=?,diagnostics=?,
                       hyperparameters=?,method=?,last_match_id=?,hmc_match_count=?,
                       sample_state=?,created=? WHERE id=?""",
                    (*values, fit_id),
                )
                self.conn.execute("DELETE FROM rating_estimates WHERE fit_id=?", (fit_id,))
                self.conn.execute("DELETE FROM rating_superiority WHERE fit_id=?", (fit_id,))
                self.conn.execute(
                    "DELETE FROM rating_matchup_information WHERE fit_id=?", (fit_id,)
                )
            self.conn.executemany(
                """INSERT INTO rating_estimates
                   (fit_id,agent_id,mean,sd,low,high,probability_best,
                    rank_median,rank_low,rank_high,draw_propensity,probability_better_parent)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        fit_id, agent.id, rating.mean, rating.sd, rating.low, rating.high,
                        rating.probability_best, rating.rank_median, rating.rank_low,
                        rating.rank_high, rating.draw_propensity,
                        rating.probability_better_parent,
                    )
                    for agent, rating in zip(agents, fit.agents, strict=True)
                ],
            )
            information = matchup_information_matrix(fit.samples)
            self.conn.executemany(
                """INSERT INTO rating_matchup_information
                   (fit_id,agent_a,agent_b,information_gain) VALUES(?,?,?,?)""",
                [
                    (fit_id, agents[i].id, agents[j].id, float(information[i, j]))
                    for i in range(len(agents))
                    for j in range(i + 1, len(agents))
                ],
            )
            superiority = superiority_matrix(fit.samples)
            self.conn.executemany(
                """INSERT INTO rating_superiority
                   (fit_id,agent_id,opponent_id,probability) VALUES(?,?,?,?)""",
                [
                    (fit_id, agent.id, opponent.id, float(superiority[i, j]))
                    for i, agent in enumerate(agents)
                    for j, opponent in enumerate(agents)
                    if i != j
                ],
            )
        return {
            "id": fit_id,
            "created": created,
            "method": method,
            "match_count": match_count,
            "agent_count": len(agents),
            "diagnostics": fit.diagnostics,
            "hyperparameters": fit.hyperparameters,
        }

    def refit_ratings(self, config: RatingConfig | None = None) -> dict[str, Any]:
        """Fit and atomically publish a full Bayesian arena posterior."""

        config = config or RatingConfig()
        agents = self.agents()
        idx = {agent.id: i for i, agent in enumerate(agents)}
        rows = list(self.conn.execute(
            "SELECT id,agent_a,agent_b,winner FROM matches ORDER BY id"
        ))
        pairs = self._rating_pairs(rows, idx)
        parents = [idx.get(agent.parent_id) for agent in agents]
        fit = fit_bayesian_ratings(len(agents), pairs, parents, config=config)
        return self._publish_rating_fit(
            fit, agents, match_count=len(rows),
            last_match_id=int(rows[-1]["id"]) if rows else 0,
            hmc_match_count=len(rows), method="hmc", config=asdict(config),
        )

    def update_ratings(
        self,
        config: RatingConfig | None = None,
        *,
        min_new_matches: int = 16,
        full_refresh_matches: int = 512,
        min_importance_ess_fraction: float = 0.50,
    ) -> dict[str, Any] | None:
        """Publish a fast exact sequential update, or refresh with HMC."""

        config = config or RatingConfig()
        latest = self.conn.execute("SELECT * FROM rating_fits ORDER BY id DESC LIMIT 1").fetchone()
        agents = self.agents()
        current_matches = int(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        if latest is None or int(latest["agent_count"]) != len(agents) or latest["sample_state"] is None:
            return self.refit_ratings(config)
        hmc_match_count = int(latest["hmc_match_count"])
        if hmc_match_count <= 0:
            return self.refit_ratings(config)
        pending = current_matches - int(latest["match_count"])
        if pending < min_new_matches:
            return None
        if current_matches - hmc_match_count >= full_refresh_matches:
            return self.refit_ratings(config)

        idx = {agent.id: i for i, agent in enumerate(agents)}
        rows = list(self.conn.execute(
            """SELECT id,agent_a,agent_b,winner FROM matches
               WHERE id>? ORDER BY id""",
            (int(latest["last_match_id"]),),
        ))
        fit = sequential_update(
            PosteriorSamples.decode(latest["sample_state"]),
            self._rating_pairs(rows, idx),
            base_diagnostics=json.loads(latest["diagnostics"]),
        )
        if float(fit.diagnostics["importance_ess_fraction"]) < min_importance_ess_fraction:
            return self.refit_ratings(config)
        fitted_match_count = int(latest["match_count"]) + len(rows)
        return self._publish_rating_fit(
            fit, agents, match_count=fitted_match_count,
            last_match_id=int(rows[-1]["id"]) if rows else int(latest["last_match_id"]),
            hmc_match_count=hmc_match_count, method="sequential",
            config=json.loads(latest["config"]), replace_fit_id=int(latest["id"]),
        )

    def _latest_ratings(self) -> tuple[dict[str, sqlite3.Row], dict[str, Any]]:
        fit = self.conn.execute("SELECT * FROM rating_fits ORDER BY id DESC LIMIT 1").fetchone()
        current_matches = int(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        side = self.conn.execute(
            """SELECT
               SUM(CASE WHEN winner='draw' THEN 1 ELSE 0 END) AS draws,
               SUM(CASE WHEN (winner='a' AND side_assignment=0)
                          OR (winner='b' AND side_assignment=1) THEN 1 ELSE 0 END) AS side0_wins,
               SUM(CASE WHEN (winner='a' AND side_assignment=1)
                          OR (winner='b' AND side_assignment=0) THEN 1 ELSE 0 END) AS side1_wins
               FROM matches"""
        ).fetchone()
        side0_wins = int(side["side0_wins"] or 0)
        side1_wins = int(side["side1_wins"] or 0)
        decisive = side0_wins + side1_wins
        side_diagnostic = {
            "side0_wins": side0_wins,
            "side1_wins": side1_wins,
            "draws": int(side["draws"] or 0),
            "side0_decisive_rate": side0_wins / decisive if decisive else None,
            "z_score": (side0_wins - side1_wins) / (decisive**0.5) if decisive else 0.0,
        }
        if fit is None:
            return {}, {
                "model": MODEL_VERSION,
                "status": "pending",
                "pending_games": current_matches,
                "side_effect": "fixed_zero",
                "side_diagnostic": side_diagnostic,
            }
        estimates = {
            row["agent_id"]: row
            for row in self.conn.execute(
                "SELECT * FROM rating_estimates WHERE fit_id=?", (fit["id"],)
            )
        }
        metadata = {
            "model": fit["model_version"],
            "status": "current" if int(fit["match_count"]) == current_matches else "updating",
            "fit_id": int(fit["id"]),
            "method": fit["method"],
            "fitted_games": int(fit["match_count"]),
            "pending_games": current_matches - int(fit["match_count"]),
            "created": fit["created"],
            "diagnostics": json.loads(fit["diagnostics"]),
            "hyperparameters": json.loads(fit["hyperparameters"]),
            "side_effect": "fixed_zero",
            "side_diagnostic": side_diagnostic,
        }
        return estimates, metadata

    def _latest_superiority(self) -> dict[str, dict[str, float]]:
        fit = self.conn.execute("SELECT id FROM rating_fits ORDER BY id DESC LIMIT 1").fetchone()
        if fit is None:
            return {}
        matrix: dict[str, dict[str, float]] = {}
        for row in self.conn.execute(
            "SELECT agent_id,opponent_id,probability FROM rating_superiority WHERE fit_id=?",
            (fit["id"],),
        ):
            matrix.setdefault(row["agent_id"], {})[row["opponent_id"]] = round(
                float(row["probability"]), 4
            )
        return matrix

    def matchup_information(self) -> dict[tuple[str, str], float]:
        fit = self.conn.execute("SELECT id FROM rating_fits ORDER BY id DESC LIMIT 1").fetchone()
        if fit is None:
            return {}
        return {
            tuple(sorted((row["agent_a"], row["agent_b"]))): float(row["information_gain"])
            for row in self.conn.execute(
                """SELECT agent_a,agent_b,information_gain
                   FROM rating_matchup_information WHERE fit_id=?""",
                (fit["id"],),
            )
        }

    def matchup_superiority(self) -> dict[str, dict[str, float]]:
        """Return posterior P(agent is stronger than opponent)."""

        return self._latest_superiority()

    def matchup_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for row in self.conn.execute(
            "SELECT agent_a,agent_b,COUNT(*) AS games FROM matches GROUP BY agent_a,agent_b"
        ):
            key = tuple(sorted((row["agent_a"], row["agent_b"])))
            counts[key] = counts.get(key, 0) + int(row["games"])
        return counts

    def snapshot(self) -> dict[str, Any]:
        agents = self.agents()
        idx = {a.id: i for i, a in enumerate(agents)}
        pair: dict[tuple[str, str], list[int]] = {}
        # Materialize the match rows once.  The worker can commit a new match
        # between separate SELECTs; re-reading here made the two collections
        # differ and crashed the dashboard's strict zip during live play.
        match_rows = list(self.conn.execute("SELECT * FROM matches ORDER BY id"))
        for row in match_rows:
            score = 1.0 if row["winner"] == "a" else 0.0 if row["winner"] == "b" else 0.5
            key = tuple(sorted((row["agent_a"], row["agent_b"])))
            rec = pair.setdefault(key, [0, 0, 0])
            if score == 0.5:
                rec[1] += 1
            elif (row["agent_a"] == key[0] and score == 1) or (row["agent_b"] == key[0] and score == 0):
                rec[0] += 1
            else:
                rec[2] += 1
        ratings, rating_metadata = self._latest_ratings()
        counts = [0] * len(agents)
        records = [{"wins": 0, "losses": 0, "draws": 0, "clears": 0, "topouts": 0}
                   for _ in agents]
        for row in match_rows:
            i, j = idx[row["agent_a"]], idx[row["agent_b"]]
            score = 1.0 if row["winner"] == "a" else 0.0 if row["winner"] == "b" else 0.5
            counts[i] += 1
            counts[j] += 1
            if score == 0.5:
                records[i]["draws"] += 1
                records[j]["draws"] += 1
                continue
            winner_i, loser_i = (i, j) if score == 1.0 else (j, i)
            records[winner_i]["wins"] += 1
            records[loser_i]["losses"] += 1
            if row["terminal_reason"] == "clear":
                records[winner_i]["clears"] += 1
            elif row["terminal_reason"] == "topout":
                records[loser_i]["topouts"] += 1
        board = []
        for i, agent in enumerate(agents):
            estimate = ratings.get(agent.id)
            mean = None if estimate is None else float(estimate["mean"])
            board.append({
                **agent.__dict__,
                "rating": None if mean is None else round(mean, 1),
                "rating_sd": None if estimate is None else round(float(estimate["sd"]), 1),
                "rating_low": None if estimate is None else round(float(estimate["low"]), 1),
                "rating_high": None if estimate is None else round(float(estimate["high"]), 1),
                "rating95": None if estimate is None else round(max(
                    mean - float(estimate["low"]), float(estimate["high"]) - mean,
                ), 1),
                "probability_best": None if estimate is None else round(
                    float(estimate["probability_best"]), 4
                ),
                "rank_median": None if estimate is None else float(estimate["rank_median"]),
                "rank_low": None if estimate is None else float(estimate["rank_low"]),
                "rank_high": None if estimate is None else float(estimate["rank_high"]),
                "draw_propensity": None if estimate is None else round(
                    float(estimate["draw_propensity"]), 3
                ),
                "lineage_los": (
                    None
                    if estimate is None or estimate["probability_better_parent"] is None
                    else round(float(estimate["probability_better_parent"]), 4)
                ),
                "games": counts[i],
                **records[i],
            })
        board.sort(key=lambda item: (
            item["rating"] is None,
            -(item["rating"] if item["rating"] is not None else 0),
            item["name"],
        ))
        events = [dict(row) for row in self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT 30"
        )]
        for event in events:
            event["detail"] = json.loads(event["detail"])
        recent = [dict(row) for row in self.conn.execute(
            "SELECT id,agent_a,agent_b,winner,match_len_sec,decisions,terminal_reason,created,replay IS NOT NULL AS has_replay FROM matches ORDER BY id DESC LIMIT 50"
        )]
        training = self._training_snapshot()
        return {
            "generated": utc_now(), "agents": board, "games": len(match_rows),
            "ratings": rating_metadata,
            "superiority": self._latest_superiority(),
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
