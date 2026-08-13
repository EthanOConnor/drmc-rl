"""Catalog sqlite store: per-seed clear-time stats, best solutions, work queue.

Schema (docs/SEED_CATALOG.md):
- games:      seed-determined content census, keyed (level, seed).
- seed_stats: clear-time aggregates + reservoir, keyed (level, speed, seed).
- solutions:  best-known placement trace per (level, speed, seed).
- work_units: lease-based shardable work queue over orbit-position ranges.
- meta:       schema version and provenance.
"""

from __future__ import annotations

import os
import random
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
RESERVOIR_CAP = 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games(
  level INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  game_hash BLOB NOT NULL,
  virus_count INTEGER NOT NULL,
  orbit_pos INTEGER NOT NULL,
  PRIMARY KEY(level, seed)
);
CREATE INDEX IF NOT EXISTS idx_games_hash ON games(game_hash);
CREATE TABLE IF NOT EXISTS seed_stats(
  level INTEGER NOT NULL,
  speed INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  n_attempts INTEGER NOT NULL DEFAULT 0,
  n_clears INTEGER NOT NULL DEFAULT 0,
  min_frames INTEGER,
  max_frames INTEGER,
  sum_frames REAL NOT NULL DEFAULT 0,
  sumsq_frames REAL NOT NULL DEFAULT 0,
  reservoir BLOB,
  best_frames INTEGER,
  best_spawns INTEGER,
  best_solver TEXT,
  best_at TEXT,
  updated_at TEXT,
  PRIMARY KEY(level, speed, seed)
);
CREATE INDEX IF NOT EXISTS idx_seed_stats_best ON seed_stats(level, speed, best_frames);
CREATE TABLE IF NOT EXISTS solutions(
  level INTEGER NOT NULL,
  speed INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  frames INTEGER NOT NULL,
  spawns INTEGER NOT NULL,
  actions BLOB NOT NULL,
  solver TEXT NOT NULL,
  created_at TEXT NOT NULL,
  verified INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(level, speed, seed)
);
CREATE TABLE IF NOT EXISTS work_units(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  level INTEGER NOT NULL,
  speed INTEGER NOT NULL,
  pass_idx INTEGER NOT NULL,
  seed_lo INTEGER NOT NULL,
  seed_hi INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'todo',
  leased_by TEXT,
  leased_at TEXT,
  done_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_work_units_key
  ON work_units(level, speed, pass_idx, seed_lo);
CREATE INDEX IF NOT EXISTS idx_work_units_status ON work_units(status);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Attempt:
    """One finished episode on (level, speed, seed)."""

    level: int
    speed: int
    seed: int
    cleared: bool
    frames: int
    spawns: int
    solver: str
    actions: Optional[bytes] = None  # uint16-LE placement ids, clears only


@dataclass(frozen=True, slots=True)
class WorkUnit:
    id: int
    level: int
    speed: int
    pass_idx: int
    seed_lo: int
    seed_hi: int


def pack_reservoir(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *[int(v) for v in values])


def unpack_reservoir(blob: Optional[bytes]) -> List[int]:
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}I", blob[: n * 4]))


def pack_actions(actions: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(actions)}H", *[int(a) for a in actions])


def unpack_actions(blob: bytes) -> List[int]:
    n = len(blob) // 2
    return list(struct.unpack(f"<{n}H", blob[: n * 2]))


class CatalogDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=60000;")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?);",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()
        self._reservoir_rng = random.Random()

    @classmethod
    def default_path(cls) -> Path:
        env = os.environ.get("DRMARIO_SEED_CATALOG_DB")
        if env:
            return Path(env).expanduser()
        return Path("data") / "seed_catalog.sqlite3"

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ census
    def upsert_games(self, rows: Iterable[Tuple[int, int, bytes, int, int]]) -> None:
        """rows: (level, seed, game_hash, virus_count, orbit_pos)."""

        self._conn.executemany(
            """
            INSERT INTO games(level, seed, game_hash, virus_count, orbit_pos)
            VALUES(?,?,?,?,?)
            ON CONFLICT(level, seed) DO UPDATE SET
              game_hash=excluded.game_hash,
              virus_count=excluded.virus_count,
              orbit_pos=excluded.orbit_pos;
            """,
            list(rows),
        )
        self._conn.commit()

    def census_count(self, *, level: int) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM games WHERE level=?;", (int(level),))
        return int(cur.fetchone()[0])

    # ---------------------------------------------------------------- attempts
    def record_attempts(self, attempts: Sequence[Attempt]) -> int:
        """Fold a batch of finished episodes into seed_stats/solutions.

        Returns the number of new per-seed best-frame records. One transaction.
        """

        if not attempts:
            return 0
        now = _utc_now_iso()
        new_bests = 0

        # Group by key so each row is read/written once per batch.
        by_key: Dict[Tuple[int, int, int], List[Attempt]] = {}
        for a in attempts:
            by_key.setdefault((int(a.level), int(a.speed), int(a.seed)), []).append(a)

        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        try:
            for (level, speed, seed), batch in by_key.items():
                row = cur.execute(
                    """
                    SELECT n_attempts, n_clears, min_frames, max_frames, sum_frames,
                           sumsq_frames, reservoir, best_frames
                    FROM seed_stats WHERE level=? AND speed=? AND seed=?;
                    """,
                    (level, speed, seed),
                ).fetchone()
                if row is None:
                    n_attempts, n_clears = 0, 0
                    min_f, max_f = None, None
                    sum_f, sumsq_f = 0.0, 0.0
                    reservoir: List[int] = []
                    best_frames = None
                else:
                    (n_attempts, n_clears, min_f, max_f, sum_f, sumsq_f, res_blob,
                     best_frames) = row
                    reservoir = unpack_reservoir(res_blob)

                best_attempt: Optional[Attempt] = None
                for a in batch:
                    n_attempts += 1
                    if not a.cleared or a.frames <= 0:
                        continue
                    f = int(a.frames)
                    n_clears += 1
                    sum_f += f
                    sumsq_f += float(f) * float(f)
                    min_f = f if min_f is None else min(int(min_f), f)
                    max_f = f if max_f is None else max(int(max_f), f)
                    # Reservoir sampling (algorithm R) over clear times.
                    if len(reservoir) < RESERVOIR_CAP:
                        reservoir.append(f)
                    else:
                        j = self._reservoir_rng.randrange(n_clears)
                        if j < RESERVOIR_CAP:
                            reservoir[j] = f
                    if best_frames is None or f < int(best_frames):
                        best_frames = f
                        best_attempt = a

                if best_attempt is not None:
                    new_bests += 1
                    cur.execute(
                        """
                        INSERT INTO seed_stats(level, speed, seed, n_attempts, n_clears,
                          min_frames, max_frames, sum_frames, sumsq_frames, reservoir,
                          best_frames, best_spawns, best_solver, best_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(level, speed, seed) DO UPDATE SET
                          n_attempts=excluded.n_attempts,
                          n_clears=excluded.n_clears,
                          min_frames=excluded.min_frames,
                          max_frames=excluded.max_frames,
                          sum_frames=excluded.sum_frames,
                          sumsq_frames=excluded.sumsq_frames,
                          reservoir=excluded.reservoir,
                          best_frames=excluded.best_frames,
                          best_spawns=excluded.best_spawns,
                          best_solver=excluded.best_solver,
                          best_at=excluded.best_at,
                          updated_at=excluded.updated_at;
                        """,
                        (level, speed, seed, n_attempts, n_clears, min_f, max_f,
                         sum_f, sumsq_f, pack_reservoir(reservoir),
                         int(best_attempt.frames), int(best_attempt.spawns),
                         str(best_attempt.solver), now, now),
                    )
                    if best_attempt.actions:
                        cur.execute(
                            """
                            INSERT INTO solutions(level, speed, seed, frames, spawns,
                              actions, solver, created_at, verified)
                            VALUES(?,?,?,?,?,?,?,?,0)
                            ON CONFLICT(level, speed, seed) DO UPDATE SET
                              frames=excluded.frames,
                              spawns=excluded.spawns,
                              actions=excluded.actions,
                              solver=excluded.solver,
                              created_at=excluded.created_at,
                              verified=0;
                            """,
                            (level, speed, seed, int(best_attempt.frames),
                             int(best_attempt.spawns), best_attempt.actions,
                             str(best_attempt.solver), now),
                        )
                else:
                    cur.execute(
                        """
                        INSERT INTO seed_stats(level, speed, seed, n_attempts, n_clears,
                          min_frames, max_frames, sum_frames, sumsq_frames, reservoir,
                          updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(level, speed, seed) DO UPDATE SET
                          n_attempts=excluded.n_attempts,
                          n_clears=excluded.n_clears,
                          min_frames=excluded.min_frames,
                          max_frames=excluded.max_frames,
                          sum_frames=excluded.sum_frames,
                          sumsq_frames=excluded.sumsq_frames,
                          reservoir=excluded.reservoir,
                          updated_at=excluded.updated_at;
                        """,
                        (level, speed, seed, n_attempts, n_clears, min_f, max_f,
                         sum_f, sumsq_f, pack_reservoir(reservoir), now),
                    )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return new_bests

    def record_best(
        self,
        *,
        level: int,
        speed: int,
        seed: int,
        frames: int,
        spawns: int,
        solver: str,
        actions: bytes,
        certified: bool = False,
    ) -> bool:
        """Fold a search-found best without touching attempt/distribution stats.

        Returns True if it improved (or set) the stored best. `certified=True`
        marks the solution as proven optimal (solutions.verified=2).
        """

        now = _utc_now_iso()
        lvl, spd, sd = int(level), int(speed), int(seed)
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        try:
            row = cur.execute(
                "SELECT best_frames FROM seed_stats WHERE level=? AND speed=? AND seed=?;",
                (lvl, spd, sd),
            ).fetchone()
            prev_best = None if row is None or row[0] is None else int(row[0])
            improved = prev_best is None or int(frames) < prev_best
            if improved:
                cur.execute(
                    """
                    INSERT INTO seed_stats(level, speed, seed, best_frames, best_spawns,
                      best_solver, best_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(level, speed, seed) DO UPDATE SET
                      best_frames=excluded.best_frames,
                      best_spawns=excluded.best_spawns,
                      best_solver=excluded.best_solver,
                      best_at=excluded.best_at,
                      updated_at=excluded.updated_at;
                    """,
                    (lvl, spd, sd, int(frames), int(spawns), str(solver), now, now),
                )
            if improved or certified:
                cur.execute(
                    """
                    INSERT INTO solutions(level, speed, seed, frames, spawns, actions,
                      solver, created_at, verified)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(level, speed, seed) DO UPDATE SET
                      frames=excluded.frames,
                      spawns=excluded.spawns,
                      actions=excluded.actions,
                      solver=excluded.solver,
                      created_at=excluded.created_at,
                      verified=excluded.verified;
                    """,
                    (lvl, spd, sd, int(frames), int(spawns), actions, str(solver), now,
                     2 if certified else 1),
                )
            self._conn.commit()
            return improved
        except Exception:
            self._conn.rollback()
            raise

    # -------------------------------------------------------------- work queue
    def enqueue_units(
        self, *, level: int, speed: int, pass_idx: int, total_seeds: int, chunk: int
    ) -> int:
        """Create todo units covering orbit positions [0, total_seeds). Idempotent."""

        rows = []
        lo = 0
        while lo < int(total_seeds):
            hi = min(lo + int(chunk), int(total_seeds))
            rows.append((int(level), int(speed), int(pass_idx), lo, hi))
            lo = hi
        cur = self._conn.executemany(
            """
            INSERT OR IGNORE INTO work_units(level, speed, pass_idx, seed_lo, seed_hi)
            VALUES(?,?,?,?,?);
            """,
            rows,
        )
        self._conn.commit()
        return int(cur.rowcount if cur.rowcount is not None and cur.rowcount > 0 else 0)

    def claim_unit(
        self, *, worker_id: str, levels: Optional[Sequence[int]] = None
    ) -> Optional[WorkUnit]:
        """Atomically lease the next todo unit (lowest pass, then level, then range)."""

        where = "status='todo'"
        params: List[object] = []
        if levels:
            where += f" AND level IN ({','.join('?' * len(levels))})"
            params.extend(int(x) for x in levels)
        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE;")
        try:
            row = cur.execute(
                f"""
                SELECT id, level, speed, pass_idx, seed_lo, seed_hi FROM work_units
                WHERE {where}
                ORDER BY pass_idx ASC, level ASC, seed_lo ASC LIMIT 1;
                """,
                params,
            ).fetchone()
            if row is None:
                self._conn.commit()
                return None
            unit = WorkUnit(*[int(v) for v in row])
            cur.execute(
                "UPDATE work_units SET status='leased', leased_by=?, leased_at=? WHERE id=?;",
                (str(worker_id), _utc_now_iso(), unit.id),
            )
            self._conn.commit()
            return unit
        except Exception:
            self._conn.rollback()
            raise

    def complete_unit(self, unit_id: int) -> None:
        self._conn.execute(
            "UPDATE work_units SET status='done', done_at=? WHERE id=?;",
            (_utc_now_iso(), int(unit_id)),
        )
        self._conn.commit()

    def release_unit(self, unit_id: int) -> None:
        self._conn.execute(
            "UPDATE work_units SET status='todo', leased_by=NULL, leased_at=NULL WHERE id=?;",
            (int(unit_id),),
        )
        self._conn.commit()

    def reclaim_stale_leases(self, *, older_than_iso: str) -> int:
        cur = self._conn.execute(
            """
            UPDATE work_units SET status='todo', leased_by=NULL, leased_at=NULL
            WHERE status='leased' AND leased_at < ?;
            """,
            (str(older_than_iso),),
        )
        self._conn.commit()
        return int(cur.rowcount or 0)

    # ----------------------------------------------------------------- queries
    def unit_counts(self) -> Dict[str, int]:
        cur = self._conn.execute("SELECT status, COUNT(*) FROM work_units GROUP BY status;")
        return {str(k): int(v) for k, v in cur.fetchall()}

    def coverage(self, *, level: int, speed: int) -> Tuple[int, int]:
        """(seeds with ≥1 attempt, seeds with ≥1 clear) for (level, speed)."""

        cur = self._conn.execute(
            """
            SELECT COUNT(*), SUM(CASE WHEN n_clears > 0 THEN 1 ELSE 0 END)
            FROM seed_stats WHERE level=? AND speed=?;
            """,
            (int(level), int(speed)),
        )
        row = cur.fetchone()
        return int(row[0] or 0), int(row[1] or 0)

    def levels_present(self, *, speed: int) -> List[int]:
        cur = self._conn.execute(
            "SELECT DISTINCT level FROM seed_stats WHERE speed=? ORDER BY level;",
            (int(speed),),
        )
        return [int(r[0]) for r in cur.fetchall()]

    def best_frames_array(self, *, level: int, speed: int) -> List[int]:
        cur = self._conn.execute(
            """
            SELECT best_frames FROM seed_stats
            WHERE level=? AND speed=? AND best_frames IS NOT NULL;
            """,
            (int(level), int(speed)),
        )
        return [int(r[0]) for r in cur.fetchall()]

    def pooled_reservoir(self, *, level: int, speed: int, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            cur = self._conn.execute(
                "SELECT reservoir FROM seed_stats WHERE level=? AND speed=? AND seed=?;",
                (int(level), int(speed), int(seed)),
            )
        else:
            cur = self._conn.execute(
                "SELECT reservoir FROM seed_stats WHERE level=? AND speed=? AND n_clears>0;",
                (int(level), int(speed)),
            )
        out: List[int] = []
        for (blob,) in cur.fetchall():
            out.extend(unpack_reservoir(blob))
        return out

    def fastest_seeds(self, *, level: int, speed: int, k: int) -> List[Tuple[int, int]]:
        cur = self._conn.execute(
            """
            SELECT seed, best_frames FROM seed_stats
            WHERE level=? AND speed=? AND best_frames IS NOT NULL
            ORDER BY best_frames ASC LIMIT ?;
            """,
            (int(level), int(speed), int(max(1, k))),
        )
        return [(int(s), int(f)) for s, f in cur.fetchall()]

    def recent_records(self, *, limit: int = 20) -> List[Tuple[str, int, int, int, int, str]]:
        """(best_at, level, speed, seed, best_frames, best_solver), newest first."""

        cur = self._conn.execute(
            """
            SELECT best_at, level, speed, seed, best_frames, best_solver
            FROM seed_stats WHERE best_at IS NOT NULL
            ORDER BY best_at DESC LIMIT ?;
            """,
            (int(max(1, limit)),),
        )
        return [
            (str(a), int(l), int(sp), int(se), int(f), str(so or ""))
            for a, l, sp, se, f, so in cur.fetchall()
        ]

    def solution(self, *, level: int, speed: int, seed: int):
        cur = self._conn.execute(
            """
            SELECT frames, spawns, actions, solver, created_at, verified
            FROM solutions WHERE level=? AND speed=? AND seed=?;
            """,
            (int(level), int(speed), int(seed)),
        )
        return cur.fetchone()

    def mark_solution_verified(self, *, level: int, speed: int, seed: int, ok: bool) -> None:
        self._conn.execute(
            "UPDATE solutions SET verified=? WHERE level=? AND speed=? AND seed=?;",
            (1 if ok else -1, int(level), int(speed), int(seed)),
        )
        self._conn.commit()
