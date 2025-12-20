from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seed_to_hex(seed: object) -> str:
    if seed is None:
        return ""
    if isinstance(seed, str):
        return seed
    if isinstance(seed, (bytes, bytearray)):
        return bytes(seed).hex()
    if isinstance(seed, Sequence):
        try:
            return bytes(int(v) & 0xFF for v in seed).hex()
        except Exception:
            return str(tuple(seed))
    return str(seed)


@dataclass(frozen=True, slots=True)
class BestTimeRecord:
    level: int
    seed_hex: str
    best_frames: int
    best_spawns: int
    updated_at: str
    run_id: str


class BestTimesDB:
    """Small sqlite DB for tracking best known clear times across runs.

    Stores one best time per (level, rng_seed) and exposes per-level minima and
    top-K seed best distributions.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_bests (
              level INTEGER NOT NULL,
              seed TEXT NOT NULL,
              best_frames INTEGER NOT NULL,
              best_spawns INTEGER NOT NULL,
              updated_at TEXT NOT NULL,
              run_id TEXT,
              PRIMARY KEY(level, seed)
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seed_bests_level_frames ON seed_bests(level, best_frames);"
        )
        self._conn.commit()

    @classmethod
    def default_path(cls) -> Path:
        env = os.environ.get("DRMARIO_BEST_TIMES_DB")
        if env:
            return Path(env).expanduser()
        return Path("data") / "best_times.sqlite3"

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def record_clear(
        self,
        *,
        level: int,
        rng_seed: object,
        frames: int,
        spawns: int,
        run_id: str = "",
    ) -> bool:
        """Record a clear time; returns True if it was a new best for that seed."""

        lvl = int(level)
        frames_i = int(frames)
        spawns_i = int(spawns)
        if frames_i <= 0:
            return False
        seed_hex = _seed_to_hex(rng_seed)
        if not seed_hex:
            return False

        cur = self._conn.execute(
            "SELECT best_frames, best_spawns FROM seed_bests WHERE level=? AND seed=?;",
            (lvl, seed_hex),
        )
        row = cur.fetchone()
        best_frames_prev: Optional[int] = None
        best_spawns_prev: Optional[int] = None
        if row is not None:
            try:
                best_frames_prev = int(row[0])
            except Exception:
                best_frames_prev = None
            try:
                best_spawns_prev = int(row[1])
            except Exception:
                best_spawns_prev = None

        updated = False
        best_frames_new = frames_i if best_frames_prev is None else min(best_frames_prev, frames_i)
        best_spawns_new = (
            max(0, spawns_i)
            if best_spawns_prev is None
            else min(best_spawns_prev, max(0, spawns_i))
        )
        if best_frames_prev is None or int(best_frames_new) < int(best_frames_prev):
            updated = True
        if best_spawns_prev is None or int(best_spawns_new) < int(best_spawns_prev):
            updated = True
        if not updated:
            return False

        self._conn.execute(
            """
            INSERT INTO seed_bests(level, seed, best_frames, best_spawns, updated_at, run_id)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(level, seed) DO UPDATE SET
              best_frames=MIN(seed_bests.best_frames, excluded.best_frames),
              best_spawns=MIN(seed_bests.best_spawns, excluded.best_spawns),
              updated_at=excluded.updated_at,
              run_id=excluded.run_id;
            """,
            (lvl, seed_hex, best_frames_new, best_spawns_new, _utc_now_iso(), str(run_id or "")),
        )
        self._conn.commit()
        return True

    def best_frames_floor(self, *, level: int) -> Optional[int]:
        cur = self._conn.execute(
            "SELECT MIN(best_frames) FROM seed_bests WHERE level=?;",
            (int(level),),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        try:
            return int(row[0])
        except Exception:
            return None

    def best_spawns_floor(self, *, level: int) -> Optional[int]:
        cur = self._conn.execute(
            "SELECT MIN(best_spawns) FROM seed_bests WHERE level=?;",
            (int(level),),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        try:
            return int(row[0])
        except Exception:
            return None

    def top_seed_best_frames(self, *, level: int, k: int) -> Tuple[int, ...]:
        k_i = int(max(0, k))
        if k_i <= 0:
            return tuple()
        cur = self._conn.execute(
            "SELECT best_frames FROM seed_bests WHERE level=? ORDER BY best_frames ASC LIMIT ?;",
            (int(level), k_i),
        )
        out = []
        for (frames,) in cur.fetchall():
            try:
                out.append(int(frames))
            except Exception:
                continue
        return tuple(out)

    def seed_best(self, *, level: int, rng_seed: object) -> Optional[BestTimeRecord]:
        seed_hex = _seed_to_hex(rng_seed)
        if not seed_hex:
            return None
        cur = self._conn.execute(
            """
            SELECT best_frames, best_spawns, updated_at, run_id
            FROM seed_bests
            WHERE level=? AND seed=?;
            """,
            (int(level), seed_hex),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            best_frames, best_spawns, updated_at, run_id = row
            return BestTimeRecord(
                level=int(level),
                seed_hex=str(seed_hex),
                best_frames=int(best_frames),
                best_spawns=int(best_spawns),
                updated_at=str(updated_at or ""),
                run_id=str(run_id or ""),
            )
        except Exception:
            return None
