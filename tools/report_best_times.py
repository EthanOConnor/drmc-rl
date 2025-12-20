#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _open(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), timeout=30.0)


def _levels(conn: sqlite3.Connection) -> List[int]:
    cur = conn.execute("SELECT DISTINCT level FROM seed_bests ORDER BY level ASC;")
    out: List[int] = []
    for (lvl,) in cur.fetchall():
        try:
            out.append(int(lvl))
        except Exception:
            continue
    return out


def _seed_bests(conn: sqlite3.Connection, *, level: int) -> Tuple[np.ndarray, np.ndarray]:
    cur = conn.execute(
        "SELECT best_frames, best_spawns FROM seed_bests WHERE level=?;",
        (int(level),),
    )
    frames: List[int] = []
    spawns: List[int] = []
    for f, s in cur.fetchall():
        try:
            frames.append(int(f))
        except Exception:
            continue
        try:
            spawns.append(int(s))
        except Exception:
            spawns.append(0)
    if not frames:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return np.asarray(frames, dtype=np.int64), np.asarray(spawns, dtype=np.int64)


def _fmt_quantiles(values: np.ndarray, qs: Iterable[float]) -> str:
    if values.size == 0:
        return "-"
    parts = []
    for q in qs:
        try:
            v = float(np.quantile(values, q))
        except Exception:
            continue
        parts.append(f"{int(round(v))}")
    return "/".join(parts) if parts else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report best known clear times from best_times.sqlite3.")
    parser.add_argument(
        "--db",
        type=str,
        default="data/best_times.sqlite3",
        help="Path to sqlite DB (default: data/best_times.sqlite3).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="If set, report only this level.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many fastest seed bests to print per level (default: 10).",
    )
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        raise FileNotFoundError(f"best-times DB not found at {db_path}")

    conn = _open(db_path)
    try:
        levels = [int(args.level)] if args.level is not None else _levels(conn)
        if not levels:
            print("No levels recorded yet.")
            return

        print(
            f"{'level':>6} {'seeds':>8} {'best_f':>10} {'q10/q50/q90_f':>14} "
            f"{'best_s':>10} {'q10/q50/q90_s':>14}"
        )
        for lvl in levels:
            frames, spawns = _seed_bests(conn, level=lvl)
            if frames.size == 0:
                continue
            frames_sorted = np.sort(frames)
            spawns_sorted = np.sort(spawns) if spawns.size else np.asarray([], dtype=np.int64)
            best_f = int(frames_sorted[0])
            best_s = int(spawns_sorted[0]) if spawns_sorted.size else 0
            q_f = _fmt_quantiles(frames_sorted, (0.10, 0.50, 0.90))
            q_s = _fmt_quantiles(spawns_sorted, (0.10, 0.50, 0.90)) if spawns_sorted.size else "-"
            print(
                f"{lvl:>6} {frames_sorted.size:>8} {best_f:>10} {q_f:>14} "
                f"{best_s:>10} {q_s:>14}"
            )
            top_k = int(max(0, int(args.top)))
            if top_k > 0:
                top_vals = frames_sorted[:top_k].tolist()
                print(f"  top{min(top_k, len(top_vals))} frames: {top_vals}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

