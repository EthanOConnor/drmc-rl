"""Live catalog TUI: coverage, throughput, record feed, active leases."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from seedlab import rng as slrng
from seedlab.db import CatalogDB
from seedlab.report import beats_human_wr, fmt_frames, level_summary


def _has_search_log(db: CatalogDB) -> bool:
    row = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='search_log';"
    ).fetchone()
    return row is not None


def _coverage_panel(db: CatalogDB, speed: int) -> Panel:
    has_log = _has_search_log(db)
    table = Table(expand=True, padding=(0, 1))
    cols = ["lvl", "census", "attempted", "cleared", "cov%", "best", "q10/q50/q90",
            "searched", "proven"]
    for col in cols:
        table.add_column(col, justify="right")
    for lvl in db.levels_present(speed=speed):
        s = level_summary(db, level=lvl, speed=speed)
        cov = float(s["coverage_pct"])
        cov_style = "green" if cov >= 99.9 else ("yellow" if cov >= 50 else "red")
        searched = "-"
        if has_log:
            searched = str(
                db._conn.execute(
                    "SELECT COUNT(DISTINCT seed) FROM search_log WHERE level=? AND speed=?;",
                    (lvl, speed),
                ).fetchone()[0]
            )
        proven = db._conn.execute(
            "SELECT COUNT(*) FROM solutions WHERE level=? AND speed=? AND verified=2;",
            (lvl, speed),
        ).fetchone()[0]
        table.add_row(
            str(lvl),
            str(s["census"]),
            str(s["attempted"]),
            str(s["cleared"]),
            f"[{cov_style}]{cov:.1f}[/]",
            # Green: faster than the human IL world record (speedrun.com).
            (f"[bold green]{fmt_frames(s['best'])}[/]"
             if beats_human_wr(lvl, s["best"]) else fmt_frames(s["best"])),
            str(s["best_q"]),
            searched,
            f"[green]{proven}[/]" if int(proven) else "0",
        )
    return Panel(
        table,
        title=f"coverage (speed={speed}, orbit={slrng.ORBIT_PERIOD}) "
              f"[green]green=beats human WR[/]",
    )


def _search_panel(db: CatalogDB, speed: int) -> Panel:
    if not _has_search_log(db):
        return Panel("no search activity yet — run: python -m seedlab explore",
                     title="search (jagged explorer)")

    def _totals(where: str = "", params: tuple = ()) -> tuple:
        row = db._conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(improved),0), COALESCE(SUM(nodes),0), "
            f"COALESCE(SUM(wall_ms),0) FROM search_log {where};",
            params,
        ).fetchone()
        return tuple(int(v) for v in row)

    it_all, rec_all, nodes_all, _ = _totals()
    it_1h, rec_1h, nodes_1h, wall_1h = _totals(
        "WHERE at >= datetime('now','-1 hour')"
    )
    head = (
        f"last hour: {it_1h} iters · {rec_1h} records · {nodes_1h:,} nodes"
        + (f" · {nodes_1h / max(1, wall_1h) * 1000:.0f} nodes/s" if wall_1h else "")
        + f"\nall time : {it_all} iters · {rec_all} records · {nodes_all:,} nodes"
    )

    tiers = Table(expand=True, padding=(0, 1))
    for col in ("tier", "iters", "rec", "rec%", "avg wall", "avg Δframes"):
        tiers.add_column(col, justify="right")
    rows = db._conn.execute(
        """
        SELECT tier, COUNT(*), SUM(improved), AVG(wall_ms),
               AVG(CASE WHEN improved=1 AND best_before IS NOT NULL
                        THEN best_before - best_after END)
        FROM search_log GROUP BY tier ORDER BY COUNT(*) DESC;
        """
    ).fetchall()
    for tier, n, rec, wall_ms, gain in rows:
        rec = int(rec or 0)
        tiers.add_row(
            str(tier), str(int(n)), str(rec),
            f"{100.0 * rec / max(1, int(n)):.0f}%",
            f"{(wall_ms or 0) / 1000:.1f}s",
            "-" if gain is None else f"-{gain:.0f}",
        )

    feed = Table(expand=True, padding=(0, 1), show_header=False)
    for _ in range(4):
        feed.add_column()
    improvements = db._conn.execute(
        """
        SELECT at, level, seed, best_before, best_after, tier FROM search_log
        WHERE improved=1 ORDER BY at DESC LIMIT 6;
        """
    ).fetchall()
    for at, lvl, seed, before, after, tier in improvements:
        feed.add_row(
            str(at)[11:19],
            f"L{int(lvl)} {int(seed):04x}",
            f"{before if before is not None else '∅'}→[green]{fmt_frames(after)}[/]",
            str(tier),
        )

    return Panel(Group(head, tiers, feed), title="search (jagged explorer)")


def _records_panel(db: CatalogDB, speed: int) -> Panel:
    table = Table(expand=True, padding=(0, 1))
    for col in ("at", "lvl", "seed", "frames", "solver"):
        table.add_column(col)
    level_best: dict = {}
    for (lvl, bf) in db._conn.execute(
        "SELECT level, MIN(best_frames) FROM seed_stats WHERE speed=? AND best_frames"
        " IS NOT NULL GROUP BY level;",
        (int(speed),),
    ).fetchall():
        level_best[int(lvl)] = int(bf)
    for at, lvl, _sp, seed, frames, solver in db.recent_records(limit=12):
        cell = fmt_frames(frames)
        if level_best.get(int(lvl)) == int(frames):
            cell = f"[bold orange1]{cell}[/]"  # current level record
        table.add_row(str(at)[11:19] or at, str(lvl), f"{seed:04x}", cell, solver[:28])
    return Panel(table, title="recent records ([orange1]orange=level record[/])")


def _queue_panel(db: CatalogDB) -> Panel:
    counts = db.unit_counts()
    lines = [f"{k}: {v}" for k, v in sorted(counts.items())] or ["queue empty"]

    cur = db._conn.execute(
        """
        SELECT leased_by, level, seed_lo, seed_hi, leased_at FROM work_units
        WHERE status='leased' ORDER BY leased_at DESC LIMIT 8;
        """
    )
    leases = [
        f"{by} L{lvl} [{lo},{hi}) since {str(at)[11:19]}"
        for by, lvl, lo, hi, at in cur.fetchall()
    ]

    # Throughput: units completed in the trailing hour.
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    done_1h = db._conn.execute(
        "SELECT COUNT(*) FROM work_units WHERE status='done' AND done_at >= ?;",
        (hour_ago,),
    ).fetchone()[0]
    lines.append(f"done last hour: {int(done_1h)}")

    body = "\n".join(lines + ([""] + leases if leases else []))
    return Panel(body, title="work queue")


def build_view(db: CatalogDB, speed: int) -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="main", ratio=3),
        Layout(name="side", ratio=2),
    )
    layout["main"].split_column(
        Layout(_coverage_panel(db, speed), name="coverage", ratio=3),
        Layout(_search_panel(db, speed), name="search", ratio=2),
    )
    layout["side"].split_column(
        Layout(_records_panel(db, speed), name="records"),
        Layout(_queue_panel(db), name="queue"),
    )
    return layout


def run_dashboard(db_path: Path, *, speed: int, refresh: float = 5.0) -> None:
    db = CatalogDB(db_path)
    try:
        with Live(build_view(db, speed), refresh_per_second=1, screen=False) as live:
            while True:
                time.sleep(max(0.5, float(refresh)))
                live.update(build_view(db, speed))
    except KeyboardInterrupt:
        pass
    finally:
        db.close()
