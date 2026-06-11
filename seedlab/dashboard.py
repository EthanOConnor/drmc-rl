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
from seedlab.report import level_summary


def _coverage_panel(db: CatalogDB, speed: int) -> Panel:
    table = Table(expand=True, padding=(0, 1))
    for col in ("lvl", "census", "attempted", "cleared", "cov%", "best", "q10/q50/q90"):
        table.add_column(col, justify="right")
    for lvl in db.levels_present(speed=speed):
        s = level_summary(db, level=lvl, speed=speed)
        cov = float(s["coverage_pct"])
        cov_style = "green" if cov >= 99.9 else ("yellow" if cov >= 50 else "red")
        table.add_row(
            str(lvl),
            str(s["census"]),
            str(s["attempted"]),
            str(s["cleared"]),
            f"[{cov_style}]{cov:.1f}[/]",
            str(s["best"] if s["best"] is not None else "-"),
            str(s["best_q"]),
        )
    return Panel(table, title=f"coverage (speed={speed}, orbit={slrng.ORBIT_PERIOD})")


def _records_panel(db: CatalogDB) -> Panel:
    table = Table(expand=True, padding=(0, 1))
    for col in ("at", "lvl", "seed", "frames", "solver"):
        table.add_column(col)
    for at, lvl, _sp, seed, frames, solver in db.recent_records(limit=12):
        table.add_row(str(at)[11:19] or at, str(lvl), f"{seed:04x}", str(frames), solver[:28])
    return Panel(table, title="recent records")


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
        Layout(_coverage_panel(db, speed), name="coverage", ratio=3),
        Layout(name="side", ratio=2),
    )
    layout["side"].split_column(
        Layout(_records_panel(db), name="records"),
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
