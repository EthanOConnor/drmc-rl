"""Catalog reporting and grading helpers (shared by CLI and TUI)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from seedlab import rng as slrng
from seedlab.db import CatalogDB


def _quantiles(values: List[int], qs=(0.10, 0.50, 0.90)) -> str:
    if not values:
        return "-"
    arr = np.asarray(values, dtype=np.float64)
    return "/".join(str(int(round(float(np.quantile(arr, q))))) for q in qs)


def level_summary(db: CatalogDB, *, level: int, speed: int) -> Dict[str, object]:
    census = db.census_count(level=level)
    attempted, cleared = db.coverage(level=level, speed=speed)
    bests = db.best_frames_array(level=level, speed=speed)
    return {
        "level": level,
        "census": census,
        "attempted": attempted,
        "cleared": cleared,
        "coverage_pct": 100.0 * attempted / slrng.ORBIT_PERIOD,
        "best": min(bests) if bests else None,
        "best_q": _quantiles(bests),
        "n_best": len(bests),
    }


def print_report(db: CatalogDB, *, speed: int, levels: Optional[List[int]] = None, top: int = 5) -> None:
    lvls = levels if levels is not None else db.levels_present(speed=speed)
    if not lvls:
        print("No catalog data yet for this speed setting.")
        units = db.unit_counts()
        if units:
            print(f"work units: {units}")
        return

    print(
        f"{'level':>5} {'census':>7} {'attempted':>9} {'cleared':>8} {'cov%':>6} "
        f"{'best':>7} {'q10/q50/q90':>16}"
    )
    for lvl in lvls:
        s = level_summary(db, level=lvl, speed=speed)
        best = s["best"] if s["best"] is not None else "-"
        print(
            f"{s['level']:>5} {s['census']:>7} {s['attempted']:>9} {s['cleared']:>8} "
            f"{s['coverage_pct']:>6.1f} {best:>7} {s['best_q']:>16}"
        )
        if top > 0:
            fastest = db.fastest_seeds(level=lvl, speed=speed, k=top)
            if fastest:
                pretty = ", ".join(f"{seed:04x}:{frames}" for seed, frames in fastest)
                print(f"      fastest: {pretty}")

    units = db.unit_counts()
    if units:
        print(f"\nwork units: {units}")
    recs = db.recent_records(limit=8)
    if recs:
        print("recent records:")
        for at, lvl, sp, seed, frames, solver in recs:
            print(f"  {at} level={lvl} speed={sp} seed={seed:04x} frames={frames} ({solver})")


def grade(
    db: CatalogDB, *, level: int, speed: int, frames: int, seed: Optional[int] = None
) -> Dict[str, object]:
    """Percentile of a clear time vs the catalog. Lower percentile = faster."""

    out: Dict[str, object] = {"level": level, "speed": speed, "frames": frames}

    bests = db.best_frames_array(level=level, speed=speed)
    if bests:
        arr = np.asarray(bests, dtype=np.float64)
        out["vs_best_pct_rank"] = float(100.0 * np.mean(arr <= frames))
        out["best_median"] = float(np.median(arr))
    pooled = db.pooled_reservoir(level=level, speed=speed)
    if pooled:
        arr = np.asarray(pooled, dtype=np.float64)
        out["vs_typical_pct_rank"] = float(100.0 * np.mean(arr <= frames))
        out["typical_median"] = float(np.median(arr))

    if seed is not None:
        res = db.pooled_reservoir(level=level, speed=speed, seed=seed)
        cur = db._conn.execute(
            "SELECT best_frames FROM seed_stats WHERE level=? AND speed=? AND seed=?;",
            (int(level), int(speed), int(seed)),
        ).fetchone()
        out["seed"] = f"{int(seed):04x}"
        out["seed_best"] = int(cur[0]) if cur and cur[0] is not None else None
        if res:
            arr = np.asarray(res, dtype=np.float64)
            out["vs_seed_pct_rank"] = float(100.0 * np.mean(arr <= frames))
    return out
