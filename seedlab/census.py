"""Analytic census: seed-determined game content for every orbit seed."""

from __future__ import annotations

import time
from typing import Iterable

from seedlab import rng as slrng
from seedlab.db import CatalogDB


def run_census(db: CatalogDB, *, levels: Iterable[int], batch: int = 2048) -> None:
    orbit = slrng.orbit()
    for level in levels:
        existing = db.census_count(level=level)
        if existing >= len(orbit):
            print(f"[census] level {level}: complete ({existing} games)", flush=True)
            continue
        t0 = time.perf_counter()
        rows = []
        for pos, seed in enumerate(orbit):
            game = slrng.generate_game(level, seed)
            rows.append((int(level), int(seed), slrng.game_hash(game), game.virus_count, pos))
            if len(rows) >= batch:
                db.upsert_games(rows)
                rows.clear()
        if rows:
            db.upsert_games(rows)
        dt = time.perf_counter() - t0
        print(f"[census] level {level}: {len(orbit)} games in {dt:.1f}s", flush=True)
