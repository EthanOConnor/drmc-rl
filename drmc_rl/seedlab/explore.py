"""Jagged-progress explorer: randomly distributed depth & effort across seeds.

Each iteration samples a (level, seed) and a solver tier from a heavy-tailed
effort distribution, runs it, replay-verifies any improvement, and logs the
spent effort. See docs/SEEDLAB_SEARCH.md.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from drmc_rl.seedlab import rng as slrng
from drmc_rl.seedlab.db import Attempt, CatalogDB, pack_actions
from drmc_rl.seedlab.search import SearchEngine, SearchResult, beam_search, exact_search

_SEARCH_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_log(
  level INTEGER NOT NULL,
  speed INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  tier TEXT NOT NULL,
  nodes INTEGER NOT NULL,
  wall_ms INTEGER NOT NULL,
  best_before INTEGER,
  best_after INTEGER,
  improved INTEGER NOT NULL,
  at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_search_log_seed ON search_log(level, speed, seed);
"""

# Tier names encode their own effort. Resource split is wall-time based:
# each tier converges to 1/n_tiers of total spent time (pick_tier), so the
# cheap tiers run many iterations and the deep ones few — jagged by
# construction. "greedy x1" is the pass-0 worker method (single argmax
# rollout), the fastest way to put a best on an uncovered seed.
DEFAULT_TIERS: Tuple[str, ...] = (
    "greedy x1",
    "rollout x4",
    "beam w8",
    "beam w32",
    "beam w128",
    "exact",
    "polish",
)

POLISH_TOP_N = 16
POLISH_WIDTHS = (32, 64, 128, 256)

# Resource share weights: cumulative wall time converges to w_t/Σw per tier.
# Polish (record deepening on the top-N front) gets a triple share.
TIER_SHARES: Dict[str, float] = {"polish": 3.0}


def pick_tier(
    tiers: Sequence[str],
    wall_ms_by_tier: Dict[str, float],
    *,
    level: int,
    exact_max_level: int,
    shares: Optional[Dict[str, float]] = None,
) -> str:
    """Least-spent-first (share-weighted) tier selection: cumulative wall
    time converges to share_t/Σshares of total resources per tier."""

    sh = TIER_SHARES if shares is None else shares
    applicable = [
        t for t in tiers if not (t == "exact" and int(level) > int(exact_max_level))
    ]
    return min(
        applicable,
        key=lambda t: float(wall_ms_by_tier.get(t, 0.0)) / float(sh.get(t, 1.0)),
    )


def frontier_index(
    levels: Sequence[int],
    covered: Dict[int, int],
    *,
    total_seeds: int,
    priority_level: Optional[int] = None,
) -> int:
    """Index of the level to anchor width-first mass on.

    With a priority floor, the build-up starts there (lower levels only get
    the uniform random backfill); if everything at or above the floor is
    finished, the scan falls back to the full list.
    """

    candidates = list(range(len(levels)))
    if priority_level is not None:
        preferred = [i for i in candidates if int(levels[i]) >= int(priority_level)]
        for i in preferred:
            if covered.get(int(levels[i]), 0) < int(total_seeds):
                return i
    for i in candidates:
        if covered.get(int(levels[i]), 0) < int(total_seeds):
            return i
    return len(levels)


def level_weights(
    levels: Sequence[int],
    frontier_idx: int,
    *,
    decay: float = 0.35,
    uniform_eps: float = 0.08,
) -> np.ndarray:
    """Width-first level distribution over `levels` (sorted ascending).

    Bulk lands on the frontier (lowest unfinished) level; levels above it get
    geometrically decaying "tentative exploration" mass; every level
    (including finished ones below the frontier) keeps a small uniform
    residue so deepening never fully stops anywhere.
    """

    n = len(levels)
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i >= frontier_idx:
            w[i] = float(decay) ** (i - frontier_idx)
    if w.sum() <= 0:
        w[:] = 1.0  # everything finished: uniform deepening
    w = (1.0 - uniform_eps) * w / w.sum() + uniform_eps / n
    return w / w.sum()


@dataclass(slots=True)
class IterationResult:
    level: int
    seed: int
    tier: str
    nodes: int
    wall_ms: int
    best_before: Optional[int]
    best_after: Optional[int]
    improved: bool
    certified: bool = False


class Explorer:
    def __init__(
        self,
        *,
        db: CatalogDB,
        levels: Sequence[int],
        speed: int = 2,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        num_envs: int = 32,
        exact_max_level: int = 0,
        exact_node_budget: int = 150_000,
        tiers: Sequence = DEFAULT_TIERS,
        seed: int = 0,
        priority_level: Optional[int] = 4,
    ) -> None:
        from drmc_rl.seedlab.worker import Solver

        self.db = db
        self.levels = [int(l) for l in levels]
        self.speed = int(speed)
        self.exact_max_level = int(exact_max_level)
        self.exact_node_budget = int(exact_node_budget)
        self.rng = np.random.default_rng(seed)
        # Accept plain names or legacy (name, weight) pairs.
        self.tiers = [t[0] if isinstance(t, (tuple, list)) else str(t) for t in tiers]
        self.priority_level = priority_level
        self.stop_requested = False

        db._conn.executescript(_SEARCH_LOG_SCHEMA)
        db._conn.commit()
        self._coverage_cache: Tuple[float, Dict[int, int]] = (0.0, {})
        # Wall-time ledger per tier, seeded from the persistent search log:
        # the 1/n resource split holds over the catalog's lifetime, so a
        # newly added (or historically light) tier deliberately monopolizes
        # until it catches up to the others. The startup banner below makes
        # that phase legible.
        self._tier_wall_ms: Dict[str, float] = {t: 0.0 for t in self.tiers}
        for tier, wall in db._conn.execute(
            "SELECT tier, COALESCE(SUM(wall_ms),0) FROM search_log GROUP BY tier;"
        ).fetchall():
            if tier in self._tier_wall_ms:
                self._tier_wall_ms[str(tier)] = float(wall)
        ledger = " | ".join(
            f"{t} {self._tier_wall_ms[t] / 60000:.1f}m" for t in self.tiers
        )
        print(f"[explore] lifetime tier ledger: {ledger}", flush=True)
        behind = min(self._tier_wall_ms, key=self._tier_wall_ms.get)
        gap_min = (max(self._tier_wall_ms.values()) - self._tier_wall_ms[behind]) / 60000
        if gap_min > 5:
            print(
                f"[explore] '{behind}' is {gap_min:.0f}m behind and will be "
                "preferred whenever applicable until the ledger balances "
                "(lifetime 1/n split).",
                flush=True,
            )

        self.solver = Solver(
            policy="checkpoint" if checkpoint else "greedy-cost",
            checkpoint=checkpoint,
            device="cpu" if device == "mps" else device,
            temperature=0.7,
            rng=self.rng,
        )
        # Wide beams (w32+) batch enough candidates that the conv trunk wins
        # on MPS (~35% wall); narrow work stays on CPU (small-batch dispatch
        # overhead dominates below that). Measured 2026-06-11.
        self.solver_wide = self.solver
        if checkpoint and device == "mps":
            self.solver_wide = Solver(
                policy="checkpoint", checkpoint=checkpoint, device="mps",
                temperature=0.7, rng=self.rng,
            )
        self.engine = SearchEngine(num_envs=int(num_envs))
        # Beam expansion runs on a lazy pool (deferred planning, see
        # docs/SEEDLAB_SEARCH.md); rollouts/exact/replay need full steps.
        # Wider than the normal engine so a whole beam layer (width × top_m
        # candidates) lands in one engine call.
        self.beam_engine = SearchEngine(num_envs=max(256, int(num_envs)), lazy=True)
        self._lambda_cache: Dict[int, float] = {}
        self._step_bounds = None  # lazy StepBounds, shared across exact runs

    # ----------------------------------------------------------------- helpers
    def install_signal_handlers(self) -> None:
        def _handler(_sig, _frame):
            self.stop_requested = True
            print("[explore] stop requested; finishing iteration...", flush=True)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _lambda_for(self, level: int) -> float:
        # Optimistic frames-per-virus from catalog q10 of per-seed bests.
        if level in self._lambda_cache:
            return self._lambda_cache[level]
        bests = self.db.best_frames_array(level=level, speed=self.speed)
        viruses = (min(level, 20) + 1) * 4
        lam = 150.0
        if len(bests) >= 32:
            lam = float(np.quantile(np.asarray(bests, dtype=np.float64), 0.10) / viruses)
        self._lambda_cache[level] = lam
        return lam

    def _best_for(self, level: int, seed: int) -> Optional[int]:
        row = self.db._conn.execute(
            "SELECT best_frames FROM seed_stats WHERE level=? AND speed=? AND seed=?;",
            (int(level), int(self.speed), int(seed)),
        ).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def _covered_counts(self) -> Dict[int, int]:
        """Per-level count of seeds with a recorded best (60 s TTL cache)."""

        ts, cached = self._coverage_cache
        now = time.monotonic()
        if cached and now - ts < 60.0:
            return cached
        counts: Dict[int, int] = {}
        for lvl in self.levels:
            row = self.db._conn.execute(
                """
                SELECT COUNT(*) FROM seed_stats
                WHERE level=? AND speed=? AND best_frames IS NOT NULL;
                """,
                (int(lvl), self.speed),
            ).fetchone()
            counts[int(lvl)] = int(row[0] or 0)
        self._coverage_cache = (now, counts)
        return counts

    def _frontier_index(self) -> int:
        return frontier_index(
            self.levels,
            self._covered_counts(),
            total_seeds=slrng.ORBIT_PERIOD,
            priority_level=self.priority_level,
        )

    def _sample_target(self) -> Tuple[int, int]:
        weights = level_weights(self.levels, self._frontier_index())
        level = int(self.rng.choice(self.levels, p=weights))
        orbit = slrng.orbit()
        covered = self._covered_counts().get(level, 0)
        if covered < slrng.ORBIT_PERIOD:
            # Width-first fill: prefer seeds without a recorded best yet
            # (bounded rejection sampling; falls through near completion).
            for _ in range(8):
                seed = int(orbit[int(self.rng.integers(0, len(orbit)))])
                if self._best_for(level, seed) is None:
                    return level, seed
            return level, seed
        if self.rng.random() < 0.5:
            return level, int(orbit[int(self.rng.integers(0, len(orbit)))])
        # Headroom-weighted: sample among the slowest decile of per-seed bests.
        rows = self.db._conn.execute(
            """
            SELECT seed, best_frames FROM seed_stats
            WHERE level=? AND speed=? AND best_frames IS NOT NULL
            ORDER BY best_frames DESC LIMIT 512;
            """,
            (level, self.speed),
        ).fetchall()
        if not rows:
            return level, int(orbit[int(self.rng.integers(0, len(orbit)))])
        seed, _bf = rows[int(self.rng.integers(0, len(rows)))]
        return level, int(seed)

    def _sample_tier(self, level: int) -> str:
        return pick_tier(
            self.tiers, self._tier_wall_ms,
            level=level, exact_max_level=self.exact_max_level,
        )

    def _polish_target(self, *, max_level: Optional[int] = None) -> Optional[Tuple[int, int, int]]:
        """(level, seed, n_prior_polishes) from the top-N records of a level."""

        levels_with_bests = [
            lvl for lvl in self.levels
            if (max_level is None or lvl <= max_level)
            and self.db.best_frames_array(level=lvl, speed=self.speed)
        ]
        if not levels_with_bests:
            return None
        level = int(self.rng.choice(levels_with_bests))
        top = self.db.fastest_seeds(level=level, speed=self.speed, k=POLISH_TOP_N)
        seed, _frames = top[int(self.rng.integers(0, len(top)))]
        n_prior = int(
            self.db._conn.execute(
                "SELECT COUNT(*) FROM search_log WHERE tier='polish' AND level=? AND speed=? AND seed=?;",
                (level, self.speed, int(seed)),
            ).fetchone()[0]
        )
        return level, int(seed), n_prior

    # ------------------------------------------------------------------ tiers
    def _run_rollouts(self, level: int, seed: int, k: int = 4) -> Tuple[int, Optional[SearchResult]]:
        """K sampled episodes (first deterministic); feeds distribution stats."""

        eng = self.engine
        E = eng.num_envs
        b = eng.runner.buffers
        attempts: List[Attempt] = []
        best: Optional[SearchResult] = None
        n_steps = 0
        k = min(k, E)
        mask = np.zeros(E, dtype=np.uint8)
        specs: List[object] = []
        for i in range(E):
            mask[i] = 1 if i < k else 0
            specs.append(
                eng.root_spec(level=level, speed=self.speed, seed=seed)
                if i < k
                else eng._noop_spec
            )
        eng.restore(specs, mask)
        nodes = [eng.read_node(i, depth=0, g=0, trace=()) for i in range(k)]
        done = [False] * k
        g = [0] * k
        traces: List[List[int]] = [[] for _ in range(k)]
        deterministic = np.zeros(E, dtype=bool)
        deterministic[0] = True

        from drmc_rl.seedlab.search import _aux_infos

        # Eval data (human_push_05): argmax clears high levels but the slow
        # tail runs hundreds of decisions (L10 p50 ≈ 300+). Scale the budget
        # with level so coverage rollouts don't truncate real clears.
        max_decisions = 200 + 80 * min(int(level), 20)
        for _step in range(max_decisions):
            if all(done) or self.stop_requested:
                break
            infos = _aux_infos(
                [n if not done[i] else nodes[i] for i, n in enumerate(nodes)],
                level=level, speed=self.speed, v_initial=(min(level, 20) + 1) * 4,
            )
            obs = np.stack([n.obs for n in nodes]).astype(np.float32)
            acts_k = self.solver.act(obs, infos, deterministic[:k])
            actions = np.zeros(E, dtype=np.int32)
            actions[:k] = acts_k
            eng.step(actions)
            n_steps += int(np.sum(~np.asarray(done)))
            for i in range(k):
                if done[i]:
                    continue
                if int(b.invalid_action[i]) != -1:
                    done[i] = True
                    continue
                g[i] += max(1, int(b.tau_frames[i]))
                traces[i].append(int(actions[i]))
                if bool(b.terminated[i]) or bool(b.truncated[i]):
                    done[i] = True
                    cleared = int(b.viruses_rem[i]) == 0
                    attempts.append(
                        Attempt(
                            level=level, speed=self.speed, seed=seed, cleared=cleared,
                            frames=g[i], spawns=len(traces[i]),
                            solver=f"explore:{self.solver.solver_id}",
                            actions=pack_actions(traces[i]) if cleared else None,
                        )
                    )
                    if cleared and (best is None or g[i] < best.frames):
                        best = SearchResult(
                            cleared=True, frames=g[i], trace=tuple(traces[i]),
                            nodes=n_steps, wall_sec=0.0,
                        )
                else:
                    nodes[i] = eng.read_node(i, depth=nodes[i].depth + 1, g=g[i], trace=())
        # Budget-exhausted episodes count as failed attempts (greedy argmax
        # can dither indefinitely on the last viruses; mirrors the worker's
        # over-budget accounting).
        for i in range(k):
            if not done[i] and traces[i]:
                attempts.append(
                    Attempt(
                        level=level, speed=self.speed, seed=seed, cleared=False,
                        frames=g[i], spawns=len(traces[i]),
                        solver=f"explore:{self.solver.solver_id}",
                    )
                )
        if attempts:
            self.db.record_attempts(attempts)
        return n_steps, best

    # -------------------------------------------------------------- iteration
    def run_iteration(self) -> IterationResult:
        level, seed = self._sample_target()
        tier = self._sample_tier(level)
        polish_visits = 0
        if tier == "polish":
            target = self._polish_target()
            if target is None:
                tier = "greedy x1"  # nothing to polish yet; do coverage instead
            else:
                level, seed, polish_visits = target
        elif tier == "exact":
            # Certificates only close near-optimal incumbents: target record
            # seeds (the polish front), where bests are strongest and a
            # "proven optimal" label is actually valuable.
            target = self._polish_target(max_level=self.exact_max_level)
            if target is None:
                tier = "greedy x1"
            else:
                level, seed, _ = target
        best_before = self._best_for(level, seed)
        t0 = time.perf_counter()
        nodes = 0
        result: Optional[SearchResult] = None
        certified = False

        if tier == "polish":
            # Record polishing: escalate width with prior visits; perturb the
            # prior ordering after the first visit so repeats explore NEW
            # subtrees instead of re-running the same deterministic beam.
            width = POLISH_WIDTHS[min(polish_visits, len(POLISH_WIDTHS) - 1)]
            solver = self.solver_wide if width >= 32 else self.solver
            result = beam_search(
                self.beam_engine,
                level=level, speed=self.speed, seed=seed,
                width=width, top_m=10,
                solver=solver if solver.policy == "checkpoint" else None,
                lambda_frames=self._lambda_for(level),
                prior_noise=0.0 if polish_visits == 0 else 0.75,
                noise_rng=self.rng,
            )
            nodes = result.nodes
        elif tier.startswith("greedy"):
            # Pass-0 worker method: one deterministic argmax rollout.
            nodes, result = self._run_rollouts(level, seed, k=1)
        elif tier.startswith("rollout"):
            nodes, result = self._run_rollouts(level, seed, k=4)
        elif tier.startswith("beam"):
            width = int(tier.split("w")[-1])
            solver = self.solver_wide if width >= 32 else self.solver
            result = beam_search(
                self.beam_engine,
                level=level, speed=self.speed, seed=seed,
                width=width, top_m=8,
                solver=solver if solver.policy == "checkpoint" else None,
                lambda_frames=self._lambda_for(level),
            )
            nodes = result.nodes
        elif tier == "exact":
            if self._step_bounds is None:
                from drmc_rl.seedlab.bounds import StepBounds

                self._step_bounds = StepBounds(self.engine, speed_setting=self.speed)
            incumbent = best_before
            result = exact_search(
                self.engine,
                level=level, speed=self.speed, seed=seed,
                incumbent_frames=incumbent,
                node_budget=self.exact_node_budget,
                bounds=self._step_bounds,
            )
            nodes = result.nodes
            certified = bool(result.certified and result.cleared)

        improved = False
        best_after = best_before
        if result is not None and result.cleared and result.trace:
            # Replay from a true reset; the replay frame count is authoritative.
            ok, frames, spawns = self.engine.replay(
                level=level, speed=self.speed, seed=seed, trace=result.trace
            )
            # A certificate is only honest if the replay agrees with the
            # search-internal accounting (checkpoint-restore drift voids it).
            certified = certified and ok and frames == result.frames
            if ok and (
                best_before is None
                or frames < best_before
                or (certified and frames <= best_before)
            ):
                improved = self.db.record_best(
                    level=level, speed=self.speed, seed=seed,
                    frames=int(frames), spawns=int(spawns),
                    solver=f"{tier}:{self.solver.solver_id}",
                    actions=pack_actions(list(result.trace)),
                    certified=certified,
                )
                best_after = min(frames, best_before) if best_before is not None else frames
            elif not ok:
                print(
                    f"[explore] replay FAILED level={level} seed={seed:04x} tier={tier} "
                    f"(search frames={result.frames})",
                    flush=True,
                )

        wall_ms = int((time.perf_counter() - t0) * 1000)
        self._tier_wall_ms[tier] = self._tier_wall_ms.get(tier, 0.0) + float(wall_ms)
        self.db._conn.execute(
            """
            INSERT INTO search_log(level, speed, seed, tier, nodes, wall_ms,
              best_before, best_after, improved, at)
            VALUES(?,?,?,?,?,?,?,?,?,datetime('now'));
            """,
            (level, self.speed, seed, tier, nodes, wall_ms, best_before, best_after,
             1 if improved else 0),
        )
        self.db._conn.commit()
        return IterationResult(
            level=level, seed=seed, tier=tier, nodes=nodes, wall_ms=wall_ms,
            best_before=best_before, best_after=best_after, improved=improved,
            certified=certified,
        )

    def run(self, *, iterations: Optional[int] = None, duration_sec: Optional[float] = None) -> None:
        t_end = None if duration_sec is None else time.monotonic() + float(duration_sec)
        i = 0
        improvements = 0
        while not self.stop_requested:
            if iterations is not None and i >= iterations:
                break
            if t_end is not None and time.monotonic() >= t_end:
                break
            r = self.run_iteration()
            i += 1
            improvements += int(r.improved)
            mark = ""
            if r.improved:
                mark = f"  ← record {r.best_before}→{r.best_after}" + (
                    " [certified]" if r.certified else ""
                )
            print(
                f"[explore] #{i} level={r.level} seed={r.seed:04x} {r.tier:<11} "
                f"nodes={r.nodes:<6} {r.wall_ms/1000:.1f}s{mark}",
                flush=True,
            )
        print(f"[explore] done: {i} iterations, {improvements} records improved", flush=True)

    def close(self) -> None:
        self.engine.close()
        self.beam_engine.close()
