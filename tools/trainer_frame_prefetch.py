"""Speculative parallel planning for the frame-by-frame reference arena.

The frame runner plans each fresh decision inline on its main thread. This
planner keeps that exact call sequence: at the start of every frame it submits
the predictable spawn-time requests (``max(delay, reaction)`` for each newly
spawned pill) to worker threads, and the runner's own ``bfs_full`` call then
reads the shared exact answer, waits for the in-flight request, or computes a
request the prediction missed. Native BFS is a pure function of its inputs, so
decisions and journals are unchanged; only wall-clock scheduling differs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from tools.trainer_event_rollout import SharedPlanner


class PrefetchingPlanner:
    def __init__(self, workers):
        self.shared = SharedPlanner()
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.seen, self.pool = {}, None

    def bfs_full(self, columns, spawn, **kwargs):
        return self.shared.bfs_full(columns, spawn, **kwargs)

    def _plan(self, state, delay, pace):
        try:
            plan_candidates(self.shared, state, delay, pace)
        except NoReachablePlacement:
            pass

    def prefetch(self, config, match, jobs, pool):
        if pool is not self.pool:
            self.seen, self.pool = {}, pool
        pace = resolve_pace(match.get("pace", "frame_perfect"))
        for side, current in enumerate(pool.states):
            spawn = (current.spawn_id, current.pill_counter_total)
            if not current.falling or current.terminal or self.seen.get(side) == spawn:
                continue
            self.seen[side] = spawn
            pair, physical = divmod(side, 2)
            params = config["variants"][match["a"] if physical == jobs[pair][1] else match["b"]]
            delay = max(int(params["delay"]), pace.reaction_frames)
            self.executor.submit(self._plan, pool.semantic(side), delay, pace)

    def close(self):
        self.executor.shutdown(wait=True)
