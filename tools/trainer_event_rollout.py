"""Batch independent causal decision boundaries while retaining exact inputs.

The frame-by-frame arena remains the independent execution reference. This
runner changes wall-clock scheduling only; it does not warp falls or park one
player while its opponent advances. No speculative preparation is performed.
"""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
import time
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drmc_rl.envs.backends.vs_frames import EventVsPool
from drmc_rl.execution.pace import resolve_pace, strategy_context
from drmc_rl.human.anticipation import execution_for_action, score_public_inputs
from drmc_rl.human.controller_context import controller_policy_inputs, uses_public_context
from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import ByteCache


class SharedPlanner:
    """Share exact answers and coalesce concurrent identical BFS requests."""
    def __init__(self, capture_path=None):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.cache = ByteCache(64*1024*1024)
        self.pending = {}
        self.capture_path, self.roots = capture_path, {}

    def bfs_full(self, columns, spawn, **kwargs):
        micro = tuple(int(getattr(v,"value",v)) for v in (spawn.x,spawn.y,spawn.rot,spawn.speed_counter,
            spawn.hor_velocity,spawn.hold_dir,spawn.rot_hold,spawn.frame_parity,spawn.locked))
        key = np.asarray(columns,dtype=np.uint16).tobytes()+repr((micro,sorted(kwargs.items()))).encode()
        with self.lock:
            if self.capture_path and len(self.roots) < 4096:
                self.roots.setdefault(key,dict(columns=np.asarray(columns).tolist(),micro=micro,kwargs=kwargs))
            result = self.cache.get(key)
            if result is not None:
                return result
            owner = key not in self.pending
            future = self.pending.setdefault(key,Future())
        if not owner:
            return future.result()
        try:
            if not hasattr(self.local,"runner"):
                self.local.runner = NativeReachabilityRunner()
            result = self.local.runner.bfs_full(columns,spawn,**kwargs).copy()
            arrays = (result.costs_u16,result.offsets_u16,result.lengths_u16,result.script_buf)
            for array in arrays:
                array.flags.writeable = False
            with self.lock:
                self.cache.put(key,result,sum(a.nbytes for a in arrays))
            future.set_result(result)
            return result
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self.lock:
                self.pending.pop(key,None)


class ParallelPlanning:
    """Per-worker native buffers, shared exact results; native BFS scratch is TLS."""
    def __init__(self, workers=4, capture_path=None):
        self.planner = SharedPlanner(capture_path)
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _plan(self, request):
        state, delay, pace = request
        try:
            return plan_candidates(self.planner, state, delay, pace)
        except NoReachablePlacement:
            return None

    def plan(self, requests):
        return list(self.executor.map(self._plan, requests))

    def submit(self, request):
        return self.executor.submit(self._plan, request)

    def close(self):
        self.executor.shutdown(wait=True)
        if self.planner.capture_path:
            Path(self.planner.capture_path).write_text(json.dumps(list(self.planner.roots.values())))


def run_event_batch(config, match, jobs, policy, planner, preparer, *, policies=None, metrics=None, activity=None):
    if preparer is not None or any(p.get("anticipation") for p in config["variants"].values()):
        raise ValueError("event rollout currently requires reaction-covered computation")
    if config.get("replay_games", 0):
        raise ValueError("use the reference frame runner for full-frame replay capture")
    started = time.perf_counter()
    next_activity = started
    measured = Counter()
    pace = resolve_pace(match.get("pace", "frame_perfect"))
    limit = config.get("max_game_frames", 60000)
    moves = [[] for _ in jobs]
    statistics = [Counter() for _ in range(2*len(jobs))]
    pending = {}
    asynchronous = config.get("async_planning", False)
    if asynchronous and not hasattr(planner,"submit"):
        raise ValueError("asynchronous rollout requires a submitting planner")
    with EventVsPool(len(jobs), lib_path=config.get("native_library")) as pool:
        pool.reset([job[0] for job in jobs], level=match["level"])
        while True:
            tick = time.perf_counter()
            progress = pool.advance(limit)
            measured["engine_seconds"] += time.perf_counter()-tick
            ready = []
            for side, value in enumerate(progress):
                for field in ("validated_input_frames", "locks", "unplanned_locks"):
                    statistics[side][field] += getattr(value, field)
                if value.needs_action and side not in pending:
                    ready.append(side)
            if activity and (time.perf_counter() >= next_activity or (not ready and not pending)):
                activity(dict(
                    games=sum(s.terminal or s.frame >= limit for s in pool.states[::2]),
                    frames=sum(int(s.frame) for s in pool.states[::2]),
                    decision_requests=sum(statistics[2*p+side]["decisions"]
                                          for p, (_, side, _) in enumerate(jobs)),
                ))
                next_activity = time.perf_counter() + 5
            if not ready and not pending:
                break
            requests, actors = [], []
            for side in ready:
                pair, physical = divmod(side, 2)
                id = match["a"] if physical == jobs[pair][1] else match["b"]
                params = config["variants"][id]
                if params.get("own_board_only"):
                    raise ValueError("own-board ablation is not a pace-training setting")
                delay = max(int(params["delay"]), pace.reaction_frames)
                statistics[side]["decisions"] += 1
                actor = policy if policies is None else policies[id]
                state = pool.semantic(side, public_context=uses_public_context(actor))
                requests.append((state, delay, pace))
                actors.append(id)
            tick = time.perf_counter()
            if asynchronous:
                for side, request, actor in zip(ready,requests,actors):
                    pending[side] = planner.submit(request), request, actor
                # A difficult position must not stall unrelated matches. Both
                # sides of its own pair stay parked in frame_advance until the
                # missing decision arrives, so its public snapshot stays valid.
                wait([p[0] for p in pending.values()],return_when=FIRST_COMPLETED)
                deadline = time.perf_counter()+.002
                while sum(p[0].done() for p in pending.values()) < min(16,len(pending)):
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0:
                        break
                    wait([p[0] for p in pending.values() if not p[0].done()],
                         timeout=remaining,return_when=FIRST_COMPLETED)
                ready = [side for side,p in pending.items() if p[0].done()][:64]
                completed = [pending.pop(side) for side in ready]
                candidates = [p[0].result() for p in completed]
                requests = [p[1] for p in completed]
                actors = [p[2] for p in completed]
            elif hasattr(planner, "plan"):
                candidates = planner.plan(requests)
            else:
                candidates = []
                for state, delay, _ in requests:
                    try:
                        candidates.append(plan_candidates(planner, state, delay, pace))
                    except NoReachablePlacement:
                        candidates.append(None)
            measured["planning_seconds"] += time.perf_counter()-tick
            observations, infos, selected_indices = [], [], []
            for i, candidate in enumerate(candidates):
                side = ready[i]
                if candidate is None:
                    statistics[side]["no_reachable_after_delay"] += 1
                    pool.install(side)
                    continue
                state, delay, _ = requests[i]
                actor = policy if policies is None else policies[actors[i]]
                obs, info = controller_policy_inputs(
                    actor, candidate, state, pace, delay,
                    int(config["variants"][actors[i]]["delay"]),
                )
                info[0]["pace/context"] = strategy_context(pace, state, delay)
                count = int(np.count_nonzero(info[0]["placements/feasible_mask"]))
                statistics[side]["feasible_candidates"] += count
                statistics[side]["forced_placements"] += int(count == 1)
                observations.append(obs)
                infos.extend(info)
                selected_indices.append(i)
            if not infos:
                continue
            obs = np.concatenate(observations)
            scores = np.empty((len(infos),512), np.float32)
            learning = {}
            tick = time.perf_counter()
            mixed = config.get("mixed_core_actor")
            if mixed:
                if set(policies) != {mixed,"parent"}:
                    raise ValueError("mixed scoring requires one adapter and its frozen parent")
                actor = policies[mixed]
                adapted = np.array([actors[i] == mixed for i in selected_indices])
                view = SimpleNamespace(score=lambda o,inf: actor.score_mixed(o,inf,adapted))
                scores[:] = score_public_inputs(view,obs,infos)
                if actor.learning_records is not None:
                    learning = {j:r for j,r in enumerate(actor.learning_records) if r is not None}
            else:
                for id in sorted(set(actors[i] for i in selected_indices)):
                    indices = [j for j,i in enumerate(selected_indices) if actors[i] == id]
                    actor = policy if policies is None else policies[id]
                    scores[indices] = score_public_inputs(actor, obs[indices], [infos[j] for j in indices])
                    records = getattr(actor, "learning_records", None)
                    if records is not None:
                        learning.update(zip(indices, records))
            measured["inference_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            for j,i in enumerate(selected_indices):
                side = ready[i]
                state, delay, _ = requests[i]
                move = execution_for_action(candidates[i], int(scores[j].argmax()), pace, delay=delay)
                sample = learning.get(j)
                if sample is not None and sample["action"] != move["placement"]["action"]:
                    raise RuntimeError("learning target differs from controller action")
                pool.install(side, move, delay=delay)
                statistics[side]["spawn_wait_frames"] += delay
                pair, physical = divmod(side, 2)
                row = {"frame":int(pool.states[side].frame), "side":physical, "delay":delay, "cache":"disabled",
                    "placement":move["placement"], "controller_frames":move["controller_frames"],
                    "board":list(pool.states[side].board), "opponent":list(pool.states[side ^ 1].board),
                    "pill":state["pill"], "preview":state["preview"], "speed_ups":state["speed_ups"]}
                if sample is not None:
                    row["learning"] = sample
                moves[pair].append(row)
            measured["witness_seconds"] += time.perf_counter()-tick
        output = []
        for pair,(seed,assignment,index) in enumerate(jobs):
            moves[pair].sort(key=lambda move:(move["frame"],move["side"]))
            a, b = 2*pair+assignment, 2*pair+1-assignment
            end = pool.states[a]
            outcome = end.outcome if end.terminal else None
            score = (
                None if outcome is None else 1.0 if outcome == 1 else 0.0 if outcome == 2 else 0.5
            )
            reason = "timeout" if not end.terminal else "clear" if any(pool.states[s].event_type == 1 for s in (a,b)) else "topout"
            row = {
                "seed": seed,
                "side": assignment,
                "index": index,
                "score": score,
                "winner": None
                if score is None
                else "a"
                if score == 1
                else "b"
                if score == 0
                else "draw",
                "reason": reason,
                "frames": int(end.frame),
                "a_stats": dict(statistics[a]),
                "b_stats": dict(statistics[b]),
            }
            output.append((row, moves[pair], []))
    elapsed = time.perf_counter()-started
    if metrics is not None:
        metrics.update(measured)
        metrics["other_seconds"] = elapsed-sum(measured.values())
    return output, elapsed
