"""Bounded exact memoization for repeated tournament positions, never the app.

Side-swapped games often ask identical questions. Reusing their full answers
saves CPU/GPU work without changing the simulated on-device compute budget.
"""
from collections import OrderedDict
import json

import numpy as np


class ByteCache:
    def __init__(self, limit):
        self.limit, self.used = limit, 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value, size):
        if key in self.entries:
            self.used -= self.entries.pop(key)[1]
        size += len(key)
        if size > self.limit:
            return
        while self.used + size > self.limit:
            _, (_, removed) = self.entries.popitem(last=False)
            self.used -= removed
        self.entries[key] = value, size
        self.used += size


class MemoPlanner:
    def __init__(self, planner, limit=64*1024*1024):
        self.planner, self.cache = planner, ByteCache(limit)

    def close(self):
        self.planner.close()

    def bfs_full(self, columns, spawn, **kwargs):
        micro = tuple(int(getattr(v, "value", v)) for v in (spawn.x, spawn.y, spawn.rot, spawn.speed_counter,
            spawn.hor_velocity, spawn.hold_dir, spawn.rot_hold, spawn.frame_parity, spawn.locked))
        key = np.asarray(columns, dtype=np.uint16).tobytes() + repr((micro, sorted(kwargs.items()))).encode()
        result = self.cache.get(key)
        if result is None:
            result = self.planner.bfs_full(columns, spawn, **kwargs).copy()
            arrays = (result.costs_u16, result.offsets_u16, result.lengths_u16, result.script_buf)
            for array in arrays:
                array.flags.writeable = False
            self.cache.put(key, result, sum(a.nbytes for a in arrays))
        return result


class MemoPolicy:
    def __init__(self, policy, limit=32*1024*1024):
        self.policy, self.cache = policy, ByteCache(limit)

    def score(self, observations, infos):
        keys = [np.ascontiguousarray(obs, dtype=np.float32).tobytes() +
            json.dumps(info, sort_keys=True, separators=(",", ":"), default=lambda a: a.tolist()).encode()
            for obs,info in zip(observations, infos)]
        values = [self.cache.get(key) for key in keys]
        missing = {}
        for i,(key,value) in enumerate(zip(keys,values)):
            if value is None:
                missing.setdefault(key, i)
        computed = {}
        if missing:
            indices = list(missing.values())
            actions,masks,logits = self.policy.score(observations[indices], [infos[i] for i in indices])
            for row,(key,_) in enumerate(missing.items()):
                scores = np.full(512, -np.inf, np.float32)
                selected = actions[row,masks[row]]
                legal = np.flatnonzero(np.asarray(infos[indices[row]]["placements/feasible_mask"]).reshape(512))
                if set(selected) != set(legal) or len(selected) != len(legal):
                    raise RuntimeError("memoized policy changed feasible candidate coverage")
                scores[selected] = logits[row,masks[row]]
                scores.flags.writeable = False
                computed[key] = scores
                self.cache.put(key, scores, scores.nbytes)
        scores = np.stack([value if value is not None else computed[key] for key,value in zip(keys,values)])
        return np.broadcast_to(np.arange(512), scores.shape), np.isfinite(scores), scores
