"""Bounded exact memoization for repeated tournament positions, never the app.

Side-swapped games often ask identical questions. Reusing their full answers
saves CPU/GPU work without changing the simulated on-device compute budget.
"""
from collections import OrderedDict
from dataclasses import asdict
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
        self.aux_spec = getattr(policy, "aux_spec", None)

    def score(self, observations, infos):
        scores, _ = self.score_values(observations, infos)
        return np.broadcast_to(np.arange(512), scores.shape), np.isfinite(scores), scores

    def score_values(self, observations, infos):
        """Complete 512-wide scores and the value head, memoized together."""
        from drmc_rl.human.lookahead import score_with_value

        keys = [np.ascontiguousarray(obs, dtype=np.float32).tobytes() +
            json.dumps(info, sort_keys=True, separators=(",", ":"), default=_public_json).encode()
            for obs,info in zip(observations, infos)]
        values = [self.cache.get(key) for key in keys]
        missing = {}
        for i,(key,value) in enumerate(zip(keys,values)):
            if value is None:
                missing.setdefault(key, i)
        computed = {}
        if missing:
            indices = list(missing.values())
            scores, heads = score_with_value(self.policy, observations[indices], [infos[i] for i in indices])
            for row,key in enumerate(missing):
                entry = scores[row].copy(), np.float32(heads[row])
                entry[0].flags.writeable = False
                computed[key] = entry
                self.cache.put(key, entry, entry[0].nbytes + 4)
        entries = [value if value is not None else computed[key] for key,value in zip(keys,values)]
        return (np.stack([e[0] for e in entries]) if entries else np.empty((0, 512), np.float32),
                np.asarray([e[1] for e in entries], np.float32))


def _public_json(value):
    from drmc_rl.game.pair_state import PublicPairState
    from drmc_rl.game.public_context import PublicExecutionContext

    if type(value) is PublicPairState:
        return value.to_dict()
    if type(value) is PublicExecutionContext:
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported public memo input {type(value).__name__}")
