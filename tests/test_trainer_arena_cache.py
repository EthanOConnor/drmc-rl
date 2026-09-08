import numpy as np
from dataclasses import replace

from drmc_rl.planning.fast_reach import FrameState
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import ByteCache, MemoPlanner, MemoPolicy


def test_byte_cache_evicts_and_replaces_within_budget():
    cache = ByteCache(12)
    cache.put(b"a", 1, 5)
    cache.put(b"b", 2, 5)
    assert cache.get(b"a") == 1
    cache.put(b"c", 3, 5)
    assert cache.get(b"b") is None
    cache.put(b"a", 4, 2)
    assert cache.used == 9 and cache.get(b"a") == 4


def test_memo_planner_owns_witnesses_and_keys_every_mechanical_input():
    planner = NativeReachabilityRunner()
    memo = MemoPlanner(planner)
    columns = np.zeros(8, np.uint16)
    spawn = FrameState(x=3, y=0, rot=0, speed_counter=0, hor_velocity=0, hold_dir=0, frame_parity=0)
    first = memo.bfs_full(columns, spawn, speed_threshold=6)
    saved = first.script_buf.copy()
    assert memo.bfs_full(columns, spawn, speed_threshold=6) is first
    memo.bfs_full(columns, spawn, speed_threshold=5)
    memo.bfs_full(columns, replace(spawn, frame_parity=1), speed_threshold=6)
    memo.bfs_full(columns, spawn, speed_threshold=6, edge_interval=2)
    columns[0] = 0x8000
    memo.bfs_full(columns, spawn, speed_threshold=6)
    np.testing.assert_array_equal(first.script_buf, saved)
    assert not first.script_buf.flags.writeable
    assert memo.cache.hits == 1 and memo.cache.misses == 5
    memo.close()


def test_memo_policy_deduplicates_without_ignoring_preview_costs_or_opponent():
    class Policy:
        calls = 0
        def score(self, obs, infos):
            self.calls += len(infos)
            logits = np.array([[float(o.sum())+i["preview"]+i["cost"]] for o,i in zip(obs,infos)])
            return np.zeros((len(infos),1), int), np.ones((len(infos),1), bool), logits
    policy = Policy()
    memo = MemoPolicy(policy)
    legal = np.arange(512) == 0
    info = {"placements/feasible_mask": legal, "preview": 1, "cost": 20}
    obs = np.zeros((2,20,16,8), np.float32)
    a = memo.score(obs, [info,info])
    assert policy.calls == 1
    b = memo.score(obs, [info,info])
    np.testing.assert_array_equal(a[2], b[2])
    obs[1,8,15,0] = 1
    c = memo.score(obs, [info,{**info,"preview":2}])
    assert policy.calls == 2 and c[2][1,0] != b[2][1,0]
    memo.score(obs[:1], [{**info,"cost":21}])
    assert policy.calls == 3
