"""Causal batching must preserve each pair's frame-level trajectory."""
import os

import numpy as np
import pytest

from drmc_rl.envs.backends.vs_frames import EventVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.anticipation import execution_for_action
from drmc_rl.human.backend import plan_candidates
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import run_batch


class FixedPolicy:
    def score(self, observations, infos):
        masks = np.stack([i["placements/feasible_mask"].reshape(512) for i in infos]).astype(bool)
        costs = np.stack([i["placements/cost_to_lock"].reshape(512) for i in infos])
        logits = -costs.astype(np.float32)-np.arange(512,dtype=np.float32)[None]*.0001
        return np.broadcast_to(np.arange(512),masks.shape), masks, logits


@pytest.mark.parametrize("pace", ["sloth","relaxed","normal","fast","top_humans"])
@pytest.mark.parametrize("asynchronous", [False,True])
def test_event_batch_matches_reference_inputs_and_outcomes(pace,asynchronous):
    config = {"native_library":os.environ.get("DRMC_FRAME_LIBRARY"),
        "variants":{"a":{"delay":4},"b":{"delay":4}},"max_game_frames":3000,"replay_games":0,
        "async_planning":asynchronous}
    match = {"a":"a","b":"b","games":4,"level":14,"pace":pace}
    jobs = [(19071,0,0),(19071,1,1),(17291,0,2),(17291,1,3)]
    planner, parallel, actor = NativeReachabilityRunner(), ParallelPlanning(2), FixedPolicy()
    try:
        reference,_ = run_batch(config,match,jobs,actor,planner,None)
        batched,_ = run_event_batch(config,match,jobs,actor,parallel,None)
        for (expected,moves,_),(actual,event_moves,_) in zip(reference,batched):
            assert event_moves == moves
            assert {k:v for k,v in actual.items() if not k.endswith("stats")} == {k:v for k,v in expected.items() if not k.endswith("stats")}
            for side in ("a_stats","b_stats"):
                for field in ("decisions","validated_input_frames","no_reachable_after_delay","feasible_candidates","spawn_wait_frames"):
                    assert actual[side].get(field,0) == expected[side].get(field,0)
    finally:
        parallel.close()
        planner.close()


def test_event_executor_rejects_a_corrupted_controller_witness():
    with EventVsPool(1,lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
        pool.reset([17291],level=14)
        assert all(p.needs_action for p in pool.advance(1000))
        planner = NativeReachabilityRunner()
        pace = resolve_pace("normal")
        for side in (0,1):
            candidate = plan_candidates(planner,pool.states[side].semantic(pool.states[side^1]),22,pace)
            move = execution_for_action(candidate,int(candidate[-2].actions[0]),pace,delay=22)
            pool.install(side,move,delay=22)
        pool.script_storage[0][0].x ^= 1
        with pytest.raises(RuntimeError,match="-5"):
            pool.advance(1000)
        pool.reset([17291],level=14)
        assert all(p.needs_action for p in pool.advance(1000))
