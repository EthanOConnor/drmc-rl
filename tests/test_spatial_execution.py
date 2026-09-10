import os

import numpy as np
import pytest
import torch

from drmc_rl.game.pair_state import DecisionBoundary, PairEvent, PairEventKind, PublicPairState, VisibleSideState
from drmc_rl.human.backend import POSE_TO_ACTION
from drmc_rl.human.spatial_execution import ConstructionObserver, payoff
from drmc_rl.human.spatial_proposer import RECURRENT_PUBLIC, SpatialProposer


class ZeroEncoder:
    def __call__(self, boards, pills, previews):
        return boards.new_zeros((len(boards), 8))


def models():
    torch.manual_seed(718)
    output = {}
    for name in ("persistent", "stateless"):
        model = SpatialProposer(8, 8, persistent=name == "persistent",
                                plan_update_schema=RECURRENT_PUBLIC).eval()
        with torch.no_grad():
            model.intent.weight.zero_()
            model.intent.bias[:] = torch.tensor([20., -20., -20., -20.])
            model.target.weight.zero_()
            model.target.bias.fill_(-20)
            model.target.bias[120] = 20
            model.horizon.weight.zero_()
            model.horizon.bias[:] = torch.tensor([-20., -20., 20., -20., -20., -20.])
        output[name] = model
    return output


def view(board, frame, events=(), *, terminal=False):
    side = VisibleSideState(bytes(board), (0, 0), (1, 1), None)
    return PublicPairState(frame, 1, (side, side),
        DecisionBoundary.TERMINAL if terminal else DecisionBoundary.P2, tuple(events))


def event(kind, frame, **payload):
    return PairEvent(kind, frame, 1, payload)


def setup():
    board = np.full(128, 0xFF, np.uint8)
    board[120:123] = 0xD1
    observer = ConstructionObserver(models(), ZeroEncoder(), [(987, 1, 0)])
    observer.observe({1: view(board, 1)})
    observer.decide([dict(side=1, frame=1, board=bytes(board), pill=(0, 0),
        preview=(1, 1), action=123, feasible=(123, 124, 125))])
    pose = int(np.flatnonzero(POSE_TO_ACTION == 123)[0])
    rotation, cell = divmod(pose, 128)
    row, column = divmod(cell, 8)
    lock = event(PairEventKind.LOCK, 2, rotation=rotation, row_top=row, column=column)
    return observer, board, lock


def test_payoff_waits_for_observed_clear_and_settled_bottle():
    observer, board, lock = setup()
    pending = observer.states[1]["pending"]
    after = pending["result"].settled_field
    observer.observe({1: view(board, 2, [lock])})
    assert not observer.plans
    assert all(record["plan"].elapsed == 0 for record in observer.states[1]["plans"].values())
    clear = event(PairEventKind.CLEAR, 3, tiles_cleared=5, viruses_cleared=3)
    observer.observe({1: view(after, 3, [lock, clear])})
    assert not observer.plans
    spawn = event(PairEventKind.SPAWN, 8)
    observer.observe({1: view(after, 8, [lock, clear, spawn])})
    assert observer.states[1]["counters"]["verified_completions"] == 1
    assert len(observer.plans) == 2
    assert all(p["reason"] == "spatial_goal_observed" and p["root_goal_observed"]
               and p["completed_placements"] == 1 for p in observer.plans)
    observer.observe({1: view(after, 8, [lock, clear, spawn])})
    assert observer.states[1]["counters"]["verified_completions"] == 1


def test_revised_anchor_payoff_does_not_credit_original_target():
    observer, board, lock = setup()
    for record in observer.states[1]["plans"].values():
        record["root_anchor"] = 0  # original top-left target was never cleared
    pending = observer.states[1]["pending"]
    observer.observe({1: view(pending["result"].settled_field, 8, [
        lock, event(PairEventKind.CLEAR, 3, tiles_cleared=5, viruses_cleared=3),
        event(PairEventKind.SPAWN, 8)])})
    assert all(p["reason"] == "spatial_goal_observed" and not p["root_goal_observed"]
               for p in observer.plans)


@pytest.mark.parametrize("fault", ["garbage", "wrong_board", "wrong_counts", "history_gap", "terminal"])
def test_incomplete_or_interrupted_transitions_are_not_payoffs(fault):
    observer, board, lock = setup()
    after = observer.states[1]["pending"]["result"].settled_field
    events = [lock, event(PairEventKind.CLEAR, 3, tiles_cleared=4 if fault=="wrong_counts" else 5,
                          viruses_cleared=3)]
    if fault == "garbage":
        events.append(event(PairEventKind.VOLLEY, 4))
    elif fault == "wrong_board":
        after = bytes(board)
    elif fault == "history_gap":
        events += [event(PairEventKind.OBSERVATION, 4)] * 29
    elif fault == "terminal":
        events = []
    if fault != "terminal":
        events.append(event(PairEventKind.SPAWN, 8))
    if fault == "history_gap":
        events = events[:31] + [event(PairEventKind.SPAWN, 8)]
    observer.observe({1: view(after, 8, events, terminal=fault=="terminal")})
    assert len(observer.plans) == 2
    assert not any(p["root_goal_observed"] for p in observer.plans)
    assert observer.states[1]["counters"]["verified_completions"] == 0


def test_goal_geometry_cannot_borrow_unrelated_vertical_clear_cells():
    from drmc_rl.game.cascade import CascadeResult, CascadeStep, ClearedCell
    horizontal = tuple(ClearedCell(15, c, 1, True) for c in range(4))
    vertical = tuple(ClearedCell(r, 7, 2, True) for r in range(8, 12))
    result = CascadeResult((CascadeStep(horizontal+vertical),), bytes([255]*128), 8, 8)
    goals, cells = payoff(result)
    assert 0 in goals and 2 in goals
    assert (8, 7, 2) not in cells[0] and (8, 7, 2) in cells[2]


def test_shadow_observer_preserves_actual_native_controller_games():
    from tests.test_event_rollout import FixedPolicy
    from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
    torch.set_num_threads(1)
    config = dict(native_library=os.environ.get("DRMC_FRAME_LIBRARY"),
                  variants={"a": {"delay": 4}, "b": {"delay": 4}},
                  max_game_frames=3000, replay_games=0)
    match = dict(a="a", b="b", games=2, level=14, pace="normal")
    jobs = [(19071, 0, 0), (19071, 1, 1)]
    observer = ConstructionObserver(models(), ZeroEncoder(), jobs)
    planner = ParallelPlanning(2)
    try:
        original, _ = run_event_batch(config, match, jobs, FixedPolicy(), planner, None)
        shadow, _ = run_event_batch(config, match, jobs, FixedPolicy(), planner, None, observer=observer)
    finally:
        planner.close()
    observer.close()
    assert original == shadow
    assert observer.decisions and all(d["feasible_count"] > 0 for d in observer.decisions)
    assert sum(s["counters"]["verified_completions"] for s in observer.states.values()) > 0
    assert all(d["arm"] in ("persistent", "stateless") for d in observer.decisions)
