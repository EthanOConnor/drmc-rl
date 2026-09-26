"""Pool worker batching: shared-base forwards and filled batches keep decisions exact."""
import os

import numpy as np
import pytest
import torch

from drmc_rl.style import knobs
from tools.trainer_arena_cache import MemoPolicy
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch


class CountingPolicy:
    """Row-independent deterministic scorer; counts forwards and rows."""

    aux_spec = None

    def __init__(self, salt=0.):
        self.salt, self.calls, self.rows = salt, 0, 0

    def score(self, observations, infos):
        self.calls += 1
        self.rows += len(infos)
        masks = np.stack([np.asarray(i["placements/feasible_mask"]).reshape(512) for i in infos]).astype(bool)
        costs = np.stack([np.asarray(i["placements/cost_to_lock"]).reshape(512) for i in infos]).astype(np.float32)
        board = np.asarray(observations, np.float32).reshape(len(infos), -1).sum(1, keepdims=True)
        logits = -costs * .01 + np.sin(np.arange(512, dtype=np.float32)[None] * .37 + board * .013 + self.salt)
        return np.broadcast_to(np.arange(512), masks.shape), masks, logits


def _config(**extra):
    return {"native_library": os.environ.get("DRMC_FRAME_LIBRARY"), "variants": {"a": {"delay": 4}, "b": {"delay": 4}},
            "max_game_frames": 4000, "replay_games": 0, "async_planning": False, **extra}


JOBS = [(19071, 0, 0), (19071, 1, 1), (17291, 0, 2), (17291, 1, 3), (40001, 0, 4), (40001, 1, 5)]


def _play(policies, pace="fast", **extra):
    planner = ParallelPlanning(2)
    try:
        match = {"a": "a", "b": "b", "games": len(JOBS), "level": 14, "pace": pace}
        return run_event_batch(_config(**extra), match, JOBS, None, planner, None, policies=policies)[0]
    finally:
        planner.close()


def _strip(batch):
    return [({k: v for k, v in row.items() if not k.endswith("stats")}, moves) for row, moves, _ in batch]


@pytest.mark.parametrize("pace", ["normal", "fast", "frame_perfect"])
def test_knob_entrant_against_its_base_plays_identical_games_with_half_the_forwards(pace):
    quad = [knobs.parse("showy-quad@1:1.5")]
    separate, other = MemoPolicy(CountingPolicy()), MemoPolicy(CountingPolicy())
    reference = _play({"a": knobs.apply(separate, quad), "b": other}, pace)
    shared_base = CountingPolicy()
    shared = MemoPolicy(shared_base)
    merged = _play({"a": knobs.apply(shared, quad), "b": shared}, pace, share_base_forwards=True)
    assert _strip(merged) == _strip(reference)
    decisions = sum(len(m) for _, m, _ in reference)
    assert decisions > 50
    # One forward per decision step for both entrants together.
    before = separate.policy.calls + other.policy.calls
    assert shared_base.calls < before, (shared_base.calls, before)
    assert shared_base.rows <= separate.policy.rows + other.policy.rows


def test_different_bases_and_two_knob_variants_share_only_their_own_network():
    quad, t2 = [knobs.parse("showy-quad@1:1.5")], [knobs.parse("showy-t2@1:1.5")]
    ref = _play({"a": knobs.apply(MemoPolicy(CountingPolicy()), quad),
                 "b": knobs.apply(MemoPolicy(CountingPolicy()), t2)})
    base = MemoPolicy(CountingPolicy())
    got = _play({"a": knobs.apply(base, quad), "b": knobs.apply(base, t2)}, share_base_forwards=True)
    assert _strip(got) == _strip(ref)
    ref = _play({"a": MemoPolicy(CountingPolicy(.5)), "b": MemoPolicy(CountingPolicy())})
    got = _play({"a": MemoPolicy(CountingPolicy(.5)), "b": MemoPolicy(CountingPolicy())}, share_base_forwards=True)
    assert _strip(got) == _strip(ref)


def test_filled_asynchronous_batches_play_the_same_games():
    ref = _play({"a": MemoPolicy(CountingPolicy(.5)), "b": MemoPolicy(CountingPolicy())})
    got = _play({"a": MemoPolicy(CountingPolicy(.5)), "b": MemoPolicy(CountingPolicy())},
                async_planning=True, fill_inference_batches=True, share_base_forwards=True)
    assert _strip(got) == _strip(ref)


def test_shared_base_forwards_refuse_training_actors():
    class Learner(CountingPolicy):
        training = True

    with pytest.raises(ValueError, match="frozen evaluation"):
        _play({"a": Learner(), "b": CountingPolicy()}, share_base_forwards=True)


def test_worker_loads_one_network_for_knob_variants_of_a_checkpoint(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from drmc_rl.pool import worker
    import tools.trainer_planning_arena as arena

    loads = []
    monkeypatch.setattr(arena, "_variant_actor", lambda config, params, parent: loads.append(params) or CountingPolicy())
    runtimes = worker.Runtimes(SimpleNamespace(), max_loaded=4)
    runtime = SimpleNamespace(config={}, policy=None)
    plain = dict(checkpoint="x.pt")
    knobbed = dict(plain, knobs=[knobs.parse("showy-quad@1:1.5")])
    a = runtimes.policy(runtime, plain, json.dumps(plain, sort_keys=True))
    b = runtimes.policy(runtime, knobbed, json.dumps(knobbed, sort_keys=True))
    assert len(loads) == 1 and b.inner is a
    torch.manual_seed(0)
