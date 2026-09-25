"""Coalesced rollout inference: same per-row results, no deadlocks, clean failure."""
import os
import random
import threading
import time

import numpy as np
import pytest
import torch

from drmc_rl.models.policy.controller_core import ControllerCorePolicy, keyed_sample
from tests.test_controller_core_training import controller_requests, parent  # noqa: F401 - fixture
from tools.coalesced_inference import InferenceClosed, InferenceHub
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.train_pace_strategy import terminal_samples
from tools.vs_head_to_head import PlainPolicy


class RowPolicy:
    """Row-independent fake: logits encode each row's own observation."""

    def __init__(self, delay=0., fail_on=None):
        self.delay, self.fail_on, self.calls, self.sizes = delay, fail_on, 0, []
        self.learning_records = None

    def score(self, obs, infos):
        self.calls += 1
        self.sizes.append(len(infos))
        if self.delay:
            time.sleep(self.delay * random.random())
        if self.fail_on is not None and any(i["id"] == self.fail_on for i in infos):
            raise ValueError("forward failed")
        logits = np.repeat(obs.reshape(len(obs), 1).astype(np.float32), 4, 1)
        self.learning_records = [dict(id=i["id"]) for i in infos]
        return np.tile(np.arange(4), (len(obs), 1)), np.ones((len(obs), 4), bool), logits


def _rows(n, start):
    return np.arange(start, start + n, dtype=np.float32), [dict(id=int(v)) for v in range(start, start + n)]


def test_seven_threads_with_random_delays_get_their_own_rows():
    random.seed(4)
    learner, opponent = RowPolicy(.002), RowPolicy(.002)
    errors = []
    with InferenceHub(window=.003, timeout=30) as hub:
        def collect(thread):
            try:
                proxies = hub.proxy(learner, "learner"), hub.proxy(opponent, "opponent")
                with hub.client():
                    for step in range(60):
                        time.sleep(random.random() * .002)  # engine / planning
                        proxy = proxies[step % 2]
                        n = random.randint(1, 5)
                        obs, infos = _rows(n, thread * 100000 + step * 10)
                        actions, masks, logits = proxy.score(obs, infos)
                        assert logits.shape == (n, 4) and np.array_equal(logits[:, 0], obs)
                        assert [r["id"] for r in proxy.learning_records] == [i["id"] for i in infos]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        threads = [threading.Thread(target=collect, args=(t,)) for t in range(7)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not any(t.is_alive() for t in threads)
    assert not errors, errors
    # Requests really were merged across threads.
    assert max(learner.sizes) > 5 and learner.calls < 7 * 30


def test_forward_errors_reach_every_waiter_and_later_requests_still_run():
    policy = RowPolicy(fail_on=3)
    with InferenceHub(window=.05, timeout=30) as hub:
        results = {}
        barrier = threading.Barrier(2)

        def call(start):
            with hub.client():
                barrier.wait()
                try:
                    results[start] = hub.proxy(policy).score(*_rows(2, start))[2]
                except ValueError as error:
                    results[start] = error
        threads = [threading.Thread(target=call, args=(s,)) for s in (2, 10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        # Both requests shared the failing forward.
        assert policy.calls == 1 and all(isinstance(r, ValueError) for r in results.values())
        assert np.array_equal(hub.proxy(policy).score(*_rows(2, 20))[2][:, 0], [20, 21])


def test_timeout_and_close_unblock_waiters_with_clear_errors():
    class Stuck(RowPolicy):
        def score(self, obs, infos):
            time.sleep(1.)
            return super().score(obs, infos)

    hub = InferenceHub(window=0, timeout=.2)
    with pytest.raises(RuntimeError, match="did not answer within"):
        hub.proxy(Stuck()).score(*_rows(1, 0))
    hub.close()
    with pytest.raises(InferenceClosed):
        hub.proxy(RowPolicy()).score(*_rows(1, 0))

    hub = InferenceHub(window=10., timeout=30)
    policy = RowPolicy()
    outcome = []

    def waiter():
        with hub.client():
            try:
                hub.proxy(policy).score(*_rows(1, 0))
                outcome.append("answered")
            except InferenceClosed:
                outcome.append("closed")
    # A second registered client that never submits keeps the 10 s window open.
    other = threading.Event()

    def idle():
        with hub.client():
            other.wait(5)
    threads = [threading.Thread(target=idle), threading.Thread(target=waiter)]
    for t in threads:
        t.start()
    time.sleep(.2)
    started = time.monotonic()
    hub.close(InferenceClosed("stopped by KeyboardInterrupt"))
    other.set()
    for t in threads:
        t.join(5)
    assert time.monotonic() - started < 3
    assert outcome in (["closed"], ["answered"])


def test_flush_is_immediate_once_every_client_waits():
    policy = RowPolicy()
    with InferenceHub(window=5., timeout=30) as hub:
        started = time.monotonic()
        with hub.client():
            hub.proxy(policy).score(*_rows(3, 0))
        assert time.monotonic() - started < 1.


def test_keyed_sampling_is_a_valid_draw_and_depends_only_on_its_key():
    logp = np.log(np.asarray([.1, 1e-300, .6, .3]))
    draws = [keyed_sample(logp, 5, (1, g, 0, d)) for g in range(40) for d in range(100)]
    counts = np.bincount(draws, minlength=4) / len(draws)
    assert counts[1] == 0 and np.allclose(counts, [.1, 0, .6, .3], atol=.02)
    assert keyed_sample(logp, 5, (1, 2, 0, 3)) == keyed_sample(logp + 1e-7, 5, (1, 2, 0, 3))
    with pytest.raises(ValueError):
        keyed_sample(logp, 5, None)


def test_coalesced_learner_rows_match_single_row_scoring(parent):  # noqa: F811
    actor = ControllerCorePolicy(parent, seed=91)
    actor.sampling_seed = 44
    obs, infos = controller_requests(actor)
    for n, info in enumerate(infos):
        info["sampling/key"] = (7, n, 0, 1)
    singles = []
    for i in range(len(infos)):
        actor.score(obs[i:i + 1], infos[i:i + 1])
        singles.append(actor.learning_records[0])
    with InferenceHub(window=.05, timeout=30) as hub:
        got = [None] * len(infos)
        barrier = threading.Barrier(len(infos))

        def call(i):
            proxy = hub.proxy(actor)
            with hub.client():
                barrier.wait()
                proxy.score(obs[i:i + 1], infos[i:i + 1])
            got[i] = proxy.learning_records[0]
        threads = [threading.Thread(target=call, args=(i,)) for i in range(len(infos))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert hub.stats["forwards"] == 1
    for single, row in zip(singles, got):
        assert row["collection_shape"][0] > 1
        assert row["slot"] == single["slot"] and row["action"] == single["action"]
        np.testing.assert_allclose(row["behavior_logp"], single["behavior_logp"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(row["base_logits"], single["base_logits"], rtol=0, atol=1e-5)
        assert abs(row["old_value"] - single["old_value"]) < 1e-5
        # The sampled likelihood comes from the forward that sampled it.
        assert row["old_logprob"] == row["behavior_logp"][row["slot"]]


def _concurrent_games(parent, coalesce):  # noqa: F811
    torch.manual_seed(3)
    learner = ControllerCorePolicy(parent, seed=12)
    learner.sampling_seed, learner.defer_reference = 99, True
    opponent = PlainPolicy(parent, "cpu", public_only=True)
    config = {"native_library": os.environ.get("DRMC_FRAME_LIBRARY"), "async_planning": True,
              "fill_inference_batches": True, "variants": {"learner": {"delay": 4}, "parent": {"delay": 4}},
              "max_game_frames": 2500, "replay_games": 0}
    schedules = [(dict(id=f"train-1-{pace}", a="learner", b="parent", games=4, pace=pace, level=14),
                  [(seed, side, 2 * i + side) for i, seed in enumerate((19071 + k, 17291 + k)) for side in (0, 1)])
                 for k, pace in enumerate(("sloth", "normal", "fast", "top_humans", "super_human",
                                           "frame_perfect", "relaxed"))]
    planner = ParallelPlanning(2)
    results = [None] * len(schedules)
    try:
        if not coalesce:
            for k, (match, jobs) in enumerate(schedules):
                results[k] = run_event_batch(config, match, jobs, None, planner, None,
                                             policies={"learner": learner, "parent": opponent})[0]
        else:
            with InferenceHub(window=.005, timeout=120) as hub:
                def run(k):
                    match, jobs = schedules[k]
                    with hub.client():
                        results[k] = run_event_batch(config, match, jobs, None, planner, None, policies={
                            "learner": hub.proxy(learner), "parent": hub.proxy(opponent)})[0]
                threads = [threading.Thread(target=run, args=(k,)) for k in range(len(schedules))]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(300)
                shapes = hub.stats
    finally:
        planner.close()
    rows = [r for batch in results for r in terminal_samples(batch)]
    learner.fill_reference_logits(rows)
    return results, rows, (shapes if coalesce else None)


def test_mixed_learner_opponent_reference_games_match_sequential_collection(parent):  # noqa: F811
    sequential, seq_rows, _ = _concurrent_games(parent, False)
    coalesced, co_rows, stats = _concurrent_games(parent, True)
    assert stats["rows"] / stats["forwards"] > 1.5
    for a, b in zip(sequential, coalesced):
        for (ra, ma, _), (rb, mb, _) in zip(a, b):
            assert {k: v for k, v in ra.items() if k != "a_stats"} == {k: v for k, v in rb.items() if k != "a_stats"}
            strip = lambda moves: [{k: v for k, v in m.items() if k != "learning"} for m in moves]  # noqa: E731
            assert strip(ma) == strip(mb)
    assert len(seq_rows) == len(co_rows) > 0
    for a, b in zip(seq_rows, co_rows):
        assert a["slot"] == b["slot"]
        np.testing.assert_allclose(a["behavior_logp"], b["behavior_logp"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(a["base_logits"], b["base_logits"], rtol=0, atol=1e-5)
