from __future__ import annotations

import threading

import numpy as np
import pytest

from envs.backends.drmario_pool import is_library_present

torch = pytest.importorskip("torch")

from models.policy.search_policy import (  # noqa: E402
    MACRO_ACTIONS,
    PonderingSearchPolicy,
    PonderResult,
    _board_key,
    _pair_index,
)

pytestmark = pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)


def _tiny_ponder(**kw) -> PonderingSearchPolicy:
    from models.policy.candidate_policy import CandidatePlacementPolicyNet

    torch.manual_seed(0)
    net = CandidatePlacementPolicyNet(
        in_channels=12,
        board_channels=8,
        board_encoder="cnn",
        encoder_blocks=0,
        d_model=32,
        pill_embed_dim=32,
        pill_embed_type="ordered_pair",
        aux_dim=0,
        pos_embed_dim=8,
        cost_embed_dim=8,
        cand_hidden_dim=32,
        patch_kernel=3,
    )
    kw.setdefault("beam", 4)
    kw.setdefault("deadline_ms", 10_000.0)
    return PonderingSearchPolicy.from_net(net, candidate_max=64, num_sim_envs=16, **kw)


def _root_state(sp: PonderingSearchPolicy):
    board = np.full((16, 8), 0xFF, dtype=np.uint8)
    board[12:, 1] = 0xD0
    board[13:, 5] = 0xD1
    board = board.reshape(-1)
    from envs.backends.drmario_pool import build_reset_spec

    spec = build_reset_spec(
        level=5,
        speed_setting=2,
        rng_state=(7, 9),
        rng_override=True,
        checkpoint_enabled=True,
        checkpoint_board=board,
        checkpoint_falling_colors=(0, 1),
        checkpoint_preview_colors=(2, 0),
    )
    sp._runner.reset(None, [spec] * sp.num_sim_envs)
    mask = sp._runner.buffers.feasible_mask[0].copy()
    cost = sp._runner.buffers.cost_to_lock[0].copy()
    return board, mask, cost


def _stub_result(board: np.ndarray, pill_raw, actions, q, refined=None) -> PonderResult:
    actions = np.asarray(actions, dtype=np.int64)
    q = np.asarray(q, dtype=np.float64)
    return PonderResult(
        board_key=_board_key(board),
        pill_raw=(int(pill_raw[0]), int(pill_raw[1])),
        actions=actions,
        q=q,
        refined=np.ones_like(q, dtype=bool) if refined is None else np.asarray(refined, bool),
        prior_action=np.full((9,), int(actions[0]), dtype=np.int64),
        depth=np.full((9,), 2, dtype=np.int8),
        wall_s=0.01,
    )


def _mask_for(actions) -> np.ndarray:
    mask = np.zeros(MACRO_ACTIONS, dtype=bool)
    mask[np.asarray(actions)] = True
    return mask


def _kick_stub(sp: PonderingSearchPolicy, board, mask, cost) -> None:
    """start_ponder with arbitrary-but-valid job args (compute is stubbed)."""

    sp.start_ponder(
        board,
        int(np.flatnonzero(mask)[0]) if mask.any() else 0,
        (0, 1),
        (2, 0),
        feasible_mask512=mask if mask.any() else _mask_for([5]),
        cost_to_lock512=cost,
        speed_setting=2,
        speed_ups=0,
        level=5,
        viruses_initial=4,
        frames_used=0,
    )


# ----------------------------------------------------------- cache semantics
def test_hit_uses_revealed_preview_column_and_is_fast() -> None:
    sp = _tiny_ponder()
    try:
        board = np.full(128, 0xFF, dtype=np.uint8)
        board[100] = 0xD0
        a1, a2 = 10, 20
        q = np.zeros((2, 9))
        c_a = _pair_index(0, 1)  # preview raw (0,1)
        c_b = _pair_index(2, 2)  # preview raw (2,2)
        q[0, c_a], q[1, c_a] = 1.0, 0.5  # pair A prefers a1
        q[0, c_b], q[1, c_b] = 0.5, 1.0  # pair B prefers a2
        result = _stub_result(board, (1, 2), [a1, a2], q)
        sp._ponder_compute = lambda job: result  # type: ignore[method-assign]
        cost = np.ones(MACRO_ACTIONS, dtype=np.uint16)
        _kick_stub(sp, board, _mask_for([a1, a2]), cost)
        assert sp.wait_ponder_idle(10.0)

        mask = _mask_for([a1, a2])
        act, info = sp.decide(board, (1, 2), (0, 1), mask, cost, 2, 0, 5)
        assert act == a1
        assert info["ponder_hit"] is True and info["stage"] == "ponder"
        assert info["value_best"] == pytest.approx(1.0)
        assert info["elapsed_ms"] < 5.0
        act_b, info_b = sp.decide(board, (1, 2), (2, 2), mask, cost, 2, 0, 5)
        assert act_b == a2 and info_b["ponder_hit"] is True
        assert sp.ponder_stats()["hits"] == 2
    finally:
        sp.close()


def test_hit_restricted_to_caller_feasible_mask() -> None:
    sp = _tiny_ponder()
    try:
        board = np.full(128, 0xFF, dtype=np.uint8)
        a1, a2 = 10, 20
        q = np.full((2, 9), 0.0)
        q[0, :], q[1, :] = 1.0, 0.5
        result = _stub_result(board, (1, 2), [a1, a2], q)
        sp._ponder_compute = lambda job: result  # type: ignore[method-assign]
        cost = np.ones(MACRO_ACTIONS, dtype=np.uint16)
        _kick_stub(sp, board, _mask_for([a1, a2]), cost)
        assert sp.wait_ponder_idle(10.0)

        # Caller mask no longer allows the pondered best (a1): take next best.
        act, info = sp.decide(board, (1, 2), (0, 0), _mask_for([a2, 30]), cost, 2, 0, 5)
        assert act == a2 and info["ponder_hit"] is True
    finally:
        sp.close()


def test_hit_prefers_refined_entries_over_higher_depth1_estimates() -> None:
    """Regression: the argmax must not mix depth-1 estimates with depth-2
    backups (the depth-1 V bias degenerated the probe to a depth-1 search)."""

    sp = _tiny_ponder()
    try:
        board = np.full(128, 0xFF, dtype=np.uint8)
        a1, a2, a3 = 10, 20, 30
        q = np.zeros((3, 9))
        q[0, :], q[1, :], q[2, :] = 0.4, 0.6, 5.0  # a3: inflated depth-1 estimate
        refined = np.zeros((3, 9), dtype=bool)
        refined[0, :] = refined[1, :] = True  # only a1, a2 got the ply-2 backup
        result = _stub_result(board, (1, 2), [a1, a2, a3], q, refined=refined)
        sp._ponder_compute = lambda job: result  # type: ignore[method-assign]
        cost = np.ones(MACRO_ACTIONS, dtype=np.uint16)
        _kick_stub(sp, board, _mask_for([a1, a2, a3]), cost)
        assert sp.wait_ponder_idle(10.0)

        mask = _mask_for([a1, a2, a3])
        act, info = sp.decide(board, (1, 2), (0, 1), mask, cost, 2, 0, 5)
        assert act == a2  # best refined, not the inflated depth-1 a3
        assert info["ponder_refined"] is True

        # With no refined entry feasible, the depth-1 ranking is the fallback.
        act2, info2 = sp.decide(board, (1, 2), (0, 1), _mask_for([a3]), cost, 2, 0, 5)
        assert act2 == a3 and info2["ponder_refined"] is False
    finally:
        sp.close()


def test_miss_on_board_divergence_falls_back_to_search() -> None:
    sp = _tiny_ponder()
    try:
        board, mask, cost = _root_state(sp)
        other = board.copy()
        other[64] = 0xD2  # garbage landed: divergence
        result = _stub_result(other, (0, 1), [int(np.flatnonzero(mask)[0])], np.ones((1, 9)))
        sp._ponder_compute = lambda job: result  # type: ignore[method-assign]
        _kick_stub(sp, board, mask, cost)
        assert sp.wait_ponder_idle(10.0)

        act, info = sp.decide(board, (0, 1), (2, 0), mask, cost, 2, 0, 5)
        assert info["ponder_hit"] is False
        assert info["stage"] == "depth2"
        assert act >= 0 and bool(mask[act])
        stats = sp.ponder_stats()
        assert stats["misses"] == 1 and stats["hits"] == 0
    finally:
        sp.close()


def test_invalidate_clears_cache() -> None:
    sp = _tiny_ponder()
    try:
        board = np.full(128, 0xFF, dtype=np.uint8)
        result = _stub_result(board, (1, 2), [10], np.ones((1, 9)))
        sp._ponder_compute = lambda job: result  # type: ignore[method-assign]
        cost = np.ones(MACRO_ACTIONS, dtype=np.uint16)
        _kick_stub(sp, board, _mask_for([10]), cost)
        assert sp.wait_ponder_idle(10.0)
        assert sp.has_ponder_for(board, (1, 2))
        sp.ponder_invalidate()
        assert not sp.has_ponder_for(board, (1, 2))
    finally:
        sp.close()


# ---------------------------------------------------------- thread lifecycle
def test_supersede_aborts_running_job_newest_wins() -> None:
    sp = _tiny_ponder()
    try:
        board_a = np.full(128, 0xFF, dtype=np.uint8)
        board_b = board_a.copy()
        board_b[0] = 0xD0
        res_b = _stub_result(board_b, (1, 2), [10], np.ones((1, 9)))
        started = threading.Event()
        calls: list[int] = []

        def stub(job):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                # Block until superseded; abort is set by the next start_ponder.
                assert job.abort.wait(10.0)
                return None
            return res_b

        sp._ponder_compute = stub  # type: ignore[method-assign]
        cost = np.ones(MACRO_ACTIONS, dtype=np.uint16)
        _kick_stub(sp, board_a, _mask_for([10]), cost)
        assert started.wait(10.0)
        _kick_stub(sp, board_b, _mask_for([10]), cost)  # supersede: newest wins
        assert sp.wait_ponder_idle(10.0)

        assert not sp.has_ponder_for(board_a, (1, 2))
        assert sp.has_ponder_for(board_b, (1, 2))
        stats = sp.ponder_stats()
        assert stats["aborts"] >= 1
        assert stats["jobs_started"] == 2 and stats["jobs_completed"] == 1
    finally:
        sp.close()


def test_close_joins_worker_thread() -> None:
    sp = _tiny_ponder()
    worker = sp._ponder_thread
    assert worker is not None and worker.is_alive()
    sp.close()
    assert sp._ponder_thread is None
    assert not worker.is_alive()


# ------------------------------------------------------------- end to end
def test_real_ponder_end_to_end_hits_next_decision() -> None:
    sp = _tiny_ponder()
    try:
        board, mask, cost = _root_state(sp)
        committed = int(np.flatnonzero(mask)[0])
        sp.start_ponder(
            board,
            committed,
            (0, 1),
            (2, 0),
            feasible_mask512=mask,
            cost_to_lock512=cost,
            speed_setting=2,
            speed_ups=0,
            level=5,
            viruses_initial=4,
            frames_used=0,
        )
        assert sp.wait_ponder_idle(120.0)
        stats = sp.ponder_stats()
        assert stats["jobs_completed"] == 1, stats
        assert stats["pairs_depth2"] == 9  # generous budget: all pairs refined

        # Replicate the committed step to get the true next decision context.
        from envs.backends.drmario_pool import build_reset_spec

        spec = build_reset_spec(
            level=5,
            speed_setting=2,
            rng_state=(31, 17),  # different seed: outcome must be color/board determined
            rng_override=True,
            checkpoint_enabled=True,
            checkpoint_board=board,
            checkpoint_falling_colors=(0, 1),
            checkpoint_preview_colors=(2, 0),
        )
        sp._runner.reset(None, [spec] * sp.num_sim_envs)
        acts = np.full((sp.num_sim_envs,), -1, dtype=np.int32)
        acts[0] = committed
        sp._runner.step(acts, None, None)
        buf = sp._runner.buffers
        assert int(buf.terminated[0]) == 0 and int(buf.invalid_action[0]) == -1
        board1 = buf.board_bytes[0].copy()
        mask1 = buf.feasible_mask[0].copy().astype(bool)
        cost1 = buf.cost_to_lock[0].copy()

        assert sp.has_ponder_for(board1, (2, 0))
        act, info = sp.decide(board1, (2, 0), (1, 1), mask1, cost1, 2, 0, 5)
        assert info["ponder_hit"] is True and info["stage"] == "ponder"
        assert info["ponder_depth"] == 2
        assert act >= 0 and bool(mask1[act])
        assert sp.ponder_stats()["hits"] == 1
    finally:
        sp.close()
