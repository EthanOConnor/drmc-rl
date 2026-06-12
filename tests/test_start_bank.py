from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.backends.drmario_pool import is_library_present

pytestmark = pytest.mark.skipif(
    not is_library_present(),
    reason="native pool library missing (build with: make -C game_engine libdrmario_pool)",
)

FIXTURE = Path(__file__).parent / "fixtures" / "fc_v2_events_small.jsonl"


@pytest.fixture(scope="module")
def bank_path(tmp_path_factory):
    """Build a tiny start bank npz from the fixture events."""

    from tools.build_start_bank import extract_positions_from_events, positions_to_arrays

    rows = extract_positions_from_events(
        FIXTURE.read_bytes(), per_game=8, rng=np.random.default_rng(0)
    )
    assert rows, "fixture produced no start positions"
    arrays = positions_to_arrays(rows)
    path = tmp_path_factory.mktemp("bank") / "bank.npz"
    np.savez_compressed(path, **arrays)
    return path


def test_bank_extraction_deterministic():
    from tools.build_start_bank import extract_positions_from_events

    raw = FIXTURE.read_bytes()
    rows_a = extract_positions_from_events(raw, rng=np.random.default_rng(7))
    rows_b = extract_positions_from_events(raw, rng=np.random.default_rng(7))
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        assert np.array_equal(a["boards"], b["boards"])
        assert a["stratum"] == b["stratum"]


def test_bank_row_round_trips_through_native_reset(bank_path):
    """bank row -> reset spec -> env reset -> board matches."""

    from envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
    from training.envs.start_bank import StartBank

    bank = StartBank(bank_path)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        for i in range(min(4, len(bank))):
            kwargs = bank.spec_kwargs(i)
            spec = build_vs_reset_spec(
                level=(14, 14),
                speed_setting=(2, 2),
                rng_state=(0x12, 0x34),
                rng_override=True,
                **kwargs,
            )
            runner.reset(None, [spec])
            got = runner.buffers.board_bytes.reshape(2, 128)
            assert np.array_equal(got, bank.boards[i].reshape(2, 128))
            # Both sides parked at a live decision with feasible placements.
            assert runner.buffers.need_action.tolist() == [1, 1]
            assert (runner.buffers.feasible_mask.sum(axis=1) > 0).all()
    finally:
        runner.close()


def _make_env(bank_path, fraction, *, enabled=True, num_pairs=4):
    from training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    cfg = None
    if enabled:
        cfg = {"enabled": True, "path": str(bank_path), "fraction": float(fraction)}
    return DrMarioVsPoolVecEnv(
        num_pairs=num_pairs,
        level=14,
        speed_setting=2,
        randomize_rng=True,
        start_bank_cfg=cfg,
    )


def test_fraction_sampling_respected(bank_path):
    from training.envs.start_bank import StartBank

    bank = StartBank(bank_path)
    bank_boards = {bank.boards[i].tobytes() for i in range(len(bank))}

    def count_bank_resets(fraction, seed=123, num_pairs=16):
        env = _make_env(bank_path, fraction, num_pairs=num_pairs)
        try:
            env.reset(seed=seed)
            boards = env._runner.buffers.board_bytes.reshape(num_pairs, 2 * 128)
            return sum(1 for b in boards if b.tobytes() in bank_boards)
        finally:
            env.close()

    assert count_bank_resets(0.0) == 0
    assert count_bank_resets(1.0) == 16
    mid = count_bank_resets(0.5)
    assert 0 < mid < 16


def test_disabled_flag_is_bit_identical(bank_path):
    obs = {}
    boards = {}
    for name, enabled in (("off", False), ("absent", None)):
        from training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

        cfg = {"enabled": False, "path": str(bank_path), "fraction": 0.5} if enabled is False else None
        env = DrMarioVsPoolVecEnv(
            num_pairs=2, level=14, speed_setting=2, randomize_rng=True, start_bank_cfg=cfg
        )
        try:
            o, _infos = env.reset(seed=42)
            obs[name] = np.array(o, copy=True)
            boards[name] = env._runner.buffers.board_bytes.copy()
        finally:
            env.close()
    assert np.array_equal(obs["off"], obs["absent"])
    assert np.array_equal(boards["off"], boards["absent"])
