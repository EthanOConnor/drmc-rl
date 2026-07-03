from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.backends.drmario_pool import is_library_present

FIXTURE = Path(__file__).parent / "fixtures" / "fc_v2_events_small.jsonl"

VIRUS_HI = 0xD0


def _extract(seed: int = 0, **kwargs):
    from tools.build_clear_endgame_bank import extract_clear_endgame_positions

    return extract_clear_endgame_positions(
        FIXTURE.read_bytes(), rng=np.random.default_rng(seed), **kwargs
    )


def test_extraction_deterministic_and_filtered():
    rows_a = _extract(seed=7)
    rows_b = _extract(seed=7)
    assert len(rows_a) == len(rows_b) > 0
    for a, b in zip(rows_a, rows_b):
        assert np.array_equal(a["boards"], b["boards"])
        assert a["stratum"] == b["stratum"]
    for r in rows_a:
        vc = ((r["boards"] & 0xF0) == VIRUS_HI).sum(axis=(1, 2))
        # Sampled side in 1..8, opponent live, virus_rem matches the board.
        assert np.array_equal(vc, r["virus_rem"])
        assert 1 <= vc.min() <= 8
        assert vc.max() >= 1
        assert r["stratum"] == (0 if vc.min() <= 2 else 1 if vc.min() <= 5 else 2)
        # Both sides spawn-safe: top 2 rows empty.
        assert (r["boards"][:, :2, :] == 0xFF).all()


def test_mirror_swaps_sides():
    from tools.build_clear_endgame_bank import mirror_expand

    positions = _extract()
    rows, extra = mirror_expand(positions, list(range(len(positions))))
    assert len(rows) == 2 * len(positions)
    assert extra["mirror"] == [0, 1] * len(positions)
    for i in range(0, len(rows), 2):
        orig, mirr = rows[i], rows[i + 1]
        for key in ("boards", "falling", "preview", "pill_counter", "speed_ups", "levels", "speeds", "virus_rem"):
            assert np.array_equal(orig[key][::-1], mirr[key])
        assert orig["stratum"] == mirr["stratum"]
        assert orig["spawn_f"] == mirr["spawn_f"]


@pytest.fixture(scope="module")
def bank_path(tmp_path_factory):
    """Tiny clear-endgame bank npz (both orientations) from the fixture."""

    from tools.build_clear_endgame_bank import mirror_expand
    from tools.build_start_bank import positions_to_arrays

    positions = _extract(per_game_side=8)
    assert positions, "fixture produced no clear-endgame positions"
    rows, extra = mirror_expand(positions, [0] * len(positions))
    arrays = positions_to_arrays(rows, extra)
    path = tmp_path_factory.mktemp("bank") / "clear_endgame.npz"
    np.savez_compressed(path, **arrays)
    return path


@pytest.mark.skipif(
    not is_library_present(),
    reason="native pool library missing (build with: make -C game_engine libdrmario_pool)",
)
def test_bank_rows_round_trip_through_native_reset(bank_path):
    """StartBank loads the npz unchanged; original AND mirrored rows reset
    cleanly to a live decision in the native pool."""

    from envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
    from training.envs.start_bank import StartBank

    bank = StartBank(bank_path)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        for i in range(min(8, len(bank))):  # rows alternate original/mirror
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
            assert runner.buffers.need_action.tolist() == [1, 1]
            assert (runner.buffers.feasible_mask.sum(axis=1) > 0).all()
    finally:
        runner.close()
