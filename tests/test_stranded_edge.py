import numpy as np

from drmc_rl.eval.stranded_edge import Definition, follow, mirror_board, stranded, track

E = 0xFF


def bottle(cells):
    g = np.full((16, 8), E, np.uint8)
    for (r, c), tile in cells.items():
        g[r, c] = tile
    return g


def test_stranded_needs_an_empty_shaft_and_few_viruses():
    g = bottle({(8, 0): 0xD1, (15, 3): 0xD0, (15, 0): 0x80})
    (v,) = stranded(g)
    assert (v["row"], v["col"], v["kind"], v["gap"], v["above"]) == (8, 0, "stranded", 6, 0)
    g[12, 0] = 0x80  # the shaft under the virus shrinks to 3 cells
    assert not stranded(g)
    assert stranded(g, Definition(min_gap=3))[0]["gap"] == 3
    many = bottle({(8, 0): 0xD1, **{(15, c): 0xD0 for c in range(1, 5)}})
    assert not stranded(many)  # four viruses left


def test_pillar_is_a_pill_tower_with_nothing_beside_it():
    g = bottle({(9, 7): 0xD2, **{(r, 7): 0x80 for r in range(10, 16)}, (15, 2): 0xD0})
    (v,) = stranded(g)
    assert v["kind"] == "pillar" and v["gap"] == 0 and v["open"] == 7


def test_support_destroying_clear_is_counted_until_the_virus_goes():
    start = bottle({(8, 0): 0xD1, (15, 5): 0xD0, **{(r, 0): 0x80 | 2 for r in range(13, 16)}})
    built = start.copy()
    built[12, 0] = 0x80  # stack rises
    knocked = built.copy()
    knocked[12:16, 0] = E  # a clear takes the stack away
    gone = knocked.copy()
    gone[8, 0] = E
    (episode,) = track([start, built, knocked, gone], [0, 10, 20, 30])
    assert episode.cleared and episode.pills == 3 and episode.frames == 30
    assert episode.support_destroying == 1 and episode.max_gap == 7
    followed = follow([start, built, knocked, gone], [0, 10, 20, 30], (8, 0, 1))
    assert (followed.pills, followed.support_destroying) == (3, 1)


def test_uncleared_episode_is_censored_unless_the_side_cleared_out():
    start = bottle({(8, 0): 0xD1, (15, 0): 0x80})
    assert track([start, start])[0].censored
    (episode,) = track([start, start], [0, 5], cleared_out=True, end_frame=40)
    assert episode.cleared and episode.pills == 2 and episode.frames == 40


def test_mirror_swaps_edges_and_horizontal_halves():
    g = bottle({(8, 0): 0xD1, (15, 5): 0x61, (15, 6): 0x72})
    m = mirror_board(g)
    assert m[8, 7] == 0xD1 and m[15, 1] == 0x62 and m[15, 2] == 0x71
    assert np.array_equal(mirror_board(m), g)


def _tiny_bank(path):
    board = bottle({(8, 0): 0xD1, (15, 3): 0xD0})
    np.savez(path, boards=np.stack([np.stack([board, board])] * 3), falling=np.ones((3, 2, 2), np.uint8),
             preview=np.zeros((3, 2, 2), np.uint8), pill_counter=np.full((3, 2), 40, np.uint8),
             speed_ups=np.zeros((3, 2), np.uint8), stratum=np.zeros(3, np.uint8))
    return path


def test_start_mix_leaves_the_natural_schedule_unchanged(tmp_path):
    from tools.train_controller_retention import StartMix, collection_schedule

    class Opponents:
        def choose(self, rng):
            return f"opp{int(rng.integers(3))}"

    config = dict(arm="mixed_retention", paces=["normal", "fast"], seed=7, games_per_update=32, games_per_pace={},
                  level20_fraction=0.15)
    mix = StartMix(dict(bank=str(_tiny_bank(tmp_path / "bank.npz")), fraction=0.5))
    available = np.arange(1, 5000)
    plain = collection_schedule(config, 3, available, Opponents())
    mixed = collection_schedule(config, 3, available, Opponents(), mix)
    assert [(m, j) for m, j, _ in plain] == [(m, j) for m, j, _ in mixed]
    for (match, jobs, starts) in mixed:
        if starts is None:
            assert match["level"] != 14
            continue
        assert len(starts) == len(jobs)
        assert all(starts[2 * i][0] == starts[2 * i + 1][0] for i in range(len(jobs) // 2))  # whole pairs
        assert 0 < sum(r is not None for r, _ in starts) < len(starts)


def test_frame_pool_starts_both_bottles_from_a_checkpoint():
    import pytest

    from drmc_rl.envs.backends.drmario_pool import is_library_present
    if not is_library_present():
        pytest.skip("native pool library not built")
    from drmc_rl.envs.backends.vs_frames import EventVsPool

    board = bottle({(8, 0): 0xD1, (15, 3): 0xD0, (15, 0): 0x80})
    overlay = dict(checkpoint_enabled=True, checkpoint_board=np.stack([board, board]).reshape(2, 128),
                   checkpoint_falling_colors=np.ones((2, 2), np.uint8),
                   checkpoint_preview_colors=np.zeros((2, 2), np.uint8),
                   checkpoint_pill_counter=(40, 40), checkpoint_speed_ups=(0, 0))
    with EventVsPool(2) as pool:
        pool.reset([0x1234, 0x2345], starts=[overlay, None])
        assert all(bytes(pool.states[s].board) == board.tobytes() for s in (0, 1))
        assert bytes(pool.states[2].board) != board.tobytes()
