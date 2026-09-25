import numpy as np
import pytest

from drmc_rl.eval import big_clear as bc

E = 0xFF
Y, R, B = 0, 1, 2  # NES low nibble
cY, cR, cB = 1, 0, 2  # canonical pill colors


def bottle(cells):
    g = np.full(128, E, np.uint8)
    for (r, c), tile in cells.items():
        g[r * 8 + c] = tile
    return bytes(g)


def test_single_four_line_scores_zero():
    board = bottle({(15, 0): 0x80 | R, (15, 1): 0x80 | R, (15, 2): 0x80 | R})
    f = bc.placement_features(board, (cR, cB), 0 * 128 + 15 * 8 + 3)  # horizontal at (15,3)-(15,4)
    assert (f.cells, f.rounds, f.lines, f.garbage) == (4, 1, 1, 0)
    assert f.score() == 0.0 and bc.tier(f.score()) == "T0"


def test_cross_counts_shared_cell_once_and_scores_simultaneous_lines():
    cells = {(15, 0): 0x81, (15, 1): 0x81, (15, 2): 0x81, (12, 3): 0x81, (13, 3): 0x81, (14, 3): 0x81}
    f = bc.placement_features(bottle(cells), (cR, cB), 0 * 128 + 15 * 8 + 3)
    assert (f.cells, f.rounds, f.lines, f.max_round_lines, f.cross, f.garbage) == (7, 1, 2, 2, 1, 2)
    # cells 3 + simultaneous 2 + cross 2; two garbage earn nothing
    assert f.score() == pytest.approx(7.0)


def test_ordinary_two_round_combo_is_not_big():
    cells = {(15, 0): 0x80 | Y, (14, 0): 0x81, (13, 0): 0x81, (12, 0): 0x81, (9, 0): 0x80, (8, 0): 0x80}
    f = bc.placement_features(bottle(cells), (cY, cR), 1 * 128 + 10 * 8 + 0)  # vertical, top Y at (10,0)
    assert (f.rounds, f.lines, f.cells, f.garbage) == (2, 2, 8, 2)
    assert f.score() == pytest.approx(7.0)  # cells 4 + one extra round 3
    assert f.score() < bc.TIERS[0][1]


def test_long_line_and_viruses_add_points():
    cells = {(15, c): 0xD0 | R for c in range(6)}
    f = bc.placement_features(bottle(cells), (cR, cR), 0 * 128 + 15 * 8 + 6)
    assert (f.cells, f.viruses, f.max_line, f.long_tiles) == (8, 6, 8, 4)
    assert f.points()["long"] == pytest.approx(6.0) and f.points()["viruses"] == pytest.approx(4.0)


def test_forms_line_agrees_with_resolution():
    rng = np.random.default_rng(0)
    for _ in range(300):
        g = np.full(128, E, np.uint8)
        for c in range(8):
            h = int(rng.integers(0, 6))
            for r in range(16 - h, 16):
                g[r * 8 + c] = 0x80 | int(rng.integers(0, 3))
        board = bytes(g)
        col = int(rng.integers(0, 7))
        heights = [next((r for r in range(16) if board[r * 8 + c] != E), 16) for c in (col, col + 1)]
        row = min(heights) - 1
        if row < 0:
            continue
        pill = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        try:
            placed = bc.place(board, pill, row * 8 + col)
        except ValueError:
            continue
        # A random stack may already hold a line; forms_line only looks at the new cells.
        before = bc.resolve(board)[1].rounds
        if before:
            continue
        assert bc.forms_line(placed, (row * 8 + col, row * 8 + col + 1)) == (bc.resolve(placed)[1].rounds > 0)


def test_bonus_is_zero_below_threshold_and_capped():
    spec = dict(threshold=20.0, base=0.05, per_point=0.005, event_cap=0.15, game_cap=0.3)
    small = bc.ClearFeatures(cells=8, rounds=2, lines=2, max_round_lines=1, garbage=2)
    assert bc.showiness_bonus(small, spec) == 0.0
    huge = bc.ClearFeatures(cells=60, rounds=6, lines=10, max_round_lines=3, viruses=10, garbage=4)
    assert bc.showiness_bonus(huge, spec) == pytest.approx(0.15)


def test_bonus_return_to_go_matches_terminal_samples(monkeypatch):
    from drmc_rl.training import showiness
    from tools.train_pace_strategy import terminal_samples

    spec = showiness.validate_spec(dict(threshold=20.0, base=0.05, per_point=0.005, event_cap=0.10, game_cap=0.12))
    monkeypatch.setattr(showiness, "move_features", lambda move: move["f"])
    feats = [bc.ClearFeatures()] + [bc.ClearFeatures(cells=int(s) + 4, rounds=1, lines=1, max_round_lines=1)
                                    for s in (25.0, 0.0, 40.0, 30.0)]
    feats[2] = bc.ClearFeatures()
    moves = []
    for k, f in enumerate(feats):
        moves.append(dict(side=0, f=f, learning=dict(action=k)))
        moves.append(dict(side=1, f=bc.ClearFeatures()))  # opponent: never rewarded
    batch = [(dict(reason="topout", score=1.0), moves, None), (dict(reason="timeout", score=None), moves, None)]
    samples = terminal_samples(batch)
    stats = showiness.apply_bonus(batch, samples, spec)
    bonus = [0.0, 0.075, 0.0, 0.045, 0.0]  # 0.075 + (0.10 capped to the 0.045 left) + nothing left
    togo = np.cumsum(bonus[::-1])[::-1]
    assert [s["return"] for s in samples] == pytest.approx([1.0 + t for t in togo])
    assert stats["events"] == 2 and stats["games"] == 1 and stats["bonus_total"] == pytest.approx(0.12)


def test_spec_rejects_a_cap_that_could_outweigh_a_win():
    from drmc_rl.training.showiness import validate_spec

    with pytest.raises(ValueError):
        validate_spec(dict(threshold=20, per_point=0.01, event_cap=0.5, game_cap=1.5))


def test_start_mix_replays_only_training_pool_seeds(tmp_path):
    from tools.train_controller_retention import StartMix, collection_schedule
    from drmc_rl.program.seed_reserve import load_reserve

    board = np.full((16, 8), E, np.uint8)
    board[15, 0] = 0xD1
    blocked = sorted(load_reserve().blocked)[0]
    path = tmp_path / "bank.npz"
    np.savez(path, boards=np.stack([np.stack([board, board])] * 3), falling=np.ones((3, 2, 2), np.uint8),
             preview=np.zeros((3, 2, 2), np.uint8), pill_counter=np.full((3, 2), 40, np.uint8),
             speed_ups=np.zeros((3, 2), np.uint8), stratum=np.zeros(3, np.uint8),
             seed=np.asarray([4321, -1, blocked]))

    class Opponents:
        def choose(self, rng):
            return f"opp{int(rng.integers(3))}"

    available = np.setdiff1d(np.arange(1, 65536), [blocked, blocked ^ 0x100])
    config = dict(arm="mixed_retention", paces=["normal"], seed=7, games_per_update=64, games_per_pace={},
                  level20_fraction=0.0)
    plain_mix = StartMix(dict(bank=str(path), fraction=0.9), available)
    replay_mix = StartMix(dict(bank=str(path), fraction=0.9, replay_share=1.0), available)
    assert replay_mix.replayable == 1  # the reserve seed and the unknown seed are never replayed
    plain = collection_schedule(config, 2, available, Opponents(), plain_mix)
    replay = collection_schedule(config, 2, available, Opponents(), replay_mix)
    (_, jobs_p, starts_p), = plain
    (_, jobs_r, starts_r), = replay
    assert [s[0] for s in starts_p] == [s[0] for s in starts_r]  # same rows drawn
    for (seed_p, side_p, _), (seed_r, side_r, _), (row, _, replayed) in zip(jobs_p, jobs_r, starts_r):
        assert side_p == side_r
        if row == 0:
            assert replayed == 4321 and seed_r == 4321
        else:
            assert replayed is None and seed_r == seed_p
