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
        validate_spec(dict(threshold=20, per_point=0.01, event_cap=0.5, game_cap=2.5))


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


def test_stepped_bonus_pays_by_tier_and_t3_dominates():
    from drmc_rl.training.showiness import validate_spec

    spec = validate_spec(dict(steps=[[27.0, 0.05], [30.0, 0.15], [42.0, 0.30]], event_cap=0.30, game_cap=0.60))

    def at(score):
        return bc.showiness_bonus(bc.ClearFeatures(cells=int(score) + 4, rounds=1, lines=1, max_round_lines=1), spec)
    assert [at(s) for s in (20, 26, 27, 29, 30, 41, 42, 70)] == [0, 0, 0.05, 0.05, 0.15, 0.15, 0.30, 0.30]
    with pytest.raises(ValueError):
        validate_spec(dict(steps=[[27.0, 0.2], [30.0, 0.1]], event_cap=0.3, game_cap=0.6))


def test_score_weights_drop_rows_below_the_first_bar(tmp_path):
    from tools.train_controller_retention import StartMix

    board = np.full((16, 8), E, np.uint8)
    board[15, 0] = 0xD1
    path = tmp_path / "bank.npz"
    np.savez(path, boards=np.stack([np.stack([board, board])] * 4), falling=np.ones((4, 2, 2), np.uint8),
             preview=np.zeros((4, 2, 2), np.uint8), pill_counter=np.full((4, 2), 40, np.uint8),
             speed_ups=np.zeros((4, 2), np.uint8), stratum=np.zeros(4, np.uint8),
             target_score=np.asarray([21.0, 28.0, 33.0, 50.0], np.float32))
    mix = StartMix(dict(bank=str(path), fraction=0.9, score_weights=[[20, 0], [27, 1], [30, 3], [42, 10]]))
    assert mix.row_p.tolist() == pytest.approx([0, 1 / 14, 3 / 14, 10 / 14])


def test_horizontal_bonus_counts_only_completed_horizontal_lines_under_its_own_cap(monkeypatch):
    from drmc_rl.training import showiness

    spec = showiness.validate_spec(dict(steps=[[27.0, 0.05]], event_cap=0.3, game_cap=0.6,
                                        horizontal=dict(per_clear=0.004, combo_extra=0.008, game_cap=0.01)))
    monkeypatch.setattr(showiness, "move_features", lambda move: move["f"])
    plain_h = bc.ClearFeatures(cells=4, rounds=1, lines=1, max_round_lines=1, horizontal_lines=1)
    combo_h = bc.ClearFeatures(cells=8, rounds=2, lines=2, max_round_lines=1, horizontal_lines=1)
    vertical = bc.ClearFeatures(cells=4, rounds=1, lines=1, max_round_lines=1, horizontal_lines=0)
    moves = [dict(side=0, f=f, learning={}) for f in (bc.ClearFeatures(), vertical, plain_h, combo_h, plain_h)]
    assert showiness.learner_bonuses(moves, spec) == pytest.approx([0, 0, 0.004, 0.006, 0])  # 0.012 capped to 0.006 left
    placed = bc.resolve(bc.place(bottle({(15, 0): 0x81, (15, 1): 0x81, (15, 2): 0x81}), (0, 2), 15 * 8 + 3))[1]
    assert placed.horizontal_lines == 1


@pytest.mark.parametrize("pressure", [0.0, 31.0])
def test_retention_hinge_is_zero_at_or_below_the_start_kl(pressure):
    import torch
    from drmc_rl.training.controller_retention import PaceRetention

    r = PaceRetention.__new__(PaceRetention)
    r.coefficient, r.batch_size, r.max_kl_increase, r.pressure_strength = 0.1, 4, 0.03, pressure
    r.paces = ("normal", "fast")
    r.by_pace = {p: [dict(i=i) for i in range(3)] for p in r.paces}
    r.weights = {p: np.full(3, 1 / 3) for p in r.paces}
    r.baseline = {"normal": 0.20, "fast": 0.10}
    r.hinge, r.hinge_counts = True, {}
    r.set_pressure(r.baseline)
    kl = {}
    r._kl = lambda rows: torch.tensor([kl[p] for p in r.paces for _ in range(len(rows) // 2)], dtype=torch.float32)
    rng = np.random.default_rng(0)
    for kl in ({"normal": 0.20, "fast": 0.05}, {"normal": 0.0, "fast": 0.10}):
        assert float(r.loss(rng)) == 0.0
    kl = {"normal": 0.26, "fast": 0.10}
    expected = 0.1 * (0.06 * r.pressure["normal"] + 0.0) / 2
    assert float(r.loss(rng)) == pytest.approx(expected, rel=1e-5)
    assert r.pop_hinge_stats() == {"normal": pytest.approx(1 / 3, abs=1e-3), "fast": 0.0}


def test_start_mixes_split_pairs_between_banks_at_constant_shares(tmp_path):
    from tools.train_controller_retention import StartMixes, collection_schedule

    board = np.full((16, 8), E, np.uint8)
    board[15, 0] = 0xD1
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.npz"
        np.savez(path, boards=np.stack([np.stack([board, board])] * 3), falling=np.ones((3, 2, 2), np.uint8),
                 preview=np.zeros((3, 2, 2), np.uint8), pill_counter=np.full((3, 2), 40, np.uint8),
                 speed_ups=np.zeros((3, 2), np.uint8), stratum=np.zeros(3, np.uint8))
        paths.append(str(path))
    mixes = StartMixes([dict(bank=paths[0], fraction=0.35), dict(bank=paths[1], fraction=0.25)])
    match = dict(pace="normal", level=14)
    counts = np.zeros(3)
    for cycle in range(100):
        starts = mixes.starts(dict(seed=5), cycle, 0, match, 32)
        for i in range(0, len(starts), 2):
            assert starts[i] == starts[i + 1] or starts[i][1] is not None  # whole pairs
            assert starts[i][3] == starts[i + 1][3]
            counts[2 if starts[i][3] is None else starts[i][3]] += 1
    assert counts / counts.sum() == pytest.approx([0.35, 0.25, 0.40], abs=0.03)
    assert mixes.starts(dict(seed=5), 0, 0, dict(pace="normal", level=20), 32) is None

    class Opponents:
        def choose(self, rng):
            return "opp"
    config = dict(arm="mixed_retention", paces=["normal"], seed=7, games_per_update=64, games_per_pace={},
                  level20_fraction=0.0)
    (_, jobs, starts), = collection_schedule(config, 1, np.arange(1, 5000), Opponents(), mixes, 0.6)
    assert len(jobs) == len(starts) == 64
    with pytest.raises(ValueError):
        StartMixes([dict(bank=paths[0], fraction=0.6), dict(bank=paths[1], fraction=0.5)])


def test_update_tf32_applies_only_inside_the_update_and_restores_strict_fp32():
    import torch
    from tools.train_controller_retention import RESUME_FREE_KEYS, update_precision

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with update_precision(dict(update_tf32=True)):
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32 and torch.backends.cudnn.benchmark
    assert not (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark)
    with update_precision({}):
        assert not torch.backends.cudnn.benchmark
    assert "update_tf32" in RESUME_FREE_KEYS


def test_accept_limit_lifts_only_the_retention_acceptance_guard():
    from drmc_rl.training.controller_retention import PaceRetention

    r = PaceRetention.__new__(PaceRetention)
    r.paces, r.baseline, r.max_kl_increase, r.pressure_strength = ("fast",), {"fast": 0.02}, 0.03, 31.0
    r.accept_limit = 0.03
    assert r.accepts({"fast": 0.049}) and not r.accepts({"fast": 0.051})
    r.accept_limit = 10.0
    assert r.accepts({"fast": 0.5})
    r.set_pressure({"fast": 0.051})
    assert r.pressure["fast"] == pytest.approx(32.0)  # the pressure scale still uses max_kl_increase
