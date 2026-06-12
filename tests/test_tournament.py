"""Unit tests for tools/tournament.py: stats + sqlite store resumability.

No real matches: the game runner is a stub. Reference numbers were computed
with an independent implementation (closed forms where they exist).
"""

import math

import numpy as np
import pytest

from tools.tournament import (
    GameResult,
    TournamentStore,
    elo_mle,
    game_seed,
    make_specs,
    render_report,
    run_sprt,
    run_tournament,
    sprt_bounds,
    sprt_llr,
    wilson_ci,
)


# ----------------------------------------------------------------- Wilson CI
def test_wilson_ci_known_values():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901568467, abs=1e-6)
    assert hi == pytest.approx(0.9433190520, abs=1e-6)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775401688, abs=1e-6)
    lo, hi = wilson_ci(10, 10)
    assert lo == pytest.approx(0.7224598312, abs=1e-6)
    assert hi == 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)


# -------------------------------------------------------------------- Elo MLE
def test_elo_mle_two_entries_closed_form():
    # No draws: MLE rating difference is 400*log10(W/L); 75-25 -> 190.8485.
    games = [(0, 1, 1.0)] * 75 + [(0, 1, 0.0)] * 25
    r, se = elo_mle(2, games)
    assert r[0] - r[1] == pytest.approx(400 * math.log10(3.0), abs=1e-3)
    assert r[0] + r[1] == pytest.approx(0.0, abs=1e-9)  # mean-zero anchor
    # Fisher info closed form: SE per entry = 1/(2*sqrt(N k^2 p(1-p))).
    k = math.log(10) / 400
    se_expect = 1.0 / (2 * math.sqrt(100 * k * k * 0.75 * 0.25))
    assert se[0] == pytest.approx(se_expect, rel=1e-3)
    assert se[1] == pytest.approx(se_expect, rel=1e-3)


def test_elo_mle_draws_are_half_points():
    # Mean score 0.7 (60-20-20 with draws=0.5) -> diff 400*log10(0.7/0.3).
    games = [(0, 1, 1.0)] * 60 + [(0, 1, 0.5)] * 20 + [(0, 1, 0.0)] * 20
    r, _se = elo_mle(2, games)
    assert r[0] - r[1] == pytest.approx(400 * math.log10(0.7 / 0.3), abs=1e-3)


def test_elo_mle_chain_transitivity():
    # A>B and B>C each 75-25, no A-C games: equally spaced ratings.
    games = (
        [(0, 1, 1.0)] * 75 + [(0, 1, 0.0)] * 25 + [(1, 2, 1.0)] * 75 + [(1, 2, 0.0)] * 25
    )
    r, se = elo_mle(3, games)
    d = 400 * math.log10(3.0)
    assert r[0] == pytest.approx(d, abs=1e-2)
    assert r[1] == pytest.approx(0.0, abs=1e-2)
    assert r[2] == pytest.approx(-d, abs=1e-2)
    assert np.all(se > 0)
    # Mirror-symmetric design: SEs of the end entries match.
    assert se[0] == pytest.approx(se[2], rel=1e-6)


# ----------------------------------------------------------------------- SPRT
def test_sprt_llr_reference_values():
    # Computed with an independent BayesElo-trinomial implementation.
    assert sprt_llr(10, 5, 5, 0.0, 5.0) == pytest.approx(0.0955029088, abs=1e-8)
    assert sprt_llr(120, 30, 90, 0.0, 5.0) == pytest.approx(0.4661302623, abs=1e-8)
    # Zero draws must not stall the test (Dr. Mario draws are rare).
    assert sprt_llr(50, 0, 70, 0.0, 5.0) == pytest.approx(-0.3014825550, abs=1e-8)


def test_sprt_llr_edge_cases_and_monotonicity():
    assert sprt_llr(0, 0, 0, 0.0, 5.0) == 0.0
    assert sprt_llr(0, 5, 0, 0.0, 5.0) == 0.0
    # More wins -> larger LLR for elo1 > elo0.
    assert sprt_llr(60, 10, 50, 0.0, 5.0) > sprt_llr(50, 10, 60, 0.0, 5.0)


def test_sprt_bounds():
    lower, upper = sprt_bounds(0.05, 0.05)
    assert lower == pytest.approx(math.log(0.05 / 0.95))
    assert upper == pytest.approx(math.log(0.95 / 0.05))
    assert lower < 0 < upper


# ------------------------------------------------------------------ scheduling
def test_game_seed_deterministic_and_nonzero():
    s1 = game_seed(123, "a", "b", 7)
    assert s1 == game_seed(123, "a", "b", 7)
    assert s1 != game_seed(123, "a", "b", 8)
    assert s1 != game_seed(124, "a", "b", 7)
    assert 0 < s1 <= 0xFFFF
    specs = make_specs(123, "a", "b", range(4))
    assert [s.a_side for s in specs] == [0, 1, 0, 1]  # alternating sides


# --------------------------------------------------------- store + resumability
ROSTER = {
    "entries": [
        {"name": "alpha", "checkpoint": "x.pt.gz", "mode": "plain"},
        {"name": "beta", "checkpoint": "y.pt.gz", "mode": "plain"},
        {"name": "gamma", "checkpoint": "z.pt.gz", "mode": "plain"},
    ]
}


class StubRunner:
    """Deterministic fake match runner; optionally crashes mid-series."""

    def __init__(self, fail_after=None, winner_seq=None):
        self.fail_after = fail_after
        self.winner_seq = winner_seq
        self.played = 0
        self.calls = []

    def play(self, entry_a, entry_b, specs):
        self.calls.append((entry_a["name"], entry_b["name"], [s.game_idx for s in specs]))
        for s in specs:
            if self.fail_after is not None and self.played >= self.fail_after:
                raise RuntimeError("simulated crash")
            if self.winner_seq is not None:
                winner = self.winner_seq[self.played % len(self.winner_seq)]
            else:
                winner = "a" if s.game_idx % 3 else "b"
            self.played += 1
            yield GameResult(spec=s, winner=winner, match_len_sec=12.5, decisions=40)


def test_run_tournament_records_all_games(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    runner = StubRunner()
    tid = run_tournament(
        store, "t1", ROSTER, games_per_pair=4, level=14, seed=99, runner=runner, verbose=False
    )
    rows = store.games_for(tid)
    assert len(rows) == 3 * 4  # 3 unordered pairs x 4 games
    # Seeds stored per game match the deterministic schedule.
    for r in rows:
        assert int(r["seed"]) == game_seed(99, r["entry_a"], r["entry_b"], int(r["game_idx"]))
        assert int(r["side_assignment"]) == int(r["game_idx"]) % 2
    store.close()


def test_run_tournament_resume_is_noop_when_complete(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    run_tournament(
        store, "t1", ROSTER, games_per_pair=4, level=14, seed=99, runner=StubRunner(),
        verbose=False,
    )
    rerun = StubRunner()
    tid = run_tournament(
        store, "t1", ROSTER, games_per_pair=4, level=14, seed=99, runner=rerun, verbose=False
    )
    assert rerun.played == 0
    assert rerun.calls == []  # complete pairs are skipped entirely
    assert len(store.games_for(tid)) == 12
    store.close()


def test_run_tournament_resumes_after_crash(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    crashy = StubRunner(fail_after=5)
    with pytest.raises(RuntimeError):
        run_tournament(
            store, "t1", ROSTER, games_per_pair=4, level=14, seed=99, runner=crashy,
            verbose=False,
        )
    tid = store.tournament_by_name("t1")["id"]
    assert len(store.games_for(tid)) == 5

    resumed = StubRunner()
    run_tournament(
        store, "t1", ROSTER, games_per_pair=4, level=14, seed=99, runner=resumed, verbose=False
    )
    rows = store.games_for(tid)
    assert len(rows) == 12
    assert resumed.played == 7
    # No duplicate (pair, game_idx) rows; full schedule covered.
    keys = {(r["entry_a"], r["entry_b"], r["game_idx"]) for r in rows}
    assert len(keys) == 12
    for a, b in [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")]:
        assert {k[2] for k in keys if k[:2] == (a, b)} == {0, 1, 2, 3}
    store.close()


def test_run_tournament_rejects_roster_mismatch(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    run_tournament(
        store, "t1", ROSTER, games_per_pair=2, level=14, seed=99, runner=StubRunner(),
        verbose=False,
    )
    other = {
        "entries": [{"name": "alpha", "checkpoint": "x"}, {"name": "delta", "checkpoint": "w"}]
    }
    with pytest.raises(SystemExit):
        run_tournament(
            store, "t1", other, games_per_pair=2, level=14, seed=99, runner=StubRunner(),
            verbose=False,
        )
    with pytest.raises(SystemExit):  # changed seed would break schedule determinism
        run_tournament(
            store, "t1", ROSTER, games_per_pair=2, level=14, seed=100, runner=StubRunner(),
            verbose=False,
        )
    store.close()


def test_render_report_smoke(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    run_tournament(
        store, "t1", ROSTER, games_per_pair=6, level=14, seed=7, runner=StubRunner(),
        verbose=False,
    )
    text = render_report(store, "t1")
    for n in ("alpha", "beta", "gamma"):
        assert n in text
    assert "Elo" in text and "Wilson" in text
    store.close()


# ------------------------------------------------------------------- SPRT runs
EA = {"name": "cand", "checkpoint": "x.pt.gz", "mode": "plain"}
EB = {"name": "base", "checkpoint": "y.pt.gz", "mode": "plain"}


def test_run_sprt_accepts_h1_on_dominant_a(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    runner = StubRunner(winner_seq=["a"])  # A wins every game
    out = run_sprt(
        store, "s1", EA, EB, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05,
        max_games=400, level=14, seed=3, runner=runner, verbose=False,
    )
    assert out["verdict"] == "H1"
    assert out["llr"] >= out["upper"]
    assert out["games"] < 400  # stopped early
    tid = store.tournament_by_name("s1")["id"]
    assert len(store.games_for(tid)) == out["games"]
    store.close()


def test_run_sprt_accepts_h0_on_dominant_b(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    out = run_sprt(
        store, "s1", EA, EB, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05,
        max_games=400, level=14, seed=3, runner=StubRunner(winner_seq=["b"]), verbose=False,
    )
    assert out["verdict"] == "H0"
    assert out["llr"] <= out["lower"]
    store.close()


def test_run_sprt_resumes_recorded_counts(tmp_path):
    store = TournamentStore(tmp_path / "t.sqlite")
    crashy = StubRunner(fail_after=10, winner_seq=["a", "b"])  # balanced, won't stop early
    with pytest.raises(RuntimeError):
        run_sprt(
            store, "s1", EA, EB, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05,
            max_games=20, level=14, seed=3, runner=crashy, verbose=False,
        )
    resumed = StubRunner(winner_seq=["a", "b"])
    out = run_sprt(
        store, "s1", EA, EB, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05,
        max_games=20, level=14, seed=3, runner=resumed, verbose=False,
    )
    assert out["games"] == 20
    assert resumed.played == 10  # only the missing half was played
    assert out["verdict"] == "inconclusive"
    store.close()
