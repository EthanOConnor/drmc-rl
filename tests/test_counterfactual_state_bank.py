from __future__ import annotations

from functools import lru_cache

from drmc_rl.search.pill_belief import PillReserveBelief
from drmc_rl.seedlab.rng import generate_game
from drmc_rl.teachers.state_bank import balance_state_rows, select_game_rows


@lru_cache(maxsize=None)
def _belief(level: int):
    board = generate_game(level, 0x8988).board
    return PillReserveBelief.from_initial_board(level=level, board=board).to_dict()


def _row(index: int, level: int, tactical: str):
    return {
        "id": f"state-{index}",
        "level": level,
        "speed": 2,
        "tactical_stratum": tactical,
        "reserve_belief": _belief(level),
    }


def test_balancing_is_deterministic_and_fills_declared_quotas() -> None:
    rows = [
        *(_row(index, 5, "midgame") for index in range(8)),
        *(_row(100 + index, 10, "race-finish") for index in range(6)),
    ]
    quota = {("5", "2", "midgame"): 3, ("10", "2", "race-finish"): 2}
    first = balance_state_rows(rows, quota=quota, seed=11)
    second = balance_state_rows(reversed(rows), quota=quota, seed=11)
    assert [row["id"] for row in first.selected] == [row["id"] for row in second.selected]
    assert first.quota_shortfall == 0
    assert first.strata == {"10/2/race-finish": 2, "5/2/midgame": 3}


def test_balancing_reports_shortfall_and_rejects_missing_belief() -> None:
    result = balance_state_rows(
        [_row(1, 5, "midgame")], quota={("5", "2", "midgame"): 3}
    )
    assert result.quota_shortfall == 2
    row = _row(2, 5, "midgame")
    row.pop("reserve_belief")
    try:
        balance_state_rows([row], quota={("5", "2", "midgame"): 1})
    except ValueError as error:
        assert "reserve-belief" in str(error)
    else:  # pragma: no cover
        raise AssertionError("missing reserve belief should be rejected")


def test_per_game_selection_retains_late_rare_tactical_states() -> None:
    rows = [
        *(_row(index, 5, "midgame") for index in range(20)),
        _row(100, 5, "race-finish"),
        _row(101, 5, "incoming-garbage"),
    ]
    counts: dict[str, int] = {}
    selected = select_game_rows(
        rows,
        limit=4,
        global_tactical_counts=counts,
        seed=17,
    )
    tactical = {row["tactical_stratum"] for row in selected}
    assert "race-finish" in tactical
    assert "incoming-garbage" in tactical
    assert sum(counts.values()) == len(selected)


def test_balancing_rejects_unknown_belief_schema() -> None:
    row = _row(3, 5, "midgame")
    row["reserve_belief"] = {"schema": "made-up"}
    try:
        balance_state_rows([row], quota={("5", "2", "midgame"): 1})
    except ValueError as error:
        assert "unsupported pill reserve belief schema" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsupported reserve-belief schema should be rejected")
