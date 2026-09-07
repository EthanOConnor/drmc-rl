from __future__ import annotations

import pytest

from tools.audit_rollout_consistency import choose_action, select_rows, summarize


def row(index, game, *, level=10, outcome="win"):
    return {"id": str(index), "game_id": str(game), "level": level, "speed": 2,
            "tactical_stratum": "midgame", "outcome": outcome,
            "natural_outcome_available": outcome is not None,
            "rollout_policy": "frozen-strong-league-mixture-argmax"}


def test_selection_is_order_independent_and_holds_out_whole_games():
    rows = [row(i, i//2, level=10 if i < 10 else 20) for i in range(20)]
    rows.append(row(20, 10, outcome=None))
    selected = select_rows(rows, 8, 42)
    assert selected == select_rows(rows[::-1], 8, 42)
    assert len({r["game_id"] for r in selected}) == 8
    assert [r["level"] for r in selected].count(10) == 4
    with pytest.raises(ValueError, match="independent eligible"):
        select_rows(rows, 11, 42)


def test_selection_rejects_duplicate_ids_and_other_rollout_policies():
    with pytest.raises(ValueError, match="unique"):
        select_rows([row(1, 1), row(1, 2)], 1, 0)
    other = row(1, 1)
    other["rollout_policy"] = "random"
    with pytest.raises(ValueError, match="ensemble argmax"):
        select_rows([other], 1, 0)


def test_choice_uses_full_legal_frontier_and_preserves_tie_order():
    assert choose_action(({0: .9, 511: 1.}, 0.), (0, 511)) == 511
    assert choose_action(({0: .5, 511: .5}, 0.), (511, 0)) == 511
    assert choose_action(({}, 0.), ()) == -1


def test_summary_never_counts_horizon_as_a_draw_or_disagreement():
    records = [
        {"game_id": "a", "initial_action_matches": True,
         "replayed_outcome": "draw", "outcome_matches": True},
        {"game_id": "b", "initial_action_matches": False,
         "replayed_outcome": None, "outcome_matches": None},
        {"game_id": "c", "initial_action_matches": False,
         "replayed_outcome": "loss", "outcome_matches": False},
    ]
    result = summarize(records)
    assert result["natural_terminal"] == 2
    assert result["incomplete"] == 1
    assert result["outcome_agreement"] == .5
