from __future__ import annotations

from types import SimpleNamespace

from tools.build_pair_state_pilot import _tactical_stratum


def _state(
    board: list[int],
    *,
    viruses: int = 20,
    pending: int = 0,
):
    own = SimpleNamespace(board=tuple(board), viruses_remaining=viruses)
    opponent = SimpleNamespace(board=tuple([0xFF] * 128), viruses_remaining=20)
    public = SimpleNamespace(sides=(own, opponent))
    privileged = SimpleNamespace(
        public=public,
        pending_attacks=(pending, 0),
    )
    return SimpleNamespace(privileged=privileged)


def test_static_high_viruses_do_not_count_as_stack_pressure() -> None:
    board = [0xFF] * 128
    board[0] = 0xD0  # virus in the top row: static seed geometry, not player stack
    assert _tactical_stratum(_state(board), 0) == "midgame"


def test_player_pill_material_drives_pressure_strata() -> None:
    top = [0xFF] * 128
    top[8] = 0x80  # single half-pill in top three rows
    assert _tactical_stratum(_state(top), 0) == "topout-defense"

    high = [0xFF] * 128
    high[4 * 8] = 0x40  # connected pill half in rows 3..6
    assert _tactical_stratum(_state(high), 0) == "high-pressure"


def test_incoming_garbage_and_race_finish_take_precedence() -> None:
    board = [0xFF] * 128
    assert _tactical_stratum(_state(board, pending=2), 0) == "incoming-garbage"
    assert _tactical_stratum(_state(board, viruses=4), 0) == "race-finish"
def test_failure_predecessors_count_own_placements_and_preserve_full_pair_records():
    from tools.build_pair_state_pilot import failure_predecessors

    rows = []
    for decision in range(20):
        for side in (0, 1):
            rows.append(
                dict(
                    id=f"{side}-{decision}",
                    root_side=side,
                    decision_index=decision,
                    outcome="loss" if side == 0 else "win",
                    natural_outcome_available=True,
                    privileged={"engine_checkpoint_b64": "opaque", "public": {"frame": decision}},
                    reserve_belief={"observations": [decision]},
                )
            )
    selected = failure_predecessors(rows)
    assert [r["decision_index"] for r in selected] == [16, 12, 4]
    assert [r["placements_before_failure"] for r in selected] == [4, 8, 16]
    assert all(r["privileged"] == rows[2 * r["decision_index"]]["privileged"] for r in selected)
    assert all(
        r["reserve_belief"] == rows[2 * r["decision_index"]]["reserve_belief"] for r in selected
    )
    assert failure_predecessors([{**r, "natural_outcome_available": False} for r in rows]) == []
