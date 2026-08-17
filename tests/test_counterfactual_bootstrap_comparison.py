from __future__ import annotations

from drmc_rl.teachers.bootstrap_comparison import BootstrapRow, compare_bootstrap
from drmc_rl.teachers.release_analysis import ReleaseDataset, ReleaseState


def _candidate(action: int, win: float, draw: float, loss: float):
    return {
        "action": action,
        "win": win,
        "draw": draw,
        "loss": loss,
        "policy_target": 1.0,
        "rank": 1,
        "regret_win_logit": 0.0,
    }


def test_counterfactual_is_compared_on_observed_action_with_game_bootstrap() -> None:
    states = {}
    rows = []
    for game in range(12):
        outcome = 0 if game < 6 else 2
        win = 0.8 if outcome == 0 else 0.2
        source_id = f"state-{game}"
        candidate = _candidate(7, win, 0.05, 0.95 - win)
        states[source_id] = ReleaseState(
            source_id,
            ("test",),
            {"best_action": 7, "candidates": [candidate]},
            {7: candidate},
        )
        rows.append(
            BootstrapRow(
                source_id=source_id,
                game_id=f"game-{game}",
                outcome=outcome,
                observed_action=7,
                baseline_wdl=(0.4, 0.2, 0.4),
            )
        )
    release = ReleaseDataset(
        settings={
            "chance_model": "nes-reserve-public-seed-belief-v2",
            "information_scope": "privileged-test",
        },
        states=states,
        manifest_paths=(),
        release_sha256=("release",),
    )
    report = compare_bootstrap(release, rows, seed=5, bootstrap_samples=300)
    assert report["rows"] == 12
    assert report["games"] == 12
    assert report["counterfactual"]["log_loss"] < report["v3_bootstrap"]["log_loss"]
    assert report["paired_game_bootstrap"]["log_loss"]["ci95_high"] < 0
