from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.search.native_pair import state_from_payload
from drmc_rl.search.pill_belief import PillReserveBelief, _matching_seed_indices, reserve_for_seed
from tools.build_public_quality_bank import collect_games, game_catalog, run
from tests.test_terminal_rollout import FirstLegal


def configuration(tmp_path):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"fixture-policy")
    return dict(output=str(tmp_path / "bank"), members={"a": str(checkpoint)},
        matchups=[["a", "a"]], partitions={"fit": 4, "anchor": 1, "confirmation": 1},
        conditions=[dict(level=14, speed=2, weight=2), dict(level=20, speed=2, weight=1)],
        excluded_reset_seeds=[[3, 7], [19, 22]], seed=1877, batch_size=3,
        states_per_game=8, max_events=512, native_workers=2, device="cpu", threads=1)


def test_catalog_is_unique_disjoint_and_independent_of_execution_batching(tmp_path):
    config = configuration(tmp_path)
    catalog = game_catalog(config)
    assert catalog == game_catalog({**config, "batch_size": 1, "native_workers": 1})
    assert len({tuple(s["reset_seed"]) for s in catalog}) == 6
    assert not ({(3, 7), (19, 22)} & {tuple(s["reset_seed"]) for s in catalog})
    assert Counter(s["partition"] for s in catalog) == config["partitions"]
    with pytest.raises(ValueError, match="distinct reset"):
        game_catalog({**config, "partitions": {"fit": 65536}})


@pytest.mark.skipif(not is_library_present(), reason="native pool library missing")
@pytest.mark.parametrize("event_public", [False, True])
def test_batched_source_games_reproduce_serial_natural_games_and_complete_public_roots(tmp_path, event_public):
    config = configuration(tmp_path)
    catalog = game_catalog(config)
    serial, batched, metrics = [], [], {}
    collect_games(catalog, {"a": FirstLegal()}, batch_size=1, max_events=512,
                  states_per_game=8, on_game=serial.append, event_public=event_public)
    collect_games(catalog, {"a": FirstLegal()}, batch_size=3, max_events=512,
                  states_per_game=8, native_workers=2, on_game=batched.append, metrics=metrics, event_public=event_public)
    serial.sort(key=lambda g: g["spec"]["index"])
    batched.sort(key=lambda g: g["spec"]["index"])
    assert batched == serial
    assert all(g["natural_outcome_available"] for g in batched)
    assert metrics["completed_games"] == 6 and max(metrics["inference_batch_rows"]) > 1
    assert metrics["policy_decisions"] == sum(g["policy_decisions"] for g in batched)
    for game in batched:
        assert game["public_observation_schema"] == f"causal-settled-pair-v{2 if event_public else 1}"
        assert len(game["rows"]) == 8
        assert len({r["id"] for r in game["rows"]}) == 8
        assert {r["temporal_bin"] for r in game["rows"]} == {"opening", "middle", "late"}
        assert sum(r["sampling_reason"] == "failure-predecessor" for r in game["rows"]) <= 1
        for row in game["rows"]:
            state = state_from_payload(row)
            assert row["candidate_count"] == len(state.legal_actions_by_side[row["root_side"]])
            assert row["observed_action"] in state.legal_actions_by_side[row["root_side"]]
            belief = PillReserveBelief.from_dict(row["reserve_belief"])
            compatible = _matching_seed_indices(belief.observations, belief.level, belief.initial_board)
            # Belief is reconstructed only from the public prefix. The actual
            # game reserve must remain compatible, without being given to actors.
            from drmc_rl.search.pill_belief import reserve_table
            actual = reserve_for_seed(*game["spec"]["reset_seed"])
            assert any((reserve_table()[i] == actual).all() for i in compatible)


@pytest.mark.skipif(not is_library_present(), reason="native pool library missing")
@pytest.mark.parametrize("event_public", [False, True])
def test_collection_resume_retains_completed_games_and_never_relabels_censoring(tmp_path, monkeypatch, event_public):
    import tools.build_public_quality_bank as module
    from pathlib import Path
    import json

    config = configuration(tmp_path)
    config.update(max_events=1)
    if event_public:
        config["public_observation_schema"] = "causal-settled-pair-v2"
    actor = FirstLegal()
    actor.policy = SimpleNamespace(aux_spec="zero_v1_vs")
    monkeypatch.setattr(module, "PublicPolicyContinuation", lambda *args, **kwargs: actor)
    original = module._atomic_gzip_jsonl
    committed = []

    def interrupt(path, rows):
        if path.parent.name == "games":
            if len(committed) == 2:
                raise RuntimeError("simulated interruption")
            committed.append(path)
        original(path, rows)

    monkeypatch.setattr(module, "_atomic_gzip_jsonl", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config)
    before = [p.read_bytes() for p in committed]
    monkeypatch.setattr(module, "_atomic_gzip_jsonl", original)
    result = run(config)
    assert [p.read_bytes() for p in committed] == before
    assert result["games"] == 6 and result["natural_games"] == 0 and result["censored_games"] == 6
    assert sum(x["games"] for x in result["partitions"].values()) == 6
    assert run(config)["games"] == 6  # completed resume does not reload an actor
    for spec in game_catalog(config):
        game = module.read_game(module.game_path(Path(config["output"]), spec))
        assert game["outcomes"] == [None, None]
        assert all(r["outcome"] is None for r in game["rows"])
    progress = json.loads((Path(config["output"])/"progress.json").read_text())
    assert progress["status"] == "Complete" and not progress["product_gates_passed"]
    assert progress["public_observation_schema"] == f"causal-settled-pair-v{2 if event_public else 1}"
    manifest = json.loads((Path(config["output"])/"fit.jsonl.gz.manifest.json").read_text())
    assert manifest["public_observation_schema"] == progress["public_observation_schema"]
    with pytest.raises(ValueError, match="frozen source collection"):
        run({**config, "seed": 1878})
