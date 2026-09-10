from copy import deepcopy
import json

import pytest
import torch

from drmc_rl.training.quality_supervision import (
    assert_disjoint_sources,
    join_quality_rows,
    make_batch,
    policy_rows,
    ranking_diagnostics,
    source_group,
    split_games,
    upgrade_public_model,
)
from tests.test_quality_supervision import checkpoint, data
from tools.fit_paired_quality import cache_reference, evaluate, fit


def labeled_rows():
    source, target = data()
    sources, targets = [], []
    for game, count in enumerate((3, 2, 1)):
        for root in range(count):
            identity = f"{game}-{root}"
            row = dict(source, id=identity, game_id=f"g{game}", reset_seed=[10, game])
            label = deepcopy(target)
            label.update(source_id=identity, game_id=row["game_id"])
            if game == 2:
                for candidate in label["candidates"]:
                    candidate["wdl"] = [0.4, 0.2, 0.4]
            sources.append(row)
            targets.append(label)
    return sources, targets


def test_policy_anchor_partitions_reject_games_and_renamed_reset_replays():
    sources, targets = labeled_rows()
    anchor = dict(sources[0], id="anchor", game_id="independent", reset_seed=[22, 33])
    assert_disjoint_sources(sources, [anchor])
    for alias in (
        dict(anchor, id=sources[0]["id"]),
        dict(anchor, game_id=sources[0]["game_id"]),
        dict(anchor, reset_seed=sources[0]["reset_seed"]),
    ):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint_sources(sources, [alias])
    # The anchor need not have any terminal outcome or artificial Q label.
    batch = make_batch(policy_rows([anchor]), schema="zero_v1_vs", device="cpu", targets=False)
    assert not {"wdl", "prior", "improved", "incumbent"} & set(batch)
    replay = dict(sources[0], id="replay", game_id="renamed-game")
    label = dict(targets[0], source_id="replay", game_id="renamed-game")
    rows = join_quality_rows(sources + [replay], targets + [label])
    train, heldout = split_games(rows, seed=12)
    assert {source_group(r) for r in train}.isdisjoint({source_group(r) for r in heldout})
    weights = make_batch(rows, schema="zero_v1_vs", device="cpu")["weights"]
    totals = {}
    for row, weight in zip(rows, weights):
        totals[source_group(row)] = totals.get(source_group(row), 0) + float(weight)
    assert max(totals.values()) == pytest.approx(min(totals.values()))


def test_rank_and_regret_metrics_ignore_padding_and_credit_prediction_ties_fairly():
    target = torch.tensor([[1.0, -1.0, 50.0], [1.0, -1.0, 50.0], [0.2, 0.2, 50.0]])
    predicted = torch.tensor([[0.4, -0.2, 99.0], [0.0, 0.0, 99.0], [0.4, -0.2, 99.0]])
    mask = torch.tensor([[True, True, False]] * 3)
    prior = torch.tensor([[0.5, 0.5, 0.0]] * 3)
    metrics = ranking_diagnostics(
        predicted, target, mask, torch.tensor([[0.0, 0.0, -1e9]] * 3).log_softmax(-1), prior
    )
    torch.testing.assert_close(metrics["informative_fraction"], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(
        metrics["informative_rank_accuracy_numerator"], torch.tensor([1.0, 0.5, 0.0])
    )
    torch.testing.assert_close(metrics["greedy_regret"], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(metrics["greedy_gain"], torch.tensor([1.0, 0.0, 0.0]))


def test_streamed_evaluation_preserves_global_game_weights_and_informative_denominators(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    sources, targets = labeled_rows()
    rows = join_quality_rows(sources, targets)
    # Include a smaller complete frontier so padding is exercised.
    rows[0] = dict(
        rows[0],
        side=1,
        wdl=rows[0]["wdl"][:2],
        prior=torch.tensor([0.6, 0.4]).numpy(),
        improved=torch.tensor([0.6, 0.4]).numpy(),
    )
    parent, _ = checkpoint()
    net, _ = upgrade_public_model(parent, mode="baseline", device="cpu")
    batch = make_batch(rows, schema="zero_v1_vs", device="cpu")
    seen = []
    hook = net.register_forward_pre_hook(lambda _, args: seen.append(len(args[0])))
    cache_reference(net, batch, batch_size=2, device="cpu")
    streamed, _ = evaluate(
        net, batch, rows, batch_size=2, device="cpu", predictions=tmp_path / "predictions.jsonl"
    )
    assert max(seen) == 2
    assert all(t.device.type == "cpu" for t in batch.values())
    hook.remove()
    whole, _ = evaluate(net, batch, rows, batch_size=len(rows), device="cpu")
    for key, expected in whole.items():
        assert streamed[key] == pytest.approx(expected, abs=2e-7)
    assert streamed["informative_fraction"] == pytest.approx(2 / 3)
    assert streamed["informative_rank_accuracy"] == pytest.approx(0.5)
    assert streamed["informative_roots"] == 5
    assert streamed["informative_games"] == 2
    exported = [
        json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert [r["source_id"] for r in exported] == [r["source_id"] for r in rows]
    assert len(exported[0]["candidate_wdl"]) == 2
    assert len(exported[-1]["candidate_wdl"]) == 3


def fit_config(tmp_path):
    source, target = data()
    sources = [dict(source, id=str(i), game_id=f"g{i}", reset_seed=[3, i]) for i in range(2)]
    targets = [dict(target, source_id=str(i), game_id=f"g{i}") for i in range(2)]
    anchors = [dict(source, id="anchor", game_id="g3", reset_seed=[3, 3])]
    for name, rows in (("sources", sources), ("targets", targets), ("anchors", anchors)):
        (tmp_path / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    parent, _ = checkpoint()
    torch.save(parent, tmp_path / "parent.pt")
    return dict(
        output=str(tmp_path / "fit"),
        checkpoint=str(tmp_path / "parent.pt"),
        state_bank=str(tmp_path / "sources.jsonl"),
        targets=str(tmp_path / "targets.jsonl"),
        anchor_bank=str(tmp_path / "anchors.jsonl"),
        device="cpu",
        mode="baseline",
        seed=19,
        threads=1,
        epochs=1,
        batch_size=1,
        evaluation_batch_size=1,
        minimum_anchor_games=1,
        lr=1e-4,
        max_policy_kl=0.02,
    ), parent


def test_independent_anchor_can_reject_an_epoch_and_rollback_all_weights(tmp_path, monkeypatch):
    import tools.fit_paired_quality as fitter

    config, parent = fit_config(tmp_path)
    config["checkpoint_directory"] = str(tmp_path / "checkpoints")
    real_evaluate = fitter.evaluate

    def drifting_anchor(*args, **kwargs):
        metrics, representation = real_evaluate(*args, **kwargs)
        if not kwargs.get("targets", True):
            metrics["anchor_kl"] = 1.0
        return metrics, representation

    monkeypatch.setattr(fitter, "evaluate", drifting_anchor)
    result = fit(config)
    assert result["accepted_examples"] == 0
    assert result["optimizer_steps"] == 5
    assert result["epochs"] == []
    assert result["rejected_policy_check"]["train"]["anchor_kl"] < config["max_policy_kl"]
    assert "independent-anchor" in result["stop_reason"]
    original, _ = upgrade_public_model(parent, mode="baseline", device="cpu")
    saved = torch.load(tmp_path / "fit" / "diagnostic.pt", weights_only=False)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(saved["state_dict"][name], value, rtol=0, atol=0)
    index = json.loads((tmp_path / "checkpoints/index.json").read_text())
    assert len(index["checkpoints"]) == 1 and index["checkpoints"][0]["epoch"] == 0
    snapshot = torch.load(
        tmp_path / "checkpoints" / index["checkpoints"][0]["filename"], weights_only=False
    )
    for name, value in original.state_dict().items():
        torch.testing.assert_close(snapshot["state_dict"][name], value, rtol=0, atol=0)


def test_quarantined_execution_labels_are_rejected_before_fitting(tmp_path):
    config, _ = fit_config(tmp_path)
    (tmp_path / "label-validity.json").write_text(
        json.dumps(
            {
                "eligible_for_quality_training": False,
                "reason": "reveal ordering predates the strict parked-input fix",
            }
        )
    )
    with pytest.raises(ValueError, match="quarantined terminal labels.*reveal ordering"):
        fit(config)
    progress = json.loads((tmp_path / "fit" / "progress.json").read_text())
    assert progress["status"] == "Failed" and progress["optimizer_steps"] == 0
    assert not (tmp_path / "fit" / "diagnostic.pt").exists()


def test_heldout_drift_is_reported_but_cannot_control_optimizer_acceptance(tmp_path, monkeypatch):
    import tools.fit_paired_quality as fitter

    config, _ = fit_config(tmp_path)
    config.update(checkpoint_directory=str(tmp_path / "checkpoints"), checkpoint_only=True)
    real_evaluate = fitter.evaluate

    def drifting_holdout(*args, **kwargs):
        metrics, representation = real_evaluate(*args, **kwargs)
        if kwargs.get("targets", True) and kwargs.get("representation_mask") is None:
            metrics["anchor_kl"] = 10.0
        return metrics, representation

    monkeypatch.setattr(fitter, "evaluate", drifting_holdout)
    result = fit(config)
    assert len(result["epochs"]) == 1
    assert result["epochs"][0]["attempts"] == 1
    assert result["accepted_examples"] == 1
    assert result["epochs"][0]["validation"]["anchor_kl"] == 10.0
    assert (
        max(result["epochs"][0][k]["anchor_kl"] for k in ("train", "anchor"))
        < config["max_policy_kl"]
    )
    assert not (tmp_path / "fit/diagnostic.pt").exists()
    index = json.loads((tmp_path / "checkpoints/index.json").read_text())
    snapshot = torch.load(
        tmp_path / "checkpoints" / index["checkpoints"][-1]["filename"], weights_only=False
    )
    assert snapshot["training_contract"]["epochs"][0]["validation"]["anchor_kl"] == 10.0


def test_independent_grown_teachers_share_the_fixed_whole_game_split(tmp_path):
    from tools.eval_policy import _build_net_from_cfg

    config, _ = fit_config(tmp_path)
    source, target = data()
    sources = [dict(source, id=str(i), game_id=f"fit-{i}", reset_seed=[30, i]) for i in range(8)]
    targets = [dict(target, source_id=str(i), game_id=f"fit-{i}") for i in range(8)]
    for name, rows in (("sources", sources), ("targets", targets)):
        (tmp_path / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    reports = []
    for seed, width in ((29, 24), (71, 32)):
        output = tmp_path / f"fit-{seed}"
        reports.append(
            fit(
                dict(
                    config,
                    seed=seed,
                    split_seed=803,
                    output=str(output),
                    encoder_growth=dict(channels=width, blocks=2),
                )
            )
        )
        saved = torch.load(output / "diagnostic.pt", weights_only=False)
        net, _, _ = _build_net_from_cfg(saved["cfg"], 20, "cpu")
        net.load_state_dict(saved["state_dict"], strict=True)
        assert net.bottle_channels == width and len(net.bottle.blocks) == 2
        assert saved["training_contract"]["encoder_growth"]["seed"] == seed
        assert reports[-1]["accepted_examples"] == 6
    assert reports[0]["train_games"] == reports[1]["train_games"]
    assert reports[0]["validation_games"] == reports[1]["validation_games"]
    assert reports[0]["split_seed"] == reports[1]["split_seed"] == 803
