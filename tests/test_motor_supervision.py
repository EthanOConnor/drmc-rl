import hashlib
import json
import os

import numpy as np
import pytest
import torch

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.backend import plan_candidates
from drmc_rl.human.controller_context import controller_policy_inputs
from drmc_rl.human.motor_opportunity import MotorOpportunityLabeler
from drmc_rl.models.policy.controller_core import ControllerCorePolicy, write_public_replay
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from drmc_rl.training.motor_supervision import (
    cache_reference, forward_motor, load_bank, make_motor_batch, motor_loss, upgrade_motor_model,
    initialize_motor_priors, cache_motor_features, cached_motor_batch, forward_cached_motor,
)
from tools.build_motor_opportunity_bank import annotate_row, split_for_seed
from tools.audit_motor_auxiliary import run as confirm_motor
from tools.eval_policy import _build_net_from_cfg
from tools.fit_motor_auxiliary import run
from tools.vs_head_to_head import PlainPolicy


def test_auxiliary_fit_updates_shared_core_and_preserves_deployment_contract(tmp_path):
    torch.manual_seed(717)
    torch.set_num_threads(1)
    cfg = {"smdp_ppo": dict(candidate_architecture="g5", candidate_board_channels=16,
        candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8, candidate_hidden_dim=24,
        candidate_cross_layers=1, candidate_interaction_layers=1, candidate_transformer_heads=2,
        candidate_patch_kernel=3, aux_spec="zero_v1_vs")}
    net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    parent = tmp_path / "parent.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), parent)
    actor = ControllerCorePolicy(parent)
    core = tmp_path / "core.pt"
    actor.save(core)
    replays = tmp_path / "public-replay"
    bank = tmp_path / "bank"
    (bank / "roots").mkdir(parents=True)
    seed = next(i for i in range(1000) if split_for_seed(17291, i) == "train" and
                split_for_seed(39577, i) == "validation")
    (bank / "config.json").write_text(json.dumps(dict(seed=seed, holdout_seeds=[61183])))
    planner = NativeReachabilityRunner()
    records, observed = [], []
    try:
        with FrameVsPool(2, lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool, MotorOpportunityLabeler(
            planner, lib_path=os.environ.get("DRMC_FRAME_LIBRARY")
        ) as labeler:
            pool.reset([17291, 39577])
            while not all(s.falling for s in pool.states):
                pool.step()
            for side, pace_name in enumerate(("sloth", "normal", "top_humans")):
                state = pool.semantic(side, public_context=True)
                pace = resolve_pace(pace_name)
                delay = max(4, pace.reaction_frames)
                candidate = plan_candidates(planner, state, delay, pace)
                obs, infos = controller_policy_inputs(actor, candidate, state, pace, delay, 4)
                observed.append((obs, infos))
                actor.score(obs, infos)
                row = actor.learning_records[0]
                row.update({"return": 1., "game_id": 0})
                game_seed = (17291, 39577)[side // 2]
                path = replays / f"update-{side + 1:05d}.npz"
                write_public_replay(path, [row], [dict(seed=game_seed, side=side % 2)],
                                    update=side + 1, pace=pace_name, level=14)
                if side == 0:
                    continue  # Independent unannotated policy anchor.
                with np.load(path, allow_pickle=False) as payload:
                    data = {key: payload[key] for key in payload.files}
                labels, data = annotate_row(data, 0, labeler)
                record = dict(id=str(side), path=f"roots/{side}.npz", source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              source_row=0, game_seed=game_seed, learner_port=side % 2,
                              split=split_for_seed(game_seed, seed), pace=pace_name, level=14,
                              **labels.summary())
                np.savez_compressed(bank / record["path"], **data)
                records.append(record)
    finally:
        planner.close()
    (tmp_path / "training.json").write_text(json.dumps(dict(updates=3)))
    (bank / "roots.jsonl").write_text("\n".join(map(json.dumps, records)) + "\n")
    (bank / "progress.json").write_text(json.dumps(dict(status="Complete", roots=2)))
    train, validation = load_bank(bank)
    saved = torch.load(core, weights_only=True)
    upgraded, upgraded_cfg = upgrade_motor_model(saved, device="cpu")
    cache_reference(upgraded, train + validation, batch_size=2, device="cpu")
    batch = make_motor_batch(train, device="cpu")
    output = forward_motor(upgraded, batch)
    original = actor.net(*batch["inputs"][:6], aux=batch["inputs"][6])
    assert torch.equal(original[0], output[0]) and torch.equal(original[1], output[1])
    assert output[2]["motor_reach_logits"].shape[-2:] == (2, 128)
    loss, metrics = motor_loss(output, batch)
    assert metrics["anchor_kl"] < 1e-6
    loss.backward()
    assert upgraded.bottle.stem.weight.grad.abs().sum() > 0
    assert upgraded.motor_auxiliary.opportunity.weight.grad.abs().sum() > 0
    # The observed counter conditions the predictor without changing the
    # existing deployed actor's tensor contract or its deterministic output.
    changed = {**batch, "geometry": batch["geometry"].clone()}
    changed["geometry"][:, 2] = 0
    changed_output = forward_motor(upgraded, changed)
    assert torch.equal(output[0], changed_output[0])
    assert not torch.equal(output[2]["motor_reach_logits"], changed_output[2]["motor_reach_logits"])
    again, again_cfg = upgrade_motor_model(dict(cfg=upgraded_cfg, state_dict=upgraded.state_dict()), device="cpu")
    assert again_cfg == upgraded_cfg
    for a, b in zip(upgraded.state_dict().values(), again.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    fit_output = tmp_path / "fit"
    result = run(dict(bank=str(bank), anchor_replay_directory=str(replays), checkpoint=str(core),
                      output=str(fit_output), seed=18, epochs=1, batch_size=1, lr=1e-5,
                      head_initialization="training_cell_prior", head_epochs=2, head_lr=1e-3,
                      joint_head_lr=1e-4,
                      anchor_rows=1, minimum_anchor_games=1, max_policy_kl=.1))
    assert result["status"] == "Complete" and result["anchor_games"] == 1
    assert result["head_accepted_examples"] == 2 and result["head_core_unchanged"]
    fitted = PlainPolicy(fit_output / "core-final.pt", public_only=True)
    assert not torch.equal(saved["state_dict"]["bottle.stem.weight"], fitted.net.bottle.stem.weight)
    # Auxiliary heads carry no inference cost when the normal live policy runs.
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary inference called the auxiliary heads")
    fitted.net.motor_auxiliary.forward = forbidden
    for obs, infos in observed:
        _, mask, logits = fitted.score(obs, infos)
        assert np.isfinite(logits[mask]).all()
    # A fresh natural controller game, not the fitting holdout, supplies the
    # independent prediction confirmation and all exact alternative labels.
    source_config = tmp_path / 'source-config.json'
    source_config.write_text(json.dumps(dict(holdout_seeds=[61183])))
    confirmation = tmp_path / 'confirmation'
    result = confirm_motor(dict(fit_directory=str(fit_output), checkpoint=str(core),
        bank=str(bank), training_config=str(source_config), output=str(confirmation),
        native_library=os.environ.get('DRMC_FRAME_LIBRARY'), seed=31, seeds=[61183],
        conditions=[dict(pace='normal', level=14)], roots_per_condition=6,
        minimum_seeds_per_condition=1, games_per_batch=2, batch_size=2))
    assert result['status'] == 'Complete'
    assessment = json.loads((confirmation / 'assessment.json').read_text())
    condition = assessment['conditions'][0]
    assert condition['games'] == 2 and condition['independent_seeds'] == 1
    assert condition['roots'] > 0 and condition['candidates'] >= condition['roots']
    assert condition['seed_metrics'][0]['seed'] == 61183
    assert condition['metrics']['reach_brier']['change_ci95'] is None
    assert condition['metrics']['reach_brier']['training_prior'] >= 0


def test_cached_motor_features_preserve_predictions_and_exclude_future_labels(tmp_path):
    """Use an actual G5 forward; cached head updates cannot touch the core."""
    torch.set_num_threads(1)
    torch.manual_seed(83)
    cfg = {"smdp_ppo": dict(candidate_architecture="g5", candidate_board_channels=16,
        candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8, candidate_hidden_dim=24,
        candidate_cross_layers=1, candidate_interaction_layers=1, candidate_transformer_heads=2,
        candidate_patch_kernel=3, aux_spec="zero_v1_vs")}
    original, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    parent_path = tmp_path / "parent.pt"
    torch.save(dict(cfg=cfg, state_dict=original.state_dict()), parent_path)
    actor = ControllerCorePolicy(parent_path)
    net, _ = upgrade_motor_model(dict(cfg=actor.cfg, state_dict=actor.net.state_dict()), device="cpu")
    rows = []
    for n in (2, 3):
        reachable = np.full((n, 2, 128), 65535, np.uint16)
        reachable[..., -8:] = 31
        rows.append(dict(record=dict(split="train"), weight=1., actions=np.arange(n),
            root_costs=np.full(n, 20, np.float32), observation=np.zeros((16, 16, 8), np.float32),
            pill=np.array([0, 1]), preview=np.array([1, 2]), public_context=np.zeros(net.aux_dim, np.float32),
            controller_geometry=np.zeros(13, np.int16), after_fields=np.full((n, 128), 255, np.uint8),
            root_terminal=np.zeros(n, np.uint8), root_viruses_cleared=np.zeros(n),
            root_nonviruses_cleared=np.zeros(n), root_clear_events=np.zeros(n),
            reachable_cells=reachable, clearable_cells=np.full_like(reachable, 65535)))
    with pytest.raises(ValueError, match="training roots only"):
        initialize_motor_priors(net, [{**rows[0], "record": {"split": "validation"}}])
    initialize_motor_priors(net, rows)
    torch.testing.assert_close(net.motor_auxiliary.opportunity.bias[:256].sigmoid().reshape(2, 128)[:, -8:],
                               torch.full((2, 8), .9999))
    cache_motor_features(net, rows, batch_size=2, device="cpu")
    cached = cached_motor_batch(rows, device="cpu")
    ordinary = forward_motor(net, make_motor_batch(rows, device="cpu"))
    from_cache = forward_cached_motor(net.motor_auxiliary, cached)
    for name in ("effect_predictions", "motor_reach_logits", "motor_clear_logits", "motor_log_cost"):
        for i, row in enumerate(rows):
            torch.testing.assert_close(ordinary[2][name][i, :len(row["actions"])],
                                       from_cache[2][name][i, :len(row["actions"])] , atol=1e-6, rtol=1e-6)
    before = {name: tensor.clone() for name, tensor in net.state_dict().items()}
    # Flipping future supervision cannot change cached public candidate features.
    features = [r["frozen_motor_features"].copy() for r in rows]
    rows[0]["clearable_cells"][:] = 1
    cache_motor_features(net, rows, batch_size=2, device="cpu")
    for expected, row in zip(features, rows, strict=True):
        np.testing.assert_array_equal(expected, row["frozen_motor_features"])
    optimizer = torch.optim.AdamW(net.motor_auxiliary.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        batch = cached_motor_batch(rows, device="cpu")
        loss, _ = motor_loss(forward_cached_motor(net.motor_auxiliary, batch), batch)
        loss.backward()
        optimizer.step()
    for name, value in net.state_dict().items():
        if not name.startswith("motor_auxiliary."):
            assert torch.equal(before[name], value)
            assert dict(net.named_parameters()).get(name, torch.empty(0)).grad is None
    assert not torch.equal(before["motor_auxiliary.opportunity.weight"], net.motor_auxiliary.opportunity.weight)
