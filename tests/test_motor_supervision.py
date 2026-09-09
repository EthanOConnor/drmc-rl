import hashlib
import json
import os

import numpy as np
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
)
from tools.build_motor_opportunity_bank import annotate_row, split_for_seed
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
                      anchor_rows=1, minimum_anchor_games=1, max_policy_kl=.1))
    assert result["status"] == "Complete" and result["anchor_games"] == 1
    fitted = PlainPolicy(fit_output / "core-final.pt", public_only=True)
    assert not torch.equal(saved["state_dict"]["bottle.stem.weight"], fitted.net.bottle.stem.weight)
    # Auxiliary heads carry no inference cost when the normal live policy runs.
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary inference called the auxiliary heads")
    fitted.net.motor_auxiliary.forward = forbidden
    for obs, infos in observed:
        _, mask, logits = fitted.score(obs, infos)
        assert np.isfinite(logits[mask]).all()
