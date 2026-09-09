import json
import os

import numpy as np
import pytest
import torch

from drmc_rl.envs.backends.vs_frames import FrameVsPool
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.backend import plan_candidates
from drmc_rl.human.controller_context import controller_policy_inputs
from drmc_rl.models.policy.controller_core import ControllerCorePolicy, write_public_replay
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.eval_policy import _build_net_from_cfg
from tools.train_pace_strategy import _policy_snapshot, update_adapter
from tools.vs_head_to_head import PlainPolicy


@pytest.fixture
def parent(tmp_path):
    torch.manual_seed(941)
    torch.set_num_threads(1)
    cfg = {"smdp_ppo": dict(
        candidate_architecture="g5", candidate_board_channels=16,
        candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8,
        candidate_hidden_dim=24, candidate_cross_layers=1,
        candidate_interaction_layers=1, candidate_transformer_heads=2,
        candidate_patch_kernel=3, aux_spec="zero_v1_vs",
    )}
    net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    path = tmp_path / "parent.pt"
    torch.save(dict(cfg=cfg, state_dict=net.state_dict()), path)
    return path


def controller_requests(actor):
    planner = NativeReachabilityRunner()
    try:
        with FrameVsPool(2, lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
            pool.reset([17291, 39577])
            while not all(s.falling for s in pool.states):
                pool.step()
            observations, infos = [], []
            for side, pace_name in enumerate(("sloth", "normal", "top_humans", "frame_perfect")):
                pace = resolve_pace(pace_name)
                state = pool.semantic(side, public_context=True)
                delay = max(4, pace.reaction_frames)
                candidate = plan_candidates(planner, state, delay, pace)
                obs, info = controller_policy_inputs(actor, candidate, state, pace, delay, 4)
                observations.append(obs)
                infos.extend(info)
            return np.concatenate(observations), infos
    finally:
        planner.close()


def test_outcome_gradients_reach_the_full_core_and_saved_policy_reloads(parent, tmp_path):
    actor = ControllerCorePolicy(parent, seed=491)
    observations, infos = controller_requests(actor)
    actions, masks, selected = actor.score(observations, infos)
    records = actor.learning_records
    assert all(row["action"] == actions[i, selected[i].argmax()] for i, row in enumerate(records))
    assert all(len(row["actions"]) == int(masks[i].sum()) for i, row in enumerate(records))
    assert not any("candidate" in row for row in records)  # No frozen trunk-feature training.
    for i, row in enumerate(records):
        row.update({"return": 1.0 if i % 2 else -1.0, "weight": 1.0, "game_id": i})
    before = {name: p.detach().clone() for name, p in actor.net.named_parameters()}
    optimizer = torch.optim.AdamW(actor.net.parameters(), lr=2e-4)
    activity = []
    metrics = update_adapter(actor, optimizer, records,
                             {"minibatch": 2, "epochs": 2, "max_update_kl": .05}, 71,
                             activity=lambda phase, **work: activity.append(dict(phase=phase, **work)))
    assert metrics["optimizer_steps"] > 0 and metrics["update_kl"] <= .05
    assert {"auditing_collection", "optimizing", "checking_update"} <= {w["phase"] for w in activity}
    assert max(w.get("step", 0) for w in activity) == 2
    assert not torch.equal(before["bottle.stem.weight"], actor.net.bottle.stem.weight)
    assert not torch.equal(before["condition.0.weight"], actor.net.condition[0].weight)
    assert all(p.grad is None for p in actor.reference.parameters())
    path = tmp_path / "trained.pt"
    actor.save(path, update=1, optimizer=optimizer.state_dict())
    actor.training = False
    restored = PlainPolicy(path, public_only=True)
    resumed = ControllerCorePolicy(parent, resume=path, training=False, seed=491)
    expected = actor.score(observations, infos)
    for actual in (restored.score(observations, infos), resumed.score(observations, infos)):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
    # Parent regularization must stay at initialization when resuming updates.
    assert torch.equal(before["bottle.stem.weight"], resumed.reference.bottle.stem.weight)
    with pytest.raises(ValueError, match="causal observation"):
        restored.score(observations, [{k: v for k, v in i.items() if k != "vs/observation_timeline"} for i in infos])


def test_public_teacher_replay_has_complete_frontiers_and_separate_outcome_labels(parent, tmp_path):
    actor = ControllerCorePolicy(parent, seed=17)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    rows = actor.learning_records
    for i, row in enumerate(rows):
        row.update({"return": 1.0 if i % 2 else -1.0, "game_id": i})
    path = tmp_path / "replay.npz"
    games = [dict(seed=17291 + i, side=i % 2) for i in range(len(rows))]
    write_public_replay(path, rows, games, update=3, pace="normal", level=14)
    with np.load(path, allow_pickle=False) as replay:
        metadata = json.loads(str(replay["metadata"]))
        assert metadata["schema"] == "drmc-public-controller-replay-v2"
        assert metadata["observation_schema"] == actor.aux_spec
        assert "return" not in metadata["actor_inputs"]
        assert not {"seed", "raw_ram", "pending_attack", "restore"}.intersection(metadata["actor_inputs"])
        for i, row in enumerate(rows):
            lo, hi = replay["offsets"][i:i + 2]
            np.testing.assert_array_equal(replay["actions"][lo:hi], row["actions"])
            np.testing.assert_array_equal(replay["costs"][lo:hi], row["costs"])
            np.testing.assert_array_equal(replay["public_context"][i], row["public_context"])
            assert replay["actions"][lo + replay["slot"][i]] == replay["action"][i]
            assert replay["game_seed"][i] == games[i]["seed"]
            from drmc_rl.human.motor_opportunity import state_from_controller_replay
            restored = state_from_controller_replay(replay, i)
            np.testing.assert_array_equal(restored["board_planes"], row["observation"][:8])
            assert restored["falling"]["frame_parity"] == infos[i]["public_controller_geometry"]["frame_parity"]


def test_collection_audit_preserves_behavior_and_detects_real_distribution_drift(parent):
    actor = ControllerCorePolicy(parent, seed=38)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    rows = actor.learning_records
    for row in rows:
        row.update({"return": 1., "advantage": 1., "weight": 1.})
    row = rows[-1]  # The unrestricted frontier has enough rare alternatives.
    original = row["behavior_logp"].copy()
    # A tiny perturbation to a low-probability move can exceed the old absolute
    # logp tolerance while moving less than 1e-5 of total probability mass.
    perturbed = torch.tensor(original)
    perturbed[perturbed.argmin()] += 1e-4
    row["behavior_logp"] = perturbed.log_softmax(-1).numpy()
    row["old_logprob"] = float(row["behavior_logp"][row["slot"]])
    actual, agreement = _policy_snapshot(actor, rows, 2)
    np.testing.assert_array_equal(actual[-1].numpy(), row["behavior_logp"])
    assert 0 < agreement["collection_max_total_variation"] <= 1e-5
    assert agreement["collection_max_logp_error"] > 3e-5
    shifted = torch.tensor(original) + torch.linspace(-.1, .1, len(original))
    row["behavior_logp"] = shifted.log_softmax(-1).numpy()
    row["old_logprob"] = float(row["behavior_logp"][row["slot"]])
    with pytest.raises(RuntimeError, match="total variation"):
        _policy_snapshot(actor, rows, 2)


def test_motor_opportunity_bank_uses_committed_complete_controller_replay(parent, tmp_path):
    from tools.build_motor_opportunity_bank import run
    from drmc_rl.human.motor_opportunity import state_from_controller_replay

    actor = ControllerCorePolicy(parent, seed=21)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    row = actor.learning_records[1]  # Actual Normal profile, seed 17291 / P2.
    row.update({"return": 1., "game_id": 0})
    replay_dir = tmp_path / "public-replay"
    path = replay_dir / "update-00001.npz"
    write_public_replay(path, [row], [dict(seed=17291, side=1)], update=1, pace="normal", level=14)
    progress = tmp_path / "training.json"
    progress.write_text(json.dumps(dict(updates=0)))
    config = dict(replay_directory=str(replay_dir), output=str(tmp_path / "labels"),
                  native_library=os.environ.get("DRMC_FRAME_LIBRARY"), seed=927,
                  holdout_seeds=[61183], max_roots=1, per_game=1, per_update=1)
    result = run(config)
    assert result["roots"] == 0  # Uncommitted replay is not a training source.
    progress.write_text(json.dumps(dict(updates=1)))
    result = run(config)
    assert result["status"] == "Complete" and result["roots"] == 1
    journal = tmp_path / "labels" / "roots.jsonl"
    record = json.loads(journal.read_text())
    with np.load(tmp_path / "labels" / record["path"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["actions"], np.sort(row["actions"]))
        assert data["next_costs"].shape == (len(row["actions"]), 2, 512)
        assert int(data["observed_action"]) == row["action"]
    original = journal.read_text()
    assert run(config)["roots"] == 1
    assert journal.read_text() == original
    with np.load(path, allow_pickle=False) as data:
        old = {key: data[key] for key in data.files}
    old_metadata = json.loads(str(old["metadata"]))
    old_metadata["schema"] = "drmc-public-controller-replay-v1"
    old["metadata"] = np.asarray(json.dumps(old_metadata))
    with pytest.raises(ValueError, match="exact v2"):
        state_from_controller_replay(old, 0)


def test_mixed_public_context_and_frozen_actors_have_frame_event_parity(parent):
    from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
    from tools.trainer_planning_arena import run_batch

    actors = {"core": ControllerCorePolicy(parent, training=False),
              "parent": PlainPolicy(parent, public_only=True)}
    config = dict(native_library=os.environ.get("DRMC_FRAME_LIBRARY"),
                  variants={name: {"delay": 4} for name in actors},
                  max_game_frames=1400, replay_games=0)
    match = dict(a="core", b="parent", games=2, level=14, pace="fast")
    jobs = [(17291, 0, 0), (17291, 1, 1)]
    reference, parallel = NativeReachabilityRunner(), ParallelPlanning(2)
    try:
        expected, _ = run_batch(config, match, jobs, None, reference, None, policies=actors)
        activity = []
        actual, _ = run_event_batch(config, match, jobs, None, parallel, None,
                                    policies=actors, activity=activity.append)
        assert activity[-1]["games"] == len(jobs)
        assert activity[-1]["frames"] == sum(row["frames"] for row, _, _ in actual)
        assert activity[-1]["decision_requests"] == sum(row["a_stats"]["decisions"] for row, _, _ in actual)
        for left, right in zip(expected, actual):
            for key in ("seed", "side", "index", "score", "winner", "reason", "frames"):
                assert left[0][key] == right[0][key]
            for side in ("a_stats", "b_stats"):
                for key in ("decisions", "feasible_candidates", "forced_placements",
                            "spawn_wait_frames", "validated_input_frames", "no_reachable_after_delay"):
                    assert left[0][side].get(key, 0) == right[0][side].get(key, 0)
                assert right[0][side]["unplanned_locks"] == 0
            assert left[1] == right[1]
    finally:
        reference.close()
        parallel.close()


def test_collection_audit_rechecks_batch_shape_outliers_without_weakening_bound(parent, monkeypatch):
    actor = ControllerCorePolicy(parent, seed=53)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    rows = actor.learning_records
    for row in rows:
        row.update({"return": 1., "advantage": 1., "weight": 1.})
    original = actor.training_forward

    def batch_rounding(features):
        logits, values = original(features)
        if len(features[0]) > 1:
            logits = logits + torch.linspace(-.01, .01, logits.shape[1])
        return logits, values

    monkeypatch.setattr(actor, "training_forward", batch_rounding)
    actual, agreement = _policy_snapshot(actor, rows, 4)
    assert agreement["collection_max_total_variation"] > 1e-4
    assert agreement["collection_rechecked_decisions"] > 0
    assert agreement["collection_max_rechecked_total_variation"] <= 1e-5
    for row, logs in zip(rows, actual):
        np.testing.assert_array_equal(row["behavior_logp"], logs)
    # The single-row fallback must not turn into permission to accept actual
    # changed distributions, even when the first forward was batched.
    shifted = torch.tensor(rows[-1]["behavior_logp"]) + torch.linspace(-.1, .1, len(rows[-1]["actions"]))
    rows[-1]["behavior_logp"] = shifted.log_softmax(-1).numpy()
    rows[-1]["old_logprob"] = float(rows[-1]["behavior_logp"][rows[-1]["slot"]])
    with pytest.raises(RuntimeError, match="total variation"):
        _policy_snapshot(actor, rows, 4)


def test_collection_versions_detect_even_tiny_network_writes(parent):
    actor = ControllerCorePolicy(parent, seed=57)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    records = actor.learning_records
    with torch.no_grad():
        next(actor.net.parameters()).add_(1e-10)
    with pytest.raises(RuntimeError, match="network changed"):
        actor.score(obs, infos)
    with pytest.raises(RuntimeError, match="network changed"):
        actor.finish_collection(records)


def test_collection_versions_cannot_mix_updates(parent):
    actor = ControllerCorePolicy(parent, seed=59)
    obs, infos = controller_requests(actor)
    actor.score(obs, infos)
    old = actor.learning_records
    actor.finish_collection(old)
    actor.score(obs, infos)
    with pytest.raises(RuntimeError, match="versions were mixed"):
        actor.finish_collection(old + actor.learning_records)


def test_activity_reports_real_work_without_inflating_completed_counters(tmp_path, monkeypatch):
    from tools.train_pace_strategy import TrainingActivity

    clock = [0.]
    monkeypatch.setattr("tools.train_pace_strategy.time.monotonic", lambda: clock[0])
    path = tmp_path / "training.json"
    completed = dict(updates=2, frames=4000, decisions=100)
    progress = dict(status="Running", **completed)
    report = TrainingActivity(path, progress)
    report("collecting", games=0, frames=10)
    before = path.read_bytes()
    clock[0] = 300.
    report("collecting", games=0, frames=10)
    assert path.read_bytes() == before  # No fresh timestamp from idle polling.
    report("collecting", games=2, frames=300)
    saved = json.loads(path.read_text())
    assert {k: saved[k] for k in completed} == completed
    assert saved["activity"]["frames"] == 300
    clock[0] += 1.
    report("optimizing", epoch=1, step=1)
    assert json.loads(path.read_text())["phase"] == "optimizing"


def test_controller_training_command_finishes_with_activity_reporting(parent, tmp_path):
    import subprocess
    import sys

    output = tmp_path / "run"
    config = dict(checkpoint=str(parent), output=str(output),
                  working_db=str(tmp_path / "working.sqlite"),
                  native_library=os.environ.get("DRMC_FRAME_LIBRARY"), device="cpu",
                  threads=1, seed=6153, updates=1, games_per_update=2, paces=["normal"],
                  holdout_seeds=[], level20_fraction=0, max_game_frames=12000,
                  training_model="public_core", public_replay=True,
                  rollout_backend="events", async_planning=False, planner_workers=1,
                  epochs=1, minibatch=32, lr=3e-6, checkpoint_keep_last=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-m", "tools.train_pace_strategy", "--config", str(path)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    state = json.loads((output / "training.json").read_text())
    assert state["status"] == "Training complete"
    assert state["updates"] == 1 and state["decisions"] > 0
    assert state["activity"] is None
    assert (output / "core-final.pt").is_file()
    assert (output / "public-replay/update-00001.npz").is_file()
