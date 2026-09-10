import hashlib
import json
import os

import numpy as np
import pytest
import torch

from drmc_rl.execution.pace import PACES, BY_ID, strategy_context
from drmc_rl.human.anticipation import public_policy_inputs
from drmc_rl.human.pace_portfolio import PacePortfolio, read_portfolio
from drmc_rl.models.policy.pace_adapter import PaceAdapter, PacePolicy
from tools.eval_policy import _build_net_from_cfg
from tools.vs_head_to_head import PlainPolicy


@pytest.fixture
def portfolio_assets(tmp_path):
    torch.manual_seed(7127)
    cfg = dict(aux_spec="zero_v1_vs", candidate_architecture="g5", candidate_board_channels=16,
               candidate_d_model=16, encoder_blocks=1, pill_embed_dim=8,
               candidate_hidden_dim=24, candidate_cross_layers=1,
               candidate_interaction_layers=1, candidate_transformer_heads=2,
               candidate_patch_kernel=3)
    net, _, _ = _build_net_from_cfg(cfg, 20, "cpu")
    parent = tmp_path / "parent.pt"
    torch.save({"cfg": cfg, "state_dict": net.state_dict()}, parent)
    digest = hashlib.sha256(parent.read_bytes()).hexdigest()
    data = {"schema": "professor-pills-pace-opponents-v1", "parent_sha256": digest,
            "context_schema": "own-motor-gravity-v1", "adapters": {},
            "paces": {p.id: (None if p.id in ("super_human", "frame_perfect") else
                            p.id if p.id in ("sloth", "relaxed") else "e1") for p in PACES}}
    for name in ("sloth", "relaxed", "e1"):
        adapter = PaceAdapter(16, hidden=12)
        with torch.no_grad():
            adapter.actor[-1].weight.normal_(std=.3)
        path = tmp_path / f"{name}.pt"
        torch.save({"schema": "drmc-pace-adapter-v1", "parent_sha256": digest,
                    "context_schema": "own-motor-gravity-v1", "adapter_config": {"hidden": 12},
                    "state_dict": adapter.state_dict()}, path)
        data["adapters"][name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = tmp_path / "opponents.json"
    manifest.write_text(json.dumps(data))
    return parent, manifest, data


def test_shared_portfolio_matches_arena_at_every_pace_and_keeps_parent_exact(portfolio_assets):
    parent, manifest, data = portfolio_assets
    plain = PlainPolicy(parent, public_only=True)
    original = plain.net
    portfolio = PacePortfolio(plain, manifest, data["parent_sha256"])
    assert portfolio.core.base is original and plain.net is portfolio.core
    assert len(portfolio.warmup_paces) == 4
    own = np.zeros((8, 16, 8), np.float32)
    own[0, 15, 0] = 1
    costs = np.full(512, 65535, np.uint16)
    costs[[1, 3, 117, 135]] = [32, 46, 80, 91]
    observations, infos = public_policy_inputs(own, own, [0, 1], [1, 2], costs, [(0, 1), (2, 0)])
    for pace in PACES:
        for info in infos:
            info.update({"pace/id": pace.id, "pace/context": strategy_context(pace, {"speed": 2, "speed_ups": 8}, max(4, pace.reaction_frames))})
        name = data["paces"][pace.id]
        expected = (PlainPolicy(parent, public_only=True) if name is None else
                    PacePolicy(parent, adapter_path=manifest.parent / data["adapters"][name]["path"]))
        actual = portfolio.score(observations, infos)
        for a, b in zip(actual, expected.score(observations, infos), strict=True):
            np.testing.assert_array_equal(a, b)
        assert portfolio.core.selected is portfolio.core.motor is None


@pytest.fixture
def core_portfolio_assets(portfolio_assets):
    from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
    from drmc_rl.models.policy.controller_core import ControllerCorePolicy
    parent, manifest, data = portfolio_assets
    core = ControllerCorePolicy(parent, training=False, seed=781)
    path = manifest.parent / "public-core.pt"
    core.save(path, update=0)
    data["schema"] = "professor-pills-pace-opponents-v2"
    data["cores"] = {"public_core": {"path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "context_schema": PUBLIC_CONTEXT_SCHEMA}}
    for pace in ("sloth", "super_human", "frame_perfect"):
        data["paces"][pace] = "public_core"
    manifest.write_text(json.dumps(data))
    return parent, manifest, data


def test_core_portfolio_routes_before_encoding_and_preserves_other_paces(core_portfolio_assets, tmp_path):
    from test_human_backend import _afterstate_checkpoint
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.human.anticipation import execution_for_action
    from drmc_rl.human.backend import HumanBackend
    from drmc_rl.human.controller_context import controller_policy_inputs, live_controller_state

    parent, manifest, data = core_portfolio_assets
    human = tmp_path / "human.pt.gz"
    _afterstate_checkpoint(human)
    backend = HumanBackend(str(human), competitive_checkpoint=str(parent), pace_manifest=str(manifest))
    try:
        assert backend.capabilities()["anticipation"]["supported_paces"] == [
            "relaxed", "normal", "fast", "top_humans"]
        assert len(backend.competitive.warmup_paces) == 3
        with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
            pool.reset([17291])
            while not pool.states[1].falling:
                pool.step()
            wire = pool.public_state(1).to_dict()
            for side in wire["sides"]:
                del side["board_b64"]
            wire.update(schema="public-controller-history-v1", compute_frames=4)
            state = pool.semantic(1) | {"public_live_context": wire}
            for pace in PACES:
                name = data["paces"][pace.id]
                is_core = name in data["cores"]
                actual_state = live_controller_state(state) if is_core else state
                delay = max(4, pace.reaction_frames)
                candidate = backend._candidates(actual_state, delay, pace)
                expected = (PlainPolicy(manifest.parent / data["cores"][name]["path"], public_only=True)
                    if is_core else PacePolicy(parent, adapter_path=manifest.parent / data["adapters"][name]["path"]))
                obs, infos = controller_policy_inputs(expected, candidate, actual_state, pace, delay, 4)
                actions, masks, logits = expected.score(obs, infos)
                action = int(actions[0, np.argmax(logits[0])])
                witness = execution_for_action(candidate, action, pace, delay=delay, frame_id=12)
                request = dict(type="decide", strength_control="quality", target_rating=1600,
                    temperature=0, pace=pace.id, frame_id=12, execution_delay_frames=delay, state=state)
                actual = backend._infer(request, remaining_ms=10000)
                assert actual["placement"] == witness["placement"]
                assert actual["controller_frames"] == witness["controller_frames"]
                assert actual["execution"] == witness["execution"]
                assert actual["candidate_count"] == int(masks.sum())
                missing = {k:v for k,v in state.items() if k != "public_live_context"}
                if is_core:
                    with pytest.raises(ValueError, match="live public history"):
                        backend._infer({**request, "state": missing}, remaining_ms=10000)
                    rejected = backend.handle({"schema":"drmc-human-backend-v1", "type":"prepare_next",
                        "pace":pace.id, "state":state, "committed":actual})
                    assert rejected["type"] == "error"
                    assert "fresh public context" in rejected["error"]["message"]
                else:
                    legacy = backend._infer({**request, "state":missing}, remaining_ms=10000)
                    assert legacy["controller_frames"] == actual["controller_frames"]
            def reject_core(*args, **kwargs):
                raise AssertionError("lower skill must not score the selected Max core")
            backend.competitive.cores["public_core"].score = reject_core
            lower = backend._infer({**request, "strength_control":"regret", "state":missing}, remaining_ms=10000)
            assert lower["timing"]["movement"]["validated"]
    finally:
        backend.close()


def test_core_portfolio_portable_and_rejects_false_schema(core_portfolio_assets, tmp_path):
    from tools.package_human_backend import copy_pace_portfolio
    parent, manifest, data = core_portfolio_assets
    target = tmp_path / "relocated"
    identity = copy_pace_portfolio(manifest, parent, target)
    actual, paths = read_portfolio(target / "pace_opponents.json", data["parent_sha256"])
    assert actual == identity and all(p.is_relative_to(target) for p in paths.values())
    PacePortfolio(PlainPolicy(parent, public_only=True), target / "pace_opponents.json", data["parent_sha256"])
    data["cores"]["public_core"].update(path=str(parent), sha256=data["parent_sha256"])
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="declared public context"):
        PacePortfolio(PlainPolicy(parent, public_only=True), manifest, data["parent_sha256"])


def test_portfolio_rejects_changed_artifacts_routes_and_motor(portfolio_assets):
    parent, manifest, data = portfolio_assets
    with pytest.raises(ValueError, match="parent or schema"):
        read_portfolio(manifest, "wrong")
    data["paces"]["frame_perfect"] = "e1"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fastest parent"):
        read_portfolio(manifest, data["parent_sha256"])
    data["paces"]["frame_perfect"] = None
    manifest.write_text(json.dumps(data))
    portfolio = PacePortfolio(PlainPolicy(parent, public_only=True), manifest, data["parent_sha256"])
    with pytest.raises(ValueError, match="motor context"):
        portfolio.score(np.zeros((1, 20, 16, 8)), [{"pace/id": "sloth", "pace/context": np.zeros(8)}])
    (manifest.parent / "sloth.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_portfolio(manifest, data["parent_sha256"])


def test_packaged_portfolio_is_portable_and_companion_is_discovered(portfolio_assets, tmp_path):
    from tools.human_backend import resolve_pace_manifest
    from tools.package_human_backend import copy_pace_portfolio

    parent, manifest, data = portfolio_assets
    packaged = tmp_path / "relocated" / "models"
    identity = copy_pace_portfolio(manifest, parent, packaged)
    resolved = resolve_pace_manifest(None, packaged / "competitive_policy.pt.gz")
    assert resolved == packaged / "pace_opponents.json"
    reread, paths = read_portfolio(resolved, data["parent_sha256"])
    assert reread == identity and all(p.is_relative_to(packaged) for p in paths.values())
    assert resolve_pace_manifest(None, None) is None
    with pytest.raises(ValueError, match="competitive"):
        resolve_pace_manifest(str(resolved), None)
    with pytest.raises(FileNotFoundError):
        resolve_pace_manifest(str(packaged / "missing.json"), parent)


def test_preparation_conditions_adapter_on_next_gravity_and_zero_delay():
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.human.anticipation import NextTurnPreparer, execution_for_action
    from drmc_rl.human.backend import plan_candidates
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    class InspectPolicy:
        def score(self, observations, infos):
            self.infos = infos
            costs = np.stack([i["placements/cost_to_lock"].reshape(512) for i in infos])
            return np.broadcast_to(np.arange(512), costs.shape), costs != 65535, -costs.astype(float)

    policy, planner = InspectPolicy(), NativeReachabilityRunner()
    prepare = NextTurnPreparer(policy, planner, lib_path=os.environ.get("DRMC_FRAME_LIBRARY"))
    try:
        with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
            pool.reset([61183])
            while not pool.states[0].falling:
                pool.step()
            state = pool.semantic(0)
            # Public BCD count 9 predicts the next gravity acceleration.
            state["pill_counter_total"], state["speed_ups"] = 9, 3
            pace = BY_ID["top_humans"]
            candidate = plan_candidates(planner, state, 0, pace)
            move = execution_for_action(candidate, int(candidate[-2].actions[0]), pace)
            prepared = prepare.prepare(state, move, pace)
            assert prepared is not None and prepared["state"]["speed_ups"] == 4
            expected = strategy_context(pace, prepared["state"], 0)
            assert len(policy.infos) == 18
            for info in policy.infos:
                assert info["pace/id"] == pace.id
                np.testing.assert_array_equal(info["pace/context"], expected)
    finally:
        prepare.close()
        planner.close()


@pytest.mark.parametrize("pace", PACES, ids=lambda p: p.id)
def test_live_backend_matches_arena_choice_and_controller_witness(portfolio_assets, tmp_path, pace):
    from test_human_backend import _afterstate_checkpoint
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.human.anticipation import execution_for_action
    from drmc_rl.human.backend import HumanBackend
    from drmc_rl.human.controller_context import controller_policy_inputs

    parent, manifest, data = portfolio_assets
    human = tmp_path / "human.pt.gz"
    _afterstate_checkpoint(human)
    backend = HumanBackend(str(human), competitive_checkpoint=str(parent), pace_manifest=str(manifest))
    try:
        assert backend.capabilities()["anticipation"]["available"]
        assert backend.competitive_identity["pace_opponents"]["paces"] == data["paces"]
        with FrameVsPool(lib_path=os.environ.get("DRMC_FRAME_LIBRARY")) as pool:
            pool.reset([17291])
            while not pool.states[1].falling:
                pool.step()
            state = pool.semantic(1)
            delay = max(4, pace.reaction_frames)
            candidate = backend._candidates(state, delay, pace)
            observation, infos = controller_policy_inputs(backend.competitive, candidate, state, pace, delay, 4)
            np.testing.assert_array_equal(infos[0]["pace/context"], strategy_context(pace, state, delay))
            name = data["paces"][pace.id]
            arena = (PlainPolicy(parent, public_only=True) if name is None else
                     PacePolicy(parent, adapter_path=manifest.parent / data["adapters"][name]["path"]))
            actions, masks, logits = arena.score(observation, infos)
            action = int(actions[0, np.argmax(logits[0])])
            witness = execution_for_action(candidate, action, pace, delay=delay, frame_id=12)
            result = backend._infer({"type": "decide", "strength_control": "quality", "target_rating": 1600,
                                     "temperature": 0, "pace": pace.id, "frame_id": 12,
                                     "execution_delay_frames": delay, "state": state}, remaining_ms=10000)
            assert result["placement"] == witness["placement"]
            assert result["controller_frames"] == witness["controller_frames"]
            assert result["execution"] == witness["execution"]
            assert result["candidate_count"] == int(masks.sum())
            if pace.id == "normal":
                def reject_competitive(*args, **kwargs):
                    raise AssertionError("lower skill must retain its existing V3 decoder")
                backend.competitive.score = reject_competitive
                result = backend._infer({"type": "decide", "strength_control": "regret", "target_rating": 1600,
                                         "temperature": 0, "pace": pace.id, "frame_id": 12,
                                         "execution_delay_frames": delay, "state": state}, remaining_ms=10000)
                assert result["timing"]["movement"]["validated"]
    finally:
        backend.close()
