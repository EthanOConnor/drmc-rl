"""Native training observations and semantic trainer inputs must stay distinct."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.game.observation import board_bytes_to_semantic_planes, legacy_vs_policy_boards
from drmc_rl.human.search import semantic_planes_to_nes_board


@pytest.fixture(params=[(False, False), (True, False), (False, True), (True, True)])
def native_bottles(request):
    if not is_library_present():
        pytest.skip("native pool library missing")
    import drmc_rl.game.specs.ram_to_state as ram_specs
    from drmc_rl.envs.backends.drmario_vs_pool import build_vs_reset_spec
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    previous = ram_specs.get_state_representation()
    boards = np.full((2, 16, 8), 0xFF, dtype=np.uint8)
    for side, col in ((0, 0), (1, 3)):
        boards[side, 14, col:col + 2] = (0x60, 0x71)
        boards[side, 15, col] = 0xD2
        boards[side, 13:16, 6] = 0xD1
    colors = np.asarray([[1, 1] if same else [1, 0] for same in request.param])
    spec = build_vs_reset_spec(
        level=(0, 0), speed_setting=(2, 2), rng_override=True, rng_state=(55, 145),
        checkpoint_enabled=True, checkpoint_board=boards,
        checkpoint_falling_colors=colors,
        checkpoint_preview_colors=np.asarray([[2, 0], [0, 2]]),
    )
    env = DrMarioVsPoolVecEnv(
        num_pairs=1, state_repr="bitplane_bottle_conn_mask_vs", level=0, speed_setting=2
    )
    env._build_reset_specs = lambda **kwargs: [spec]
    try:
        obs, infos = env.reset(seed=7)
        yield env, obs, infos
    finally:
        env.close()
        ram_specs.set_state_representation(previous)


def test_live_and_teacher_boards_match_native_frozen_actor_inputs(native_bottles):
    from drmc_rl.search.native_pair import capture_native_state
    from drmc_rl.search.strong_league import _policy_inputs
    from tools.eval_policy import _make_aux_builder

    env, obs, infos = native_bottles
    state = capture_native_state(env._runner, level=0, speed_setting=2)
    # Search snapshots carry native clocks. The training wrapper separately
    # accumulates max(1, tau); compare builders under the same elapsed context.
    aux_infos = [
        {**info, "task/frames_used": state.privileged.pair_clocks[side]}
        for side, info in enumerate(infos)
    ]
    aux = _make_aux_builder(72, aux_spec="v1_vs")._build_aux_batch(obs, aux_infos)
    for side in range(2):
        own = board_bytes_to_semantic_planes(infos[side]["board"])
        opponent = board_bytes_to_semantic_planes(infos[side]["vs/opponent_board"])
        assert own[6:8].any() and opponent[6:8].any()
        np.testing.assert_array_equal(
            semantic_planes_to_nes_board(own), infos[side]["board"]
        )
        encoded = legacy_vs_policy_boards(
            own, opponent, infos[side]["next_pill_colors"],
            infos[side]["vs/opponent_pill_colors"],
        )
        np.testing.assert_array_equal(encoded, obs[side, :16])
        assert own[6:8].any() and opponent[6:8].any()  # no semantic mutation

        teacher_obs, pill, preview, actions, costs, mask, teacher_aux = _policy_inputs(state, side)
        np.testing.assert_array_equal(teacher_obs, obs[side, :16])
        np.testing.assert_allclose(teacher_aux, aux[side], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(pill, env._runner.buffers.pill_colors[side])
        np.testing.assert_array_equal(preview, env._runner.buffers.preview_colors[side])
        valid = np.flatnonzero(infos[side]["placements/feasible_mask"].reshape(-1))
        np.testing.assert_array_equal(np.sort(actions[mask]), valid)
        np.testing.assert_array_equal(
            costs[mask], infos[side]["placements/cost_to_lock"].reshape(-1)[actions[mask]]
        )


def test_arena_and_training_pool_pass_complete_bonds_to_afterstates(native_bottles):
    from tools.tournament import _EntryPolicy

    env, obs, infos = native_bottles
    captured = []

    def score_batch(requests):
        captured.extend(requests)
        return [dict(competitive_score=np.zeros(len(row["candidate_actions"]))) for row in requests]

    runtime = SimpleNamespace(
        score_batch=score_batch,
        choose_quality=lambda scores, mask: int(np.flatnonzero(mask)[0]),
    )
    entry = _EntryPolicy.__new__(_EntryPolicy)
    entry.mode, entry.human, entry.human_v3 = "human", runtime, True
    entry.human_rating, entry.human_rating_sd, entry.human_opponent_rating = 1600, 0, 1600
    entry.human_temperature, entry.human_strength_control = 0, "quality"
    entry.runner = SimpleNamespace(env=env, level=0, speed_setting=2)
    arena_actions = entry.decide([0, 1], obs, infos, np.zeros(2))
    pool_actions = env._afterstate_opponent_actions(
        SimpleNamespace(candidate_max=128, rating=1600, rating_sd=0,
                        runtime=runtime, selection="quality"),
        [0, 1],
    )
    np.testing.assert_array_equal(arena_actions, pool_actions)
    assert len(captured) == 4
    for index, row in enumerate(captured):
        side = index % 2
        np.testing.assert_array_equal(
            semantic_planes_to_nes_board(row["board_planes"]), infos[side]["board"]
        )
        np.testing.assert_array_equal(
            semantic_planes_to_nes_board(row["opponent_board_planes"]),
            infos[side]["vs/opponent_board"],
        )


def test_teacher_preserves_every_candidate_beyond_128(native_bottles):
    from drmc_rl.search.native_pair import capture_native_state
    from drmc_rl.search.strong_league import _policy_inputs

    env, _obs, _infos = native_bottles
    state = capture_native_state(env._runner, level=0, speed_setting=2)
    # A large frontier is supplied independently of the usual small bottle;
    # the continuation adapter must preserve the engine's entire frontier.
    state = replace(state, legal_actions_by_side=(tuple(range(140)), ()),
                    action_costs_by_side=(tuple(range(1, 141)), ()))
    _obs, _pill, _preview, actions, costs, mask, _aux = _policy_inputs(state, 0)
    assert mask.sum() == 140
    np.testing.assert_array_equal(actions[mask], np.arange(140))
    np.testing.assert_array_equal(costs[mask], np.arange(1, 141))


def test_calibration_can_select_a_candidate_beyond_the_old_128_limit():
    import torch
    from tools.calibrate_strong_league_wdl import _batch_infer

    feasible = np.zeros((2, 4, 16, 8), dtype=bool)
    feasible[0].reshape(-1)[:140] = True
    batch = SimpleNamespace(
        feasible_mask=feasible,
        cost_to_lock=np.broadcast_to(np.arange(1, 513).reshape(1, 4, 16, 8), feasible.shape),
        pill_colors=np.zeros((2, 2), dtype=np.int64),
        preview_pill_colors=np.zeros((2, 2), dtype=np.int64),
        aux=np.zeros((2, 72), dtype=np.float32),
    )
    seen = []

    def net(obs, pill, preview, actions, costs, masks, *, aux):
        seen.append(actions[masks].tolist())
        return actions.float(), torch.tensor([.25, -.5])

    mixture = SimpleNamespace(weights=np.asarray([1.0]),
                              members=[SimpleNamespace(net=net, device="cpu")])
    chosen, values, members = _batch_infer(
        mixture, SimpleNamespace(policy_batch=lambda spec: batch), np.zeros((2, 16, 16, 8))
    )
    np.testing.assert_array_equal(chosen, [139, -1])
    np.testing.assert_array_equal(values, [.25, -.5])
    np.testing.assert_array_equal(members[:, 0], values)
    assert seen == [list(range(140))]


def test_public_value_audit_matches_actor_inputs_without_pending_attack_leak(native_bottles):
    import torch
    from drmc_rl.search.native_pair import capture_native_state, state_to_payload
    from tools.audit_search_predictions import score_public_values

    env, obs, infos = native_bottles
    state = capture_native_state(env._runner, level=0, speed_setting=2)
    captured = []

    def net(boards, pills, previews, actions, costs, masks, *, aux):
        captured.append((boards.numpy(), pills.numpy(), previews.numpy(), actions.numpy(),
                         masks.numpy(), aux.numpy()))
        return torch.zeros_like(costs), torch.arange(len(boards)).float()

    policy = SimpleNamespace(net=net, device="cpu", aux_dim=72)
    rows = [{**state_to_payload(state), "root_side": side} for side in (0, 1)]
    np.testing.assert_array_equal(score_public_values(policy, rows), [0, 1])
    boards, pills, previews, actions, masks, aux = captured[0]
    np.testing.assert_array_equal(boards[:, :16], obs[:, :16])
    np.testing.assert_array_equal(pills, env._runner.buffers.pill_colors)
    np.testing.assert_array_equal(previews, env._runner.buffers.preview_colors)
    assert not aux.any()
    for side in (0, 1):
        # Native observation tails retain duplicate same-color orientations;
        # actor candidates and live reachability use the canonical frontier.
        np.testing.assert_array_equal(boards[side, 16:], infos[side]["placements/feasible_mask"])
        np.testing.assert_array_equal(np.sort(actions[side, masks[side]]),
            np.flatnonzero(infos[side]["placements/feasible_mask"]))
    changed = replace(state, privileged=replace(state.privileged, pending_attacks=(4, 3)))
    changed_rows = [{**state_to_payload(changed), "root_side": side} for side in (0, 1)]
    score_public_values(policy, changed_rows)
    for first, second in zip(captured[0], captured[1], strict=True):
        np.testing.assert_array_equal(first, second)


def test_search_uses_live_public_policy_value_and_color_contract(native_bottles):
    from collections import OrderedDict
    from drmc_rl.search.native_pair import capture_native_state
    from drmc_rl.search.public_policy import PublicPolicyContinuation
    from drmc_rl.search.strong_league import DavidsonCalibration
    from tools.vs_head_to_head import PlainPolicy

    env, obs, _infos = native_bottles
    state = capture_native_state(env._runner, level=0, speed_setting=2)
    captured = []

    def net(boards, pills, previews, actions, costs, masks, *, aux):
        captured.append((boards.numpy(), pills.numpy(), previews.numpy(), aux.numpy()))
        return actions.float()/10, previews[:, 0].float()/10 + pills[:, 1].float()/10

    policy = object.__new__(PlainPolicy)
    policy.net, policy.device, policy.in_channels = net, "cpu", 20
    policy.public_only, policy.aux_dim, policy.aux_shim = True, 72, object()
    continuation = object.__new__(PublicPolicyContinuation)
    continuation.policy = policy
    continuation.calibration = DavidsonCalibration(1.0, 0.0, -3.0, "test")
    continuation._cache = OrderedDict()
    for side in (0, 1):
        prior, value = continuation._infer(state, side)
        assert sum(prior.values()) == pytest.approx(1.0)
        boards, pills, previews, aux = captured[-1]
        np.testing.assert_array_equal(boards[0, :16], obs[side, :16])
        np.testing.assert_array_equal(pills[0], env._runner.buffers.pill_colors[side])
        np.testing.assert_array_equal(previews[0], env._runner.buffers.preview_colors[side])
        assert not aux.any()
        assert continuation.evaluate(state, side) == continuation.calibration.wdl(value)
    assert len(captured) == 2  # value/prior share the same network forward
    changed = replace(state, privileged=replace(state.privileged, pending_attacks=(4, 3)))
    assert continuation._infer(changed, 0) == continuation._infer(state, 0)
    assert len(captured) == 2  # hidden attack bytes cannot change the public cache key
    opponent_only = replace(state, privileged=replace(state.privileged, need_action=(False, True)))
    opponent_value = continuation.evaluate(opponent_only, 1)
    root_value = continuation.evaluate(opponent_only, 0)
    assert (root_value.win, root_value.draw, root_value.loss) == (
        opponent_value.loss, opponent_value.draw, opponent_value.win)
    with pytest.raises(ValueError, match="acting side"):
        continuation._infer(opponent_only, 0)
    requests = [(state, 0), (state, 1), (changed, 0)]
    batched = continuation.infer_batch(requests)
    for (item, value), (source, side) in zip(batched, requests, strict=True):
        expected, expected_value = continuation._infer(source, side)
        assert item == pytest.approx(expected)
        assert value == pytest.approx(expected_value)
    assert continuation.infer_batch([]) == []
    with pytest.raises(ValueError, match="acting side"):
        continuation.infer_batch([(opponent_only, 0)])
    continuation.calibration = None
    with pytest.raises(ValueError, match="fitted W/D/L"):
        continuation.evaluate(state, 0)


@pytest.mark.skipif(not is_library_present(), reason="native pool library missing")
def test_losing_capsule_bonds_changes_exact_clear_and_gravity_outcomes():
    from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator

    board = np.full((16, 8), 0xFF, dtype=np.uint8)
    board[14, :2] = (0x60, 0x71)  # a hanging right half, supported through its bond
    board[15, 0] = 0xD2
    board[13:16, 6] = 0xD1
    planes = board_bytes_to_semantic_planes(board)
    masked = planes.copy()
    masked[6:8] = 0
    fields = np.stack((semantic_planes_to_nes_board(planes), semantic_planes_to_nes_board(masked)))
    # Vertical red/red at column 6, top row 11, clears the three red viruses.
    action = 128 + 11 * 8 + 6
    with NativeAfterstateSimulator(num_envs=2) as simulator:
        result = simulator.simulate_packed(
            fields=fields, pills=np.asarray([[1, 1], [1, 1]], dtype=np.uint8),
            previews=np.asarray([[0, 2], [0, 2]], dtype=np.uint8),
            candidate_actions=np.asarray([[action], [action]]),
            candidate_costs=np.asarray([[30], [30]], dtype=np.uint16),
            candidate_count=np.ones(2, dtype=np.int64), speed=np.asarray([2, 2]),
            speed_ups=np.zeros(2, dtype=np.int64),
        )
    assert not result.invalid.any()
    assert np.all(result.viruses_cleared == 3)
    assert result.fields[0, 14 * 8 + 1] == 0x71
    assert result.fields[0, 15 * 8 + 1] == 0xFF
    assert result.fields[1, 14 * 8 + 1] == 0xFF
    assert result.fields[1, 15 * 8 + 1] == 0x81
