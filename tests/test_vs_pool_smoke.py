from __future__ import annotations

import numpy as np
import pytest

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present


def _has_vspool_symbols() -> bool:
    if not is_library_present():
        return False
    try:
        import ctypes

        from drmc_rl.envs.backends.drmario_pool import resolve_library_path

        lib = ctypes.CDLL(str(resolve_library_path()))
        return hasattr(lib, "drm_vspool_create")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_vspool_symbols(),
    reason="cpp-vs-pool library missing (build with: make -C vendor/drmario_native libdrmario_pool)",
)


def _random_feasible_actions(infos, rng) -> np.ndarray:
    actions = []
    for info in infos:
        mask = np.asarray(info.get("placements/feasible_mask"), dtype=np.uint8)
        assert mask.shape == (4, 16, 8)
        idxs = np.flatnonzero(mask.reshape(-1))
        if bool(info.get("placements/needs_action")) and idxs.size > 0:
            actions.append(int(rng.choice(idxs)))
        else:
            actions.append(-1)
    return np.asarray(actions, dtype=np.int32)


def test_vs_pool_smoke() -> None:
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    num_pairs = 2
    env = DrMarioVsPoolVecEnv(
        num_pairs=num_pairs,
        state_repr="bitplane_bottle_conn_mask",
        level=0,
        speed_setting=2,
        randomize_rng=True,
    )
    rng = np.random.default_rng(0)
    try:
        assert env.num_envs == 2 * num_pairs

        obs, infos = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (2 * num_pairs, 12, 16, 8)
        assert isinstance(infos, (list, tuple)) and len(infos) == env.num_envs

        # Fresh pairs: every side parked at its first decision with options.
        for info in infos:
            assert bool(info.get("placements/needs_action"))
            assert int(info.get("placements/options", 0)) > 0
            assert isinstance(info.get("next_pill_colors"), np.ndarray)
            assert "preview_pill" in info
            assert isinstance(info.get("placements/cost_to_lock"), np.ndarray)

        # Obs parity sanity: feasible planes mirror the info mask. For
        # same-color pills the obs planes are pre-symmetry-reduction (native
        # 1P build_obs parity), so they are a superset of the reduced mask.
        for i, info in enumerate(infos):
            mask = np.asarray(info["placements/feasible_mask"], dtype=np.float32)
            colors = np.asarray(info["next_pill_colors"])
            np.testing.assert_array_equal(obs[i, 8:10], mask[:2])
            if int(colors[0]) != int(colors[1]):
                np.testing.assert_array_equal(obs[i, 10:12], mask[2:])
            else:
                assert np.all(obs[i, 10:12] >= mask[2:])

        for _ in range(50):
            actions = _random_feasible_actions(infos, rng)
            obs, rewards, terminated, truncated, infos = env.step(actions)
            assert obs.shape == (2 * num_pairs, 12, 16, 8)
            assert np.asarray(rewards).shape == (env.num_envs,)
            assert np.asarray(terminated).shape == (env.num_envs,)
            assert np.asarray(truncated).shape == (env.num_envs,)
            for info in infos:
                assert int(info.get("placements/tau", 0)) >= 1
                assert "placements/feasible_mask" in info
                assert "placements/cost_to_lock" in info
                assert "next_pill_colors" in info
                assert "preview_pill" in info
            # Both sides of a pair share done flags.
            done = np.asarray(terminated) | np.asarray(truncated)
            for p in range(num_pairs):
                assert bool(done[2 * p]) == bool(done[2 * p + 1])
            for i in range(env.num_envs):
                if done[i]:
                    assert "episode" in infos[i]
                    assert "drm" in infos[i]

        # Run until matches complete (random play at level 0 tops out quickly;
        # the bound is generous to keep this robust).
        max_decisions = 5000
        for _ in range(max_decisions):
            if env.get_vs_metrics().get("vs/matches_total", 0.0) >= 2.0:
                break
            actions = _random_feasible_actions(infos, rng)
            obs, rewards, terminated, truncated, infos = env.step(actions)
        metrics = env.get_vs_metrics()
        assert metrics["vs/matches_total"] >= 1.0
        assert "vs/win_rate" in metrics
        assert "vs/match_len_p50_sec" in metrics
        assert metrics["vs/opponent_pool"] == 1.0
        # Volleys may or may not occur under random play; the counters must at
        # least be consistent and non-negative.
        assert metrics["vs/cpm"] >= 0.0
        assert metrics["vs/cpm_received"] == metrics["vs/cpm"]

        # Completed matches produce per-side skill records with the DrMC
        # feature names tools/skill_grade.py expects.
        games = env.pop_skill_games()
        assert len(games) >= 2 and len(games) % 2 == 0
        for g in games:
            for key in ("cpm", "cur", "spd", "salt_per_min", "pills_per_min", "garbage_per_min", "length_s"):
                assert key in g
        assert env.skill_games_pending() == 0
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)


def test_vs_pool_runner_need_action_noop() -> None:
    """Direct runner check: -1 (noop-fall) is accepted for parked sides and
    sides that are not parked ignore their action slot."""

    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec

    runner = DrMarioVsPoolRunner(num_pairs=1)
    try:
        spec = build_vs_reset_spec(level=(0, 0), speed_setting=(2, 2), rng_state=(7, 7), rng_override=True)
        runner.reset(None, [spec])
        buf = runner.buffers
        assert buf.need_action.shape == (2,)
        assert int(buf.need_action.sum()) == 2  # both parked after reset
        assert buf.board_bytes.shape == (2, 128)
        assert int(buf.feasible_mask.sum()) > 0

        # noop-fall for both sides: accepted, advances the pair clock.
        runner.step(np.asarray([-1, -1], dtype=np.int32), None, None)
        assert int(buf.invalid_action[0]) == -1
        assert int(buf.invalid_action[1]) == -1
        assert int(buf.side_frames.max()) > 0

        # Out-of-range action is surfaced as invalid for parked sides.
        if int(buf.terminated[0]) == 0:
            parked = buf.need_action.astype(bool)
            acts = np.where(parked, 512, -2).astype(np.int32)
            runner.step(acts, None, None)
            for i in range(2):
                if parked[i]:
                    assert int(buf.invalid_action[i]) == 512
    finally:
        runner.close()
