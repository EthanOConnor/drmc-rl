"""Deferred GPU planning parity: the vs pool with a reach_cuda plan_solver
must produce bit-identical decision outputs and trajectories vs the internal
CPU planner (drm_reach_bfs_v4)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.backends.drmario_pool import is_library_present

pytestmark = pytest.mark.skipif(
    not is_library_present(),
    reason="native pool library missing (build with: make -C game_engine libdrmario_pool)",
)


def _cuda_available() -> bool:
    try:
        from reach_cuda import CudaReach  # noqa: F401

        return True
    except Exception:
        return False


def _make_runners(num_pairs: int):
    from envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
    from training.envs.drmario_vs_vec import _make_gpu_plan_solver

    cpu = DrMarioVsPoolRunner(num_pairs=num_pairs)
    gpu = DrMarioVsPoolRunner(num_pairs=num_pairs, plan_solver=_make_gpu_plan_solver(2048))
    return cpu, gpu


def _reset_specs(num_pairs: int):
    from envs.backends.drmario_vs_pool import build_vs_reset_spec

    return [
        build_vs_reset_spec(
            level=(14, 14),
            speed_setting=(2, 2),
            rng_state=(0x10 + i, 0x37 + i),
            rng_override=True,
            frame_counter_base=100 + i,
        )
        for i in range(num_pairs)
    ]


@pytest.mark.skipif(not _cuda_available(), reason="reach_cuda unavailable")
def test_gpu_planner_trajectory_parity():
    num_pairs = 4
    decisions = 120
    cpu, gpu = _make_runners(num_pairs)
    specs = _reset_specs(num_pairs)

    cpu.reset(None, specs)
    gpu.reset(None, specs)

    rng = np.random.default_rng(0)
    for step_i in range(decisions):
        bc, bg = cpu.buffers, gpu.buffers
        np.testing.assert_array_equal(bc.need_action, bg.need_action, err_msg=f"step {step_i}")
        np.testing.assert_array_equal(
            bc.feasible_mask, bg.feasible_mask, err_msg=f"step {step_i} feasible"
        )
        np.testing.assert_array_equal(
            bc.cost_to_lock, bg.cost_to_lock, err_msg=f"step {step_i} costs"
        )
        np.testing.assert_array_equal(
            bc.board_bytes, bg.board_bytes, err_msg=f"step {step_i} boards"
        )
        assert not bg.plan_needed.any(), f"step {step_i}: uninjected plans remain"

        # Identical (randomly sampled) feasible action per parked side.
        acts = np.full(cpu.num_sides, -2, dtype=np.int32)
        for gi in range(cpu.num_sides):
            if bc.need_action[gi] == 0:
                continue
            feas = np.flatnonzero(bc.feasible_mask[gi])
            acts[gi] = -1 if feas.size == 0 else int(rng.choice(feas))

        reset_mask = bc.terminated | bc.truncated
        reset_specs = None
        if reset_mask.any():
            reset_specs = _reset_specs(num_pairs)
            np.testing.assert_array_equal(
                bc.terminated | bc.truncated, bg.terminated | bg.truncated
            )
        cpu.step(acts, reset_mask if reset_specs else None, reset_specs)
        gpu.step(acts, reset_mask if reset_specs else None, reset_specs)


@pytest.mark.skipif(not _cuda_available(), reason="reach_cuda unavailable")
def test_gpu_planner_vec_env_smoke():
    """DrMarioVsPoolVecEnv(gpu_planner=True) steps and produces sane masks."""

    from training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    env = DrMarioVsPoolVecEnv(num_pairs=2, level=14, speed_setting=2, gpu_planner=True)
    obs, infos = env.reset()
    assert obs.shape[0] == env.num_envs
    rng = np.random.default_rng(1)
    saw_feasible = False
    for _ in range(20):
        acts = []
        for i in range(env.num_envs):
            mask = np.asarray(infos[i]["placements/feasible_mask"]).reshape(-1)
            feas = np.flatnonzero(mask)
            saw_feasible = saw_feasible or feas.size > 0
            acts.append(-1 if feas.size == 0 else int(rng.choice(feas)))
        obs, rew, term, trunc, infos = env.step(acts)
    env.close()
    assert saw_feasible, "GPU-planned env never produced a feasible action"
