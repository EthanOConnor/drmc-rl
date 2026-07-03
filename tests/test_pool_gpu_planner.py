"""Deferred GPU planning parity for the 1P pool + SearchPolicy.

The pool with a reach_cuda plan_solver must produce bit-identical decision
outputs and trajectories vs the internal CPU planner, and SearchPolicy with
gpu_planner=True must reproduce the CPU-planned search decisions exactly.
"""

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


def _make_runner(num_envs: int, gpu: bool):
    from envs.backends.drmario_pool import DrMarioPoolRunner
    from envs.backends.gpu_plan_solver import make_gpu_plan_solver

    return DrMarioPoolRunner(
        num_envs=num_envs,
        obs_spec=4,
        obs_channels=12,
        emit_board=True,
        plan_solver=make_gpu_plan_solver() if gpu else None,
    )


def _reset_specs(num_envs: int):
    from envs.backends.drmario_pool import build_reset_spec

    return [
        build_reset_spec(
            level=14,
            speed_setting=2,
            rng_state=(0x20 + i, 0x51 + i),
            rng_override=True,
        )
        for i in range(num_envs)
    ]


@pytest.mark.skipif(not _cuda_available(), reason="reach_cuda unavailable")
def test_pool_gpu_planner_trajectory_parity():
    num_envs = 8
    decisions = 100
    cpu = _make_runner(num_envs, gpu=False)
    gpu = _make_runner(num_envs, gpu=True)
    specs = _reset_specs(num_envs)

    cpu.reset(None, specs)
    gpu.reset(None, specs)

    rng = np.random.default_rng(0)
    for step_i in range(decisions):
        bc, bg = cpu.buffers, gpu.buffers
        np.testing.assert_array_equal(
            bc.feasible_mask, bg.feasible_mask, err_msg=f"step {step_i} feasible"
        )
        np.testing.assert_array_equal(
            bc.cost_to_lock, bg.cost_to_lock, err_msg=f"step {step_i} costs"
        )
        np.testing.assert_array_equal(
            bc.board_bytes, bg.board_bytes, err_msg=f"step {step_i} boards"
        )
        np.testing.assert_array_equal(bc.obs, bg.obs, err_msg=f"step {step_i} obs")
        assert not bg.plan_needed.any(), f"step {step_i}: uninjected plans remain"

        acts = np.zeros(num_envs, dtype=np.int32)
        reset_mask = (bc.terminated | bc.truncated).astype(np.uint8)
        np.testing.assert_array_equal(
            reset_mask, (bg.terminated | bg.truncated).astype(np.uint8)
        )
        for i in range(num_envs):
            feas = np.flatnonzero(bc.feasible_mask[i])
            acts[i] = -1 if feas.size == 0 else int(rng.choice(feas))
        reset_specs = _reset_specs(num_envs) if reset_mask.any() else None
        cpu.step(acts, reset_mask if reset_specs else None, reset_specs)
        gpu.step(acts, reset_mask if reset_specs else None, reset_specs)


CHAMPION = "runs/best_agents/smdp_ppo_step535164979.pt.gz"


@pytest.mark.skipif(not _cuda_available(), reason="reach_cuda unavailable")
def test_search_policy_gpu_planner_decision_parity():
    from pathlib import Path

    if not Path(CHAMPION).is_file():
        pytest.skip("champion checkpoint not staged")

    from models.policy.search_policy import SearchPolicy

    kw = dict(beam=4, num_sim_envs=16, deadline_ms=10_000.0, seed=0, warmup=False)
    sp_cpu = SearchPolicy(CHAMPION, gpu_planner=False, **kw)
    sp_gpu = SearchPolicy(CHAMPION, gpu_planner=True, **kw)

    # Drive real states from a CPU-planned pool and compare decisions.
    env = _make_runner(4, gpu=False)
    env.reset(None, _reset_specs(4))
    rng = np.random.default_rng(3)
    compared = 0
    for _ in range(12):
        buf = env.buffers
        acts = np.zeros(4, dtype=np.int32)
        for i in range(4):
            mask = buf.feasible_mask[i].copy()
            feas = np.flatnonzero(mask)
            if feas.size == 0 or buf.terminated[i] or buf.truncated[i]:
                acts[i] = -1
                continue
            board = buf.board_bytes[i].copy()  # flat 128, row 0 = top
            # pool colors are canonical (0=R,1=Y,2=B); decide wants raw NES
            # (0=Y,1=R,2=B)
            can2raw = {0: 1, 1: 0, 2: 2}
            pills = tuple(can2raw[int(v)] for v in buf.pill_colors[i])
            prev = tuple(can2raw[int(v)] for v in buf.preview_colors[i])
            costs = buf.cost_to_lock[i].copy()
            args = (board, pills, prev, mask, costs, 2, 0, 14)
            a_cpu, info_cpu = sp_cpu.decide(*args)
            a_gpu, info_gpu = sp_gpu.decide(*args)
            assert a_cpu == a_gpu, (
                f"search decision diverged: cpu={a_cpu} gpu={a_gpu} "
                f"(cpu info {info_cpu}, gpu info {info_gpu})"
            )
            compared += 1
            acts[i] = a_cpu
        reset_mask = (buf.terminated | buf.truncated).astype(np.uint8)
        reset_specs = _reset_specs(4) if reset_mask.any() else None
        env.step(acts, reset_mask if reset_specs else None, reset_specs)
    assert compared >= 20, f"only {compared} decisions compared"
