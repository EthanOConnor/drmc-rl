"""reach_cuda-backed PlanSolver factory for the deferred-planning pools.

Shared by the VS training env (`drmc_rl/training/envs/drmario_vs_vec.py`), search
(`drmc_rl/models/policy/search_policy.py`), and any other caller that hands a
`plan_solver` to `DrMarioPoolRunner` / `DrMarioVsPoolRunner`. Bit-exact vs
the in-pool CPU planner (tests/test_vs_gpu_planner.py,
tests/test_pool_gpu_planner.py).
"""

from __future__ import annotations

import numpy as np


def make_gpu_plan_solver(max_lock_frames: int = 2048, *, max_batch: int = 2048):
    """Batched reach_cuda solver matching the runners' PlanSolver contract.

    Constructs the CUDA context (NVRTC JIT on first use, cached) eagerly so
    misconfigured hosts fail at setup, not mid-rollout.
    """

    from drmc_rl.planning.cuda import CudaReach

    ctx = CudaReach(max_batch=int(max_batch))

    def solve(ps: np.ndarray) -> np.ndarray:
        return ctx.solve_costs(
            np.ascontiguousarray(ps["cols"]),
            np.ascontiguousarray(ps["parity"]),
            np.ascontiguousarray(ps["thr"]),
            sx=np.ascontiguousarray(ps["x"]),
            sy=np.ascontiguousarray(ps["y_top"]),
            srot=np.ascontiguousarray(ps["rot"]),
            sc=np.ascontiguousarray(ps["sc"]),
            hv=np.ascontiguousarray(ps["hv"]),
            hd=np.ascontiguousarray(ps["hd"]),
            rh=np.ascontiguousarray(ps["rh"]),
            max_frames=int(max_lock_frames),
        )

    return solve
