"""The single-upload CUDA opponent path must choose exactly the same actions
as the per-group reference path (same nets, same inputs, same argmax)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drmc_rl.envs.backends.drmario_pool import is_library_present

pytestmark = [
    pytest.mark.skipif(
        not is_library_present(),
        reason="native pool library missing",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_cuda_opponent_path_matches_reference(tmp_path):
    from pathlib import Path

    seeds = [
        Path("runs/bc_opponents/bc_lt1600.pt.gz"),
        Path("runs/bc_opponents/bc_gt2000.pt.gz"),
        Path("runs/best_agents/smdp_ppo_step535164979.pt.gz"),  # aux entry
    ]
    if not all(p.is_file() for p in seeds):
        pytest.skip("seed checkpoints not staged")

    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    env = DrMarioVsPoolVecEnv(
        num_pairs=6,
        level=14,
        speed_setting=2,
        opponent_pool_cfg={
            "enabled": True,
            "max_pool": 6,
            "dir": str(tmp_path / "pool"),
            "seed_paths": [str(p) for p in seeds],
            "device": "cuda",
        },
    )
    obs, infos = env.reset(seed=5)
    rng = np.random.default_rng(5)
    compared = 0
    for _ in range(25):
        # Recompute both paths on the same pre-step state.
        acts_fast = env._opponent_actions()
        # Reference: force the per-group path by rebuilding the group map.
        groups, entries = {}, {}
        for pair_i in range(env.num_pairs):
            gi = pair_i * 2 + 1
            if bool(env._pending_reset[pair_i]) or int(env._need_action[gi]) == 0:
                continue
            entry = env._pair_opponents[pair_i]
            if entry is None:
                continue
            groups.setdefault(entry.id, []).append(gi)
            entries[entry.id] = entry
        acts_ref = np.full((env.num_pairs,), -2, dtype=np.int32)
        for pair_i in range(env.num_pairs):
            gi = pair_i * 2 + 1
            if bool(env._pending_reset[pair_i]) or int(env._need_action[gi]) == 0:
                continue
            if env._pair_opponents[pair_i] is None:
                acts_ref[pair_i] = -1
        for entry_id, side_idxs in groups.items():
            chosen = env._forward_opponent(entries[entry_id], side_idxs)
            for k, gi in enumerate(side_idxs):
                acts_ref[gi // 2] = int(chosen[k])
            compared += len(side_idxs)

        np.testing.assert_array_equal(acts_fast, acts_ref)

        # Advance with random learner actions.
        acts = []
        for i in range(env.num_envs):
            mask = np.asarray(infos[i]["placements/feasible_mask"]).reshape(-1)
            feas = np.flatnonzero(mask)
            acts.append(-1 if feas.size == 0 else int(rng.choice(feas)))
        obs, rew, term, trunc, infos = env.step(acts)
    env.close()
    assert compared >= 40, f"only {compared} opponent decisions compared"
