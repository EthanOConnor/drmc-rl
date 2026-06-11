"""Replay stored best solutions and check their recorded frame totals."""

from __future__ import annotations

from typing import Optional, Tuple

from seedlab import rng as slrng
from seedlab.db import CatalogDB, unpack_actions


def verify_solution(
    db: CatalogDB, *, level: int, speed: int, seed: int, state_repr: str = "bitplane_bottle_mask"
) -> Tuple[bool, str]:
    """Re-execute the stored trace in warp mode; check clear + frame total."""

    from training.envs.drmario_pool_vec import DrMarioPoolVecEnv

    row = db.solution(level=level, speed=speed, seed=seed)
    if row is None:
        return False, "no stored solution"
    frames_expected, spawns_expected, actions_blob, _solver, _at, _verified = row
    actions = unpack_actions(actions_blob)

    env = DrMarioPoolVecEnv(
        num_envs=1,
        state_repr=state_repr,
        level=int(level),
        speed_setting=int(speed),
        randomize_rng=False,
        seed_provider=lambda _i: slrng.seed_to_bytes(int(seed)),
    )
    try:
        _obs, infos = env.reset()
        frames = 0
        cleared = False
        for k, action in enumerate(actions):
            _obs, _r, terminated, truncated, infos = env.step([int(action)])
            info = infos[0]
            if info.get("placements/invalid_action") is not None and int(
                info.get("placements/invalid_action", -1)
            ) != -1:
                return False, f"invalid action at decision {k}"
            frames += max(1, int(info.get("placements/tau", 1)))
            if terminated[0] or truncated[0]:
                drm = info.get("drm", {}) if isinstance(info.get("drm"), dict) else {}
                cleared = bool(drm.get("cleared", False))
                if k != len(actions) - 1:
                    return False, f"episode ended early at decision {k + 1}/{len(actions)}"
                break
        if not cleared:
            return False, "trace did not clear the level"
        if frames != int(frames_expected):
            return False, f"frame mismatch: replay={frames} stored={int(frames_expected)}"
        if len(actions) != int(spawns_expected):
            return False, f"spawn mismatch: replay={len(actions)} stored={int(spawns_expected)}"
        return True, f"ok: {frames} frames, {len(actions)} spawns"
    finally:
        env.close()


def verify_and_mark(db: CatalogDB, *, level: int, speed: int, seed: int) -> Tuple[bool, str]:
    ok, msg = verify_solution(db, level=level, speed=speed, seed=seed)
    db.mark_solution_verified(level=level, speed=speed, seed=seed, ok=ok)
    return ok, msg
