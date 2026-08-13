from __future__ import annotations

import numpy as np
import pytest

from drmc_rl.seedlab import rng as slrng


def test_orbit_structure() -> None:
    seeds = slrng.orbit()
    assert len(seeds) == slrng.ORBIT_PERIOD == 32767
    assert seeds[0] == 0x8988
    assert len(set(seeds)) == len(seeds)
    assert slrng.orbit_pos(0x8988) == 0
    # 0x0000 is the lockup fixed point; never on the console orbit.
    assert slrng.orbit_pos(0x0000) == -1
    assert slrng.rng_step(0, 0) == (0, 0)


def test_seed_byte_packing_roundtrip() -> None:
    for seed in (0x0000, 0x8988, 0xFFFF, 0x0102):
        r0, r1 = slrng.seed_to_bytes(seed)
        assert slrng.bytes_to_seed(r0, r1) == seed


def test_game_content_deterministic() -> None:
    a = slrng.generate_game(10, 0x8988)
    b = slrng.generate_game(10, 0x8988)
    assert a == b
    assert slrng.game_hash(a) == slrng.game_hash(b)
    c = slrng.generate_game(10, slrng.orbit()[1])
    assert slrng.game_hash(a) != slrng.game_hash(c)
    assert a.virus_count == 44  # (10+1)*4


# ---------------------------------------------------------------- engine parity

_RAW_TO_CANONICAL = {0: 1, 1: 0, 2: 2}  # raw Y/R/B -> canonical R/Y/B


def _is_pool_present() -> bool:
    try:
        from drmc_rl.envs.backends.drmario_pool import is_library_present

        return bool(is_library_present())
    except Exception:
        return False


@pytest.mark.skipif(
    not _is_pool_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
@pytest.mark.parametrize("level", [0, 10, 20])
def test_engine_parity_board_and_pills(level: int) -> None:
    from drmc_rl.training.envs.drmario_pool_vec import DrMarioPoolVecEnv

    orbit = slrng.orbit()
    sample_seeds = [0x8988, orbit[1], orbit[1234], orbit[30000], 0x0102]

    env = DrMarioPoolVecEnv(
        num_envs=1,
        state_repr="bitplane_bottle_mask",
        level=level,
        speed_setting=2,
        randomize_rng=False,
        emit_board=True,
    )
    try:
        for seed in sample_seeds:
            r_bytes = slrng.seed_to_bytes(seed)
            _obs, infos = env.reset(options={"rng_seed_bytes": r_bytes})
            info = infos[0]

            expected = slrng.generate_game(level, seed)

            engine_board = np.asarray(info["board"], dtype=np.uint8).reshape(-1)
            mirror_board = np.frombuffer(expected.board, dtype=np.uint8)
            np.testing.assert_array_equal(
                engine_board, mirror_board,
                err_msg=f"virus board mismatch level={level} seed={seed:#06x}",
            )
            assert int(info["viruses_remaining"]) == expected.virus_count

            # First played pill is reserve[0]; preview is reserve[1].
            first_l, first_r = slrng.pill_colors_raw(expected.pills[0])
            next_colors = np.asarray(info["next_pill_colors"], dtype=np.int64)
            assert int(next_colors[0]) == _RAW_TO_CANONICAL[first_l]
            assert int(next_colors[1]) == _RAW_TO_CANONICAL[first_r]

            prev_l, prev_r = slrng.pill_colors_raw(expected.pills[1])
            preview = info["preview_pill"]
            assert int(preview["first_color"]) == prev_l
            assert int(preview["second_color"]) == prev_r
    finally:
        env.close()


@pytest.mark.skipif(
    not _is_pool_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
def test_seed_provider_per_env_seeds() -> None:
    from drmc_rl.training.envs.drmario_pool_vec import DrMarioPoolVecEnv

    seeds = [0x8988, slrng.orbit()[100]]
    env = DrMarioPoolVecEnv(
        num_envs=2,
        state_repr="bitplane_bottle_mask",
        level=7,
        speed_setting=2,
        randomize_rng=True,  # provider must take precedence
        emit_board=True,
        seed_provider=lambda i: slrng.seed_to_bytes(seeds[i]),
    )
    try:
        _obs, infos = env.reset()
        for i, seed in enumerate(seeds):
            expected = slrng.generate_game(7, seed)
            engine_board = np.asarray(infos[i]["board"], dtype=np.uint8).reshape(-1)
            np.testing.assert_array_equal(engine_board, np.frombuffer(expected.board, dtype=np.uint8))
    finally:
        env.close()
