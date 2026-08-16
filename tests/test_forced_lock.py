from drmc_rl.envs.backends.vs_forced import ForcedLock, build_forced_lock_array


def test_forced_lock_array_preserves_exact_pair_frame() -> None:
    array = build_forced_lock_array(
        [ForcedLock(column=3, row_bottom=5, rotation=2, lock_frame=1234), ForcedLock.spectator()],
        num_sides=2,
    )
    assert array[0].col == 3
    assert array[0].lock_frame == 1234
    assert array[1].lock_frame == -2
