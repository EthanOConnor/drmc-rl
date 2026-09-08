import numpy as np
import pytest

from drmc_rl.human.repertoire import clear_geometry, placement_geometry


def test_cross_counts_union_once_and_ignores_clearing_markers():
    field = np.full((16, 8), 255, np.uint8)
    field[10, 1:7] = 0xD1
    field[8:12, 3] = 0x81
    field[15, :4] = 0xB1
    assert clear_geometry(field) == dict(horizontal_lines=1, vertical_lines=1,
        longest_horizontal=6, longest_vertical=4, crossing_cells=1, first_wave_cells=9)


def test_vertical_capsule_can_complete_horizontal_clear_without_mutating_root():
    field = np.full((16, 8), 255, np.uint8)
    field[15, :3] = 0xD1
    original = field.copy()
    result = placement_geometry(field, (0, 2), 3*128 + 15*8 + 3)
    assert result["horizontal_lines"] == 1
    assert result["vertical_lines"] == 0
    np.testing.assert_array_equal(field, original)


def test_geometry_rejects_overlapping_placement():
    with pytest.raises(ValueError, match="empty"):
        placement_geometry(np.full(128, 0xD0, np.uint8), (0, 1), 0)


def test_audit_rejects_incomplete_policy_candidate_coverage():
    from tools.audit_trainer_repertoire import infer

    class DroppingPolicy:
        def score(self, obs, infos):
            return np.asarray([[0]]), np.asarray([[True]]), np.asarray([[1.0]])

    row = dict(board=np.full(128, 255, np.uint8), opponent=np.full(128, 255, np.uint8),
        pill=[0, 1], preview=[0, 2], opponent_pill=[0, 1])
    cost = np.full(512, 65535, np.uint16)
    cost[:2] = 10
    with pytest.raises(ValueError, match="every feasible candidate"):
        infer(DroppingPolicy(), [row], [cost])


def test_audit_geometry_agrees_with_native_resolution_for_every_color():
    from drmc_rl.envs.backends.drmario_pool import is_library_present
    from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from drmc_rl.execution.pace import resolve_pace
    from tools.audit_trainer_repertoire import audit_root

    if not is_library_present():
        pytest.skip("native pool library missing")

    class FirstPolicy:
        def score(self, obs, infos):
            actions = np.broadcast_to(np.arange(512), (len(infos), 512))
            masks = np.asarray([info["placements/feasible_mask"].reshape(-1) for info in infos])
            return actions, masks, -actions.astype(float)

    planner = NativeReachabilityRunner(max_frames=2048)
    with NativeAfterstateSimulator(num_envs=128) as simulator:
        for canonical, raw in enumerate((1, 0, 2)):
            board = np.full((16, 8), 255, np.uint8)
            board[15, :3] = 0xD0 | raw
            row = dict(board=board.reshape(-1), opponent=board.reshape(-1),
                pill=[canonical, (canonical+1) % 3], preview=[0, 2], opponent_pill=[0, 1],
                columns=np.asarray([1 << 15 if col < 3 else 0 for col in range(8)], np.uint16),
                spawn=FrameState(3, 0, 0, 0, 0, HoldDir(0), 0, Rotation(0)),
                threshold=compute_speed_threshold(2, 0), speed=2, speed_ups=0)
            result = audit_root(row, resolve_pace("frame_perfect"), planner, FirstPolicy(), simulator)
            horizontal = [c for c in result["candidates_detail"] if c["geometry"]["horizontal_lines"]]
            assert horizontal
            assert all(c["viruses_cleared"] == 3 for c in horizontal)
