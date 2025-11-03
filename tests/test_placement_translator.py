import numpy as np

from envs.retro.placement_actions import action_count, edge_from_index
from envs.retro.placement_wrapper import PlacementTranslator
from envs.retro.placement_planner import PlacementPlanner, PlannerParams


class _StubEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self._ram_offsets = {
            "falling_pill": {
                "row_addr": "0x0306",
                "col_addr": "0x0305",
                "orient_addr": "0x0325",
                "left_color_addr": "0x0301",
                "right_color_addr": "0x0302",
            },
            "gravity_lock": {
                "gravity_counter_addr": "0x0312",
                "lock_counter_addr": "0x0307",
                "speed_setting_addr": "0x030B",
                "speed_index_addr": "0x0320",
            },
            "pill_counter": {"addr": "0x0310"},
            "bottle": {"base_addr": "0x0400", "stride": 8},
            "timers": {"frame_counter_addr": "0x0043"},
            "inputs": {"p1_buttons_held_addr": "0x00F7"},
        }
        self._ram = np.zeros(0x800, dtype=np.uint8)
        self._ram[0x0305] = 3  # column
        self._ram[0x0306] = 15  # row from bottom -> board row 0
        self._ram[0x0325] = 0  # orientation
        self._ram[0x0301] = 1
        self._ram[0x0302] = 2
        self._ram[0x0312] = 6
        self._ram[0x0307] = 0
        self._ram[0x0310] = 5
        self._ram[0x030B] = 0
        self._ram[0x0320] = 0x40
        self._ram[0x0092] = 6
        self._ram[0x0093] = 0
        self._ram[0x008A] = 0x40
        self._ram[0x0043] = 0
        self._ram[0x00F7] = 0

    def _read_ram_array(self, refresh: bool = True) -> np.ndarray:
        return self._ram.copy()


def test_translator_refresh_populates_masks():
    env = _StubEnv()
    translator = PlacementTranslator(env, planner=PlacementPlanner(params=PlannerParams(max_search_frames=40)))
    translator.refresh()
    translator.prepare_options(force=True)
    info = translator.info()
    assert info["placements/options"] >= 1
    assert info["placements/costs"].shape[0] == action_count()
    assert info["placements/path_indices"].shape[0] == action_count()
    assert info["pill/speed_counter"] == 6
    assert "placements/stats/expanded" in info
    diag = translator.diagnostics()
    assert diag["plan_latency_ms"] >= 0.0
    assert diag["plan_count"] >= info["placements/options"]
    assert "planner/expanded" in diag


def test_identical_color_masks_canonical_direction():
    env = _StubEnv()
    env._ram[0x0301] = 2
    env._ram[0x0302] = 2
    translator = PlacementTranslator(env, planner=PlacementPlanner(params=PlannerParams(max_search_frames=40)))
    translator.refresh()
    translator.prepare_options(force=True)
    info = translator.info()
    legal = info["placements/legal_mask"]
    feasible = info["placements/feasible_mask"]
    path_indices = info["placements/path_indices"]
    duplicates = np.flatnonzero(legal)
    assert duplicates.size > 0
    # Ensure that for identical colors, at most one direction per undirected pair remains legal.
    seen = {}
    for idx in np.flatnonzero(legal):
        edge = edge_from_index(int(idx))
        key = tuple(sorted((edge.origin, edge.dest)))
        assert key not in seen
        seen[key] = idx
    # All feasible actions must reference a valid path index.
    for idx in np.flatnonzero(feasible):
        assert path_indices[idx] >= 0
