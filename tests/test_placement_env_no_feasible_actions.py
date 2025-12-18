from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from envs.retro.placement_env import DrMarioPlacementEnv
from envs.retro.placement_planner import SpawnReachability
from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH, ORIENTATIONS


class _NoFeasiblePlanner:
    """Planner stub that reports no feasible macro actions."""

    def build_spawn_reachability(self, board, snapshot) -> SpawnReachability:  # noqa: ANN001
        legal = np.zeros((ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
        legal[0, 0, 0] = True  # non-empty, to mimic a non-trivial board
        feasible = np.zeros_like(legal)
        costs = np.full_like(legal, np.inf, dtype=np.float32)
        return SpawnReachability(legal_mask=legal, feasible_mask=feasible, costs=costs)

    def plan_action(self, spawn, action):  # noqa: ANN001
        return None


def _make_dummy_state():
    ram = bytearray(0x0100)
    ram[0x0097] = 0x00  # currentP_nextAction == pillFalling
    ram[0x0086] = 0x0F  # fallingPillY from bottom (15) -> base_row=0 (top)
    ram[0x0085] = 0x03  # fallingPillX
    ram[0x00A5] = 0x03  # falling pill rotation
    ram[0x0081] = 0x01  # color1
    ram[0x0082] = 0x02  # color2
    ram[0x008B] = 0x00  # speed_setting
    ram[0x008A] = 0x00  # speed_ups
    ram[0x0092] = 0x00  # speed_counter
    ram[0x0093] = 0x00  # hor_velocity
    ram[0x0043] = 0x00  # frame_counter
    ram[0x00F7] = 0x00  # P1 buttons held

    planes = np.zeros((16, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    falling_mask = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    calc = SimpleNamespace(planes=planes, falling_mask=falling_mask)

    return SimpleNamespace(
        ram=SimpleNamespace(bytes=bytes(ram)),
        ram_vals=SimpleNamespace(pill_counter=1),
        calc=calc,
    )


class _FakeFrameEnv(gym.Env):
    """Minimal per-frame env that immediately terminates after one step."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self._t = 0
        self._state_cache = _make_dummy_state()
        self._ram_offsets = {}
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        self._t = 0
        self._state_cache = _make_dummy_state()
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001
        self._t += 1
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_skips_dead_decision_points_when_no_feasible_actions():
    base = _FakeFrameEnv()
    env = DrMarioPlacementEnv(base, planner=_NoFeasiblePlanner(), max_wait_frames=10)
    _obs, info = env.reset()
    # Regression: we must advance the underlying env instead of returning a
    # decision point with an all-false feasible mask (which would cause an
    # infinite invalid-action loop).
    assert base._t > 0
    assert not bool(info.get("placements/needs_action", False))
