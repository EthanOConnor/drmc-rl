from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from envs.retro.placement_env import DrMarioPlacementEnv
from envs.retro.placement_planner import PlanResult, SpawnReachability
from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH, ORIENTATIONS


class _TrivialPlanner:
    """Planner stub: exactly one feasible macro action and an empty script."""

    def build_spawn_reachability(self, board, snapshot) -> SpawnReachability:  # noqa: ANN001
        legal = np.zeros((ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
        feasible = np.zeros_like(legal)
        costs = np.full_like(legal, np.inf, dtype=np.float32)
        legal[0, 0, 0] = True
        feasible[0, 0, 0] = True
        costs[0, 0, 0] = 0.0
        return SpawnReachability(legal_mask=legal, feasible_mask=feasible, costs=costs)

    def plan_action(self, spawn, action):  # noqa: ANN001
        if int(action) != 0:
            return None
        return PlanResult(action=0, controller=[], cost=0, terminal_pose=(0, 0, 0))


def _make_dummy_state(*, pill_counter: int) -> SimpleNamespace:
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
        ram_vals=SimpleNamespace(pill_counter=int(pill_counter)),
        calc=calc,
    )


class _SpawnCounterFrameEnv(gym.Env):
    """Per-frame env stub: pill counter increments after N steps.

    The wrapper must expose exactly one macro decision per spawn, i.e. once it
    has executed a plan for spawn 1 it must *not* return another decision until
    `pill_counter` changes to spawn 2.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, next_spawn_after_steps: int = 5) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(10)
        self._t = 0
        self._steps = 0
        self._next_spawn_after_steps = int(next_spawn_after_steps)
        self._state_cache = _make_dummy_state(pill_counter=1)
        self._ram_offsets = {}
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        self._t = 0
        self._steps = 0
        self._state_cache = _make_dummy_state(pill_counter=1)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001
        self._t += 1
        self._steps += 1
        if self._steps == self._next_spawn_after_steps:
            self._state_cache = _make_dummy_state(pill_counter=2)
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_macro_env_is_spawn_latched_by_pill_counter() -> None:
    base = _SpawnCounterFrameEnv(next_spawn_after_steps=5)
    env = DrMarioPlacementEnv(base, planner=_TrivialPlanner(), max_wait_frames=50)

    _obs, info = env.reset()
    assert bool(info.get("placements/needs_action", False))
    assert int(info.get("placements/spawn_id", -1)) == 1

    # Consume the spawn with the only feasible macro action.
    _obs2, _r, terminated, truncated, info2 = env.step(0)
    assert not terminated
    assert not truncated
    assert bool(info2.get("placements/needs_action", False))
    assert int(info2.get("placements/spawn_id", -1)) == 2

    # Regression: without spawn-latching the wrapper would return immediately at
    # the next falling-pill frame and never advance the underlying env.
    assert base._t >= 5
