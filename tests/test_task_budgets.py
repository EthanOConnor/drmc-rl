from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from envs.retro.placement_env import DrMarioPlacementEnv
from envs.retro.placement_planner import PlanResult, SpawnReachability
from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH, ORIENTATIONS
from training.envs.curriculum import CurriculumConfig


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
    """Per-frame env stub: pill counter increments after N steps."""

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


class _TerminalFrameEnv(gym.Env):
    """Env stub that terminates with a clear and a terminal bonus on the first step."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(10)
        self._t = 0
        self._state_cache = _make_dummy_state(pill_counter=1)
        self._ram_offsets = {}
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        self._t = 0
        self._state_cache = _make_dummy_state(pill_counter=1)
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001
        self._t += 1
        obs = np.zeros((1,), dtype=np.float32)
        reward = 5.0
        terminated = True
        truncated = False
        info = {
            "goal_achieved": True,
            "cleared": True,
            "terminal_bonus_reward": 5.0,
            "r_env": 5.0,
            "r_total": 5.0,
        }
        return obs, reward, terminated, truncated, info


def test_spawn_budget_truncates_episode() -> None:
    base = _SpawnCounterFrameEnv(next_spawn_after_steps=3)
    env = DrMarioPlacementEnv(base, planner=_TrivialPlanner(), max_wait_frames=50)
    env.task_max_spawns = 1

    _obs, info = env.reset()
    assert int(info.get("task/spawns_used", -1)) == 0

    _obs2, _r2, terminated2, truncated2, info2 = env.step(0)
    assert not terminated2
    assert not truncated2
    assert int(info2.get("task/spawns_used", -1)) == 1

    _obs3, _r3, terminated3, truncated3, info3 = env.step(0)
    assert not terminated3
    assert truncated3
    assert bool(info3.get("task/budget_exceeded_spawns", False))


def test_frame_budget_truncates_episode() -> None:
    base = _SpawnCounterFrameEnv(next_spawn_after_steps=5)
    env = DrMarioPlacementEnv(base, planner=_TrivialPlanner(), max_wait_frames=50)
    env.task_max_frames = 4

    env.reset()
    _obs2, _r2, terminated2, truncated2, info2 = env.step(0)
    assert not terminated2
    assert truncated2
    assert bool(info2.get("task/budget_exceeded_frames", False))


def test_clear_over_budget_strips_terminal_bonus() -> None:
    base = _TerminalFrameEnv()
    env = DrMarioPlacementEnv(base, planner=_TrivialPlanner(), max_wait_frames=50)
    env.task_max_frames = 0

    env.reset()
    _obs2, r2, terminated2, truncated2, info2 = env.step(0)
    assert terminated2
    assert not truncated2
    assert bool(info2.get("task/budget_exceeded_frames", False))
    assert not bool(info2.get("goal_achieved", True))
    assert float(info2.get("terminal_bonus_reward", -1.0)) == 0.0
    assert float(r2) == 0.0


def test_curriculum_config_parses_task_budgets() -> None:
    cfg = CurriculumConfig.from_cfg(
        {
            "enabled": True,
            "task_max_frames": 1000,
            "task_max_spawns": {"-15": 20},
        }
    )
    assert cfg.task_max_frames == 1000
    assert cfg.task_max_spawns is None
    assert cfg.task_max_spawns_by_level.get(-15) == 20

