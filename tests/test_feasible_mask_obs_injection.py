from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import envs.specs.ram_to_state as ram_specs
from envs.retro.placement_env import DrMarioPlacementEnv
from envs.retro.placement_planner import SpawnReachability
from envs.retro.placement_space import GRID_HEIGHT, GRID_WIDTH, ORIENTATIONS


class _TrivialPlanner:
    """Planner stub: exactly one feasible macro action."""

    def build_spawn_reachability(self, board, snapshot) -> SpawnReachability:  # noqa: ANN001
        legal = np.zeros((ORIENTATIONS, GRID_HEIGHT, GRID_WIDTH), dtype=np.bool_)
        feasible = np.zeros_like(legal)
        costs = np.full_like(legal, np.inf, dtype=np.float32)
        legal[0, 0, 0] = True
        feasible[0, 0, 0] = True
        costs[0, 0, 0] = 0.0
        return SpawnReachability(legal_mask=legal, feasible_mask=feasible, costs=costs)

    def plan_action(self, spawn, action):  # noqa: ANN001
        return None


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


class _ObsStackEnv(gym.Env):
    """Per-frame env stub that emits a (4,C,16,8) observation stack."""

    metadata = {"render_modes": []}

    def __init__(self, *, channels: int) -> None:
        super().__init__()
        self._t = 0
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4, int(channels), GRID_HEIGHT, GRID_WIDTH), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(10)
        self._state_cache = _make_dummy_state(pill_counter=1)
        self._ram_offsets = {}
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False

    def reset(self, *, seed=None, options=None):  # noqa: ANN001
        self._t = 0
        self._state_cache = _make_dummy_state(pill_counter=1)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):  # noqa: ANN001
        self._t += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize(
    ("state_repr", "channels", "mask_channels"),
    [
        ("bitplane_reduced_mask", 10, (6, 7, 8, 9)),
        ("bitplane_bottle_mask", 8, (4, 5, 6, 7)),
    ],
)
def test_feasible_mask_planes_are_injected_into_obs_when_enabled(
    state_repr: str, channels: int, mask_channels: tuple[int, int, int, int]
) -> None:
    prev = ram_specs.get_state_representation()
    ram_specs.set_state_representation(state_repr)
    try:
        assert int(ram_specs.STATE_CHANNELS) == int(channels)
        assert getattr(ram_specs.STATE_IDX, "feasible_mask_channels", None) == mask_channels

        base = _ObsStackEnv(channels=int(ram_specs.STATE_CHANNELS))
        env = DrMarioPlacementEnv(base, planner=_TrivialPlanner(), max_wait_frames=5)

        obs, info = env.reset()
        assert bool(info.get("placements/needs_action", False))
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, int(channels), GRID_HEIGHT, GRID_WIDTH)

        # Trivial planner: only feasible[0,0,0] is True => orientation-0 channel should have a 1 there.
        ch0, ch1, ch2, ch3 = mask_channels
        assert float(obs[:, ch0, 0, 0].min()) == 1.0
        assert float(obs[:, ch0, :, :].sum()) == 4.0  # repeated across the 4-frame stack
        assert float(obs[:, ch1, :, :].sum()) == 0.0
        assert float(obs[:, ch2, :, :].sum()) == 0.0
        assert float(obs[:, ch3, :, :].sum()) == 0.0
    finally:
        ram_specs.set_state_representation(prev)
