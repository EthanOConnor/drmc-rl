from __future__ import annotations

import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.retro.drmario_env import DrMarioRetroEnv, RewardConfig


def _make_env(*, pair: float = 1.0, triplet: float = 2.0) -> DrMarioRetroEnv:
    cfg = RewardConfig(
        virus_adjacency_pair_bonus=float(pair),
        virus_adjacency_triplet_bonus=float(triplet),
        # Keep other shaping terms off for isolation (doesn't affect the method under test).
        adjacency_pair_bonus=0.0,
        adjacency_triplet_bonus=0.0,
    )
    return DrMarioRetroEnv(obs_mode="state", backend="mock", reward_config=cfg, auto_start=False)


def test_virus_adjacency_pair_bonus_awarded() -> None:
    prev_repr = ram_specs.get_state_representation()
    try:
        ram_specs.set_state_representation("extended")
        env = _make_env(pair=1.25, triplet=9.0)

        C = ram_specs.STATE_CHANNELS
        prev = np.zeros((C, 16, 8), dtype=np.float32)
        nxt = np.zeros((C, 16, 8), dtype=np.float32)

        # Red virus at (10, 3).
        virus_red = int(ram_specs.STATE_IDX.virus_color_channels[0])
        prev[virus_red, 10, 3] = 1.0
        nxt[virus_red, 10, 3] = 1.0

        # New red pill cell adjacent at (10, 4).
        pill_red = int(ram_specs.STATE_IDX.static_color_channels[0])
        nxt[pill_red, 10, 4] = 1.0

        bonus = env._compute_virus_adjacency_bonus(prev, nxt)
        assert bonus == 1.25
    finally:
        ram_specs.set_state_representation(prev_repr)


def test_virus_adjacency_triplet_bonus_awarded_when_run_includes_virus() -> None:
    prev_repr = ram_specs.get_state_representation()
    try:
        ram_specs.set_state_representation("extended")
        env = _make_env(pair=1.0, triplet=2.5)

        C = ram_specs.STATE_CHANNELS
        prev = np.zeros((C, 16, 8), dtype=np.float32)
        nxt = np.zeros((C, 16, 8), dtype=np.float32)

        virus_red = int(ram_specs.STATE_IDX.virus_color_channels[0])
        pill_red = int(ram_specs.STATE_IDX.static_color_channels[0])

        # Virus at (10,3) and existing pill at (10,5).
        prev[virus_red, 10, 3] = 1.0
        prev[pill_red, 10, 5] = 1.0
        nxt[virus_red, 10, 3] = 1.0
        nxt[pill_red, 10, 5] = 1.0

        # New pill bridges to make a length-3 run: (10,3)-(10,4)-(10,5).
        nxt[pill_red, 10, 4] = 1.0

        bonus = env._compute_virus_adjacency_bonus(prev, nxt)
        assert bonus == 2.5
    finally:
        ram_specs.set_state_representation(prev_repr)


def test_virus_adjacency_bonus_not_awarded_without_virus_in_run() -> None:
    prev_repr = ram_specs.get_state_representation()
    try:
        ram_specs.set_state_representation("extended")
        env = _make_env(pair=1.0, triplet=2.0)

        C = ram_specs.STATE_CHANNELS
        prev = np.zeros((C, 16, 8), dtype=np.float32)
        nxt = np.zeros((C, 16, 8), dtype=np.float32)

        pill_red = int(ram_specs.STATE_IDX.static_color_channels[0])

        # Existing pill at (10,3), new pill at (10,4) forms a pair but with no virus.
        prev[pill_red, 10, 3] = 1.0
        nxt[pill_red, 10, 3] = 1.0
        nxt[pill_red, 10, 4] = 1.0

        bonus = env._compute_virus_adjacency_bonus(prev, nxt)
        assert bonus == 0.0
    finally:
        ram_specs.set_state_representation(prev_repr)

