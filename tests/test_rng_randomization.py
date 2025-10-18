from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.retro.drmario_env import DrMarioRetroEnv


class _StubBackend:
    def __init__(self) -> None:
        self.frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self.ram = np.zeros(0x10000, dtype=np.uint8)
        self.events: list[tuple] = []

    def reset(self) -> None:
        self.events.append(("reset",))

    def get_frame(self) -> np.ndarray:
        return self.frame

    def get_ram(self) -> np.ndarray:
        return self.ram

    def write_ram(self, addr: int, values) -> None:
        data = np.asarray(list(values), dtype=np.uint8)
        self.events.append(("write", addr, data.copy()))
        self.ram[addr : addr + data.size] = data

    def step(self, buttons, repeat: int = 1) -> None:
        self.events.append(("step", list(buttons), int(repeat)))

    def get_state(self):
        raise NotImplementedError


def _make_env(auto_start: bool) -> tuple[DrMarioRetroEnv, _StubBackend]:
    env = DrMarioRetroEnv(obs_mode="pixel", backend="mock", auto_start=auto_start)
    stub = _StubBackend()
    env._backend = stub  # type: ignore[assignment]
    env._using_backend = True
    env.auto_start = auto_start
    return env, stub


def test_randomize_rng_defers_until_start_sequence():
    env, stub = _make_env(auto_start=True)
    options = {
        "randomize_rng": True,
        "start_presses": 1,
        "start_hold_frames": 1,
        "start_gap_frames": 1,
        "start_settle_frames": 0,
        "start_wait_viruses": 0,
        "start_level_taps": 0,
    }

    _, info = env.reset(options=options)

    write_indices = [idx for idx, event in enumerate(stub.events) if event[0] == "write"]
    assert write_indices, "RNG randomization never triggered"
    first_write = write_indices[0]
    assert any(event[0] == "step" for event in stub.events[:first_write]), (
        "RNG randomization should occur after start inputs are issued"
    )
    written = stub.events[first_write]
    addr, values = written[1], written[2]
    assert addr == 0x0017
    assert values.shape[0] == 2
    assert tuple(int(v) for v in values) == info.get("rng_seed")


def test_randomize_rng_immediate_when_auto_start_disabled():
    env, stub = _make_env(auto_start=False)
    override = [0x01, 0xAB]
    options = {"randomize_rng": True, "rng_seed_bytes": override}

    _, info = env.reset(options=options)

    write_events = [event for event in stub.events if event[0] == "write"]
    assert len(write_events) == 1
    assert not any(event[0] == "step" for event in stub.events)
    addr, values = write_events[0][1], write_events[0][2]
    assert addr == 0x0017
    assert tuple(int(v) for v in values) == tuple(override)
    assert info.get("rng_seed") == tuple(override)

