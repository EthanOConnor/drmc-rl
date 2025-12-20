from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from envs.backends.base import NES_BUTTONS
from envs.backends.cpp_engine_backend import CppEngineBackend


ENGINE_PATH = Path("game_engine/drmario_engine")


@pytest.mark.skipif(
    not ENGINE_PATH.is_file() or not os.access(ENGINE_PATH, os.X_OK),
    reason="C++ engine binary not available",
)
def test_cpp_backend_multienv_isolation():
    backends: list[CppEngineBackend] = []
    seeds = [(0x12, 0x34), (0x56, 0x78), (0x9A, 0xBC)]
    try:
        for seed in seeds:
            backend = CppEngineBackend(engine_path=str(ENGINE_PATH), build_if_missing=False)
            backend.set_next_reset_rng_seed_bytes(seed)
            backend.reset()
            backends.append(backend)

        shm_paths = [b._shm_file for b in backends]
        assert all(path is not None for path in shm_paths)
        assert len({str(path) for path in shm_paths}) == len(shm_paths)

        seed_bytes = {(int(b.get_ram()[0x0017]), int(b.get_ram()[0x0018])) for b in backends}
        assert len(seed_bytes) == len(backends)

        buttons = [0 for _ in NES_BUTTONS]
        backends[0].step(buttons, repeat=1)
        ram0 = np.asarray(backends[0].get_ram()).copy()
        ram1 = np.asarray(backends[1].get_ram()).copy()
        assert int(ram0[0x0043]) != int(ram1[0x0043])
    finally:
        for backend in backends:
            try:
                backend.close()
            except Exception:
                pass
