import re
import subprocess
import time
from pathlib import Path
import os

import ctypes
import pytest

from game_engine.engine_shm import DrMarioStatePy, SHM_SIZE, open_shared_memory, repo_root


ENGINE_DIR = repo_root() / "game_engine"
DEMO_DATA_PATH = repo_root() / "dr-mario-disassembly" / "data" / "drmario_data_demo_field_pills.asm"

COLOR_COMBO_LEFT = [0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02]
COLOR_COMBO_RIGHT = [0x00, 0x01, 0x02, 0x00, 0x01, 0x02, 0x00, 0x01, 0x02]


def parse_block(label: str, length: int) -> list[int]:
    data = []
    seen = False
    for line in DEMO_DATA_PATH.read_text().splitlines():
        if label in line:
            seen = True
            continue
        if not seen:
            continue
        if line.startswith(";;") or "demo_pills_UNUSED" in line:
            break
        for match in re.findall(r"\$([0-9A-Fa-f]{2})", line):
            data.append(int(match, 16))
            if len(data) == length:
                return data
    raise ValueError(f"Failed to parse {label}")


def parse_demo_field() -> list[int]:
    return parse_block("demo_field:", 128)


def parse_demo_pills() -> list[int]:
    return parse_block("demo_pills:", 45)


@pytest.fixture(scope="module")
def demo_data():
    return {
        "field": parse_demo_field(),
        "pills": parse_demo_pills(),
    }


@pytest.fixture
def engine_proc(tmp_path, monkeypatch):
    shm_file = tmp_path / "drmario_shm.bin"
    monkeypatch.setenv("DRMARIO_SHM_FILE", str(shm_file))

    env = os.environ.copy()
    env["DRMARIO_SHM_FILE"] = str(shm_file)
    proc = subprocess.Popen(
        ["./drmario_engine", "--demo", "--wait-start"],
        cwd=ENGINE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    time.sleep(0.02)  # allow shm to come up and reset
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_struct_size_matches_c():
    assert ctypes.sizeof(DrMarioStatePy) == SHM_SIZE


def test_demo_reset_matches_disassembly(engine_proc, demo_data):
    mm, state = open_shared_memory()
    state.control_flags = 1  # release wait-start gate
    state.buttons = 0
    # Ensure we read right after start
    while state.frame_count == 0:
        pass
    try:
        # After reset, engine should be in demo mode with demo field copied.
        assert list(state.board) == demo_data["field"]

        expected_pills = demo_data["pills"]
        # First falling pill uses pill 0, preview uses pill 1 (reserve advanced once)
        pill0 = expected_pills[0]
        pill1 = expected_pills[1]

        assert state.falling_pill_color_l == COLOR_COMBO_LEFT[pill0]
        assert state.falling_pill_color_r == COLOR_COMBO_RIGHT[pill0]

        assert state.preview_pill_color_l == COLOR_COMBO_LEFT[pill1]
        assert state.preview_pill_color_r == COLOR_COMBO_RIGHT[pill1]
    finally:
        del state
        mm.close()
