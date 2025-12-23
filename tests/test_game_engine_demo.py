import re
import subprocess
import time
from pathlib import Path
import os

import ctypes
import pytest

from game_engine.engine_shm import DrMarioStatePy, SHM_SIZE, open_shared_memory, repo_root
from tools.game_transcript import compare_transcripts, load_json
from tools.record_demo import record_cpp_demo


ENGINE_DIR = repo_root() / "game_engine"
DEMO_DATA_PATH = repo_root() / "dr-mario-disassembly" / "data" / "drmario_data_demo_field_pills.asm"
NES_DEMO_TRANSCRIPT_PATH = repo_root() / "data" / "nes_demo.json"

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
    # demo_pills is a 128-byte ROM table. The disassembly splits it into the
    # first 45 bytes (reachable in a normal demo playback) and an UNUSED
    # continuation (still present in ROM and addressable by the &0x7F index).
    data: list[int] = []
    seen = False
    for line in DEMO_DATA_PATH.read_text().splitlines():
        if line.startswith("demo_pills:"):
            seen = True
            continue
        if not seen:
            continue
        if line.strip() == "endif":
            break
        for match in re.findall(r"\$([0-9A-Fa-f]{2})", line):
            data.append(int(match, 16))
            if len(data) == 128:
                return data
    raise ValueError("Failed to parse demo_pills (expected 128 bytes)")


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
    # Wait for the post-gate `reset()` to finish populating the demo board.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if state.frame_count != 0 and state.board[0] == 0xFF:
            break
        time.sleep(0.001)
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
        try:
            mm.close()
        except BufferError:
            # pytest can keep references alive for assertion introspection; force
            # a collection so the mmap can be closed cleanly.
            import gc

            gc.collect()
            mm.close()


def test_demo_trace_matches_nes_ground_truth():
    expected = load_json(NES_DEMO_TRANSCRIPT_PATH)
    actual = record_cpp_demo(
        engine_path=ENGINE_DIR / "drmario_engine",
        max_frames=8000,  # should stop naturally at demo end (~5701)
        verbose=False,
    )

    divergences = compare_transcripts(expected, actual, stop_on_first=True)
    if divergences:
        d = divergences[0]
        pytest.fail(
            f"Demo transcript diverged at frame {d.frame} ({d.field}): {d.message}\n"
            f"expected={d.expected}\n"
            f"actual={d.actual}"
        )
