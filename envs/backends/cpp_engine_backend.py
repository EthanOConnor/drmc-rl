"""C++ Dr. Mario engine backend (shared memory + subprocess).

This backend provides a `EmulatorBackend`-compatible interface for the
high-performance C++ game-logic reimplementation under `game_engine/`.

Key design goal: expose a **2 KB NES RAM view** so the rest of the Python stack
(RAM→state mapping, placement planner, curriculum patching) can run unchanged.

Protocol summary (see `game_engine/GameState.h`):
  - Driver writes `state.buttons` (NES button bitmask, 0x01=R ... 0x80=A)
  - Manual stepping: driver sets `control_flags|=0x04` for one frame; engine clears it
  - Reset: driver can pre-seed `rng_state` + set `rng_override!=0`, then set `control_flags|=0x10`
  - Exit: driver sets `control_flags|=0x08`
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from envs.backends import register_backend
from envs.backends.base import EmulatorBackend, NES_BUTTONS


_BTN_MASK: dict[str, int] = {
    "RIGHT": 0x01,
    "LEFT": 0x02,
    "DOWN": 0x04,
    "UP": 0x08,
    "START": 0x10,
    "SELECT": 0x20,
    "B": 0x40,
    "A": 0x80,
}


def _buttons_vector_to_mask(buttons: Sequence[int]) -> int:
    """Convert an 8-length 0/1 vector in `NES_BUTTONS` order to engine bitmask."""

    if len(buttons) != len(NES_BUTTONS):
        raise ValueError(f"Expected {len(NES_BUTTONS)} buttons, got {len(buttons)}")
    mask = 0
    for name, pressed in zip(NES_BUTTONS, buttons):
        if int(pressed):
            bit = _BTN_MASK.get(str(name), 0)
            mask |= int(bit)
    return int(mask) & 0xFF


class CppEngineBackend(EmulatorBackend):
    """Backend that drives the C++ engine via a file-backed shared-memory map."""

    _RUN_MODE_FRAMES = 1
    _RUN_MODE_UNTIL_DECISION = 2

    _RUN_REASON_DONE = 1
    _RUN_REASON_DECISION = 2
    _RUN_REASON_TERMINAL = 3
    _RUN_REASON_TIMEOUT = 4

    def __init__(
        self,
        *,
        engine_path: Optional[str] = None,
        level: int = 0,
        speed_setting: int = 1,
        demo: bool = False,
        build_if_missing: bool = True,
        step_timeout_sec: float = 0.5,
    ) -> None:
        self.engine_path = (
            (Path(engine_path) if engine_path else Path("game_engine/drmario_engine"))
            .expanduser()
            .resolve()
        )
        self.level = int(level)
        self.speed_setting = int(speed_setting)
        self.demo = bool(demo)
        self.build_if_missing = bool(build_if_missing)
        self.step_timeout_sec = float(step_timeout_sec)

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._shm_file: Optional[Path] = None
        self._mm = None
        self._state = None

        self._frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self._ram = np.zeros((0x800,), dtype=np.uint8)
        self._rng_seed_bytes_next_reset: Optional[tuple[int, int]] = None
        self._intro_wait_frames_next_reset: Optional[int] = None
        self._intro_frame_counter_lo_next_reset: Optional[int] = None

    # --------------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self.engine_path.is_file():
            return
        if not self.build_if_missing:
            raise FileNotFoundError(f"Engine binary not found at {self.engine_path}")
        self._build_engine()

    def _build_engine(self) -> None:
        engine_dir = self.engine_path.parent
        if not (engine_dir / "Makefile").is_file():
            raise FileNotFoundError(f"Expected Makefile under {engine_dir}")
        subprocess.check_call(["make"], cwd=str(engine_dir))
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"Engine build completed but binary missing at {self.engine_path}")

    def reset(self) -> None:
        """Reset the engine to a fresh level state.

        If the engine process is not running, start it in manual-step mode and
        perform the initial gated reset after attaching to shared memory.
        """

        self.load()
        if self._proc is None or self._proc.poll() is not None:
            self._start_engine_process()
            self._initial_gate_reset()
        else:
            self._request_reset()
        self._refresh_ram_from_state()

    def close(self) -> None:
        self._shutdown()

    # --------------------------------------------------------------------- stepping

    def step(self, buttons: Sequence[int], repeat: int = 1) -> None:
        if self._proc is None or self._state is None:
            raise RuntimeError("Engine backend not initialized; call load/reset first.")
        repeats = max(1, int(repeat))
        raw_mask = _buttons_vector_to_mask(buttons)
        if repeats > 1:
            self._run_request(
                mode=self._RUN_MODE_FRAMES,
                max_frames=repeats,
                buttons_mask=raw_mask,
                last_spawn_id=None,
            )
        else:
            self._state.buttons = raw_mask
            self._step_once()
            self._refresh_ram_from_state()

    def run_until_decision(self, *, last_spawn_id: Optional[int], max_frames: int) -> dict[str, int]:
        """Fast-forward until the next controllable pill spawn (decision point).

        This avoids per-frame Python handshakes by letting the engine run many
        frames internally and stopping when:
          - nextAction == pillFalling, and
          - pill_counter != last_spawn_id (unless last_spawn_id is None), and
          - gameplay mode is active.

        Returns a dict of counters for telemetry/reward aggregation.
        """

        return self._run_request(
            mode=self._RUN_MODE_UNTIL_DECISION,
            max_frames=int(max(1, max_frames)),
            buttons_mask=0,
            last_spawn_id=last_spawn_id,
        )

    def run_frames_mask(self, *, buttons_mask: int, frames: int) -> dict[str, int]:
        """Run exactly `frames` frames with a constant raw NES button bitmask."""

        return self._run_request(
            mode=self._RUN_MODE_FRAMES,
            max_frames=int(max(0, frames)),
            buttons_mask=int(buttons_mask) & 0xFF,
            last_spawn_id=None,
        )

    def _step_once(self) -> None:
        if self._proc is None or self._state is None:
            raise RuntimeError("Engine backend not initialized.")

        expected = int(self._state.frame_count) + 1
        self._state.control_flags |= 0x04
        t0 = time.perf_counter()
        spin_until = t0 + 0.00005  # 50µs: avoids oversleeping when the engine is fast
        sleep_s = 0.0
        polls = 0
        while int(self._state.frame_count) < expected:
            now = time.perf_counter()
            if now - t0 > self.step_timeout_sec:
                raise TimeoutError("Timed out waiting for engine step.")

            polls += 1
            if (polls & 0x7F) == 0:  # every 128 iterations
                if self._proc.poll() is not None:
                    out, err = self._proc.communicate(timeout=1)
                    raise RuntimeError(
                        "Engine exited unexpectedly during step.\n"
                        f"stdout:\n{out.decode(errors='replace')}\n"
                        f"stderr:\n{err.decode(errors='replace')}\n"
                    )

            if now < spin_until:
                continue

            # Back off gently; `time.sleep` releases the GIL and helps under
            # heavy multi-env loads (prevents spin contention with engine procs).
            sleep_s = 0.00005 if sleep_s <= 0.0 else min(0.002, sleep_s * 1.35)
            time.sleep(sleep_s)

    def _run_request(
        self,
        *,
        mode: int,
        max_frames: int,
        buttons_mask: int,
        last_spawn_id: Optional[int],
    ) -> dict[str, int]:
        if self._proc is None or self._state is None:
            raise RuntimeError("Engine backend not initialized.")

        mode_i = int(mode)
        max_frames_i = int(max(0, max_frames))
        buttons_i = int(buttons_mask) & 0xFF
        last_spawn_u8 = 0xFF if last_spawn_id is None else (int(last_spawn_id) & 0xFF)

        # Bump request id (uint32 wrap).
        req = (int(getattr(self._state, "run_request_id")) + 1) & 0xFFFFFFFF

        self._state.run_mode = mode_i
        self._state.run_frames = max_frames_i
        self._state.run_buttons = buttons_i
        self._state.run_last_spawn_id = last_spawn_u8
        self._state.run_request_id = req

        t0 = time.perf_counter()
        # Conservative timeout: allow longer for large fast-forwards.
        timeout = max(self.step_timeout_sec, float(max_frames_i) / 2000.0)
        sleep_s = 0.0
        while int(getattr(self._state, "run_ack_id")) != req:
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError("Timed out waiting for engine run request.")
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited unexpectedly during run.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            # Back off gently; `time.sleep` releases the GIL. We intentionally
            # start with a slightly larger sleep to reduce scheduler thrash when
            # running many envs in parallel.
            sleep_s = 0.00005 if sleep_s <= 0.0 else min(0.005, sleep_s * 1.35)
            time.sleep(sleep_s)

        # Populate synthetic RAM view after the run completes.
        self._refresh_ram_from_state()

        return {
            "frames": int(getattr(self._state, "run_frames_executed")),
            "reason": int(getattr(self._state, "run_reason")) & 0xFF,
            "tiles_cleared_total": int(getattr(self._state, "run_tiles_cleared_total")),
            "tiles_cleared_virus": int(getattr(self._state, "run_tiles_cleared_virus")),
            "tiles_cleared_nonvirus": int(getattr(self._state, "run_tiles_cleared_nonvirus")),
        }

    # --------------------------------------------------------------------- outputs

    def get_frame(self) -> np.ndarray:
        return self._frame

    def get_ram(self) -> Optional[np.ndarray]:
        return self._ram

    def serialize(self) -> Optional[bytes]:
        return None

    def deserialize(self, blob: bytes) -> None:
        raise NotImplementedError("CppEngineBackend does not support savestates yet.")

    # --------------------------------------------------------------------- extra (non-protocol) helpers

    def write_ram(self, addr: int, values: Sequence[int]) -> None:
        """Best-effort RAM patching for env utilities.

        DrMarioRetroEnv uses this hook for RNG randomization and synthetic virus
        curricula (patching the bottle buffer + virus counter). We implement the
        subset of the NES RAM map that is meaningful for the engine.
        """

        if self._state is None:
            raise RuntimeError("Engine backend not initialized.")
        start = int(addr)
        for i, v in enumerate(values):
            a = start + i
            if 0 <= a < self._ram.shape[0]:
                self._ram[a] = int(v) & 0xFF
            # RNG ($0017,$0018)
            if a == 0x0017:
                self._state.rng_state[0] = int(v) & 0xFF
                self._state.rng_override = 1
            elif a == 0x0018:
                self._state.rng_state[1] = int(v) & 0xFF
                self._state.rng_override = 1
            # Bottle buffer ($0400..$047F)
            elif 0x0400 <= a < 0x0480:
                self._state.board[a - 0x0400] = int(v) & 0xFF
            # Virus counter ($0324, BCD)
            elif a == 0x0324:
                self._state.viruses_remaining = int(v) & 0xFF

    def set_next_reset_rng_seed_bytes(self, seed_bytes: Optional[Sequence[int]]) -> None:
        """Configure RNG seed for the *next* engine reset (2 bytes)."""

        if seed_bytes is None:
            self._rng_seed_bytes_next_reset = None
            return
        if len(seed_bytes) != 2:
            raise ValueError("Expected exactly 2 rng seed bytes.")
        self._rng_seed_bytes_next_reset = (int(seed_bytes[0]) & 0xFF, int(seed_bytes[1]) & 0xFF)

    def set_next_reset_intro_wait_frames(self, wait_frames: Optional[int]) -> None:
        """Configure the *next* engine reset to start mid level-intro delay.

        This mirrors the ROM's `waitFrames` ($0051) value during `waitFor_A_frames`
        in `levelIntro`. It exists primarily for emulator parity harnesses.
        """

        if wait_frames is None:
            self._intro_wait_frames_next_reset = None
            return
        self._intro_wait_frames_next_reset = int(wait_frames) & 0xFF

    def set_next_reset_intro_frame_counter_lo(self, frame_counter_lo: Optional[int]) -> None:
        """Configure the *next* engine reset to match the ROM's NMI `frameCounter` low byte.

        Dr. Mario uses the low bit of `frameCounter` for soft-drop timing; matching
        it at a parity sync point avoids off-by-one drift.
        """

        if frame_counter_lo is None:
            self._intro_frame_counter_lo_next_reset = None
            return
        self._intro_frame_counter_lo_next_reset = int(frame_counter_lo) & 0xFF

    # --------------------------------------------------------------------- process + shared memory

    def _start_engine_process(self) -> None:
        self._shutdown()

        from envs.backends.cpp_engine_shm import SHM_SIZE, open_shared_memory_file

        # File-backed shm lets us run multiple engine instances without global
        # `shm_open` name collisions.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tmp.close()
        shm_file = Path(tmp.name)
        shm_file.write_bytes(b"\x00" * int(SHM_SIZE))
        self._shm_file = shm_file

        env = os.environ.copy()
        env["DRMARIO_SHM_FILE"] = str(shm_file)
        env["DRMARIO_LEVEL"] = str(max(0, min(int(self.level), 25)))
        env["DRMARIO_SPEED"] = str(max(0, min(int(self.speed_setting), 2)))
        env["DRMARIO_MODE"] = "demo" if self.demo else "play"

        argv = [str(self.engine_path), "--wait-start", "--manual-step"]
        if self.demo:
            argv.insert(1, "--demo")

        self._proc = subprocess.Popen(
            argv,
            cwd=str(self.engine_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Give the process time to mmap the file.
        time.sleep(0.05)
        self._mm, self._state = open_shared_memory_file(shm_file)

    def _initial_gate_reset(self) -> None:
        if self._state is None:
            raise RuntimeError("Engine shared state not mapped.")

        # Wait until the engine has completed its startup memset/arg parsing.
        # If we set the start gate too early, it can be wiped and the engine
        # will block forever in the --wait-start loop.
        t0 = time.perf_counter()
        while (int(self._state.control_flags) & 0x02) == 0:
            if self._proc is None:
                raise RuntimeError("Engine process missing during initialization.")
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited unexpectedly during initialization.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            if time.perf_counter() - t0 > 1.0:
                raise TimeoutError("Timed out waiting for engine initialization.")
            time.sleep(0.001)

        # If a seed was configured, set it before releasing the start gate so
        # `GameLogic::reset()` will use it during level setup.
        if self._rng_seed_bytes_next_reset is not None:
            a, b = self._rng_seed_bytes_next_reset
            self._state.rng_state[0] = int(a) & 0xFF
            self._state.rng_state[1] = int(b) & 0xFF
            self._state.rng_override = 1
            self._rng_seed_bytes_next_reset = None
        if self._intro_wait_frames_next_reset is not None:
            self._state.reset_wait_frames = int(self._intro_wait_frames_next_reset) & 0xFF
            self._intro_wait_frames_next_reset = None
        if self._intro_frame_counter_lo_next_reset is not None:
            self._state.reset_framecounter_lo_plus1 = (int(self._intro_frame_counter_lo_next_reset) & 0xFF) + 1
            self._intro_frame_counter_lo_next_reset = None

        # Release start gate so engine performs its initial reset.
        self._state.control_flags |= 0x01
        # Wait until reset has populated the board/virus count.
        t0 = time.perf_counter()
        while int(self._state.viruses_remaining) == 0:
            if self._proc is None:
                raise RuntimeError("Engine process missing during gate reset.")
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited unexpectedly during initial reset.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            if time.perf_counter() - t0 > 1.0:
                break
            time.sleep(0.001)

    def _request_reset(self) -> None:
        if self._state is None:
            raise RuntimeError("Engine shared state not mapped.")

        if self._rng_seed_bytes_next_reset is not None:
            a, b = self._rng_seed_bytes_next_reset
            self._state.rng_state[0] = int(a) & 0xFF
            self._state.rng_state[1] = int(b) & 0xFF
            self._state.rng_override = 1
            self._rng_seed_bytes_next_reset = None
        if self._intro_wait_frames_next_reset is not None:
            self._state.reset_wait_frames = int(self._intro_wait_frames_next_reset) & 0xFF
            self._intro_wait_frames_next_reset = None
        if self._intro_frame_counter_lo_next_reset is not None:
            self._state.reset_framecounter_lo_plus1 = (int(self._intro_frame_counter_lo_next_reset) & 0xFF) + 1
            self._intro_frame_counter_lo_next_reset = None

        self._state.control_flags |= 0x10
        t0 = time.perf_counter()
        while (int(self._state.control_flags) & 0x10) != 0:
            if self._proc is None:
                raise RuntimeError("Engine process missing during reset.")
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited unexpectedly during reset.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            if time.perf_counter() - t0 > 1.0:
                raise TimeoutError("Timed out waiting for engine reset.")
            time.sleep(0.0001)

    def _refresh_ram_from_state(self) -> None:
        """Populate the synthetic 0x800-byte NES RAM view from shared state."""

        if self._state is None:
            return

        ram = self._ram
        ram.fill(0)

        # RNG state ($0017,$0018)
        ram[0x0017] = int(self._state.rng_state[0]) & 0xFF
        ram[0x0018] = int(self._state.rng_state[1]) & 0xFF

        # NMI frameCounter ($0043): low byte.
        ram[0x0043] = int(self._state.frame_count) & 0xFF

        # Gameplay flags
        ram[0x0046] = int(self._state.mode) & 0xFF
        # Level intro countdown (`waitFrames`) in ZP ($0051).
        ram[0x0051] = int(self._state.wait_frames) & 0xFF
        ram[0x0053] = int(self._state.ending_active) & 0xFF
        ram[0x0055] = int(self._state.stage_clear) & 0xFF

        # Player count (1P)
        ram[0x0727] = 0x01

        # Buttons held mirror (used by planner tooling).
        ram[0x00F7] = int(self._state.buttons_held) & 0xFF

        # Core per-player state machine + falling pose mirrors (ZP + main RAM).
        ram[0x0097] = int(self._state.next_action) & 0xFF

        # Falling capsule pose/counters (ZP mirrors)
        ram[0x0085] = int(self._state.falling_pill_col) & 0xFF
        ram[0x0086] = int(self._state.falling_pill_row) & 0xFF
        ram[0x00A5] = int(self._state.falling_pill_orient) & 0xFF
        ram[0x0092] = int(self._state.speed_counter) & 0xFF
        ram[0x0093] = int(self._state.hor_velocity) & 0xFF
        ram[0x008A] = int(self._state.speed_ups) & 0xFF
        ram[0x008B] = int(self._state.speed_setting) & 0xFF

        # Falling capsule colors (ZP mirrors).
        ram[0x0081] = int(self._state.falling_pill_color_l) & 0xFF
        ram[0x0082] = int(self._state.falling_pill_color_r) & 0xFF

        # Main RAM falling capsule registers.
        ram[0x0301] = int(self._state.falling_pill_color_l) & 0xFF
        ram[0x0302] = int(self._state.falling_pill_color_r) & 0xFF
        ram[0x0305] = int(self._state.falling_pill_col) & 0xFF
        ram[0x0306] = int(self._state.falling_pill_row) & 0xFF
        ram[0x0307] = int(self._state.lock_counter) & 0xFF
        ram[0x0309] = int(self._state.level_fail) & 0xFF
        # p1_pillsCounter_decimal/hundreds (BCD) at $0310/$0311.
        bcd_total = int(self._state.pill_counter_total) & 0xFFFF
        ram[0x0310] = bcd_total & 0xFF
        ram[0x0311] = (bcd_total >> 8) & 0xFF
        # p1_nextAction at $0317 (also mirrored in ZP at $0097).
        if int(self._state.level_fail) & 0xFF:
            # Retail behavior (see `mainLoop_level`): once a player fails, the
            # fixed-RAM `p1_nextAction` is set to `nextAction_doNothing` (0x04)
            # even though the ZP `currentP_nextAction` is left unchanged.
            ram[0x0317] = 0x04
        else:
            ram[0x0317] = int(self._state.next_action) & 0xFF
        # p1_speedCounter/horVelocity at $0312/$0313.
        ram[0x0312] = int(self._state.speed_counter) & 0xFF
        ram[0x0313] = int(self._state.hor_velocity) & 0xFF
        ram[0x0316] = int(self._state.level) & 0xFF
        # p1_speedUps/speedSetting at $030A/$030B (also mirrored in ZP at $008A/$008B).
        ram[0x030A] = int(self._state.speed_ups) & 0xFF
        ram[0x030B] = int(self._state.speed_setting) & 0xFF

        # Preview capsule registers.
        ram[0x031A] = int(self._state.preview_pill_color_l) & 0xFF
        ram[0x031B] = int(self._state.preview_pill_color_r) & 0xFF
        ram[0x0322] = int(self._state.preview_pill_rotation) & 0xFF
        ram[0x0323] = int(self._state.preview_pill_size) & 0xFF
        ram[0x0324] = int(self._state.viruses_remaining) & 0xFF
        ram[0x0325] = int(self._state.falling_pill_orient) & 0xFF
        ram[0x0326] = int(self._state.falling_pill_size) & 0xFF
        # p1_pillsCounter (reserve index) at $0327.
        ram[0x0327] = int(self._state.pill_counter) & 0xFF

        # Bottle buffer (P1 board at $0400).
        ram[0x0400 : 0x0400 + 128] = np.ctypeslib.as_array(self._state.board)

    def _shutdown(self) -> None:
        # Best-effort graceful stop.
        if self._proc is not None and self._proc.poll() is None and self._state is not None:
            try:
                self._state.control_flags |= 0x08
            except Exception:
                pass

        if self._state is not None:
            self._state = None
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._shm_file is not None:
            try:
                self._shm_file.unlink()
            except Exception:
                pass
            self._shm_file = None


register_backend("cpp-engine", CppEngineBackend)
