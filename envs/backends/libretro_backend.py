"""Native libretro backend implemented via ctypes (no external binding)."""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import register_backend
from .base import EmulatorBackend, NES_BUTTONS


# libretro environment/pixel constants we need
RETRO_ENVIRONMENT_SET_PIXEL_FORMAT = 10
RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY = 9
RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY = 31
RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME = 18
RETRO_ENVIRONMENT_GET_VARIABLE = 15
RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE = 17
RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES = 24
RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION = 52

RETRO_MEMORY_SYSTEM_RAM = 2

RETRO_DEVICE_JOYPAD = 1
RETRO_DEVICE_ID_JOYPAD_MAP = {
    "B": 0,
    "Y": 1,
    "SELECT": 2,
    "START": 3,
    "UP": 4,
    "DOWN": 5,
    "LEFT": 6,
    "RIGHT": 7,
    "A": 8,
    "X": 9,
    "L": 10,
    "R": 11,
}

RETRO_PIXEL_FORMAT_0RGB1555 = 0
RETRO_PIXEL_FORMAT_XRGB8888 = 1
RETRO_PIXEL_FORMAT_RGB565 = 2


class RetroGameInfo(C.Structure):
    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


class RetroSystemInfo(C.Structure):
    _fields_ = [
        ("library_name", C.c_char_p),
        ("library_version", C.c_char_p),
        ("valid_extensions", C.c_char_p),
        ("need_fullpath", C.c_bool),
        ("block_extract", C.c_bool),
    ]


class RetroGameGeometry(C.Structure):
    _fields_ = [
        ("base_width", C.c_uint),
        ("base_height", C.c_uint),
        ("max_width", C.c_uint),
        ("max_height", C.c_uint),
        ("aspect_ratio", C.c_float),
    ]


class RetroSystemTiming(C.Structure):
    _fields_ = [
        ("fps", C.c_double),
        ("sample_rate", C.c_double),
    ]


class RetroSystemAVInfo(C.Structure):
    _fields_ = [
        ("geometry", RetroGameGeometry),
        ("timing", RetroSystemTiming),
    ]


retro_environment_t = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
retro_video_refresh_t = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
retro_audio_sample_t = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
retro_audio_sample_batch_t = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
retro_input_poll_t = C.CFUNCTYPE(None)
retro_input_state_t = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


class _LibretroCore:
    """Minimal libretro driver using ctypes."""

    def __init__(self, core_path: str, system_dir: str, save_dir: str) -> None:
        self.core_path = core_path
        self.system_dir = system_dir
        self.save_dir = save_dir

        self._lib = C.CDLL(core_path)
        self._need_fullpath = True
        self._rom_bytes: Optional[bytes] = None
        self._rom_buffer = None
        self._pixel_format = RETRO_PIXEL_FORMAT_XRGB8888
        self._frame_ready = False
        self._frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self._width = 256
        self._height = 240
        self._joypad_state = [0] * len(NES_BUTTONS)

        # Buffers we must keep alive for libretro
        self._system_dir_buf = C.create_string_buffer(system_dir.encode("utf-8"))
        self._save_dir_buf = C.create_string_buffer(save_dir.encode("utf-8"))

        # Prepare callbacks (keep references on self to avoid GC)
        self._env_cb = retro_environment_t(self._environment_cb)
        self._video_cb = retro_video_refresh_t(self._video_cb_wrapper)
        self._audio_cb = retro_audio_sample_t(self._audio_cb_wrapper)
        self._audio_batch_cb = retro_audio_sample_batch_t(self._audio_batch_cb_wrapper)
        self._input_poll_cb = retro_input_poll_t(self._input_poll_cb_wrapper)
        self._input_state_cb = retro_input_state_t(self._input_state_cb_wrapper)

        # Wire callback setters
        self._lib.retro_set_environment.argtypes = [retro_environment_t]
        self._lib.retro_set_video_refresh.argtypes = [retro_video_refresh_t]
        self._lib.retro_set_audio_sample.argtypes = [retro_audio_sample_t]
        self._lib.retro_set_audio_sample_batch.argtypes = [retro_audio_sample_batch_t]
        self._lib.retro_set_input_poll.argtypes = [retro_input_poll_t]
        self._lib.retro_set_input_state.argtypes = [retro_input_state_t]
        self._lib.retro_set_controller_port_device.argtypes = [C.c_uint, C.c_uint]

        self._lib.retro_set_environment(self._env_cb)
        self._lib.retro_set_video_refresh(self._video_cb)
        self._lib.retro_set_audio_sample(self._audio_cb)
        self._lib.retro_set_audio_sample_batch(self._audio_batch_cb)
        self._lib.retro_set_input_poll(self._input_poll_cb)
        self._lib.retro_set_input_state(self._input_state_cb)

        # Introspect system info
        self._lib.retro_init()
        sys_info = RetroSystemInfo()
        self._lib.retro_get_system_info.argtypes = [C.POINTER(RetroSystemInfo)]
        self._lib.retro_get_system_info(C.byref(sys_info))
        self._need_fullpath = bool(sys_info.need_fullpath)

        av_info = RetroSystemAVInfo()
        self._lib.retro_get_system_av_info.argtypes = [C.POINTER(RetroSystemAVInfo)]
        self._lib.retro_get_system_av_info(C.byref(av_info))
        self._width = int(av_info.geometry.base_width or 256)
        self._height = int(av_info.geometry.base_height or 240)

    def load_game(self, rom_path: str) -> None:
        """Load the ROM (fullpath required)."""
        self._lib.retro_load_game.argtypes = [C.POINTER(RetroGameInfo)]
        self._lib.retro_load_game.restype = C.c_bool
        game = RetroGameInfo()
        if self._need_fullpath:
            game.path = rom_path.encode("utf-8")
            game.data = None
            game.size = 0
        else:
            with open(rom_path, "rb") as f:
                self._rom_bytes = f.read()
            self._rom_buffer = C.create_string_buffer(self._rom_bytes)
            game.path = None
            game.size = len(self._rom_bytes or b"")
            game.data = C.cast(self._rom_buffer, C.c_void_p)  # type: ignore[arg-type]
        ok = self._lib.retro_load_game(C.byref(game))
        if not ok:
            raise RuntimeError("retro_load_game failed")
        self._lib.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD)
        self.reset()

    def reset(self) -> None:
        self._lib.retro_reset()
        self._frame_ready = False

    def step(self) -> None:
        self._frame_ready = False
        self._lib.retro_run()
        if not self._frame_ready:
            # no frame produced -> reuse previous frame
            pass

    def unload(self) -> None:
        try:
            self._lib.retro_unload_game()
        finally:
            self._lib.retro_deinit()

    # Callback implementations -------------------------------------------------

    def _environment_cb(self, cmd: C.c_uint, data: C.c_void_p) -> bool:
        cmd_int = int(cmd)
        # Debug prints (remove or guard in production)
        # print("env cmd", cmd_int)
        if cmd_int == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            ptr = C.cast(data, C.POINTER(C.c_uint))
            if ptr:
                ptr[0] = RETRO_PIXEL_FORMAT_XRGB8888
            self._pixel_format = RETRO_PIXEL_FORMAT_XRGB8888
            return True
        if cmd_int == RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
            ptr = C.cast(data, C.POINTER(C.c_char_p))
            ptr[0] = C.cast(self._system_dir_buf, C.c_char_p)
            return True
        if cmd_int == RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            ptr = C.cast(data, C.POINTER(C.c_char_p))
            ptr[0] = C.cast(self._save_dir_buf, C.c_char_p)
            return True
        if cmd_int == RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME:
            return False
        if cmd_int == RETRO_ENVIRONMENT_GET_VARIABLE:
            return False
        if cmd_int == RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            return False
        if cmd_int == RETRO_ENVIRONMENT_GET_INPUT_DEVICE_CAPABILITIES:
            return False
        if cmd_int == RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
            return False
        return False

    def _video_cb_wrapper(self, data: C.c_void_p, width: int, height: int, pitch: int) -> None:
        if not data:
            return
        w = int(width)
        h = int(height)
        if h <= 0 or w <= 0:
            return
        self._frame_ready = False
        if self._pixel_format == RETRO_PIXEL_FORMAT_XRGB8888:
            row_bytes = pitch
            raw = C.string_at(data, row_bytes * h)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, row_bytes // 4, 4)
            usable_w = min(w, arr.shape[1])
            if usable_w <= 0:
                return
            if self._frame.shape[0] != h or self._frame.shape[1] != usable_w:
                self._frame = np.zeros((h, usable_w, 3), dtype=np.uint8)
            rgb = arr[:, :usable_w, 1:4]
            self._frame[:, :, :] = rgb
        elif self._pixel_format == RETRO_PIXEL_FORMAT_RGB565:
            row_bytes = pitch
            raw = C.string_at(data, row_bytes * h)
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, row_bytes // 2)
            usable_w = min(w, arr.shape[1])
            if usable_w <= 0:
                return
            if self._frame.shape[0] != h or self._frame.shape[1] != usable_w:
                self._frame = np.zeros((h, usable_w, 3), dtype=np.uint8)
            slice_arr = arr[:, :usable_w]
            r = ((slice_arr >> 11) & 0x1F).astype(np.uint8)
            g = ((slice_arr >> 5) & 0x3F).astype(np.uint8)
            b = (slice_arr & 0x1F).astype(np.uint8)
            self._frame[:, :, 0] = (r * 255) // 31
            self._frame[:, :, 1] = (g * 255) // 63
            self._frame[:, :, 2] = (b * 255) // 31
        else:
            raw = C.string_at(data, pitch * h)
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, pitch // 2)
            usable_w = min(w, arr.shape[1])
            if usable_w <= 0:
                return
            if self._frame.shape[0] != h or self._frame.shape[1] != usable_w:
                self._frame = np.zeros((h, usable_w, 3), dtype=np.uint8)
            slice_arr = arr[:, :usable_w]
            r = ((slice_arr >> 10) & 0x1F).astype(np.uint8)
            g = ((slice_arr >> 5) & 0x1F).astype(np.uint8)
            b = (slice_arr & 0x1F).astype(np.uint8)
            self._frame[:, :, 0] = (r * 255) // 31
            self._frame[:, :, 1] = (g * 255) // 31
            self._frame[:, :, 2] = (b * 255) // 31
        self._frame_ready = True

    def _audio_cb_wrapper(self, left: int, right: int) -> None:
        # Audio not needed for now.
        return

    def _audio_batch_cb_wrapper(self, data: C.c_void_p, frames: C.c_size_t) -> C.c_size_t:
        return frames

    def _input_poll_cb_wrapper(self) -> None:
        # Nothing to do; state pulled via _input_state_cb_wrapper.
        return

    def _input_state_cb_wrapper(
        self, port: int, device: int, index: int, key_id: int
    ) -> C.c_int16:
        if port != 0 or device != RETRO_DEVICE_JOYPAD:
            return 0
        for idx, name in enumerate(NES_BUTTONS):
            mapped = RETRO_DEVICE_ID_JOYPAD_MAP.get(name, -1)
            if mapped == key_id:
                return 1 if self._joypad_state[idx] else 0
        return 0

    # Public helpers -----------------------------------------------------------

    def set_buttons(self, buttons: Sequence[int]) -> None:
        self._joypad_state = list(buttons)

    def get_frame(self) -> np.ndarray:
        return np.ascontiguousarray(self._frame)

    def get_ram(self) -> Optional[bytes]:
        self._lib.retro_get_memory_size.argtypes = [C.c_uint]
        self._lib.retro_get_memory_size.restype = C.c_size_t
        self._lib.retro_get_memory_data.argtypes = [C.c_uint]
        self._lib.retro_get_memory_data.restype = C.c_void_p
        size = int(self._lib.retro_get_memory_size(RETRO_MEMORY_SYSTEM_RAM))
        ptr = self._lib.retro_get_memory_data(RETRO_MEMORY_SYSTEM_RAM)
        if not ptr or size == 0:
            return None
        return C.string_at(ptr, size)


class LibretroBackend(EmulatorBackend):
    """Backend driving a libretro core via ctypes."""

    def __init__(
        self,
        *,
        core_path: Optional[str] = None,
        rom_path: Optional[str] = None,
        system_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        resolved_core = Path(core_path or os.environ.get("DRMARIO_CORE_PATH", "")).expanduser()
        if not resolved_core.is_file():
            raise FileNotFoundError(
                f"Libretro core not found. Set DRMARIO_CORE_PATH or pass core_path. "
                f"Tried: {resolved_core}"
            )
        resolved_rom = Path(rom_path or os.environ.get("DRMARIO_ROM_PATH", "")).expanduser()
        if not resolved_rom.is_file():
            raise FileNotFoundError(
                "NES ROM not found. Set DRMARIO_ROM_PATH or pass rom_path pointing to your Dr. Mario ROM."
            )
        self._core_path = str(resolved_core)
        self._rom_path = str(resolved_rom)
        self._system_dir = system_dir or resolved_core.parent.as_posix()
        self._save_dir = save_dir or resolved_core.parent.as_posix()

        self._core: Optional[_LibretroCore] = None
        self._frame_cache = np.zeros((240, 256, 3), dtype=np.uint8)
        self._ram_cache: Optional[np.ndarray] = None

    def load(self) -> None:
        self._core = _LibretroCore(self._core_path, self._system_dir, self._save_dir)
        self._core.load_game(self._rom_path)
        self._frame_cache = self._core.get_frame()
        self._ram_cache = None

    def reset(self) -> None:
        if self._core is None:
            raise RuntimeError("Backend not loaded.")
        self._core.reset()
        self._frame_cache = self._core.get_frame()
        self._ram_cache = None

    def step(self, buttons: Sequence[int], repeat: int = 1) -> None:
        if self._core is None:
            raise RuntimeError("Backend not loaded.")
        if len(buttons) != len(NES_BUTTONS):
            raise ValueError(f"Expected {len(NES_BUTTONS)} buttons, got {len(buttons)}.")
        self._core.set_buttons(buttons)
        for _ in range(max(1, int(repeat))):
            self._core.step()
        self._frame_cache = self._core.get_frame()
        self._ram_cache = None

    def get_frame(self) -> np.ndarray:
        return np.ascontiguousarray(self._frame_cache)

    def get_ram(self) -> Optional[np.ndarray]:
        if self._core is None:
            return None
        if self._ram_cache is None:
            ram_bytes = self._core.get_ram()
            if ram_bytes is not None:
                self._ram_cache = np.frombuffer(ram_bytes, dtype=np.uint8).copy()
        return self._ram_cache

    def serialize(self) -> Optional[bytes]:
        # Implement if needed (requires retro_serialize bindings).
        return None

    def deserialize(self, blob: bytes) -> None:
        # Implement if needed (requires retro_unserialize bindings).
        return None

    def close(self) -> None:
        if self._core is not None:
            try:
                self._core.unload()
            except Exception:
                pass
            self._core = None


register_backend("libretro", LibretroBackend)
