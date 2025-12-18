from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.specs.timeouts import get_level_timeout
import envs.specs.ram_to_state as ram_specs
from envs.retro.state_viz import state_to_rgb
from envs.reward_shaping import PotentialShaper
from envs.backends import make_backend
from envs.backends.base import NES_BUTTONS
from envs.state_core import DrMarioState, build_state

A_BUTTON_INDEX = NES_BUTTONS.index("A")
B_BUTTON_INDEX = NES_BUTTONS.index("B")
from envs.retro.action_adapters import discrete10_to_buttons
from time import perf_counter
import os

def _env_flag(name: str) -> bool:
    """Parse a boolean-like environment variable.

    Recognizes: 1/true/yes/on (case-insensitive) as True; 0/false/no/off as False.
    Any other value or unset -> False.
    """
    try:
        val = os.environ.get(name)
        if val is None:
            return False
        v = str(val).strip().lower()
        return v in ("1", "true", "yes", "on")
    except Exception:
        return False


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    ROTATE_A = 4
    ROTATE_B = 5
    LEFT_HOLD = 6
    RIGHT_HOLD = 7
    DOWN_HOLD = 8
    BOTH_ROT = 9


@dataclass
class RewardConfig:
    pill_place_base: float = 1.2
    pill_place_growth: float = 0.0008
    virus_clear_bonus: float = 300.0
    non_virus_clear_bonus: float = 50.0
    terminal_clear_bonus: float = 150.0
    topout_penalty: float = -50.0
    time_bonus_topout_per_60_frames: float = 2.0
    time_penalty_clear_per_60_frames: float = 3.0
    adjacency_pair_bonus: float = 10.0
    adjacency_triplet_bonus: float = 25.0
    column_height_penalty: float = 5.0
    action_penalty_scale: float = 0.25
    placement_height_threshold: float = 3.0
    placement_height_penalty_multiplier: float = -2.0
    punish_high_placements: bool = True



def _default_seed_registry() -> Path:
    return Path(__file__).with_suffix("").parent / "seeds" / "registry.json"


class DrMarioRetroEnv(gym.Env):
    """Single-agent Dr. Mario environment with pluggable emulator backends.

    Observation modes:
      - pixel: 128x128 RGB, frame-stack 4, normalized to [0,1]
      - state: structured 16x8 board tensor (channel count depends on the
        selected representation via ``state_repr`` or
        :func:`envs.specs.ram_to_state.set_state_representation`), frame-stack 4.
        See docs/STATE_OBS_AND_RAM_MAPPING.md for channel spec and mapping;
        RAM offsets are configured via envs/specs/ram_offsets.json (for this
        ROM revision) or override with env var `DRMARIO_RAM_OFFSETS`.

    Notes:
      - Defaults to the libretro backend using a direct ctypes binding and a provided NES core.
        Stable-Retro remains available as an alternate backend, and the env falls back
        to deterministic mock dynamics if no backend can be constructed.
      - Automatic START/LEFT sequences can be enabled (default) to skip menus and enforce
        level-0 restarts; top-outs are detected via virus-count reset and incur a penalty.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        obs_mode: str = "pixel",
        frame_skip: int = 1,
        include_risk_tau: bool = True,
        risk_tau: float = 1.0,
        reward_config: Optional[RewardConfig] = None,
        reward_config_path: Optional[Union[str, Path]] = None,
        seed_registry: Optional[Path] = None,
        core_path: Optional[Path] = None,
        rom_path: Optional[Path] = None,
        backend: Optional[str] = None,
        backend_kwargs: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        level: int = 0,
        auto_start: bool = True,
        # Evaluator-based shaping controls
        use_potential_shaping: bool = False,
        evaluator: Optional[Any] = None,
        potential_gamma: float = 0.997,
        potential_kappa: float = 250.0,
        learner_discount: Optional[float] = None,
        state_viz_interval: Optional[int] = None,
        state_repr: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert obs_mode in {"pixel", "state"}
        assert frame_skip in {1, 2}
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.include_risk_tau = include_risk_tau
        self.default_risk_tau = float(risk_tau)
        self._reward_config_path: Optional[Path] = None
        self.reward_cfg, self._reward_cfg_descriptions = self._resolve_reward_config(
            reward_config, reward_config_path
        )
        self._reward_config_mtime: Optional[float] = None
        if self._reward_config_path and self._reward_config_path.is_file():
            try:
                self._reward_config_mtime = self._reward_config_path.stat().st_mtime
            except OSError:
                self._reward_config_mtime = None
        self.core_path = Path(core_path).expanduser() if core_path is not None else None
        self.rom_path = Path(rom_path).expanduser() if rom_path is not None else None
        self.render_mode = render_mode
        self.level = level
        self.backend_name = (backend or os.environ.get("DRMARIO_BACKEND", "libretro")).lower()
        self._backend_kwargs = dict(backend_kwargs or {})
        if self.core_path is not None and "core_path" not in self._backend_kwargs:
            self._backend_kwargs["core_path"] = str(self.core_path)
        if self.rom_path is not None and "rom_path" not in self._backend_kwargs:
            self._backend_kwargs["rom_path"] = str(self.rom_path)
        self.auto_start = auto_start
        self._first_boot = True

        self._t = 0  # frames elapsed
        self._viruses_remaining = 8  # placeholder; varies by level
        self._viruses_initial = self._viruses_remaining
        self._rng = np.random.RandomState(0)
        self._t_max = get_level_timeout(self.level)
        self._fps = 60.0
        if state_viz_interval is None:
            env_interval = os.environ.get("DRMARIO_STATE_VIZ_INTERVAL")
            if env_interval:
                try:
                    state_viz_interval = int(env_interval)
                except ValueError:
                    state_viz_interval = None
        self._state_viz_interval = max(1, int(state_viz_interval)) if state_viz_interval else None
        self._state_viz_last_t = -1
        # Potential shaping setup
        self._learner_discount = None if learner_discount is None else float(learner_discount)
        self._use_shaping = use_potential_shaping and evaluator is not None
        self._shaper: Optional[PotentialShaper] = (
            PotentialShaper(
                evaluator,
                potential_gamma=potential_gamma,
                kappa=potential_kappa,
                learner_discount=self._learner_discount,
            )
            if self._use_shaping
            else None
        )
        if state_repr is not None:
            ram_specs.set_state_representation(state_repr)
        self.state_repr = ram_specs.get_state_representation()
        self._state_prev: Optional[np.ndarray] = None
        self._state_cache: Optional[DrMarioState] = None
        # Load RAM offsets for state-mode mapping
        self._ram_offsets = self._load_ram_offsets()

        # Observation space
        if obs_mode == "pixel":
            # 4 x 128 x 128 x 3 float32 in [0,1]
            obs_space = spaces.Box(low=0.0, high=1.0, shape=(4, 128, 128, 3), dtype=np.float32)
        else:
            # 4 x (C x 16 x 8)
            obs_space = spaces.Box(
                low=0.0, high=1.0, shape=(4, ram_specs.STATE_CHANNELS, 16, 8), dtype=np.float32
            )

        if include_risk_tau:
            self.observation_space = spaces.Dict(
                {
                    "obs": obs_space,
                    "risk_tau": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                }
            )
        else:
            self.observation_space = obs_space

        self.action_space = spaces.Discrete(10)

        # Attempt to bind a backend (libretro by default)
        self._backend = None
        self._using_backend = False
        if self.backend_name != "mock":
            try:
                self._backend = make_backend(self.backend_name, **self._backend_kwargs)
                self._backend.load()
                self._using_backend = True
            except Exception as exc:
                warnings.warn(
                    f"Failed to initialize backend '{self.backend_name}': {exc}. "
                    "Falling back to mock dynamics."
                )
                self._backend = None
                self._using_backend = False
        self._buttons_layout: Sequence[str] = NES_BUTTONS
        self._button_index = {name: idx for idx, name in enumerate(self._buttons_layout)}
        self._last_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self._pix_stack: Optional[np.ndarray] = None  # (4,128,128,3)
        self._ram_cache: Optional[np.ndarray] = None
        # Hold state (used when retro is active)
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False
        self._prev_move_dir = "none"
        self._in_game = False
        self._last_terminal: Optional[str] = None
        self._prev_terminal: Optional[str] = None
        self._frames_without_active_pill = 0
        self._topout_inactive_threshold = 240
        self._game_mode_val: Optional[int] = None
        self._gameplay_active: Optional[bool] = None
        self._stage_clear_flag: Optional[bool] = None
        self._ending_active: Optional[bool] = None
        self._player_count: Optional[int] = None
        self._pill_spawn_counter: Optional[int] = None
        self._prev_pill_count = 0
        self._frames_until_drop_val: Optional[int] = None
        self._pending_rng_randomize = False
        self._pending_rng_override: Optional[Sequence[int]] = None
        self._last_rng_seed_bytes: Optional[Tuple[int, ...]] = None
        self._backend_reset_done = False
        self._prev_height_penalty = 0.0

    # ------------------------------------------------------------------
    # Canonical termination detection (state/RAM mode)
    # ------------------------------------------------------------------

    def _canonical_ram_outcome(self) -> Tuple[Optional[bool], Optional[bool]]:
        """Return (fail, clear) booleans from canonical RAM flags when available.

        For emulator-backed state modes, we rely on the game's own flags:
          - Failure: p1_levelFailFlag at $0309 (non-zero when player topped out)
          - Success: p1_virusLeft at $0324 equals 0, or whoWon at $0055 equals 0x01

        If RAM is unavailable or not in state mode, returns (None, None).
        """
        if self.obs_mode != "state" or not self._using_backend or self._state_cache is None:
            return None, None

        ram_arr = self._state_cache.ram.arr
        try:
            p1_fail = int(ram_arr[0x0309]) if ram_arr.shape[0] > 0x0309 else None
        except Exception:
            p1_fail = None
        try:
            p1_virus_left = int(ram_arr[0x0324]) if ram_arr.shape[0] > 0x0324 else None
        except Exception:
            p1_virus_left = None
        try:
            who_won = int(ram_arr[0x0055]) if ram_arr.shape[0] > 0x0055 else None
        except Exception:
            who_won = None

        fail_flag: Optional[bool] = None
        clear_flag: Optional[bool] = None

        if p1_fail is not None:
            fail_flag = bool(p1_fail != 0)
        # Success if either the zero-page winner flag is set to P1, or viruses reach zero.
        if who_won is not None and who_won == 0x01:
            clear_flag = True
        elif p1_virus_left is not None:
            clear_flag = bool(p1_virus_left == 0)

        return fail_flag, clear_flag
        self._elapsed_frames = 0

    def _resolve_reward_config(
        self,
        reward_config: Optional[RewardConfig],
        reward_config_path: Optional[Union[str, Path]],
    ) -> Tuple[RewardConfig, Dict[str, str]]:
        if reward_config is not None:
            self._reward_config_path = None
            return reward_config, {}

        candidates: list[Path] = []
        if reward_config_path is not None:
            candidates.append(Path(reward_config_path).expanduser())
        env_override = os.environ.get("DRMARIO_REWARD_CONFIG")
        if env_override:
            candidates.append(Path(env_override).expanduser())
        candidates.append(Path(__file__).with_suffix("").parent.parent / "specs" / "reward_config.json")

        for candidate in candidates:
            try:
                if not candidate or not candidate.is_file():
                    continue
                with candidate.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                warnings.warn(f"Failed to load reward config from {candidate}: {exc}")
                continue
            cfg_obj, descriptions = self._reward_config_from_payload(payload)
            self._reward_config_path = candidate
            return cfg_obj, descriptions

        self._reward_config_path = None
        return RewardConfig(), {}

    @staticmethod
    def _reward_config_from_payload(payload: Dict[str, Any]) -> Tuple[RewardConfig, Dict[str, str]]:
        if not isinstance(payload, dict):
            return RewardConfig(), {}

        values: Dict[str, Any] = {}
        descriptions: Dict[str, str] = {}
        for key, meta in payload.items():
            if isinstance(meta, dict):
                if "value" in meta:
                    values[key] = meta["value"]
                if "description" in meta:
                    descriptions[key] = str(meta["description"])
            else:
                values[key] = meta

        cfg_kwargs: Dict[str, Any] = {}
        for field_name in RewardConfig.__dataclass_fields__:
            if field_name in values:
                cfg_kwargs[field_name] = values[field_name]

        return RewardConfig(**cfg_kwargs), descriptions

    def reload_reward_config(self) -> bool:
        if self._reward_config_path is None:
            return False
        try:
            mtime = self._reward_config_path.stat().st_mtime
        except OSError:
            return False
        if self._reward_config_mtime is not None and mtime <= self._reward_config_mtime:
            return False
        try:
            with self._reward_config_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            cfg_obj, descriptions = self._reward_config_from_payload(payload)
            self.reward_cfg = cfg_obj
            self._reward_cfg_descriptions = descriptions
            self._reward_config_mtime = mtime
            return True
        except Exception as exc:
            warnings.warn(f"Failed to reload reward config from {self._reward_config_path}: {exc}")
            return False

    def _resize_rgb(self, rgb: np.ndarray, out_hw=(128, 128)) -> np.ndarray:
        # Nearest-neighbor resize without external deps
        h, w = rgb.shape[:2]
        oh, ow = out_hw
        ys = (np.linspace(0, h - 1, oh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, ow)).astype(np.int32)
        return rgb[ys][:, xs]

    def _update_pixel_stack(self, frame: np.ndarray) -> None:
        small = self._resize_rgb(frame, (128, 128)).astype(np.float32) / 255.0
        if self._pix_stack is None:
            self._pix_stack = np.stack([small for _ in range(4)], axis=0)
        else:
            self._pix_stack = np.concatenate([self._pix_stack[1:], small[None, ...]], axis=0)

    def _read_ram_array(self, refresh: bool = True) -> Optional[np.ndarray]:
        if not self._using_backend or self._backend is None:
            return None
        if refresh:
            try:
                ram = self._backend.get_ram()
            except Exception:
                ram = None
            if ram is not None:
                self._ram_cache = np.asarray(ram, dtype=np.uint8).reshape(-1)
        return self._ram_cache

    def _raw_ram_bytes(self) -> Optional[bytes]:
        if not self._using_backend or self._backend is None:
            return None
        try:
            ram_arr = self._read_ram_array(refresh=False)
            if ram_arr is not None:
                return ram_arr.tobytes()
            if hasattr(self._backend, "get_ram"):
                raw = self._backend.get_ram()
            else:
                raw = None
            if raw is None:
                return None
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
            try:
                return np.asarray(raw, dtype=np.uint8).reshape(-1).tobytes()
            except Exception:
                return None
        except Exception:
            return None

    def _read_ram_value(self, addr: int) -> Optional[int]:
        ram_arr = self._read_ram_array(refresh=False)
        if ram_arr is None:
            return None
        if 0 <= addr < ram_arr.shape[0]:
            return int(ram_arr[addr])
        return None

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str):
            try:
                base = 16 if value.lower().startswith("0x") else 10
                return int(value, base)
            except ValueError:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _read_offset_value(self, group: str, key: str = "addr") -> Optional[int]:
        offsets = self._ram_offsets.get(group)
        if not offsets:
            return None
        addr_val = offsets.get(key)
        addr = self._parse_int(addr_val)
        if addr is None:
            return None
        return self._read_ram_value(addr)

    def _read_gameplay_flag(self) -> Tuple[Optional[int], Optional[bool]]:
        if self._state_cache is None:
            return None, None
        return self._state_cache.ram_vals.mode, self._state_cache.ram_vals.gameplay_active

    def _read_stage_clear_flag(self) -> Optional[bool]:
        if self._state_cache is None:
            return None
        return self._state_cache.ram_vals.stage_clear

    def _read_ending_flag(self) -> Optional[bool]:
        if self._state_cache is None:
            return None
        return self._state_cache.ram_vals.ending_active

    def _read_player_count(self) -> Optional[int]:
        if self._state_cache is None:
            return None
        return self._state_cache.ram_vals.player_count

    def _read_pill_counter(self) -> Optional[int]:
        if self._state_cache is None:
            return None
        return self._state_cache.ram_vals.pill_counter

    def _read_frames_until_drop(self) -> Optional[int]:
        if self._state_cache is None:
            return None
        return self._state_cache.ram_vals.gravity_counter

    def _randomize_rng_state(
        self, override: Optional[Sequence[int]] = None
    ) -> Optional[Tuple[int, ...]]:
        if not self._using_backend or self._backend is None:
            return None
        offsets = self._ram_offsets.get("rng", {})
        state_addrs = offsets.get("state_addrs")
        if not state_addrs:
            return None
        writer = getattr(self._backend, "write_ram", None)
        if writer is None:
            return None
        addresses: list[int] = []
        for addr_hex in state_addrs:
            try:
                addresses.append(int(addr_hex, 16))
            except (TypeError, ValueError):
                continue
        if not addresses:
            return None
        addresses.sort()
        if override is not None:
            data = np.asarray(list(override), dtype=np.uint8)
        else:
            rng = np.random.default_rng()
            data = rng.integers(0, 256, size=len(addresses), dtype=np.uint8)
        data_list = [int(v) for v in data.tolist()]
        chunk_start: Optional[int] = None
        chunk_values: list[int] = []
        for addr, value in zip(addresses, data_list):
            if chunk_start is None:
                chunk_start = addr
                chunk_values = [value]
            elif addr == chunk_start + len(chunk_values):
                chunk_values.append(value)
            else:
                try:
                    writer(chunk_start, chunk_values)
                except Exception:
                    return None
                chunk_start = addr
                chunk_values = [value]
        if chunk_start is not None and chunk_values:
            try:
                writer(chunk_start, chunk_values)
            except Exception:
                return None
        self._ram_cache = None
        self._read_ram_array(refresh=True)
        return tuple(data_list)

    def _apply_pending_rng_randomization(self) -> None:
        if not self._pending_rng_randomize or not self._using_backend or self._backend is None:
            self._pending_rng_randomize = False
            self._pending_rng_override = None
            return
        override = self._pending_rng_override
        try:
            result = self._randomize_rng_state(override)
        except Exception as exc:
            warnings.warn(f"RNG randomization failed: {exc}")
            result = None
        self._last_rng_seed_bytes = tuple(int(v) & 0xFF for v in result) if result else None
        self._pending_rng_randomize = False
        self._pending_rng_override = None

    def _update_active_pill_tracker(self, v_now: Optional[int]) -> None:
        if not self._in_game or self._state_cache is None:
            self._frames_without_active_pill = 0
            return
        if v_now is not None and v_now == 0:
            self._frames_without_active_pill = 0
            return
        size_hex = self._ram_offsets.get("falling_pill", {}).get("size_addr")
        if not size_hex:
            self._frames_without_active_pill = 0
            return
        try:
            size_addr = int(size_hex, 16)
        except (TypeError, ValueError):
            self._frames_without_active_pill = 0
            return

        ram_arr = self._state_cache.ram.arr
        size_val = ram_arr[size_addr] if 0 <= size_addr < ram_arr.shape[0] else None

        if size_val is None or size_val >= 2:
            self._frames_without_active_pill = 0
            return

        lock_val = self._state_cache.ram_vals.lock_counter
        if lock_val is not None and lock_val > 0:
            self._frames_without_active_pill = 0
            return
        self._frames_without_active_pill += self.frame_skip

    def _explicit_topout_flag(self) -> Optional[bool]:
        if self._state_cache is None:
            return None
        ram_arr = self._state_cache.ram.arr

        offsets = self._ram_offsets.get("topout")
        if not offsets:
            return None
        flag_hex = offsets.get("flag_addr")
        if flag_hex:
            try:
                flag_addr = int(flag_hex, 16)
            except (TypeError, ValueError):
                flag_addr = None
            if flag_addr is not None:
                flag_val = ram_arr[flag_addr] if 0 <= flag_addr < ram_arr.shape[0] else None
                if flag_val is not None:
                    return bool(flag_val)
        sentinel = offsets.get("sentinel_values")
        value_hex = offsets.get("value_addr")
        if value_hex and sentinel:
            try:
                val_addr = int(value_hex, 16)
            except (TypeError, ValueError):
                val_addr = None
            if val_addr is not None:
                val = ram_arr[val_addr] if 0 <= val_addr < ram_arr.shape[0] else None
                if val is not None:
                    values: list[int] = []
                    try:
                        for s in sentinel:
                            values.append(int(s, 16) if isinstance(s, str) else int(s))
                    except Exception:
                        values = []
                    if int(val) in values:
                        return True
        return None

    def _detect_topout(self, v_now: Optional[int]) -> bool:
        explicit = self._explicit_topout_flag()
        if explicit is True:
            self._frames_without_active_pill = 0
            return True
        if explicit is False:
            self._frames_without_active_pill = 0
            return False
        self._update_active_pill_tracker(v_now)
        if self._frames_without_active_pill >= self._topout_inactive_threshold:
            return True
        if self.obs_mode == "state":
            if self._state_cache is not None:
                planes = self._state_cache.calc.planes
                if self._detect_state_game_over_pattern(planes):
                    self._frames_without_active_pill = 0
                    return True
        return False

    @staticmethod
    def _detect_state_game_over_pattern(state_stack: np.ndarray) -> bool:
        try:
            stack = np.asarray(state_stack)
        except Exception:
            return False
        if stack.ndim == 4:
            if stack.shape[0] == 0:
                return False
            latest = stack[-1]
        elif stack.ndim == 3:
            latest = stack
        else:
            return False
        if latest.shape[-2:] != (16, 8):
            return False
        static_colors = ram_specs.get_static_color_planes(latest)
        if static_colors.shape[0] < 3:
            return False
        static_masks = static_colors > 0.5
        static_red = static_masks[0]
        static_yellow = static_masks[1]
        static_blue = static_masks[2]
        occupancy = static_red | static_yellow | static_blue
        H, W = occupancy.shape
        if W < 8:
            return False
        for top in range(0, H - 4):
            row_y = static_yellow[top]
            row_r = static_red[top]
            if row_y.sum() != 6:
                continue
            if row_y[0] or row_y[7]:
                continue
            if not row_y[1:7].all():
                continue
            if not row_r[7] or row_r[:7].any():
                continue
            if static_blue[top, 0]:
                continue
            consecutive_blue = 0
            for r in range(top + 1, H):
                if not static_blue[r, 0]:
                    break
                consecutive_blue += 1
            if consecutive_blue < 4:
                continue
            row_occ = occupancy[top]
            if row_occ.sum() != 7:
                continue
            return True
        return False

    def _compute_adjacency_bonus(self, prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
        static_prev = ram_specs.get_static_color_planes(prev_frame)
        static_next = ram_specs.get_static_color_planes(next_frame)
        if static_prev.shape[0] < 3 or static_next.shape[0] < 3:
            return 0.0
        static_prev_mask = static_prev > 0.5
        static_next_mask = static_next > 0.5
        new_cells = static_next_mask & ~static_prev_mask
        total_bonus = 0.0

        for color_idx in range(3):
            coords = np.argwhere(new_cells[color_idx])
            if coords.size == 0:
                continue
            mask_prev = static_prev_mask[color_idx]
            mask_new = static_next_mask[color_idx]
            pair_awarded = False
            triplet_awarded = False

            for r, c in coords:
                # Horizontal runs
                left_prev = 0
                cc = c - 1
                while cc >= 0 and mask_prev[r, cc]:
                    left_prev += 1
                    cc -= 1
                right_prev = 0
                cc = c + 1
                while cc < mask_prev.shape[1] and mask_prev[r, cc]:
                    right_prev += 1
                    cc += 1

                left_new = 0
                cc = c - 1
                while cc >= 0 and mask_new[r, cc]:
                    left_new += 1
                    cc -= 1
                right_new = 0
                cc = c + 1
                while cc < mask_new.shape[1] and mask_new[r, cc]:
                    right_new += 1
                    cc += 1

                run_prev_best = max(left_prev, right_prev)
                run_new_total = left_new + 1 + right_new

                if run_prev_best < 3 and run_new_total >= 3:
                    triplet_awarded = True
                elif run_prev_best < 2 and run_new_total >= 2:
                    pair_awarded = True

                # Vertical runs
                up_prev = 0
                rr = r - 1
                while rr >= 0 and mask_prev[rr, c]:
                    up_prev += 1
                    rr -= 1
                down_prev = 0
                rr = r + 1
                while rr < mask_prev.shape[0] and mask_prev[rr, c]:
                    down_prev += 1
                    rr += 1

                up_new = 0
                rr = r - 1
                while rr >= 0 and mask_new[rr, c]:
                    up_new += 1
                    rr -= 1
                down_new = 0
                rr = r + 1
                while rr < mask_new.shape[0] and mask_new[rr, c]:
                    down_new += 1
                    rr += 1

                run_prev_best_v = max(up_prev, down_prev)
                run_new_total_v = up_new + 1 + down_new

                if run_prev_best_v < 3 and run_new_total_v >= 3:
                    triplet_awarded = True
                elif run_prev_best_v < 2 and run_new_total_v >= 2:
                    pair_awarded = True

            if triplet_awarded:
                total_bonus += self.reward_cfg.adjacency_triplet_bonus
            elif pair_awarded:
                total_bonus += self.reward_cfg.adjacency_pair_bonus

        return total_bonus

    def _extract_virus_count(self) -> Optional[int]:
        # Prefer the raw RAM counter when available (works during reset/startup
        # sequences where `_state_cache` may be stale).
        raw = self._read_offset_value("viruses", "remaining_addr")
        if raw is not None:
            return int(raw)
        if self._state_cache is None:
            return None
        return self._state_cache.calc.viruses_remaining

    def _decode_preview_pill(self) -> Optional[Dict[str, int]]:
        if self._state_cache is None:
            return None
        return self._state_cache.calc.preview

    def _augment_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        augmented = dict(info)
        preview = self._decode_preview_pill()
        if preview is not None:
            augmented["preview_pill"] = preview

        augmented["placement_height_threshold"] = float(
            self.reward_cfg.placement_height_threshold
        )
        augmented["placement_height_penalty_multiplier"] = float(
            self.reward_cfg.placement_height_penalty_multiplier
        )
        augmented["punish_high_placements"] = bool(self.reward_cfg.punish_high_placements)

        state_stack = getattr(self, "_state_stack", None)
        if state_stack is not None:
            latest = np.asarray(state_stack[-1])
            if ram_specs.STATE_IDX.level is not None:
                level_norm = float(np.clip(ram_specs.get_level_value(latest), 0.0, 1.0))
                level_val = int(round(level_norm * 20.0))
                level_val = int(min(max(level_val, 0), 20))
                augmented["level_state"] = level_val
                augmented["level"] = level_val
            if ram_specs.STATE_IDX.lock is not None:
                lock_norm = float(np.clip(ram_specs.get_lock_value(latest), 0.0, 1.0))
                lock_raw = int(round(lock_norm * 255.0))
                augmented["lock_counter"] = lock_raw
                augmented["is_locked"] = bool(lock_raw > 0)

        return augmented

    def _button_vector(self, names: Optional[Sequence[str]] = None) -> Sequence[int]:
        vec = [0 for _ in range(len(self._buttons_layout))]
        if names:
            for name in names:
                idx = self._button_index.get(name)
                if idx is not None:
                    vec[idx] = 1
        return vec

    def _backend_step_buttons(self, buttons: Sequence[int], repeat: int = 1) -> None:
        if not self._using_backend or self._backend is None:
            return
        self._backend.step(list(buttons), repeat=max(1, int(repeat)))
        self._last_frame = self._backend.get_frame()
        self._update_pixel_stack(self._last_frame)
        self._read_ram_array(refresh=True)

    def backend_reset(self) -> None:
        """Reset the underlying backend and mark the next env.reset as already handled."""

        if not self._using_backend or self._backend is None:
            self._backend_reset_done = False
            return

        self._backend.reset()
        self._backend_reset_done = True

    def _run_start_sequence(
        self,
        presses: int,
        options: Optional[Dict[str, Any]],
        *,
        from_topout: bool = False,
        apply_rng: bool = False,
    ) -> None:
        if presses <= 0 or not self._using_backend or self._backend is None:
            return
        opts = options or {}
        hold_frames = int(opts.get("start_hold_frames", 6))
        gap_frames = int(opts.get("start_gap_frames", 40))
        settle_frames = int(opts.get("start_settle_frames", 180))
        wait_frames = int(opts.get("start_wait_viruses", 600))
        level_taps = int(opts.get("start_level_taps", 12))
        noop = self._button_vector(None)
        start_vec = self._button_vector(["START"])
        left_vec = self._button_vector(["LEFT"])

        def press_start() -> None:
            self._backend_step_buttons(start_vec, repeat=hold_frames)
            self._backend_step_buttons(noop, repeat=gap_frames)

        rng_applied = False

        def apply_rng_now() -> None:
            nonlocal rng_applied
            if apply_rng and not rng_applied:
                self._apply_pending_rng_randomization()
                rng_applied = True

        presses = int(presses)
        if presses <= 0:
            return

        def align_level() -> None:
            if level_taps > 0:
                for _ in range(level_taps):
                    self._backend_step_buttons(left_vec, repeat=1)
                    self._backend_step_buttons(noop, repeat=1)

        if from_topout:
            presses = max(2, presses)
            press_start()  # leave game-over screen
            align_level()
            apply_rng_now()
            for _ in range(presses - 1):
                press_start()
        elif presses == 1:
            align_level()
            apply_rng_now()
            press_start()
        else:
            press_start()
            align_level()
            apply_rng_now()
            for _ in range(presses - 1):
                press_start()

        apply_rng_now()

        if settle_frames > 0:
            self._backend_step_buttons(noop, repeat=settle_frames)
        if wait_frames > 0:
            for _ in range(wait_frames):
                vcount = self._extract_virus_count()
                if vcount is not None and vcount > 0:
                    break
                self._backend_step_buttons(noop, repeat=1)
        self._hold_left = self._hold_right = self._hold_down = False
        vcount = self._extract_virus_count()
        if vcount is not None and vcount > 0:
            self._in_game = True


    def _maybe_auto_start(self, options: Optional[Dict[str, Any]]) -> None:
        if not self.auto_start:
            return
        prev_terminal = getattr(self, "_prev_terminal", None)
        presses_opt = (options or {}).get("start_presses")
        if presses_opt is not None:
            presses = int(presses_opt)
        else:
            if self._first_boot:
                presses = 3
            elif prev_terminal == "topout":
                presses = 2
            else:
                presses = 1
        apply_rng = bool(self._pending_rng_randomize)
        if presses > 0:
            self._run_start_sequence(
                presses,
                options,
                from_topout=(prev_terminal == "topout"),
                apply_rng=apply_rng,
            )
        if self._pending_rng_randomize:
            self._apply_pending_rng_randomization()
        if self._using_backend and self._backend is not None:
            self._read_ram_array(refresh=True)
            vcount = self._extract_virus_count()
            if vcount is not None and vcount > 0:
                self._in_game = True
        self._first_boot = False
        self._prev_terminal = None

    def _load_ram_offsets(self) -> Dict[str, Any]:
        import json, os
        candidates = [
            os.environ.get("DRMARIO_RAM_OFFSETS"),
            str(Path(__file__).with_suffix("").parent.parent / "specs" / "ram_offsets.json"),
            str(Path(__file__).with_suffix("").parent.parent / "specs" / "ram_offsets_example.json"),
        ]
        for p in candidates:
            if not p:
                continue
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:
                continue
        return {}

    def _mock_obs(self) -> np.ndarray:
        if self.obs_mode == "pixel":
            # Make a simple moving gradient to simulate pixel stacks
            base = np.linspace(0, 1, 128, dtype=np.float32)
            frame = np.outer(base, np.ones(128, dtype=np.float32))
            rgb = np.stack([frame, np.roll(frame, self._t % 5, axis=0), frame.T], axis=-1)
            rgb = np.clip(rgb, 0.0, 1.0)
            stacked = np.stack([rgb for _ in range(4)], axis=0)
            return stacked
        else:
            # Random one-hot-ish channels for 16x8 board over 4 frames
            C = ram_specs.STATE_CHANNELS
            board = np.zeros((4, C, 16, 8), dtype=np.float32)
            for f in range(4):
                idx = self._rng.randint(0, C, size=(16, 8))
                board[f, idx, np.arange(16)[:, None], np.arange(8)[None, :]] = 1.0
            # Broadcast scalars could be encoded in reserved channels later
            return board

    def _pixel_obs(self) -> np.ndarray:
        if self._using_backend:
            if self._pix_stack is None:
                self._update_pixel_stack(self._last_frame)
            return self._pix_stack
        return self._mock_obs()

    def _state_obs(self) -> np.ndarray:
        # Always return the per-step cached stack so we don't remap RAM
        if self._state_cache is None:
            # Fallback for very early calls; could also raise
            return np.zeros((4, ram_specs.STATE_CHANNELS, 16, 8), dtype=np.float32)  # or your configured shape
        return self._state_cache.stack4

    def _observe(self, risk_tau: Optional[float] = None) -> Any:
        if self.obs_mode == "pixel":
            core = self._pixel_obs()
        else:
            core = self._state_obs()
        tau = self.default_risk_tau if risk_tau is None else float(risk_tau)
        if self.include_risk_tau:
            return {"obs": core, "risk_tau": np.asarray(tau, dtype=np.float32)}
        return core

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._t = 0
        self._viruses_remaining = 8
        self._t_max = get_level_timeout(self.level)
        self._state_viz_last_t = -1
        self._pix_stack = None
        self._state_stack = None
        self._state_cache = None
        self._in_game = False
        self._prev_terminal = getattr(self, "_last_terminal", None)
        self._last_terminal = None
        self._frames_without_active_pill = 0
        self._game_mode_val = None
        self._gameplay_active = None
        self._stage_clear_flag = None
        self._ending_active = None
        self._player_count = None
        self._pill_spawn_counter = None
        self._prev_pill_count = 0
        self._frames_until_drop_val = None
        self._pending_rng_randomize = False
        self._pending_rng_override = None
        self._last_rng_seed_bytes = None
        self._prev_height_penalty = 0.0
        backend_reset_pending = getattr(self, "_backend_reset_done", False)
        self._elapsed_frames = 0
        self._prev_move_dir = "none"
        if self._using_backend and self._backend is not None:
            try:
                if not backend_reset_pending:
                    self._backend.reset()
                self._last_frame = self._backend.get_frame()
                self._update_pixel_stack(self._last_frame)
                self._read_ram_array(refresh=True)

                ram_arr = self._read_ram_array(refresh=False)
                ram_bytes = ram_arr.tobytes()

                # Freeze/update canonical state
                self._state_cache = build_state(
                    ram_bytes=ram_bytes,
                    ram_offsets=self._ram_offsets,
                    prev_stack4=None,
                    t=self._t,
                    elapsed_frames=self._elapsed_frames,
                    frame_skip=self.frame_skip,
                    last_terminal=self._last_terminal if hasattr(self, "_last_terminal") else None,
                )
                self._state_stack = self._state_cache.stack4

            except Exception as exc:
                warnings.warn(f"Backend reset failed: {exc}. Falling back to mock dynamics.")
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
                self._using_backend = False
            finally:
                self._backend_reset_done = False
        else:
            self._backend_reset_done = False
        opts = dict(options or {})
        if opts.get("randomize_rng") and self._using_backend and self._backend is not None:
            try:
                seed_bytes = opts.get("rng_seed_bytes")
                override_seq: Optional[Sequence[int]]
                if isinstance(seed_bytes, Sequence) and not isinstance(
                    seed_bytes, (str, bytes, bytearray)
                ):
                    override_seq = [int(v) & 0xFF for v in seed_bytes]
                else:
                    override_seq = None
                self._pending_rng_randomize = True
                self._pending_rng_override = override_seq
                if not self.auto_start:
                    self._apply_pending_rng_randomization()
            except Exception as exc:
                warnings.warn(f"RNG randomization setup failed: {exc}")
                self._pending_rng_randomize = False
                self._pending_rng_override = None
        # Optional frame offset to influence ROM RNG by letting frames elapse before we start
        frame_offset = int(opts.get("frame_offset", 0))
        if frame_offset > 0 and self._using_backend and self._backend is not None:
            noop = self._button_vector(None)
            try:
                self._backend_step_buttons(noop, repeat=frame_offset)
            except Exception as exc:
                warnings.warn(f"Backend frame_offset failed: {exc}")
        if self._using_backend and self._backend is not None:
            try:
                self._maybe_auto_start(opts)
            except Exception as exc:
                warnings.warn(f"Auto-start sequence failed: {exc}")
            # Auto-start advances emulator frames but `_backend_step_buttons`
            # intentionally does not rebuild `_state_cache` (to keep step() fast).
            # Rebuild canonical state once here so reset() returns a consistent
            # observation/info snapshot.
            try:
                ram_arr = self._read_ram_array(refresh=True)
                if ram_arr is not None:
                    ram_bytes = ram_arr.tobytes()
                    self._state_cache = build_state(
                        ram_bytes=ram_bytes,
                        ram_offsets=self._ram_offsets,
                        prev_stack4=None,
                        t=self._t,
                        elapsed_frames=self._elapsed_frames,
                        frame_skip=self.frame_skip,
                        last_terminal=self._last_terminal if hasattr(self, "_last_terminal") else None,
                    )
                    self._state_stack = self._state_cache.stack4
                    self._game_mode_val = self._state_cache.ram_vals.mode
                    self._gameplay_active = self._state_cache.ram_vals.gameplay_active
                    self._stage_clear_flag = self._state_cache.ram_vals.stage_clear
                    self._ending_active = self._state_cache.ram_vals.ending_active
                    self._player_count = self._state_cache.ram_vals.player_count
                    self._pill_spawn_counter = self._state_cache.ram_vals.pill_counter
                    self._frames_until_drop_val = self._state_cache.ram_vals.gravity_counter
            except Exception as exc:
                warnings.warn(f"Failed to rebuild state cache after auto-start: {exc}")
        # Initialize viruses remaining from RAM if available
        vcount = self._extract_virus_count()
        if vcount is not None:
            self._viruses_remaining = vcount
        self._viruses_initial = max(1, self._viruses_remaining)
        self._viruses_prev = self._viruses_remaining
        # Reset holds
        self._hold_left = self._hold_right = self._hold_down = False
        obs = self._observe(risk_tau=(options.get("risk_tau") if options else self.default_risk_tau))
        self.last_obs = obs
        if self.obs_mode == "state":
            self._state_prev = obs["obs"] if isinstance(obs, dict) else obs
        elif self._use_shaping and self._shaper is not None:
            self._state_prev = obs["obs"] if isinstance(obs, dict) else obs
        else:
            self._state_prev = None
        info: Dict[str, Any] = {"viruses_remaining": self._viruses_remaining, "level": self.level}
        info["raw_ram"] = self._raw_ram_bytes()
        if self._last_rng_seed_bytes is not None:
            info["rng_seed"] = self._last_rng_seed_bytes
        return obs, self._augment_info(info)


    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        self._elapsed_frames += self.frame_skip
        done = False
        truncated = False

        delta_v = 0
        topout = False

        self._game_mode_val = None
        self._gameplay_active = None
        self._stage_clear_flag = None
        self._ending_active = None
        self._player_count = None
        self._pill_spawn_counter = None
        self._frames_until_drop_val = None

        # Drive Retro forward for frame_skip steps; update pixel stack and RAM-based counters
        held = {"LEFT": self._hold_left, "RIGHT": self._hold_right, "DOWN": self._hold_down}
        buttons = discrete10_to_buttons(int(action), held)
        # Optional low-noise debug of input state
        try:
            if _env_flag("DRMARIO_DEBUG_INPUT"):
                print(
                    f"[input] t={int(self._t)} action={int(action)} holds L:{int(self._hold_left)} R:{int(self._hold_right)} D:{int(self._hold_down)} "
                    f"buttons={buttons}",
                    flush=True,
                )
        except Exception:
            pass
        self._hold_left, self._hold_right, self._hold_down = (
            bool(held["LEFT"]),
            bool(held["RIGHT"]),
            bool(held["DOWN"]),
        )

        action_events = 0
        rotation_pressed = bool(buttons[A_BUTTON_INDEX] or buttons[B_BUTTON_INDEX])
        if rotation_pressed:
            action_events += 1

        if self._hold_left:
            new_move_dir = "left"
        elif self._hold_right:
            new_move_dir = "right"
        elif self._hold_down:
            new_move_dir = "down"
        else:
            new_move_dir = "none"

        if new_move_dir != "none" and new_move_dir != self._prev_move_dir:
            action_events += 1

        self._prev_move_dir = new_move_dir

        if self._using_backend and self._backend is not None:
            gameplay_flag: Optional[bool] = None
            try:
                _timing = _env_flag("DRMARIO_TIMING")
                _t0 = perf_counter() if _timing else 0.0
                if _timing:
                    try:
                        print(
                            f"[backend] pre_step t={_t0:.6f} action={int(action)} holds L:{int(self._hold_left)} R:{int(self._hold_right)} D:{int(self._hold_down)}",
                            flush=True,
                        )
                    except Exception:
                        pass
                self._backend_step_buttons(buttons, repeat=self.frame_skip)
                if _timing:
                    _t1 = perf_counter()
                    try:
                        print(
                            f"[backend] post_step dt_ms={( _t1-_t0 )*1000:.3f}",
                            flush=True,
                        )
                    except Exception:
                        pass

                # Snapshot RAM exactly once per step
                ram_arr = self._read_ram_array()  # existing helper; returns np.uint8 array
                ram_bytes = ram_arr.tobytes()

                # Freeze/update canonical state
                self._state_cache = build_state(
                    ram_bytes=ram_bytes,
                    ram_offsets=self._ram_offsets,
                    prev_stack4=None if self._state_cache is None else self._state_cache.stack4,
                    t=self._t,
                    elapsed_frames=self._elapsed_frames,
                    frame_skip=self.frame_skip,
                    last_terminal=self._last_terminal if hasattr(self, "_last_terminal") else None,
                )
                self._state_stack = self._state_cache.stack4

                mode_val = self._state_cache.ram_vals.mode
                gameplay_flag = self._state_cache.ram_vals.gameplay_active
                self._game_mode_val = self._state_cache.ram_vals.mode
                self._gameplay_active = self._state_cache.ram_vals.gameplay_active
                self._stage_clear_flag = self._state_cache.ram_vals.stage_clear
                self._ending_active = self._state_cache.ram_vals.ending_active
                self._player_count = self._state_cache.ram_vals.player_count
                self._pill_spawn_counter = self._state_cache.ram_vals.pill_counter
                self._frames_until_drop_val = self._state_cache.ram_vals.gravity_counter

                if gameplay_flag is False:
                    self._in_game = False
                    self._frames_without_active_pill = 0
                v_now = self._state_cache.calc.viruses_remaining
                if v_now is not None:
                    if gameplay_flag is True and v_now > 0:
                        self._in_game = True
                    elif v_now > 0 and not self._in_game:
                        self._in_game = True
                    detected_topout = False
                    if self.obs_mode != "state":
                        if self._in_game and self._viruses_prev > 0:
                            if self._detect_topout(v_now):
                                detected_topout = True
                        else:
                            self._frames_without_active_pill = 0
                    else:
                        # In state mode, skip heuristic topout detection (fast path via RAM flags)
                        self._frames_without_active_pill = 0
                    if detected_topout:
                        topout = True
                        done = True
                        self._in_game = False
                        self._viruses_remaining = v_now
                        self._viruses_prev = v_now
                    else:
                        delta_v = max(0, self._viruses_prev - v_now)
                        self._viruses_prev = self._viruses_remaining = v_now
            except Exception as exc:
                warnings.warn(f"Backend step failed: {exc}. Falling back to mock dynamics.")
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
                self._using_backend = False
        else:
            # Mock dynamics if Retro not available
            delta_v = 1 if self._rng.rand() < 0.02 else 0
            self._viruses_remaining = max(0, self._viruses_remaining - delta_v)

        obs = self._observe()
        self.last_obs = obs

        # New reward calculation
        r_env = 0.0

        state_prev_frame: Optional[np.ndarray] = None
        state_next_frame: Optional[np.ndarray] = None
        if self.obs_mode == "state" and self._state_prev is not None:
            state_prev_frame = self._state_prev[-1]
            state_next_frame = (obs["obs"] if isinstance(obs, dict) else obs)[-1]

        placement_bonus_adjusted = 0.0
        placement_height_diff: Optional[float] = None
        penalty_unit = float(self.reward_cfg.action_penalty_scale)

        # Fast-path feature flags (skip heavy computations when disabled by config)
        adjacency_enabled = (
            float(self.reward_cfg.adjacency_pair_bonus) != 0.0
            or float(self.reward_cfg.adjacency_triplet_bonus) != 0.0
        )
        height_penalty_enabled = float(self.reward_cfg.column_height_penalty) != 0.0
        placement_bonus_enabled = float(self.reward_cfg.pill_place_base) != 0.0
        placement_height_analysis_enabled = placement_bonus_enabled and bool(self.reward_cfg.punish_high_placements)

        # Pill placement bonus (subject to placement height adjustment)
        current_pill_count = self._pill_spawn_counter or 0
        placement_bonus = 0.0
        if self._prev_pill_count > 0 and current_pill_count > self._prev_pill_count:
            placements = max(1, current_pill_count)
            base_bonus = float(self.reward_cfg.pill_place_base)
            growth = float(self.reward_cfg.pill_place_growth)
            placement_bonus = base_bonus * (1.0 + growth * float((placements - 1) ** 2))
            placement_bonus_adjusted = placement_bonus
            if placement_height_analysis_enabled and state_prev_frame is not None and state_next_frame is not None:
                height_threshold = float(self.reward_cfg.placement_height_threshold)
                penalty_multiplier = float(self.reward_cfg.placement_height_penalty_multiplier)
                prev_static = ram_specs.get_static_color_planes(state_prev_frame)
                next_static = ram_specs.get_static_color_planes(state_next_frame)
                new_static = (next_static > 0.1) & (prev_static <= 0.1)
                board_h = next_static.shape[1] if next_static.ndim >= 3 else 0
                placement_heights: Optional[np.ndarray] = None
                if new_static.any() and board_h > 0:
                    new_rows = np.nonzero(new_static.any(axis=0))[0]
                    if new_rows.size > 0:
                        placement_heights = board_h - new_rows
                virus_mask = ram_specs.get_virus_mask(state_next_frame)
                virus_height = 0
                if virus_mask.any():
                    virus_rows = np.nonzero(virus_mask)[0]
                    if virus_rows.size > 0:
                        highest_virus_row = int(virus_rows.max())
                        virus_height = board_h - highest_virus_row
                if placement_heights is not None and placement_heights.size > 0:
                    placement_height_diff = float(placement_heights.max() - virus_height)
                    if virus_mask.any():
                        if np.any(placement_heights <= virus_height):
                            placement_bonus_adjusted = placement_bonus
                        elif np.all(placement_heights > virus_height + height_threshold):
                            placement_bonus_adjusted = penalty_multiplier * abs(placement_bonus)
                        else:
                            placement_bonus_adjusted = placement_bonus
                # If analysis disabled or insufficient data, keep adjusted=base and diff=None
            r_env += placement_bonus_adjusted
        self._prev_pill_count = current_pill_count
        if placement_bonus > 0.0:
            penalty_unit = float(self.reward_cfg.action_penalty_scale)

        # Virus and non-virus clear bonus
        adjacency_bonus = 0.0
        nonvirus_bonus = 0.0
        height_penalty_raw = 0.0
        height_penalty_delta = 0.0
        if state_prev_frame is not None and state_next_frame is not None:
            s_prev_latest_frame = state_prev_frame
            s_next_latest_frame = state_next_frame
            if not topout:
                prev_occ_mask = ram_specs.get_occupancy_mask(s_prev_latest_frame)
                next_occ_mask = ram_specs.get_occupancy_mask(s_next_latest_frame)
                s_prev_occupied = float(prev_occ_mask.sum())
                s_next_occupied = float(next_occ_mask.sum())

                total_cleared = s_prev_occupied - s_next_occupied
                if total_cleared < 0:
                    total_cleared = 0 # Pills can be added, so don't count negative clears

                # Heuristics below are superseded by canonical RAM checks in state mode; keep
                # reward signals but gate them based on coefficients for efficiency.
                cleared_non_virus = total_cleared - delta_v
                if cleared_non_virus > 0 and float(self.reward_cfg.non_virus_clear_bonus) != 0.0:
                    nonvirus_bonus = self.reward_cfg.non_virus_clear_bonus * cleared_non_virus
                    r_env += nonvirus_bonus
                if adjacency_enabled:
                    adjacency_bonus = self._compute_adjacency_bonus(s_prev_latest_frame, s_next_latest_frame)
                static_pills = ram_specs.get_static_mask(s_next_latest_frame)
                if height_penalty_enabled:
                    if static_pills.any():
                        board_h = static_pills.shape[0]
                        tallest_height = 0
                        for c in range(static_pills.shape[1]):
                            rows = np.nonzero(static_pills[:, c])[0]
                            if rows.size > 0:
                                highest_row = int(rows[-1])
                                height_from_bottom = board_h - highest_row
                                if height_from_bottom > tallest_height:
                                    tallest_height = height_from_bottom
                        if tallest_height > 0:
                            virus_mask = ram_specs.get_virus_mask(s_next_latest_frame)
                            virus_height = 0
                            if virus_mask.any():
                                virus_rows = np.nonzero(virus_mask)[0]
                                highest_virus_row = int(virus_rows.max())
                                virus_height = board_h - highest_virus_row
                            lines_above_virus = max(0, tallest_height - virus_height)
                            penalty_units = min(tallest_height + 2 * lines_above_virus, 40)
                            height_penalty_raw = -self.reward_cfg.column_height_penalty * float(penalty_units)
                            height_penalty_delta = height_penalty_raw - self._prev_height_penalty
                            r_env += height_penalty_delta
                            self._prev_height_penalty = height_penalty_raw
                    else:
                        # No static pills: decay penalty to 0 when enabled
                        height_penalty_raw = 0.0
                        height_penalty_delta = -self._prev_height_penalty
                        r_env += height_penalty_delta
                        self._prev_height_penalty = 0.0
                else:
                    # Disabled: ensure internal state is neutral without adding reward cost
                    height_penalty_raw = 0.0
                    height_penalty_delta = 0.0
                    self._prev_height_penalty = 0.0
        if adjacency_bonus > 0.0:
            r_env += adjacency_bonus

        r_env += self.reward_cfg.virus_clear_bonus * float(delta_v)

        # Canonical termination from RAM (state mode): overrides any heuristic earlier.
        can_fail, can_clear = self._canonical_ram_outcome()
        if self.obs_mode == "state":
            if can_fail is True:
                topout = True
                done = True
            elif can_clear is True:
                # Prefer clear over any prior heuristic failure
                topout = False
                done = True

        # Terminal conditions
        if self._viruses_remaining == 0 and not topout:
            r_env += self.reward_cfg.terminal_clear_bonus
            done = True
            self._in_game = False
            self._prev_terminal = self._last_terminal = "clear"
        if topout:
            r_env += self.reward_cfg.topout_penalty
            self._prev_terminal = self._last_terminal = "topout"
            self._prev_height_penalty = 0.0

        time_reward = 0.0
        if done:
            elapsed_seconds = self._elapsed_frames / 60.0
            if topout:
                time_reward = self.reward_cfg.time_bonus_topout_per_60_frames * elapsed_seconds
            elif self._viruses_remaining == 0:
                time_reward = -self.reward_cfg.time_penalty_clear_per_60_frames * elapsed_seconds
            if time_reward != 0.0:
                r_env += time_reward

        if self._t > self._t_max:
            truncated = True
        step_action_penalty = penalty_unit * float(action_events)
        if step_action_penalty != 0.0:
            r_env -= step_action_penalty

        r_shape = 0.0
        if self._use_shaping and self._shaper is not None and self._state_prev is not None:
            s_prev = self._state_prev
            # If terminal, set s_next=None so Phi(terminal)=0 by convention
            s_next = None if (done or truncated) else (obs["obs"] if isinstance(obs, dict) else obs)
            r_shape = float(self._shaper.potential_delta(s_prev, s_next))

        if self.obs_mode == 'state':
            self._state_prev = obs["obs"] if isinstance(obs, dict) else obs
        elif self._use_shaping:
            s_next = None if (done or truncated) else (obs["obs"] if isinstance(obs, dict) else obs)
            self._state_prev = s_next

        if done or truncated:
            self._elapsed_frames = 0

        r_total = float(r_env + r_shape)
        base_info: Dict[str, Any] = {
            "t": self._t,
            "viruses_remaining": self._viruses_remaining,
            "delta_v": delta_v,
            "r_env": float(r_env),
            "r_shape": float(r_shape),
            "r_total": float(r_total),
            "time_reward": float(time_reward),
            "adjacency_bonus": float(adjacency_bonus),
            "pill_bonus": float(placement_bonus),
            "pill_bonus_adjusted": float(placement_bonus_adjusted),
            "non_virus_bonus": float(nonvirus_bonus),
            "height_penalty": float(height_penalty_raw),
            "height_penalty_delta": float(height_penalty_delta),
            "placement_height_diff": None if placement_height_diff is None else float(placement_height_diff),
            "action_penalty": float(step_action_penalty),
            "action_events": int(action_events),
            "cleared": bool(self._viruses_remaining == 0 and not topout),
            "topout": bool(topout),
            "backend_active": bool(self._using_backend),
            "terminal_reason": self._last_terminal,
            "level": self.level,
            "raw_ram": self._raw_ram_bytes(),
        }
        if self._game_mode_val is not None:
            base_info["game_mode"] = int(self._game_mode_val)
        if self._gameplay_active is not None:
            base_info["gameplay_active"] = bool(self._gameplay_active)
        if self._stage_clear_flag is not None:
            base_info["stage_clear_flag"] = bool(self._stage_clear_flag)
        if self._ending_active is not None:
            base_info["ending_active"] = bool(self._ending_active)
        if self._player_count is not None:
            base_info["player_count"] = int(self._player_count)
        if self._pill_spawn_counter is not None:
            base_info["pill_counter"] = int(self._pill_spawn_counter)
        if self._frames_until_drop_val is not None:
            base_info["frames_until_drop"] = int(self._frames_until_drop_val)
        if self._reward_config_path is not None:
            base_info["reward_config_path"] = str(self._reward_config_path)
        if (
            self._state_viz_interval
            and self.obs_mode == "state"
            and getattr(self, "_state_stack", None) is not None
            and (self._t % self._state_viz_interval == 0)
        ):
            base_info["state_rgb"] = state_to_rgb(self._state_stack, base_info)
        if not (done or truncated):
            self._last_terminal = None
        info = self._augment_info(base_info)
        return obs, r_total, done, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # For now, return a simple mock frame
        if self.render_mode == "rgb_array":
            if self._using_backend:
                return np.ascontiguousarray(self._last_frame)
            return (self._mock_obs()[0] * 255).astype(np.uint8)
        return None

    # Planning support (mock-friendly)
    def snapshot(self) -> Dict[str, Any]:
        return {
            "t": self._t,
            "viruses_remaining": self._viruses_remaining,
            "rng_state": self._rng.get_state(),
            "state_prev": None if self._state_prev is None else np.copy(self._state_prev),
            "last_obs": self.last_obs if hasattr(self, "last_obs") else None,
            "backend_state": (
                self._backend.serialize() if self._using_backend and self._backend is not None else None
            ),
            "first_boot": self._first_boot,
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self._t = snap["t"]
        self._viruses_remaining = snap["viruses_remaining"]
        self._rng.set_state(snap["rng_state"])  # type: ignore[arg-type]
        self._state_prev = snap["state_prev"]
        self.last_obs = snap.get("last_obs")
        self._first_boot = bool(snap.get("first_boot", False))
        backend_blob = snap.get("backend_state")
        if (
            backend_blob
            and self._using_backend
            and self._backend is not None
            and isinstance(backend_blob, (bytes, bytearray))
        ):
            try:
                self._backend.deserialize(bytes(backend_blob))
                self._last_frame = self._backend.get_frame()
                self._pix_stack = None
                self._update_pixel_stack(self._last_frame)
                self._read_ram_array(refresh=True)
            except Exception as exc:
                warnings.warn(f"Failed to restore backend state: {exc}")

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        super().close()

    def peek_step(self, action: int):
        snap = self.snapshot()
        obs, r, term, trunc, info = self.step(action)
        self.restore(snap)
        return obs, r, term, trunc, info
