from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.specs.timeouts import get_level_timeout
from envs.reward_shaping import PotentialShaper
from envs.specs.ram_to_state import ram_to_state
from envs.backends import make_backend
from envs.backends.base import NES_BUTTONS
from envs.retro.action_adapters import discrete10_to_buttons


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
    # Finalized defaults per design review
    alpha: float = 8.0   # progress shaping for virus clears
    beta: float = 0.5    # tiny bonus for multi-clears
    gamma: float = 0.1   # settle penalty weight
    terminal_clear_bonus: float = 500.0
    topout_penalty: float = -500.0
    mode: str = "speedrun"  # "speedrun" | "mean_safe" | "cvar"
    cvar_alpha: float = 0.25


def _default_seed_registry() -> Path:
    return Path(__file__).with_suffix("").parent / "seeds" / "registry.json"


class DrMarioRetroEnv(gym.Env):
    """Single-agent Dr. Mario environment with pluggable emulator backends.

    Observation modes:
      - pixel: 128x128 RGB, frame-stack 4, normalized to [0,1]
      - state: structured 16x8 board tensor (14 channels), frame-stack 4.
        See docs/STATE_OBS_AND_RAM_MAPPING.md for channel spec and mapping; RAM offsets are
        configured via envs/specs/ram_offsets.json (for this ROM revision) or override with
        env var `DRMARIO_RAM_OFFSETS`.

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
    ) -> None:
        super().__init__()
        assert obs_mode in {"pixel", "state"}
        assert frame_skip in {1, 2}
        self.obs_mode = obs_mode
        self.frame_skip = frame_skip
        self.include_risk_tau = include_risk_tau
        self.default_risk_tau = float(risk_tau)
        self.reward_cfg = reward_config or RewardConfig()
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
        self._rng = np.random.RandomState(0)
        self._t_max = get_level_timeout(self.level)
        # Potential shaping setup
        self._use_shaping = use_potential_shaping and evaluator is not None
        self._shaper: Optional[PotentialShaper] = (
            PotentialShaper(evaluator, gamma=potential_gamma, kappa=potential_kappa)
            if self._use_shaping
            else None
        )
        self._state_prev: Optional[np.ndarray] = None
        # Load RAM offsets for state-mode mapping
        self._ram_offsets = self._load_ram_offsets()

        # Observation space
        if obs_mode == "pixel":
            # 4 x 128 x 128 x 3 float32 in [0,1]
            obs_space = spaces.Box(low=0.0, high=1.0, shape=(4, 128, 128, 3), dtype=np.float32)
        else:
            # 4 x (C x 16 x 8), typical C ~= 14
            obs_space = spaces.Box(low=0.0, high=1.0, shape=(4, 14, 16, 8), dtype=np.float32)

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
        self._in_game = False
        self._last_terminal: Optional[str] = None
        self._prev_terminal: Optional[str] = None

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

    def _extract_virus_count(self) -> Optional[int]:
        ram_arr = self._read_ram_array(refresh=False)
        if ram_arr is None:
            return None
        addr_hex = self._ram_offsets.get("viruses", {}).get("remaining_addr") if "viruses" in self._ram_offsets else None
        if not addr_hex:
            return None
        idx = int(addr_hex, 16)
        if 0 <= idx < ram_arr.shape[0]:
            return int(ram_arr[idx])
        return None

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

    def _run_start_sequence(
        self, presses: int, options: Optional[Dict[str, Any]], *, from_topout: bool = False
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
            for _ in range(presses - 1):
                press_start()
        elif presses == 1:
            align_level()
            press_start()
        else:
            press_start()
            align_level()
            for _ in range(presses - 1):
                press_start()

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
        if presses > 0:
            self._run_start_sequence(presses, options, from_topout=(prev_terminal == "topout"))
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
            C = 14
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
        # RAM->state mapping (C=14, H=16, W=8). If backend present, parse RAM; else zeros
        planes = np.zeros((14, 16, 8), dtype=np.float32)
        if self._using_backend:
            ram_arr = self._read_ram_array()
            if ram_arr is not None:
                try:
                    planes = ram_to_state(ram_arr.tobytes(), self._ram_offsets)
                except Exception:
                    planes = np.zeros((14, 16, 8), dtype=np.float32)
        # Maintain a 4-frame stack along axis 0
        if getattr(self, "_state_stack", None) is None:
            self._state_stack = np.stack([planes for _ in range(4)], axis=0)
        else:
            self._state_stack = np.concatenate([self._state_stack[1:], planes[None, ...]], axis=0)
        return self._state_stack

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
        self._pix_stack = None
        self._state_stack = None
        self._in_game = False
        self._prev_terminal = getattr(self, "_last_terminal", None)
        self._last_terminal = None
        if self._using_backend and self._backend is not None:
            try:
                self._backend.reset()
                self._last_frame = self._backend.get_frame()
                self._update_pixel_stack(self._last_frame)
                self._read_ram_array(refresh=True)
            except Exception as exc:
                warnings.warn(f"Backend reset failed: {exc}. Falling back to mock dynamics.")
                try:
                    self._backend.close()
                except Exception:
                    pass
                self._backend = None
                self._using_backend = False
        # Optional frame offset to influence ROM RNG by letting frames elapse before we start
        frame_offset = int((options or {}).get("frame_offset", 0))
        if frame_offset > 0 and self._using_backend and self._backend is not None:
            noop = self._button_vector(None)
            try:
                self._backend_step_buttons(noop, repeat=frame_offset)
            except Exception as exc:
                warnings.warn(f"Backend frame_offset failed: {exc}")
        if self._using_backend and self._backend is not None:
            try:
                self._maybe_auto_start(options)
            except Exception as exc:
                warnings.warn(f"Auto-start sequence failed: {exc}")
        # Initialize viruses remaining from RAM if available
        vcount = self._extract_virus_count()
        if vcount is not None:
            self._viruses_remaining = vcount
        self._viruses_prev = self._viruses_remaining
        # Reset holds
        self._hold_left = self._hold_right = self._hold_down = False
        obs = self._observe(risk_tau=(options.get("risk_tau") if options else self.default_risk_tau))
        self.last_obs = obs
        if self._use_shaping and self._shaper is not None:
            self._state_prev = obs["obs"] if isinstance(obs, dict) else obs
        else:
            self._state_prev = None
        info: Dict[str, Any] = {"viruses_remaining": self._viruses_remaining, "level": self.level}
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        done = False
        truncated = False

        delta_v = 0
        topout = False

        # Drive Retro forward for frame_skip steps; update pixel stack and RAM-based counters
        held = {"LEFT": self._hold_left, "RIGHT": self._hold_right, "DOWN": self._hold_down}
        buttons = discrete10_to_buttons(int(action), held)
        self._hold_left, self._hold_right, self._hold_down = (
            bool(held["LEFT"]),
            bool(held["RIGHT"]),
            bool(held["DOWN"]),
        )

        if self._using_backend and self._backend is not None:
            try:
                self._backend_step_buttons(buttons, repeat=self.frame_skip)
                v_now = self._extract_virus_count()
                if v_now is not None:
                    if v_now > 0 and not self._in_game:
                        self._in_game = True
                    if self._in_game and self._viruses_prev > 0 and v_now >= 200:
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

        # Reward shaping per finalized spec
        r_env = -1.0 + self.reward_cfg.alpha * float(delta_v)
        if self._viruses_remaining == 0 and not topout:
            r_env += self.reward_cfg.terminal_clear_bonus
            done = True
            self._in_game = False
            self._prev_terminal = self._last_terminal = "clear"
        if topout:
            r_env += self.reward_cfg.topout_penalty
            self._prev_terminal = self._last_terminal = "topout"

        if self._t > self._t_max:
            truncated = True

        obs = self._observe()
        self.last_obs = obs
        r_shape = 0.0
        if self._use_shaping and self._shaper is not None and self._state_prev is not None:
            s_prev = self._state_prev
            # If terminal, set s_next=None so Phi(terminal)=0 by convention
            s_next = None if (done or truncated) else (obs["obs"] if isinstance(obs, dict) else obs)
            r_shape = float(self._shaper.potential_delta(s_prev, s_next))
            self._state_prev = s_next
        r_total = float(r_env + r_shape)
        info: Dict[str, Any] = {
            "t": self._t,
            "viruses_remaining": self._viruses_remaining,
            "delta_v": delta_v,
            "r_env": float(r_env),
            "r_shape": float(r_shape),
            "r_total": float(r_total),
            "cleared": bool(self._viruses_remaining == 0 and not topout),
            "topout": bool(topout),
            "backend_active": bool(self._using_backend),
            "terminal_reason": self._last_terminal,
            "level": self.level,
        }
        if not (done or truncated):
            self._last_terminal = None
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
