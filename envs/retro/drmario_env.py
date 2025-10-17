from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.specs.timeouts import get_level_timeout
from envs.reward_shaping import PotentialShaper
from envs.specs.ram_to_state import ram_to_state
try:
    from envs.retro.stable_retro_utils import make_retro_env, get_buttons_layout
except Exception:  # optional import
    make_retro_env = None  # type: ignore
    get_buttons_layout = None  # type: ignore


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
    mode: str = "speedrun"  # "speedrun" | "mean_safe" | "cvar"
    cvar_alpha: float = 0.25


def _default_seed_registry() -> Path:
    return Path(__file__).with_suffix("").parent / "seeds" / "registry.json"


class DrMarioRetroEnv(gym.Env):
    """Single-agent Dr. Mario environment (Stable-Retro wrapper skeleton).

    Observation modes:
      - pixel: 128x128 RGB, frame-stack 4, normalized to [0,1]
      - state: structured 16x8 board tensor (14 channels), frame-stack 4.
        See docs/STATE_OBS_AND_RAM_MAPPING.md for channel spec and mapping; RAM offsets are
        configured via envs/specs/ram_offsets.json (for this ROM revision) or override with
        env var `DRMARIO_RAM_OFFSETS`.

    Notes:
      - This skeleton does not yet bind to stable-retro. It mocks transitions to
        enable wiring of training/eval code paths. Replace TODOs once retro is available.
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
        render_mode: Optional[str] = None,
        level: int = 0,
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
        self.core_path = core_path
        self.render_mode = render_mode
        self.level = level

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

        # Attempt to bind a Stable-Retro env (optional at bootstrap)
        self._retro = None
        self._using_retro = False
        self._buttons_layout = None
        if make_retro_env is not None:
            try:
                self._retro = make_retro_env()
                self._using_retro = True
                self._buttons_layout = get_buttons_layout() if get_buttons_layout else None
            except Exception:
                self._retro = None
                self._using_retro = False
        self._last_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self._pix_stack: Optional[np.ndarray] = None  # (4,128,128,3)
        # Hold state (used when retro is active)
        self._hold_left = False
        self._hold_right = False
        self._hold_down = False

    def _resize_rgb(self, rgb: np.ndarray, out_hw=(128, 128)) -> np.ndarray:
        # Nearest-neighbor resize without external deps
        h, w = rgb.shape[:2]
        oh, ow = out_hw
        ys = (np.linspace(0, h - 1, oh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, ow)).astype(np.int32)
        return rgb[ys][:, xs]

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
        if self._using_retro:
            if self._pix_stack is None:
                small = self._resize_rgb(self._last_frame, (128, 128)).astype(np.float32) / 255.0
                self._pix_stack = np.stack([small for _ in range(4)], axis=0)
            return self._pix_stack
        return self._mock_obs()

    def _state_obs(self) -> np.ndarray:
        # RAM->state mapping (C=14, H=16, W=8). If Retro present, parse RAM; else zeros
        if self._using_retro and self._retro is not None:
            try:
                ram_bytes = self._retro.get_ram()
                planes = ram_to_state(ram_bytes, self._ram_offsets)
            except Exception:
                planes = np.zeros((14, 16, 8), dtype=np.float32)
        else:
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
            return {"obs": core, "risk_tau": np.float32(tau)}
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
        # Reset underlying retro env if present
        if self._using_retro and self._retro is not None:
            try:
                ob, _ = self._retro.reset()
                if isinstance(ob, np.ndarray):
                    self._last_frame = ob
                self._pix_stack = None
            except Exception:
                pass
        # Optional frame offset to influence ROM RNG by letting frames elapse before we start
        frame_offset = int((options or {}).get("frame_offset", 0))
        if frame_offset > 0 and self._using_retro and self._retro is not None:
            # Advance with NOOP inputs for frame_offset frames
            try:
                import numpy as _np
                _noop = _np.zeros((len(self._buttons_layout or ("B","A","SELECT","START","UP","DOWN","LEFT","RIGHT"))), dtype=_np.uint8)
                for _ in range(frame_offset):
                    ob, *_ = self._retro.step(_noop)
                    if isinstance(ob, _np.ndarray):
                        self._last_frame = ob
            except Exception:
                pass
        # Reset holds
        self._hold_left = self._hold_right = self._hold_down = False
        obs = self._observe(risk_tau=(options.get("risk_tau") if options else self.default_risk_tau))
        self.last_obs = obs
        if self._use_shaping and self._shaper is not None:
            self._state_prev = obs["obs"] if isinstance(obs, dict) else obs
        else:
            self._state_prev = None
        info: Dict[str, Any] = {"viruses_remaining": self._viruses_remaining}
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # Placeholder dynamics: random chance to clear a virus
        self._t += 1
        done = False
        truncated = False

        # Drive Retro forward if available (for pixels); env reward remains per spec/mock
        if self._using_retro and self._retro is not None:
            # Map our discrete action to NES MultiBinary via adapter
            from envs.retro.action_adapters import discrete10_to_buttons
            held = {"LEFT": self._hold_left, "RIGHT": self._hold_right, "DOWN": self._hold_down}
            a_list = discrete10_to_buttons(int(action), held)
            # Update held state from adapter's dict
            self._hold_left, self._hold_right, self._hold_down = held["LEFT"], held["RIGHT"], held["DOWN"]
            a_vec = np.array(a_list, dtype=np.uint8)
            try:
                ob, _, _, _, _ = self._retro.step(a_vec)
                if isinstance(ob, np.ndarray):
                    self._last_frame = ob
                    small = self._resize_rgb(self._last_frame, (128, 128)).astype(np.float32) / 255.0
                    if self._pix_stack is None:
                        self._pix_stack = np.stack([small for _ in range(4)], axis=0)
                    else:
                        self._pix_stack = np.concatenate([self._pix_stack[1:], small[None, ...]], axis=0)
            except Exception:
                pass

        # Mock gameplay dynamics for reward/termination until RAM mapping is implemented
        delta_v = 1 if self._rng.rand() < 0.02 else 0
        self._viruses_remaining = max(0, self._viruses_remaining - delta_v)

        # Reward shaping per finalized spec (placeholder mechanics)
        r_env = -1.0 + self.reward_cfg.alpha * float(delta_v)
        if self._viruses_remaining == 0:
            r_env += self.reward_cfg.terminal_clear_bonus
            done = True

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
            "viruses_remaining": self._viruses_remaining if not self._using_retro else -1,
            "delta_v": delta_v,
            "r_env": float(r_env),
            "r_shape": float(r_shape),
            "r_total": float(r_total),
            "cleared": bool(self._viruses_remaining == 0) if not self._using_retro else done,
        }
        return obs, r_total, done, truncated, info

    def render(self) -> Optional[np.ndarray]:
        # For now, return a simple mock frame
        if self.render_mode == "rgb_array":
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
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self._t = snap["t"]
        self._viruses_remaining = snap["viruses_remaining"]
        self._rng.set_state(snap["rng_state"])  # type: ignore[arg-type]
        self._state_prev = snap["state_prev"]
        self.last_obs = snap.get("last_obs")

    def peek_step(self, action: int):
        snap = self.snapshot()
        obs, r, term, trunc, info = self.step(action)
        self.restore(snap)
        return obs, r, term, trunc, info
