"""SMDP-PPO trainer for placement policies.

Implements PPO with SMDP (Semi-Markov Decision Process) discounting where
actions span variable durations τ and credit assignment uses γ^τ.
"""

from __future__ import annotations

import re
import time
from contextlib import nullcontext
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    from torch.optim.swa_utils import get_ema_multi_avg_fn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore
    get_ema_multi_avg_fn = None  # type: ignore

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.models.policy.candidate_packing import (
    candidate_bucket_width,
    pack_feasible_candidates_tensor_batch,
)
from drmc_rl.models.policy.candidate_policy import CandidatePlacementPolicyNet
from drmc_rl.models.policy.placement_dist import MaskedPlacementDist
from drmc_rl.models.policy.placement_heads import PlacementPolicyNet
from drmc_rl.training.algo.base import AlgoAdapter
from drmc_rl.training.algo.search_distill import (
    SearchDistillConfig,
    blend_value_targets,
    masked_distill_kl,
)
from drmc_rl.training.rollout.decision_buffer import DecisionBatch, DecisionRolloutBuffer
from drmc_rl.training.utils.checkpoint_io import checkpoint_path, load_checkpoint, save_checkpoint
from drmc_rl.training.utils.reproducibility import git_commit

_AUX_SPEC_NONE = "none"
_AUX_SPEC_V1 = "v1"
_AUX_SPEC_V1_VS = "v1_vs"

_AUX_V1_LEVEL_MIN = -15
_AUX_V1_LEVEL_MAX = 20
_AUX_V1_LEVEL_DIM = _AUX_V1_LEVEL_MAX - _AUX_V1_LEVEL_MIN + 1  # 36
_AUX_V1_VIRUS_NORM = 84.0  # Max viruses at level 20: (20+1)*4 = 84

# v1 feature layout (float32, [B, 57]):
#   speed_onehot[3]
#   virus_total/84
#   virus_by_color/84 [3] (R,Y,B)
#   level_onehot[36] for levels [-15..20] (out-of-range => all zeros)
#   frame_count_norm [1] (task/frames_used normalized)
#   max_height/16 [1]
#   col_heights/16 [8]
#   clearance_progress [1] (matches or viruses)
#   feasible_fraction [1] (placements/options / 512)
#   occupancy_fraction [1] (occupied tiles / 128)
#   virus_max_height/16 [1]
_AUX_V1_DIM = 3 + 1 + 3 + _AUX_V1_LEVEL_DIM + 1 + 1 + 8 + 1 + 1 + 1 + 1  # 57

# v1_vs = the 57 v1 features (built identically) + VS opponent scalars
# (`vs/*` info keys from `DrMarioVsPoolVecEnv`; all zeros in 1P envs):
#   opp_virus_total/84 [1]
#   garbage_pending_self/4 [1] (NES attack buffer holds at most 4 columns)
#   garbage_pending_opp/4 [1]
#   opp_pill_onehot [6] (2 halves x 3 canonical colors R,Y,B)
#   opp_preview_onehot [6]
_AUX_V1_VS_EXTRA = 1 + 1 + 1 + 6 + 6  # 15
_AUX_V1_VS_DIM = _AUX_V1_DIM + _AUX_V1_VS_EXTRA  # 72
_AUX_GARBAGE_PENDING_NORM = 4.0

_AUX_DIM_BY_SPEC = {_AUX_SPEC_V1: _AUX_V1_DIM, _AUX_SPEC_V1_VS: _AUX_V1_VS_DIM}

# Order must match the stacked per-minibatch metric rows in `_update_policy`.
_UPDATE_METRIC_KEYS = (
    "loss/policy",
    "loss/value",
    "loss/total",
    "policy/entropy",
    "policy/kl",
    "policy/clip_frac",
)


@dataclass(slots=True)
class SMDPPPOConfig:
    """Configuration for SMDP-PPO."""

    # Learning
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95

    # PPO
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Rollout
    decisions_per_update: int = 512
    num_epochs: int = 4
    minibatch_size: int = 128

    # Policy head
    head_type: str = "dense"  # dense, shift_score, or factorized
    pill_embed_dim: int = 32
    pill_embed_type: str = "unordered"  # unordered|ordered_onehot|ordered_pair
    encoder_blocks: int = 0
    policy_type: str = "heatmap"  # heatmap|candidate

    # Candidate-scoring policy params (policy_type=candidate)
    candidate_board_encoder: str = "cnn"  # cnn|col_transformer
    candidate_architecture: str = "g4"  # g4|g5
    candidate_board_channels: int = 0  # 0=auto: non-feasibility bottle channels
    candidate_max_candidates: int = 512
    candidate_d_model: int = 128
    candidate_pos_embed_dim: int = 32
    candidate_cost_embed_dim: int = 32
    candidate_hidden_dim: int = 256
    candidate_transformer_layers: int = 4
    candidate_cross_layers: int = 0
    candidate_transformer_heads: int = 4
    candidate_transformer_ff_mult: int = 4
    candidate_interaction_layers: int = 2
    candidate_value_atoms: int = 51
    candidate_conditioned_trunk: bool = True
    candidate_opponent_features: bool = True
    candidate_cross_ff_mult: int = 2
    candidate_patch_kernel: int = 3

    # Optional auxiliary vector inputs (derived from obs + info).
    aux_spec: str = "none"  # none|v1

    # Exploration
    entropy_schedule_end: float = 0.003
    entropy_schedule_steps: int = 1000000
    use_gumbel_topk: bool = False
    gumbel_k: int = 2

    # Misc
    value_loss_type: str = "mse"  # mse or huber
    compile_mode: str = "off"  # off|default
    rollout_compute_dtype: str = "fp32"  # fp32|bf16
    update_compute_dtype: str = "fp32"  # fp32|bf16


@dataclass(slots=True)
class _DeviceRolloutWave:
    observations: torch.Tensor
    pill_colors: torch.Tensor
    preview_pill_colors: torch.Tensor
    aux: Optional[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_cost: torch.Tensor


@dataclass(slots=True)
class _DeviceRolloutBatch:
    observations: torch.Tensor
    pill_colors: torch.Tensor
    preview_pill_colors: torch.Tensor
    aux: Optional[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    candidate_actions: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_cost: torch.Tensor


class _DeviceRolloutBuffer:
    """Retain rollout inference tensors and concatenate them once for PPO."""

    def __init__(self, *, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.waves: List[_DeviceRolloutWave] = []
        self.size = 0

    def add(
        self,
        wave: _DeviceRolloutWave,
        *,
        actions: np.ndarray,
        log_probs: np.ndarray,
        replace_policy_outputs: bool,
    ) -> None:
        count = int(wave.observations.shape[0])
        end = self.size + count
        if end > self.capacity:
            raise BufferError(
                f"Device rollout batch exceeds capacity: {self.size}+{count}>{self.capacity}"
            )
        if replace_policy_outputs:
            wave = _DeviceRolloutWave(
                observations=wave.observations,
                pill_colors=wave.pill_colors,
                preview_pill_colors=wave.preview_pill_colors,
                aux=wave.aux,
                actions=torch.from_numpy(actions).to(self.device),
                log_probs=torch.from_numpy(log_probs).to(self.device),
                candidate_actions=wave.candidate_actions,
                candidate_mask=wave.candidate_mask,
                candidate_cost=wave.candidate_cost,
            )
        self.waves.append(wave)
        self.size = end

    def batch(self, size: int) -> _DeviceRolloutBatch:
        if int(size) != self.size:
            raise RuntimeError(
                f"CPU/device rollout size mismatch: CPU={int(size)}, device={self.size}"
            )
        aux = [wave.aux for wave in self.waves]
        candidate_width = max(int(wave.candidate_actions.shape[1]) for wave in self.waves)

        def _pad_candidates(value: torch.Tensor, fill: float | int) -> torch.Tensor:
            missing = candidate_width - int(value.shape[1])
            return value if missing == 0 else F.pad(value, (0, missing), value=fill)

        return _DeviceRolloutBatch(
            observations=torch.cat([wave.observations for wave in self.waves]),
            pill_colors=torch.cat([wave.pill_colors for wave in self.waves]),
            preview_pill_colors=torch.cat(
                [wave.preview_pill_colors for wave in self.waves]
            ),
            aux=None if aux[0] is None else torch.cat(aux),  # type: ignore[arg-type]
            actions=torch.cat([wave.actions for wave in self.waves]),
            log_probs=torch.cat([wave.log_probs for wave in self.waves]),
            candidate_actions=torch.cat(
                [_pad_candidates(wave.candidate_actions, -1) for wave in self.waves]
            ),
            candidate_mask=torch.cat(
                [_pad_candidates(wave.candidate_mask, False) for wave in self.waves]
            ),
            candidate_cost=torch.cat(
                [_pad_candidates(wave.candidate_cost, 0.0) for wave in self.waves]
            ),
        )

    def clear(self) -> None:
        self.waves.clear()
        self.size = 0


class SMDPPPOAdapter(AlgoAdapter):
    """PPO trainer for placement policies with SMDP discounting."""

    def __init__(
        self,
        cfg: Any,
        env: Any,
        logger: Any,
        event_bus: Any,
        device: Optional[str] = None,
    ):
        super().__init__(cfg, env, logger, event_bus, device=device)

        if torch is None:
            raise RuntimeError("PyTorch is required for SMDP-PPO")

        # Parse config
        ppo_cfg_dict = getattr(cfg, "smdp_ppo", {})
        if hasattr(ppo_cfg_dict, "to_dict"):
            ppo_cfg_dict = ppo_cfg_dict.to_dict()

        self.hparams = SMDPPPOConfig(
            lr=float(ppo_cfg_dict.get("lr", 3e-4)),
            gamma=float(ppo_cfg_dict.get("gamma", 0.995)),
            gae_lambda=float(ppo_cfg_dict.get("gae_lambda", 0.95)),
            clip_epsilon=float(ppo_cfg_dict.get("clip_epsilon", 0.2)),
            value_coef=float(ppo_cfg_dict.get("value_coef", 0.5)),
            entropy_coef=float(ppo_cfg_dict.get("entropy_coef", 0.01)),
            max_grad_norm=float(ppo_cfg_dict.get("max_grad_norm", 0.5)),
            decisions_per_update=int(ppo_cfg_dict.get("decisions_per_update", 512)),
            num_epochs=int(ppo_cfg_dict.get("num_epochs", 4)),
            minibatch_size=int(ppo_cfg_dict.get("minibatch_size", 128)),
            head_type=str(ppo_cfg_dict.get("head_type", "dense")),
            pill_embed_dim=int(ppo_cfg_dict.get("pill_embed_dim", 32)),
            pill_embed_type=str(ppo_cfg_dict.get("pill_embed_type", "unordered")),
            encoder_blocks=int(ppo_cfg_dict.get("encoder_blocks", 0)),
            policy_type=str(ppo_cfg_dict.get("policy_type", "heatmap")),
            candidate_board_encoder=str(ppo_cfg_dict.get("candidate_board_encoder", "cnn")),
            candidate_architecture=str(ppo_cfg_dict.get("candidate_architecture", "g4")),
            candidate_board_channels=int(ppo_cfg_dict.get("candidate_board_channels", 0)),
            candidate_max_candidates=int(ppo_cfg_dict.get("candidate_max_candidates", 512)),
            candidate_d_model=int(ppo_cfg_dict.get("candidate_d_model", 128)),
            candidate_pos_embed_dim=int(ppo_cfg_dict.get("candidate_pos_embed_dim", 32)),
            candidate_cost_embed_dim=int(ppo_cfg_dict.get("candidate_cost_embed_dim", 32)),
            candidate_hidden_dim=int(ppo_cfg_dict.get("candidate_hidden_dim", 256)),
            candidate_transformer_layers=int(ppo_cfg_dict.get("candidate_transformer_layers", 4)),
            candidate_cross_layers=int(ppo_cfg_dict.get("candidate_cross_layers", 0)),
            candidate_transformer_heads=int(ppo_cfg_dict.get("candidate_transformer_heads", 4)),
            candidate_transformer_ff_mult=int(ppo_cfg_dict.get("candidate_transformer_ff_mult", 4)),
            candidate_interaction_layers=int(ppo_cfg_dict.get("candidate_interaction_layers", 2)),
            candidate_value_atoms=int(ppo_cfg_dict.get("candidate_value_atoms", 51)),
            candidate_conditioned_trunk=bool(
                ppo_cfg_dict.get("candidate_conditioned_trunk", True)
            ),
            candidate_opponent_features=bool(
                ppo_cfg_dict.get("candidate_opponent_features", True)
            ),
            candidate_cross_ff_mult=int(ppo_cfg_dict.get("candidate_cross_ff_mult", 2)),
            candidate_patch_kernel=int(ppo_cfg_dict.get("candidate_patch_kernel", 3)),
            aux_spec=str(ppo_cfg_dict.get("aux_spec", "none")),
            entropy_schedule_end=float(ppo_cfg_dict.get("entropy_schedule_end", 0.003)),
            entropy_schedule_steps=int(ppo_cfg_dict.get("entropy_schedule_steps", 1000000)),
            use_gumbel_topk=bool(ppo_cfg_dict.get("use_gumbel_topk", False)),
            gumbel_k=int(ppo_cfg_dict.get("gumbel_k", 2)),
            value_loss_type=str(ppo_cfg_dict.get("value_loss_type", "mse")),
            compile_mode=str(ppo_cfg_dict.get("compile_mode", "off")),
            rollout_compute_dtype=str(ppo_cfg_dict.get("rollout_compute_dtype", "fp32")),
            update_compute_dtype=str(ppo_cfg_dict.get("update_compute_dtype", "fp32")),
        )

        self._autocast_dtypes: Dict[str, Optional[torch.dtype]] = {}
        for phase, value in (
            ("rollout", self.hparams.rollout_compute_dtype),
            ("update", self.hparams.update_compute_dtype),
        ):
            dtype_name = value.strip().lower()
            if dtype_name not in {"fp32", "bf16"}:
                raise ValueError(
                    f"Unknown smdp_ppo.{phase}_compute_dtype: {value!r}; "
                    "expected fp32 or bf16"
                )
            if dtype_name == "bf16" and torch.device(self.device).type != "cuda":
                raise ValueError(
                    f"smdp_ppo.{phase}_compute_dtype=bf16 requires a CUDA device"
                )
            self._autocast_dtypes[phase] = (
                torch.bfloat16 if dtype_name == "bf16" else None
            )

        policy_type_norm = str(self.hparams.policy_type or "heatmap").strip().lower()
        if policy_type_norm not in {"heatmap", "candidate"}:
            raise ValueError(f"Unknown smdp_ppo.policy_type: {self.hparams.policy_type!r}")
        self.policy_type = policy_type_norm

        pill_embed_type_norm = str(self.hparams.pill_embed_type or "unordered").strip().lower()
        if pill_embed_type_norm not in {
            "unordered",
            "deepsets",
            "unordered_embed",
            "ordered_onehot",
            "ordered",
            "onehot",
            "ordered_pair",
        }:
            raise ValueError(f"Unknown smdp_ppo.pill_embed_type: {self.hparams.pill_embed_type!r}")
        self.pill_embed_type = pill_embed_type_norm

        aux_spec_norm = str(self.hparams.aux_spec or "none").strip().lower()
        if aux_spec_norm not in {_AUX_SPEC_NONE, _AUX_SPEC_V1, _AUX_SPEC_V1_VS}:
            raise ValueError(f"Unknown smdp_ppo.aux_spec: {self.hparams.aux_spec!r}")
        self.aux_spec = aux_spec_norm
        self.aux_dim = int(_AUX_DIM_BY_SPEC.get(self.aux_spec, 0))

        self.candidate_max = int(max(1, int(self.hparams.candidate_max_candidates)))

        # Search-amplified training targets (docs/SEARCH_DISTILL.md); OFF by
        # default — the flag-off path must stay bit-identical to plain PPO.
        sd_dict = ppo_cfg_dict.get("search_distill", {}) or {}
        if hasattr(sd_dict, "to_dict"):
            sd_dict = sd_dict.to_dict()
        self.search_distill_cfg = SearchDistillConfig.from_dict(sd_dict)
        self._sd_runner = None

        # Environment info
        obs_space = getattr(env, "single_observation_space", env.observation_space)
        obs_shape = obs_space.shape  # expected [C, 16, 8] when frame_stack == 1
        in_channels = obs_shape[0] if len(obs_shape) == 3 else 12

        # Create policy network
        if self.policy_type == "candidate":
            # Validate that the observation channel layout matches the candidate policy assumptions
            # (first 4 channels must be color_{r,y,b} + virus_mask).
            try:
                env_cfg = getattr(cfg, "env", None)
                state_repr = getattr(env_cfg, "state_repr", None) if env_cfg is not None else None
                names: Tuple[str, ...] = tuple()
                if state_repr is not None:
                    names = ram_specs.get_plane_names(str(state_repr))
                    if len(names) >= 4 and tuple(names[:4]) != (
                        "color_red",
                        "color_yellow",
                        "color_blue",
                        "virus_mask",
                    ):
                        raise ValueError(
                            "Candidate policy assumes obs[:4] == "
                            "(color_red,color_yellow,color_blue,virus_mask); "
                            f"got {tuple(names[:4])!r} for state_repr={state_repr!r}."
                        )
            except Exception as e:
                # Raise as ValueError to keep config errors actionable.
                raise ValueError(str(e)) from e

            candidate_board_channels = int(self.hparams.candidate_board_channels)
            if candidate_board_channels <= 0:
                candidate_board_channels = int(in_channels)
                if names:
                    for i, name in enumerate(names):
                        if str(name).startswith("feasible_"):
                            candidate_board_channels = int(i)
                            break
            candidate_board_channels = int(
                max(4, min(int(candidate_board_channels), int(in_channels)))
            )
            if names:
                included = tuple(names[:candidate_board_channels])
                if any(str(name).startswith("feasible_") for name in included):
                    raise ValueError(
                        "candidate_board_channels must exclude feasible mask planes; "
                        f"got {candidate_board_channels} for state_repr={state_repr!r}."
                    )

            architecture = self.hparams.candidate_architecture.strip().lower()
            common = dict(
                in_channels=int(in_channels),
                board_channels=int(candidate_board_channels),
                encoder_blocks=self.hparams.encoder_blocks,
                d_model=self.hparams.candidate_d_model,
                pill_embed_dim=self.hparams.pill_embed_dim,
                pill_embed_type=self.pill_embed_type,
                num_colors=3,
                aux_dim=self.aux_dim,
                pos_embed_dim=self.hparams.candidate_pos_embed_dim,
                cost_embed_dim=self.hparams.candidate_cost_embed_dim,
                cand_hidden_dim=self.hparams.candidate_hidden_dim,
                transformer_heads=self.hparams.candidate_transformer_heads,
                transformer_ff_mult=self.hparams.candidate_transformer_ff_mult,
                cross_layers=self.hparams.candidate_cross_layers,
                patch_kernel=self.hparams.candidate_patch_kernel,
            )
            if architecture == "g5":
                from drmc_rl.models.policy.candidate_policy_g5 import (
                    G5CandidatePlacementPolicyNet,
                )

                self.net = G5CandidatePlacementPolicyNet(
                    **common,
                    interaction_layers=self.hparams.candidate_interaction_layers,
                    value_atoms=self.hparams.candidate_value_atoms,
                    conditioned_trunk=self.hparams.candidate_conditioned_trunk,
                    opponent_features=self.hparams.candidate_opponent_features,
                    cross_ff_mult=self.hparams.candidate_cross_ff_mult,
                ).to(self.device)
            elif architecture == "g4":
                self.net = CandidatePlacementPolicyNet(
                    **common,
                    board_encoder=str(self.hparams.candidate_board_encoder),
                    transformer_layers=self.hparams.candidate_transformer_layers,
                ).to(self.device)
            else:
                raise ValueError(f"Unknown candidate_architecture: {architecture!r}")
        else:
            self.net = PlacementPolicyNet(
                in_channels=in_channels,
                head_type=self.hparams.head_type,
                pill_embed_dim=self.hparams.pill_embed_dim,
                pill_embed_type=self.pill_embed_type,
                encoder_blocks=self.hparams.encoder_blocks,
                num_colors=3,
                aux_dim=self.aux_dim,
            ).to(self.device)

        compile_mode = str(self.hparams.compile_mode).strip().lower()
        if compile_mode not in {"off", "default"}:
            raise ValueError(f"Unknown smdp_ppo.compile_mode: {self.hparams.compile_mode!r}")
        if compile_mode != "off":
            if torch.device(self.device).type != "cuda":
                raise ValueError("smdp_ppo.compile_mode requires a CUDA device")
            # Learner rollout and minibatch shapes are stable. Opponent nets
            # remain eager because their PFSP group sizes change every wave.
            self.net.compile(mode=compile_mode, dynamic=False)

        optimizer_kwargs = {"fused": True} if torch.device(self.device).type == "cuda" else {}
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.hparams.lr, **optimizer_kwargs
        )

        # EMA of the policy weights: PPO iterates wander, and evaluating the
        # raw iterate produced large sweep-to-sweep skill swings; the EMA is
        # the deployable/evaluated policy.
        self.ema_decay = float(getattr(self.hparams, "ema_decay", 0.995) or 0.995)
        self._ema_update = get_ema_multi_avg_fn(self.ema_decay)
        self._ema_state = {
            k: v.detach().clone() for k, v in self.net.state_dict().items()
        }

        # Rollout buffer (decision-wise)
        self.buffer = DecisionRolloutBuffer(
            capacity=self.hparams.decisions_per_update,
            obs_shape=obs_shape,
            num_envs=env.num_envs,
            gamma=self.hparams.gamma,
            gae_lambda=self.hparams.gae_lambda,
            aux_dim=self.aux_dim,
            store_costs_to_lock=(self.policy_type == "candidate"),
            search_target_dim=(self.candidate_max if self.search_distill_cfg.enabled else 0),
        )
        self._rollout_env_ids = np.arange(env.num_envs, dtype=np.int32)
        self._pending_device_wave: Optional[_DeviceRolloutWave] = None
        self._device_rollout = (
            _DeviceRolloutBuffer(
                capacity=self.hparams.decisions_per_update,
                device=torch.device(self.device),
            )
            if self.policy_type == "candidate" and torch.device(self.device).type == "cuda"
            else None
        )

        # Tracking
        self.global_step = 0  # Total environment steps (frames)
        self.decision_step = 0  # Total decisions made
        self.total_steps = int(getattr(cfg.train, "total_steps", 5000000))
        self.checkpoint_interval = int(getattr(cfg.train, "checkpoint_interval", 100000))
        # 0 = keep everything (historical behavior). N > 0 = after each save,
        # delete this run's oldest step checkpoints beyond the newest N.
        self.checkpoint_keep_last = int(getattr(cfg.train, "checkpoint_keep_last", 0))
        self._episodes_total = 0
        self._curriculum_last_level: Optional[int] = None
        self._curriculum_last_frames: int = 0
        self._curriculum_last_episodes: int = 0

        self.batch_returns: deque[float] = deque(maxlen=100)
        self.batch_lengths: deque[int] = deque(maxlen=100)
        self.batch_viruses: deque[float] = deque(maxlen=100)
        self.batch_decisions: deque[int] = deque(maxlen=100)

        # Lightweight perf counters (used by debug UI via RateLimitedVecEnv hooks).
        self._perf_inference_calls: int = 0
        self._perf_inference_sec_total: float = 0.0
        self._perf_last_inference_sec: float = 0.0
        self._last_update_step: int = 0

        self.checkpoint_dir = Path(getattr(cfg, "logdir", "runs/smdp_ppo")) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_checkpoint = self.checkpoint_interval

        train_cfg = getattr(cfg, "train", None)
        init_ckpt = getattr(train_cfg, "init_checkpoint", None) if train_cfg is not None else None
        resume_opt = bool(getattr(train_cfg, "resume_optimizer", True)) if train_cfg is not None else True
        resume_step = bool(getattr(train_cfg, "resume_step", True)) if train_cfg is not None else True
        strict_ckpt = bool(getattr(train_cfg, "strict_checkpoint", True)) if train_cfg is not None else True
        if init_ckpt:
            self._load_checkpoint(
                Path(str(init_ckpt)).expanduser(),
                resume_optimizer=resume_opt,
                resume_step=resume_step,
                strict=strict_ckpt,
            )

        # Search-distillation rollout runner (built after any checkpoint load
        # so the first searched rollout uses the loaded weights).
        if self.search_distill_cfg.enabled:
            self._sd_runner = self._build_search_distill_runner()

        # Entropy annealing
        self._entropy_coef_initial = self.hparams.entropy_coef

    def _autocast(self, phase: str):
        dtype = self._autocast_dtypes[phase]
        if dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=True,
        )

    # ----------------------------------------------------- search distillation
    def _build_search_distill_runner(self):
        """Validate + build the rollout-side search runner (enabled=true only).

        Supported obs (docs/SEARCH_DISTILL.md): candidate policy over the
        12-channel `bitplane_bottle_conn_mask` obs, or the 20-channel `_vs`
        opponent-obs layout (search sims rebuild the own-board planes natively;
        opponent planes + the v1_vs aux tail are frozen from the live decision
        and spliced into search node evaluations).
        """

        from drmc_rl.training.algo.search_distill import SearchDistillRunner

        if self.policy_type != "candidate":
            raise ValueError("search_distill requires smdp_ppo.policy_type=candidate")
        obs_space = getattr(self.env, "single_observation_space", self.env.observation_space)
        in_channels = int(obs_space.shape[0]) if len(obs_space.shape) == 3 else 0
        if in_channels not in (12, 20):
            raise ValueError(
                "search_distill requires the 12-channel bitplane_bottle_conn_mask obs "
                "or the 20-channel bitplane_bottle_conn_mask_vs obs; "
                f"got {in_channels} channels."
            )
        # VS rollouts: search the learner's own board with the 1P sim pool and
        # replicate the VS reward (garbage shaping + terminal +-1). 1P rollouts
        # replicate the configured env reward exactly.
        reward_mode = "vs" if callable(getattr(self.env, "get_vs_metrics", None)) else "1p"
        if self.search_distill_cfg.opponent_model == "self":
            if in_channels != 20:
                raise ValueError(
                    "search_distill.opponent_model=self requires the 20-channel "
                    "bitplane_bottle_conn_mask_vs obs (the opponent planes are what "
                    f"the advance updates); got {in_channels} channels."
                )
            if reward_mode != "vs":
                raise ValueError(
                    "search_distill.opponent_model=self requires the VS env "
                    "(vs/opponent_board decision context)."
                )
        return SearchDistillRunner(
            self.search_distill_cfg,
            net=self.net,
            aux_spec=self.aux_spec,
            candidate_max=self.candidate_max,
            num_envs=int(self.env.num_envs),
            reward_mode=reward_mode,
            gamma=float(self.hparams.gamma),
            garbage_reward_coef=float(getattr(self.env, "garbage_reward_coef", 0.05)),
            device=str(self.device),
            seed=int(getattr(self.cfg, "seed", 0) or 0),
        )

    # ---------------------------------------------------------------- training
    def train_forever(self) -> None:
        self._train_sequential()

    def _train_sequential(self) -> None:
        """Main training loop."""
        obs, info = self.env.reset(seed=getattr(self.cfg, "seed", None))
        obs_arr = self._ensure_batched_obs(self._unwrap_obs(obs)).astype(np.float32)

        # Decision-level tracking per environment
        decision_obs = obs_arr.copy()
        decision_info = self._normalize_infos(info)

        start_time = time.time()
        start_step = int(self.global_step)
        start_decision_step = int(self.decision_step)

        while self.global_step < self.total_steps:
            # Collect decisions until buffer is full
            decisions_collected = 0

            while decisions_collected < self.hparams.decisions_per_update:
                # Make a vectorized decision for all environments
                (
                    actions,
                    log_probs,
                    values,
                    masks,
                    costs_to_lock,
                    pill_colors,
                    preview_pill_colors,
                    aux_batch,
                ) = self._select_actions_batch(decision_obs, decision_info, deterministic=False)

                # Search-amplified targets for a sampled subset of decisions.
                # With act_from_search the returned actions/log_probs replace
                # the behavior samples on searched rows (the behavior policy
                # there is the improved policy); otherwise they pass through.
                sd_targets = sd_values = sd_flags = None
                if self._sd_runner is not None:
                    (
                        sd_targets,
                        sd_values,
                        sd_flags,
                        actions,
                        log_probs,
                    ) = self._sd_runner.run(
                        decision_info,
                        masks,
                        costs_to_lock,
                        actions,
                        log_probs,
                        obs=decision_obs,
                        aux=aux_batch,
                    )

                # Step environment once for the full batch.
                obs_after, rewards, terminated, truncated, info_after = self.env.step(actions)

                obs_after_arr = self._ensure_batched_obs(self._unwrap_obs(obs_after)).astype(
                    np.float32
                )
                info_after_list = self._normalize_infos(info_after)
                rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(self.env.num_envs)
                terminated_arr = np.asarray(terminated, dtype=bool).reshape(self.env.num_envs)
                truncated_arr = np.asarray(truncated, dtype=bool).reshape(self.env.num_envs)

                direct_tau = getattr(self.env, "transition_tau", None)
                if callable(direct_tau):
                    tau_arr = np.asarray(direct_tau(), dtype=np.int32).reshape(
                        self.env.num_envs
                    )
                else:
                    tau_arr = np.array(
                        [
                            self._extract_tau(info_after_list[i])
                            for i in range(self.env.num_envs)
                        ],
                        dtype=np.int32,
                    )

                done_arr = terminated_arr | truncated_arr
                if self._sd_runner is not None:
                    self._sd_runner.note_dones(done_arr)

                frames_total = int(np.sum(tau_arr))
                self.global_step += frames_total
                self.decision_step += int(self.env.num_envs)
                decisions_collected += int(self.env.num_envs)

                self.buffer.add_arrays(
                    observations=decision_obs,
                    masks=masks,
                    costs_to_lock=costs_to_lock,
                    pill_colors=pill_colors,
                    preview_pill_colors=preview_pill_colors,
                    aux=aux_batch,
                    actions=actions,
                    log_probs=log_probs,
                    values=values,
                    taus=tau_arr,
                    rewards=rewards_arr,
                    observations_next=obs_after_arr,
                    dones=done_arr,
                    env_ids=self._rollout_env_ids,
                    search_targets=sd_targets,
                    search_values=sd_values,
                    search_mask=sd_flags,
                )
                if self._device_rollout is not None:
                    if self._pending_device_wave is None:
                        raise RuntimeError("CUDA rollout is missing its device wave")
                    self._device_rollout.add(
                        self._pending_device_wave,
                        actions=actions,
                        log_probs=log_probs,
                        replace_policy_outputs=self._sd_runner is not None,
                    )

                advance_from: Optional[int] = None
                advance_to: Optional[int] = None
                for env_idx in range(self.env.num_envs):
                    info_i = info_after_list[env_idx] if env_idx < len(info_after_list) else {}
                    # Track episodes
                    if bool(done_arr[env_idx]):
                        self._episodes_total += 1
                        ep_info = info_i.get("episode", {})
                        drm_info = info_i.get("drm", {})

                        self.batch_returns.append(float(ep_info.get("r", 0.0)))
                        self.batch_lengths.append(int(ep_info.get("l", 0)))
                        self.batch_viruses.append(float(drm_info.get("viruses_cleared", 0.0)))
                        self.batch_decisions.append(int(ep_info.get("decisions", 0)))

                        payload = {
                            "step": self.global_step,
                            "ret": float(ep_info.get("r", 0.0)),
                            "len": int(ep_info.get("l", 0)),
                            "env_index": int(env_idx),
                        }
                        if "decisions" in ep_info:
                            payload["decisions"] = int(ep_info.get("decisions", 0))
                        payload.update({f"drm/{k}": v for k, v in drm_info.items()})
                        self.event_bus.emit("episode_end", **payload)

                    if advance_to is None:
                        adv_to = self._extract_int(info_i.get("curriculum/advanced_to"))
                        if adv_to is not None:
                            advance_to = int(adv_to)
                            advance_from = self._extract_int(info_i.get("curriculum/advanced_from"))

                if advance_to is not None:
                    self._log_curriculum_advance(advance_from, advance_to)

                # Update decision state for next batch.
                decision_obs = obs_after_arr.copy()
                decision_info = info_after_list

                # Keep PPO rollouts stage-pure: once the curriculum advances,
                # stop collecting and update immediately so we don't mix levels
                # within a single PPO update batch.
                if advance_to is not None:
                    break

            # Update policy
            update_start = time.time()

            # Bootstrap values for the last observation per environment.
            with torch.no_grad():
                (
                    _actions,
                    _log_probs,
                    bootstrap_values,
                    _masks,
                    _costs_to_lock,
                    _pill_colors,
                    _preview_pill_colors,
                    _aux_batch,
                ) = self._select_actions_batch(decision_obs, decision_info, deterministic=True)

            batch = self.buffer.get_batch(
                bootstrap_value=np.asarray(bootstrap_values, dtype=np.float32),
                copy_storage=self._device_rollout is None,
            )

            # Candidate-packing telemetry (helps detect truncation / action-set issues).
            candidate_stats: Dict[str, float] = {}
            if self.policy_type == "candidate":
                feasible_counts = (
                    batch.masks.reshape(batch.masks.shape[0], -1).sum(axis=1).astype(np.int32)
                )
                if feasible_counts.size > 0:
                    candidate_stats["candidate/feasible_mean"] = float(np.mean(feasible_counts))
                    candidate_stats["candidate/feasible_p95"] = float(
                        np.percentile(feasible_counts, 95)
                    )
                    candidate_stats["candidate/feasible_max"] = float(np.max(feasible_counts))
                    candidate_stats["candidate/truncation_frac"] = float(
                        np.mean(feasible_counts > int(self.candidate_max))
                    )

            device_batch = (
                None
                if self._device_rollout is None
                else self._device_rollout.batch(len(batch.actions))
            )
            metrics = self._update_policy(batch, device_batch=device_batch)
            metrics.update(candidate_stats)

            if self._sd_runner is not None:
                metrics.update(self._sd_runner.pop_metrics())
                # Next rollout's search targets come from the just-updated net.
                self._sd_runner.refresh_weights(
                    {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
                )

            self.buffer.clear()
            if self._device_rollout is not None:
                self._device_rollout.clear()

            update_time = time.time() - update_start
            metrics["perf/update_sec"] = update_time

            frames_since_update = int(self.global_step - self._last_update_step)
            self._last_update_step = int(self.global_step)
            try:
                if hasattr(self.env, "record_update"):
                    self.env.record_update(float(update_time), frames=frames_since_update)
            except Exception:
                pass

            elapsed = time.time() - start_time
            frames_since_start = max(0, int(self.global_step) - start_step)
            decisions_since_start = max(0, int(self.decision_step) - start_decision_step)
            metrics["perf/sps_frames_total"] = float(frames_since_start / max(elapsed, 1e-6))
            metrics["perf/dps_decisions_total"] = float(decisions_since_start / max(elapsed, 1e-6))
            # Backwards-compatible aliases (used by existing TUI/event handlers).
            metrics["perf/sps"] = float(metrics["perf/sps_frames_total"])
            metrics["perf/dps"] = float(metrics["perf/dps_decisions_total"])

            # Inference timing (policy forward passes outside the PPO update).
            metrics["perf/inference_calls"] = float(self._perf_inference_calls)
            metrics["perf/inference_sec_total"] = float(self._perf_inference_sec_total)
            metrics["perf/last_inference_ms"] = float(self._perf_last_inference_sec) * 1000.0
            if self._perf_inference_calls > 0:
                metrics["perf/inference_ms_avg"] = (
                    float(self._perf_inference_sec_total)
                    * 1000.0
                    / float(self._perf_inference_calls)
                )
            if frames_since_start > 0:
                metrics["perf/inference_ms_per_frame"] = (
                    float(self._perf_inference_sec_total) * 1000.0 / float(frames_since_start)
                )

            curriculum_snapshot = self._extract_curriculum_snapshot(decision_info)
            if curriculum_snapshot is not None:
                metrics.update(self._curriculum_scalar_metrics(curriculum_snapshot))

            self._log_metrics(metrics)
            if curriculum_snapshot is not None:
                self.event_bus.emit(
                    "update_end", step=self.global_step, curriculum=curriculum_snapshot, **metrics
                )
            else:
                self.event_bus.emit("update_end", step=self.global_step, **metrics)
            self._maybe_checkpoint()
            self.logger.flush()

    # ---------------------------------------------------------- policy methods
    def _forward_policy(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        aux: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, float]:
        """Forward pass through policy network.

        Returns:
            Tuple of (logits_map [4, 16, 8], value scalar)
        """
        if self.policy_type != "heatmap":
            raise RuntimeError("_forward_policy is only supported for policy_type=heatmap")
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        colors_t = torch.from_numpy(pill_colors).unsqueeze(0).to(self.device)
        preview_t = torch.from_numpy(preview_pill_colors).unsqueeze(0).to(self.device)
        aux_t = None if aux is None else torch.from_numpy(aux).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with self._autocast("rollout"):
            logits_map, value = self.net(obs_t, colors_t, preview_t, mask_t, aux=aux_t)
        dt = float(time.perf_counter() - t0)
        self._perf_inference_calls += 1
        self._perf_inference_sec_total += dt
        self._perf_last_inference_sec = dt
        try:
            if hasattr(self.env, "record_inference"):
                self.env.record_inference(dt)
        except Exception:
            pass

        return logits_map.squeeze(0), float(value.squeeze().item())

    def _forward_policy_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        aux: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network for a batch.

        Returns:
            Tuple of (logits_map [B, 4, 16, 8], values [B])
        """
        if self.policy_type != "heatmap":
            raise RuntimeError("_forward_policy_batch is only supported for policy_type=heatmap")
        obs_t = torch.from_numpy(obs).to(self.device)
        mask_t = torch.from_numpy(mask).to(self.device)
        colors_t = torch.from_numpy(pill_colors).to(self.device)
        preview_t = torch.from_numpy(preview_pill_colors).to(self.device)
        aux_t = None if aux is None else torch.from_numpy(aux).to(self.device)

        t0 = time.perf_counter()
        with self._autocast("rollout"):
            logits_map, value = self.net(obs_t, colors_t, preview_t, mask_t, aux=aux_t)
        dt = float(time.perf_counter() - t0)
        batch_size = int(obs_t.shape[0])
        self._perf_inference_calls += batch_size
        self._perf_inference_sec_total += dt
        self._perf_last_inference_sec = dt
        try:
            if hasattr(self.env, "record_inference"):
                self.env.record_inference(dt, calls=batch_size)
        except Exception:
            pass

        return logits_map, value.squeeze(-1)

    def _select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        pill_colors: np.ndarray,
        preview_pill_colors: np.ndarray,
        aux: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select action using masked policy.

        Returns:
            Tuple of (action_idx, log_prob, value)
        """
        logits_map, value = self._forward_policy(obs, mask, pill_colors, preview_pill_colors, aux)

        # Create masked distribution
        dist = MaskedPlacementDist(logits_map, torch.from_numpy(mask).to(self.device))

        # Sample
        action_idx, log_prob = dist.sample(deterministic=deterministic)

        return int(action_idx.item()), float(log_prob.item()), value

    def _select_actions_batch(
        self,
        obs_batch: np.ndarray,
        infos: List[Dict[str, Any]],
        deterministic: bool = False,
        policy_net: Optional[nn.Module] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """Select actions for a batch of environments."""
        net = self.net if policy_net is None else policy_net
        num_envs = int(self.env.num_envs)
        obs_arr = self._ensure_batched_obs(obs_batch).astype(np.float32, copy=False)
        direct_batch_fn = getattr(self.env, "policy_batch", None)
        direct = (
            direct_batch_fn(self.aux_spec)
            if callable(direct_batch_fn)
            and bool(getattr(self.env, "direct_policy_batch", False))
            else None
        )
        if direct is not None:
            masks = np.asarray(direct.feasible_mask)
            costs_to_lock = np.asarray(direct.cost_to_lock)
            pill_colors = np.asarray(direct.pill_colors, dtype=np.int64)
            preview_pill_colors = np.asarray(
                direct.preview_pill_colors, dtype=np.int64
            )
            aux_batch = direct.aux
            if masks.shape != (num_envs, 4, 16, 8):
                raise RuntimeError(
                    f"Direct policy batch has mask shape {masks.shape}, "
                    f"expected {(num_envs, 4, 16, 8)}"
                )
        else:
            masks = np.zeros((num_envs, 4, 16, 8), dtype=bool)
            costs_to_lock = np.full(
                (num_envs, 4, 16, 8), np.inf, dtype=np.float32
            )
            pill_colors = np.zeros((num_envs, 2), dtype=np.int64)
            preview_pill_colors = np.zeros((num_envs, 2), dtype=np.int64)
            aux_batch: Optional[np.ndarray] = None
            if self.aux_dim > 0:
                aux_batch = np.zeros((num_envs, self.aux_dim), dtype=np.float32)
            for i in range(num_envs):
                info_i = infos[i] if i < len(infos) else {}
                masks[i] = self._extract_mask(info_i)
                costs_to_lock[i] = self._extract_cost_to_lock(info_i)
                pill_colors[i] = self._extract_pill_colors(info_i)
                preview_pill_colors[i] = self._extract_preview_pill_colors(info_i)
            if aux_batch is not None:
                aux_batch = self._build_aux_batch(obs_arr, infos)

        if self.policy_type == "candidate":
            t0 = time.perf_counter()
            with torch.inference_mode():
                obs_t = torch.from_numpy(obs_arr).to(self.device)
                colors_t = torch.from_numpy(pill_colors).to(self.device)
                preview_t = torch.from_numpy(preview_pill_colors).to(self.device)
                aux_t = None if aux_batch is None else torch.from_numpy(aux_batch).to(self.device)
                packed_width = (
                    self.candidate_max
                    if self._sd_runner is not None
                    else candidate_bucket_width(masks, max_candidates=self.candidate_max)
                )
                packed = pack_feasible_candidates_tensor_batch(
                    torch.from_numpy(masks).to(self.device),
                    torch.from_numpy(costs_to_lock).to(self.device),
                    max_candidates=packed_width,
                )
                cand_actions_t = packed.actions
                cand_cost_t = packed.cost
                cand_mask_t = packed.mask

                with self._autocast("rollout"):
                    logits, values = net(  # type: ignore[misc]
                        obs_t,
                        colors_t,
                        preview_t,
                        cand_actions_t,
                        cand_cost_t,
                        cand_mask_t,
                        aux=aux_t,
                    )
                dist = MaskedPlacementDist(logits.float(), cand_mask_t)
                if deterministic:
                    slot = dist.mode()
                    log_probs = dist.log_prob(slot)
                else:
                    slot, log_probs = dist.sample(deterministic=False)
                actions_t = cand_actions_t.gather(1, slot.unsqueeze(1)).squeeze(1)

                # The native engine needs host actions. Resolve and sample on
                # device, then synchronize once for only the three selected
                # scalars per environment instead of copying every candidate
                # logit back to the CPU.
                selected = torch.stack(
                    (actions_t.float(), log_probs.float(), values.float().reshape(-1)), dim=1
                ).cpu().numpy()
                if self._device_rollout is not None:
                    self._pending_device_wave = _DeviceRolloutWave(
                        observations=obs_t,
                        pill_colors=colors_t,
                        preview_pill_colors=preview_t,
                        aux=aux_t,
                        actions=actions_t,
                        log_probs=log_probs,
                        candidate_actions=cand_actions_t,
                        candidate_mask=cand_mask_t,
                        candidate_cost=cand_cost_t,
                    )
            dt = float(time.perf_counter() - t0)
            batch_size = int(obs_arr.shape[0])
            self._perf_inference_calls += batch_size
            self._perf_inference_sec_total += dt
            self._perf_last_inference_sec = dt
            try:
                if hasattr(self.env, "record_inference"):
                    self.env.record_inference(dt, calls=batch_size)
            except Exception:
                pass

            actions_np = selected[:, 0].astype(np.int64)
            log_probs_np = selected[:, 1].astype(np.float32)
            values_np = selected[:, 2].astype(np.float32)
            return (
                actions_np,
                log_probs_np,
                values_np,
                masks,
                costs_to_lock,
                pill_colors,
                preview_pill_colors,
                aux_batch,
            )

        with torch.inference_mode():
            logits_map, values = self._forward_policy_batch(
                obs_arr, masks, pill_colors, preview_pill_colors, aux_batch
            )
            logits_cpu = logits_map.float().cpu()
            values_np = values.float().cpu().numpy().astype(np.float32)
        dist = MaskedPlacementDist(logits_cpu, torch.from_numpy(masks))
        if deterministic:
            action_idx = dist.mode()
            log_probs = dist.log_prob(action_idx)
        else:
            action_idx, log_probs = dist.sample(deterministic=False)

        actions_np = action_idx.numpy().astype(np.int64)
        log_probs_np = log_probs.numpy().astype(np.float32)

        return (
            actions_np,
            log_probs_np,
            values_np,
            masks,
            costs_to_lock,
            pill_colors,
            preview_pill_colors,
            aux_batch,
        )

    def _execute_placement(
        self,
        env_idx: int,
        action_idx: int,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, List[Dict]]:
        """Execute placement action (may span multiple frames).

        The placement wrapper handles the multi-frame execution internally.
        We just step once and it returns the cumulative result.

        Returns:
            Tuple of (obs, total_reward, terminated, truncated, info)
        """
        # Create action array for vectorized env
        actions = np.full(self.env.num_envs, action_idx, dtype=np.int64)

        # Step environment (wrapper handles τ frames internally)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # Track frames consumed (get from info if available)
        tau = 1  # Default
        if env_idx < len(infos):
            # Try to extract actual tau from info
            tau = infos[env_idx].get("placements/tau", 1)
            if isinstance(tau, np.ndarray):
                tau = int(tau.item())
            else:
                tau = int(tau) if tau else 1

        self.global_step += tau  # Track actual frames consumed

        return obs, float(rewards[env_idx]), terminated, truncated, infos

    # ------------------------------------------------------------------ update
    def _update_policy(
        self,
        batch: DecisionBatch,
        *,
        device_batch: Optional[_DeviceRolloutBatch] = None,
        schedule_step: Optional[int] = None,
    ) -> Dict[str, float]:
        """Update policy using PPO on decision-level batch."""
        T = len(batch.actions)

        masks_np = batch.masks
        costs_to_lock_np = batch.costs_to_lock

        # Convert to tensors
        obs = (
            device_batch.observations
            if device_batch is not None
            else torch.from_numpy(batch.observations).to(self.device)
        )
        masks = torch.from_numpy(batch.masks).to(self.device)
        pill_colors = (
            device_batch.pill_colors
            if device_batch is not None
            else torch.from_numpy(batch.pill_colors).to(self.device)
        )
        preview_pill_colors = (
            device_batch.preview_pill_colors
            if device_batch is not None
            else torch.from_numpy(batch.preview_pill_colors).to(self.device)
        )
        aux = (
            device_batch.aux
            if device_batch is not None
            else None if batch.aux is None else torch.from_numpy(batch.aux).to(self.device)
        )
        actions = (
            device_batch.actions
            if device_batch is not None
            else torch.from_numpy(batch.actions).to(self.device)
        )
        log_probs_old = (
            device_batch.log_probs
            if device_batch is not None
            else torch.from_numpy(batch.log_probs).to(self.device)
        )
        returns = torch.from_numpy(batch.returns).to(self.device)
        advantages = torch.from_numpy(batch.advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Current entropy coefficient (annealed)
        entropy_coef = self._get_entropy_coef(step=schedule_step)

        # Search-distillation targets (docs/SEARCH_DISTILL.md). None unless the
        # feature is enabled AND this batch contains searched decisions, so the
        # flag-off path below is bit-identical to plain PPO.
        sd_targets_t: Optional[torch.Tensor] = None
        sd_values_t: Optional[torch.Tensor] = None
        sd_mask_t: Optional[torch.Tensor] = None
        sd_kl_sum: Optional[torch.Tensor] = None
        sd_kl_n: Optional[torch.Tensor] = None
        sd_beta = float(self.search_distill_cfg.beta)
        sd_value_mix = float(self.search_distill_cfg.value_mix)
        if (
            self.search_distill_cfg.enabled
            and batch.search_targets is not None
            and batch.search_mask is not None
            and bool(batch.search_mask.any())
        ):
            sd_targets_t = torch.from_numpy(batch.search_targets).to(self.device)
            sd_values_t = torch.from_numpy(batch.search_values).to(self.device)
            sd_mask_t = torch.from_numpy(batch.search_mask).to(self.device)
            sd_kl_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            sd_kl_n = torch.zeros((), dtype=torch.float32, device=self.device)

        # Candidate-packing precompute: build packed candidates once per PPO update batch
        # (instead of re-packing inside each minibatch loop).
        cand_actions_all_t: Optional[torch.Tensor] = None
        cand_mask_all_t: Optional[torch.Tensor] = None
        cand_cost_all_t: Optional[torch.Tensor] = None
        feasible_counts_np: Optional[np.ndarray] = None
        if self.policy_type == "candidate":
            if costs_to_lock_np is None:
                raise ValueError(
                    "DecisionBatch.costs_to_lock is required for policy_type=candidate"
                )
            kmax = int(self.candidate_max)
            feasible_counts_np = masks_np.reshape(T, -1).sum(axis=1).astype(np.int32, copy=False)
            if device_batch is not None:
                cand_actions_all_t = device_batch.candidate_actions
                cand_mask_all_t = device_batch.candidate_mask
                cand_cost_all_t = device_batch.candidate_cost
            else:
                packed = pack_feasible_candidates_tensor_batch(
                    masks,
                    torch.from_numpy(costs_to_lock_np).to(self.device),
                    max_candidates=kmax,
                )
                cand_actions_all_t = packed.actions
                cand_mask_all_t = packed.mask
                cand_cost_all_t = packed.cost

        # Multiple epochs over the batch
        metrics_accum = {key: 0.0 for key in _UPDATE_METRIC_KEYS}
        metric_rows: List[torch.Tensor] = []

        for epoch in range(self.hparams.num_epochs):
            # Shuffle indices
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, self.hparams.minibatch_size):
                end = min(start + self.hparams.minibatch_size, T)
                mb_indices = indices[start:end]

                # Mini-batch
                mb_obs = obs[mb_indices]
                mb_masks = masks[mb_indices]
                mb_colors = pill_colors[mb_indices]
                mb_preview = preview_pill_colors[mb_indices]
                mb_aux = None if aux is None else aux[mb_indices]
                mb_actions = actions[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                value_aux = None
                if self.policy_type == "candidate":
                    if (
                        cand_actions_all_t is None
                        or cand_mask_all_t is None
                        or cand_cost_all_t is None
                    ):
                        raise RuntimeError("candidate prepack missing (internal error)")
                    cand_actions_t = cand_actions_all_t[mb_indices]
                    cand_mask_t = cand_mask_all_t[mb_indices]
                    cand_cost_t = cand_cost_all_t[mb_indices]

                    # Forward pass
                    with self._autocast("update"):
                        if self.hparams.candidate_architecture.strip().lower() == "g5":
                            logits, values, value_aux = self.net(  # type: ignore[misc]
                                mb_obs,
                                mb_colors,
                                mb_preview,
                                cand_actions_t,
                                cand_cost_t,
                                cand_mask_t,
                                aux=mb_aux,
                                return_aux=True,
                            )
                        else:
                            logits, values = self.net(  # type: ignore[misc]
                                mb_obs,
                                mb_colors,
                                mb_preview,
                                cand_actions_t,
                                cand_cost_t,
                                cand_mask_t,
                                aux=mb_aux,
                            )
                            value_aux = None
                    logits = logits.float()
                    values = values.float()

                    # Compute log probs and entropy in candidate-slot space.
                    eq = cand_actions_t == mb_actions.unsqueeze(1)
                    found = eq.any(dim=1)
                    if not bool(found.all()):
                        missing = (~found).nonzero(as_tuple=False).squeeze(-1)
                        first_missing_action = int(mb_actions[missing[0]].item())
                        src = int(mb_indices[missing[0]].item())
                        feasible_count = (
                            int(feasible_counts_np[src])  # type: ignore[index]
                            if feasible_counts_np is not None
                            else int(masks_np[src].sum())
                        )
                        raise RuntimeError(
                            "PPO update: macro action missing from repacked candidate list "
                            f"({int(missing.numel())}/{int(found.numel())} missing; "
                            f"first={first_missing_action}; feasible_count={feasible_count}; "
                            f"candidate_max={self.candidate_max}). "
                            "This indicates nondeterministic packing or truncation."
                        )
                    slots = eq.to(torch.int64).argmax(dim=1)

                    dist = MaskedPlacementDist(logits, cand_mask_t)
                    log_probs = dist.log_prob(slots)
                    entropy = dist.entropy().mean()

                    # Distillation: KL(pi_target || pi_net) on searched rows only.
                    if sd_targets_t is not None:
                        log_probs_all = torch.log(dist.probs + 1e-9)
                        kl_sum, kl_n = masked_distill_kl(
                            sd_targets_t[mb_indices], log_probs_all, sd_mask_t[mb_indices]
                        )
                        sd_kl_loss = kl_sum / kl_n.clamp(min=1.0)
                        with torch.no_grad():
                            sd_kl_sum += kl_sum.detach()
                            sd_kl_n += kl_n.detach()
                    else:
                        sd_kl_loss = None
                else:
                    sd_kl_loss = None
                    # Forward pass
                    with self._autocast("update"):
                        logits_map, values = self.net(
                            mb_obs, mb_colors, mb_preview, mb_masks, aux=mb_aux
                        )
                    logits_map = logits_map.float()
                    values = values.float()

                    # Compute log probs and entropy
                    dist = MaskedPlacementDist(logits_map, mb_masks)
                    log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                # PPO policy loss
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.hparams.clip_epsilon,
                        1.0 + self.hparams.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (searched decisions optionally blend in the
                # search-backup value; identical to mb_returns when disabled).
                mb_value_targets = mb_returns
                if sd_values_t is not None and sd_value_mix > 0.0:
                    mb_value_targets = blend_value_targets(
                        mb_returns,
                        sd_values_t[mb_indices],
                        sd_mask_t[mb_indices],
                        sd_value_mix,
                    )
                if value_aux is not None:
                    value_loss = self.net.distributional_value_loss(  # type: ignore[attr-defined]
                        value_aux["value_logits"], mb_value_targets
                    )
                elif self.hparams.value_loss_type == "huber":
                    value_loss = F.huber_loss(values.squeeze(-1), mb_value_targets)
                else:
                    value_loss = F.mse_loss(values.squeeze(-1), mb_value_targets)

                # Total loss
                loss = policy_loss + self.hparams.value_coef * value_loss - entropy_coef * entropy
                if sd_kl_loss is not None:
                    loss = loss + sd_beta * sd_kl_loss

                # Optimize
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    self.hparams.max_grad_norm,
                )
                self.optimizer.step()
                with torch.no_grad():
                    ema_tensors: List[torch.Tensor] = []
                    model_tensors: List[torch.Tensor] = []
                    for k, v in self.net.state_dict().items():
                        e = self._ema_state[k]
                        if v.dtype.is_floating_point:
                            ema_tensors.append(e)
                            model_tensors.append(v)
                        else:
                            e.copy_(v)
                    if ema_tensors:
                        self._ema_update(ema_tensors, model_tensors, None)

                # Track metrics on-device; a `.item()` per metric per minibatch
                # forces an accelerator sync each time (dominated profiles).
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.hparams.clip_epsilon).float().mean()
                    # Approximate KL(old || new) under the sampled actions.
                    # Use the mini-batch's old log-probs to match `ratio` and avoid shape mismatch.
                    kl = (mb_log_probs_old - log_probs).mean()
                    metric_rows.append(
                        torch.stack(
                            [
                                policy_loss.detach(),
                                value_loss.detach(),
                                loss.detach(),
                                entropy.detach(),
                                kl,
                                clip_frac,
                            ]
                        )
                    )

        # Average metrics with a single host sync.
        if metric_rows:
            stacked = torch.stack(metric_rows).mean(dim=0).cpu()
            for idx, key in enumerate(_UPDATE_METRIC_KEYS):
                metrics_accum[key] = float(stacked[idx])

        if sd_kl_sum is not None:
            # On-device accumulation; one host sync for the whole update.
            metrics_accum["search_distill/kl_target_net"] = float(
                (sd_kl_sum / sd_kl_n.clamp(min=1.0)).cpu()
            )

        metrics_accum["optim/lr"] = self.optimizer.param_groups[0]["lr"]
        metrics_accum["optim/entropy_coef"] = entropy_coef

        return metrics_accum

    # --------------------------------------------------------------- utilities
    def _extract_mask(self, info: Dict) -> np.ndarray:
        """Extract action mask from info dict."""
        for key in ("placements/feasible_mask", "placements/legal_mask", "mask"):
            mask = info.get(key)
            if mask is not None:
                if isinstance(mask, np.ndarray) and mask.shape == (4, 16, 8):
                    return mask.astype(bool)
        # Fallback: all valid
        return np.ones((4, 16, 8), dtype=bool)

    def _extract_cost_to_lock(self, info: Dict[str, Any]) -> np.ndarray:
        """Extract frames-to-lock cost array from info dict.

        Supported keys:
          - cpp-pool backend: `placements/cost_to_lock` (uint16, 0xFFFF sentinel)
          - python/libretro wrapper: `placements/costs` (float, inf for unreachable)
        """

        value = info.get("placements/cost_to_lock")
        if value is None:
            value = info.get("placements/costs")
        if value is None:
            return np.full((4, 16, 8), np.inf, dtype=np.float32)
        try:
            arr = np.asarray(value)
        except Exception:
            return np.full((4, 16, 8), np.inf, dtype=np.float32)
        if arr.shape != (4, 16, 8):
            return np.full((4, 16, 8), np.inf, dtype=np.float32)
        if arr.dtype == np.uint16:
            out = arr.astype(np.float32)
            out[out >= np.float32(0xFFFE)] = np.inf
            return out
        return arr.astype(np.float32, copy=False)

    def _unwrap_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs.get("obs")
        return np.asarray(obs)

    def _ensure_batched_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == len(self.buffer.obs_shape):
            return obs_arr[None, ...]
        return obs_arr

    def _normalize_infos(self, infos: Any) -> List[Dict[str, Any]]:
        num_envs = int(self.env.num_envs)
        if infos is None:
            return [{} for _ in range(num_envs)]
        if isinstance(infos, (list, tuple)):
            out = [dict(i) if isinstance(i, dict) else {} for i in infos]
        elif isinstance(infos, dict):
            out = [dict(infos) for _ in range(num_envs)]
        else:
            out = [{} for _ in range(num_envs)]
        if len(out) < num_envs:
            out.extend({} for _ in range(num_envs - len(out)))
        return out[:num_envs]

    @staticmethod
    def _extract_tau(info: Dict[str, Any]) -> int:
        tau = info.get("placements/tau", 1)
        if isinstance(tau, np.ndarray):
            try:
                tau = tau.item()
            except Exception:
                return 1
        try:
            return max(1, int(tau))
        except Exception:
            return 1

    def _extract_pill_colors(self, info: Dict) -> np.ndarray:
        """Extract current pill colors (canonical indices 0=R,1=Y,2=B) from info dict."""
        colors = info.get("next_pill_colors")
        if colors is not None:
            arr = np.asarray(colors, dtype=np.int64)
            if arr.shape == (2,):
                return arr
        # Fallback: [0, 0]
        return np.array([0, 0], dtype=np.int64)

    def _extract_preview_pill_colors(self, info: Dict) -> np.ndarray:
        """Extract preview pill colors (canonical indices 0=R,1=Y,2=B) from info dict."""

        raw_left: Optional[int] = None
        raw_right: Optional[int] = None

        raw_ram = info.get("raw_ram")
        try:
            if isinstance(raw_ram, (bytes, bytearray, memoryview)) and len(raw_ram) > 0x031B:
                raw_left = int(raw_ram[0x031A]) & 0x03
                raw_right = int(raw_ram[0x031B]) & 0x03
        except Exception:
            raw_left = None
            raw_right = None

        if raw_left is None or raw_right is None:
            preview = info.get("preview_pill")
            if isinstance(preview, dict):
                try:
                    raw_left = int(preview.get("first_color", 0)) & 0x03
                    raw_right = int(preview.get("second_color", 0)) & 0x03
                except Exception:
                    raw_left = None
                    raw_right = None
            elif isinstance(preview, (list, tuple)) and len(preview) >= 2:
                try:
                    raw_left = int(preview[0]) & 0x03
                    raw_right = int(preview[1]) & 0x03
                except Exception:
                    raw_left = None
                    raw_right = None

        if raw_left is None or raw_right is None:
            return np.array([0, 0], dtype=np.int64)

        def _map_color(raw: int) -> int:
            return int(ram_specs.COLOR_VALUE_TO_INDEX.get(int(raw) & 0x03, 0))

        return np.array([_map_color(raw_left), _map_color(raw_right)], dtype=np.int64)

    @staticmethod
    def _extract_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            try:
                value = value.item()
            except Exception:
                return None
        try:
            return int(value)
        except Exception:
            return None

    def _build_aux(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.aux_spec in {_AUX_SPEC_V1, _AUX_SPEC_V1_VS}:
            return self._build_aux_v1(obs, info)
        raise ValueError(f"aux_spec={self.aux_spec!r} does not define auxiliary inputs")

    def _fill_aux_vs(self, out_row: np.ndarray, k: int, info: Dict[str, Any]) -> int:
        """Append the v1_vs opponent scalars (see `_AUX_V1_VS_EXTRA` layout)."""

        opp_viruses = self._extract_int(info.get("vs/opponent_viruses_remaining")) or 0
        out_row[k] = float(np.clip(float(opp_viruses) / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0))
        k += 1
        gp_self = self._extract_int(info.get("vs/garbage_pending")) or 0
        out_row[k] = float(np.clip(float(gp_self) / float(_AUX_GARBAGE_PENDING_NORM), 0.0, 1.0))
        k += 1
        gp_opp = self._extract_int(info.get("vs/garbage_pending_opp")) or 0
        out_row[k] = float(np.clip(float(gp_opp) / float(_AUX_GARBAGE_PENDING_NORM), 0.0, 1.0))
        k += 1
        for key in ("vs/opponent_pill_colors", "vs/opponent_preview_colors"):
            colors = np.asarray(info.get(key, ()), dtype=np.int64).reshape(-1)
            for j in range(2):
                c = int(colors[j]) if j < colors.shape[0] else -1
                if 0 <= c <= 2:
                    out_row[k + c] = 1.0
                k += 3
        return k

    def _build_aux_batch(self, obs_arr: np.ndarray, infos: List[Dict[str, Any]]) -> np.ndarray:
        """Batched aux_v1[_vs]: vectorizes the plane-derived features across envs.

        Must stay output-identical to per-env `_build_aux_v1` (covered by
        tests); scalar info lookups remain per-env, plane math is batched.
        """
        if self.aux_spec not in {_AUX_SPEC_V1, _AUX_SPEC_V1_VS}:
            raise ValueError(f"aux_spec={self.aux_spec!r} does not define auxiliary inputs")
        B = int(obs_arr.shape[0])
        frames = np.asarray(obs_arr, dtype=np.float32)
        if frames.ndim == 5 and frames.shape[-2:] == (16, 8):
            frames = frames[:, -1]
        if frames.ndim != 4 or frames.shape[2:] != (16, 8):
            raise ValueError(f"Expected obs shape (B,C,16,8), got {frames.shape!r}")

        out = np.zeros((B, self.aux_dim), dtype=np.float32)

        # Batched plane features. Keep this equivalent to ram_to_state's
        # single-frame helpers without crossing Python once per environment.
        idx = ram_specs.STATE_IDX
        if ram_specs.STATE_USE_BITPLANES:
            colors = frames[:, list(idx.color_channels)]
            virus_mask = frames[:, int(idx.virus_mask)] > 0.5
            falling_idx = getattr(idx, "falling_mask", None)
            falling = (
                frames[:, int(falling_idx)] > 0.5
                if falling_idx is not None
                else np.zeros_like(virus_mask)
            )
            preview_idx = getattr(idx, "preview_mask", None)
            preview = (
                frames[:, int(preview_idx)] > 0.5
                if preview_idx is not None
                else np.zeros_like(virus_mask)
            )
            locked_idx = getattr(idx, "locked_mask", None)
            if locked_idx is not None:
                locked = frames[:, int(locked_idx)] > 0.5
            else:
                locked = (colors > 0.5).any(axis=1) & ~virus_mask & ~falling & ~preview
            occ = locked | virus_mask | falling | preview
            virus_planes = colors * frames[:, int(idx.virus_mask), None]
        else:
            virus_planes = frames[:, list(idx.virus_color_channels)]
            virus_mask = (virus_planes > 0.1).any(axis=1)
            static = (frames[:, list(idx.static_color_channels)] > 0.1).any(axis=1)
            falling = (frames[:, list(idx.falling_color_channels)] > 0.1).any(axis=1)
            occ = static | virus_mask | falling
        virus_total = virus_mask.reshape(B, -1).sum(axis=1).astype(np.float32)
        virus_by_color = (virus_planes > 0.5).reshape(B, 3, -1).sum(axis=2).astype(np.float32)

        def _heights(masks_b: np.ndarray) -> np.ndarray:
            any_occ = masks_b.any(axis=1)  # (B, 8)
            first = masks_b.argmax(axis=1)  # (B, 8) first occupied row from top
            return np.where(any_occ, 16 - first, 0).astype(np.float32)

        heights = _heights(occ.astype(bool))
        virus_heights = _heights(virus_mask.astype(bool))

        for i in range(B):
            info = infos[i] if i < len(infos) else {}
            k = 0
            speed = self._extract_int(info.get("pill/speed_setting"))
            if speed is None:
                speed = self._extract_int(info.get("speed_setting"))
            if speed is None:
                speed = 2
            out[i, k + int(max(0, min(int(speed), 2)))] = 1.0
            k += 3

            out[i, k] = min(1.0, virus_total[i] / float(_AUX_V1_VIRUS_NORM))
            k += 1
            out[i, k : k + 3] = np.clip(virus_by_color[i] / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0)
            k += 3

            lvl = self._extract_int(info.get("curriculum/env_level"))
            if lvl is None:
                lvl = self._extract_int(info.get("curriculum_level"))
            if lvl is None:
                lvl = self._extract_int(info.get("level"))
            if lvl is None:
                lvl = 0
            lvl_i = int(lvl)
            if _AUX_V1_LEVEL_MIN <= lvl_i <= _AUX_V1_LEVEL_MAX:
                out[i, k + (lvl_i - _AUX_V1_LEVEL_MIN)] = 1.0
            k += _AUX_V1_LEVEL_DIM

            frames_used = self._extract_int(info.get("task/frames_used")) or 0
            max_frames = self._extract_int(info.get("task/max_frames"))
            if max_frames is not None and int(max_frames) > 0:
                out[i, k] = float(np.clip(float(frames_used) / float(max_frames), 0.0, 1.0))
            else:
                out[i, k] = float(np.tanh(float(frames_used) / 8000.0))
            k += 1

            out[i, k] = min(1.0, float(heights[i].max()) / 16.0)
            k += 1
            out[i, k : k + 8] = np.clip(heights[i] / 16.0, 0.0, 1.0)
            k += 8

            task_mode = str(info.get("task_mode") or "viruses").strip().lower()
            progress = 0.0
            if task_mode in {"matches", "any_clear"}:
                mc = self._extract_int(info.get("matches_completed")) or 0
                target = self._extract_int(info.get("match_target")) or 0
                if target > 0:
                    progress = float(mc) / float(max(1, target))
            else:
                v0 = self._extract_int(info.get("drm/viruses_initial"))
                if v0 is None:
                    v0 = self._extract_int(info.get("viruses_initial"))
                v_now = self._extract_int(info.get("viruses_remaining"))
                if v_now is None:
                    v_now = int(virus_total[i])
                if v0 is not None and int(v0) > 0:
                    progress = float(int(v0) - int(v_now)) / float(int(v0))
            out[i, k] = float(np.clip(progress, 0.0, 1.0))
            k += 1

            options = self._extract_int(info.get("placements/options")) or 0
            out[i, k] = float(np.clip(float(options) / 512.0, 0.0, 1.0))
            k += 1
            out[i, k] = float(np.clip(float(occ[i].sum()) / 128.0, 0.0, 1.0))
            k += 1
            out[i, k] = min(1.0, float(virus_heights[i].max()) / 16.0)
            k += 1
            if k != _AUX_V1_DIM:
                raise RuntimeError(f"aux_v1 packing mismatch: k={k} dim={_AUX_V1_DIM}")
            if self.aux_spec == _AUX_SPEC_V1_VS:
                k = self._fill_aux_vs(out[i], k, info)
            if k != self.aux_dim:
                raise RuntimeError(f"aux packing mismatch: k={k} dim={self.aux_dim}")
        return out

    @staticmethod
    def _column_heights(mask: np.ndarray) -> np.ndarray:
        occ = np.asarray(mask, dtype=bool)
        if occ.shape != (16, 8):
            raise ValueError(f"Expected mask shape (16,8), got {occ.shape!r}")
        heights = np.zeros((8,), dtype=np.int32)
        for c in range(8):
            rows = np.nonzero(occ[:, c])[0]
            if rows.size == 0:
                heights[c] = 0
            else:
                heights[c] = int(16 - int(rows.min()))
        return heights

    def _build_aux_v1(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.aux_dim != _AUX_DIM_BY_SPEC.get(self.aux_spec):
            raise ValueError(
                f"aux_dim mismatch: expected {_AUX_DIM_BY_SPEC.get(self.aux_spec)}, "
                f"got {self.aux_dim}"
            )
        frame = np.asarray(obs, dtype=np.float32)
        if frame.ndim == 4 and frame.shape[-2:] == (16, 8):
            # Allow passing a fixed frame stack (T,C,16,8); use the latest frame.
            frame = frame[-1]
        if frame.ndim != 3 or frame.shape[1:] != (16, 8):
            raise ValueError(f"Expected obs shape (C,16,8), got {frame.shape!r}")

        out = np.zeros((self.aux_dim,), dtype=np.float32)
        k = 0

        # speed_onehot[3]
        speed = self._extract_int(info.get("pill/speed_setting"))
        if speed is None:
            speed = self._extract_int(info.get("speed_setting"))
        if speed is None:
            speed = 2
        speed = int(max(0, min(int(speed), 2)))
        out[k + speed] = 1.0
        k += 3

        # Virus counts from the bottle planes.
        virus_mask = ram_specs.get_virus_mask(frame)
        virus_total = int(virus_mask.sum())
        out[k] = float(np.clip(float(virus_total) / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0))
        k += 1

        virus_planes = ram_specs.get_virus_color_planes(frame)
        if virus_planes.shape[0] != 3:
            raise ValueError(f"Expected 3 virus color planes, got {virus_planes.shape!r}")
        for c in range(3):
            out[k + c] = float(
                np.clip(float((virus_planes[c] > 0.5).sum()) / float(_AUX_V1_VIRUS_NORM), 0.0, 1.0)
            )
        k += 3

        # level_onehot[36] for [-15..20]
        lvl = self._extract_int(info.get("curriculum/env_level"))
        if lvl is None:
            lvl = self._extract_int(info.get("curriculum_level"))
        if lvl is None:
            lvl = self._extract_int(info.get("level"))
        if lvl is None:
            lvl = 0
        lvl_i = int(lvl)
        if _AUX_V1_LEVEL_MIN <= lvl_i <= _AUX_V1_LEVEL_MAX:
            out[k + (lvl_i - _AUX_V1_LEVEL_MIN)] = 1.0
        k += _AUX_V1_LEVEL_DIM

        # frame_count_norm (task timer; normalized to [0,1])
        frames_used = self._extract_int(info.get("task/frames_used"))
        if frames_used is None:
            frames_used = 0
        max_frames = self._extract_int(info.get("task/max_frames"))
        if max_frames is not None and int(max_frames) > 0:
            out[k] = float(np.clip(float(frames_used) / float(max_frames), 0.0, 1.0))
        else:
            out[k] = float(np.tanh(float(frames_used) / 8000.0))
        k += 1

        # heights from occupancy mask (bottle-only for bitplane_bottle*).
        occ = ram_specs.get_occupancy_mask(frame)
        heights = self._column_heights(occ)
        max_h = int(heights.max())
        out[k] = float(np.clip(float(max_h) / 16.0, 0.0, 1.0))
        k += 1

        out[k : k + 8] = np.clip(heights.astype(np.float32) / 16.0, 0.0, 1.0)
        k += 8

        # clearance_progress (matches or viruses)
        task_mode = str(info.get("task_mode") or "viruses").strip().lower()
        progress = 0.0
        if task_mode in {"matches", "any_clear"}:
            mc = self._extract_int(info.get("matches_completed")) or 0
            target = self._extract_int(info.get("match_target")) or 0
            if target > 0:
                progress = float(mc) / float(max(1, target))
        else:
            v0 = self._extract_int(info.get("drm/viruses_initial"))
            if v0 is None:
                v0 = self._extract_int(info.get("viruses_initial"))
            v_now = self._extract_int(info.get("viruses_remaining"))
            if v_now is None:
                v_now = virus_total
            if v0 is not None and int(v0) > 0:
                progress = float(int(v0) - int(v_now)) / float(int(v0))
        out[k] = float(np.clip(progress, 0.0, 1.0))
        k += 1

        # feasible_fraction (placements/options / 512)
        options = self._extract_int(info.get("placements/options"))
        if options is None:
            options = 0
        out[k] = float(np.clip(float(options) / 512.0, 0.0, 1.0))
        k += 1

        # occupancy_fraction (occupied / 128)
        out[k] = float(np.clip(float(occ.sum()) / 128.0, 0.0, 1.0))
        k += 1

        # virus_max_height/16
        virus_heights = self._column_heights(virus_mask)
        virus_max_h = int(virus_heights.max())
        out[k] = float(np.clip(float(virus_max_h) / 16.0, 0.0, 1.0))
        k += 1

        if k != _AUX_V1_DIM:
            raise RuntimeError(f"aux_v1 packing mismatch: k={k} dim={_AUX_V1_DIM}")
        if self.aux_spec == _AUX_SPEC_V1_VS:
            k = self._fill_aux_vs(out, k, info)
        if k != self.aux_dim:
            raise RuntimeError(f"aux packing mismatch: k={k} dim={self.aux_dim}")
        return out

    def _extract_curriculum_snapshot(self, infos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract a compact curriculum snapshot from per-env info dicts."""

        if not infos:
            return None

        source: Optional[Dict[str, Any]] = None
        for info in infos:
            if isinstance(info, dict) and "curriculum/current_level" in info:
                source = info
                break
        if source is None:
            return None

        current_level = self._extract_int(source.get("curriculum/current_level"))
        if current_level is None:
            return None

        def _float(key: str, default: float = 0.0) -> float:
            val = source.get(key, default)
            if isinstance(val, np.ndarray):
                try:
                    val = val.item()
                except Exception:
                    return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)

        def _int(key: str, default: int = 0) -> int:
            val = self._extract_int(source.get(key))
            return int(default) if val is None else int(val)

        snapshot: Dict[str, Any] = {
            "current_level": int(current_level),
            "rate_current": _float("curriculum/rate_current", 0.0),
            "window_n": _int("curriculum/window_n", 0),
            "window_size": _int("curriculum/window_size", 0),
            "episodes_current_total": _int("curriculum/episodes_current_total", 0),
            "start_level": _int("curriculum/start_level", 0),
            "max_level": _int("curriculum/max_level", 0),
            "success_threshold": _float("curriculum/success_threshold", 0.0),
            "min_episodes": _int("curriculum/min_episodes", 0),
            "rehearsal_prob": _float("curriculum/rehearsal_prob", 0.0),
        }

        confidence_sigmas = source.get("curriculum/confidence_sigmas")
        if confidence_sigmas is not None:
            try:
                snapshot["confidence_sigmas"] = float(confidence_sigmas)
            except Exception:
                pass
        confidence_lb = source.get("curriculum/confidence_lower_bound")
        if confidence_lb is not None:
            try:
                snapshot["confidence_lower_bound"] = float(confidence_lb)
            except Exception:
                pass
        window_successes = self._extract_int(source.get("curriculum/window_successes"))
        if window_successes is not None:
            snapshot["window_successes"] = int(window_successes)

        time_budget_frames = self._extract_int(source.get("curriculum/time_budget_frames"))
        if time_budget_frames is not None:
            snapshot["time_budget_frames"] = int(time_budget_frames)
        time_budget_spawns = self._extract_int(source.get("curriculum/time_budget_spawns"))
        if time_budget_spawns is not None:
            snapshot["time_budget_spawns"] = int(time_budget_spawns)
        time_mean_frames = source.get("curriculum/time_mean_frames")
        if time_mean_frames is not None:
            try:
                snapshot["time_mean_frames"] = float(time_mean_frames)
            except Exception:
                pass
        time_mad_frames = source.get("curriculum/time_mad_frames")
        if time_mad_frames is not None:
            try:
                snapshot["time_mad_frames"] = float(time_mad_frames)
            except Exception:
                pass
        time_mean_spawns = source.get("curriculum/time_mean_spawns")
        if time_mean_spawns is not None:
            try:
                snapshot["time_mean_spawns"] = float(time_mean_spawns)
            except Exception:
                pass
        time_mad_spawns = source.get("curriculum/time_mad_spawns")
        if time_mad_spawns is not None:
            try:
                snapshot["time_mad_spawns"] = float(time_mad_spawns)
            except Exception:
                pass
        time_k = self._extract_int(source.get("curriculum/time_k"))
        if time_k is not None:
            snapshot["time_k"] = int(time_k)
        time_target = source.get("curriculum/time_target")
        if time_target is not None:
            try:
                snapshot["time_target"] = float(time_target)
            except Exception:
                pass

        mode = source.get("curriculum/mode")
        if isinstance(mode, str) and mode:
            snapshot["mode"] = str(mode)

        stage_index = self._extract_int(source.get("curriculum/stage_index"))
        if stage_index is not None:
            snapshot["stage_index"] = int(stage_index)
        stage_count = self._extract_int(source.get("curriculum/stage_count"))
        if stage_count is not None:
            snapshot["stage_count"] = int(stage_count)

        probe_threshold = _float("curriculum/probe_threshold", 0.0)
        if probe_threshold > 0.0:
            snapshot["probe_threshold"] = float(probe_threshold)

        decisions_current_total = self._extract_int(
            source.get("curriculum/decisions_current_total")
        )
        if decisions_current_total is not None:
            snapshot["decisions_current_total"] = int(decisions_current_total)
        min_stage_decisions = self._extract_int(source.get("curriculum/min_stage_decisions"))
        if min_stage_decisions is not None:
            snapshot["min_stage_decisions"] = int(min_stage_decisions)

        # Distribution of active env levels.
        env_levels: List[int] = []
        for info in infos:
            if not isinstance(info, dict):
                continue
            lvl = self._extract_int(info.get("curriculum/env_level"))
            if lvl is not None:
                env_levels.append(int(lvl))
        if env_levels:
            counts = Counter(env_levels)
            snapshot["env_level_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}

        # Advancement (present only on steps that trigger it).
        adv_from = None
        adv_to = None
        for info in infos:
            if not isinstance(info, dict):
                continue
            adv_to = self._extract_int(info.get("curriculum/advanced_to"))
            if adv_to is None:
                continue
            adv_from = self._extract_int(info.get("curriculum/advanced_from"))
            break
        if adv_to is not None:
            snapshot["advanced_to"] = int(adv_to)
            if adv_from is not None:
                snapshot["advanced_from"] = int(adv_from)

        return snapshot

    @staticmethod
    def _curriculum_scalar_metrics(snapshot: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            out["curriculum/current_level"] = float(int(snapshot.get("current_level", 0)))
            out["curriculum/rate_current"] = float(snapshot.get("rate_current", 0.0) or 0.0)
            out["curriculum/window_n"] = float(int(snapshot.get("window_n", 0)))
            out["curriculum/window_size"] = float(int(snapshot.get("window_size", 0)))
            out["curriculum/episodes_current_total"] = float(
                int(snapshot.get("episodes_current_total", 0))
            )
            decisions_total = snapshot.get("decisions_current_total")
            if decisions_total is not None:
                out["curriculum/decisions_current_total"] = float(int(decisions_total))
            min_stage_decisions = snapshot.get("min_stage_decisions")
            if min_stage_decisions is not None:
                out["curriculum/min_stage_decisions"] = float(int(min_stage_decisions))
            out["curriculum/start_level"] = float(int(snapshot.get("start_level", 0)))
            out["curriculum/max_level"] = float(int(snapshot.get("max_level", 0)))
            out["curriculum/success_threshold"] = float(
                snapshot.get("success_threshold", 0.0) or 0.0
            )
            out["curriculum/min_episodes"] = float(int(snapshot.get("min_episodes", 0)))
            out["curriculum/rehearsal_prob"] = float(snapshot.get("rehearsal_prob", 0.0) or 0.0)
            stage_index = snapshot.get("stage_index")
            if stage_index is not None:
                out["curriculum/stage_index"] = float(int(stage_index))
            conf_sigmas = snapshot.get("confidence_sigmas")
            if conf_sigmas is not None:
                out["curriculum/confidence_sigmas"] = float(conf_sigmas)
            conf_lb = snapshot.get("confidence_lower_bound")
            if conf_lb is not None:
                out["curriculum/confidence_lower_bound"] = float(conf_lb)
            window_successes = snapshot.get("window_successes")
            if window_successes is not None:
                out["curriculum/window_successes"] = float(int(window_successes))
            time_budget_frames = snapshot.get("time_budget_frames")
            if time_budget_frames is not None:
                out["curriculum/time_budget_frames"] = float(int(time_budget_frames))
            time_budget_spawns = snapshot.get("time_budget_spawns")
            if time_budget_spawns is not None:
                out["curriculum/time_budget_spawns"] = float(int(time_budget_spawns))
            time_mean = snapshot.get("time_mean_frames")
            if time_mean is not None:
                out["curriculum/time_mean_frames"] = float(time_mean)
            time_mad = snapshot.get("time_mad_frames")
            if time_mad is not None:
                out["curriculum/time_mad_frames"] = float(time_mad)
            time_mean_spawns = snapshot.get("time_mean_spawns")
            if time_mean_spawns is not None:
                out["curriculum/time_mean_spawns"] = float(time_mean_spawns)
            time_mad_spawns = snapshot.get("time_mad_spawns")
            if time_mad_spawns is not None:
                out["curriculum/time_mad_spawns"] = float(time_mad_spawns)
            time_k = snapshot.get("time_k")
            if time_k is not None:
                out["curriculum/time_k"] = float(int(time_k))
            time_target = snapshot.get("time_target")
            if time_target is not None:
                out["curriculum/time_target"] = float(time_target)
        except Exception:
            return out

        counts = snapshot.get("env_level_counts")
        if isinstance(counts, dict) and counts:
            try:
                levels = [int(k) for k in counts.keys()]
                out["curriculum/env_level_min"] = float(min(levels))
                out["curriculum/env_level_max"] = float(max(levels))
                out["curriculum/env_levels_unique"] = float(len(levels))
            except Exception:
                pass
        return out

    def _log_curriculum_advance(self, level_from: Optional[int], level_to: int) -> None:
        if self._curriculum_last_level is not None and int(level_to) <= int(
            self._curriculum_last_level
        ):
            return
        frames_total = int(self.global_step)
        episodes_total = int(self._episodes_total)
        frames_delta = frames_total - int(self._curriculum_last_frames)
        episodes_delta = episodes_total - int(self._curriculum_last_episodes)

        step = int(self.global_step)
        if level_from is not None:
            self.logger.log_scalar("curriculum/advanced_from", float(level_from), step)
        self.logger.log_scalar("curriculum/advanced_to", float(level_to), step)
        self.logger.log_scalar("curriculum/advanced_frames_total", float(frames_total), step)
        self.logger.log_scalar("curriculum/advanced_episodes_total", float(episodes_total), step)
        self.logger.log_scalar("curriculum/advanced_frames_delta", float(frames_delta), step)
        self.logger.log_scalar("curriculum/advanced_episodes_delta", float(episodes_delta), step)

        self._curriculum_last_level = int(level_to)
        self._curriculum_last_frames = frames_total
        self._curriculum_last_episodes = episodes_total

    def _get_entropy_coef(self, *, step: Optional[int] = None) -> float:
        """Get current entropy coefficient (annealed over training)."""
        schedule_step = self.global_step if step is None else int(step)
        progress = min(1.0, schedule_step / self.hparams.entropy_schedule_steps)
        return (
            self._entropy_coef_initial
            + (self.hparams.entropy_schedule_end - self._entropy_coef_initial) * progress
        )

    def _seed_ema_from_net(self) -> None:
        if hasattr(self, "_ema_state"):
            self._ema_state = {
                k: v.detach().clone() for k, v in self.net.state_dict().items()
            }

    def _load_checkpoint(
        self,
        path: Path,
        *,
        resume_optimizer: bool,
        resume_step: bool,
        strict: bool,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            payload = load_checkpoint(path, map_location=self.device)
        except Exception as exc:
            msg = f"Failed to load checkpoint {path}: {exc}"
            if "PytorchStreamReader" in str(exc) or "central directory" in str(exc):
                msg = f"{msg}. The checkpoint may be incomplete or corrupted; try an earlier file."
            raise RuntimeError(msg) from exc
        state_dict = payload.get("state_dict") or payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint missing state_dict")
        self.net.load_state_dict(state_dict, strict=bool(strict))
        ema_sd = payload.get("ema_state_dict")
        if ema_sd is not None and hasattr(self, "_ema_state"):
            self._ema_state = {k: v.detach().clone().to(self.device) for k, v in ema_sd.items()}
        else:
            self._seed_ema_from_net()
        if resume_optimizer:
            opt_state = payload.get("optimizer")
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
        if resume_step:
            step = int(payload.get("step", self.global_step) or 0)
            decision_step = int(payload.get("decision_step", self.decision_step) or 0)
            if step > 0:
                self.global_step = step
            if decision_step > 0:
                self.decision_step = decision_step
            self._last_update_step = int(self.global_step)
            if self.checkpoint_interval > 0:
                self._next_checkpoint = (
                    (int(self.global_step) // int(self.checkpoint_interval)) + 1
                ) * int(self.checkpoint_interval)

    def _log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        """Log metrics to logger."""
        step = self.global_step if step is None else int(step)
        values: Dict[str, float] = dict(metrics)

        if self.batch_returns:
            returns = np.array(self.batch_returns, dtype=np.float32)
            values["train/return_mean"] = float(returns.mean())
            values["train/return_std"] = float(returns.std())

        if self.batch_viruses:
            viruses = np.array(self.batch_viruses, dtype=np.float32)
            values["drm/viruses_per_ep"] = float(viruses.mean())

        # VS self-play metrics + skill grading (vs vec env only).
        get_vs_metrics = getattr(self.env, "get_vs_metrics", None)
        if callable(get_vs_metrics):
            try:
                for key, value in get_vs_metrics().items():
                    values[str(key)] = float(value)
            except Exception:
                pass
            self.logger.log_scalars(values, step)
            self._maybe_grade_vs_skill(step)
            # Opponent-pool snapshots (vs env with opponent_pool enabled):
            # freeze the EMA weights into the pool every N learner matches.
            maybe_snapshot = getattr(self.env, "maybe_snapshot", None)
            if callable(maybe_snapshot):
                try:
                    maybe_snapshot(
                        lambda: {k: v.detach().cpu().clone() for k, v in self._ema_state.items()},
                        cfg=getattr(self.cfg, "to_dict", lambda: {})(),
                        step=int(step),
                    )
                except Exception:
                    pass
        else:
            self.logger.log_scalars(values, step)

    # Grade a batch of completed VS matches every N matches (2 per-side
    # samples per match) and append to <logdir>/skill_history.jsonl.
    _VS_SKILL_GRADE_MATCHES = 50

    def _maybe_grade_vs_skill(self, step: int) -> None:
        pop_games = getattr(self.env, "pop_skill_games", None)
        pending = getattr(self.env, "skill_games_pending", None)
        if not callable(pop_games) or not callable(pending):
            return
        try:
            if int(pending()) < 2 * self._VS_SKILL_GRADE_MATCHES:
                return
            games = list(pop_games())
        except Exception:
            return
        if not games:
            return

        import json

        try:
            import tools.skill_grade as skill_grade

            model_path = Path(skill_grade.DEFAULT_MODEL)
            if not model_path.is_file():
                return
            model = json.loads(model_path.read_text())

            X = np.asarray(
                [[float(g[name]) for name in skill_grade.BASE_FEATURES] for g in games],
                dtype=float,
            )
            # Self-play: the opponent is the agent itself, so solve the rating
            # fixed point r = f(metrics, opp_whr=r). Falls back to the plain
            # mean prediction for version-1 (no opp_whr) model files.
            whr, _converged, _iters = skill_grade.self_play_rating(model, X)

            n_matches = len(games) // 2
            row: Dict[str, float] = {
                "step": int(step),
                "whr": float(whr),
                "whr_std": float(model["resid_std"]) / float(max(1, len(games))) ** 0.5,
                "n_games": int(n_matches),
                "n_samples": int(len(games)),
                "win_rate": float(np.mean([g.get("won", 0.0) for g in games])),
            }
            for j, name in enumerate(skill_grade.BASE_FEATURES):
                row[name] = float(np.mean(X[:, j]))
            # Dashboard-friendly alias (tools/vs_dashboard.py shows "salt").
            row["salt"] = row["salt_per_min"]

            logdir = Path(str(getattr(self.cfg, "logdir", "runs/smdp_ppo")))
            logdir.mkdir(parents=True, exist_ok=True)
            with (logdir / "skill_history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            return

    def _maybe_checkpoint(
        self, *, step: Optional[int] = None, decision_step: Optional[int] = None
    ) -> None:
        """Save checkpoint if interval reached."""
        checkpoint_step = self.global_step if step is None else int(step)
        checkpoint_decision_step = (
            self.decision_step if decision_step is None else int(decision_step)
        )
        if checkpoint_step < self._next_checkpoint:
            return

        payload = {
            "state_dict": self.net.state_dict(),
            "ema_state_dict": self._ema_state,
            "optimizer": self.optimizer.state_dict(),
            "cfg": getattr(self.cfg, "to_dict", lambda: {})(),
            "step": checkpoint_step,
            "decision_step": checkpoint_decision_step,
            "sha": git_commit(),
        }

        path = checkpoint_path(
            self.checkpoint_dir, "smdp_ppo", checkpoint_step, compress=True
        )
        save_checkpoint(payload, path)

        self.event_bus.emit(
            "checkpoint", step=checkpoint_step, path=str(path), walltime=time.time()
        )
        self._next_checkpoint += self.checkpoint_interval
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        """Keep only the newest `checkpoint_keep_last` step checkpoints of this run.

        Only touches smdp_ppo_step<N>.pt(.gz) files inside this run's own
        checkpoints/ dir; anything else there (renamed/gate-best copies) is
        left alone. Opponent-pool snapshots live elsewhere and hold their own
        copies, so pruning never invalidates the pool manifest.
        """
        if self.checkpoint_keep_last <= 0:
            return
        pat = re.compile(r"^smdp_ppo_step(\d+)\.pt(\.gz)?$")
        found = []
        for p in self.checkpoint_dir.iterdir():
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort()
        for _, p in found[: max(0, len(found) - self.checkpoint_keep_last)]:
            try:
                p.unlink()
            except OSError:
                pass


__all__ = ["SMDPPPOAdapter", "SMDPPPOConfig"]
