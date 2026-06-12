"""Inference-time policy-guided truncated expectimax for placement decisions.

Design: docs/SEARCH_DESIGN.md. Depth 2, beam K: ply-1 = current-pill
placements, ply-2 = preview-pill placements (fully known). Backup is
reward-augmented and training-consistent: Q = r̂1 + γ^τ1·(r̂2 + γ^τ2·V(leaf)),
with the training reward replicated from sim event counters and γ from the
checkpoint cfg. The unknown pill after the preview is neutralized by
analytically marginalizing the leaf value over the 81 (pill, preview)
canonical color pairs — decide() is exactly deterministic despite random sim
seeds. Anytime: the policy argmax is the instant fallback; a deadline is
enforced between stages and the best result at the deepest completed stage
commits.

Simulation primitive: the 1P pool's checkpoint reset
(`envs/backends/drmario_pool.build_reset_spec(checkpoint_enabled=True, ...)`)
with `inject_plan` round-trips so no board is ever BFS-planned twice: root
reset injects the caller's plan, ply-1 runs on one representative env per
branch, and a second checkpoint reset fans each surviving branch out to a
block of envs (injecting the branch's just-computed plan) for the ply-2 step
and leaf evaluation.

Color conventions: `decide()` takes **raw NES** colors (0=Y,1=R,2=B) for the
falling and preview pills, matching the live bridge and the checkpoint reset
spec; the network sees canonical indices (0=R,1=Y,2=B) internally.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import envs.specs.ram_to_state as ram_specs
from envs.backends.drmario_pool import (
    GRID_H,
    GRID_W,
    MACRO_ACTIONS,
    DrMarioPoolRunner,
    build_reset_spec,
)
from models.policy.candidate_packing import pack_feasible_candidates

# Raw NES color (0=Y,1=R,2=B) <-> canonical index (0=R,1=Y,2=B): an involution.
_COLOR_SWAP = (1, 0, 2)

# DrmPoolTerminalReason values.
_TERMINAL_CLEAR = 1
_TERMINAL_TOPOUT = 2

# Per-block status codes.
_BLOCK_ALIVE = 0
_BLOCK_TERMINAL = 1  # ply-1 placement ended the game (value already final)
_BLOCK_INVALID = 2  # sim rejected the ply-1 action (spawn-planner mismatch)


def backup_block_values(
    block_status: np.ndarray,
    block_values: np.ndarray,
    leaf_values: Sequence[Sequence[float]],
    *,
    r1: Optional[np.ndarray] = None,
    disc1: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pure backup: Q(a1) = r1 + disc1 * max over the block's ply-2 leaf Qs.

    `leaf_values[i]` holds the ply-2 contributions (r2 + disc2*V or terminal
    values) for block i. With `r1`/`disc1` omitted the backup is a plain max.
    Blocks with no evaluated leaves keep their current (depth-1 or terminal)
    value; `_BLOCK_INVALID` blocks keep -inf.
    """

    out = np.asarray(block_values, dtype=np.float64).copy()
    for i, leaves in enumerate(leaf_values):
        if int(block_status[i]) != _BLOCK_ALIVE:
            continue
        vals = [float(v) for v in leaves]
        if vals:
            best = max(vals)
            if r1 is not None:
                d = 1.0 if disc1 is None else float(disc1[i])
                out[i] = float(r1[i]) + d * best
            else:
                out[i] = best
    out[np.asarray(block_status) == _BLOCK_INVALID] = -np.inf
    return out


def aux_v1_batch_fast(obs: np.ndarray, infos: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Vectorized aux-v1 for the bitplane_bottle_conn_mask layout.

    Output-identical to `SMDPPPOAdapter._build_aux_batch` for (B,12,16,8) obs
    (pinned by tests/test_search_policy.py); avoids its per-frame Python loops
    on the hot leaf-evaluation path.
    """

    obs = np.asarray(obs, dtype=np.float32)
    B = int(obs.shape[0])
    out = np.zeros((B, 57), dtype=np.float32)

    colors = obs[:, :3] > 0.5  # (B,3,16,8); viruses carry color planes too
    virus = obs[:, 3] > 0.5  # (B,16,8)
    occ = colors.any(axis=1)  # == static|virus (no falling/preview planes)
    virus_total = virus.reshape(B, -1).sum(axis=1).astype(np.float32)
    virus_by_color = (colors & virus[:, None]).reshape(B, 3, -1).sum(axis=2).astype(np.float32)

    def _heights(m: np.ndarray) -> np.ndarray:
        any_occ = m.any(axis=1)  # (B,8)
        first = m.argmax(axis=1)  # first occupied row from top
        return np.where(any_occ, 16 - first, 0).astype(np.float32)

    heights = _heights(occ)
    virus_heights = _heights(virus)

    for i in range(B):
        info = infos[i] if i < len(infos) else {}
        k = 0
        speed = info.get("pill/speed_setting", info.get("speed_setting", 2))
        out[i, k + int(max(0, min(int(speed), 2)))] = 1.0
        k += 3
        out[i, k] = min(1.0, virus_total[i] / 84.0)
        k += 1
        out[i, k : k + 3] = np.clip(virus_by_color[i] / 84.0, 0.0, 1.0)
        k += 3
        lvl = int(info.get("level", 0))
        if -15 <= lvl <= 20:
            out[i, k + (lvl + 15)] = 1.0
        k += 36
        frames_used = int(info.get("task/frames_used", 0) or 0)
        max_frames = info.get("task/max_frames")
        if max_frames is not None and int(max_frames) > 0:
            out[i, k] = float(np.clip(float(frames_used) / float(max_frames), 0.0, 1.0))
        else:
            out[i, k] = float(np.tanh(float(frames_used) / 8000.0))
        k += 1
        out[i, k] = min(1.0, float(heights[i].max()) / 16.0)
        k += 1
        out[i, k : k + 8] = np.clip(heights[i] / 16.0, 0.0, 1.0)
        k += 8
        # task_mode is always "viruses" for sim states.
        progress = 0.0
        v0 = info.get("drm/viruses_initial")
        v_now = info.get("viruses_remaining")
        if v_now is None:
            v_now = int(virus_total[i])
        if v0 is not None and int(v0) > 0:
            progress = float(int(v0) - int(v_now)) / float(int(v0))
        out[i, k] = float(np.clip(progress, 0.0, 1.0))
        k += 1
        options = int(info.get("placements/options", 0) or 0)
        out[i, k] = float(np.clip(float(options) / 512.0, 0.0, 1.0))
        k += 1
        out[i, k] = float(np.clip(float(occ[i].sum()) / 128.0, 0.0, 1.0))
        k += 1
        out[i, k] = min(1.0, float(virus_heights[i].max()) / 16.0)
    return out


def _build_obs_from_board(board: np.ndarray, feasible_512: np.ndarray) -> np.ndarray:
    """12-channel bitplane_bottle_conn_mask obs (mirrors DrMarioPool::build_obs)."""

    field = board.reshape(GRID_H, GRID_W)
    obs = np.zeros((12, GRID_H, GRID_W), dtype=np.float32)
    type_hi = field & 0xF0
    color_lo = field & 0x03
    is_empty = field == 0xFF
    is_zero = field == 0x00
    just_emptied = (type_hi == 0xF0) & ~is_empty
    clearing = (type_hi == 0xB0) | just_emptied
    color_valid = ~(is_empty | is_zero | clearing)
    for raw_value, ch in ((1, 0), (0, 1), (2, 2)):  # canonical R, Y, B planes
        obs[ch] = ((color_lo == raw_value) & color_valid).astype(np.float32)
    obs[3] = (type_hi == 0xD0).astype(np.float32)  # virus
    for code, ch in ((0x50, 4), (0x40, 5), (0x70, 6), (0x60, 7)):  # conn up/down/left/right
        obs[ch] = (type_hi == code).astype(np.float32)
    obs[8:12] = feasible_512.reshape(4, GRID_H, GRID_W).astype(np.float32)
    return obs


class SearchPolicy:
    """Depth-2 beam search over the native pool engine, guided by a policy net."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        beam: int = 8,
        deadline_ms: float = 60.0,
        device: str = "cpu",
        num_sim_envs: int = 64,
        lib_path: Optional[str] = None,
        win_value: float = 8.0,
        loss_value: float = -8.0,
        depth_penalty: float = 0.01,
        reward_mode: Optional[str] = None,
        gamma: Optional[float] = None,
        garbage_reward_coef: float = 0.05,
        seed: int = 0,
        warmup: bool = True,
    ) -> None:
        from tools.eval_policy import _build_net_from_cfg, _make_aux_builder
        from training.utils.checkpoint_io import load_checkpoint

        path = Path(checkpoint_path)
        payload = load_checkpoint(path, map_location="cpu")
        cfg = payload.get("cfg", {})
        net, aux_dim, candidate_max = _build_net_from_cfg(cfg, 12, "cpu")
        net.load_state_dict(payload.get("ema_state_dict") or payload["state_dict"])
        self.checkpoint_name = path.name
        self.checkpoint_step = payload.get("step")
        if reward_mode is None:
            env_cfg = cfg.get("env", {}) if isinstance(cfg.get("env"), dict) else {}
            tag = f"{env_cfg.get('backend', '')} {env_cfg.get('id', '')}".lower()
            reward_mode = "vs" if "vs" in tag else "1p"
        if gamma is None:
            sp = cfg.get("smdp_ppo", cfg) if isinstance(cfg, dict) else {}
            gamma = float(sp.get("gamma", 1.0))
        self._init_common(
            net=net,
            aux_shim=_make_aux_builder(aux_dim),
            candidate_max=int(candidate_max),
            beam=beam,
            deadline_ms=deadline_ms,
            device=device,
            num_sim_envs=num_sim_envs,
            lib_path=lib_path,
            win_value=win_value,
            loss_value=loss_value,
            depth_penalty=depth_penalty,
            reward_mode=str(reward_mode),
            gamma=float(gamma),
            garbage_reward_coef=float(garbage_reward_coef),
            seed=seed,
            warmup=warmup,
        )

    @classmethod
    def from_net(
        cls,
        net: Any,
        *,
        aux_shim: Any = None,
        candidate_max: int = 128,
        beam: int = 8,
        deadline_ms: float = 60.0,
        device: str = "cpu",
        num_sim_envs: int = 64,
        lib_path: Optional[str] = None,
        win_value: float = 8.0,
        loss_value: float = -8.0,
        depth_penalty: float = 0.01,
        reward_mode: str = "1p",
        gamma: float = 1.0,
        garbage_reward_coef: float = 0.05,
        seed: int = 0,
        warmup: bool = False,
    ) -> "SearchPolicy":
        """Build from an in-memory net (tests / callers that loaded their own)."""

        self = cls.__new__(cls)
        self.checkpoint_name = None
        self.checkpoint_step = None
        self._init_common(
            net=net,
            aux_shim=aux_shim,
            candidate_max=int(candidate_max),
            beam=beam,
            deadline_ms=deadline_ms,
            device=device,
            num_sim_envs=num_sim_envs,
            lib_path=lib_path,
            win_value=win_value,
            loss_value=loss_value,
            depth_penalty=depth_penalty,
            reward_mode=reward_mode,
            gamma=gamma,
            garbage_reward_coef=garbage_reward_coef,
            seed=seed,
            warmup=warmup,
        )
        return self

    def _init_common(
        self,
        *,
        net: Any,
        aux_shim: Any,
        candidate_max: int,
        beam: int,
        deadline_ms: float,
        device: str,
        num_sim_envs: int,
        lib_path: Optional[str],
        win_value: float,
        loss_value: float,
        depth_penalty: float,
        reward_mode: str,
        gamma: float,
        garbage_reward_coef: float,
        seed: int,
        warmup: bool,
    ) -> None:
        ram_specs.set_state_representation("bitplane_bottle_conn_mask")
        # Two net copies: the small full forwards (root B=1, ply-2 priors B<=K)
        # run on CPU where dispatch latency is lowest; only the big marginal
        # leaf batch runs on `device` (one transfer in, one sync out).
        self.net = net.to("cpu").eval()
        if str(device) != "cpu":
            import copy as _copy

            self.net_dev = _copy.deepcopy(self.net).to(device).eval()
        else:
            self.net_dev = self.net
        self.aux_shim = aux_shim
        self.candidate_max = int(max(1, candidate_max))
        self.beam = int(max(1, beam))
        self.deadline_ms = float(deadline_ms)
        self.device = str(device)
        self.num_sim_envs = int(max(self.beam, num_sim_envs))
        self.win_value = float(win_value)
        self.loss_value = float(loss_value)
        self.depth_penalty = float(depth_penalty)
        reward_mode = str(reward_mode).strip().lower()
        if reward_mode not in {"1p", "vs"}:
            raise ValueError(f"reward_mode must be '1p' or 'vs', got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.gamma = float(gamma)
        self.garbage_reward_coef = float(garbage_reward_coef)
        if self.reward_mode == "1p":
            from training.envs.drmario_pool_vec import _RewardCfg

            self._reward_cfg, _ = _RewardCfg.load()
        else:
            self._reward_cfg = None
        self._rng = np.random.default_rng(seed)

        self._runner = DrMarioPoolRunner(
            num_envs=self.num_sim_envs,
            obs_spec=4,  # DRM_POOL_OBS_BITPLANE_BOTTLE_CONN_MASK
            obs_channels=12,
            lib_path=lib_path,
            emit_board=True,
        )

        # Precompute the 81 (pill, preview) canonical color-pair conditioning
        # embeddings used by the marginalized leaf value (docs/SEARCH_DESIGN.md).
        combos9 = np.array([(a, b) for a in range(3) for b in range(3)], dtype=np.int64)
        pill_idx = np.repeat(np.arange(9), 9)
        prev_idx = np.tile(np.arange(9), 9)
        self._combo_pills = combos9[pill_idx]  # [81, 2] canonical pill colors
        self._combo_prevs = combos9[prev_idx]  # [81, 2] canonical preview colors
        same = combos9[pill_idx][:, 0] == combos9[pill_idx][:, 1]
        self._combo_same = torch.from_numpy(same).to(self.device)  # [81] bool
        self._refresh_combo_embeddings()

        # EMA of marginal-value cost per leaf (ms), seeded by warmup; used to
        # decide whether the leaf stage fits the remaining deadline in one batch.
        self._leaf_ms_per_env = 1.0

        if warmup:
            self._warmup()

    def _refresh_combo_embeddings(self) -> None:
        """(Re)build the cached [81, P] pill/preview combo embeddings from the
        current net weights (called at init and after `refresh_weights`)."""

        with torch.inference_mode():
            p3 = torch.from_numpy(self._combo_pills).to(self.device)
            p4 = torch.from_numpy(self._combo_prevs).to(self.device)
            self._p_combo = self.net_dev.pill_fusion(
                torch.cat(
                    [self.net_dev.pill_embedding(p3), self.net_dev.preview_embedding(p4)], dim=-1
                )
            )

    def refresh_weights(self, state_dict: Dict[str, Any]) -> None:
        """Reload net weights in place (CPU tensors expected).

        Used by training-time distillation (`training/algo/search_distill.py`)
        where the guiding net moves every PPO update. Refreshes both net copies
        and the cached combo embeddings.
        """

        self.net.load_state_dict(state_dict)
        if self.net_dev is not self.net:
            self.net_dev.load_state_dict(state_dict)
        self._refresh_combo_embeddings()

    def close(self) -> None:
        if getattr(self, "_runner", None) is not None:
            self._runner.close()

    # ------------------------------------------------------------------ public
    def decide(
        self,
        board_bytes128: np.ndarray | Sequence[int],
        pill: Sequence[int],
        preview: Sequence[int],
        feasible_mask512: np.ndarray,
        cost_to_lock512: np.ndarray,
        speed_setting: int,
        speed_ups: int,
        level: int,
        *,
        viruses_initial: Optional[int] = None,
        frames_used: int = 0,
        deadline_ms: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Pick a macro action for the given decision context.

        Args:
            board_bytes128: NES tile bytes, row 0 = top (live-bridge `field`).
            pill / preview: raw NES colors (0=Y,1=R,2=B), 2 each.
            feasible_mask512 / cost_to_lock512: root planner outputs (flat 512).
            speed_setting / speed_ups / level: engine context for the sim.
            viruses_initial / frames_used: aux-v1 context (defaults: current
                virus count / 0).
        Returns:
            (macro_action, info). `macro_action` is -1 when no placement is
            feasible. info: fallback_action, agreed_with_policy,
            nodes_expanded, elapsed_ms, value_best, value_fallback, stage.
        """

        t0 = time.perf_counter()
        budget_s = (self.deadline_ms if deadline_ms is None else float(deadline_ms)) / 1e3
        deadline = t0 + budget_s

        info: Dict[str, Any] = {
            "fallback_action": -1,
            "agreed_with_policy": True,
            "nodes_expanded": 0,
            "elapsed_ms": 0.0,
            "value_best": None,
            "value_fallback": None,
            "stage": "fallback",
            # Distillation exports (training/algo/search_distill.py): filled at
            # the root forward / commit. None when the search never got there.
            "value_root": None,
            "cand_actions": None,
            "cand_mask": None,
            "prior_logits": None,
            "ply1_actions": None,
            "q_values": None,
        }

        def _finish(action: int, stage: str) -> Tuple[int, Dict[str, Any]]:
            info["stage"] = stage
            info["agreed_with_policy"] = bool(action == info["fallback_action"])
            info["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
            return int(action), info

        board = np.ascontiguousarray(np.asarray(board_bytes128, dtype=np.uint8).reshape(128))
        pill_raw = (int(pill[0]) & 0x03, int(pill[1]) & 0x03)
        prev_raw = (int(preview[0]) & 0x03, int(preview[1]) & 0x03)
        pill_can = (_COLOR_SWAP[pill_raw[0]], _COLOR_SWAP[pill_raw[1]])
        prev_can = (_COLOR_SWAP[prev_raw[0]], _COLOR_SWAP[prev_raw[1]])

        mask512 = np.asarray(feasible_mask512).reshape(MACRO_ACTIONS).astype(bool)
        cost512 = np.asarray(cost_to_lock512).reshape(MACRO_ACTIONS)
        if not bool(mask512.any()):
            return _finish(-1, "no-feasible")

        # Training-parity symmetry reduction (same-color pill: drop o=2,3
        # duplicates; obs channels 6,7 zeroed below). Guard: keep the full
        # mask if reduction would empty it (mid-fall live masks only).
        mask_red = mask512.copy()
        if pill_can[0] == pill_can[1]:
            cand = mask_red.reshape(4, GRID_H, GRID_W)
            if bool(cand[:2].any()):
                cand[2:] = False

        v_count = int(((board & 0xF0) == 0xD0).sum())
        v0 = v_count if viruses_initial is None else int(viruses_initial)

        # ---------------------------------------------------------- root prior
        obs0 = _build_obs_from_board(board, mask512)  # ch 8..11 = unreduced mask
        if pill_can[0] == pill_can[1]:
            obs0[6:8] = 0.0
        packed0 = pack_feasible_candidates(
            mask_red.reshape(4, GRID_H, GRID_W),
            cost512.reshape(4, GRID_H, GRID_W),
            max_candidates=self.candidate_max,
            sort_by_cost=True,
        )
        infos0 = [
            self._sim_info(
                speed_setting=speed_setting,
                level=level,
                frames_used=int(frames_used),
                viruses_initial=v0,
                viruses_remaining=v_count,
                options=int(mask_red.sum()),
            )
        ]
        logits0, value0 = self._forward_full(
            obs0[None],
            np.array([pill_can], dtype=np.int64),
            np.array([prev_can], dtype=np.int64),
            packed0.actions[None],
            packed0.cost[None],
            packed0.mask[None],
            infos0,
        )
        lg0 = logits0[0]
        lg0[~packed0.mask] = -np.inf
        order = np.argsort(-lg0, kind="stable")
        order = order[packed0.mask[order]]
        fallback_action = int(packed0.actions[order[0]])
        info["fallback_action"] = fallback_action
        info["value_fallback"] = float(value0[0])
        info["value_root"] = float(value0[0])
        info["cand_actions"] = packed0.actions.copy()
        info["cand_mask"] = packed0.mask.copy()
        info["prior_logits"] = lg0.copy()  # -inf on padding slots

        if time.perf_counter() >= deadline:
            return _finish(fallback_action, "fallback")

        # ------------------------------------------------------- ply-1 branches
        K = min(self.beam, int(order.size), self.num_sim_envs)
        block = max(1, self.num_sim_envs // K)
        ply1_actions = packed0.actions[order[:K]].astype(np.int64)

        # Root checkpoint reset with the caller's plan injected: no native BFS
        # here, and the sim accepts exactly the caller-feasible actions.
        inject_costs = self._inject_costs(cost512, mask512)
        lvl = int(max(0, min(int(level), 25)))
        spd = int(max(0, min(int(speed_setting), 2)))
        sups = int(max(0, min(int(speed_ups), 255)))

        def _root_spec() -> object:
            return build_reset_spec(
                level=lvl,
                speed_setting=spd,
                rng_state=(int(self._rng.integers(0, 256)), int(self._rng.integers(0, 256))),
                rng_override=True,
                checkpoint_enabled=True,
                checkpoint_board=board,
                checkpoint_falling_colors=pill_raw,
                checkpoint_preview_colors=prev_raw,
                checkpoint_speed_ups=sups,
                inject_plan=True,
                inject_feasible=mask512.astype(np.uint8),
                inject_costs=inject_costs,
            )

        self._runner.reset(None, [_root_spec() for _ in range(self.num_sim_envs)])

        # Ply-1: one representative env per branch (envs 0..K-1). The step's
        # decision replan (one BFS per surviving branch) is reused below via
        # planner-cache injection, so no board state is planned twice.
        acts = np.full((self.num_sim_envs,), -1, dtype=np.int32)
        acts[:K] = ply1_actions.astype(np.int32)
        self._runner.step(acts, None, None)
        info["nodes_expanded"] = K

        buf = self._runner.buffers
        reps = np.arange(K)
        block_status = np.full((K,), _BLOCK_ALIVE, dtype=np.int32)
        # Rank order as tiny prior tie-break until depth-1 values arrive.
        block_values = -1e-3 * np.arange(K, dtype=np.float64)
        tau1 = np.zeros((K,), dtype=np.int64)
        r1 = np.zeros((K,), dtype=np.float64)
        disc1 = np.ones((K,), dtype=np.float64)
        for i, rep in enumerate(reps):
            tau1[i] = int(buf.tau_frames[rep])
            disc1[i] = self.gamma ** float(tau1[i])
            if int(buf.invalid_action[rep]) != -1:
                block_status[i] = _BLOCK_INVALID
                block_values[i] = -np.inf
                continue
            r1[i] = self._step_reward(
                rep,
                v_prev=v_count,
                v0=v0,
                elapsed_frames=int(frames_used) + int(tau1[i]),
            )
            if int(buf.terminated[rep]) != 0 or int(buf.truncated[rep]) != 0:
                block_status[i] = _BLOCK_TERMINAL
                block_values[i] = self._terminal_q(
                    reason=int(buf.terminal_reason[rep]), ply=1, reward=float(r1[i])
                )

        def _commit(stage: str) -> Tuple[int, Dict[str, Any]]:
            # Per-branch backed-up Qs (aligned with ply1_actions); -inf marks
            # invalid branches. Consumed by training-time distillation.
            info["ply1_actions"] = ply1_actions.copy()
            info["q_values"] = block_values.copy()
            best = int(np.argmax(block_values))
            if not np.isfinite(block_values[best]):
                return _finish(fallback_action, stage)
            info["value_best"] = float(block_values[best])
            fb = np.flatnonzero(ply1_actions == fallback_action)
            if fb.size and np.isfinite(block_values[int(fb[0])]):
                info["value_fallback"] = float(block_values[int(fb[0])])
            return _finish(int(ply1_actions[best]), stage)

        alive = np.flatnonzero(block_status == _BLOCK_ALIVE)
        if alive.size == 0 or time.perf_counter() >= deadline:
            return _commit("ply1")

        # ------------------------------------------------- ply-2 priors (depth 1)
        # Copy the representative outputs before the phase-B reset reuses the
        # buffers.
        same2 = prev_can[0] == prev_can[1]
        B2 = int(alive.size)
        obs2 = np.ascontiguousarray(buf.obs[alive], dtype=np.float32)
        boards2 = buf.board_bytes[alive].copy()  # type: ignore[union-attr]
        mask2_flat = buf.feasible_mask[alive].copy()
        cost2_flat = buf.cost_to_lock[alive].copy()
        masks2 = mask2_flat.reshape(B2, 4, GRID_H, GRID_W).astype(bool)
        costs2 = cost2_flat.reshape(B2, 4, GRID_H, GRID_W)
        viruses2 = buf.viruses_rem[alive].astype(np.int64)
        if same2:
            obs2[:, 6:8] = 0.0
            keep = masks2[:, :2].reshape(B2, -1).any(axis=1)
            masks2[keep, 2:] = False
        ca2 = np.full((B2, self.candidate_max), -1, dtype=np.int32)
        cm2 = np.zeros((B2, self.candidate_max), dtype=np.bool_)
        cc2 = np.zeros((B2, self.candidate_max), dtype=np.float32)
        infos2 = []
        for k, i in enumerate(alive):
            pk = pack_feasible_candidates(
                masks2[k], costs2[k], max_candidates=self.candidate_max, sort_by_cost=True
            )
            ca2[k], cm2[k], cc2[k] = pk.actions, pk.mask, pk.cost
            infos2.append(
                self._sim_info(
                    speed_setting=speed_setting,
                    level=level,
                    frames_used=int(frames_used) + int(tau1[i]),
                    viruses_initial=v0,
                    viruses_remaining=int(viruses2[k]),
                    options=int(masks2[k].sum()),
                )
            )
        pills2 = np.repeat(np.array([prev_can], dtype=np.int64), B2, axis=0)
        prevs2 = buf.preview_colors[alive].astype(np.int64)  # seed garbage; priors only
        logits2, values2 = self._forward_full(obs2, pills2, prevs2, ca2, cc2, cm2, infos2)
        # Depth-1 estimate: Q(a1) = r1 + gamma^tau1 * V(s after ply-1).
        block_values[alive] = r1[alive] + disc1[alive] * values2.astype(np.float64)

        if time.perf_counter() >= deadline:
            return _commit("depth1")

        # ----------------------------------------- phase B: re-checkpoint + ply-2
        # Fan each surviving branch out to a block of envs by checkpoint-reset
        # to its post-ply-1 board (falling pill = the known preview) with the
        # branch's just-computed plan injected — again no BFS at reset. The
        # ply-2 step then replans once per *distinct* leaf board.
        alive_set = set(int(i) for i in alive)
        rep_of_block = {int(i): k for k, i in enumerate(alive)}
        specs_b: List[object] = []
        for j in range(self.num_sim_envs):
            i = j // block
            if i not in alive_set:
                specs_b.append(_root_spec())  # parked; stepped with -1 below
                continue
            k = rep_of_block[i]
            specs_b.append(
                build_reset_spec(
                    level=lvl,
                    speed_setting=spd,
                    rng_state=(
                        int(self._rng.integers(0, 256)),
                        int(self._rng.integers(0, 256)),
                    ),
                    rng_override=True,
                    checkpoint_enabled=True,
                    checkpoint_board=boards2[k],
                    checkpoint_falling_colors=prev_raw,
                    checkpoint_preview_colors=(0xFF, 0xFF),  # garbage; marginalized
                    checkpoint_speed_ups=sups,
                    inject_plan=True,
                    inject_feasible=mask2_flat[k],
                    inject_costs=cost2_flat[k],
                )
            )
        self._runner.reset(None, specs_b)

        acts2 = np.full((self.num_sim_envs,), -1, dtype=np.int32)
        n2 = np.zeros((K,), dtype=np.int64)
        for k, i in enumerate(alive):
            lg = logits2[k].copy()
            lg[~cm2[k]] = -np.inf
            o2 = np.argsort(-lg, kind="stable")
            o2 = o2[cm2[k][o2]][:block]
            n2[i] = int(o2.size)
            for s, slot in enumerate(o2):
                acts2[int(i) * block + s] = int(ca2[k][slot])
        self._runner.step(acts2, None, None)

        # --------------------------------------------------- leaves (depth 2)
        # Per-leaf Q contribution: r2 + gamma^tau2 * V(leaf) (terminal sims get
        # terminal Qs); blocks back up the max, discounted behind r1.
        viruses_of_block = {int(i): int(viruses2[k]) for k, i in enumerate(alive)}
        leaf_envs: List[int] = []
        leaf_block: List[int] = []
        leaf_r2: Dict[int, float] = {}
        leaf_disc2: Dict[int, float] = {}
        leaf_values: List[List[float]] = [[] for _ in range(K)]
        for i in alive:
            for s in range(int(n2[i])):
                j = int(i) * block + s
                if int(buf.invalid_action[j]) != -1:
                    continue
                tau2 = int(buf.tau_frames[j])
                r2 = self._step_reward(
                    j,
                    v_prev=viruses_of_block[int(i)],
                    v0=v0,
                    elapsed_frames=int(frames_used) + int(tau1[i]) + tau2,
                )
                if int(buf.terminated[j]) != 0 or int(buf.truncated[j]) != 0:
                    leaf_values[int(i)].append(
                        self._terminal_q(
                            reason=int(buf.terminal_reason[j]), ply=2, reward=float(r2)
                        )
                    )
                else:
                    leaf_envs.append(j)
                    leaf_block.append(int(i))
                    leaf_r2[j] = float(r2)
                    leaf_disc2[j] = self.gamma ** float(tau2)
        info["nodes_expanded"] += int(n2.sum())

        if leaf_envs:
            # Best-first chunking: refine the most promising blocks before the
            # deadline; a single batch when the estimate fits.
            order_blocks = sorted(set(leaf_block), key=lambda b: -block_values[b])
            remaining_ms = (deadline - time.perf_counter()) * 1e3
            if remaining_ms >= 1.3 * self._leaf_ms_per_env * len(leaf_envs) + 3.0:
                chunks = [order_blocks]
            else:
                chunks = [[b] for b in order_blocks]
            by_block: Dict[int, List[int]] = {}
            for j, b in zip(leaf_envs, leaf_block):
                by_block.setdefault(b, []).append(j)
            for chunk in chunks:
                if time.perf_counter() >= deadline:
                    break
                idxs = [j for b in chunk for j in by_block[b]]
                tchunk = time.perf_counter()
                vals = self._marginal_leaf_values(
                    idxs,
                    speed_setting=speed_setting,
                    level=level,
                    frames_used_base=int(frames_used),
                    tau1=tau1,
                    block=block,
                    viruses_initial=v0,
                )
                dt_ms = (time.perf_counter() - tchunk) * 1e3
                per_env = dt_ms / max(1, len(idxs))
                self._leaf_ms_per_env = 0.7 * self._leaf_ms_per_env + 0.3 * per_env
                for j, v in zip(idxs, vals):
                    leaf_values[j // block].append(leaf_r2[j] + leaf_disc2[j] * float(v))

        block_values = backup_block_values(
            block_status, block_values, leaf_values, r1=r1, disc1=disc1
        )
        return _commit("depth2")

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _inject_costs(cost512: np.ndarray, mask512: np.ndarray) -> np.ndarray:
        """Caller costs -> uint16 planner-injection costs (>=1; 0xFFFF infeasible)."""

        cost = np.asarray(cost512, dtype=np.float64).reshape(MACRO_ACTIONS).copy()
        cost[~np.isfinite(cost)] = float(0xFFFF)
        out = np.clip(cost, 1.0, float(0xFFFF)).astype(np.uint16)
        out[~np.asarray(mask512, dtype=bool)] = 0xFFFF
        return out

    @staticmethod
    def _sim_info(
        *,
        speed_setting: int,
        level: int,
        frames_used: int,
        viruses_initial: int,
        viruses_remaining: int,
        options: int,
    ) -> Dict[str, Any]:
        return {
            "pill/speed_setting": int(speed_setting),
            "speed_setting": int(speed_setting),
            "level": int(level),
            "task_mode": "viruses",
            "task/frames_used": int(frames_used),
            "drm/viruses_initial": int(viruses_initial),
            "viruses_remaining": int(viruses_remaining),
            "placements/options": int(options),
        }

    def _terminal_q(self, *, reason: int, ply: int, reward: float) -> float:
        """Q of a terminal sim node.

        VS nets: dominating win/loss constants (docs/SEARCH_DESIGN.md) with a
        small depth penalty so an immediate win beats a win one pill later.
        1P nets: the replicated env reward already contains the terminal
        bonus/penalty and there is no future value, so Q == reward.
        """

        if self.reward_mode != "vs":
            return float(reward)
        scale = 1.0 - self.depth_penalty * float(max(0, ply - 1))
        if reason == _TERMINAL_CLEAR:
            return self.win_value * scale
        # Topout, and (conservatively) wait-cap truncation.
        return self.loss_value * scale

    def _step_reward(
        self, j: int, *, v_prev: int, v0: int, elapsed_frames: int, buf: Optional[Any] = None
    ) -> float:
        """Replicate the training reward for the sim step that env `j` just took.

        1P: `DrMarioPoolVecEnv` reward components from the pool's event
        counters. VS: the garbage-shaping term with the volley size estimated
        from cleared tiles (the 1P sim has no garbage plumbing); terminal
        outcome rewards are handled by `_terminal_q`.
        """

        buf = self._runner.buffers if buf is None else buf
        if self.reward_mode == "vs":
            lines = int(buf.tiles_cleared_total[j]) // 4
            volley = min(4, lines) if lines >= 2 else 0
            return self.garbage_reward_coef * float(volley)

        rc = self._reward_cfg
        r = 0.0
        delta_v = max(0, int(v_prev) - int(buf.viruses_rem[j]))
        if delta_v > 0 and int(v0) > 0:
            r += rc.virus_clear_bonus * float(delta_v) / float(v0)
        r += rc.non_virus_clear_bonus * float(int(buf.tiles_cleared_nonvirus[j]))
        r += rc.adjacency_triplet_bonus * float(int(buf.adj_triplet[j].sum()))
        r += rc.adjacency_pair_bonus * float(int(buf.adj_pair[j].sum()))
        r += rc.virus_adjacency_triplet_bonus * float(int(buf.virus_adj_triplet[j].sum()))
        r += rc.virus_adjacency_pair_bonus * float(int(buf.virus_adj_pair[j].sum()))

        terminated = int(buf.terminated[j]) != 0
        reason = int(buf.terminal_reason[j])
        topout = bool(terminated and reason == _TERMINAL_TOPOUT)
        cleared = bool(
            not topout
            and ((terminated and reason == _TERMINAL_CLEAR) or int(buf.viruses_rem[j]) == 0)
        )
        elapsed_sec = float(max(0, int(elapsed_frames))) / 60.0
        if cleared:
            r += rc.terminal_clear_bonus
            r -= rc.time_penalty_clear_per_60_frames * elapsed_sec
        if topout:
            r += rc.topout_penalty
            r += rc.time_bonus_topout_per_60_frames * elapsed_sec
        return float(r)

    def _forward_full(
        self,
        obs: np.ndarray,
        pills: np.ndarray,
        prevs: np.ndarray,
        cand_actions: np.ndarray,
        cand_cost: np.ndarray,
        cand_mask: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full candidate forward: (logits [B,Kmax], values [B]) as numpy."""

        aux = None
        if self.aux_shim is not None:
            aux = aux_v1_batch_fast(obs, infos)
        dev = "cpu"
        with torch.inference_mode():
            logits, values = self.net(
                torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(dev),
                torch.from_numpy(np.ascontiguousarray(pills, dtype=np.int64)).to(dev),
                torch.from_numpy(np.ascontiguousarray(prevs, dtype=np.int64)).to(dev),
                torch.from_numpy(np.ascontiguousarray(cand_actions, dtype=np.int32)).to(dev),
                torch.from_numpy(np.ascontiguousarray(cand_cost, dtype=np.float32)).to(dev),
                torch.from_numpy(np.ascontiguousarray(cand_mask)).to(dev),
                aux=None if aux is None else torch.from_numpy(aux).to(dev),
            )
            return (
                logits.float().cpu().numpy(),
                values.float().cpu().numpy().reshape(-1),
            )

    def _marginal_leaf_values(
        self,
        env_idxs: Sequence[int],
        *,
        speed_setting: int,
        level: int,
        frames_used_base: int,
        tau1: np.ndarray,
        block: int,
        viruses_initial: int,
        buf: Optional[Any] = None,
    ) -> np.ndarray:
        """Leaf values with the unknown (pill, preview) colors marginalized.

        Mean over the 81 ordered canonical color pairs; the board trunk runs
        on two obs variants (normal / symmetry-reduced ch-6,7 zeroed), each
        combo selecting the variant + aux matching its same-color bit.
        """

        buf = self._runner.buffers if buf is None else buf
        idxs = list(env_idxs)
        B = len(idxs)
        obs = np.ascontiguousarray(buf.obs[idxs], dtype=np.float32)

        infos_n: List[Dict[str, Any]] = []
        infos_z: List[Dict[str, Any]] = []
        for j in idxs:
            mask = buf.feasible_mask[j].reshape(4, -1)
            opt_n = int(mask.sum())
            opt_z = int(mask[:2].sum()) if bool(mask[:2].any()) else opt_n
            common = dict(
                speed_setting=speed_setting,
                level=level,
                frames_used=int(frames_used_base)
                + int(tau1[j // block])
                + int(buf.tau_frames[j]),
                viruses_initial=int(viruses_initial),
                viruses_remaining=int(buf.viruses_rem[j]),
            )
            infos_n.append(self._sim_info(options=opt_n, **common))
            infos_z.append(self._sim_info(options=opt_z, **common))

        v = self._combo_values(obs, infos_n, infos_z)  # [B,81]
        return v.mean(dim=1).reshape(-1).float().cpu().numpy()

    def _combo_values(
        self,
        obs: np.ndarray,
        infos_n: List[Dict[str, Any]],
        infos_z: List[Dict[str, Any]],
    ) -> "torch.Tensor":
        """Value head over all 81 (pill, preview) canonical color combos.

        Returns a [B, 81] tensor (combo index = 9*pill_pair + preview_pair,
        canonical-pair-major). Same trunk-sharing trick as documented on
        `_marginal_leaf_values`.
        """

        B = int(np.asarray(obs).shape[0])
        net = self.net_dev
        dev = self.device
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs).to(dev)
            obs_z = obs_t.clone()
            obs_z[:, 6:8] = 0.0
            board = torch.cat([obs_t, obs_z], dim=0)[:, : net.board_channels]
            trunk = net.board_trunk(board)
            if trunk.ndim == 4:
                g_board = trunk.mean(dim=(2, 3))
            else:
                g_board = trunk.mean(dim=1)
            gb_n, gb_z = g_board[:B], g_board[B:]  # [B,D]

            p = self._p_combo  # [81, P]
            n_combo = p.shape[0]
            if self.aux_shim is not None and net.aux_encoder is not None:
                aux = np.concatenate(
                    [aux_v1_batch_fast(obs, infos_n), aux_v1_batch_fast(obs, infos_z)], axis=0
                )
                ae = net.aux_encoder(torch.from_numpy(aux).to(dev))
                ae_n, ae_z = ae[:B], ae[B:]  # [B,P]
                p_exp = p.unsqueeze(0).expand(B, n_combo, p.shape[1])
                cond_n = net.cond_fusion(
                    torch.cat([p_exp, ae_n.unsqueeze(1).expand(B, n_combo, ae_n.shape[1])], dim=-1)
                )
                cond_z = net.cond_fusion(
                    torch.cat([p_exp, ae_z.unsqueeze(1).expand(B, n_combo, ae_z.shape[1])], dim=-1)
                )
                cm_n = net.cond_to_model(cond_n)  # [B,81,D]
                cm_z = net.cond_to_model(cond_z)
            else:
                cm = net.cond_to_model(p)  # [81,D]
                cm_n = cm.unsqueeze(0).expand(B, n_combo, cm.shape[1])
                cm_z = cm_n

            same = self._combo_same.view(1, n_combo, 1)
            pre = torch.where(same, gb_z.unsqueeze(1) + cm_z, gb_n.unsqueeze(1) + cm_n)
            v = net.value_head(net.ln_g(pre))  # [B,81,1]
            return v.reshape(B, n_combo)

    def _warmup(self) -> None:
        """Run one full-depth decide on a synthetic state (JIT/dispatch warmup,
        seeds the leaf-cost estimate)."""

        board = np.full((GRID_H, GRID_W), 0xFF, dtype=np.uint8)
        g = np.random.default_rng(1234)
        for _ in range(12):
            board[int(g.integers(10, GRID_H)), int(g.integers(0, GRID_W))] = 0xD0 + int(
                g.integers(0, 3)
            )
        # Plan the root with one sim env via a checkpoint reset.
        spec = build_reset_spec(
            level=10,
            speed_setting=2,
            rng_state=(1, 2),
            rng_override=True,
            checkpoint_enabled=True,
            checkpoint_board=board.reshape(-1),
            checkpoint_falling_colors=(0, 1),
            checkpoint_preview_colors=(1, 2),
        )
        self._runner.reset(None, [spec] * self.num_sim_envs)
        mask = self._runner.buffers.feasible_mask[0].copy()
        cost = self._runner.buffers.cost_to_lock[0].copy()
        # Twice: the first full-depth pass compiles/dispatches device kernels;
        # the second calibrates a realistic leaf-cost estimate.
        for _ in range(2):
            self.decide(
                board.reshape(-1),
                (0, 1),
                (1, 2),
                mask,
                cost,
                2,
                0,
                10,
                deadline_ms=10_000.0,
            )


# --------------------------------------------------------------------- ponder
def _board_key(board_bytes128: np.ndarray | Sequence[int]) -> bytes:
    """Canonical cache key for a board: empty-ish tiles (>= 0xF0) -> 0xFF."""

    b = np.asarray(board_bytes128, dtype=np.uint8).reshape(128).copy()
    b[b >= 0xF0] = 0xFF
    return b.tobytes()


def _pair_index(raw0: int, raw1: int) -> int:
    """Ordered raw NES color pair -> canonical pair index in [0, 9)."""

    return _COLOR_SWAP[int(raw0) & 3] * 3 + _COLOR_SWAP[int(raw1) & 3]


@dataclass
class PonderResult:
    """Cached next-decision search keyed by (post-commit board, next pill).

    `q[:, c]` are the backed-up Q values of the full-width ply-1 `actions`
    conditioned on canonical preview-pair index `c`; `depth[c]` is 2 when the
    pair got the ply-2 refinement (1 = depth-1 value estimate only).
    `refined[i, c]` marks entries whose q is a depth-2 backup (or an exact
    ply-1 terminal value) rather than a depth-1 estimate: the decision argmax
    must not mix the two — the depth-1 V estimate is systematically biased
    relative to the depth-2 backup, so an unrestricted argmax degenerates to
    a depth-1 search (measured: probe win rate collapsed to ~13% vs the
    plain depth-2 search before this restriction).
    """

    board_key: bytes
    pill_raw: Tuple[int, int]
    actions: np.ndarray  # (F,) int64 macro actions, full feasible width
    q: np.ndarray  # (F, 9) float64
    refined: np.ndarray  # (F, 9) bool
    prior_action: np.ndarray  # (9,) int64 root policy argmax per preview pair
    depth: np.ndarray  # (9,) int8
    wall_s: float


@dataclass
class _PonderJob:
    board: np.ndarray  # (128,) uint8 decision-N board
    action: int  # committed macro action
    pill: Tuple[int, int]  # raw NES colors of the committed pill
    preview: Tuple[int, int]  # raw NES colors of the next pill
    mask: np.ndarray  # (512,) bool caller-feasible mask at decision N
    cost: np.ndarray  # (512,) float64 caller cost-to-lock at decision N
    speed_setting: int
    speed_ups: int
    level: int
    viruses_initial: Optional[int]
    frames_used: int
    abort: threading.Event = field(default_factory=threading.Event)


class PonderingSearchPolicy(SearchPolicy):
    """SearchPolicy + dead-time pondering of the *next* decision.

    After a placement is committed, `start_ponder()` hands the decision
    context to a single background worker (newest job wins). The worker
    resolves the committed placement (deterministic, one sim), then searches
    the next decision root (post-commit board B', pill = the known preview,
    preview = unknown) on its own dedicated pool runner:

    - full-width ply-1: every feasible placement of the next pill is simmed
      (chunked over the runner) and given a depth-1 value conditioned on each
      of the 9 candidate preview pairs in one marginal-style forward;
    - per preview pair: ply-2 beam (priors conditioned on that pair) +
      marginalized leaf values + the training-consistent Q backup, exactly as
      in `SearchPolicy.decide` (budget-checked; pairs not reached keep their
      depth-1 values).

    The result is cached keyed by (normalized board bytes, next pill colors).
    `decide()` consults the cache first: on a hit the revealed preview pair
    selects a column and the answer is an argmax over that column's
    depth-2-refined entries restricted to the caller's feasible mask (~1 ms).
    Any mismatch (garbage, replan, divergence) — or a column without a
    feasible depth-2-refined entry — falls back to the normal deadline search
    and aborts the stale job.
    """

    # Entries are small (~F*9 floats); the cap only needs to cover the number
    # of concurrently-pondered games (offline probes share one policy across
    # several envs) times a few decisions of slack.
    _PONDER_CACHE_CAP = 64
    _PONDER_TIMES_CAP = 512

    def __init__(self, checkpoint_path: str | Path, *, ponder_budget_s: float = 1.0, **kwargs) -> None:
        super().__init__(checkpoint_path, **kwargs)
        self.ponder_budget_s = float(ponder_budget_s)

    def _init_common(self, **kw: Any) -> None:
        # Ponder plumbing must exist before the base warmup calls decide().
        self.ponder_budget_s = 1.0
        self._ponder_cv = threading.Condition()
        self._ponder_cache: "OrderedDict[Tuple[bytes, Tuple[int, int]], PonderResult]" = OrderedDict()
        self._ponder_pending: Optional[_PonderJob] = None
        self._ponder_running: Optional[_PonderJob] = None
        self._ponder_shutdown = False
        self._ponder_times: List[float] = []
        self._ponder_counters: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "aborts": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "pairs_total": 0,
            "pairs_depth2": 0,
        }
        self._ponder_last_error: Optional[str] = None
        super()._init_common(**kw)
        self._ponder_rng = np.random.default_rng(int(self._rng.integers(2**31)))
        self._ponder_runner = DrMarioPoolRunner(
            num_envs=self.num_sim_envs,
            obs_spec=4,
            obs_channels=12,
            lib_path=kw.get("lib_path"),
            emit_board=True,
        )
        self._ponder_thread = threading.Thread(
            target=self._ponder_loop, name="ponder-worker", daemon=True
        )
        self._ponder_thread.start()
        for k in self._ponder_counters:  # warmup decide() counts as misses; reset
            self._ponder_counters[k] = 0

    def close(self) -> None:
        cv = getattr(self, "_ponder_cv", None)
        if cv is not None and getattr(self, "_ponder_thread", None) is not None:
            with cv:
                self._ponder_shutdown = True
                self._abort_ponder_locked()
                cv.notify_all()
            self._ponder_thread.join(timeout=10.0)
            self._ponder_thread = None
        runner = getattr(self, "_ponder_runner", None)
        if runner is not None:
            runner.close()
            self._ponder_runner = None
        super().close()

    # ------------------------------------------------------------- public API
    def start_ponder(
        self,
        board_bytes128: np.ndarray | Sequence[int],
        committed_action: int,
        pill: Sequence[int],
        preview: Sequence[int],
        *,
        feasible_mask512: np.ndarray,
        cost_to_lock512: np.ndarray,
        speed_setting: int,
        speed_ups: int,
        level: int,
        viruses_initial: Optional[int] = None,
        frames_used: int = 0,
    ) -> None:
        """Queue a ponder job for the decision that follows `committed_action`.

        Non-blocking. A running or pending job is superseded (newest wins).
        """

        job = _PonderJob(
            board=np.ascontiguousarray(np.asarray(board_bytes128, dtype=np.uint8).reshape(128)),
            action=int(committed_action),
            pill=(int(pill[0]) & 3, int(pill[1]) & 3),
            preview=(int(preview[0]) & 3, int(preview[1]) & 3),
            mask=np.asarray(feasible_mask512).reshape(MACRO_ACTIONS).astype(bool),
            cost=np.asarray(cost_to_lock512, dtype=np.float64).reshape(MACRO_ACTIONS),
            speed_setting=int(speed_setting),
            speed_ups=int(speed_ups),
            level=int(level),
            viruses_initial=None if viruses_initial is None else int(viruses_initial),
            frames_used=int(frames_used),
        )
        with self._ponder_cv:
            self._abort_ponder_locked()
            self._ponder_pending = job
            self._ponder_counters["jobs_started"] += 1
            self._ponder_cv.notify_all()

    def has_ponder_for(
        self, board_bytes128: np.ndarray | Sequence[int], pill: Sequence[int]
    ) -> bool:
        """Cheap peek: is a completed ponder result cached for this root?"""

        key = (_board_key(board_bytes128), (int(pill[0]) & 3, int(pill[1]) & 3))
        with self._ponder_cv:
            return key in self._ponder_cache

    def ponder_invalidate(self) -> None:
        """Drop all cached results and abort any running/pending job."""

        with self._ponder_cv:
            self._ponder_cache.clear()
            self._abort_ponder_locked()

    def wait_ponder_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until no job is pending or running (tests / offline probes)."""

        with self._ponder_cv:
            return self._ponder_cv.wait_for(
                lambda: self._ponder_pending is None and self._ponder_running is None,
                timeout,
            )

    def ponder_stats(self) -> Dict[str, Any]:
        with self._ponder_cv:
            out: Dict[str, Any] = dict(self._ponder_counters)
            times = list(self._ponder_times)
            out["last_error"] = self._ponder_last_error
        n = out["hits"] + out["misses"]
        out["hit_rate"] = out["hits"] / n if n else None
        if times:
            arr = np.asarray(times)
            out["job_wall_s_p50"] = float(np.percentile(arr, 50))
            out["job_wall_s_p95"] = float(np.percentile(arr, 95))
            out["job_wall_s_max"] = float(arr.max())
        return out

    def decide(
        self,
        board_bytes128: np.ndarray | Sequence[int],
        pill: Sequence[int],
        preview: Sequence[int],
        feasible_mask512: np.ndarray,
        cost_to_lock512: np.ndarray,
        speed_setting: int,
        speed_ups: int,
        level: int,
        *,
        viruses_initial: Optional[int] = None,
        frames_used: int = 0,
        deadline_ms: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        t0 = time.perf_counter()
        mask512 = np.asarray(feasible_mask512).reshape(MACRO_ACTIONS).astype(bool)
        if mask512.any():
            key = (
                _board_key(board_bytes128),
                (int(pill[0]) & 3, int(pill[1]) & 3),
            )
            with self._ponder_cv:
                res = self._ponder_cache.get(key)
            if res is not None:
                cidx = _pair_index(int(preview[0]), int(preview[1]))
                qcol = res.q[:, cidx]
                ok = mask512[res.actions] & np.isfinite(qcol)
                # Commit only when the revealed pair's column got the ply-2
                # refinement and a refined entry is still feasible: depth-1
                # estimates must never decide (their V bias degenerated the
                # probe to a depth-1 search), and a depth-1 column's refined
                # entries are terminals only (an argmax restricted to those
                # can commit a topout). Anything else is a miss -> live search.
                okr = ok & res.refined[:, cidx]
                if int(res.depth[cidx]) == 2 and bool(okr.any()):
                    qm = np.where(okr, qcol, -np.inf)
                    sel = int(np.argmax(qm))
                    action = int(res.actions[sel])
                    fb = int(res.prior_action[cidx])
                    vf = None
                    fbpos = np.flatnonzero(res.actions == fb)
                    if fbpos.size and np.isfinite(qcol[int(fbpos[0])]):
                        vf = float(qcol[int(fbpos[0])])
                    with self._ponder_cv:
                        self._ponder_counters["hits"] += 1
                    info: Dict[str, Any] = {
                        "fallback_action": fb,
                        "agreed_with_policy": bool(action == fb),
                        "nodes_expanded": 0,
                        "elapsed_ms": (time.perf_counter() - t0) * 1e3,
                        "value_best": float(qm[sel]),
                        "value_fallback": vf,
                        "stage": "ponder",
                        "ponder_hit": True,
                        "ponder_depth": int(res.depth[cidx]),
                        "ponder_refined": True,
                    }
                    return action, info
            # Miss (or unusable entry): the running job is for a stale context.
            with self._ponder_cv:
                self._ponder_counters["misses"] += 1
                self._abort_ponder_locked()
        action, info = super().decide(
            board_bytes128,
            pill,
            preview,
            feasible_mask512,
            cost_to_lock512,
            speed_setting,
            speed_ups,
            level,
            viruses_initial=viruses_initial,
            frames_used=frames_used,
            deadline_ms=deadline_ms,
        )
        info["ponder_hit"] = False
        return action, info

    # ----------------------------------------------------------- worker side
    def _abort_ponder_locked(self) -> None:
        """Abort running + drop pending job. Caller holds `_ponder_cv`."""

        if self._ponder_running is not None and not self._ponder_running.abort.is_set():
            self._ponder_running.abort.set()
            self._ponder_counters["aborts"] += 1
        if self._ponder_pending is not None:
            self._ponder_pending = None
            self._ponder_counters["aborts"] += 1

    def _ponder_loop(self) -> None:
        while True:
            with self._ponder_cv:
                self._ponder_cv.wait_for(
                    lambda: self._ponder_shutdown or self._ponder_pending is not None
                )
                if self._ponder_shutdown:
                    return
                job = self._ponder_pending
                self._ponder_pending = None
                self._ponder_running = job
            result: Optional[PonderResult] = None
            failed = False
            try:
                result = self._ponder_compute(job)
            except Exception as exc:  # keep the worker alive
                failed = True
                with self._ponder_cv:
                    self._ponder_last_error = repr(exc)
            with self._ponder_cv:
                self._ponder_running = None
                if failed:
                    self._ponder_counters["jobs_failed"] += 1
                elif result is not None and not job.abort.is_set():
                    self._ponder_cache[(result.board_key, result.pill_raw)] = result
                    while len(self._ponder_cache) > self._PONDER_CACHE_CAP:
                        self._ponder_cache.popitem(last=False)
                    self._ponder_counters["jobs_completed"] += 1
                    self._ponder_counters["pairs_total"] += 9
                    self._ponder_counters["pairs_depth2"] += int((result.depth == 2).sum())
                    self._ponder_times.append(float(result.wall_s))
                    if len(self._ponder_times) > self._PONDER_TIMES_CAP:
                        del self._ponder_times[: -self._PONDER_TIMES_CAP]
                self._ponder_cv.notify_all()

    def _ponder_compute(self, job: _PonderJob) -> Optional[PonderResult]:
        """Resolve the committed action, then search the next decision root.

        Runs on the worker thread against `self._ponder_runner` only. Returns
        None when there is nothing to cache (terminal, rejected commit, abort).
        """

        t0 = time.perf_counter()
        runner = self._ponder_runner
        buf = runner.buffers
        N = runner.num_envs
        rng = self._ponder_rng
        lvl = int(max(0, min(job.level, 25)))
        spd = int(max(0, min(job.speed_setting, 2)))
        sups = int(max(0, min(job.speed_ups, 255)))

        def _spec(board: np.ndarray, falling: Tuple[int, int], feas: np.ndarray, costs: np.ndarray) -> object:
            return build_reset_spec(
                level=lvl,
                speed_setting=spd,
                rng_state=(int(rng.integers(0, 256)), int(rng.integers(0, 256))),
                rng_override=True,
                checkpoint_enabled=True,
                checkpoint_board=board,
                checkpoint_falling_colors=falling,
                checkpoint_preview_colors=(0xFF, 0xFF),  # colors marginalized/conditioned later
                checkpoint_speed_ups=sups,
                inject_plan=True,
                inject_feasible=feas,
                inject_costs=costs,
            )

        # ---- resolve the committed placement (deterministic): B = board -> B'
        inj0 = self._inject_costs(job.cost, job.mask)
        spec0 = build_reset_spec(
            level=lvl,
            speed_setting=spd,
            rng_state=(int(rng.integers(0, 256)), int(rng.integers(0, 256))),
            rng_override=True,
            checkpoint_enabled=True,
            checkpoint_board=job.board,
            checkpoint_falling_colors=job.pill,
            checkpoint_preview_colors=job.preview,
            checkpoint_speed_ups=sups,
            inject_plan=True,
            inject_feasible=job.mask.astype(np.uint8),
            inject_costs=inj0,
        )
        runner.reset(None, [spec0] * N)
        acts = np.full((N,), -1, dtype=np.int32)
        acts[0] = int(job.action)
        runner.step(acts, None, None)
        if job.abort.is_set():
            return None
        if (
            int(buf.invalid_action[0]) != -1
            or int(buf.terminated[0]) != 0
            or int(buf.truncated[0]) != 0
        ):
            return None  # no next decision to ponder
        tau0 = int(buf.tau_frames[0])
        board1 = buf.board_bytes[0].copy()  # type: ignore[union-attr]
        mask1_u8 = buf.feasible_mask[0].copy()
        cost1 = buf.cost_to_lock[0].copy()
        v1 = int(buf.viruses_rem[0])
        frames1 = int(job.frames_used) + tau0
        v0 = v1 if job.viruses_initial is None else int(job.viruses_initial)

        mask1 = mask1_u8.reshape(MACRO_ACTIONS).astype(bool)
        if not bool(mask1.any()):
            return None
        k_raw = job.preview  # the next decision's falling pill (known)
        k_can = (_COLOR_SWAP[k_raw[0]], _COLOR_SWAP[k_raw[1]])
        mask_red = mask1.copy()
        if k_can[0] == k_can[1]:
            cand = mask_red.reshape(4, GRID_H, GRID_W)
            if bool(cand[:2].any()):
                cand[2:] = False

        # ---- full-width ply-1 action set + root priors for all 9 pairs
        packed = pack_feasible_candidates(
            mask_red.reshape(4, GRID_H, GRID_W),
            cost1.reshape(4, GRID_H, GRID_W).astype(np.float32),
            max_candidates=self.candidate_max,
            sort_by_cost=True,
        )
        actions_full = packed.actions[packed.mask].astype(np.int64)
        F = int(actions_full.size)
        if F == 0:
            return None

        obs_root = _build_obs_from_board(board1, mask1)
        if k_can[0] == k_can[1]:
            obs_root[6:8] = 0.0
        prevs9 = np.array([(a, b) for a in range(3) for b in range(3)], dtype=np.int64)
        infos_root = [
            self._sim_info(
                speed_setting=job.speed_setting,
                level=job.level,
                frames_used=frames1,
                viruses_initial=v0,
                viruses_remaining=v1,
                options=int(mask_red.sum()),
            )
        ] * 9
        logits9, _ = self._forward_full(
            np.repeat(obs_root[None], 9, axis=0),
            np.repeat(np.array([k_can], dtype=np.int64), 9, axis=0),
            prevs9,
            np.repeat(packed.actions[None], 9, axis=0),
            np.repeat(packed.cost[None], 9, axis=0),
            np.repeat(packed.mask[None], 9, axis=0),
            infos_root,
        )
        lg9 = logits9.copy()
        lg9[:, ~packed.mask] = -np.inf
        prior_action = packed.actions[np.argmax(lg9, axis=1)].astype(np.int64)
        # Per-pair prior over the full-width actions (slot order == actions_full
        # order): the ply-1 beam below is prior-selected, exactly as decide()'s
        # is — beam selection by depth-1 *values* measurably degraded play (the
        # value head misranks off-distribution ply-1 outcomes).
        prior_full = lg9[:, packed.mask]  # (9, F)

        # ---- sim every ply-1 placement (chunked over the runner)
        status = np.full((F,), _BLOCK_ALIVE, dtype=np.int32)
        tau1 = np.zeros((F,), dtype=np.int64)
        r1 = np.zeros((F,), dtype=np.float64)
        disc1 = np.ones((F,), dtype=np.float64)
        term_q = np.zeros((F,), dtype=np.float64)
        obs1 = np.zeros((F, 12, GRID_H, GRID_W), dtype=np.float32)
        boards1b = np.zeros((F, 128), dtype=np.uint8)
        maskF = np.zeros((F, MACRO_ACTIONS), dtype=np.uint8)
        costF = np.zeros((F, MACRO_ACTIONS), dtype=np.uint16)
        virF = np.zeros((F,), dtype=np.int64)
        prevF = np.zeros((F, 2), dtype=np.int64)
        for c0 in range(0, F, N):
            chunk = actions_full[c0 : c0 + N]
            runner.reset(None, [_spec(board1, k_raw, mask1_u8, cost1)] * N)
            acts = np.full((N,), -1, dtype=np.int32)
            acts[: chunk.size] = chunk.astype(np.int32)
            runner.step(acts, None, None)
            if job.abort.is_set():
                return None
            for s in range(int(chunk.size)):
                i = c0 + s
                tau1[i] = int(buf.tau_frames[s])
                disc1[i] = self.gamma ** float(tau1[i])
                if int(buf.invalid_action[s]) != -1:
                    status[i] = _BLOCK_INVALID
                    continue
                r1[i] = self._step_reward(
                    s, v_prev=v1, v0=v0, elapsed_frames=frames1 + int(tau1[i]), buf=buf
                )
                if int(buf.terminated[s]) != 0 or int(buf.truncated[s]) != 0:
                    status[i] = _BLOCK_TERMINAL
                    term_q[i] = self._terminal_q(
                        reason=int(buf.terminal_reason[s]), ply=1, reward=float(r1[i])
                    )
                else:
                    obs1[i] = buf.obs[s]
                    boards1b[i] = buf.board_bytes[s]  # type: ignore[index]
                    maskF[i] = buf.feasible_mask[s]
                    costF[i] = buf.cost_to_lock[s]
                    virF[i] = int(buf.viruses_rem[s])
                    prevF[i] = buf.preview_colors[s]

        # ---- depth-1: V(s after a1 | preview pair) for all pairs in one pass
        q = np.full((F, 9), -np.inf, dtype=np.float64)
        refined = np.zeros((F, 9), dtype=bool)
        term_idx = np.flatnonzero(status == _BLOCK_TERMINAL)
        q[term_idx] = term_q[term_idx, None]
        refined[term_idx] = True  # exact ply-1 terminal values
        alive_idx = np.flatnonzero(status == _BLOCK_ALIVE)
        if alive_idx.size == 0:
            if not bool(np.isfinite(q).any()):
                return None
            return PonderResult(
                board_key=_board_key(board1),
                pill_raw=k_raw,
                actions=actions_full,
                q=q,
                refined=refined,
                prior_action=prior_action,
                depth=np.ones((9,), dtype=np.int8),
                wall_s=time.perf_counter() - t0,
            )
        if job.abort.is_set():
            return None
        v9 = self._pill_conditioned_values(
            obs1[alive_idx],
            maskF[alive_idx],
            speed_setting=job.speed_setting,
            level=job.level,
            frames=frames1 + tau1[alive_idx],
            viruses=virF[alive_idx],
            viruses_initial=v0,
        )
        q[alive_idx] = r1[alive_idx, None] + disc1[alive_idx, None] * v9.astype(np.float64)

        # ---- depth-2 per preview pair: ply-2 beam + marginalized leaves
        depth = np.ones((9,), dtype=np.int8)
        K = min(self.beam, int(alive_idx.size), N)
        block = max(1, N // K) if K > 0 else 1
        pk_cache: Dict[Tuple[int, bool], Any] = {}

        def _packed_for(i: int, same_p: bool) -> Any:
            pk = pk_cache.get((i, same_p))
            if pk is None:
                m = maskF[i].reshape(4, GRID_H, GRID_W).astype(bool).copy()
                if same_p and bool(m[:2].any()):
                    m[2:] = False
                pk = pack_feasible_candidates(
                    m,
                    costF[i].reshape(4, GRID_H, GRID_W).astype(np.float32),
                    max_candidates=self.candidate_max,
                    sort_by_cost=True,
                )
                pk_cache[(i, same_p)] = pk
            return pk

        for cidx in range(9):
            if job.abort.is_set():
                return None
            if K == 0 or time.perf_counter() - t0 >= self.ponder_budget_s:
                break
            p_can = (cidx // 3, cidx % 3)
            p_raw = (_COLOR_SWAP[p_can[0]], _COLOR_SWAP[p_can[1]])
            same_p = p_can[0] == p_can[1]
            order = np.argsort(-prior_full[cidx], kind="stable")
            beam = [int(i) for i in order if status[i] == _BLOCK_ALIVE][:K]
            if not beam:
                continue
            B2 = len(beam)
            obs2 = obs1[beam].copy()
            if same_p:
                obs2[:, 6:8] = 0.0
            ca2 = np.full((B2, self.candidate_max), -1, dtype=np.int32)
            cm2 = np.zeros((B2, self.candidate_max), dtype=np.bool_)
            cc2 = np.zeros((B2, self.candidate_max), dtype=np.float32)
            infos2 = []
            for b, i in enumerate(beam):
                pk = _packed_for(i, same_p)
                ca2[b], cm2[b], cc2[b] = pk.actions, pk.mask, pk.cost
                infos2.append(
                    self._sim_info(
                        speed_setting=job.speed_setting,
                        level=job.level,
                        frames_used=frames1 + int(tau1[i]),
                        viruses_initial=v0,
                        viruses_remaining=int(virF[i]),
                        options=int(cm2[b].sum()),
                    )
                )
            logits2, _ = self._forward_full(
                obs2,
                np.repeat(np.array([p_can], dtype=np.int64), B2, axis=0),
                np.clip(prevF[beam], 0, 2),
                ca2,
                cc2,
                cm2,
                infos2,
            )

            specs_b: List[object] = []
            for j in range(N):
                b = j // block
                if b >= B2:
                    specs_b.append(_spec(board1, k_raw, mask1_u8, cost1))  # parked
                    continue
                i = beam[b]
                specs_b.append(_spec(boards1b[i], p_raw, maskF[i], costF[i]))
            runner.reset(None, specs_b)
            acts2 = np.full((N,), -1, dtype=np.int32)
            n2 = np.zeros((B2,), dtype=np.int64)
            for b in range(B2):
                lg = logits2[b].copy()
                lg[~cm2[b]] = -np.inf
                o2 = np.argsort(-lg, kind="stable")
                o2 = o2[cm2[b][o2]][:block]
                n2[b] = int(o2.size)
                for s, slot in enumerate(o2):
                    acts2[b * block + s] = int(ca2[b][slot])
            runner.step(acts2, None, None)
            if job.abort.is_set():
                return None

            tau1_beam = tau1[beam]
            leaf_envs: List[int] = []
            leaf_r2: Dict[int, float] = {}
            leaf_disc2: Dict[int, float] = {}
            leaves: List[List[float]] = [[] for _ in range(B2)]
            for b in range(B2):
                i = beam[b]
                for s in range(int(n2[b])):
                    j = b * block + s
                    if int(buf.invalid_action[j]) != -1:
                        continue
                    tau2 = int(buf.tau_frames[j])
                    r2 = self._step_reward(
                        j,
                        v_prev=int(virF[i]),
                        v0=v0,
                        elapsed_frames=frames1 + int(tau1[i]) + tau2,
                        buf=buf,
                    )
                    if int(buf.terminated[j]) != 0 or int(buf.truncated[j]) != 0:
                        leaves[b].append(
                            self._terminal_q(
                                reason=int(buf.terminal_reason[j]), ply=2, reward=float(r2)
                            )
                        )
                    else:
                        leaf_envs.append(j)
                        leaf_r2[j] = float(r2)
                        leaf_disc2[j] = self.gamma ** float(tau2)
            if leaf_envs:
                vals = self._marginal_leaf_values(
                    leaf_envs,
                    speed_setting=job.speed_setting,
                    level=job.level,
                    frames_used_base=frames1,
                    tau1=tau1_beam,
                    block=block,
                    viruses_initial=v0,
                    buf=buf,
                )
                for j, v in zip(leaf_envs, vals):
                    leaves[j // block].append(leaf_r2[j] + leaf_disc2[j] * float(v))
            for b in range(B2):
                if leaves[b]:
                    i = beam[b]
                    q[i, cidx] = r1[i] + disc1[i] * max(leaves[b])
                    refined[i, cidx] = True
            depth[cidx] = 2

        return PonderResult(
            board_key=_board_key(board1),
            pill_raw=k_raw,
            actions=actions_full,
            q=q,
            refined=refined,
            prior_action=prior_action,
            depth=depth,
            wall_s=time.perf_counter() - t0,
        )

    def _pill_conditioned_values(
        self,
        obs: np.ndarray,
        masks_flat: np.ndarray,
        *,
        speed_setting: int,
        level: int,
        frames: np.ndarray,
        viruses: np.ndarray,
        viruses_initial: int,
    ) -> np.ndarray:
        """[B, 9] values conditioned on the falling-pill canonical color pair,
        with the (unknown) preview pair marginalized — one trunk pass."""

        B = int(np.asarray(obs).shape[0])
        infos_n: List[Dict[str, Any]] = []
        infos_z: List[Dict[str, Any]] = []
        for i in range(B):
            mask = np.asarray(masks_flat[i]).reshape(4, -1)
            opt_n = int(mask.sum())
            opt_z = int(mask[:2].sum()) if bool(mask[:2].any()) else opt_n
            common = dict(
                speed_setting=speed_setting,
                level=level,
                frames_used=int(frames[i]),
                viruses_initial=int(viruses_initial),
                viruses_remaining=int(viruses[i]),
            )
            infos_n.append(self._sim_info(options=opt_n, **common))
            infos_z.append(self._sim_info(options=opt_z, **common))
        v81 = self._combo_values(np.ascontiguousarray(obs, dtype=np.float32), infos_n, infos_z)
        with torch.inference_mode():
            v9 = v81.view(B, 9, 9).mean(dim=2)
        return v9.float().cpu().numpy()


__all__ = [
    "PonderingSearchPolicy",
    "PonderResult",
    "SearchPolicy",
    "aux_v1_batch_fast",
    "backup_block_values",
]
