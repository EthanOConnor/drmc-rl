"""Frozen Strong League continuation mixture for strict pair search.

The frozen mixture is a privileged teacher: its legacy auxiliary vector uses
pending-attack counts. Reserve bytes remain exclusive to restore/chance
transitions and never enter network input.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from drmc_rl.game.observation import board_bytes_to_semantic_planes, legacy_vs_policy_boards
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.search.joint_event import WDL
from drmc_rl.search.native_pair import NativePairSearchState
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DavidsonCalibration:
    """Three-outcome Davidson link fitted on held-out native match states."""

    slope: float
    bias: float
    draw_logit: float
    artifact_sha256: str

    def wdl(self, score: float) -> WDL:
        strength = float(np.clip(self.slope * float(score) + self.bias, -30.0, 30.0))
        logits = np.asarray((strength, self.draw_logit, -strength), dtype=np.float64)
        return WDL.from_logits(logits)

    @classmethod
    def from_path(cls, path: Path) -> "DavidsonCalibration":
        payload = json.loads(path.read_text())
        if payload.get("schema") != "drmc-strong-league-wdl-calibration-v1":
            raise ValueError(f"unsupported W/D/L calibration schema in {path}")
        return cls(
            slope=float(payload["parameters"]["slope"]),
            bias=float(payload["parameters"]["bias"]),
            draw_logit=float(payload["parameters"]["draw_logit"]),
            artifact_sha256=_sha256(path),
        )


@dataclass(frozen=True, slots=True)
class MixtureMember:
    id: str
    checkpoint: Path
    sha256: str
    weight: float


class _FrozenMember:
    def __init__(self, member: MixtureMember, *, device: str) -> None:
        from tools.eval_policy import _build_net_from_cfg

        if _sha256(member.checkpoint) != member.sha256:
            raise ValueError(f"checkpoint hash mismatch for mixture member {member.id}")
        payload = load_checkpoint(member.checkpoint, map_location="cpu")
        cfg = dict(payload.get("cfg") or {})
        sp = cfg.get("smdp_ppo", cfg)
        board_channels = int(sp.get("candidate_board_channels", 8))
        self.in_channels = board_channels + 4
        self.net, self.aux_dim, self.candidate_max = _build_net_from_cfg(
            cfg, self.in_channels, device
        )
        self.net.load_state_dict(payload.get("ema_state_dict") or payload["state_dict"])
        self.net.eval()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.member = member
        self.device = device

    def infer(
        self,
        obs: np.ndarray,
        pill: np.ndarray,
        preview: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        mask: np.ndarray,
        aux: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        logits, values = self.infer_batch(
            obs[None], pill[None], preview[None], actions[None], costs[None],
            mask[None], aux[None],
        )
        return logits[0], float(values[0])

    def infer_batch(
        self,
        obs: np.ndarray,
        pill: np.ndarray,
        preview: np.ndarray,
        actions: np.ndarray,
        costs: np.ndarray,
        mask: np.ndarray,
        aux: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            logits, value = self.net(
                torch.from_numpy(obs).to(self.device),
                torch.from_numpy(pill).to(self.device),
                torch.from_numpy(preview).to(self.device),
                torch.from_numpy(actions).to(self.device),
                torch.from_numpy(costs).to(self.device),
                torch.from_numpy(mask).to(self.device),
                aux=(None if self.aux_dim == 0 else torch.from_numpy(aux).to(self.device)),
            )
        return logits.float().cpu().numpy(), value.reshape(-1).float().cpu().numpy()


class FrozenStrongLeagueMixture:
    """Immutable policy/value ensemble with held-out W/D/L calibration."""

    def __init__(
        self,
        members: Sequence[MixtureMember],
        calibration: DavidsonCalibration,
        *,
        device: str = "cpu",
        cache_size: int = 8192,
    ) -> None:
        if not members or any(member.weight <= 0 for member in members):
            raise ValueError("continuation mixture requires positive member weights")
        total = sum(member.weight for member in members)
        self.weights = np.asarray([member.weight / total for member in members], dtype=np.float64)
        self.members = tuple(_FrozenMember(member, device=device) for member in members)
        if len({member.member.sha256 for member in self.members}) != len(self.members):
            raise ValueError("continuation mixture contains duplicate checkpoints")
        self.calibration = calibration
        self.cache_size = int(max(1, cache_size))
        self._cache: OrderedDict[tuple[str, int], tuple[dict[int, float], float]] = OrderedDict()

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        calibration_path: Path,
        *,
        device: str = "cpu",
    ) -> "FrozenStrongLeagueMixture":
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != "drmc-strong-league-continuation-mixture-v1":
            raise ValueError(f"unsupported continuation mixture schema in {manifest_path}")
        base = manifest_path.parent
        members = []
        for item in payload["members"]:
            checkpoint = Path(item["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = base / checkpoint
            members.append(
                MixtureMember(
                    id=str(item["id"]),
                    checkpoint=checkpoint,
                    sha256=str(item["sha256"]),
                    weight=float(item["weight"]),
                )
            )
        return cls(
            members,
            DavidsonCalibration.from_path(calibration_path),
            device=device,
        )

    def prior(
        self, state: NativePairSearchState, side: int, actions: Sequence[int]
    ) -> Sequence[float]:
        probabilities, _score = self._infer(state, int(side))
        floor = 1e-8
        return tuple(max(floor, probabilities.get(int(action), floor)) for action in actions)

    def evaluate(self, state: NativePairSearchState, root_side: int) -> WDL:
        root_side = int(root_side)
        need = state.privileged.need_action
        if not any(need):
            raise ValueError("Strong League value requires an actionable decision boundary")
        # The frozen value was trained only where its own side can act. At an
        # asynchronous opponent decision, evaluate that supported perspective
        # and reverse W/L after applying its calibrated link.
        value_side = root_side if need[root_side] else 1 - root_side
        _probabilities, score = self._infer(state, value_side)
        value = self.calibration.wdl(score)
        return value if value_side == root_side else WDL(value.loss, value.draw, value.win)

    def infer_batch(
        self, requests: Sequence[tuple[NativePairSearchState, int]],
    ) -> list[tuple[dict[int, float], float]]:
        """Batch independent acting sides without changing the frozen mixture.

        Every frontier is retained. Candidate padding carries zero probability;
        it never participates in normalization or action selection.
        """
        if not requests:
            return []
        inputs = []
        for state, side in requests:
            if not state.privileged.need_action[side] or not state.legal_actions_by_side[side]:
                raise ValueError("Strong League inference requires an acting side with legal candidates")
            inputs.append(_policy_inputs(state, side))
        width = max(len(item[3]) for item in inputs)
        batch = []
        for index in range(7):
            values = []
            for item in inputs:
                value = item[index]
                if index in (3, 4, 5):
                    fill = 0xFFFF if index == 4 else 0
                    value = np.pad(value, (0, width-len(value)), constant_values=fill)
                values.append(value)
            batch.append(np.stack(values))
        mask = batch[5]
        probability = np.zeros(mask.shape, dtype=np.float64)
        value = np.zeros(len(requests), dtype=np.float64)
        for weight, member in zip(self.weights, self.members, strict=True):
            logits, member_value = member.infer_batch(*batch)
            if not np.isfinite(logits[mask]).all() or not np.isfinite(member_value).all():
                raise ValueError("frozen mixture returned non-finite predictions")
            logits = np.where(mask, logits, -np.inf).astype(np.float64)
            scores = np.exp(np.clip(logits-logits.max(axis=1, keepdims=True), -60, 0))
            scores[~mask] = 0
            scores /= scores.sum(axis=1, keepdims=True)
            probability += weight*scores
            value += weight*member_value
        return [
            (dict(zip(map(int, actions[valid]), map(float, weights[valid]), strict=True)),
             float(score))
            for actions, valid, weights, score in zip(batch[3], mask, probability, value, strict=True)
        ]

    def _infer(self, state: NativePairSearchState, side: int) -> tuple[dict[int, float], float]:
        if not state.privileged.need_action[side] or not state.legal_actions_by_side[side]:
            raise ValueError("Strong League inference requires an acting side with legal candidates")
        key = (hashlib.sha256(state.privileged.engine_checkpoint).hexdigest(), side)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        obs, pill, preview, actions, costs, mask, aux = _policy_inputs(state, side)
        member_probabilities: list[np.ndarray] = []
        member_values: list[float] = []
        for member in self.members:
            logits, value = member.infer(obs, pill, preview, actions, costs, mask, aux)
            valid_logits = logits[mask]
            valid_logits = valid_logits - valid_logits.max(initial=0.0)
            probability = np.exp(np.clip(valid_logits, -60.0, 0.0))
            probability /= probability.sum()
            expanded = np.zeros(len(actions), dtype=np.float64)
            expanded[mask] = probability
            member_probabilities.append(expanded)
            member_values.append(value)
        mixture = self.weights @ np.asarray(member_probabilities)
        by_action = {
            int(action): float(probability)
            for action, probability, valid in zip(actions, mixture, mask, strict=True)
            if valid
        }
        result = (by_action, float(self.weights @ np.asarray(member_values)))
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result


def _policy_inputs(state: NativePairSearchState, side: int):
    public = state.privileged.public
    own = public.sides[side]
    opponent = public.sides[1 - side]
    own_planes = board_bytes_to_semantic_planes(own.board)
    opponent_planes = board_bytes_to_semantic_planes(opponent.board)
    legal = state.legal_actions_by_side[side]
    costs_legal = state.action_costs_by_side[side]
    feasible = np.zeros(512, dtype=bool)
    costs = np.full(512, 0xFFFF, dtype=np.uint16)
    feasible[list(legal)] = True
    costs[list(legal)] = np.asarray(costs_legal, dtype=np.uint16)
    pill = np.asarray(own.pill, dtype=np.int64)
    preview = np.asarray(own.preview, dtype=np.int64)
    if pill[0] == pill[1]:
        feasible.reshape(4, 16, 8)[2:] = False
        costs.reshape(4, 16, 8)[2:] = 0xFFFF
    packed = pack_feasible_candidates(
        feasible.reshape(4, 16, 8),
        costs.reshape(4, 16, 8),
        max_candidates=max(128, int(feasible.sum())),
        sort_by_cost=True,
    )
    obs = legacy_vs_policy_boards(own_planes, opponent_planes, pill, opponent.pill)
    aux = _aux_v1_vs(state, side, obs, feasible)
    return (
        obs,
        pill,
        preview,
        packed.actions,
        packed.cost,
        packed.mask,
        aux,
    )


def _aux_v1_vs(
    state: NativePairSearchState, side: int, obs: np.ndarray, feasible: np.ndarray
) -> np.ndarray:
    out = np.zeros(72, dtype=np.float32)
    own = obs[:8]
    virus = own[3] > 0.5
    colors = own[:3] > 0.5
    occupancy = colors.any(axis=0)
    virus_by_color = (colors & virus[None]).reshape(3, -1).sum(axis=1)

    def heights(mask: np.ndarray) -> np.ndarray:
        occupied = mask.any(axis=0)
        first = mask.argmax(axis=0)
        return np.where(occupied, 16 - first, 0).astype(np.float32)

    column_heights = heights(occupancy)
    virus_heights = heights(virus)
    viruses = float(virus.sum())
    initial = float(state.viruses_initial[side])
    k = 0
    out[k + state.speed_setting] = 1.0
    k += 3
    out[k] = np.clip(viruses / 84.0, 0.0, 1.0)
    k += 1
    out[k : k + 3] = np.clip(virus_by_color / 84.0, 0.0, 1.0)
    k += 3
    out[k + state.level + 15] = 1.0
    k += 36
    out[k] = np.tanh(state.privileged.pair_clocks[side] / 8000.0)
    k += 1
    out[k] = np.clip(column_heights.max(initial=0.0) / 16.0, 0.0, 1.0)
    k += 1
    out[k : k + 8] = np.clip(column_heights / 16.0, 0.0, 1.0)
    k += 8
    out[k] = np.clip((initial - viruses) / initial if initial > 0 else 0.0, 0.0, 1.0)
    k += 1
    out[k] = np.clip(feasible.sum() / 512.0, 0.0, 1.0)
    k += 1
    out[k] = np.clip(occupancy.sum() / 128.0, 0.0, 1.0)
    k += 1
    out[k] = np.clip(virus_heights.max(initial=0.0) / 16.0, 0.0, 1.0)
    k += 1
    opponent_side = 1 - side
    out[k] = np.clip(
        state.privileged.public.sides[opponent_side].viruses_remaining / 84.0, 0.0, 1.0
    )
    k += 1
    out[k] = np.clip(state.privileged.pending_attacks[side] / 4.0, 0.0, 1.0)
    k += 1
    out[k] = np.clip(state.privileged.pending_attacks[opponent_side] / 4.0, 0.0, 1.0)
    k += 1
    opponent = state.privileged.public.sides[opponent_side]
    for values in (opponent.pill, opponent.preview):
        for color in values:
            if 0 <= int(color) < 3:
                out[k + int(color)] = 1.0
            k += 3
    if k != 72:
        raise RuntimeError(f"aux_v1_vs packing mismatch: {k}")
    return out


def frozen_strong_league_factory(args: Any):
    """CLI adapter for promotion-quality continuation labels."""

    if not args.mixture_manifest or not args.wdl_calibration:
        raise ValueError("frozen Strong League adapter requires mixture and calibration paths")
    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
    from drmc_rl.search.native_pair import NativePairSearchModel, state_from_payload

    mixture = FrozenStrongLeagueMixture.from_manifest(
        Path(args.mixture_manifest),
        Path(args.wdl_calibration),
        device=str(args.device),
    )
    model = NativePairSearchModel(
        DrMarioVsPoolRunner(num_pairs=1),
        continuation=mixture,
        reveal_chance=True,
    )
    return model, state_from_payload


__all__ = [
    "DavidsonCalibration",
    "FrozenStrongLeagueMixture",
    "MixtureMember",
    "frozen_strong_league_factory",
]
