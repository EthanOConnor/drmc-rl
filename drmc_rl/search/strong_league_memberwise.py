"""Belief-aware Strong League adapters for quality and uncertainty releases.

The frozen G4 checkpoints were trained with exact native pending-attack scalars.
They are therefore a *privileged continuation teacher*, not a deployable
public-information search evaluator.  Reserve reveals are handled separately by
:class:`BeliefNativePairSearchModel` using a public seed posterior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
from drmc_rl.search.native_pair import NativePairSearchState, state_from_payload
from drmc_rl.search.pill_belief import PillReserveBelief
from drmc_rl.search.strong_league import (
    DavidsonCalibration,
    FrozenStrongLeagueMixture,
    MixtureMember,
)
from drmc_rl.teachers.counterfactual import WeightedTeacherModels

INFORMATION_SCOPE = "privileged-pending-attack-continuation-v1"


def read_mixture_members(manifest_path: Path) -> tuple[MixtureMember, ...]:
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != "drmc-strong-league-continuation-mixture-v1":
        raise ValueError(f"unsupported continuation mixture schema in {manifest_path}")
    base = manifest_path.parent
    members: list[MixtureMember] = []
    for item in payload.get("members", ()):
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
    if not members:
        raise ValueError("continuation mixture manifest contains no members")
    if len({member.id for member in members}) != len(members):
        raise ValueError("continuation mixture member ids must be unique")
    if any(member.weight <= 0 for member in members):
        raise ValueError("continuation mixture weights must be positive")
    return tuple(members)


def _register_payload_belief(
    model: BeliefNativePairSearchModel,
    state: NativePairSearchState,
    payload: Mapping[str, Any],
) -> None:
    raw = payload.get("reserve_belief")
    if isinstance(raw, Mapping):
        model.register_belief(state, PillReserveBelief.from_dict(raw))


def frozen_strong_league_belief_factory(args: Any):
    """Aggregate mixture values with exact public reserve-belief branching."""

    if not args.mixture_manifest or not args.wdl_calibration:
        raise ValueError("Strong League adapter requires mixture and calibration paths")
    mixture = FrozenStrongLeagueMixture.from_manifest(
        Path(args.mixture_manifest),
        Path(args.wdl_calibration),
        device=str(args.device),
    )
    model = BeliefNativePairSearchModel(
        DrMarioVsPoolRunner(num_pairs=1), continuation=mixture
    )
    model.information_scope = INFORMATION_SCOPE

    def decode(payload: Mapping[str, Any]) -> NativePairSearchState:
        state = state_from_payload(payload)
        _register_payload_belief(model, state, payload)
        return state

    return model, decode


def frozen_strong_league_memberwise_factory(args: Any):
    """Run one complete search per frozen member and export weighted disagreement."""

    if not args.mixture_manifest or not args.wdl_calibration:
        raise ValueError("Strong League adapter requires mixture and calibration paths")
    members = read_mixture_members(Path(args.mixture_manifest))
    calibration = DavidsonCalibration.from_path(Path(args.wdl_calibration))
    models: list[BeliefNativePairSearchModel] = []
    for member in members:
        continuation = FrozenStrongLeagueMixture(
            (MixtureMember(member.id, member.checkpoint, member.sha256, 1.0),),
            calibration,
            device=str(args.device),
        )
        model = BeliefNativePairSearchModel(
            DrMarioVsPoolRunner(num_pairs=1), continuation=continuation
        )
        model.information_scope = INFORMATION_SCOPE
        models.append(model)
    ensemble = WeightedTeacherModels(
        models=tuple(models),
        weights=tuple(member.weight for member in members),
        ids=tuple(member.id for member in members),
    )

    def decode(payload: Mapping[str, Any]) -> NativePairSearchState:
        state = state_from_payload(payload)
        for model in models:
            _register_payload_belief(model, state, payload)
        return state

    return ensemble, decode


__all__ = [
    "INFORMATION_SCOPE",
    "frozen_strong_league_belief_factory",
    "frozen_strong_league_memberwise_factory",
    "read_mixture_members",
]
