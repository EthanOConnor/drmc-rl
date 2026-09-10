"""Frozen per-pace public cores and residuals for the live competitive scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from drmc_rl.execution.pace import BY_ID
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.models.policy.pace_adapter import PaceAdapter

PORTFOLIO_SCHEMA = "professor-pills-pace-opponents-v1"
CORE_PORTFOLIO_SCHEMA = "professor-pills-pace-opponents-v2"
MOTOR_SCHEMA = "own-motor-gravity-v1"


def read_portfolio(path, parent_sha256):
    """Validate the portable manifest and every artifact before loading weights.

    Model paths are relative to the manifest's directory, or explicit absolute
    paths during source development. Public identities omit those local paths.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if (data.get("schema") not in (PORTFOLIO_SCHEMA, CORE_PORTFOLIO_SCHEMA)
            or data.get("context_schema") != MOTOR_SCHEMA
            or data.get("parent_sha256") != parent_sha256):
        raise ValueError("pace portfolio belongs to a different parent or schema")
    adapters, paces = data.get("adapters", {}), data.get("paces", {})
    cores = data.get("cores", {})
    if (set(adapters) & set(cores) or (cores and data["schema"] != CORE_PORTFOLIO_SCHEMA)
            or any(c.get("context_schema") != PUBLIC_CONTEXT_SCHEMA for c in cores.values())):
        raise ValueError("core routes require distinct identities and the public context schema")
    if (set(paces) != set(BY_ID)
            or any(name is not None and name not in adapters and name not in cores for name in paces.values())
            or any(paces[p] is not None and paces[p] not in cores for p in ("super_human", "frame_perfect"))):
        raise ValueError("pace portfolio must cover every pace; fastest parent overrides require public cores")
    paths, identities = {}, {}
    for name, item in {**adapters, **cores}.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError("invalid pace model identifier")
        artifact = (path.parent / item["path"]).resolve()
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"pace model hash mismatch: {name}")
        paths[name] = artifact
        identities[name] = {"sha256": digest, "label": item.get("label", name)}
        if name in cores:
            identities[name]["context_schema"] = PUBLIC_CONTEXT_SCHEMA
    identity = {"schema": data["schema"], "parent_sha256": parent_sha256,
                "context_schema": MOTOR_SCHEMA,
                "adapters": {n: identities[n] for n in adapters}, "paces": paces}
    if data["schema"] == CORE_PORTFOLIO_SCHEMA:
        identity["cores"] = {n: identities[n] for n in cores}
    return identity, paths


class _PortfolioCore(nn.Module):
    def __init__(self, base, adapters):
        super().__init__()
        self.base, self.adapters = base, nn.ModuleDict(adapters)
        self.selected, self.motor = None, None

    def forward(self, *args, **kwargs):
        if self.selected is None:
            return self.base(*args, **kwargs)
        logits, value, features = self.base(*args, **kwargs, return_aux=True)
        return self.adapters[self.selected](
            features["candidate_context"], features["global_context"],
            self.motor.to(logits.device), logits, value.reshape(-1), args[5].bool(),
        )


class PacePortfolio:
    """Synchronous live scorer; each controller batch has one selected pace.

    Legacy residuals share one parent. Complete public cores are selected before
    observation encoding; lower-skill decoding remains outside this scorer.
    """

    def __init__(self, plain, manifest, parent_sha256):
        if plain.aux_spec == PUBLIC_CONTEXT_SCHEMA:
            raise ValueError("these pace residuals require their original zero-aux parent")
        self.identity, paths = read_portfolio(manifest, parent_sha256)
        adapters = {}
        for name in self.identity["adapters"]:
            path = paths[name]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("schema") != "drmc-pace-adapter-v1"
                    or payload.get("context_schema") != MOTOR_SCHEMA
                    or payload.get("parent_sha256") != parent_sha256):
                raise ValueError(f"pace adapter belongs to a different parent or schema: {name}")
            adapter = PaceAdapter(plain.net.d_model, **payload.get("adapter_config", {}))
            adapter.load_state_dict(payload["state_dict"], strict=True)
            adapters[name] = adapter.to(plain.device).eval()
        self.plain = plain
        self.cores = {}
        from tools.vs_head_to_head import PlainPolicy
        for name in self.identity.get("cores", {}):
            policy = PlainPolicy(paths[name], device=plain.device, public_only=True)
            if policy.aux_spec != PUBLIC_CONTEXT_SCHEMA:
                raise ValueError(f"core route lacks its declared public context: {name}")
            self.cores[name] = policy
        self.core = _PortfolioCore(plain.net, adapters).eval()
        plain.net = self.core
        seen, self.warmup_paces = set(), []
        for pace, name in self.identity["paces"].items():
            if name not in seen:
                seen.add(name)
                self.warmup_paces.append(pace)

    def for_pace(self, pace):
        """Select the complete policy before encoding observations or planning."""
        if pace not in self.identity["paces"]:
            raise ValueError("unknown portfolio pace")
        return self.cores.get(self.identity["paces"][pace], self)

    def __getattr__(self, name):
        return getattr(self.plain, name)

    def score(self, observations, infos):
        paces = {info["pace/id"] for info in infos}
        if len(paces) != 1 or not paces <= self.identity["paces"].keys():
            raise ValueError("a controller batch must use one known pace")
        pace = paces.pop()
        if self.identity["paces"][pace] in self.cores:
            raise ValueError("select the public core before encoding its controller inputs")
        motor = np.asarray([info["pace/context"] for info in infos], dtype=np.float32)
        enabled = pace not in ("super_human", "frame_perfect")
        if (motor.shape != (len(infos), 8) or not np.isfinite(motor).all()
                or np.any(motor[:, -1] != float(enabled))):
            raise ValueError("invalid motor context for the selected pace")
        self.core.selected = self.identity["paces"][pace]
        self.core.motor = torch.as_tensor(motor)
        try:
            return self.plain.score(observations, infos)
        finally:
            self.core.selected = self.core.motor = None
