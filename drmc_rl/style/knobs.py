"""Decision-time knobs: a registry of versioned, stackable per-candidate biases.

A knob is identified by ``id@version`` (e.g. ``showy-t2@1``). It maps one
decision's inputs (the root bottle, the pill, the candidate actions and the
legal mask) plus its model spec to a per-candidate bias that is centered over
the legal candidates. An entrant (or any player) lists knobs in order::

    knobs: [{"id": "showy-t2", "version": 1, "lambda": 1.5, "model": <inline spec | "sha256:<hex>">}, ...]

and its logits become ``logits + sum_i lambda_i * bias_i`` over legal candidates.
An empty list, or every lambda = 0, installs nothing: the player is
byte-identical to the unbiased one. A new knob or a changed algorithm is a new
registry entry or version, and so a new worker capability ``knob:<id>@<version>``.

Model specs are inline JSON (workers never read local files) or the sha256 of
a model bundled in ``drmc_rl/style/models`` (canonical JSON). Everything here
is numpy/Python, without torch.

Browser mirror (professorPills, later): the web build's single ``showy`` lambda
maps to ``knobs: [{id: "showy-t2", version: 1, lambda, model: "sha256:..."}]``;
the browser config should carry the same list and bundle the models by hash.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

MODELS = Path(__file__).with_name("models")
MAX_MODEL_BYTES = 65536
KNOB_KEYS = ("id", "version", "lambda", "model", "tier_bar")


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def model_hash(spec: dict) -> str:
    return hashlib.sha256(canonical(spec).encode()).hexdigest()


@dataclass(frozen=True)
class Knob:
    id: str
    version: int
    schema: str                       # the model-spec schema this knob accepts
    tier_bar: float                   # default immediate-clear override bar (inf: none)
    bias: Callable                    # (model, root_field, pill, actions, mask, tier_bar[, decision]) -> centered bias [K]
    summary: str
    default_model: str = ""           # bundled model file used when an entrant names none

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def capability(self) -> str:
        return f"knob:{self.key}"


def _showy_bias(model, root_field, pill, actions, mask, tier_bar, decision=None):
    from drmc_rl.style.showy_knob import showy_bias
    return showy_bias(model, root_field, pill, actions, mask, 1.0, tier_bar=tier_bar, decision=decision)


def _quad_bias(model, root_field, pill, actions, mask, tier_bar, decision=None):
    from drmc_rl.style.showy_knob import quad_bias
    return quad_bias(model, root_field, pill, actions, mask, 1.0, tier_bar=tier_bar, decision=decision)


REGISTRY: dict[str, Knob] = {k.key: k for k in (
    Knob("showy-t2", 1, "drmc-showy-knob-v1", 30.0, _showy_bias,
         "P(T2+ clear within 4 placements | afterstate), human-fit logistic model; a T2+ clear now counts as certain",
         "showy_t2k4_v1.json"),
    Knob("showy-hcombo", 1, "drmc-showy-knob-v1", math.inf, _showy_bias,
         "P(combo with a horizontal line within 4 placements | afterstate), human-fit logistic model",
         "showy_hc_k4_v1.json"),
    Knob("showy-quad", 1, "drmc-showy-knob-v1", 4.0, _quad_bias,
         "P(quad: an attack of 4+ matched lines, the ROM's 4-piece cap, within 4 placements | afterstate), "
         "human-fit logistic model; a 4+ line attack now counts as certain, optional per-wasted-line penalty "
         "(model waste_penalty)",
         "showy_quad_k4_v1.json"),
)}


def capabilities() -> set[str]:
    return {k.capability for k in REGISTRY.values()}


def bundled_models() -> dict[str, dict]:
    out = {}
    for path in sorted(MODELS.glob("*.json")):
        spec = json.loads(path.read_text())
        out[model_hash(spec)] = spec
    return out


def default_model_ref(key: str) -> str:
    """``sha256:<hash>`` of a registered knob's bundled default model."""
    knob = REGISTRY[key]
    return "sha256:" + model_hash(json.loads((MODELS / knob.default_model).read_text()))


def parse(text: str, model=None) -> dict:
    """``id@version:lambda`` -> a knob entry (bundled default model unless given)."""
    key, _, lam = text.partition(":")
    ident, _, version = key.partition("@")
    if not (ident and version.isdigit() and lam):
        raise ValueError(f"knob {text!r} must be id@version:lambda")
    entry = {"id": ident, "version": int(version), "lambda": float(lam)}
    entry["model"] = model or default_model_ref(f"{ident}@{version}")
    return entry


def resolve_model(ref) -> dict:
    if isinstance(ref, dict):
        return ref
    if isinstance(ref, str) and ref.startswith("sha256:"):
        spec = bundled_models().get(ref[7:])
        if spec is None:
            raise ValueError(f"no bundled knob model {ref[:19]}...")
        return spec
    raise ValueError("a knob model is an inline spec or sha256:<hash> of a bundled model (never a path)")


def validate_knobs(knobs, *, known=None) -> list[dict]:
    """Check an entrant's knob list against the registry (``known``: allowed ``id@version`` keys)."""
    if knobs is None:
        return []
    if not isinstance(knobs, list):
        raise ValueError("knobs must be a list of {id, version, lambda, model}")
    known = set(REGISTRY) if known is None else set(known)
    seen = set()
    for entry in knobs:
        if not isinstance(entry, dict) or set(entry) - set(KNOB_KEYS) or not {"id", "version", "lambda", "model"} <= set(entry):
            raise ValueError(f"knob entries are {{id, version, lambda, model[, tier_bar]}}: {entry!r:.80}")
        key = f"{entry['id']}@{entry['version']}"
        if key not in known or key not in REGISTRY:
            raise ValueError(f"unknown knob {key}")
        if key in seen:
            raise ValueError(f"knob {key} listed twice")
        seen.add(key)
        lam = entry["lambda"]
        if isinstance(lam, bool) or not isinstance(lam, (int, float)) or not math.isfinite(lam):
            raise ValueError(f"knob {key} lambda must be a finite number")
        if "tier_bar" in entry and (isinstance(entry["tier_bar"], bool) or not isinstance(entry["tier_bar"], (int, float))):
            raise ValueError(f"knob {key} tier_bar must be a number")
        spec = resolve_model(entry["model"])
        n = len(spec.get("features", []))
        if spec.get("schema") != REGISTRY[key].schema or not n or \
                any(len(spec.get(k, [])) != n for k in ("mean", "scale", "coef")) or len(canonical(spec)) > MAX_MODEL_BYTES:
            raise ValueError(f"knob {key}: malformed, oversized or wrong-schema model spec")
    return knobs


def active(knobs) -> list[dict]:
    return [k for k in (knobs or []) if k["lambda"] != 0]


def requirements(knobs) -> set[str]:
    return {f"knob:{k['id']}@{k['version']}" for k in active(knobs)}


def suffix(knobs) -> str:
    """Identity suffix, e.g. ``+showy-t2@1:1.5+showy-hcombo@1:0.5`` ('' without active knobs)."""
    return "".join(f"+{k['id']}@{k['version']}:{k['lambda']:g}" for k in active(knobs))


def total_bias(knobs, root_field, pill, actions, mask, *, models=None) -> np.ndarray:
    """sum_i lambda_i * bias_i for one decision (each bias centered over legal candidates).

    The knobs share one ``Decision``: candidate afterstates and features are computed once
    per decision (natively when ``drmc_rl.style.native`` is built), not once per knob.
    """
    from drmc_rl.style.showy_knob import Decision
    out = np.zeros(np.asarray(actions).shape, np.float32)
    decision = Decision(root_field, pill, actions, mask)
    for i, entry in enumerate(active(knobs)):
        knob = REGISTRY[f"{entry['id']}@{entry['version']}"]
        model = models[i] if models is not None else load_model(entry)
        bar = float(entry.get("tier_bar", knob.tier_bar))
        out += np.float32(entry["lambda"]) * knob.bias(model, root_field, pill, actions, mask, bar, decision).astype(np.float32)
    return out


def load_model(entry):
    from drmc_rl.style.showy_knob import ShowyModel
    return ShowyModel.load(resolve_model(entry["model"]))


class KnobPolicy:
    """Wrap a candidate-scoring actor (``score(obs, infos) -> (actions, mask, logits)``) with knobs."""

    def __init__(self, inner, knobs):
        self.inner, self.knobs = inner, active(knobs)
        self.models = [load_model(k) for k in self.knobs]

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score(self, obs, infos):
        ca, cm, lg = self.inner.score(obs, infos)
        return self.adjust(obs, infos, ca, cm, lg)

    def adjust(self, obs, infos, ca, cm, lg):
        """Add the knob biases to scores the inner actor produced for these rows."""
        from drmc_rl.game.afterstate import planes_to_fields
        fields = planes_to_fields(np.asarray(obs)[:, :8])
        lg = np.array(lg, dtype=np.float32, copy=True)
        for i, info in enumerate(infos):
            pill = np.asarray(info["next_pill_colors"], np.int64)
            b = total_bias(self.knobs, fields[i], pill, ca[i], cm[i], models=self.models)
            lg[i] = np.where(cm[i], lg[i] + b, lg[i])
        return ca, cm, lg


def apply(actor, knobs):
    """The actor with its active knobs installed; unchanged (the same object) when none are active."""
    validate_knobs(knobs)
    return KnobPolicy(actor, knobs) if active(knobs) else actor
