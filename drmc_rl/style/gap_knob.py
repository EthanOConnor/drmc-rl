"""Skill-gap knobs: human-fit P(event within k | afterstate) models over showy + gap features.

The knobs-v2 family targets measured differences between the AI and strong humans
(``tools/skill_gaps``) rather than style. A model is a standardized logistic regression
(schema ``drmc-gap-knob-v1``) over any of

* ``drmc_rl.style.showy_knob`` board features (``FEATURE_NAMES``),
* its exact trigger features (``TRIGGER_NAMES``), and
* the gap features here (``GAP_NAMES``): spawn-column height and danger, covered and
  buried viruses, isolated (stranded) pill tiles, stranded edge viruses.

The knob's bias is ``sign * lambda * (logit V - mean over legal candidates)``: ``sign``
+1 pushes toward the event (e.g. a virus clear soon), -1 away from it (e.g. danger soon).
A registry entry may name an immediate rule that marks a candidate certain when the
placement itself completes the event (``virus_clear``: it clears a virus). A spec's optional
``max_root_viruses`` makes the knob inert on bottles with more viruses than that (the model
was fit only below it). Pure numpy.
"""
from __future__ import annotations

import numpy as np

from drmc_rl.style import showy_knob as sk

SCHEMA = "drmc-gap-knob-v1"
_EMPTY = 0xFF
GAP_NAMES = ("gap_h34", "gap_top3", "gap_near_topout", "gap_covered_viruses", "gap_buried", "gap_exposed_viruses",
             "gap_isolated", "gap_isolated_covered", "gap_edge_strand", "gap_pill_cells", "gap_viruses")


def _shift(a, dr, dc, fill):
    """out[r, c] = a[r + dr, c + dc] (``fill`` outside the bottle)."""
    out = np.full_like(a, fill)
    _, h, w = a.shape
    rs, re = max(0, -dr), min(h, h - dr)
    cs, ce = max(0, -dc), min(w, w - dc)
    out[:, rs:re, cs:ce] = a[:, rs + dr:re + dr, cs + dc:ce + dc]
    return out


def gap_features(fields) -> np.ndarray:
    """``[N,128]`` settled bottles (NES tile bytes) -> ``[N, len(GAP_NAMES)]`` float32."""
    f = np.asarray(fields, np.uint8).reshape(-1, 16, 8)
    occ = f != _EMPTY
    color = np.where(occ, f & 3, 3).astype(np.int8)
    virus = occ & ((f & 0xF0) == 0xD0)
    pill = occ & ~virus
    h = 16 - np.where(occ.any(axis=1), occ.argmax(axis=1), 16)
    above_occ = _shift(occ, -1, 0, False)
    above_color = _shift(color, -1, 0, 3)
    above_pill = _shift(pill, -1, 0, False)
    occ_above = np.cumsum(occ, axis=1) - occ
    nvir = virus.sum(axis=(1, 2))
    same = np.zeros_like(occ)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        same |= _shift(occ, dr, dc, False) & (_shift(color, dr, dc, 3) == color)
    iso = pill & ~same
    strand = np.zeros(len(f), np.float32)
    for i in np.flatnonzero((nvir >= 1) & (nvir <= 3)):
        for col in (0, 7):
            for r in np.flatnonzero(virus[i, :, col]):
                below = occ[i, r + 1:, col]
                if (int(below.argmax()) if below.any() else 15 - r) >= 4:
                    strand[i] = 1
    cols = (
        h[:, 3:5].max(axis=1),
        occ[:, :3, 3:5].sum(axis=(1, 2)),
        occ[:, :2, 3:5].any(axis=(1, 2)),
        (virus & above_pill & (above_color != color)).sum(axis=(1, 2)),
        (occ_above * virus).sum(axis=(1, 2)),
        (virus & (occ_above == 0)).sum(axis=(1, 2)),
        iso.sum(axis=(1, 2)),
        (iso & above_occ & (above_color != color)).sum(axis=(1, 2)),
        strand,
        pill.sum(axis=(1, 2)),
        nvir,
    )
    return np.stack([np.asarray(c, np.float32) for c in cols], axis=1)


def all_names() -> list[str]:
    if not sk.FEATURE_NAMES:
        sk.board_features(np.full((1, 128), _EMPTY, np.uint8))
    return list(sk.FEATURE_NAMES) + list(sk.TRIGGER_NAMES) + list(GAP_NAMES)


class GapModel:
    """Standardized logistic regression over ``all_names()`` columns (JSON spec)."""

    def __init__(self, spec: dict):
        if spec.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} model")
        self.spec = spec
        self.names = list(spec["features"])
        unknown = set(self.names) - set(all_names())
        if unknown:
            raise ValueError(f"unknown gap-knob features {sorted(unknown)[:3]}")
        self.mean = np.asarray(spec["mean"], np.float32)
        self.scale = np.asarray(spec["scale"], np.float32)
        self.coef = np.asarray(spec["coef"], np.float32)
        self.intercept = float(spec["intercept"])
        self.sign = float(spec.get("sign", 1.0))
        self.uses_triggers = any(k in sk.TRIGGER_NAMES for k in self.names)
        self.uses_gap = any(k in GAP_NAMES for k in self.names)
        pos = {k: i for i, k in enumerate(all_names())}
        self.index = np.asarray([pos[k] for k in self.names])

    @classmethod
    def load(cls, spec: dict) -> "GapModel":
        return cls(spec)

    def columns(self, fields, board=None, trig=None) -> np.ndarray:
        """``[board | trigger | gap]`` columns for settled bottles (unused blocks are zeros)."""
        fields = np.asarray(fields, np.uint8).reshape(-1, 128)
        n = len(fields)
        board = sk.board_features(fields) if board is None else board[:, :len(sk.FEATURE_NAMES)]
        if trig is None:
            trig = sk.trigger_features(fields) if self.uses_triggers else np.zeros((n, len(sk.TRIGGER_NAMES)), np.float32)
        gap = gap_features(fields) if self.uses_gap else np.zeros((n, len(GAP_NAMES)), np.float32)
        return np.concatenate([board, trig, gap], axis=1)

    def logit_columns(self, x) -> np.ndarray:
        return ((x[:, self.index] - self.mean) / self.scale) @ self.coef + self.intercept

    def logit(self, fields) -> np.ndarray:
        return self.logit_columns(self.columns(fields))

    def prob(self, fields) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(fields)))


def _viruses(fields) -> np.ndarray:
    return ((np.asarray(fields, np.uint8).reshape(-1, 128) & 0xF0) == 0xD0).sum(axis=1)


IMMEDIATE = {
    "none": None,
    "virus_clear": lambda root, after: _viruses(after) < _viruses(root)[0],   # the placement clears a virus
}


def gap_bias(model: GapModel, root_field, pill, actions, mask, lam: float, *, immediate: str = "none",
             eps: float = 1e-3, decision=None) -> np.ndarray:
    """Per-candidate ``sign * lam * (logit V - mean logit V over legal)`` for one decision ``[K]``."""
    actions = np.asarray(actions)
    bias = np.zeros(actions.shape, np.float32)
    d = decision if decision is not None else sk.Decision(root_field, pill, actions, mask)
    if lam == 0 or len(d.legal) == 0:
        return bias
    limit = model.spec.get("max_root_viruses")
    if limit is not None and _viruses(d.root)[0] > limit:
        return bias                     # outside the model's fitted range (e.g. an endgame-only knob)
    x = d.features(model.uses_triggers)
    after = d.after
    trig = x[:, len(sk.FEATURE_NAMES):] if model.uses_triggers else None
    cols = model.columns(after, board=x, trig=trig)
    z = model.logit_columns(cols).astype(np.float64)
    rule = IMMEDIATE[immediate]
    if rule is not None:
        z[np.asarray(rule(d.root, after)).reshape(-1)] = np.log((1 - eps) / eps)
    z = np.clip(z, -12.0, 12.0)
    bias[d.legal] = model.sign * lam * (z - z.mean())
    return bias


__all__ = ["GAP_NAMES", "GapModel", "SCHEMA", "all_names", "gap_bias", "gap_features"]
