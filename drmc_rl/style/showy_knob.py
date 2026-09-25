"""Showy-setup knob: a learned P(showy clear soon | afterstate) as a decision-time bias.

A small interpretable model (engineered board features -> logistic regression)
is fit offline on human (Fightcade 14-Hi, rating > 2000) placements: the label
is "this player makes a T2+ clear (``drmc_rl.eval.big_clear``) within the next
k placements". At decision time every legal candidate's settled afterstate is
featurized and the policy logits become

    logits + lambda * (logit(V) - logit(V_ref))

where V is the candidate's showy value: 1 - eps when the placement itself is a
T2+ clear, else the model's probability. Only legal candidates change; at
lambda = 0 the wrapper is not installed at all, so the policy is unchanged.
Everything is pure numpy/Python (no torch), deterministic, and small enough for
Pyodide.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "drmc-showy-knob-v1"
_EMPTY = 0xFF


def _planes(fields: np.ndarray):
    f = np.asarray(fields, dtype=np.uint8).reshape(-1, 16, 8)
    occ = f != _EMPTY
    color = np.where(occ, f & 3, 3)
    virus = occ & ((f & 0xF0) == 0xD0)
    return occ, color, virus


def _hwindows(a: np.ndarray, w: int) -> np.ndarray:
    """Sum over horizontal windows of width w: [N,16,8] -> [N,16,9-w]."""
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,), np.int16), np.cumsum(a, axis=-1, dtype=np.int16)], axis=-1)
    return c[..., w:] - c[..., :-w]


def _vwindows(a: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([np.zeros((a.shape[0], 1, a.shape[2]), np.int16), np.cumsum(a, axis=1, dtype=np.int16)], axis=1)
    return c[:, w:, :] - c[:, :-w, :]


_PER_NAMES = ("h3", "h3s", "h2", "h2s", "h5_4", "h6_4", "h6_5", "v3", "v2", "hv3", "vir_h3", "vir_v3",
              "surf_run3", "surf_run2", "stack")
# board_features' columns, in order (checked against the computed dict on every call).
FEATURE_NAMES: list[str] = ["occupied", "viruses", "height_max", "height_mean", "bumpiness", "holes", "spawn_danger",
                            "tall_cols", *(f"{k}_{a}" for k in _PER_NAMES for a in ("sum", "max", "colors")),
                            "threats", "threats_sq", "threat_rows", "surface_cells", "log_occ"]


def board_features(fields: np.ndarray) -> np.ndarray:
    """Engineered features of settled bottles ``[N,128]`` (NES tile bytes) -> ``[N,F]`` float32."""
    occ, color, virus = _planes(fields)
    n = occ.shape[0]
    empty = ~occ
    below_solid = np.concatenate([occ[:, 1:, :], np.ones((n, 1, 8), bool)], axis=1)
    above_clear = np.cumsum(occ, axis=1) == 0              # nothing at or above this cell in the column
    surface = empty & below_solid & above_clear              # where a dropped tile comes to rest
    supported = empty & below_solid
    heights = 16 - np.where(occ.any(axis=1), occ.argmax(axis=1), 16)
    holes = (empty & ~above_clear).sum(axis=(1, 2))
    feats: dict[str, np.ndarray] = {}
    feats["occupied"] = occ.sum(axis=(1, 2))
    feats["viruses"] = virus.sum(axis=(1, 2))
    feats["height_max"] = heights.max(axis=1)
    feats["height_mean"] = heights.mean(axis=1)
    feats["bumpiness"] = np.abs(np.diff(heights, axis=1)).sum(axis=1)
    feats["holes"] = holes
    feats["spawn_danger"] = occ[:, :3, 3:5].sum(axis=(1, 2))
    feats["tall_cols"] = (heights >= 12).sum(axis=1)
    e_h4 = _hwindows(empty, 4); sup_h4 = _hwindows(supported, 4)
    e_v4 = _vwindows(empty, 4)
    per = {k: [] for k in _PER_NAMES}
    for c in range(3):
        cc = color == c
        n4 = _hwindows(cc, 4)
        h3 = (n4 == 3) & (e_h4 == 1)
        h2 = (n4 == 2) & (e_h4 == 2)
        per["h3"].append(h3.sum(axis=(1, 2)))
        per["h3s"].append((h3 & (sup_h4 >= 1)).sum(axis=(1, 2)))
        per["h2"].append(h2.sum(axis=(1, 2)))
        per["h2s"].append((h2 & (sup_h4 >= 1)).sum(axis=(1, 2)))
        n5, e5 = _hwindows(cc, 5), _hwindows(empty, 5)
        per["h5_4"].append(((n5 == 4) & (e5 == 1)).sum(axis=(1, 2)))
        n6, e6 = _hwindows(cc, 6), _hwindows(empty, 6)
        per["h6_4"].append(((n6 == 4) & (e6 == 2)).sum(axis=(1, 2)))
        per["h6_5"].append(((n6 == 5) & (e6 == 1)).sum(axis=(1, 2)))
        v4 = _vwindows(cc, 4)
        # top-of-window empty, three below same color: completing needs a drop into the column
        top_empty = empty[:, :13, :]
        v3 = (v4 == 3) & (e_v4 == 1) & top_empty
        v2 = (v4 == 2) & (e_v4 == 2) & top_empty & empty[:, 1:14, :]
        per["v3"].append(v3.sum(axis=(1, 2)))
        per["v2"].append(v2.sum(axis=(1, 2)))
        per["hv3"].append(((h3.sum(axis=2) > 0).sum(axis=1) > 0) * (v3.sum(axis=(1, 2)) > 0))
        cv = cc & virus
        per["vir_h3"].append((h3 & (_hwindows(cv, 4) > 0)).sum(axis=(1, 2)))
        per["vir_v3"].append((v3 & (_vwindows(cv, 4) > 0)).sum(axis=(1, 2)))
        # same-color run length directly under each surface cell (vertical threat at the top)
        run = np.zeros((n, 8), np.int16)
        depth = np.where(occ.any(axis=1), occ.argmax(axis=1), 16)
        alive = np.ones((n, 8), bool)
        for k in range(4):
            r = depth + k
            ok = r < 16
            rr = np.clip(r, 0, 15)
            same = np.take_along_axis(cc, rr[:, None, :], axis=1)[:, 0, :] & ok & alive
            run += same
            alive &= same
        per["surf_run3"].append((run >= 3).sum(axis=1))
        per["surf_run2"].append((run == 2).sum(axis=1))
        # stacked threats: rows (distinct) holding a near-complete horizontal of this color
        per["stack"].append((h3.sum(axis=2) > 0).sum(axis=1))
    for k, v in per.items():
        arr = np.stack(v, axis=1).astype(np.float32)
        feats[f"{k}_sum"] = arr.sum(axis=1)
        feats[f"{k}_max"] = arr.max(axis=1)
        feats[f"{k}_colors"] = (arr > 0).sum(axis=1)
    threats = feats["h3s_sum"] + feats["surf_run3_sum"]
    feats["threats"] = threats
    feats["threats_sq"] = threats ** 2
    feats["threat_rows"] = feats["stack_sum"]
    feats["surface_cells"] = surface.sum(axis=(1, 2))
    feats["log_occ"] = np.log1p(feats["occupied"])
    if list(feats) != FEATURE_NAMES:
        raise AssertionError("board_features columns drifted from FEATURE_NAMES")
    return np.stack([np.asarray(feats[k], np.float32) for k in FEATURE_NAMES], axis=1)


class ShowyModel:
    """Standardized logistic regression over ``board_features`` (JSON-serializable)."""

    def __init__(self, spec: dict):
        if spec.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} model")
        self.spec = spec
        self.names = list(spec["features"])
        self.mean = np.asarray(spec["mean"], np.float32)
        self.scale = np.asarray(spec["scale"], np.float32)
        self.coef = np.asarray(spec["coef"], np.float32)
        self.intercept = float(spec["intercept"])
        self.index = None

    @classmethod
    def load(cls, source) -> "ShowyModel":
        if isinstance(source, dict):
            return cls(source)
        return cls(json.loads(Path(source).read_text()))

    @property
    def uses_triggers(self) -> bool:
        return any(k in TRIGGER_NAMES for k in self.names)

    def logit(self, fields: np.ndarray) -> np.ndarray:
        x = board_features(fields)
        if self.uses_triggers:
            x = np.concatenate([x, trigger_features(fields)], axis=1)
        return self.logit_features(x)

    def logit_features(self, x: np.ndarray) -> np.ndarray:
        """``logit`` from precomputed ``[board_features | trigger_features]`` columns (float32)."""
        if self.index is None:
            pos = {k: i for i, k in enumerate(list(FEATURE_NAMES) + list(TRIGGER_NAMES))}
            self.index = np.asarray([pos[k] for k in self.names])
        z = (x[:, self.index] - self.mean) / self.scale
        return z @ self.coef + self.intercept

    def prob(self, fields: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(fields)))


def immediate_scores(fields: np.ndarray, pill, actions) -> np.ndarray:
    """Exact showiness score of each placement (``-1`` for no clear) from one root bottle."""
    return immediate_clears(fields, pill, actions)[0]


def immediate_clears(fields: np.ndarray, pill, actions):
    """Exact (showiness score or -1, matched lines) of each placement from one root bottle."""
    from drmc_rl.eval import big_clear as bc
    root = bytes(np.asarray(fields, np.uint8).reshape(128))
    out = np.full(len(actions), -1.0, np.float32)
    lines = np.zeros(len(actions), np.int32)
    for k, action in enumerate(actions):
        action = int(action)
        if action < 0:
            continue
        o, cell = divmod(action, 128)
        r, c = divmod(cell, 8)
        dr, dc = bc._SECOND[o]
        try:
            placed = bc.place(root, (int(pill[0]), int(pill[1])), action)
        except ValueError:
            continue
        if bc.forms_line(placed, (cell, (r + dr) * 8 + c + dc)):
            f = bc.resolve(placed)[1]
            if f.rounds:
                out[k] = f.score()
                lines[k] = f.lines
    return out, lines


def showy_bias(model: ShowyModel, root_field, pill, actions, mask, lam: float, *, tier_bar: float = 30.0,
               eps: float = 1e-3, decision: "Decision | None" = None) -> np.ndarray:
    """Per-candidate bias ``lam * (logit V - mean logit V over legal)`` for one decision ``[K]``."""
    return _bias(model, root_field, pill, actions, mask, lam, eps=eps, decision=decision,
                 certain=lambda score, lines: score >= tier_bar)


def quad_bias(model: ShowyModel, root_field, pill, actions, mask, lam: float, *, tier_bar: float = 4.0,
              eps: float = 1e-3, decision: "Decision | None" = None) -> np.ndarray:
    """The quad knob: V = P(quad within the next placements | afterstate), certain when this
    placement itself matches ``tier_bar`` (4) or more lines, minus ``model["waste_penalty"]``
    logit units per line beyond four in this placement's own attack (the ROM sends at most 4)."""
    penalty = float(model.spec.get("waste_penalty", 0.0))
    return _bias(model, root_field, pill, actions, mask, lam, eps=eps, decision=decision,
                 certain=lambda score, lines: lines >= tier_bar,
                 adjust=(lambda score, lines: -penalty * np.maximum(lines - 4, 0)) if penalty else None)


class Decision:
    """One decision's shared candidate facts: legal slots, settled afterstates, features
    ``[board_features | trigger_features]`` and each placement's own clear (score or -1, lines).

    Stacked knobs share one instance, so the afterstates and features are computed once.
    The native library (``drmc_rl.style.native``) produces the same values when present.
    """

    def __init__(self, root_field, pill, actions, mask):
        self.actions = np.asarray(actions)
        self.legal = np.flatnonzero(np.asarray(mask, bool))
        self.root = np.asarray(root_field, np.uint8).reshape(128)
        self.colors = (int(pill[0]), int(pill[1]))
        self._x = None
        self._trig = False
        self.after = self.score = self.lines = None

    def features(self, want_trig: bool):
        if self._x is not None and (self._trig or not want_trig):
            return self._x
        want_trig = want_trig or self._trig
        from drmc_rl.style import native
        out = native.decision_features(self.root, self.colors, self.actions[self.legal], want_trig)
        if out is None:
            if self.after is None:
                self.after = reference_afterstates(self.root, self.colors, self.actions[self.legal])
            x = board_features(self.after)
            if want_trig:
                x = np.concatenate([x, trigger_features(self.after)], axis=1)
            if self.score is None:
                self.score, self.lines = immediate_clears(self.root, self.colors, self.actions[self.legal])
        else:
            self.after, x, self.score, self.lines = out
        self._x, self._trig = x, want_trig
        return x


def reference_afterstates(root, colors, actions) -> np.ndarray:
    """Settled afterstate of each placement (numpy/Python reference)."""
    from drmc_rl.game.afterstate import resolve_placement
    try:  # fast path for quiet placements where available (newer afterstate module)
        from drmc_rl.game.afterstate import _quiet_placement, _stable_root
    except ImportError:
        _quiet_placement = _stable_root = None
    stable = _stable_root(root) if _stable_root is not None else None
    after = np.empty((len(actions), 128), np.uint8)
    for j, action in enumerate(actions):
        quiet = _quiet_placement(stable, colors, int(action)) if stable is not None else None
        after[j] = np.frombuffer((quiet or resolve_placement(root, colors, int(action)))[0], np.uint8)
    return after


def _bias(model, root_field, pill, actions, mask, lam, *, eps, certain, adjust=None, decision=None):
    actions = np.asarray(actions)
    bias = np.zeros(actions.shape, np.float32)
    d = decision if decision is not None else Decision(root_field, pill, actions, mask)
    legal = d.legal
    if lam == 0 or len(legal) == 0:
        return bias
    z = model.logit_features(d.features(model.uses_triggers)).astype(np.float64)
    score, lines = d.score, d.lines
    z[certain(score, lines)] = np.log((1 - eps) / eps)
    z = np.clip(z, -12.0, 12.0)
    if adjust is not None:
        z = z + adjust(score, lines)
    bias[legal] = lam * (z - z.mean())
    return bias


class ShowyPolicy:
    """Wrap a candidate-scoring actor (``score(obs, infos) -> (actions, mask, logits)``)."""

    def __init__(self, inner, model: ShowyModel, lam: float, *, tier_bar: float = 30.0):
        self.inner, self.model, self.lam, self.tier_bar = inner, model, float(lam), float(tier_bar)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score(self, obs, infos):
        from drmc_rl.game.afterstate import planes_to_fields
        ca, cm, lg = self.inner.score(obs, infos)
        if self.lam == 0:
            return ca, cm, lg
        fields = planes_to_fields(np.asarray(obs)[:, :8])
        lg = np.array(lg, dtype=np.float32, copy=True)
        for i, info in enumerate(infos):
            pill = np.asarray(info["next_pill_colors"], np.int64)
            b = showy_bias(self.model, fields[i], pill, ca[i], cm[i], self.lam, tier_bar=self.tier_bar)
            lg[i] = np.where(cm[i], lg[i] + b, lg[i])
        return ca, cm, lg

    def act(self, obs, infos):
        ca, cm, lg = self.score(obs, infos)
        slots = np.argmax(lg, axis=1)
        acts = ca[np.arange(len(ca)), slots]
        acts[~cm.any(axis=1)] = -1
        return acts.astype(np.int32)


__all__ = ["Decision", "FEATURE_NAMES", "SCHEMA", "TRIGGER_NAMES", "ShowyModel", "ShowyPolicy", "board_features", "immediate_clears",
           "immediate_scores", "quad_bias", "showy_bias", "trigger_features"]


TRIGGER_NAMES = ("trig_n", "trig_multi", "trig_max_score", "trig_sum_score", "trig_max_rounds", "trig_max_lines",
                 "trig_h", "trig_t1",
                 # attack view of the same drops (ROM: 2+ matched lines send min(lines, 4) pieces)
                 "trig_attack", "trig_quad", "trig_max_garbage", "trig_max_waste")


def trigger_features(fields: np.ndarray) -> np.ndarray:
    """Cascade potential by exact simulation: drop one tile of each color on each column's surface.

    For each of the 8 x 3 (column, color) single-tile drops that completes a line, the
    bottle is resolved exactly (``drmc_rl.eval.big_clear.resolve``). Returns ``[N,12]``
    (columns appended later never change earlier ones, so older models are unaffected).
    """
    from drmc_rl.eval import big_clear as bc
    fields = np.asarray(fields, np.uint8).reshape(-1, 128)
    out = np.zeros((len(fields), len(TRIGGER_NAMES)), np.float32)
    nes = (1, 0, 2)
    for i, f in enumerate(fields):
        occ = (f != _EMPTY).reshape(16, 8)
        depth = np.where(occ.any(axis=0), occ.argmax(axis=0), 16)
        row = out[i]
        base = bytearray(f.tobytes())
        for col in range(8):
            r = int(depth[col]) - 1
            if r < 0:
                continue
            index = r * 8 + col
            for c in range(3):
                board = bytearray(base)
                board[index] = 0x80 | nes[c]
                if not bc.forms_line(board, (index, index)):
                    continue
                feat = bc.resolve(board)[1]
                if not feat.rounds:
                    continue
                s = feat.score()
                row[0] += 1
                row[1] += feat.lines >= 2 or feat.rounds >= 2
                row[2] = max(row[2], s)
                row[3] += s
                row[4] = max(row[4], feat.rounds)
                row[5] = max(row[5], feat.lines)
                row[6] += feat.horizontal_lines > 0
                row[7] += s >= 27
                row[8] += feat.lines >= 2
                row[9] += feat.lines >= 4
                row[10] = max(row[10], min(feat.lines, 4) if feat.lines >= 2 else 0)
                row[11] = max(row[11], feat.lines - 4)
    return out
