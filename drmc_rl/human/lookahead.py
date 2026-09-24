"""One-ply value lookahead at decision time (inference only, public information).

The core scores the current pill's complete motor-feasible frontier as usual.
With a ``lookahead`` variant block the arena keeps the top ``k`` placements
(plus any within ``margin`` logits of the best, at most ``max_kept``), settles
each one exactly (``drmc_rl.game.afterstate.resolve_placement``), plans the next
pill on every settled bottle with the same pace and delay, and scores those
frontiers in one batched pass. Each kept placement is then valued by

* ``value``: the value head on the state after its best ``followups`` next-pill
  placements (one more batched value pass), maximized over them;
* ``root_value``: the value head of the settled bottle's own follow-up pass;
* ``logit``: the follow-up policy's best logit.

Immediate terminal facts override the network: a winning placement is taken at
once, a placement whose settled bottle blocks the spawn is a loss, and a settled
bottle with no reachable next placement is a loss.

Only public information is used. The next pill is the root view's preview; the
after-next preview is unknown and the next pill stands in for it (the
``early_preview=repeat`` convention). At a pre-spawn decision the root view
already repeats the current pill as its preview, so the next pill is unknown
too and the same repeat rule applies. Incoming garbage, the opponent's later
moves and the next spawn's controller micro-state (the current spawn's is
reused) are not modeled. The opponent side is frozen at the root view.

``k = 1`` keeps only the argmax and performs no extra computation, reproducing
the plain core exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from drmc_rl.game.afterstate import resolve_placement
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.early_decision import early_public_view

LOOKAHEAD_MODES = ("value", "root_value", "logit")
LOOKAHEAD_WHEN = ("always", "early_only")
WIN, SPAWN_BLOCKED, NO_REACH = 3.0, -3.0, -1.0
FOLLOWUP_WIN, FOLLOWUP_BLOCKED = 1.0, -1.0


def lookahead_params(params):
    """Validated lookahead block of a variant, or None."""
    block = params.get("lookahead")
    if block is None:
        return None
    k = block.get("k")
    if type(k) is not int or not 1 <= k <= 32:
        raise ValueError("lookahead.k must be an integer in [1, 32]")
    followups = block.get("followups", 1)
    if followups not in (1, 2):
        raise ValueError("lookahead.followups must be 1 or 2")
    mode = block.get("mode", "value")
    if mode not in LOOKAHEAD_MODES:
        raise ValueError(f"lookahead.mode must be one of {LOOKAHEAD_MODES}")
    margin = float(block.get("margin", 0.0))
    charge = block.get("charge_frames", 0)
    if margin < 0 or type(charge) is not int or not 0 <= charge <= 60:
        raise ValueError("lookahead.margin must be >= 0 and charge_frames an integer frame count")
    when = block.get("when", "always")
    if when not in LOOKAHEAD_WHEN:
        raise ValueError(f"lookahead.when must be one of {LOOKAHEAD_WHEN}")
    if params.get("preview_input", "visible") != "visible" or (
            params.get("decision_point", "spawn") != "spawn"
            and params.get("early_preview", "marginal") != "repeat"):
        raise ValueError("lookahead needs one visible or repeat root view per decision")
    if params.get("anticipation") or params.get("own_board_only") or "compute_input_frames" in params:
        raise ValueError("lookahead is defined only on the plain fresh-decision contract")
    return dict(k=k, followups=followups, mode=mode, margin=margin, charge_frames=charge, when=when,
                max_kept=int(block.get("max_kept", 2 * k)), prior=float(block.get("prior", 0.0)))


def charged(params, active):
    """Variant parameters with the lookahead's compute charged as decision delay."""
    block = lookahead_params(params)
    if block is None or not active:
        return params
    return {**params, "delay": int(params["delay"]) + block["charge_frames"]}


def score_with_value(actor, observations, infos):
    """Complete 512-wide scores and the scalar value head for each row."""
    if hasattr(actor, "score_values"):
        return actor.score_values(observations, infos)
    actions, masks, logits, values = actor.score_and_value(observations, infos)
    scores = np.full((len(infos), 512), -np.inf, np.float32)
    for row, info in enumerate(infos):
        legal = np.flatnonzero(np.asarray(info["placements/feasible_mask"]).reshape(512))
        selected = actions[row, masks[row]]
        if set(selected) != set(legal) or len(selected) != len(legal):
            raise RuntimeError("value scoring changed legal candidate coverage")
        scores[row, selected] = logits[row, masks[row]]
    return scores, np.asarray(values, np.float32)


@dataclass
class Root:
    """One decision whose root frontier is already scored."""
    actor: Any
    state: dict
    candidate: tuple
    scores: np.ndarray
    pace: Any
    delay: int
    public: Any               # the root public view the network saw
    compute_input: int
    delay_input: int
    params: dict              # validated lookahead block
    diagnostics: dict = field(default_factory=dict)


def _settle(board, pill, action):
    return resolve_placement(np.frombuffer(bytes(board), np.uint8), pill, action)


def _kept(scores, block):
    legal = np.flatnonzero(np.isfinite(scores))
    # Stable descending order: ties keep ascending action order, like argmax.
    order = legal[np.argsort(-scores[legal], kind="stable")]
    if block["k"] == 1:
        return order[:1]
    best = scores[order[0]]
    wide = [a for a in order if scores[a] >= best - block["margin"]]
    count = min(max(block["k"], len(wide)), block["max_kept"])
    return order[:count]


def _follow_state(root, board, preview=None, **_):
    """The next pill's spawn state on a settled own bottle (public, repeat preview)."""
    viewer = root.public.viewer_side
    nxt = tuple(int(c) for c in root.public.sides[viewer].preview)
    preview = nxt if preview is None else tuple(preview)
    view = early_public_view(root.public, board=board, pill=nxt, preview=preview, falling=root.state["falling"])
    state = {**root.state, "board_planes": board_bytes_to_semantic_planes(board),
             "pill": list(nxt), "preview": list(preview),
             "pill_counter_total": int(root.state["pill_counter_total"]) + 1,
             "public_pair_state": view}
    return state, view, nxt


def select_lookahead(roots: list[Root], plan: Callable, inputs: Callable):
    """Choose one placement per root; returns ``(action, diagnostics)`` pairs.

    ``plan(requests)`` returns one candidate tuple (or None) per
    ``(state, delay, pace)``. ``inputs(actor, candidate, state, pace, delay,
    compute_input, public, delay_input)`` returns the network ``(obs, infos)``.
    """
    results = [None] * len(roots)
    follow = []           # (root index, action, state, view, next pill)
    q = [dict() for _ in roots]
    for r, root in enumerate(roots):
        block = root.params
        kept = _kept(root.scores, block)
        root.diagnostics.update(kept=len(kept), argmax=int(kept[0]))
        if len(kept) == 1:
            results[r] = int(kept[0])
            continue
        board = bytes(root.public.sides[root.public.viewer_side].board)
        pill = tuple(int(c) for c in root.state["pill"])
        for action in map(int, kept):
            after, facts = _settle(board, pill, action)
            if facts[10]:          # win
                q[r][action] = WIN
            elif facts[8]:         # the settled bottle blocks the next spawn
                q[r][action] = SPAWN_BLOCKED
            else:
                state, view, nxt = _follow_state(root, after)
                follow.append((r, action, state, view, nxt))
    candidates = plan([(state, roots[r].delay, roots[r].pace) for r, _, state, _, _ in follow])
    rows, observations, infos = [], [], []
    for (r, action, state, view, _), candidate in zip(follow, candidates):
        if candidate is None:
            q[r][action] = NO_REACH
            continue
        root = roots[r]
        obs, info = inputs(root.actor, candidate, state, root.pace, root.delay,
                           root.compute_input, view, root.delay_input)
        rows.append((r, action, state, view, candidate))
        observations.append(obs)
        infos.extend(info)
    first = _batched(roots, rows, observations, infos)
    second = []
    for (r, action, state, view, candidate), (scores, value) in zip(rows, first):
        root = roots[r]
        mode = root.params["mode"]
        if mode == "root_value":
            q[r][action] = float(value)
            continue
        if mode == "logit":
            q[r][action] = float(scores[np.isfinite(scores)].max())
            continue
        legal = np.flatnonzero(np.isfinite(scores))
        order = legal[np.argsort(-scores[legal], kind="stable")][:root.params["followups"]]
        board = bytes(view.sides[view.viewer_side].board)
        pill = tuple(int(c) for c in state["pill"])
        best = -np.inf
        for follow_action in map(int, order):
            after, facts = _settle(board, pill, follow_action)
            if facts[10]:
                best = max(best, FOLLOWUP_WIN)
            elif facts[8]:
                best = max(best, FOLLOWUP_BLOCKED)
            else:
                second.append((r, action, follow_action, after, state, candidate))
        q[r][action] = best
    observations, infos, value_rows = [], [], []
    for r, action, follow_action, after, state, candidate in second:
        root = roots[r]
        leaf, view, _ = _follow_state(root, after)
        leaf["pill_counter_total"] = int(state["pill_counter_total"]) + 1
        # The G5 critic reads only bottles and context, never the frontier; one
        # placeholder candidate satisfies the input contract.
        costs = np.full(512, 0xFFFF, np.uint16)
        costs[follow_action] = 0
        obs, info = inputs(root.actor, (costs,), leaf, root.pace, root.delay,
                           root.compute_input, view, root.delay_input)
        observations.append(obs)
        infos.extend(info)
        value_rows.append((r, action))
    for (r, action), (_, value) in zip(value_rows, _batched(roots, value_rows, observations, infos)):
        q[r][action] = max(q[r][action], float(value))
    for r, root in enumerate(roots):
        if results[r] is not None:
            continue
        kept = _kept(root.scores, root.params)
        finite = root.scores[np.isfinite(root.scores)]
        log_z = float(finite.max() + np.log(np.exp(finite - finite.max()).sum()))
        total = {a: q[r][int(a)] + root.params["prior"] * (float(root.scores[a]) - log_z) for a in kept}
        # Ties keep the root policy's order.
        choice = int(max(kept, key=lambda a: total[a]))
        if not np.isfinite(root.scores[choice]):
            raise RuntimeError("lookahead chose a placement outside the feasible frontier")
        results[r] = choice
        root.diagnostics.update(
            q=[round(float(q[r][int(a)]), 4) for a in kept], rank=int(list(kept).index(choice)),
            changed=choice != int(kept[0]))
    return [(action, root.diagnostics) for action, root in zip(results, roots)]


def _batched(roots, rows, observations, infos):
    """One scoring pass per actor over every row; returns (scores, value) per row."""
    if not rows:
        return []
    obs = np.concatenate(observations)
    out_scores = np.empty((len(infos), 512), np.float32)
    out_values = np.empty(len(infos), np.float32)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(id(roots[row[0]].actor), (roots[row[0]].actor, []))[1].append(i)
    for actor, indices in groups.values():
        scores, values = score_with_value(actor, obs[indices], [infos[i] for i in indices])
        out_scores[indices], out_values[indices] = scores, values
    return list(zip(out_scores, out_values))


__all__ = ["LOOKAHEAD_MODES", "Root", "charged", "lookahead_params", "score_with_value",
           "select_lookahead"]
