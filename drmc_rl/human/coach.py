"""Backend-only, presentation-neutral decision analysis."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def analyze_choice(
    actions: Sequence[int],
    human_logits: Sequence[float],
    *,
    chosen_action: int | None = None,
    competitive_scores: Sequence[float] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Describe a decision without pretending imitation logits are game value.

    Human logits answer "what do comparable humans tend to choose?". Optional
    competitive scores answer "what does the strongest available policy
    prefer?". Keeping the two axes separate prevents a common coaching error:
    calling an unusual move bad merely because it is unusual.
    """

    action_arr = np.asarray(actions, dtype=np.int64)
    logits = np.asarray(human_logits, dtype=np.float64)
    if action_arr.ndim != 1 or logits.shape != action_arr.shape or action_arr.size == 0:
        raise ValueError("actions and human_logits must be non-empty vectors of equal length")
    probs = _softmax(logits)
    human_order = np.argsort(-probs, kind="stable")
    comp = None if competitive_scores is None else np.asarray(competitive_scores, dtype=np.float64)
    if comp is not None and comp.shape != action_arr.shape:
        raise ValueError("competitive_scores must match actions")

    top = []
    for idx in human_order[: max(1, int(limit))]:
        row: dict[str, Any] = {
            "action": int(action_arr[idx]),
            "human_probability": float(probs[idx]),
            "human_rank": int(np.flatnonzero(human_order == idx)[0]) + 1,
        }
        if comp is not None:
            row["competitive_score"] = float(comp[idx])
            row["competitive_rank"] = int((comp > comp[idx]).sum()) + 1
        top.append(row)

    result: dict[str, Any] = {
        "interpretation": {
            "human_probability": "frequency under the requested human-strength model",
            "competitive_score": "optional policy preference; only this axis is quality-oriented",
        },
        "alternatives": top,
    }
    if chosen_action is not None:
        matches = np.flatnonzero(action_arr == int(chosen_action))
        if matches.size == 0:
            result["chosen"] = {"action": int(chosen_action), "feasible": False}
        else:
            idx = int(matches[0])
            chosen: dict[str, Any] = {
                "action": int(chosen_action),
                "feasible": True,
                "human_probability": float(probs[idx]),
                "human_rank": int((probs > probs[idx]).sum()) + 1,
                "surprisal_nats": float(-np.log(max(probs[idx], 1e-12))),
            }
            if comp is not None:
                chosen["competitive_score"] = float(comp[idx])
                chosen["competitive_rank"] = int((comp > comp[idx]).sum()) + 1
                chosen["competitive_gap"] = float(comp.max() - comp[idx])
            result["chosen"] = chosen
    return result
