"""Requested-goal imitation with an immutable public construction objective.

Hindsight payoff cells are requests during supervised fitting, never assertions
about a live game's future. Live goals come from a frozen root-only proposer.
This module provides no competitive-quality permission.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from drmc_rl.human.spatial_proposer import RECURRENT_PUBLIC, SpatialProposal, SpatialProposer

SCHEMA = "drmc-target-construction-v1"


class TargetConstruction(SpatialProposer):
    def __init__(self, feature_dim, width=128):
        super().__init__(feature_dim, width, persistent=True, plan_update_schema=RECURRENT_PUBLIC)

    def requested_actions(self, memory, current, goal, elapsed, anchor, remaining):
        if (anchor.shape != goal.shape or remaining.shape != goal.shape
                or bool(((anchor < 0) | (anchor >= 384)).any())
                or bool(((remaining < 1) | (remaining > 6)).any())):
            raise ValueError("invalid explicit construction request")
        spatial = F.one_hot(anchor, 384).to(current.dtype)
        horizon = F.one_hot(remaining-1, 6).to(current.dtype)
        return self.actions(memory, current, goal, elapsed, spatial, horizon)


@dataclass
class TargetProposal(SpatialProposal):
    """Keep the original target while replanning the route after each placement."""

    @classmethod
    def start_requested(cls, model, inputs, *, frame, goal, anchor, budget):
        if not 0 <= int(goal) < 4 or not 0 <= int(anchor) < 384 or not 2 <= int(budget) <= 6:
            raise ValueError("a new request needs a goal, colored cell and 2–6 placement budget")
        with torch.inference_mode():
            current = model.encode(*inputs)
            if len(current) != 1:
                raise ValueError("one proposal belongs to one player")
            committed = torch.zeros_like(current)
            memory = model.update_memory(current, committed)
            spatial = F.one_hot(torch.tensor([anchor], device=current.device), 384).to(current.dtype)
            horizon = F.one_hot(torch.tensor([budget-1], device=current.device), 6).to(current.dtype)
        return cls(memory, int(goal), spatial, horizon, int(anchor), int(budget), 0,
                   int(frame), committed_memory=committed)

    def rank(self, model, inputs, feasible_actions):
        if self.reason is not None:
            return []
        actions = torch.as_tensor(feasible_actions, dtype=torch.long, device=self.memory.device)
        if (actions.ndim != 1 or not len(actions) or len(actions.unique()) != len(actions)
                or bool(((actions < 0) | (actions >= 512)).any())):
            raise ValueError("invalid complete feasible action inventory")
        with torch.inference_mode():
            current = model.encode(*inputs)
            self.memory = model.update_memory(current, self.committed_memory)
            integer = lambda x: torch.tensor([x], device=current.device, dtype=torch.long)
            logits = model.requested_actions(self.memory, current, integer(self.goal),
                integer(self.elapsed), integer(self.anchor), integer(self.remaining))[0]
        return actions[torch.argsort(logits[actions], descending=True, stable=True)].tolist()


def propose_target(proposer, inputs):
    """Only the actual root board and its two visible pills choose the request."""
    with torch.inference_mode():
        current = proposer.encode(*inputs)
        memory = proposer.update_memory(current, torch.zeros_like(current))
        goal = proposer.intent(memory).argmax(-1)
        target, _ = proposer.plan_at(memory, goal, torch.zeros_like(goal))
    return int(goal.item()), int(target.argmax(-1).item())
