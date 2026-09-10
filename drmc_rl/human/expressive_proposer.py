"""Auxiliary persistent construction proposals, never competitive value scores.

Root memory is causal and fixed throughout a 2–6-placement construction. Each
step reads the actual current board/pill/preview. No future pill queue enters
the actor; callers retain the common quality model and exact feasible set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from drmc_rl.human.expressive_sequences import GOALS

MODEL_SCHEMA = "drmc-persistent-expressive-proposer-v1"


def public_planes(board):
    board = np.asarray(board, dtype=np.uint8).reshape(-1, 16, 8)
    color, kind = board & 0x0F, board & 0xF0
    planes = [((color == c) & (kind >= 0x40) & (kind <= 0x80)) for c in range(3)]
    planes += [((color == c) & (kind == 0xD0)) for c in range(3)]
    planes += [(kind == 0x60) | (kind == 0x70), (kind == 0x40) | (kind == 0x50)]
    return np.stack(planes, axis=1).astype(np.float32)


class ExpressiveProposer(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = int(width)
        self.encoder = nn.Sequential(nn.Conv2d(8, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(), nn.Flatten(),
            nn.Linear(32*16*8, width), nn.SiLU())
        self.pill = nn.Embedding(9, 16)
        self.state = nn.Sequential(nn.Linear(width+32, width), nn.SiLU())
        self.intent = nn.Linear(width, len(GOALS)*5)
        self.goal = nn.Embedding(len(GOALS), 16)
        self.remaining = nn.Embedding(7, 8)
        self.action = nn.Sequential(nn.Linear(2*width+24, width), nn.SiLU(), nn.Linear(width, 512))

    def encode(self, board, pill, preview):
        return self.state(torch.cat((self.encoder(board), self.pill(pill[:, 0]*3+pill[:, 1]),
                                     self.pill(preview[:, 0]*3+preview[:, 1])), dim=-1))

    def forward(self, root, current, goal, remaining):
        return self.action(torch.cat((root, current, self.goal(goal), self.remaining(remaining)), dim=-1))


@dataclass
class PersistentProposal:
    """One causal root memory; a surprise ends it before any stale ranking."""
    root: torch.Tensor
    goal: int
    remaining: int
    started_frame: int
    last_frame: int
    reason: str | None = None
    last_completion_frame: int | None = None

    @classmethod
    def start(cls, model, board, pill, preview, *, frame, goal=None, horizon=None):
        with torch.inference_mode():
            root = model.encode(board, pill, preview).detach().clone()
            if root.shape[0] != 1:
                raise ValueError("one proposal belongs to one player")
            choice = int(model.intent(root).argmax(-1).item())
        goal = choice//5 if goal is None else int(goal)
        horizon = 2+choice%5 if horizon is None else int(horizon)
        if not 0 <= goal < len(GOALS) or not 2 <= horizon <= 6:
            raise ValueError("proposal requires a named geometry goal and 2–6 placements")
        return cls(root, goal, horizon, int(frame), int(frame))

    def observe(self, *, frame, completed_placement=False, observed_goals=(),
                incoming_garbage=False, terminal=False, own_state_mismatch=False):
        if self.reason is not None:
            return
        if int(frame) < self.last_frame:
            raise ValueError("public observations must be chronological")
        self.last_frame = int(frame)
        if terminal:
            self.reason = "terminal"
        elif incoming_garbage or own_state_mismatch:
            self.reason = "board_changed"
        elif completed_placement:
            if self.last_completion_frame == int(frame):
                return
            self.last_completion_frame = int(frame)
            self.remaining -= 1
            if self.goal in observed_goals:
                self.reason = "goal_observed"
            elif self.remaining == 0:
                self.reason = "placement_budget"

    def rank(self, model, board, pill, preview, feasible_actions):
        """Return proposal order only. This does not admit a quality sacrifice."""
        if self.reason is not None:
            return []
        actions = torch.as_tensor(feasible_actions, dtype=torch.long, device=self.root.device)
        if actions.ndim != 1 or bool(((actions < 0) | (actions >= 512)).any()):
            raise ValueError("invalid complete feasible action inventory")
        if len(actions.unique()) != len(actions):
            raise ValueError("duplicate feasible actions")
        with torch.inference_mode():
            current = model.encode(board, pill, preview)
            logits = model(self.root, current, torch.tensor([self.goal], device=self.root.device),
                           torch.tensor([self.remaining], device=self.root.device))[0]
        return actions[torch.argsort(logits[actions], descending=True, stable=True)].tolist()
