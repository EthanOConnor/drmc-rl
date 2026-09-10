"""Spatial human construction proposals on frozen competitive bottle features.

These are candidate preferences, never outcome values or quality admission.
Only the actual own bottle and currently visible two pills enter this encoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from drmc_rl.human.expressive_sequences import GOALS

SCHEMA = "drmc-spatial-expressive-proposer-v1"
FIXED_ROOT = "fixed-root-v1"
RECURRENT_PUBLIC = "recurrent-public-v1"
COLOR_MAP = np.asarray([1, 0, 2], np.int64)  # native Y/R/B -> policy R/Y/B


def spatial_clear_target(result, goal):
    """A distribution over the actual colored cells in the chosen clear event."""
    selected = set()
    for step in result.steps:
        cells = {(c.row, c.col, c.color) for c in step.cleared}
        axes = []
        for dr, dc in ((0, 1), (1, 0)):
            marked = set()
            for r, c, color in cells:
                run = {(r+i*dr, c+i*dc, color) for i in range(4)}
                if run <= cells:
                    marked |= run
            axes.append(marked)
        if goal == 0 and axes[0]:
            selected = axes[0]
        elif goal == 1 and axes[0] & axes[1]:
            selected = axes[0] | axes[1]
        elif goal == 2 and len(cells) >= 8:
            selected = cells
        elif goal == 3 and len(result.steps) >= 2:
            selected |= cells
        if selected and goal != 3:
            break
    if not selected:
        raise ValueError("the verified payoff does not contain the selected spatial goal")
    target = np.zeros((3, 16, 8), np.float32)
    for r, c, color in selected:
        target[COLOR_MAP[color], r, c] = 1
    return target / target.sum()


class FrozenConstructionEncoder(nn.Module):
    """Reuse the legacy competitive own-bottle path, with no invented pair state.

    The separate proposer sees complete semantic bonds. Its shared feature
    input reproduces the frozen core's same-color bond encoding exactly.
    Public-history cores require a real full-context adapter and are rejected.
    """
    def __init__(self, core, *, zero_auxiliary=False):
        super().__init__()
        if (core.aux_dim and not zero_auxiliary) or core.public_context_schema is not None:
            raise ValueError("own-only replay cannot supply a full public-context core")
        self.core = core.eval().requires_grad_(False)
        self.feature_dim = 2 * core.d_model

    def forward(self, board, pill, preview):
        own = board.clone()
        own[pill[:, 0] == pill[:, 1], 6:8] = 0
        pieces = [self.core.pill_embedding(pill),self.core.preview_embedding(preview)]
        if self.core.aux_dim:
            # Explicit zero_v1_vs public export contract, not reconstructed
            # history or a guessed hidden pending-attack observation.
            pieces.append(torch.zeros(len(board),self.core.aux_dim,device=board.device,dtype=board.dtype))
        cond = self.core.condition(torch.cat(pieces, -1))
        if not self.core.conditioned_trunk:
            cond = torch.zeros_like(cond)
        features = self.core.bottle_projection(self.core.bottle(own, cond))
        return torch.cat((features.mean((2, 3)), features.amax((2, 3))), -1)


class SpatialProposer(nn.Module):
    def __init__(self, feature_dim, width=128, *, persistent=True, plan_update_schema=FIXED_ROOT):
        super().__init__()
        if plan_update_schema not in (FIXED_ROOT, RECURRENT_PUBLIC):
            raise ValueError("unknown spatial plan update schema")
        self.feature_dim, self.width = int(feature_dim), int(width)
        self.persistent = bool(persistent)
        self.plan_update_schema = plan_update_schema
        self.geometry = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(), nn.Flatten(),
            nn.Linear(16*16*8, width), nn.SiLU())
        self.pill = nn.Embedding(9, 16)
        self.state = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(2*width+32, width), nn.SiLU())
        self.intent = nn.Linear(width, len(GOALS))
        self.goal = nn.Embedding(len(GOALS), 16)
        self.target = nn.Linear(width+16, 3*16*8)
        self.horizon = nn.Linear(width+16, 6)
        self.plan_embedding = nn.Sequential(nn.Linear(3*16*8, 32), nn.SiLU())
        self.elapsed = nn.Embedding(7, 8)
        self.action = nn.Sequential(nn.Linear(2*width+32+6+16+8, width), nn.SiLU(),
                                    nn.Linear(width, 512))
        if self.replanning:
            # The control has exactly the same capacity, with history reset.
            self.recurrence = nn.GRUCell(width, width)

    @property
    def replanning(self):
        return self.plan_update_schema == RECURRENT_PUBLIC

    def encode(self, board, features, pill, preview):
        return self.fuse(torch.cat((self.geometry(board), self.state(features),
            self.pill(pill[:, 0]*3+pill[:, 1]), self.pill(preview[:, 0]*3+preview[:, 1])), -1))

    def plan(self, memory, goal):
        condition = torch.cat((memory, self.goal(goal)), -1)
        return self.target(condition), self.horizon(condition)

    def update_memory(self, current, previous):
        if not self.replanning:
            raise ValueError("fixed-root proposals do not update memory")
        return self.recurrence(current, previous if self.persistent else torch.zeros_like(previous))

    def sequence_memory(self, current, owner, elapsed, windows):
        """Causal, differentiable updates on actual decision prefixes only."""
        if not self.persistent:
            return self.update_memory(current, torch.zeros_like(current))
        hidden = current.new_zeros((windows, self.width))
        memories = torch.zeros_like(current)
        for step in range(int(elapsed.max())+1):
            rows = torch.nonzero(elapsed == step, as_tuple=True)[0]
            updated = self.update_memory(current[rows], hidden[owner[rows]])
            # Out-of-place updates preserve gradients through previous steps.
            hidden = hidden.index_copy(0, owner[rows], updated)
            memories = memories.index_copy(0, rows, updated)
        return memories

    def plan_at(self, memory, goal, elapsed):
        spatial, horizon = self.plan(memory, goal)
        if self.replanning:
            # Both new arms start with a 2–6 placement proposal; later predictions
            # describe the remaining duration, including an imminent payoff.
            horizon = horizon.masked_fill((elapsed == 0)[:, None] &
                (torch.arange(6, device=horizon.device) == 0)[None, :], -torch.inf)
        elif self.persistent:
            horizon = horizon.clone()
            horizon[:, 0] = -torch.inf
        return spatial, horizon

    def actions(self, memory, current, goal, elapsed, spatial, horizon):
        return self.action(torch.cat((memory, current, self.goal(goal),
            self.plan_embedding(spatial), horizon, self.elapsed(elapsed)), -1))

    def forward(self, memory, current, goal, elapsed):
        spatial, horizon = self.plan_at(memory, goal, elapsed)
        # The action decoder always sees predicted plans, including in fitting.
        # Outcome-cell and duration labels never become action inputs.
        logits = self.actions(memory, current, goal, elapsed,
                              spatial.softmax(-1).detach(), horizon.softmax(-1).detach())
        return logits, spatial, horizon


@dataclass
class SpatialProposal:
    memory: torch.Tensor
    goal: int
    spatial: torch.Tensor
    horizon_distribution: torch.Tensor
    anchor: int
    remaining: int
    elapsed: int
    last_frame: int
    last_completion_frame: int | None = None
    reason: str | None = None
    committed_memory: torch.Tensor | None = None

    @classmethod
    def start(cls, model, inputs, *, frame, goal=None):
        if not model.persistent and not model.replanning:
            raise ValueError("a stateless control cannot start a persistent proposal")
        with torch.inference_mode():
            current = model.encode(*inputs).detach().clone()
            committed = torch.zeros_like(current) if model.replanning else None
            memory = model.update_memory(current, committed) if model.replanning else current
            if len(memory) != 1:
                raise ValueError("one proposal belongs to one player")
            goal = int(model.intent(memory).argmax(-1)) if goal is None else int(goal)
            if not 0 <= goal < len(GOALS):
                raise ValueError("unknown construction goal")
            spatial, horizon = model.plan_at(memory, torch.tensor([goal], device=memory.device),
                                             torch.zeros(1, dtype=torch.long, device=memory.device))
            # A new persistent construction lasts 2–6 placements. Class one is
            # used by the stateless control for already-imminent clear events.
            horizon = horizon.clone()
            horizon[:, 0] = -torch.inf
            spatial, horizon = spatial.softmax(-1), horizon.softmax(-1)
        return cls(memory, goal, spatial, horizon, int(spatial.argmax(-1)),
                   1+int(horizon.argmax(-1)), 0, int(frame), committed_memory=committed)

    def rank(self, model, inputs, feasible_actions):
        if self.reason is not None:
            return []
        actions = torch.as_tensor(feasible_actions, dtype=torch.long, device=self.memory.device)
        if (actions.ndim != 1 or len(actions.unique()) != len(actions)
                or bool(((actions < 0) | (actions >= 512)).any())):
            raise ValueError("invalid complete feasible action inventory")
        with torch.inference_mode():
            current = model.encode(*inputs)
            if len(current) != 1 or model.replanning != (self.committed_memory is not None):
                raise ValueError("proposal and model update contracts differ")
            goal = torch.tensor([self.goal], device=self.memory.device)
            elapsed = torch.tensor([self.elapsed], device=self.memory.device)
            if model.replanning:
                # Repeated planning in one placement revises the same history
                # slot; only an observed completion commits it for the next one.
                self.memory = model.update_memory(current, self.committed_memory)
                spatial, horizon = model.plan_at(self.memory, goal, elapsed)
                self.spatial, self.horizon_distribution = spatial.softmax(-1), horizon.softmax(-1)
                self.anchor = int(self.spatial.argmax(-1))
            logits = model.actions(self.memory, current,
                goal, elapsed,
                self.spatial, self.horizon_distribution)[0]
        return actions[torch.argsort(logits[actions], descending=True, stable=True)].tolist()

    def observe(self, *, frame, completed_placement=False, cleared_cells=(), observed_goals=(),
                incoming_garbage=False, own_state_mismatch=False, terminal=False):
        """Cleared cells are actual (row, column, canonical R/Y/B color) tuples."""
        if self.reason is not None:
            return
        if int(frame) < self.last_frame:
            raise ValueError("public observations must be chronological")
        self.last_frame = int(frame)
        if terminal:
            self.reason = "terminal"
        elif incoming_garbage or own_state_mismatch:
            self.reason = "board_changed"
        elif completed_placement and self.last_completion_frame != int(frame):
            self.last_completion_frame = int(frame)
            if self.committed_memory is not None:
                self.committed_memory = self.memory.detach().clone()
            self.remaining -= 1
            self.elapsed += 1
            color, cell = divmod(self.anchor, 128)
            row, col = divmod(cell, 8)
            if self.goal in observed_goals and (row, col, color) in cleared_cells:
                self.reason = "spatial_goal_observed"
            elif self.remaining <= 0:
                self.reason = "placement_budget"
