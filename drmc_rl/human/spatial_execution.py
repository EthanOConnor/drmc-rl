"""Observe construction proposals during autonomous controller games.

No proposal controls an action. Payoff geometry is accepted only after an
undisturbed, actually locked placement reproduces the next observed bottle.
This is a conditional realization diagnostic, not competitive quality admission.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.game.pair_state import DecisionBoundary, PairEventKind
from drmc_rl.human.backend import POSE_TO_ACTION
from drmc_rl.human.expressive_sequences import locked_field, observed_goals
from drmc_rl.human.spatial_proposer import COLOR_MAP, SpatialProposal, spatial_clear_target


def payoff(result):
    goals = np.flatnonzero(observed_goals(result)).tolist()
    # Use the same goal-specific geometry as fitting: a separate vertical clear
    # cannot satisfy a horizontal target just because both happened this turn.
    cells = {goal: {(int(r), int(c), int(color)) for color, r, c in
                    np.argwhere(spatial_clear_target(result, goal) > 0)} for goal in goals}
    return goals, cells


def anchor_hit(anchor, goal, goals, cells):
    color, cell = divmod(anchor, 128)
    row, col = divmod(cell, 8)
    return goal in goals and (row, col, color) in cells


class ConstructionObserver:
    def __init__(self, models, encoder, jobs, *, feature_device="cpu"):
        self.models, self.encoder, self.feature_device = models, encoder, feature_device
        self.sides = {2*i+assignment for i, (_, assignment, _) in enumerate(jobs)}
        self.states = {side: dict(frame=-1, pending=None, plans={}, counters=Counter()) for side in self.sides}
        self.decisions, self.plans, self.transitions = [], [], []

    def start_proposal(self, model, inputs, frame):
        return SpatialProposal.start(model, inputs, frame=frame)

    def model_inputs(self, records):
        boards = torch.tensor(np.stack([board_bytes_to_semantic_planes(r["board"]) for r in records]))
        pills = torch.tensor([r["pill"] for r in records], dtype=torch.long)
        previews = torch.tensor([r["preview"] for r in records], dtype=torch.long)
        with torch.inference_mode():
            features = self.encoder(boards.to(self.feature_device), pills.to(self.feature_device),
                                    previews.to(self.feature_device)).cpu()
        return [(boards[i:i+1], features[i:i+1], pills[i:i+1], previews[i:i+1])
                for i in range(len(records))]

    def end(self, side, name, frame, reason):
        state = self.states[side]
        record = state["plans"].pop(name, None)
        if record is not None:
            plan = record.pop("plan")
            record.update(side=side, arm=name, end_frame=frame, reason=reason,
                          completed_placements=plan.elapsed, final_anchor=plan.anchor)
            self.plans.append(record)

    def abort(self, side, frame, reason):
        for name in list(self.states[side]["plans"]):
            self.end(side, name, frame, reason)

    def finish_placement(self, side, view):
        state = self.states[side]
        pending = state["pending"]
        if pending is None:
            return
        state["pending"] = None
        if not pending["locked"]:
            self.abort(side, view.frame_id, "unobserved_lock")
            state["counters"]["unobserved_locks"] += 1
            return
        result = pending["result"]
        expected_tiles = sum(len(step.cleared) for step in result.steps)
        expected_viruses = sum(sum(c.is_virus for c in step.cleared) for step in result.steps)
        board_matches = result.settled_field == view.own.board
        verified = (not pending["interrupted"] and board_matches
                    and pending["tiles"] == expected_tiles and pending["viruses"] == expected_viruses)
        self.transitions.append(dict(side=side, start_frame=pending["frame"], end_frame=view.frame_id,
            action=pending["action"], verified=verified, interrupted=pending["interrupted"],
            terminal=view.decision_boundary == DecisionBoundary.TERMINAL,
            board_matches=board_matches, observed_tiles=pending["tiles"], expected_tiles=expected_tiles,
            observed_viruses=pending["viruses"], expected_viruses=expected_viruses))
        if not verified:
            reason = ("interrupted_transitions" if pending["interrupted"] else
                      "terminal_unresolved" if view.decision_boundary == DecisionBoundary.TERMINAL
                      else "unverified_transitions")
            state["counters"][reason] += 1
            self.abort(side, view.frame_id, reason)
            return
        state["counters"]["verified_completions"] += 1
        goals, goal_cells = payoff(result)
        for name, record in list(state["plans"].items()):
            plan = record["plan"]
            cells = goal_cells.get(plan.goal, set())
            record["root_goal_observed"] |= anchor_hit(record["root_anchor"], plan.goal, goals, cells)
            plan.observe(frame=view.frame_id, completed_placement=True, observed_goals=goals, cleared_cells=cells)
            if plan.reason is not None:
                self.end(side, name, view.frame_id, plan.reason)

    def observe(self, views):
        for side, view in views.items():
            state = self.states[side]
            previous = state["frame"]
            if view.frame_id < previous:
                raise ValueError("autonomous observations must be chronological")
            if view.frame_id == previous:
                continue
            events = view.recent_events
            if len(events) == 32 and previous >= 0 and events[0].frame_id > previous:
                state["counters"]["history_gaps"] += 1
                self.abort(side, view.frame_id, "history_gap")
                if state["pending"] is not None:
                    state["pending"]["interrupted"] = True
            for event in events:
                if event.frame_id <= previous or event.side != view.viewer_side:
                    continue
                pending = state["pending"]
                if event.kind == PairEventKind.VOLLEY:
                    self.abort(side, event.frame_id, "incoming_garbage")
                    state["counters"]["incoming_volleys"] += 1
                    if pending is not None:
                        pending["interrupted"] = True
                elif event.kind == PairEventKind.LOCK:
                    if pending is None:
                        state["counters"]["unplanned_locks"] += 1
                        self.abort(side, event.frame_id, "unplanned_lock")
                    else:
                        p = event.public_payload
                        pose = p["rotation"]*128 + p["row_top"]*8 + p["column"]
                        if not 0 <= pose < 512 or int(POSE_TO_ACTION[pose]) != pending["action"]:
                            raise ValueError("observed lock differs from the installed controller decision")
                        pending["locked"] = True
                elif event.kind == PairEventKind.CLEAR and pending is not None:
                    pending["tiles"] += event.public_payload["tiles_cleared"]
                    pending["viruses"] += event.public_payload["viruses_cleared"]
                elif event.kind == PairEventKind.SPAWN:
                    self.finish_placement(side, view)
            if view.decision_boundary == DecisionBoundary.TERMINAL:
                self.finish_placement(side, view)
                self.abort(side, view.frame_id, "terminal")
            state["frame"] = view.frame_id

    def decide(self, records):
        if not records:
            return
        inputs_by_row = self.model_inputs(records)
        for i, row in enumerate(records):
            side, frame = row["side"], row["frame"]
            state = self.states[side]
            if state["pending"] is not None or row["action"] not in row["feasible"]:
                raise ValueError("decision lacks a closed preceding placement or complete legal inventory")
            raw_pill = COLOR_MAP[np.asarray(row["pill"])]
            result = resolve_cascade(locked_field(np.frombuffer(row["board"], np.uint8), raw_pill, row["action"]))
            state["pending"] = dict(result=result, action=row["action"], frame=frame, locked=False,
                                    interrupted=False, tiles=0, viruses=0)
            inputs = inputs_by_row[i]
            for name, model in self.models.items():
                if name not in state["plans"]:
                    plan = self.start_proposal(model, inputs, frame)
                    state["plans"][name] = dict(plan=plan, start_frame=frame, goal=plan.goal,
                        root_anchor=plan.anchor, budget=plan.remaining, root_goal_observed=False,
                        anchor_revisions=0)
                record = state["plans"][name]
                plan, old_anchor = record["plan"], record["plan"].anchor
                ranking = plan.rank(model, inputs, range(512))
                feasible = set(row["feasible"])
                constrained = [a for a in ranking if a in feasible]
                if len(constrained) != len(feasible):
                    raise ValueError("proposal truncated the feasible inventory")
                record["anchor_revisions"] += int(plan.anchor != old_anchor)
                self.decisions.append(dict(side=side, frame=frame, arm=name, goal=plan.goal,
                    feasible_count=len(feasible), raw_preference_reachable=ranking[0] in feasible,
                    proposed_action=constrained[0], actual_action=row["action"],
                    agrees_with_actor=constrained[0] == row["action"],
                    actual_action_rank=constrained.index(row["action"]),
                    anchor=plan.anchor, root_anchor=record["root_anchor"]))

    def close(self):
        for side, state in self.states.items():
            self.abort(side, state["frame"], "censored")
