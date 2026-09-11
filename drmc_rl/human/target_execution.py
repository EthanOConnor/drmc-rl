"""Explicit experimental controls for immutable-target construction studies.

This is not installed in the trainer. Calibrated competitive admission is absent.
"""
from __future__ import annotations

from drmc_rl.human.spatial_execution import ConstructionObserver
from drmc_rl.human.target_construction import TargetProposal, propose_target


class ConstructionController(ConstructionObserver):
    def __init__(self, model, root_proposer, encoder, jobs, *, control, budget=6, feature_device="cpu"):
        super().__init__({"target":model}, encoder, jobs, feature_device=feature_device)
        if not 2 <= budget <= 6:
            raise ValueError("a fixed 2–6 placement budget is required")
        self.root_proposer, self.control, self.budget = root_proposer, bool(control), budget
        self.selections, self._cached_inputs = [], None

    @staticmethod
    def input_key(records):
        return tuple((r['side'],r['frame'],r['board'],r['pill'],r['preview']) for r in records)

    def model_inputs(self, records):
        if self._cached_inputs is None or self._cached_inputs[0] != self.input_key(records):
            raise ValueError("installed controls must use the exact public inputs already ranked")
        inputs = self._cached_inputs[1]
        self._cached_inputs = None
        return inputs

    def start_proposal(self, model, inputs, frame):
        goal, anchor = propose_target(self.root_proposer, inputs)
        return TargetProposal.start_requested(model, inputs, frame=frame, goal=goal,
                                               anchor=anchor, budget=self.budget)

    def select_decisions(self, records):
        if not records:
            return {}
        if self._cached_inputs is not None:
            raise ValueError("a previous decision was not installed")
        inputs_by_row = super().model_inputs(records)
        chosen = {}
        for row, inputs in zip(records, inputs_by_row, strict=True):
            side, frame = row['side'], row['frame']
            state = self.states[side]
            if state['pending'] is not None:
                raise ValueError("previous placement is not closed")
            model = self.models['target']
            if 'target' not in state['plans']:
                plan = self.start_proposal(model, inputs, frame)
                state['plans']['target'] = dict(plan=plan, start_frame=frame, goal=plan.goal,
                    root_anchor=plan.anchor, budget=plan.remaining, root_goal_observed=False,
                    anchor_revisions=0)
            plan = state['plans']['target']['plan']
            ranking = plan.rank(model, inputs, row['feasible'])
            selected = ranking[0] if self.control else row['incumbent_action']
            chosen[side] = selected
            self.selections.append(dict(side=side, frame=frame, requested_anchor=plan.anchor,
                goal=plan.goal, remaining=plan.remaining, proposed_action=ranking[0],
                incumbent_action=row['incumbent_action'], selected_action=selected,
                controls_player=self.control, quality_admitted=False))
        self._cached_inputs = self.input_key(records), inputs_by_row
        return chosen
