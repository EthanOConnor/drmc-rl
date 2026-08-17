"""Native pair search with a public belief over the private pill reserve.

The retail reserve is generated once by a deterministic two-byte RNG process.
Consequently, a newly revealed ordered color pair is neither independent nor
uniform after conditioning on the visible falling and preview pills.  This
adapter keeps the opaque native checkpoint for exact transitions, but chooses
chance probabilities only from a public seed posterior.  The actual hidden
reserve byte in the checkpoint is always overwritten before its reveal.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Mapping, Sequence

from drmc_rl.search.joint_event import ChanceOutcome
from drmc_rl.search.native_pair import (
    NativePairSearchModel,
    NativePairSearchState,
    capture_native_state,
    state_from_payload,
)
from drmc_rl.search.pill_belief import (
    CHANCE_MODEL_ID,
    PillReserveBelief,
    pill_id_to_canonical_pair,
)


class BeliefNativePairSearchModel(NativePairSearchModel):
    """Exact native transitions plus posterior-predictive reveal branching."""

    chance_model = CHANCE_MODEL_ID

    def __init__(self, *args: Any, belief_cache_size: int = 65536, **kwargs: Any) -> None:
        kwargs["reveal_chance"] = True
        super().__init__(*args, **kwargs)
        self.belief_cache_size = max(1, int(belief_cache_size))
        self._belief_by_checkpoint: OrderedDict[str, PillReserveBelief] = OrderedDict()

    @staticmethod
    def _checkpoint_key(state: NativePairSearchState) -> str:
        return hashlib.sha256(state.privileged.engine_checkpoint).hexdigest()

    def register_belief(
        self, state: NativePairSearchState, belief: PillReserveBelief
    ) -> None:
        key = self._checkpoint_key(state)
        self._belief_by_checkpoint[key] = belief
        self._belief_by_checkpoint.move_to_end(key)
        if len(self._belief_by_checkpoint) > self.belief_cache_size:
            self._belief_by_checkpoint.popitem(last=False)

    def belief(self, state: NativePairSearchState) -> PillReserveBelief:
        key = self._checkpoint_key(state)
        cached = self._belief_by_checkpoint.get(key)
        if cached is not None:
            self._belief_by_checkpoint.move_to_end(key)
            return cached
        self.runner.restore(0, state.privileged.engine_checkpoint)
        belief = self._condition_visible(PillReserveBelief())
        self.register_belief(state, belief)
        return belief

    def _condition_visible(self, belief: PillReserveBelief) -> PillReserveBelief:
        buffers = self.runner.buffers
        result = belief
        for side in range(2):
            counter = int(buffers.spawn_id[side]) & 0x7F
            # Normal pair play has already generated falling and preview pills
            # when counter>=2.  Counters 0/1 are setup/checkpoint edge cases for
            # which the two-entry relation is not guaranteed.
            if counter < 2:
                continue
            result = result.condition_visible(
                reserve_counter=counter,
                falling_colors=tuple(int(value) for value in buffers.pill_colors[side]),
                preview_colors=tuple(
                    int(value) for value in buffers.preview_colors[side]
                ),
            )
        return result

    def key(self, state: NativePairSearchState):
        return super().key(state), self.belief(state).stable_hash()

    def apply_actions(
        self,
        state: NativePairSearchState,
        action_p1: int | None,
        action_p2: int | None,
    ) -> NativePairSearchState:
        parent_belief = self.belief(state)
        child = super().apply_actions(state, action_p1, action_p2)
        self.register_belief(child, self._condition_visible(parent_belief))
        return child

    def chance_outcomes(
        self, state: NativePairSearchState
    ) -> Sequence[ChanceOutcome[NativePairSearchState]]:
        self.runner.restore(0, state.privileged.engine_checkpoint)
        reveal = self.runner.search_reveal_info(0)
        if reveal is None:
            return ()
        side, reserve_index = reveal
        belief = self.belief(state)
        probability = belief.probabilities(reserve_index)
        outcomes: list[ChanceOutcome[NativePairSearchState]] = []
        for pill_id, mass in enumerate(probability):
            if float(mass) <= 0.0:
                continue
            self.runner.restore(0, state.privileged.engine_checkpoint)
            colors = pill_id_to_canonical_pair(pill_id)
            self.runner.search_reveal(0, side, colors)
            child = capture_native_state(
                self.runner,
                level=state.level,
                speed_setting=state.speed_setting,
                viruses_initial=state.viruses_initial,
            )
            child_belief = belief.condition(reserve_index, pill_id)
            child_belief = self._condition_visible(child_belief)
            self.register_belief(child, child_belief)
            outcomes.append(ChanceOutcome(float(mass), child))
        if not outcomes:
            raise RuntimeError("reserve belief produced no possible reveal outcomes")
        return tuple(outcomes)


def belief_diagnostic_factory(args: Any):
    """Restore/coverage adapter using exact chance belief and diagnostic leaves."""

    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner

    model = BeliefNativePairSearchModel(DrMarioVsPoolRunner(num_pairs=1))

    def decode(payload: Mapping[str, Any]) -> NativePairSearchState:
        state = state_from_payload(payload)
        belief_payload = payload.get("reserve_belief")
        if isinstance(belief_payload, Mapping):
            model.register_belief(state, PillReserveBelief.from_dict(belief_payload))
        return state

    return model, decode


__all__ = ["BeliefNativePairSearchModel", "belief_diagnostic_factory"]
