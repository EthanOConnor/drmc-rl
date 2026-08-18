"""Native pair search with a public belief over the private pill reserve.

The native reserve is generated once by a deterministic two-byte RNG process.
Consequently, a newly revealed ordered color pair is neither independent nor
uniform after conditioning on the public initial virus bottle and the visible
falling and preview pills. This adapter keeps the opaque native checkpoint for
exact transitions, but chooses chance probabilities only from the declared
public seed posterior. The actual hidden reserve byte in the checkpoint is
always overwritten before its reveal.
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
    canonical_pair_to_pill_id,
    pill_id_to_raw_pair,
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
        reveal = self.runner.search_reveal_info(0)
        for side in range(2):
            counter = int(buffers.spawn_id[side]) & 0x7F
            if not bool(buffers.need_action[side]):
                # Pill/preview buffers are authoritative public observations
                # at actionable decision boundaries. During asynchronous
                # advance and reveal stops they may intentionally retain the
                # preceding boundary's values while the reserve counter moves.
                continue
            if reveal is not None and reveal[0] == side:
                if reveal[1] != counter:
                    raise RuntimeError(
                        "pending reveal index does not match the native reserve counter"
                    )
                # stop_before_reveal has advanced the reserve counter to the
                # pending entry while leaving both visible pill buffers at the
                # preceding decision boundary. Those observations are already
                # in the parent belief; re-indexing either buffer relative to
                # the advanced counter would attach stale colors to the wrong
                # reserve entry. The chance branch below conditions the new
                # preview and then normal visible conditioning resumes.
                continue
            # At a normal decision boundary the reserve counter points one
            # past the visible preview entry. The modulo relation remains true
            # across the 128-entry wrap: falling=counter-2, preview=counter-1.
            try:
                result = result.condition(
                    counter - 2,
                    canonical_pair_to_pill_id(
                        tuple(int(value) for value in buffers.pill_colors[side])
                    ),
                )
                result = result.condition(
                    counter - 1,
                    canonical_pair_to_pill_id(
                        tuple(int(value) for value in buffers.preview_colors[side])
                    ),
                )
            except ValueError as error:
                raise ValueError(
                    "visible reserve conditioning failed "
                    f"(side={side}, counter={counter}, reveal={reveal}, "
                    f"need_action={buffers.need_action.tolist()}, "
                    f"pill_colors={buffers.pill_colors.tolist()}, "
                    f"preview_colors={buffers.preview_colors.tolist()}, "
                    f"falling_prior={dict(result.observations).get((counter - 2) & 0x7F)}, "
                    f"preview_prior={dict(result.observations).get((counter - 1) & 0x7F)}, "
                    f"observations_tail={result.observations[-8:]})"
                ) from error
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
            colors = pill_id_to_raw_pair(pill_id)
            self.runner.search_reveal(0, side, colors)
            child = capture_native_state(
                self.runner,
                level=state.level,
                speed_setting=state.speed_setting,
                viruses_initial=state.viruses_initial,
            )
            child_belief = belief.condition(reserve_index, pill_id)
            try:
                child_belief = self._condition_visible(child_belief)
            except ValueError as error:
                raise ValueError(
                    "post-reveal public conditioning failed "
                    f"(revealed_side={side}, reserve_index={reserve_index}, "
                    f"pill_id={pill_id})"
                ) from error
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
