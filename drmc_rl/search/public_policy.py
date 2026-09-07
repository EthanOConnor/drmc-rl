"""Use the deployed public competitive core as an offline pair-search teacher.

Only public board/pill/reachability inputs reach the network. Native transition
checkpoints remain privileged, so this adapter alone is not a deployable public
search agent. Its current calibration is diagnostic until the quality gate.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from drmc_rl.game.observation import board_bytes_to_semantic_planes, legacy_vs_policy_boards
from drmc_rl.game.pair_state import PublicPairState
from drmc_rl.search.joint_event import WDL
from drmc_rl.search.strong_league import DavidsonCalibration


def policy_request(public: PublicPairState, side: int, legal, action_costs):
    """Construct the same input contract used by the live competitive opponent."""
    if not legal or len(legal) != len(action_costs):
        raise ValueError("public continuation requires a nonempty complete frontier")
    own, opponent = public.sides[side], public.sides[1-side]
    feasible = np.zeros(512, dtype=bool)
    costs = np.full(512, 0xFFFF, dtype=np.uint16)
    feasible[list(legal)] = True
    costs[list(legal)] = action_costs
    if own.pill[0] == own.pill[1]:
        feasible[256:] = False
        costs[256:] = 0xFFFF
    feasible = feasible.reshape(4, 16, 8)
    boards = legacy_vs_policy_boards(
        board_bytes_to_semantic_planes(own.board),
        board_bytes_to_semantic_planes(opponent.board), own.pill, opponent.pill)
    # PlainPolicy's preview dictionary uses raw NES colors. Reuse its exact
    # canonical/raw involution while preserving canonical board planes.
    from tools.vs_head_to_head import _CANON_TO_RAW
    raw_color = _CANON_TO_RAW
    return np.concatenate((boards, feasible.astype(np.float32))), {
        "placements/feasible_mask": feasible,
        "placements/cost_to_lock": costs.reshape(4, 16, 8),
        "next_pill_colors": np.asarray(own.pill, dtype=np.int64),
        "preview_pill": {"first_color": int(raw_color[own.preview[0]]),
                         "second_color": int(raw_color[own.preview[1]])},
    }


class PublicPolicyContinuation:
    def __init__(self, checkpoint: Path, calibration: DavidsonCalibration | None = None, *, device="cpu"):
        from tools.vs_head_to_head import PlainPolicy
        self.policy = PlainPolicy(checkpoint, device=device, public_only=True)
        if self.policy.in_channels != 20 or self.policy.aux_spec != "zero_v1_vs":
            raise ValueError("public continuation requires the frozen full-pair zero-aux actor")
        self.calibration = calibration
        self._cache = OrderedDict()

    def infer_batch(self, requests):
        """Score complete public frontiers for independent rollout decisions."""
        if not requests:
            return []
        observations, infos = [], []
        for state, side in requests:
            if not state.privileged.need_action[side]:
                raise ValueError("public continuation requires an acting side")
            obs, info = policy_request(
                state.privileged.public, side, state.legal_actions_by_side[side],
                state.action_costs_by_side[side])
            observations.append(obs)
            infos.append(info)
        actions, masks, logits, values = self.policy.score_and_value(np.stack(observations), infos)
        results = []
        for action, mask, scores, value, info in zip(actions, masks, logits, values, infos, strict=True):
            if int(mask.sum()) != int(info["placements/feasible_mask"].sum()):
                raise RuntimeError("public continuation truncated a candidate")
            scores = scores[mask].astype(np.float64)
            if not np.isfinite(scores).all() or not np.isfinite(value):
                raise ValueError("non-finite public continuation prediction")
            weights = np.exp(np.clip(scores-scores.max(), -60, 0))
            weights /= weights.sum()
            results.append((dict(zip(map(int, action[mask]), map(float, weights), strict=True)),
                            float(value)))
        return results

    def _infer(self, state, side):
        if not state.privileged.need_action[side]:
            raise ValueError("public continuation requires an acting side")
        public = state.privileged.public
        legal, costs = state.legal_actions_by_side[side], state.action_costs_by_side[side]
        key = (public.stable_hash(), side, legal, costs)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        obs, info = policy_request(public, side, legal, costs)
        actions, mask, logits, values = self.policy.score_and_value(obs[None], [info])
        valid = mask[0]
        if int(valid.sum()) != int(info["placements/feasible_mask"].sum()):
            raise RuntimeError("public continuation truncated a candidate")
        scores = logits[0, valid].astype(np.float64)
        weights = np.exp(np.clip(scores - scores.max(), -60, 0))
        weights /= weights.sum()
        result = (dict(zip(map(int, actions[0, valid]), map(float, weights), strict=True)),
                  float(values[0]))
        if not np.isfinite(result[1]):
            raise ValueError("non-finite public continuation value")
        self._cache[key] = result
        if len(self._cache) > 8192:
            self._cache.popitem(last=False)
        return result

    def prior(self, state, side, actions):
        probability, _value = self._infer(state, side)
        return [max(1e-8, probability.get(int(action), 0)) for action in actions]

    def evaluate(self, state, root_side):
        if self.calibration is None:
            raise ValueError("public continuation value requires a fitted W/D/L calibration")
        need = state.privileged.need_action
        if not any(need):
            raise ValueError("public continuation value requires an actionable boundary")
        side = root_side if need[root_side] else 1-root_side
        _probability, score = self._infer(state, side)
        value = self.calibration.wdl(score)
        return value if side == root_side else WDL(value.loss, value.draw, value.win)


def public_policy_belief_factory(args):
    from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner
    from drmc_rl.search.belief_native_pair import BeliefNativePairSearchModel
    from drmc_rl.search.native_pair import state_from_payload
    from drmc_rl.search.strong_league_memberwise import _register_payload_belief
    from drmc_rl.teachers.counterfactual import WeightedTeacherModels
    from drmc_rl.teachers.counterfactual_release import sha256_file

    calibration_path = Path(args.public_search_calibration)
    artifact = json.loads(calibration_path.read_text())
    if artifact.get("schema") != "drmc-public-value-audit-v1":
        raise ValueError("unsupported public continuation calibration")
    if artifact["checkpoint_sha256"] != sha256_file(Path(args.public_checkpoint)):
        raise ValueError("public continuation checkpoint/calibration mismatch")
    if not artifact.get("game_sets_disjoint"):
        raise ValueError("public continuation calibration leaks evaluation games")
    calibration = DavidsonCalibration(**artifact["parameters"], artifact_sha256=sha256_file(calibration_path))
    continuation = PublicPolicyContinuation(Path(args.public_checkpoint), calibration, device=args.device)
    model = BeliefNativePairSearchModel(DrMarioVsPoolRunner(num_pairs=1), continuation=continuation)
    model.information_scope = "public-continuation-privileged-transition-v1"

    def decode(payload):
        state = state_from_payload(payload)
        _register_payload_belief(model, state, payload)
        return state

    return WeightedTeacherModels((model,), (1.0,), ("public-outcome-competitive",)), decode
