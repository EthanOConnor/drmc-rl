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


def policy_request(
    public: PublicPairState,
    side: int,
    legal,
    action_costs,
    *,
    context_schema="zero_v1_vs",
    execution=None,
):
    """Construct the same input contract used by the live competitive opponent."""
    if not legal or len(legal) != len(action_costs):
        raise ValueError("public continuation requires a nonempty complete frontier")
    from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA

    if context_schema not in ("zero_v1_vs", PUBLIC_CONTEXT_SCHEMA):
        raise ValueError("unknown public policy input schema")
    if len(set(legal)) != len(legal) or any(not 0 <= a < 512 for a in legal):
        raise ValueError("public frontier must contain unique valid actions")
    own, opponent = public.sides[side], public.sides[1-side]
    feasible = np.zeros(512, dtype=bool)
    costs = np.full(512, 0xFFFF, dtype=np.uint16)
    feasible[list(legal)] = True
    costs[list(legal)] = action_costs
    if context_schema == "zero_v1_vs" and own.pill[0] == own.pill[1]:
        feasible[256:] = False
        costs[256:] = 0xFFFF
    feasible = feasible.reshape(4, 16, 8)
    own_planes = board_bytes_to_semantic_planes(own.board)
    opponent_planes = board_bytes_to_semantic_planes(opponent.board)
    boards = (
        legacy_vs_policy_boards(own_planes, opponent_planes, own.pill, opponent.pill)
        if context_schema == "zero_v1_vs"
        else np.concatenate((own_planes, opponent_planes))
    )
    # PlainPolicy's preview dictionary uses raw NES colors. Reuse its exact
    # canonical/raw involution while preserving canonical board planes.
    from tools.vs_head_to_head import _CANON_TO_RAW
    raw_color = _CANON_TO_RAW
    info = {
        "placements/feasible_mask": feasible,
        "placements/cost_to_lock": costs.reshape(4, 16, 8),
        "next_pill_colors": np.asarray(own.pill, dtype=np.int64),
        "preview_pill": {
            "first_color": int(raw_color[own.preview[0]]),
            "second_color": int(raw_color[own.preview[1]]),
        },
    }
    if context_schema == PUBLIC_CONTEXT_SCHEMA:
        info.update(
            public_context_schema=PUBLIC_CONTEXT_SCHEMA,
            public_pair_state=public,
            public_acting_side=side,
            public_execution=execution,
        )
    return np.concatenate((boards, feasible.astype(np.float32))), info


class PublicPolicyContinuation:
    def __init__(self, checkpoint: Path, calibration: DavidsonCalibration | None = None, *,
                 device="cpu", cache_size=0):
        from tools.vs_head_to_head import PlainPolicy
        self.policy = PlainPolicy(checkpoint, device=device, public_only=True)
        from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA

        if self.policy.in_channels != 20 or self.policy.aux_spec not in (
            "zero_v1_vs",
            PUBLIC_CONTEXT_SCHEMA,
        ):
            raise ValueError("public continuation requires an explicitly public full-pair actor")
        self.calibration = calibration
        self._cache = OrderedDict()
        self._batch_cache = OrderedDict()
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError("public inference cache size must be nonnegative")
        self.last_inference_batch_rows = ()
        self.last_cache_hits = self.last_batch_duplicates = 0

    def infer_batch(self, requests):
        """Score complete public frontiers for independent rollout decisions."""
        self.last_inference_batch_rows = ()
        self.last_cache_hits = self.last_batch_duplicates = 0
        if not requests:
            return []
        if not getattr(self, "cache_size", 0):
            result = self._infer_uncached(requests)
            self.last_inference_batch_rows = (len(requests),)
            return result
        keys, missing, values = [], {}, {}
        for state, side in requests:
            key = self._policy_key(state, side)
            keys.append(key)
            if key in self._batch_cache:
                values[key] = self._batch_cache[key]
                self._batch_cache.move_to_end(key)
                self.last_cache_hits += 1
            elif key in missing:
                self.last_batch_duplicates += 1
            else:
                missing[key] = (state, side)
        if missing:
            answers = self._infer_uncached(list(missing.values()))
            self.last_inference_batch_rows = (len(missing),)
            for key, (probability, value) in zip(missing, answers, strict=True):
                result = dict(probability), value
                values[key] = result
                self._batch_cache[key] = result
                while len(self._batch_cache) > self.cache_size:
                    self._batch_cache.popitem(last=False)
        # Callers cannot mutate the cached probability dictionary. Keep local
        # answers even if this batch itself exceeds the bounded cache capacity.
        return [(dict(values[key][0]), values[key][1]) for key in keys]

    def _policy_key(self, state, side):
        from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMAS

        if state.public_observation_schema not in CAUSAL_PUBLIC_SCHEMAS:
            raise ValueError("public continuation requires a causal public timeline")
        if not state.privileged.need_action[side]:
            raise ValueError("public continuation requires an acting side")
        if self.policy.aux_spec != "zero_v1_vs":
            return self._request_key(state, side)
        # Exactly the public fields used by legacy_vs_policy_boards and
        # policy_request for zero_v1_vs. Clock/age and opponent preview are
        # absent from that actor's tensors; opponent board and pill are not.
        public = state.privileged.public
        own, opponent = public.sides[side], public.sides[1-side]
        return (own.board, opponent.board, own.pill, own.preview, opponent.pill,
                state.legal_actions_by_side[side], state.action_costs_by_side[side])

    def _infer_uncached(self, requests):
        observations, infos = [], []
        for state, side in requests:
            from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMAS

            if state.public_observation_schema not in CAUSAL_PUBLIC_SCHEMAS:
                raise ValueError(
                    "public continuation requires a causal public timeline; legacy warped buffers expose future locks"
                )
            if not state.privileged.need_action[side]:
                raise ValueError("public continuation requires an acting side")
            obs, info = policy_request(
                state.privileged.public,
                side,
                state.legal_actions_by_side[side],
                state.action_costs_by_side[side],
                context_schema=self.policy.aux_spec,
            )
            # Attach this only after validating the native source contract.
            # A bare PublicPairState cannot prove how its producer observed it.
            info["vs/observation_timeline"] = state.public_observation_schema
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

    @staticmethod
    def _request_key(state, side):
        from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMAS

        if state.public_observation_schema not in CAUSAL_PUBLIC_SCHEMAS:
            raise ValueError(
                "public continuation requires a causal public timeline; legacy warped buffers expose future locks"
            )
        return (
            state.privileged.public.stable_hash(),
            side,
            state.legal_actions_by_side[side],
            state.action_costs_by_side[side],
            state.public_observation_schema,
        )

    def prefetch(self, requests, *, batch_size=64):
        """Prime independent public requests without changing search backups."""
        if batch_size < 1:
            raise ValueError("frontier batch size must be positive")
        missing = {}
        for state, side in requests:
            key = self._request_key(state, side)
            if key not in self._cache:
                missing.setdefault(key, (state, side))
        pending = list(missing.items())
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            values = self.infer_batch([request for _, request in chunk])
            for (key, _), value in zip(chunk, values, strict=True):
                self._cache[key] = value
            while len(self._cache) > 8192:
                self._cache.popitem(last=False)

    def _infer(self, state, side):
        if not state.privileged.need_action[side]:
            raise ValueError("public continuation requires an acting side")
        key = self._request_key(state, side)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self.infer_batch([(state, side)])[0]
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
