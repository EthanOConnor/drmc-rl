"""Paired candidate/continuation panels and conservative policy targets.

Full posterior enumeration is exact for the declared finite continuation
panel. Its chance variance and policy sensitivity are NOT a confidence interval
on optimal play. Natural game promotion remains an independent experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from drmc_rl.teachers.terminal_rollout import RolloutTask


@dataclass(frozen=True)
class ContinuationPair:
    actor: str
    opponent: str
    weight: float
    execution: str = "native-smdp-v1"

    def __post_init__(self):
        if not self.actor or not self.opponent or not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("continuation pair requires named members and positive mass")
        if self.execution != "native-smdp-v1":
            raise ValueError("this terminal teacher supports native SMDP execution only")


def build_panel(state, side, hypotheses, continuations):
    """Every feasible root uses exactly the same reserve/continuation panel."""
    if not np.isclose(sum(c.weight for c in continuations), 1.0, atol=1e-12, rtol=0):
        raise ValueError("continuation panel mass must sum to one")
    pairs = [(c.actor, c.opponent, c.execution) for c in continuations]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate deterministic continuation pair")
    if not hypotheses or len({r for r, _ in hypotheses}) != len(hypotheses):
        raise ValueError("reserve hypotheses must be unique and nonempty")
    if not np.isclose(sum(w for _, w in hypotheses), 1.0, atol=1e-12, rtol=0):
        raise ValueError("posterior mass must sum to one")
    tasks, inventory = [], []
    for action in state.legal_actions_by_side[side]:
        for continuation_index, member in enumerate(continuations):
            for reserve, weight in hypotheses:
                task = RolloutTask(
                    len(tasks),
                    state,
                    side,
                    action,
                    reserve,
                    weight * member.weight,
                    member.actor,
                    member.opponent,
                )
                tasks.append(task)
                inventory.append(
                    dict(
                        id=task.id,
                        action=action,
                        continuation=continuation_index,
                        reserve=hashlib.sha256(reserve).hexdigest(),
                        weight=task.weight,
                    )
                )
    return tasks, inventory


def conservative_policy_target(prior, gaps, paired_std, *, sensitivity_penalty=1.0, kl_budget=0.02):
    """pi_ref * exp(shrunk paired advantage / eta), with an explicit KL cap.

    The penalty is risk sensitivity, not an uncalibrated statistical lower
    confidence bound. All candidates must have supported terminal labels.
    """
    p, a, s = (np.asarray(x, dtype=np.float64) for x in (prior, gaps, paired_std))
    if p.ndim != 1 or p.shape != a.shape or p.shape != s.shape or not p.size:
        raise ValueError("policy target inputs require matching nonempty vectors")
    if not np.isfinite([p, a, s]).all() or (p <= 0).any() or (s < 0).any():
        raise ValueError("supported finite candidates and positive reference mass required")
    if (
        not np.isfinite(sensitivity_penalty)
        or sensitivity_penalty < 0
        or not np.isfinite(kl_budget)
        or kl_budget <= 0
    ):
        raise ValueError("invalid policy-improvement budget")
    p = p / p.sum()
    shrunk = np.sign(a) * np.maximum(0, np.abs(a) - sensitivity_penalty * s)

    def target(eta):
        logp = np.log(p) + shrunk / eta
        result = np.exp(logp - logp.max())
        result /= result.sum()
        kl = max(0.0, float(np.sum(result * np.log(np.maximum(result, 1e-300) / p))))
        return result, kl

    lo, hi = 1e-8, 1.0
    while target(hi)[1] > kl_budget:
        hi *= 2
    for _ in range(64):
        mid = (lo + hi) / 2
        if target(mid)[1] > kl_budget:
            lo = mid
        else:
            hi = mid
    result, kl = target(hi)
    return dict(
        probability=result.tolist(),
        kl_to_reference=kl,
        eta=hi,
        shrunk_advantage=shrunk.tolist(),
        sensitivity_penalty=sensitivity_penalty,
    )


def aggregate_panel(
    actions,
    incumbent,
    inventory,
    results,
    reference_prior,
    *,
    kl_budget=0.02,
    sensitivity_penalty=1.0,
):
    expected = {r["id"]: r for r in inventory}
    actual = {r["id"]: r for r in results}
    if (
        len(expected) != len(inventory)
        or len(actual) != len(results)
        or expected.keys() != actual.keys()
    ):
        raise ValueError("paired rollout coverage differs from the complete inventory")
    if (
        len(set(actions)) != len(actions)
        or set(actions) != {r["action"] for r in inventory}
        or incumbent not in actions
    ):
        raise ValueError("full feasible action inventory and incumbent are required")
    by_action = {a: {} for a in actions}
    for id, item in expected.items():
        result = actual[id]
        if result["weight"] != item["weight"] or result["outcome"] not in (None, 1, 2, 3):
            raise ValueError("invalid paired terminal outcome or changed scenario mass")
        key = (item["continuation"], item["reserve"])
        if key in by_action[item["action"]]:
            raise ValueError("duplicate deterministic scenario cannot count as new evidence")
        by_action[item["action"]][key] = (item["weight"], result["outcome"])
    ref = by_action[incumbent]
    if not np.isclose(sum(w for w, _ in ref.values()), 1.0, atol=1e-12, rtol=0):
        raise ValueError("candidate scenario mass must sum to one")
    for rows in by_action.values():
        if rows.keys() != ref.keys() or any(rows[k][0] != ref[k][0] for k in ref):
            raise ValueError("root candidates require identical paired scenarios")
    utility = {1: 1.0, 2: -1.0, 3: 0.0}
    positions = {1: 0, 3: 1, 2: 2}
    records = []
    for action, rows in by_action.items():
        record = dict(
            action=action,
            wdl=None,
            paired_gap=None,
            paired_std=None,
            unknown_mass=sum(w for w, o in rows.values() if o is None),
        )
        if all(o is not None for _, o in rows.values()):
            wdl = np.zeros(3)
            means = {}
            mass = {}
            second = {}
            for (member, _reserve), (w, o) in rows.items():
                wdl[positions[o]] += w
                means[member] = means.get(member, 0.0) + w * utility[o]
                second[member] = second.get(member, 0.0) + w * utility[o] ** 2
                mass[member] = mass.get(member, 0.0) + w
            mean = sum(means.values())
            chance = sum(second[k] - means[k] ** 2 / mass[k] for k in means)
            sensitivity = sum(mass[k] * (means[k] / mass[k] - mean) ** 2 for k in means)
            record.update(
                wdl=wdl.tolist(),
                utility=mean,
                chance_variance=max(0.0, chance),
                continuation_sensitivity=max(0.0, sensitivity),
                continuation_utilities={str(k): means[k] / mass[k] for k in means},
            )
            if all(o is not None for _, o in ref.values()):
                differences = [(w, utility[o] - utility[ref[k][1]]) for k, (w, o) in rows.items()]
                gap = sum(w * d for w, d in differences)
                record.update(
                    paired_gap=gap,
                    paired_std=float(np.sqrt(sum(w * (d - gap) ** 2 for w, d in differences))),
                )
        records.append(record)
    target = None
    if all(r["paired_gap"] is not None for r in records):
        target = conservative_policy_target(
            reference_prior,
            [r["paired_gap"] for r in records],
            [r["paired_std"] for r in records],
            kl_budget=kl_budget,
            sensitivity_penalty=sensitivity_penalty,
        )
    return dict(
        schema="drmc-paired-terminal-quality-v1",
        actions=list(actions),
        incumbent=incumbent,
        reference_prior=list(reference_prior),
        candidates=records,
        policy_target=target,
        distinct_scenarios=len(ref),
        continuation_pairs=len({k[0] for k in ref}),
        candidate_truncation=0,
        posterior_enumerated=True,
        sampling_standard_error=None,
        learned_evaluator_disagreement=None,
        optimal_quality_claim=False,
    )
