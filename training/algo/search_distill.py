"""Search-amplified training targets for SMDP-PPO (Gumbel-AZ-lite, phase 1).

Design: docs/SEARCH_DISTILL.md. During rollout collection a sampled fraction
of decisions is re-analyzed with the depth-2 checkpoint-reset search
(`models/policy/search_policy.SearchPolicy`) guided by the *current* trainer
net. The search output is converted into an improved policy target over the
packed candidate slots (Gumbel-MuZero-style completed Q-values + a monotone
sigma transform added to the prior logits) plus a search-consistent value
estimate. The executed action is still sampled from the behavior policy —
search output is used only as auxiliary targets:

    L = L_PPO + beta * mean_searched KL(pi_target || pi_net)

and, with ``value_mix > 0``, the value regression target of searched
decisions is blended toward the search value.

VS approximation (same as inference-time search): the learner's own board is
simulated with the single-player pool; garbage arrival between decisions and
opponent progress are ignored at depth 2. Terminal sims use the
training-consistent +1/-1 outcome values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch required by the trainer anyway
    torch = None  # type: ignore

from models.policy.candidate_packing import pack_feasible_candidates

# Raw NES color (0=Y,1=R,2=B) <-> canonical index (0=R,1=Y,2=B): an involution.
_COLOR_SWAP = (1, 0, 2)

# Effectively "no deadline": the sim budget (cfg.sims) bounds the search.
_NO_DEADLINE_MS = 1e9

# speed_ups approximation (matches tools/eval_policy.py / tools/tournament.py):
# +1 every 10 pills, capped at the NES table end.
_SPEEDUPS_MAX = 0x31


@dataclass(slots=True)
class SearchDistillConfig:
    """Config for the ``smdp_ppo.search_distill`` block (default OFF)."""

    enabled: bool = False
    sims: int = 12  # native sim envs per searched decision (fan-out budget)
    beam: int = 8  # ply-1 beam width (K)
    decision_fraction: float = 0.25  # sampled fraction of decisions searched
    beta: float = 1.0  # weight of the KL(pi_target || pi_net) term
    value_mix: float = 0.5  # blend of search value into value targets (0 = off)
    sigma_scale: float = 1.0  # c in improved_logits = prior + c * norm(Q)
    # Terminal sim values for reward_mode="vs" (training-consistent +-1; the
    # 1P reward replication carries its own terminal bonuses).
    win_value: float = 1.0
    loss_value: float = -1.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SearchDistillConfig":
        d = dict(d or {})
        cfg = cls(
            enabled=bool(d.get("enabled", False)),
            sims=int(d.get("sims", 12)),
            beam=int(d.get("beam", 8)),
            decision_fraction=float(d.get("decision_fraction", 0.25)),
            beta=float(d.get("beta", 1.0)),
            value_mix=float(d.get("value_mix", 0.5)),
            sigma_scale=float(d.get("sigma_scale", 1.0)),
            win_value=float(d.get("win_value", 1.0)),
            loss_value=float(d.get("loss_value", -1.0)),
        )
        if cfg.enabled:
            if cfg.sims < 1 or cfg.beam < 1:
                raise ValueError("search_distill: sims and beam must be >= 1")
            if not (0.0 < cfg.decision_fraction <= 1.0):
                raise ValueError("search_distill: decision_fraction must be in (0, 1]")
            if cfg.beta < 0.0:
                raise ValueError("search_distill: beta must be >= 0")
            if not (0.0 <= cfg.value_mix <= 1.0):
                raise ValueError("search_distill: value_mix must be in [0, 1]")
        return cfg


# ------------------------------------------------------------- target math
def improved_policy_target(
    prior_logits: np.ndarray,
    cand_mask: np.ndarray,
    q: np.ndarray,
    evaluated: np.ndarray,
    *,
    baseline: float,
    sigma_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gumbel-AZ-lite improved policy over packed candidate slots.

    Completed Q-values: evaluated slots keep their search-backup Q; unevaluated
    feasible slots take ``baseline`` (the net's root value — the value-equivalent
    estimate of acting from the prior). Q is then normalized to [0, 1] within
    the decision and ``improved_logits = prior_logits + sigma_scale * q_norm``
    (a monotone transform, cf. sigma(q) in Gumbel MuZero).

    Args:
        prior_logits: [K] float; net logits over packed slots (-inf ok on padding).
        cand_mask: [K] bool; True for valid candidate slots.
        q: [K] float; search-backup Q per slot (only read where ``evaluated``).
        evaluated: [K] bool; slots with a search-backup Q.
        baseline: completed-Q fill for unevaluated slots.
        sigma_scale: c >= 0 scaling of the normalized Q.

    Returns:
        (target_probs [K] float32 summing to 1 over ``cand_mask``,
         q_completed [K] float64).
    """

    mask = np.asarray(cand_mask, dtype=bool).reshape(-1)
    K = int(mask.shape[0])
    prior = np.asarray(prior_logits, dtype=np.float64).reshape(K)
    evaluated = np.asarray(evaluated, dtype=bool).reshape(K) & mask

    q_completed = np.full((K,), float(baseline), dtype=np.float64)
    q_completed[evaluated] = np.asarray(q, dtype=np.float64).reshape(K)[evaluated]

    target = np.zeros((K,), dtype=np.float32)
    if not bool(mask.any()):
        return target, q_completed

    feas_q = q_completed[mask]
    q_min = float(feas_q.min())
    q_max = float(feas_q.max())
    if q_max - q_min > 1e-12:
        q_norm = (q_completed - q_min) / (q_max - q_min)
    else:
        q_norm = np.zeros((K,), dtype=np.float64)

    logits = np.where(mask, prior + float(sigma_scale) * q_norm, -np.inf)
    # Guard against -inf priors on feasible slots (e.g. a slot missing from the
    # search's packing): they simply get zero target mass.
    finite = np.isfinite(logits)
    if not bool(finite.any()):
        target[mask] = 1.0 / float(mask.sum())
        return target, q_completed
    z = logits - logits[finite].max()
    e = np.where(finite, np.exp(np.clip(z, -60.0, 0.0)), 0.0)
    target[:] = (e / e.sum()).astype(np.float32)
    return target, q_completed


def masked_distill_kl(
    target_probs: "torch.Tensor",
    log_probs: "torch.Tensor",
    searched: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Sum over searched rows of KL(pi_target || pi_net), plus the row count.

    Args:
        target_probs: [B, K] float; rows sum to 1 (zeros on unsearched rows ok).
        log_probs: [B, K] float; net log-probabilities over the same slots.
        searched: [B] bool/float; rows contributing to the loss.

    Returns:
        (kl_sum scalar, n_searched scalar) — caller normalizes across
        minibatches (kl_sum / max(n, 1)). Unsearched rows contribute exactly 0.
    """

    t = target_probs
    t_log_t = torch.where(t > 0, t * torch.log(t.clamp_min(1e-12)), torch.zeros_like(t))
    per_row = (t_log_t - t * log_probs).sum(dim=-1)
    m = searched.to(per_row.dtype)
    return (per_row * m).sum(), m.sum()


def blend_value_targets(
    returns: "torch.Tensor",
    search_values: "torch.Tensor",
    searched: "torch.Tensor",
    value_mix: float,
) -> "torch.Tensor":
    """Blend search values into the value regression target on searched rows."""

    mix = float(value_mix)
    blended = (1.0 - mix) * returns + mix * search_values
    return torch.where(searched.to(torch.bool), blended, returns)


# ---------------------------------------------------------------- rollout side
def _make_aux_shim(aux_spec: str):
    spec = str(aux_spec or "none").strip().lower()
    if spec == "none":
        return None
    if spec != "v1":
        raise ValueError(
            f"search_distill supports aux_spec in {{none, v1}}; got {spec!r} "
            "(v1_vs nets need opponent context the 1P search sims cannot provide)"
        )
    from training.algo.ppo_smdp import _AUX_DIM_BY_SPEC, SMDPPPOAdapter

    shim = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    shim.aux_spec = spec
    shim.aux_dim = int(_AUX_DIM_BY_SPEC[spec])
    return shim


class SearchDistillRunner:
    """Rollout-side search runner: per-decision targets + metric accumulation.

    Owns a `SearchPolicy` built from a CPU copy of the trainer net (root/prior
    forwards on CPU, the big marginal leaf batch on ``device``) and a native
    pool of ``cfg.sims`` checkpoint-reset sim envs. Call `run()` once per
    vectorized decision, `note_dones()` after the env step, `refresh_weights()`
    after each PPO update, and `pop_metrics()` once per update.
    """

    def __init__(
        self,
        cfg: SearchDistillConfig,
        *,
        net: Any,
        aux_spec: str,
        candidate_max: int,
        num_envs: int,
        reward_mode: str,
        gamma: float,
        garbage_reward_coef: float = 0.05,
        device: str = "cpu",
        seed: int = 0,
        lib_path: Optional[str] = None,
    ) -> None:
        import copy

        from models.policy.search_policy import SearchPolicy

        self.cfg = cfg
        self.candidate_max = int(candidate_max)
        net_copy = copy.deepcopy(net).to("cpu")
        self._search = SearchPolicy.from_net(
            net_copy,
            aux_shim=_make_aux_shim(aux_spec),
            candidate_max=self.candidate_max,
            beam=int(cfg.beam),
            deadline_ms=_NO_DEADLINE_MS,
            device=str(device),
            num_sim_envs=int(cfg.sims),
            win_value=float(cfg.win_value),
            loss_value=float(cfg.loss_value),
            reward_mode=str(reward_mode),
            gamma=float(gamma),
            garbage_reward_coef=float(garbage_reward_coef),
            seed=int(seed),
            warmup=False,
        )
        self._rng = np.random.default_rng(int(seed))
        # Decisions since episode start, per env (speed_ups approximation).
        self._ep_decisions = np.zeros((int(num_envs),), dtype=np.int64)

        # Per-update metric accumulators (drained by pop_metrics()).
        self._decisions_total = 0
        self._searched_total = 0
        self._search_ms: List[float] = []
        self._q_gaps: List[float] = []

    def close(self) -> None:
        self._search.close()

    def refresh_weights(self, state_dict: Dict[str, Any]) -> None:
        self._search.refresh_weights(state_dict)

    def note_dones(self, dones: np.ndarray) -> None:
        """Reset the per-env episode decision counters on terminal steps."""

        self._ep_decisions[np.asarray(dones, dtype=bool)] = 0

    def run(
        self,
        infos: Sequence[Dict[str, Any]],
        masks: np.ndarray,
        costs_to_lock: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search a sampled subset of this decision's envs.

        Args:
            infos: per-env decision info dicts (must carry ``board`` to be
                searchable; envs without it are skipped).
            masks: [N, 4, 16, 8] bool feasible masks (as stored in the buffer).
            costs_to_lock: [N, 4, 16, 8] float32 (inf = unreachable).
            actions: [N] executed macro actions (behavior-policy samples; used
                only for the q_gap metric).

        Returns:
            (targets [N, Kmax] float32, search_values [N] float32,
             searched [N] bool). Unsearched rows are zeros/False.
        """

        N = int(masks.shape[0])
        targets = np.zeros((N, self.candidate_max), dtype=np.float32)
        values = np.zeros((N,), dtype=np.float32)
        searched = np.zeros((N,), dtype=bool)
        self._decisions_total += N

        # Sample the searched subset up front (Bernoulli per decision; sampled,
        # not strided, to avoid phase artifacts).
        pick = self._rng.random(N) < float(self.cfg.decision_fraction)
        for i in range(N):
            info = infos[i] if i < len(infos) else {}
            ep_dec = int(self._ep_decisions[i])
            self._ep_decisions[i] += 1
            if not bool(pick[i]):
                continue
            board = info.get("board")
            mask = np.asarray(masks[i], dtype=bool)
            if board is None or not bool(mask.any()):
                continue
            # VS env: sides not parked at a decision boundary carry a stale
            # decision context (the executed action is ignored there too).
            if info.get("placements/needs_action") is False:
                continue

            pill_can = np.asarray(info.get("next_pill_colors", (0, 0))).reshape(-1)
            pill_raw = (
                _COLOR_SWAP[int(pill_can[0]) % 3],
                _COLOR_SWAP[int(pill_can[1]) % 3],
            )
            pv = info.get("preview_pill") or {}
            prev_raw = (
                int(pv.get("first_color", 0)) & 0x03,
                int(pv.get("second_color", 0)) & 0x03,
            )
            level = int(info.get("level", 0) or 0)
            speed = int(info.get("pill/speed_setting", info.get("speed_setting", 2)) or 0)
            spd_ups = min(_SPEEDUPS_MAX, max(0, level - 20) + ep_dec // 10)
            v0 = info.get("drm/viruses_initial")
            frames_used = int(info.get("task/frames_used", 0) or 0)

            t0 = time.perf_counter()
            _action, sinfo = self._search.decide(
                np.asarray(board, dtype=np.uint8).reshape(-1),
                pill_raw,
                prev_raw,
                mask.reshape(-1),
                np.asarray(costs_to_lock[i], dtype=np.float64).reshape(-1),
                speed,
                spd_ups,
                level,
                viruses_initial=None if v0 is None else int(v0),
                frames_used=frames_used,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            if sinfo.get("q_values") is None or sinfo.get("prior_logits") is None:
                continue  # search never reached the sim stage; no target

            # Re-pack with the trainer's packing (deterministic; identical to
            # the PPO update's prepack of the stored mask/cost) and map the
            # search outputs by macro action id.
            packed = pack_feasible_candidates(
                mask,
                np.asarray(costs_to_lock[i], dtype=np.float32),
                max_candidates=self.candidate_max,
                sort_by_cost=True,
            )
            slot_of = {int(a): k for k, a in enumerate(packed.actions[: packed.count])}

            prior = np.full((self.candidate_max,), -np.inf, dtype=np.float64)
            s_actions = np.asarray(sinfo["cand_actions"]).reshape(-1)
            s_prior = np.asarray(sinfo["prior_logits"], dtype=np.float64).reshape(-1)
            for k in np.flatnonzero(np.asarray(sinfo["cand_mask"], dtype=bool).reshape(-1)):
                s = slot_of.get(int(s_actions[k]))
                if s is not None:
                    prior[s] = s_prior[k]

            qvals = np.zeros((self.candidate_max,), dtype=np.float64)
            evaluated = np.zeros((self.candidate_max,), dtype=bool)
            for a, qv in zip(
                np.asarray(sinfo["ply1_actions"]).reshape(-1),
                np.asarray(sinfo["q_values"], dtype=np.float64).reshape(-1),
            ):
                if not np.isfinite(qv):
                    continue
                s = slot_of.get(int(a))
                if s is not None:
                    evaluated[s] = True
                    qvals[s] = float(qv)

            target, q_completed = improved_policy_target(
                prior,
                packed.mask,
                qvals,
                evaluated,
                baseline=float(sinfo["value_root"]),
                sigma_scale=float(self.cfg.sigma_scale),
            )
            targets[i] = target
            values[i] = float(np.dot(target.astype(np.float64), q_completed))
            searched[i] = True

            self._searched_total += 1
            self._search_ms.append(float(elapsed_ms))
            s_exec = slot_of.get(int(actions[i]))
            if s_exec is not None:
                q_max = float(q_completed[packed.mask].max())
                self._q_gaps.append(q_max - float(q_completed[s_exec]))

        return targets, values, searched

    def pop_metrics(self) -> Dict[str, float]:
        """Drain the per-update rollout metrics (search_distill/* scalars)."""

        out: Dict[str, float] = {}
        if self._decisions_total > 0:
            out["search_distill/searched_fraction_actual"] = float(
                self._searched_total / self._decisions_total
            )
        if self._q_gaps:
            out["search_distill/q_gap_mean"] = float(np.mean(self._q_gaps))
        if self._search_ms:
            out["search_distill/search_ms_p50"] = float(np.percentile(self._search_ms, 50))
        self._decisions_total = 0
        self._searched_total = 0
        self._search_ms = []
        self._q_gaps = []
        return out


__all__ = [
    "SearchDistillConfig",
    "SearchDistillRunner",
    "blend_value_targets",
    "improved_policy_target",
    "masked_distill_kl",
]
