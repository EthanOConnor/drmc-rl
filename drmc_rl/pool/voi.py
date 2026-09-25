"""Value-of-information background scheduling.

For every candidate batch (pair, condition) the scheduler estimates how much one
batch of ``n`` games would shrink the posterior variance of the rating
*differences* the pool cares about, and plays the most valuable batch.

Per condition the fit gives the covariance Sigma of the free ratings (logits;
the anchor is fixed, unrated entrants get the prior variance). One batch of the
pair (i, j) adds Fisher information lambda = n p(1-p) / kappa along u = e_i - e_j,
where p is the model win probability and kappa the condition's seed-pair design
effect (robust over model variance). The variance of a target difference t
shrinks by the rank-one update

    delta(t) = lambda (t' Sigma u)^2 / (1 + lambda u' Sigma u).

Targets (a, b, weight) for a condition set, each counted at (w_c / sum w)^2,
w_c the condition's pace weight, because the pooled rating is the w-weighted
mean over conditions:

* frontier (weight 4): each active run's newest vs its best so far, both vs the
  anchor, and each resolving snapshot vs its unresolved neighbours;
* new or uncertain entrants (weight ``new_entrant_boost``): own rating vs the
  anchor while its pooled 95% half-width exceeds ``new_entrant_ci``;
* table (weight 1.5 * 0.85^k * 4q(1-q), q their LOS): adjacent rows k, k+1 of the
  default ranking while their order is uncertain.

A separate coverage budget (``coverage_share`` of background leases) targets
every active entrant's own rating at weight 1, so no rating goes stale and
uncertain entrants improve.

Safety rails: a new entrant (unrated, or pooled half-width above ``new_entrant_ci``
with under ``new_entrant_games`` games per condition) plays only the anchor and the ``new_entrant_peers`` most
established entrants nearest its provisional rating; over each entrant's last
``mix_window`` pool games in a set, an opponent above ``opponent_cap`` of them is
discounted 20x and an anchor share below ``anchor_floor`` triples the value of
playing the anchor.
"""
from __future__ import annotations

import math

import numpy as np

from drmc_rl.pool.ratings import pooled, superiority

FRONTIER, RESOLVE = 4.0, 3.0


def design_effect(fit):
    """Robust over model variance of the fitted ratings (>= 1): the seed-pair correlation."""
    return max(1.0, float(getattr(fit, "design", 1.0) or 1.0))


def set_targets(state, cset, view, roles, anchor, active, *, coverage, settings):
    """[(a, b, weight, label)] rating differences worth measuring in this set."""
    targets = []
    side = {e for e, r in roles.items() if r[0] in ("resolving", "maintenance")}
    if coverage:
        for e in active:
            if e != anchor:
                targets.append((e, anchor, 1.0, f"coverage {e}"))
        return targets
    for run in state.lineage_runs():
        members = [e for e in state.lineage_members(run) if e in roles]
        newest = next((e for e in members if roles[e][0] == "newest"), None)
        best = next((e for e in members if roles[e][0] == "best"), newest)
        if newest is None:
            continue
        if best != newest:
            targets.append((newest, best, FRONTIER, f"frontier {newest} vs {best}"))
            targets.append((best, anchor, FRONTIER / 2, f"frontier {best} vs anchor"))
        targets.append((newest, anchor, FRONTIER, f"frontier {newest} vs anchor"))
        for e in members:
            if roles[e][0] == "resolving":
                for o in roles[e][2]:
                    targets.append((e, o, RESOLVE, f"resolving {e} vs {o}"))
    for e in active:
        if e == anchor or e in side:
            continue
        if is_new(e, view, settings, len(cset["conditions"])):
            targets.append((e, anchor, settings["new_entrant_boost"], f"new entrant {e}"))
    table = [e for e in sorted(view, key=lambda e: -view[e]["rating"]) if e in active and e not in side]
    for k in range(len(table) - 1):
        a, b = view[table[k]], view[table[k + 1]]
        # Worth measuring only while the order is uncertain: 4 q (1 - q) with q the rows' LOS.
        q = superiority(a["rating"] - b["rating"], math.hypot(a["se"], b["se"])) or 0.5
        weight = 1.5 * 0.85 ** k * 4 * q * (1 - q)
        if weight > 1e-3:
            targets.append((table[k], table[k + 1], weight, f"table #{k + 1} {table[k]} vs {table[k + 1]}"))
    return targets


def is_new(e, view, settings, conditions):
    """Unrated, or uncertain (pooled 95% half-width above new_entrant_ci) with fewer than
    new_entrant_games games per condition of the set: a lopsided record (e.g. a floor entrant
    that loses every game) stays wide forever and must not keep new-entrant priority."""
    if e not in view:
        return True
    return 1.96 * view[e]["se"] > settings["new_entrant_ci"] and \
        view[e]["games"] < settings["new_entrant_games"] * conditions


def allowed_opponents(view, active, anchor, settings, conditions=1):
    """New or uncertain entrants: anchor plus the most established entrants near their provisional rating."""
    new = {e for e in active if e != anchor and is_new(e, view, settings, conditions)}
    # Established = well determined, not merely past the new-entrant game count.
    established = [e for e in active if e not in new and e != anchor and e in view
                   and 1.96 * view[e]["se"] <= settings["new_entrant_ci"]]
    allowed = {}
    for e in new:
        guess = view[e]["rating"] if e in view else settings["anchor_rating"]
        near = sorted(established, key=lambda o: (abs(view[o]["rating"] - guess) + 1.96 * view[o]["se"], o))
        allowed[e] = {anchor, *near[:settings["new_entrant_peers"]]}
    return new, allowed


def condition_values(fit, condition_weight, targets, active, anchor, pairs, *, games, prior_var):
    """VoI of one batch for each candidate pair in one condition; returns (values, best target labels)."""
    ids = [e for e in active if e != anchor]
    pos = {e: k for k, e in enumerate(ids)}
    n = len(ids)
    sigma = np.eye(n) * prior_var
    theta = np.zeros(n)
    if fit is not None and fit.covariance is not None and len(fit.index):
        known = [e for e in ids if e in fit.index]
        src = np.array([fit.index[e] for e in known], dtype=int)
        dst = np.array([pos[e] for e in known], dtype=int)
        if len(known):
            sigma[np.ix_(dst, dst)] = fit.covariance[np.ix_(src, src)]
            for e in known:
                theta[pos[e]] = fit.ratings[e].theta
    kappa = design_effect(fit) if fit is not None else 1.0

    def vector(a, b):
        v = np.zeros(n)
        if a in pos:
            v[pos[a]] += 1.0
        if b in pos:
            v[pos[b]] -= 1.0
        return v
    usable = [(a, b, w, label) for a, b, w, label in targets if a in pos or b in pos]
    if not usable or not pairs:
        return np.zeros(len(pairs)), [""] * len(pairs)
    T = np.stack([vector(a, b) for a, b, _, _ in usable], axis=1)          # n x K
    omega = np.array([w for _, _, w, _ in usable]) * condition_weight
    U = np.stack([vector(a, b) for a, b in pairs], axis=1)                  # n x P
    SU = sigma @ U
    denom_var = np.einsum("ij,ij->j", U, SU)
    d = np.array([(theta[pos[a]] if a in pos else 0.0) - (theta[pos[b]] if b in pos else 0.0) for a, b in pairs])
    p = 1.0 / (1.0 + np.exp(-d))
    lam = games * p * (1 - p) / kappa
    TSU = T.T @ SU                                                          # K x P
    delta = lam * TSU ** 2 / (1.0 + lam * denom_var)                        # variance reduction per target
    contrib = omega[:, None] * delta
    values = contrib.sum(axis=0)
    labels = [usable[int(k)][3] for k in contrib.argmax(axis=0)]
    return values, labels


def background_voi(scheduler, fits, worker_caps, inflight, *, coverage):
    """The highest-value background batch, or None."""
    state, settings = scheduler.state, scheduler.state.settings
    roles = scheduler.roles()
    candidates = []
    for cset in sorted(state.condition_sets.values(), key=lambda s: s["name"]):
        if cset["weight"] <= 0:
            continue
        anchor, keys = cset["anchor"], cset["conditions"]
        weights = scheduler.pace_weights(keys)
        total = sum(weights) or 1.0
        view = pooled(fits, keys, min_games=1, weights=weights) if all(k in fits for k in keys) else {}
        active_all = [e for e in state.entrant_ids(("active",))]
        targets = set_targets(state, cset, view, roles, anchor, active_all, coverage=coverage, settings=settings)
        if not targets:
            continue
        new, allowed = allowed_opponents(view, active_all, anchor, settings, len(keys))
        mix = scheduler.opponent_mix(cset["name"])
        for condition, w_c in zip(keys, weights):
            active = [e for e in active_all if scheduler.playable(condition, e, worker_caps)]
            if anchor not in active:
                if not scheduler.playable(condition, anchor, worker_caps):
                    continue
                active.append(anchor)
            pairs = []
            for i, a in enumerate(active):
                for b in active[i + 1:]:
                    if (a in new and b not in allowed[a]) or (b in new and a not in allowed[b]):
                        continue
                    if state.blocked(condition, a, b):
                        continue
                    pairs.append(tuple(sorted((a, b))))
            if not pairs:
                continue
            games = 2 * scheduler.pairs_per_batch(condition)
            values, labels = condition_values(fits.get(condition), cset["weight"] * (w_c / total) ** 2, targets,
                                              active, anchor, pairs, games=games,
                                              prior_var=settings["prior_sd"] ** 2)
            for (a, b), value, label in zip(pairs, values, labels):
                if value <= 0:
                    continue
                factor = 1.0
                for me, other in ((a, b), (b, a)):
                    if me == anchor:
                        continue
                    seen = mix.get(me)
                    if seen and sum(seen.values()) >= settings["min_rated_games"]:
                        count = sum(seen.values())
                        # The anchor is exempt from the cap (it has a floor instead).
                        if other != anchor and seen.get(other, 0) / count > settings["opponent_cap"]:
                            factor *= 0.05
                        if other == anchor and seen.get(anchor, 0) / count < settings["anchor_floor"]:
                            factor *= 3.0
                busy = len(inflight.get((condition, a, b, "leases"), ()))
                candidates.append((value * factor / (1 + busy), condition, a, b, label, cset["name"]))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
    for value, condition, a, b, label, set_name in candidates[:50]:
        seed_set, seeds = scheduler.background_seeds(condition, a, b, inflight)
        batch = scheduler._batch(condition, a, b, seeds, inflight, scheduler.pairs_per_batch(condition), why=(
            f"{'coverage' if coverage else 'background'} {set_name} ({seed_set} seeds): voi "
            f"{1e3 * value:.3g} for {label}"))
        if batch is not None:
            batch["value"] = 1e3 * value
            return batch
    return None

