"""Which batch a free worker plays next.

Priority list (highest first), evaluated at every lease:

1. Replicate audits and calibration (fidelity; handled by the coordinator).
2. Focused jobs whose ``priority`` exceeds ``background_priority``, in order of
   priority, then deadline, then submission. Within a job the (pairing,
   condition) furthest below its game target goes first.
3. Background rating fill over every weighted condition set: for each active
   entrant ``e`` and candidate opponent ``o`` (the anchor, the
   ``nearest_opponents`` closest rated entrants, and each era's best), the
   value of one more seed pair is

       set_weight * boost(e) * p(1 - p) * (var_e + var_o) / (1 + in_flight)

   the expected Fisher information about the rating difference, where
   ``boost(e) = 1 + new_entrant_boost * [games < min_rated_games]
   + underplayed_boost * max(0, 1 - games / target_games)``. Unrated
   entrants have the prior variance and play the anchor first.
4. Jobs at or below ``background_priority``.

Every ``1 / background_min_share``-th lease tries background first so ratings
stay fresh while long jobs run. Every batch is whole side-swapped seed pairs
from the pairing's seed list, skipping pairs already played or leased.
"""
from __future__ import annotations

from drmc_rl.pool.ratings import ELO_SCALE
from drmc_rl.pool.store import condition_requirements, entrant_requirements


class Scheduler:
    def __init__(self, state, *, seed_source, available, capabilities, background_seeds=None, roles=None):
        """``seed_source(job, condition) -> list[int]``; ``available(entrant) -> bool`` (artifacts servable)."""
        self.state = state
        self.seed_source = seed_source
        self.available = available
        self.capabilities = capabilities
        # (condition, a, b, inflight) -> (seed-set name, seeds) for background pairs; default: the bank.
        self.background_seeds = background_seeds or (lambda c, a, b, inflight: ("reserve", seed_source(None, c)))
        # entrant -> (kind, detail, targets) for snapshots of active runs: newest/best get full priority,
        # resolving ones a real boost against their unresolved neighbours, maintenance a thin share.
        self.roles = roles or (lambda: {e: (("newest" if r == "newest" else "maintenance"), "", [])
                                        for e in state.entrants if (r := state.snapshot_role(e))})
        self.leases = 0

    # -- eligibility --------------------------------------------------------------
    def playable(self, condition, entrant, worker_caps):
        record = self.state.entrants.get(entrant)
        if record is None or record["status"] == "retired" or not self.available(entrant):
            return False
        needs = entrant_requirements(record) | condition_requirements(self.state, condition)
        return needs <= worker_caps and self.state.blocked(condition, entrant) is None

    def _free_seeds(self, condition, a, b, seeds, inflight, count):
        played = self.state.played_seeds(condition, a, b)
        busy = inflight.get((condition, a, b), set())
        out = []
        for seed in seeds:
            if seed not in played and seed not in busy:
                out.append(seed)
                if len(out) == count:
                    break
        return out

    def _batch(self, condition, a, b, seeds, inflight, pairs, job=None, why=""):
        a, b = sorted((a, b))
        if self.state.blocked(condition, a, b):
            return None
        if len(inflight.get((condition, a, b, "leases"), ())) >= self.state.settings["max_inflight_per_pairing"]:
            return None
        chosen = self._free_seeds(condition, a, b, seeds, inflight, pairs)
        if not chosen:
            return None
        return dict(condition=condition, a=a, b=b, seeds=chosen, job=job, why=why)

    def pairs_per_batch(self, condition):
        backend = self.state.conditions[condition]["spec"]["backend"]
        return max(1, int(self.state.settings["batch_games"].get(backend, 16)) // 2)

    # -- jobs -----------------------------------------------------------------------
    def job_items(self, job):
        """(condition, a, b) triples of one job, canonical, current entrant patterns resolved."""
        state = self.state
        conditions = state.expand_conditions(job["conditions"])
        entrants = state.resolve_entrants(job.get("entrants", []))
        if job.get("step_every"):
            # Stop-rule panels stay on their pre-registered snapshot marks (e.g. every 50M frames).
            entrants = [e for e in entrants if state.lineage_of(e) is None or state.step_of(e) % job["step_every"] == 0]
        items = []
        for condition in conditions:
            anchor = state.anchor_for(condition)
            mode = job.get("mode", "vs")
            if mode == "explicit":
                pairs = [tuple(p) for p in job["pairings"]]
            elif mode == "round_robin":
                pairs = [(x, y) for i, x in enumerate(entrants) for y in entrants[i + 1:]]
            elif mode == "vs_parent":
                pairs = []
                for e in entrants:
                    parent = (state.entrants[e].get("lineage") or {}).get("parent")
                    for o in [parent, anchor]:
                        if o and o != e and o in state.entrants:
                            pairs.append((e, o))
            else:
                opponents = state.resolve_entrants(job.get("opponents") or [anchor])
                pairs = [(e, o) for e in entrants for o in opponents if e != o]
            for a, b in pairs:
                items.append((condition, *sorted((a, b))))
        return list(dict.fromkeys(items))

    def job_progress(self, job):
        rows = []
        for condition, a, b in self.job_items(job):
            seeds = set(self.seed_source(job, condition))
            rows.append(dict(condition=condition, a=a, b=b, target=job["games"],
                             blocked=self.state.blocked(condition, a, b),
                             games=self.state.pairing_games(condition, a, b, None if job.get("seeds", "bank") == "bank"
                                                            else seeds)))
        return rows

    def job_batch(self, job, worker_caps, inflight):
        best = None
        for condition, a, b in self.job_items(job):
            if not (self.playable(condition, a, worker_caps) and self.playable(condition, b, worker_caps)):
                continue
            seeds = self.seed_source(job, condition)
            counted = None if job.get("seeds", "bank") == "bank" else set(seeds)
            games = self.state.pairing_games(condition, a, b, counted)
            games += 2 * len(inflight.get((condition, a, b), ()))
            if games >= job["games"]:
                continue
            ratio = games / job["games"]
            if best is None or ratio < best[0]:
                want = min(self.pairs_per_batch(condition), (job["games"] - games) // 2)
                batch = self._batch(condition, a, b, seeds, inflight, max(1, want), job=job["id"],
                                    why=f"job {job['id']}")
                if batch is not None:
                    best = (ratio, batch)
        return None if best is None else best[1]

    def job_complete(self, job):
        if job.get("open"):
            return False
        items = self.job_items(job)
        if not items:
            return False
        for condition, a, b in items:
            seeds = self.seed_source(job, condition)
            counted = None if job.get("seeds", "bank") == "bank" else set(seeds)
            if self.state.pairing_games(condition, a, b, counted) < job["games"]:
                free = self._free_seeds(condition, a, b, seeds, {}, 1)
                if free:
                    return False
        return True

    # -- background -----------------------------------------------------------------
    def background_batch(self, fits, worker_caps, inflight):
        state, settings = self.state, self.state.settings
        prior_var = settings["prior_sd"] ** 2
        best = None
        for cset in sorted(state.condition_sets.values(), key=lambda s: s["name"]):
            if cset["weight"] <= 0:
                continue
            anchor = cset["anchor"]
            for condition in cset["conditions"]:
                active = [e for e in state.entrant_ids(("active",)) if self.playable(condition, e, worker_caps)]
                if anchor not in active and not self.playable(condition, anchor, worker_caps):
                    continue
                fit = fits.get(condition)
                ratings = fit.ratings if fit is not None else {}

                def games(e):
                    return ratings[e].games if e in ratings else 0

                def variance(e):
                    if e == anchor:
                        return 0.0
                    return (ratings[e].se / ELO_SCALE) ** 2 if e in ratings and ratings[e].games else prior_var
                # Snapshots of an active run other than its newest and best are not opponents for
                # others: the run's background budget is carried by those two.
                roles = self.roles()
                role = {e: roles.get(e, (None, "", []))[0] for e in active}
                side = {e for e in active if role[e] in ("resolving", "maintenance")}
                rated = sorted((e for e in active if e in ratings and ratings[e].games and e not in side),
                               key=lambda e: ratings[e].rating)
                era_best = {}
                for e in rated:
                    era = state.entrants[e]["era"]
                    if era not in era_best or ratings[e].rating > ratings[era_best[era]].rating:
                        era_best[era] = e
                for e in active:
                    if e == anchor:
                        continue
                    g = games(e)
                    boost = 1.0 + settings["new_entrant_boost"] * (g < settings["min_rated_games"]) \
                        + settings["underplayed_boost"] * max(0.0, 1.0 - g / settings["target_games"])
                    if role[e] == "resolving":
                        boost = 1.0 + settings["underplayed_boost"]
                    elif role[e] == "maintenance":
                        half = 1.96 * ratings[e].se if e in ratings and g else float("inf")
                        boost = settings["maintenance_share"] if half > settings["maintenance_ci"] \
                            else settings["maintenance_idle"]
                    # An unplayed entrant is anchored first; afterwards eras and neighbours join.
                    opponents = [anchor] if not g else [anchor, *era_best.values()]
                    if role[e] == "resolving":
                        opponents += [o for o in roles[e][2] if o in active]   # its unresolved neighbours
                    if e in ratings and g:
                        near = sorted((o for o in rated if o != e),
                                      key=lambda o: (abs(ratings[o].rating - ratings[e].rating), o))
                        opponents += near[:settings["nearest_opponents"]]
                    for o in dict.fromkeys(opponents):
                        if o == e or o not in state.entrants:
                            continue
                        p = fit.expected(e, o) if fit is not None else None
                        p = 0.5 if p is None else p
                        a, b = sorted((e, o))
                        busy = len(inflight.get((condition, a, b, "leases"), ()))
                        value = cset["weight"] * boost * p * (1 - p) * (variance(e) + variance(o)) / (1 + busy)
                        key = (value, condition, a, b)
                        if best is not None and key <= best[0]:
                            continue
                        seed_set, seeds = self.background_seeds(condition, a, b, inflight)
                        batch = self._batch(condition, a, b, seeds, inflight,
                                            self.pairs_per_batch(condition), why=(
                                                f"background {cset['name']} ({seed_set} seeds): value {value:.3g}"
                                                f"{f' ({role[e]})' if role[e] else ' (new entrant)' if g < settings['min_rated_games'] else ''}"))
                        if batch is not None:
                            best = (key, batch)
        return None if best is None else best[1]

    # -- the priority list -----------------------------------------------------------
    def next_batch(self, fits, worker_caps, inflight):
        state, settings = self.state, self.state.settings
        self.leases += 1
        jobs = sorted((j for j in state.jobs.values() if j["status"] == "active"),
                      key=lambda j: (-j.get("priority", 50), j.get("deadline") or "9999", j.get("created_at", ""),
                                     j["id"]))
        share = settings["background_min_share"]
        background_first = share > 0 and self.leases % max(1, round(1 / share)) == 0
        if background_first:
            batch = self.background_batch(fits, worker_caps, inflight)
            if batch is not None:
                return batch
        for job in jobs:
            if job.get("priority", 50) <= settings["background_priority"]:
                break
            batch = self.job_batch(job, worker_caps, inflight)
            if batch is not None:
                return batch
        if not background_first:
            batch = self.background_batch(fits, worker_caps, inflight)
            if batch is not None:
                return batch
        for job in jobs:
            if job.get("priority", 50) > settings["background_priority"]:
                continue
            batch = self.job_batch(job, worker_caps, inflight)
            if batch is not None:
                return batch
        return None

