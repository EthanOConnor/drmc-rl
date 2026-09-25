"""Which batch a free worker plays next.

Priority list (highest first), evaluated at every lease:

1. Replicate audits and calibration (fidelity; handled by the coordinator).
2. Focused jobs whose ``priority`` exceeds ``background_priority``, in order of
   priority, then deadline, then leases so far per target game (so equal-priority
   jobs share workers in proportion to their targets), then submission. Within a job the (pairing,
   condition) furthest below its game target goes first.
3. Background fill by value of information (drmc_rl/pool/voi.py): the batch
   that most reduces the posterior variance of the rating differences that
   matter (run frontiers, new entrants, adjacent rows of the default ranking),
   with a coverage budget and opponent diversity rails.
4. Jobs at or below ``background_priority``.

Every ``1 / background_min_share``-th lease tries background first so ratings
stay fresh while long jobs run. Every batch is whole side-swapped seed pairs
from the pairing's seed list, skipping pairs already played or leased.
"""
from __future__ import annotations

from drmc_rl.pool.store import condition_requirements, entrant_requirements


class Scheduler:
    def __init__(self, state, *, seed_source, available, capabilities, background_seeds=None, roles=None,
                 pace_weights=None, opponent_mix=None):
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
        self.pace_weights = pace_weights or (lambda keys: [1.0] * len(keys))
        # set name -> {entrant: Counter(opponent)} over each entrant's recent pool games in that set.
        self.opponent_mix = opponent_mix or (lambda name: {})
        self.leases = 0
        self.background_leases = 0
        self.job_leases = {}

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
        if job.get("exclude"):
            # e.g. exclude ["*+*"]: the trained snapshots only, never their knob variants.
            dropped = set(state.resolve_entrants(job["exclude"]))
            entrants = [e for e in entrants if e not in dropped]
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
        """Value-of-information background fill (drmc_rl/pool/voi.py); every
        ``1 / coverage_share``-th background lease is a coverage lease."""
        from drmc_rl.pool.voi import background_voi
        self.background_leases += 1
        share = self.state.settings["coverage_share"]
        coverage = share > 0 and self.background_leases % max(1, round(1 / share)) == 0
        batch = background_voi(self, fits, worker_caps, inflight, coverage=coverage)
        if batch is None:
            # Nothing undecided of the other kind: never leave a worker idle while ratings can improve.
            batch = background_voi(self, fits, worker_caps, inflight, coverage=not coverage)
        return batch

    # -- the priority list -----------------------------------------------------------
    def next_batch(self, fits, worker_caps, inflight):
        state, settings = self.state, self.state.settings
        self.leases += 1
        active = [j for j in state.jobs.values() if j["status"] == "active"]

        def behind(job):
            # Equal-priority jobs share workers in proportion to their targets: the job with the
            # fewest leases (this coordinator session) per target game goes first, so a new job
            # neither starves nor monopolises an old one.
            target = sum(r["target"] for r in self.job_progress(job))
            return self.job_leases.get(job["id"], 0) / target if target else 1.0
        share = {j["id"]: behind(j) for j in active} if len({j.get("priority", 50) for j in active}) < len(active) else {}
        jobs = sorted(active, key=lambda j: (-j.get("priority", 50), j.get("deadline") or "9999",
                                             share.get(j["id"], 0.0), j.get("created_at", ""), j["id"]))
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
                self.job_leases[job["id"]] = self.job_leases.get(job["id"], 0) + 1
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

