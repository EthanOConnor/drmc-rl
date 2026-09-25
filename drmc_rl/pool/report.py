"""Pool reports: JSON export, CLI summary, the LAN web page and the stop-rule query."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from drmc_rl.pool import intentions as intent
from drmc_rl.pool.ratings import MODEL, pooled, pooled_difference_se, superiority


def _entrant_view(state, e):
    r = state.entrants[e]
    return dict(id=e, name=r.get("name", e), era=r["era"], status=r["status"], tags=r.get("tags", []),
                lineage=r.get("lineage", {}), added_at=r.get("added_at"), notes=r.get("notes", ""))


def frames_label(step):
    step = int(step or 0)
    if step >= 10 ** 9:
        return f"{step / 1e9:.3g}B"
    return f"{step / 1e6:.4g}M" if step else "0"


def _round(value):
    return None if value is None else round(value, 4)


def _bounds(r):
    return [round(r["rating"] - 1.96 * r["se"]), round(r["rating"] + 1.96 * r["se"])]


def collapse_lineages(rows, state, difference_se=None):
    """One row per training run: its best snapshot once concluded, else its newest rated one.

    The row carries the run's trajectory (every rated snapshot, retired ones included,
    by frames). Entrants outside lineages pass through unchanged.
    """
    if state is None:
        return rows
    out, groups = [], {}
    for r in rows:
        run = state.lineage_of(r["entrant"])
        if run is None:
            out.append(r)
        else:
            groups.setdefault(run, []).append(r)
    for run, members in groups.items():
        members.sort(key=lambda r: (state.step_of(r["entrant"]), r["entrant"]))
        record = state.lineages.get(run, {})
        present = {r["entrant"]: r for r in members}
        if state.lineage_status(run) == "concluded":
            pick = record.get("best") if record.get("best") in present else max(present, key=lambda e: present[e]["rating"])
        else:
            live = [r["entrant"] for r in members if state.entrants[r["entrant"]]["status"] != "retired"]
            pick = (live or [members[-1]["entrant"]])[-1]
        rep = dict(present[pick], lineage=run, lineage_status=state.lineage_status(run),
                   label=f"{run} · {frames_label(state.step_of(pick))}",
                   trajectory=[dict(entrant=r["entrant"], step=state.step_of(r["entrant"]),
                                    frames=frames_label(state.step_of(r["entrant"])), rating=round(r["rating"]),
                                    ci95=_bounds(r), games=r["games"],
                                    retired=state.entrants[r["entrant"]]["status"] == "retired",
                                    shown=r["entrant"] == pick,
                                    # P(this snapshot is stronger than the next one by frames).
                                    los=None if i + 1 == len(members) or difference_se is None else _round(superiority(
                                        r["rating"] - members[i + 1]["rating"],
                                        difference_se(r["entrant"], members[i + 1]["entrant"]))))
                                for i, r in enumerate(members)])
        out.append(rep)
    return out


def display_rows(rows, difference_se, state=None):
    """Rows sorted strongest first, for display: integer rating and 95% bounds, and the
    likelihood of superiority over the next row, P(rating_i > rating_i+1), from the fitted
    covariance of the two estimates (None for the last row or when it is undefined).
    With ``state``, snapshots of a training run collapse to one row with its trajectory."""
    rows = sorted(collapse_lineages(rows, state, difference_se), key=lambda r: -r["rating"])
    out = []
    for i, r in enumerate(rows):
        below = rows[i + 1] if i + 1 < len(rows) else None
        los = None if below is None else superiority(r["rating"] - below["rating"],
                                                       difference_se(r["entrant"], below["entrant"]))
        out.append(dict(r, rating=round(r["rating"]), se=round(r["se"], 1),
                        ci95=_bounds(r), los=None if los is None else round(los, 4)))
    return out


def los_text(value):
    return "–" if value is None else f"{round(100 * value)}%"


def primary_set(state):
    sets = sorted(state.condition_sets.values(), key=lambda s: (not s["primary"], s["name"]))
    return sets[0] if sets else None


def build_report(coordinator):
    state, settings = coordinator.state, coordinator.settings
    fits = coordinator.all_fits()
    now = coordinator.clock()
    batches = list(state.batches_journal.read())
    recent = [b for b in batches if b.get("source") == "pool" and now - b.get("unix", 0) < 3600]
    day = [b for b in batches if b.get("source") == "pool" and now - b.get("unix", 0) < 86400]
    games_by_source = {}
    for row in state.games.values():
        source = "pool" if row["source"] == "pool" else "imported"
        games_by_source[source] = games_by_source.get(source, 0) + 1
    conditions = []
    for key, c in sorted(state.conditions.items(), key=lambda kv: kv[1]["name"]):
        f = fits.get(key)
        games = sum(len(sides) for seeds in state.by_pairing.get(key, {}).values() for sides in seeds.values())
        conditions.append(dict(key=key, name=c["name"], spec=c["spec"], games=games,
                               anchor=None if f is None else f.anchor,
                               ratings=[] if f is None else display_rows([r.to_dict() for r in f.ratings.values()],
                                                                         f.difference_se, state),
                               unanchored=[] if f is None else f.unanchored))
    sets = []
    view_fits = {v: coordinator.view_fits(v) for v in ("real_play", "uniform")}

    def set_rows(fitmap, keys, weighting):
        weights = coordinator.pace_weights(keys, weighting)
        view = pooled(fitmap, keys, min_games=1, weights=weights)
        return display_rows(list(view.values()), lambda a, b: pooled_difference_se(fitmap, keys, a, b, weights), state)
    for cset in sorted(state.condition_sets.values(), key=lambda s: (not s["primary"], s["name"])):
        keys = cset["conditions"]
        by_key = {c["key"]: c for c in conditions}
        paces = [dict(pace=state.conditions[k]["spec"]["pace"], condition=state.conditions[k]["name"], key=k,
                      ratings=by_key[k]["ratings"]) for k in cset["conditions"]]
        sets.append(dict(name=cset["name"], anchor=cset["anchor"], primary=cset["primary"], weight=cset["weight"],
                         paces=paces,
                         notes=cset["notes"], conditions=[state.conditions[k]["name"] for k in cset["conditions"]],
                         weights={state.conditions[k]["spec"]["pace"]: w
                                  for k, w in zip(keys, coordinator.pace_weights(keys))},
                         # Default ranking: pace-weighted pooled rating over all comparable games.
                         pooled=set_rows(fits, keys, "pace"), pooled_equal=set_rows(fits, keys, "equal"),
                         views=dict(real_play=set_rows(view_fits["real_play"], keys, "pace"),
                                    uniform=set_rows(view_fits["uniform"], keys, "pace"))))
    primary = primary_set(state)
    primary_view = pooled(fits, primary["conditions"], min_games=1,
                          weights=coordinator.pace_weights(primary["conditions"])) if primary else {}
    eras = {}
    for e in state.entrants:
        era = state.entrants[e]["era"]
        slot = eras.setdefault(era, dict(era=era, entrants=0, best=None))
        slot["entrants"] += 1
        if e in primary_view and (slot["best"] is None or primary_view[e]["rating"] > slot["best"]["rating"]):
            slot["best"] = dict(primary_view[e], name=state.entrants[e].get("name", e))
    for slot in eras.values():
        if slot["best"]:
            b = slot["best"]
            slot["best"] = dict(b, rating=round(b["rating"]), se=round(b["se"], 1),
                                ci95=[round(b["rating"] - 1.96 * b["se"]), round(b["rating"] + 1.96 * b["se"])])
    min_rated = settings["min_rated_games"]
    new = []
    week = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for e in state.entrant_ids(("active", "benched")):
        if state.snapshot_role(e) == "older":
            continue
        per = {}
        for key in (primary["conditions"] if primary else []):
            f = fits.get(key)
            per[state.conditions[key]["name"]] = f.ratings[e].games if f and e in f.ratings else 0
        if (state.entrants[e].get("added_at") or "") >= week or any(g < min_rated for g in per.values()):
            new.append(dict(_entrant_view(state, e), games=per, rated=e in primary_view,
                            pooled=None if e not in primary_view else round(primary_view[e]["rating"])))
    memorization = memorization_rows(coordinator)
    jobs = []
    for job in sorted(state.jobs.values(), key=lambda j: (j["status"] != "active", -j.get("priority", 50), j["id"])):
        progress = coordinator.scheduler.job_progress(job)
        jobs.append(dict(id=job["id"], title=job.get("title", ""), status=job["status"],
                         priority=job.get("priority", 50), deadline=job.get("deadline"), owner=job.get("owner", ""),
                         intention=job.get("intention"), games=sum(min(p["games"], p["target"]) for p in progress),
                         target=sum(p["target"] for p in progress), items=len(progress),
                         blocked_items=sum(1 for p in progress if p["blocked"])))
    workers = []
    for w in sorted(coordinator.workers.values(), key=lambda w: w["worker_id"]):
        workers.append(dict(worker=w["worker_id"], host=w.get("host"), numerics=w.get("numerics"),
                            admitted=coordinator.admission(w.get("numerics", "")),
                            seen_seconds_ago=round(now - w["seen"]), status=w.get("status"),
                            batches=w.get("batches", 0), games=w.get("games", 0),
                            games_per_hour=round(3600 * w.get("games", 0) / w["seconds"]) if w.get("seconds") else None,
                            failures=w.get("failures", 0), last_error=w.get("last_error")))
    return dict(
        schema="drmc-rating-pool-report-v1", generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=MODEL, anchor_rating=settings["anchor_rating"], source=coordinator.source,
        capabilities=sorted(coordinator.capabilities),
        totals=dict(games=len(state.games), **games_by_source, entrants=len(state.entrants),
                    active_entrants=len(state.entrant_ids(("active",))), conditions=len(state.conditions),
                    games_last_hour=sum(b["new_games"] for b in recent),
                    games_last_day=sum(b["new_games"] for b in day), leases=len(coordinator.leases),
                    trace_mb=round(coordinator.trace_bytes / 2 ** 20, 1)),
        condition_sets=sets, conditions=conditions,
        eras=sorted(eras.values(), key=lambda s: -(s["best"] or {}).get("rating", -1e9)),
        memorization=memorization, pace_weights=dict(settings["pace_weights"]),
        pace_weights_status=settings["pace_weights_status"],
        seed_sets={k: len(v) for k, v in coordinator.seed_sets().items()},
        new_entrants=new, entrants=[_entrant_view(state, e) for e in sorted(state.entrants)], jobs=jobs,
        lineages=[dict(run=run, status=state.lineage_status(run), members=len(state.lineage_members(run)),
                       newest=state.newest(run), **{k: v for k, v in state.lineages.get(run, {}).items()
                                                    if k in ("best", "final", "reason", "stop_rule")})
                  for run in state.lineage_runs()],
        intentions=intent.roadmap(state, coordinator.capabilities),
        workers=workers, blocks=list(state.blocks.values()),
        fidelity=dict(mode=settings["fidelity"], min_agreement=settings["min_agreement"],
                      admitted=state.admitted, classes=coordinator.fidelity_stats,
                      pending_audits=len(coordinator.audits)))


def memorization_rows(coordinator):
    """Per entrant: seen-minus-reserve score gap (percentage points) on pool games.

    Seen = the uniform non-reserve seed set, reserve = the rating bank, so both sides are
    uniform draws of console games and differ only in whether training may have seen them.
    Flagged when the 95% interval excludes 0 by more than ``memorization_flag_points``.
    """
    from types import SimpleNamespace
    from drmc_rl.program.seed_reserve import detectable_gap, memorization_report
    state = coordinator.state
    sets = coordinator.seed_sets()
    reserve, uniform = set(sets["reserve"]), set(sets.get("uniform", ()))
    rows = {}
    for game in state.games.values():
        if game.get("source") != "pool" or game["score"] is None or not state.counts(game):
            continue
        if game["seed"] not in reserve and game["seed"] not in uniform:
            continue
        name = state.conditions[game["condition"]]["name"]
        for e, score in ((game["a"], game["score"]), (game["b"], 1.0 - game["score"])):
            rows.setdefault(e, []).append(dict(seed=game["seed"], score=score, condition=name))
    flag = float(coordinator.settings["memorization_flag_points"]) / 100
    out = []
    for e, items in sorted(rows.items()):
        report = memorization_report(items, reserve=SimpleNamespace(blocked=frozenset(reserve)))
        held = {r["seed"] for r in items if r["seed"] in reserve}
        seen = {r["seed"] for r in items if r["seed"] not in reserve}
        plays = len(items) / 2 / max(len(held | seen), 1)
        pooled_gap = report.get("pooled")
        row = dict(entrant=e, reserve_seeds=len(held), seen_seeds=len(seen), games=len(items),
                   detectable_points=round(100 * detectable_gap(max(len(held), 1), max(plays, 1e-9),
                                                                max(len(seen), 1)), 1) if held and seen else None)
        if pooled_gap:
            low, high = pooled_gap["ci95"]
            row.update(gap_points=round(100 * pooled_gap["gap"], 1), ci95_points=[round(100 * low, 1), round(100 * high, 1)],
                       flagged=low > flag or high < -flag)
        out.append(row)
    return sorted(out, key=lambda r: (not r.get("flagged", False), -(r.get("gap_points") or -1e9)))


def stop_rule(coordinator, *, run, condition_set=None, min_games=128, patience=2, step_every=None,
              weighting="pace"):
    """Pre-registered-style stop rule computed from pool ratings.

    Snapshots are the entrants whose ``lineage.run`` is ``run``, ordered by
    ``lineage.step``. The baseline is the first snapshot's ``lineage.parent``.
    A snapshot improves when its pooled rating over ``condition_set`` (the
    run's set, default the primary set) exceeds the best of the baseline and
    every earlier snapshot. The rule fires after ``patience`` consecutive
    non-improving snapshots; a snapshot with fewer than ``min_games`` games in
    any condition of the set is pending and stops evaluation there.
    ``step_every`` keeps the rule on its registered snapshot marks (e.g. every
    50M frames); snapshots between marks are trajectory only.
    """
    state = coordinator.state
    snapshots = sorted((e for e, r in state.entrants.items() if (r.get("lineage") or {}).get("run") == run),
                       key=lambda e: (state.entrants[e]["lineage"].get("step", 0), e))
    if step_every:
        snapshots = [e for e in snapshots if state.step_of(e) % int(step_every) == 0]
    if not snapshots:
        raise KeyError(f"no entrants with lineage.run {run!r}")
    cset = state.condition_sets[condition_set] if condition_set else primary_set(state)
    fits = coordinator.all_fits()
    view = pooled(fits, cset["conditions"], min_games=1, weights=coordinator.pace_weights(cset["conditions"], weighting))
    parent = state.entrants[snapshots[0]]["lineage"].get("parent")

    def games(e):
        return min((fits[c].ratings[e].games if c in fits and e in fits[c].ratings else 0) for c in cset["conditions"])
    rows, best, streak, fired, pending = [], None, 0, False, False
    if parent and parent in view:
        best = dict(entrant=parent, rating=view[parent]["rating"])
    for e in snapshots:
        g = games(e)
        row = dict(entrant=e, step=state.entrants[e]["lineage"].get("step"), min_condition_games=g,
                   pooled=None if e not in view else dict(view[e], rating=round(view[e]["rating"], 1),
                                                          se=round(view[e]["se"], 1)))
        if pending or fired or g < min_games or e not in view:
            row["status"] = "pending" if not fired else "after stop"
            pending = pending or not fired
            rows.append(row)
            continue
        improved = best is None or view[e]["rating"] > best["rating"]
        row.update(status="improved" if improved else "not improved", best_before=best)
        if improved:
            best, streak = dict(entrant=e, rating=view[e]["rating"]), 0
        else:
            streak += 1
            fired = streak >= patience
        rows.append(row)
    return dict(run=run, condition_set=cset["name"], weighting=weighting,
                rule=f"stop after {patience} consecutive snapshots whose {weighting}-weighted pooled "
                f"rating does not exceed the best of the parent and earlier snapshots", parent=parent,
                min_games=min_games, step_every=step_every, snapshots=rows, fired=fired, selected=best,
                pending=pending)


def summary_text(report):
    lines = [f"pool {report['generated']}  source {report['source'][:12]}  model {report['model']}",
             "games {games} (pool {pool}, imported {imported}); last hour {games_last_hour}, last day "
             "{games_last_day}; {active_entrants}/{entrants} entrants active; {leases} leases".format(
                 **{"pool": 0, "imported": 0, **report["totals"]})]
    for s in report["condition_sets"]:
        lines.append(f"\n[{s['name']}] pace-weighted pooled over {len(s['conditions'])} conditions, anchor "
                     f"{s['anchor']} = {report['anchor_rating']:.0f}  (weights "
                     + ", ".join(f"{k} {v:g}" for k, v in s.get("weights", {}).items()) + ")")
        lines.append(f"  {'rating':>6}  {'95% CI':>11}  {'LOS':>4}  {'entrant':<44} {'games':>6}")
        for r in s["pooled"][:25]:
            lines.append(f"  {r['rating']:>6}  {r['ci95'][0]:>5}–{r['ci95'][1]:<5}  {los_text(r['los']):>4}  "
                         f"{r.get('label', r['entrant']):<44} {r['games']:>6}")
    if report["condition_sets"]:
        lines.append("  equal-weight: " + ", ".join(f"{r.get('label', r['entrant'])} {r['rating']}"
                                                  for r in report["condition_sets"][0]["pooled_equal"][:8]))
    mem = [m for m in report.get("memorization", []) if m.get("gap_points") is not None]
    if mem:
        lines.append("\n[memorization] seen (uniform) minus reserve, score points")
        for m in mem[:15]:
            lines.append(f"  {m['gap_points']:+6.1f} [{m['ci95_points'][0]:+.1f}, {m['ci95_points'][1]:+.1f}]"
                         f"{'  FLAG' if m['flagged'] else ''}  {m['entrant']}  ({m['reserve_seeds']}/{m['seen_seeds']} seeds)")
    lines.append("\n[eras] best per era (primary set)")
    for era in report["eras"]:
        b = era["best"]
        lines.append(f"  {era['era']:<16} " + (f"{b['rating']:>6}  {b['ci95'][0]:>5}–{b['ci95'][1]:<5}  {b['entrant']}"
                                                  if b else "unrated") + f"  ({era['entrants']} entrants)")
    if report["new_entrants"]:
        lines.append("\n[new / underplayed]")
        for e in report["new_entrants"][:20]:
            lines.append(f"  {e['id']:<34} {e['era']:<12} " + " ".join(f"{k}:{v}" for k, v in e["games"].items()))
    lines.append("\n[jobs]")
    for j in report["jobs"]:
        blocked = f"  ({j['blocked_items']} of {j['items']} pairings blocked)" if j["blocked_items"] else ""
        lines.append(f"  {j['status']:<9} p{j['priority']:<4} {j['id']:<36} {j['games']}/{j['target']} games{blocked}")
    lines.append("\n[intentions]")
    for i in report["intentions"]:
        wait = f" waiting on {', '.join(i['waiting_on'])}" if i["waiting_on"] else ""
        lines.append(f"  {i['view']:<8}{' OVERDUE' if i['overdue'] else ''} {i['id']:<34} {i['title']}{wait}")
    lines.append("\n[workers]")
    for w in report["workers"]:
        lines.append(f"  {w['worker']:<30} {str(w['numerics'])[:40]:<40} seen {w['seen_seconds_ago']}s  "
                     f"{w['games']} games  {w['games_per_hour']} g/h  fail {w['failures']}  {w['status']}")
    return "\n".join(lines)




PAGE = (Path(__file__).with_name("report.html")).read_text()
