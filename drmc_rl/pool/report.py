"""Pool reports: JSON export, CLI summary, the LAN web page and the stop-rule query."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from drmc_rl.pool import intentions as intent
from drmc_rl.pool.ratings import MODEL, pooled


def _entrant_view(state, e):
    r = state.entrants[e]
    return dict(id=e, name=r.get("name", e), era=r["era"], status=r["status"], tags=r.get("tags", []),
                lineage=r.get("lineage", {}), added_at=r.get("added_at"), notes=r.get("notes", ""))


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
                               ratings=[] if f is None else sorted((r.to_dict() for r in f.ratings.values()),
                                                                   key=lambda r: -r["rating"]),
                               unanchored=[] if f is None else f.unanchored))
    sets = []
    for cset in sorted(state.condition_sets.values(), key=lambda s: (not s["primary"], s["name"])):
        view = pooled(fits, cset["conditions"], min_games=1)
        sets.append(dict(name=cset["name"], anchor=cset["anchor"], primary=cset["primary"], weight=cset["weight"],
                         notes=cset["notes"], conditions=[state.conditions[k]["name"] for k in cset["conditions"]],
                         pooled=sorted(view.values(), key=lambda r: -r["rating"])))
    primary = primary_set(state)
    primary_view = pooled(fits, primary["conditions"], min_games=1) if primary else {}
    eras = {}
    for e in state.entrants:
        era = state.entrants[e]["era"]
        slot = eras.setdefault(era, dict(era=era, entrants=0, best=None))
        slot["entrants"] += 1
        if e in primary_view and (slot["best"] is None or primary_view[e]["rating"] > slot["best"]["rating"]):
            slot["best"] = dict(primary_view[e], name=state.entrants[e].get("name", e))
    min_rated = settings["min_rated_games"]
    new = []
    week = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for e in state.entrant_ids(("active", "benched")):
        per = {}
        for key in (primary["conditions"] if primary else []):
            f = fits.get(key)
            per[state.conditions[key]["name"]] = f.ratings[e].games if f and e in f.ratings else 0
        if (state.entrants[e].get("added_at") or "") >= week or any(g < min_rated for g in per.values()):
            new.append(dict(_entrant_view(state, e), games=per, rated=e in primary_view,
                            pooled=primary_view.get(e)))
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
        new_entrants=new, entrants=[_entrant_view(state, e) for e in sorted(state.entrants)], jobs=jobs,
        intentions=intent.roadmap(state, coordinator.capabilities),
        workers=workers, blocks=list(state.blocks.values()),
        fidelity=dict(mode=settings["fidelity"], min_agreement=settings["min_agreement"],
                      admitted=state.admitted, classes=coordinator.fidelity_stats,
                      pending_audits=len(coordinator.audits)))


def stop_rule(coordinator, *, run, condition_set=None, min_games=128, patience=2):
    """Pre-registered-style stop rule computed from pool ratings.

    Snapshots are the entrants whose ``lineage.run`` is ``run``, ordered by
    ``lineage.step``. The baseline is the first snapshot's ``lineage.parent``.
    A snapshot improves when its pooled rating over ``condition_set`` (the
    run's set, default the primary set) exceeds the best of the baseline and
    every earlier snapshot. The rule fires after ``patience`` consecutive
    non-improving snapshots; a snapshot with fewer than ``min_games`` games in
    any condition of the set is pending and stops evaluation there.
    """
    state = coordinator.state
    snapshots = sorted((e for e, r in state.entrants.items() if (r.get("lineage") or {}).get("run") == run),
                       key=lambda e: (state.entrants[e]["lineage"].get("step", 0), e))
    if not snapshots:
        raise KeyError(f"no entrants with lineage.run {run!r}")
    cset = state.condition_sets[condition_set] if condition_set else primary_set(state)
    fits = coordinator.all_fits()
    view = pooled(fits, cset["conditions"], min_games=1)
    parent = state.entrants[snapshots[0]]["lineage"].get("parent")

    def games(e):
        return min((fits[c].ratings[e].games if c in fits and e in fits[c].ratings else 0) for c in cset["conditions"])
    rows, best, streak, fired, pending = [], None, 0, False, False
    if parent and parent in view:
        best = dict(entrant=parent, rating=view[parent]["rating"])
    for e in snapshots:
        g = games(e)
        row = dict(entrant=e, step=state.entrants[e]["lineage"].get("step"), min_condition_games=g,
                   pooled=view.get(e))
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
    return dict(run=run, condition_set=cset["name"], rule=f"stop after {patience} consecutive snapshots whose pooled "
                f"rating does not exceed the best of the parent and earlier snapshots", parent=parent,
                min_games=min_games, snapshots=rows, fired=fired, selected=best, pending=pending)


def summary_text(report):
    lines = [f"pool {report['generated']}  source {report['source'][:12]}  model {report['model']}",
             "games {games} (pool {pool}, imported {imported}); last hour {games_last_hour}, last day "
             "{games_last_day}; {active_entrants}/{entrants} entrants active; {leases} leases".format(
                 **{"pool": 0, "imported": 0, **report["totals"]})]
    for s in report["condition_sets"]:
        lines.append(f"\n[{s['name']}] pooled over {len(s['conditions'])} conditions, anchor {s['anchor']} = "
                     f"{report['anchor_rating']:.0f}")
        for r in s["pooled"][:25]:
            lines.append(f"  {r['rating']:7.1f} ± {1.96 * r['se']:5.1f}  {r['entrant']:<34} {r['games']:>6} games")
    lines.append("\n[eras] best per era (primary set)")
    for era in report["eras"]:
        b = era["best"]
        lines.append(f"  {era['era']:<16} " + (f"{b['rating']:7.1f} ± {1.96 * b['se']:5.1f}  {b['entrant']}"
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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rating Pool</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a64;--line:#e3e0d8;--accent:#2f6f5e;--warn:#9a5b13;--bad:#a33a2a;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecebe6;--muted:#9c9a92;--line:#2e2d2a;--accent:#7cc4ad;--warn:#e0a95e;--bad:#ec8b7b;--card:#1e1e1c}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;margin:0;padding:16px}
main{max-width:1180px;margin:auto}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}td.n{text-align:right}.wrap{overflow-x:auto}.tag{font-size:12px;padding:1px 6px;border-radius:9px;border:1px solid var(--line)}
.overdue,.bad{color:var(--bad)}.warn{color:var(--warn)}.ok{color:var(--accent)}details{margin:6px 0}summary{cursor:pointer}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:4px;vertical-align:middle}
</style></head><body><main>
<h1>Rating pool</h1><div id="meta" class="muted">loading…</div>
<h2>Ratings by condition set (pooled mean over the set's conditions; anchor fixed)</h2><div id="sets"></div>
<h2>Era leaderboard</h2><div class="wrap" id="eras"></div>
<h2>New and underplayed entrants</h2><div class="wrap" id="new"></div>
<h2>Jobs</h2><div class="wrap" id="jobs"></div>
<h2>Roadmap (intentions)</h2><div class="wrap" id="intentions"></div>
<h2>Workers</h2><div class="wrap" id="workers"></div>
<h2>Per-condition ratings</h2><div id="conditions"></div>
<p class="muted">Raw data: <a href="report.json">report.json</a>. Ratings are recomputable from games.jsonl alone (docs/RATING_POOL.md).</p>
</main><script>
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const f1=x=>x==null?"":Number(x).toFixed(1);
function table(cols,rows){return `<table><tr>${cols.map(c=>`<th>${c[0]}</th>`).join("")}</tr>${rows.map(r=>`<tr>${cols.map(c=>`<td class="${c[2]||""}">${c[1](r)}</td>`).join("")}</tr>`).join("")}</table>`}
function ratingTable(rows){return table([["Rating",r=>f1(r.rating),"n"],["95% CI",r=>r.ci95?`${f1(r.ci95[0])} – ${f1(r.ci95[1])}`:"","n"],["Entrant",r=>esc(r.entrant)],["Games",r=>r.games,"n"]],rows)}
fetch("report.json").then(r=>r.json()).then(d=>{
 const t=d.totals;document.getElementById("meta").textContent=`${d.generated} · ${t.games.toLocaleString()} games (${(t.pool||0).toLocaleString()} pool, ${(t.imported||0).toLocaleString()} imported) · ${t.games_last_hour} last hour · ${t.games_last_day} last day · ${t.active_entrants}/${t.entrants} entrants active · ${t.leases} leases · anchor = ${d.anchor_rating}`;
 document.getElementById("sets").innerHTML=d.condition_sets.map(s=>`<details ${s.primary?"open":""}><summary><b>${esc(s.name)}</b> <span class="muted">anchor ${esc(s.anchor)} · ${s.conditions.length} conditions · weight ${s.weight}</span></summary><p class="muted">${esc(s.notes)}</p><div class="wrap">${ratingTable(s.pooled)}</div></details>`).join("");
 document.getElementById("eras").innerHTML=table([["Era",e=>esc(e.era)],["Best",e=>e.best?esc(e.best.entrant):"<span class=muted>unrated</span>"],["Rating",e=>e.best?f1(e.best.rating):"","n"],["±95%",e=>e.best?f1(1.96*e.best.se):"","n"],["Entrants",e=>e.entrants,"n"]],d.eras);
 document.getElementById("new").innerHTML=table([["Entrant",e=>esc(e.id)],["Era",e=>esc(e.era)],["Status",e=>esc(e.status)],["Games per primary condition",e=>Object.entries(e.games).map(([k,v])=>`${esc(k)}: ${v}`).join("<br>")],["Pooled",e=>e.pooled?f1(e.pooled.rating):"","n"],["Added",e=>esc((e.added_at||"").slice(0,10))]],d.new_entrants);
 document.getElementById("jobs").innerHTML=table([["Job",j=>esc(j.id)+(j.title?`<br><span class=muted>${esc(j.title)}</span>`:"")],["Status",j=>esc(j.status)],["Priority",j=>j.priority,"n"],["Progress",j=>`<span class="bar" style="width:${Math.round(80*j.games/Math.max(j.target,1))}px"></span> ${j.games}/${j.target}`+(j.blocked_items?` <span class=warn>${j.blocked_items} blocked</span>`:"")],["Deadline",j=>esc(j.deadline||"")],["Owner",j=>esc(j.owner)]],d.jobs);
 document.getElementById("intentions").innerHTML=table([["Intention",i=>`<b>${esc(i.id)}</b><br>${esc(i.title)}`],["Status",i=>`<span class="${i.overdue?"overdue":i.view=="blocked"?"warn":"ok"}">${esc(i.view)}${i.overdue?" · overdue":""}</span>`+(i.job?`<br><span class=muted>job ${esc(i.job.status)}</span>`:"")],["Waiting on",i=>i.waiting_on.map(esc).join("<br>")],["Hypothesis / decision rule",i=>`${esc(i.hypothesis)}<br><span class=muted>${esc(i.decision_rule)}</span>`],["Due",i=>esc(i.due||"")],["Owner",i=>esc(i.owner)]],d.intentions);
 document.getElementById("workers").innerHTML=table([["Worker",w=>esc(w.worker)],["Numerics",w=>esc(w.numerics)],["Admitted",w=>w.admitted===true?"<span class=ok>yes</span>":w.admitted==null?"pending":`<span class=bad>${esc(w.admitted)}</span>`],["Seen",w=>`${w.seen_seconds_ago}s ago`,"n"],["Games",w=>w.games,"n"],["Games/h",w=>w.games_per_hour??"","n"],["Failures",w=>w.failures,"n"],["Status",w=>esc(w.status)+(w.last_error?`<br><span class=bad>${esc(w.last_error)}</span>`:"")]],d.workers);
 document.getElementById("conditions").innerHTML=d.conditions.map(c=>`<details><summary><b>${esc(c.name)}</b> <span class="muted">${c.key} · ${c.games.toLocaleString()} games · anchor ${esc(c.anchor)}</span></summary><div class="wrap">${ratingTable(c.ratings)}</div>${c.unanchored.length?`<p class=warn>Unanchored (not connected to the anchor): ${c.unanchored.map(esc).join(", ")}</p>`:""}</details>`).join("");
}).catch(e=>{document.getElementById("meta").textContent="report unavailable: "+e});
</script></body></html>
"""

