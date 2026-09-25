"""Intentions: planned experiments recorded before their entrants or engines exist.

An intention names a hypothesis, the entrants it will compare (ids or glob
patterns, e.g. ``stronger3-armC-*``), its conditions, metrics, decision rule
and dependencies. Dependencies are typed strings:

    intention:<id>      another intention must be done
    capability:<cap>    the pool's source tree must execute it (e.g. movement:human)
    entrant:<pattern>   at least one matching entrant must be registered
    external:<text>     outside the pool (data, hardware, a decision); met once
                        listed in the intention's ``resolved``

When every dependency is met and every planned entrant pattern matches, the
coordinator submits the intention's ``job`` (a job spec without an id; the job
takes the intention's id) and moves it to ``running``. ``done`` and
``dropped`` are set by people, with a note; the report shows planned, blocked,
running (and whether its job finished) and overdue items as a roadmap.
"""
from __future__ import annotations

from datetime import date

from drmc_rl.pool.conditions import check_name

STATUSES = ("planned", "blocked", "running", "done", "dropped")
DEPENDENCY_KINDS = ("intention", "capability", "entrant", "external")


def validate_intention(record):
    check_name(record["id"])
    if record.get("status") not in STATUSES:
        raise ValueError(f"intention status must be one of {STATUSES}")
    if not record.get("title"):
        raise ValueError("intention needs a title")
    for dep in record.get("depends", []):
        kind = str(dep).split(":", 1)[0]
        if kind not in DEPENDENCY_KINDS or ":" not in str(dep):
            raise ValueError(f"dependency {dep!r} must be one of {DEPENDENCY_KINDS} as kind:value")
    if record.get("due"):
        date.fromisoformat(record["due"])
    job = record.get("job")
    if job is not None and ("id" in job or not isinstance(job, dict)):
        raise ValueError("an intention's job spec takes the intention's id; omit job.id")


def unmet(state, record, capabilities):
    """Dependencies and planned entrants that are not yet satisfied."""
    missing = []
    resolved = set(record.get("resolved", []))
    for dep in record.get("depends", []):
        kind, value = dep.split(":", 1)
        if dep in resolved:
            continue
        if kind == "intention":
            other = state.intentions.get(value)
            if other is None or other["status"] != "done":
                missing.append(dep)
        elif kind == "capability":
            if value not in capabilities:
                missing.append(dep)
        elif kind == "entrant":
            if not state.resolve_entrants([value]):
                missing.append(dep)
        else:
            missing.append(dep)
    for pattern in record.get("entrants", []):
        if not state.resolve_entrants([pattern]):
            missing.append(f"entrant:{pattern}")
    for ref in record.get("conditions", []):
        try:
            state.expand_conditions([ref])
        except KeyError:
            missing.append(f"condition:{ref}")
    return list(dict.fromkeys(missing))


def roadmap(state, capabilities, today=None):
    today = today or date.today()
    rows = []
    for record in sorted(state.intentions.values(), key=lambda r: (STATUSES.index(r["status"]), r.get("due") or "9999",
                                                                   r["id"])):
        missing = unmet(state, record, capabilities) if record["status"] in ("planned", "blocked") else []
        job = state.jobs.get(record["id"])
        overdue = bool(record.get("due")) and record["status"] in ("planned", "blocked", "running") \
            and date.fromisoformat(record["due"]) < today
        view = record["status"]
        if record["status"] == "planned" and missing:
            view = "blocked"
        rows.append(dict(id=record["id"], title=record["title"], status=record["status"], view=view,
                         owner=record.get("owner", ""), due=record.get("due"), overdue=overdue,
                         waiting_on=missing, hypothesis=record.get("hypothesis", ""),
                         decision_rule=record.get("decision_rule", ""), metrics=record.get("metrics", []),
                         entrants=record.get("entrants", []), conditions=record.get("conditions", []),
                         job=None if job is None else dict(status=job["status"]),
                         notes=record.get("notes", "")))
    return rows


def ready_to_start(state, capabilities):
    """Intentions whose job can be submitted now."""
    for record in sorted(state.intentions.values(), key=lambda r: r["id"]):
        if record["status"] in ("planned", "blocked") and record.get("job") is not None \
                and record["id"] not in state.jobs and not unmet(state, record, capabilities):
            yield record
