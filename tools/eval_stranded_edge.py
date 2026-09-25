"""Stranded edge-virus benchmark: play fixed endgame start states and score the resolution.

Every bank row puts the same stranded bottle and pill in both bottles; the
row's evaluation-reserve seed fixes the pill stream that follows. Two modes:

- ``solo``: the candidate on both sides (a mirror race). Argmax play from
  identical inputs keeps the sides in step, so each row is one clean
  measurement of the candidate's own resolution, uncensored by an opponent.
- ``race``: candidate against a fixed reference, side-swapped, from the same
  identical bottles. Win rate is the vs-setting check; the resolution metrics
  are censored when the reference finishes first.

Per candidate side it records pills and frames until the target virus is gone,
support-destroying clears (the build zone under/beside the virus got lower
while it stood), clears, and whether the side cleared the bottle. Rows are
journaled so a run resumes; ``summary.json`` pools rows with whole-position
bootstrap intervals.

Seeds come only from a recorded allocation (``python -m
drmc_rl.program.seed_reserve allocate STUDY COUNT``), one per bank row.

  python -m tools.eval_stranded_edge --config CONFIG.json
  python -m tools.eval_stranded_edge --summarize OUT_DIR
  python -m tools.eval_stranded_edge --compare CANDIDATE_OUT BASELINE_OUT [--baseline-id champion]
  python -m tools.eval_stranded_edge --reference-gap OUT REFERENCE.jsonl [--whole-round]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.eval.stranded_edge import SCHEMA, follow

RESULT_SCHEMA = "drmc-stranded-edge-benchmark-v1"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank_rows(path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def start_overlay(bank, i: int) -> dict:
    return {
        "checkpoint_enabled": True,
        "checkpoint_board": bank["boards"][i].reshape(2, 128),
        "checkpoint_falling_colors": bank["falling"][i],
        "checkpoint_preview_colors": bank["preview"][i],
        "checkpoint_pill_counter": tuple(int(v) for v in bank["pill_counter"][i]),
        "checkpoint_speed_ups": tuple(int(v) for v in bank["speed_ups"][i]),
    }


def side_metrics(bank, i: int, row: dict, moves: list, physical: int, winner_is_side: bool) -> dict:
    mine = [m for m in moves if int(m["side"]) == physical]
    target = tuple(int(v) for v in bank["target"][i])
    # A "clear" draw means both bottles emptied on the same frame.
    cleared_out = (winner_is_side or row["winner"] == "draw") and row["reason"] == "clear"
    episode = follow([m["board"] for m in mine], [m["frame"] for m in mine], target,
                     cleared_out=cleared_out, end_frame=row["frames"])
    return dict(cleared=episode.cleared, pills=episode.pills, frames=episode.frames,
                support_destroying=episode.support_destroying, clears=episode.clears,
                max_gap=episode.max_gap, censored=episode.censored,
                round_pills=len(mine) if cleared_out else None, placements=len(mine),
                won=winner_is_side, draw=row["winner"] == "draw", reason=row["reason"])


def build_runtime(config):
    from tools.trainer_planning_arena import ArenaRuntime, bind_execution_profiles, variant_policy

    bind_execution_profiles(config)
    runtime = ArenaRuntime(config)
    runtime.policies = runtime.policies if runtime.policies is not None else {}
    for id, params in config["variants"].items():
        runtime.policies[id] = variant_policy(config, params, runtime.policy)
    return runtime


def run(config: dict) -> None:
    from drmc_rl.program.seed_reserve import allocated_seeds

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    bank = bank_rows(config["bank"])
    n = len(bank["boards"])
    rows_wanted = config.get("rows") or list(range(n))
    seeds = allocated_seeds(config["study"])
    groups = int(bank["group"].max()) + 1
    if len(seeds) < groups:
        raise ValueError(f"study {config['study']!r} holds {len(seeds)} seeds for {groups} bank groups")
    identity = dict(schema=RESULT_SCHEMA, detector=SCHEMA, bank=str(config["bank"]), bank_sha256=_sha(config["bank"]),
                    study=config["study"], seeds_sha256=hashlib.sha256(json.dumps(seeds[:groups]).encode()).hexdigest(),
                    variants={k: {**v, "sha256": _sha(v["checkpoint"])} for k, v in config["variants"].items()})
    ident_path = out / "identity.json"
    if ident_path.exists():
        if json.loads(ident_path.read_text()) != identity:
            raise ValueError("output holds a different bank, seed allocation or checkpoints; use a fresh output")
    else:
        ident_path.write_text(json.dumps(identity, indent=1))
    journal = out / "games.jsonl"
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            done.add((r["match"], r["row"], r["side"]))
    runtime = build_runtime(config)
    pairs = int(config.get("pairs", 16))
    started = time.time()
    try:
        for plan in config["schedule"]:
            mode, pace, a = plan["mode"], plan["pace"], plan["candidate"]
            b = a if mode == "solo" else plan["reference"]
            match_id = f"{mode}-{pace}-{a}-{b}"
            match = dict(id=match_id, a=a, b=b, games=0, pace=pace, level=int(plan.get("level", 14)))
            jobs = []
            for i in rows_wanted:
                for side in ((0,) if mode == "solo" else (0, 1)):
                    if (match_id, int(i), side) not in done:
                        jobs.append((int(seeds[int(bank["group"][i])]), side, int(i)))
            for start in range(0, len(jobs), pairs):
                chunk = jobs[start:start + pairs]
                batch, elapsed = runtime.rollout(
                    config, match, [(s, side, k) for k, (s, side, _) in enumerate(chunk)], runtime.policy,
                    runtime.planner, runtime.preparer, policies=runtime.policies,
                    starts=[start_overlay(bank, i) for (_, _, i) in chunk])
                with journal.open("a") as sink:
                    for (seed, side, i), (row, moves, _) in zip(chunk, batch):
                        a_won = row["winner"] == "a"
                        entry = dict(match=match_id, mode=mode, pace=pace, candidate=a, reference=b, row=i,
                                     seed=seed, side=side, stratum=int(bank["stratum"][i]),
                                     source=str(bank["source"][i]), frames_total=row["frames"],
                                     winner=row["winner"], reason=row["reason"],
                                     candidate_side=side_metrics(bank, i, row, moves, side, a_won))
                        if mode == "solo":
                            entry["mirror_side"] = side_metrics(bank, i, row, moves, 1 - side,
                                                                row["winner"] == "b")
                        sink.write(json.dumps(entry) + "\n")
                        if config.get("keep_traces"):
                            trace = out / "moves" / f"{match_id}-{i:04d}-{side}.json.gz"
                            trace.parent.mkdir(exist_ok=True)
                            with gzip.open(trace, "wt") as t:
                                json.dump(dict(game=row, moves=moves), t)
                print(json.dumps(dict(match=match_id, done=min(start + pairs, len(jobs)), of=len(jobs),
                                      batch_seconds=round(elapsed, 1), wall=round(time.time() - started))),
                      flush=True)
    finally:
        runtime.close()
    summarize(out)


def _bootstrap(values_by_row: dict[int, list[float]], stat, draws: int = 2000, seed: int = 0):
    keys = sorted(values_by_row)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    point = stat([v for k in keys for v in values_by_row[k]])
    samples = []
    for _ in range(draws):
        pick = rng.choice(len(keys), len(keys))
        vals = [v for j in pick for v in values_by_row[keys[j]]]
        samples.append(stat(vals))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [round(float(point), 4), round(float(lo), 4), round(float(hi), 4)]


def restricted_pills(m: dict, horizon: int) -> float:
    """Pills to clear the target capped at ``horizon``; an uncleared target counts as the horizon."""
    return float(min(m["pills"], horizon)) if m["cleared"] else float(horizon)


def summarize(out: Path, horizon: int = 40) -> dict:
    out = Path(out)
    rows = [json.loads(l) for l in (out / "games.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for r in rows:
        groups[r["match"]].append(r)
        groups[r["match"] + f"|stratum{r['stratum']}"].append(r)
    report = {}
    for key, items in sorted(groups.items()):
        by_row = defaultdict(list)
        for r in items:
            by_row[r["row"]].append(r["candidate_side"])
        def metric(fn):
            return {k: [fn(m) for m in v] for k, v in by_row.items()}
        cleared = metric(lambda m: float(m["cleared"]))
        pills = {k: [m["pills"] for m in v if m["cleared"]] for k, v in by_row.items()}
        frames = {k: [m["frames"] for m in v if m["cleared"] and m["frames"] is not None] for k, v in by_row.items()}
        sd = metric(lambda m: float(m["support_destroying"]))
        clears = metric(lambda m: float(m["clears"]))
        report[key] = dict(
            games=len(items), rows=len(by_row),
            target_cleared=_bootstrap(cleared, np.mean),
            pills_median=_bootstrap({k: v for k, v in pills.items() if v}, np.median),
            pills_mean=_bootstrap({k: v for k, v in pills.items() if v}, np.mean),
            pills_restricted_mean=_bootstrap(metric(lambda m: restricted_pills(m, horizon)), np.mean),
            frames_median=_bootstrap({k: v for k, v in frames.items() if v}, np.median),
            support_destroying_per_position=_bootstrap(sd, np.mean),
            support_destroying_rate=_bootstrap(
                {k: [sd[k][j] > 0 for j in range(len(sd[k]))] for k in sd}, np.mean),
            support_destroying_per_clear=round(sum(map(sum, sd.values())) / max(1.0, sum(map(sum, clears.values()))), 4),
            round_cleared=_bootstrap(metric(lambda m: float(m["round_pills"] is not None)), np.mean),
            score=_bootstrap({k: [1.0 if r["candidate_side"]["won"] else 0.5 if r["winner"] == "draw" else 0.0
                                  for r in items if r["row"] == k] for k in by_row}, np.mean),
        )
        if items[0]["mode"] == "solo":
            agree = [r["candidate_side"]["pills"] == r["mirror_side"]["pills"] for r in items]
            report[key]["mirror_side_agreement"] = round(float(np.mean(agree)), 4)
    report.update(handicap(rows, bank_rows(json.loads((out / "identity.json").read_text())["bank"]), horizon))
    summary = dict(schema=RESULT_SCHEMA, horizon=horizon, created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                   identity=json.loads((out / "identity.json").read_text()), results=report)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def handicap(rows: list[dict], bank, horizon: int) -> dict:
    """Stranded minus grounded-twin cost on the same group, mirror and pill stream (paired control)."""
    kind, group, mirror = bank["kind"], bank["group"], bank["mirror"]
    out = {}
    for match in sorted({r["match"] for r in rows}):
        cells = defaultdict(dict)
        for r in rows:
            if r["match"] == match and kind[r["row"]] in (0, 2):
                cells[(int(group[r["row"]]), int(mirror[r["row"]]))].setdefault(int(kind[r["row"]]), []).append(
                    r["candidate_side"])
        by_group = defaultdict(lambda: defaultdict(list))
        for (g, _), pair in cells.items():
            if 0 in pair and 2 in pair:
                for name, fn in (("pills", lambda m: restricted_pills(m, horizon)),
                                 ("cleared", lambda m: float(m["cleared"])),
                                 ("support_destroying", lambda m: float(m["support_destroying"]))):
                    by_group[name][g].append(np.mean([fn(m) for m in pair[0]]) - np.mean([fn(m) for m in pair[2]]))
        out[f"{match}|handicap_vs_grounded"] = {
            name: _bootstrap(values, np.mean) for name, values in by_group.items()} | {
            "pairs": sum(len(v) for v in by_group["pills"].values())}
    return out


def compare(candidate: Path, baseline: Path, *, candidate_id: str | None = None, baseline_id: str | None = None,
            strata=(0, 1), horizon: int = 40) -> dict:
    """Paired per-row differences (candidate minus baseline) for solo matches at each pace.

    Rows are paired by bank row and pace; the bootstrap resamples bank groups,
    so a position, its mirror and its twins move together.
    """
    def load(out, id):
        ident = json.loads((Path(out) / "identity.json").read_text())
        rows = [json.loads(l) for l in (Path(out) / "games.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["mode"] == "solo"]
        ids = {r["candidate"] for r in rows}
        if id is None and len(ids) != 1:
            raise ValueError(f"{out} holds solo rows for {sorted(ids)}; name one")
        id = id or ids.pop()
        return ident, {(r["pace"], r["row"]): r["candidate_side"] for r in rows if r["candidate"] == id}
    ident_c, cand = load(candidate, candidate_id)
    ident_b, base = load(baseline, baseline_id)
    if (ident_c["bank_sha256"], ident_c["seeds_sha256"]) != (ident_b["bank_sha256"], ident_b["seeds_sha256"]):
        raise ValueError("candidate and baseline used different banks or seed allocations")
    bank = bank_rows(ident_c["bank"])
    report = {}
    paces = sorted({p for p, _ in cand} & {p for p, _ in base})
    for pace in [*paces, "pooled"]:
        diffs = defaultdict(lambda: defaultdict(list))
        for (p, i), m in cand.items():
            if (p != pace and pace != "pooled") or (p, i) not in base or int(bank["stratum"][i]) not in strata:
                continue
            b, g = base[(p, i)], int(bank["group"][i])
            diffs["pills_restricted"][g].append(restricted_pills(m, horizon) - restricted_pills(b, horizon))
            diffs["cleared"][g].append(float(m["cleared"]) - float(b["cleared"]))
            diffs["support_destroying"][g].append(float(m["support_destroying"]) - float(b["support_destroying"]))
        report[pace] = {k: _bootstrap(v, np.mean) for k, v in diffs.items()}
        report[pace]["rows"] = sum(len(v) for v in diffs["cleared"].values())
    return dict(candidate=str(candidate), baseline=str(baseline), strata=list(strata), horizon=horizon,
                paired=report)


def reference_gap(out: Path, reference: Path, *, candidate_id: str | None = None, horizon: int = 40,
                  whole_round: bool = False) -> dict:
    """Candidate solo pills minus a search reference (``tools.oracle_stranded_edge``) on the same rows.

    ``whole_round`` compares pills to empty the bottle (reference ``preview-round``
    rows) instead of pills to clear the target. Both sides are capped at
    ``horizon``; a failure counts as the horizon. Bootstrap by bank group.
    """
    ident = json.loads((Path(out) / "identity.json").read_text())
    bank = bank_rows(ident["bank"])
    ref = {r["row"]: r for r in map(json.loads, Path(reference).read_text().splitlines())}
    rows = [r for r in map(json.loads, (Path(out) / "games.jsonl").read_text().splitlines()) if r["mode"] == "solo"]
    ids = {r["candidate"] for r in rows}
    candidate_id = candidate_id or (ids.pop() if len(ids) == 1 else None)
    if candidate_id is None:
        raise ValueError(f"{out} holds solo rows for {sorted(ids)}; name one")
    names = {0: "stranded", 1: "stranded", 2: "pillar", 3: "pillar", 4: "grounded", 5: "grounded"}

    def agent(m):
        value = m["round_pills"] if whole_round else (m["pills"] if m["cleared"] else None)
        return float(horizon) if value is None else float(min(value, horizon))

    report = {}
    for pace in sorted({r["pace"] for r in rows if r["candidate"] == candidate_id}):
        cells = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["candidate"] != candidate_id or r["pace"] != pace or r["row"] not in ref:
                continue
            value = ref[r["row"]]["pills"]
            reference_pills = float(horizon) if value is None else float(min(value, horizon))
            cells[names[r["stratum"]]][int(bank["group"][r["row"]])].append((agent(r["candidate_side"]),
                                                                              reference_pills))
        report[pace] = {}
        for stratum, groups in cells.items():
            pairs = [p for v in groups.values() for p in v]
            report[pace][stratum] = dict(
                rows=len(pairs), agent=round(float(np.mean([a for a, _ in pairs])), 3),
                reference=round(float(np.mean([b for _, b in pairs])), 3),
                ratio=round(float(np.mean([a for a, _ in pairs]) / max(1e-9, np.mean([b for _, b in pairs]))), 3),
                excess=_bootstrap({g: [a - b for a, b in v] for g, v in groups.items()}, np.mean),
                reference_failed=round(float(np.mean([b >= horizon for _, b in pairs])), 3))
    return dict(out=str(out), reference=str(reference), candidate=candidate_id, horizon=horizon,
                whole_round=whole_round, gap=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("CANDIDATE_OUT", "BASELINE_OUT"))
    parser.add_argument("--reference-gap", type=Path, nargs=2, metavar=("OUT", "REFERENCE_JSONL"))
    parser.add_argument("--whole-round", action="store_true")
    parser.add_argument("--candidate-id")
    parser.add_argument("--baseline-id")
    args = parser.parse_args()
    if args.reference_gap:
        print(json.dumps(reference_gap(*args.reference_gap, candidate_id=args.candidate_id,
                                       whole_round=args.whole_round), indent=1))
    elif args.compare:
        print(json.dumps(compare(*args.compare, candidate_id=args.candidate_id, baseline_id=args.baseline_id),
                         indent=1))
    elif args.summarize:
        print(json.dumps(summarize(args.summarize)["results"], indent=1))
    else:
        run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
