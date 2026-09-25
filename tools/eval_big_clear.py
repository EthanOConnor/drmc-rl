"""Big-clear setup benchmark: can a candidate realise a human big-clear setup?

Bank rows (``tools.build_big_clear_bank write --split benchmark``) are held-out
human positions ``L`` placements before a T2+ clear (``kind`` mirror, both
bottles the setup) and matched negative controls (``kind`` control: the same
held-out side-game at a similar virus count where the human made no T1+ clear
in the next 20 placements). A group is one target clear: its setups at each
lookback and its control.

Each row is played ``solo`` (the candidate in both bottles, argmax, so the
sides stay in step and a row is one measurement of the candidate's own play)
under two pill conditions:

* ``replay``: the source game's seed and reserve index, so the pills are the
  human's and the target clear is reachable by the human's own line;
* ``fresh``: the group's evaluation-reserve seed (``python -m
  drmc_rl.program.seed_reserve allocate big-clear-benchmark-v1 N``), which
  asks whether the setup generalises to other pills.

``natural`` plan entries play the candidate against a reference (the
parent) from level-14 starts on the ``big-clear-natural-v1`` allocation,
side-swapped, with every placement of both sides resolved: the style panel for
big-clear rate and size in ordinary games (and the score, as a sanity check;
strength itself is measured by the rating pool).

Games stop at ``max_game_frames`` (the horizon is counted in pills). Per side
the journal keeps every clear of the first ``record_pills`` placements as
``(pill index, score, cells, rounds)``. ``summary.json`` reports, per match
and seed condition, the rate of realising a T1+ (and T2+) clear within the
horizon (setups: ``lookback + slack`` pills; controls: the same horizons),
the mean best score within the horizon, and the setup-minus-control lift,
with bootstrap intervals over groups.

  python -m tools.eval_big_clear --config CONFIG.json
  python -m tools.eval_big_clear --summarize OUT_DIR
  python -m tools.eval_big_clear --compare CANDIDATE_OUT BASELINE_OUT [--candidate-id X --baseline-id Y]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.eval import big_clear as bc
from tools.eval_stranded_edge import _bootstrap, bank_rows, build_runtime, start_overlay

RESULT_SCHEMA = "drmc-big-clear-benchmark-v1"
SLACK = 6
SEED_MODES = ("replay", "fresh")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def side_clears(moves: list, physical: int, record_pills: int) -> dict:
    mine = [m for m in moves if int(m["side"]) == physical][:record_pills]
    clears = []
    for k, m in enumerate(mine):
        try:
            f = bc.placement_features(bytes(m["board"]), m["pill"], int(m["placement"]["action"]))
        except ValueError:
            continue
        if f.rounds:
            clears.append([k + 1, round(f.score(), 1), f.cells, f.rounds])
    return dict(pills=len(mine), clears=clears)


def run(config: dict) -> None:
    from drmc_rl.program.seed_reserve import allocated_seeds

    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    bank = bank_rows(config["bank"])
    n = len(bank["boards"])
    groups = int(bank["group"].max()) + 1
    fresh = allocated_seeds(config["study"])
    if len(fresh) < groups:
        raise ValueError(f"study {config['study']!r} holds {len(fresh)} seeds for {groups} bank groups")
    record = int(config.get("record_pills", 24))
    identity = dict(schema=RESULT_SCHEMA, score=bc.SCHEMA, tiers=bc.TIERS, bank=str(config["bank"]),
                    bank_sha256=_sha(config["bank"]), study=config["study"],
                    seeds_sha256=hashlib.sha256(json.dumps(fresh[:groups]).encode()).hexdigest(),
                    record_pills=record,
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
            done.add((r["match"], r["seed_mode"], r["row"]))
    natural_seeds = allocated_seeds(config.get("natural_study", "big-clear-natural-v1")) if any(
        p.get("mode") == "natural" for p in config["schedule"]) else []
    runtime = build_runtime(config)
    pairs = int(config.get("pairs", 16))
    started = time.time()
    try:
        for plan in config["schedule"]:
            if plan.get("mode") == "natural":
                natural(config, runtime, plan, natural_seeds, done, journal, pairs, started)
                continue
            pace, a = plan["pace"], plan["candidate"]
            match_id = f"solo-{pace}-{a}"
            match = dict(id=match_id, a=a, b=a, games=0, pace=pace, level=int(plan.get("level", 14)))
            jobs = []
            for mode in plan.get("seed_modes", SEED_MODES):
                for i in (config.get("rows") or range(n)):
                    seed = int(bank["seed"][i]) if mode == "replay" else int(fresh[int(bank["group"][i])])
                    if seed <= 0 or (match_id, mode, i) in done:
                        continue
                    jobs.append((seed, mode, i))
            for start in range(0, len(jobs), pairs):
                chunk = jobs[start:start + pairs]
                batch, elapsed = runtime.rollout(
                    config, match, [(s, 0, k) for k, (s, _, _) in enumerate(chunk)], runtime.policy,
                    runtime.planner, runtime.preparer, policies=runtime.policies,
                    starts=[start_overlay(bank, i) for (_, _, i) in chunk])
                with journal.open("a") as sink:
                    for (seed, mode, i), (row, moves, _) in zip(chunk, batch):
                        sink.write(json.dumps(dict(
                            match=match_id, pace=pace, candidate=a, seed_mode=mode, seed=seed, row=i,
                            group=int(bank["group"][i]), kind=int(bank["kind"][i]),
                            lookback=int(bank["lookback"][i]), target_score=float(bank["target_score"][i]),
                            frames=row["frames"], reason=row["reason"],
                            side0=side_clears(moves, 0, record), side1=side_clears(moves, 1, record))) + "\n")
                print(json.dumps(dict(match=match_id, done=min(start + pairs, len(jobs)), of=len(jobs),
                                      batch_seconds=round(elapsed, 1), wall=round(time.time() - started))),
                      flush=True)
    finally:
        runtime.close()
    summarize(out)


def natural(config, runtime, plan, seeds, done, journal, pairs, started) -> None:
    pace, a, b = plan["pace"], plan["candidate"], plan["reference"]
    match_id = f"natural-{pace}-{a}-{b}"
    match = dict(id=match_id, a=a, b=b, games=0, pace=pace, level=int(plan.get("level", 14)))
    count = int(plan.get("seed_pairs", len(seeds)))
    jobs = [(int(seed), side, 2 * k + side) for k, seed in enumerate(seeds[:count]) for side in (0, 1)
            if (match_id, "natural", 2 * k + side) not in done]
    for start in range(0, len(jobs), 2 * pairs):
        chunk = jobs[start:start + 2 * pairs]
        batch, elapsed = runtime.rollout(dict(config, max_game_frames=int(plan.get("max_game_frames", 60000))),
                                         match, [(s, side, k) for k, (s, side, _) in enumerate(chunk)],
                                         runtime.policy, runtime.planner, runtime.preparer, policies=runtime.policies)
        with journal.open("a") as sink:
            for (seed, side, row_id), (row, moves, _) in zip(chunk, batch):
                sink.write(json.dumps(dict(
                    match=match_id, pace=pace, candidate=a, reference=b, seed_mode="natural", seed=seed,
                    row=row_id, candidate_side=side, winner=row["winner"], reason=row["reason"], frames=row["frames"],
                    candidate_clears=side_clears(moves, side, 10_000),
                    reference_clears=side_clears(moves, 1 - side, 10_000))) + "\n")
        print(json.dumps(dict(match=match_id, done=min(start + 2 * pairs, len(jobs)), of=len(jobs),
                              batch_seconds=round(elapsed, 1), wall=round(time.time() - started))), flush=True)


def natural_summary(items: list[dict]) -> dict:
    by_seed = defaultdict(list)
    for r in items:
        by_seed[r["seed"]].append(r)
    out = {}
    for who in ("candidate", "reference"):
        for name, bar in bc.TIERS:
            per = {s: [sum(c[1] >= bar for c in r[f"{who}_clears"]["clears"]) for r in v] for s, v in by_seed.items()}
            pills = {s: [r[f"{who}_clears"]["pills"] for r in v] for s, v in by_seed.items()}
            keys = sorted(per)
            point = 100.0 * sum(map(sum, per.values())) / max(1, sum(map(sum, pills.values())))
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(2000):
                pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
                draws.append(100.0 * sum(sum(per[k]) for k in pick) / max(1, sum(sum(pills[k]) for k in pick)))
            out[f"{who}_{name}_per_100"] = [round(point, 4), *[round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])]]
        scores = [c[1] for r in items for c in r[f"{who}_clears"]["clears"]]
        out[f"{who}_clear_mean_score"] = round(float(np.mean(scores)), 3) if scores else 0.0
        out[f"{who}_max_score"] = max(scores, default=0.0)
    out["candidate_score"] = _bootstrap({s: [1.0 if r["winner"] == "a" else 0.5 if r["winner"] == "draw" else 0.0
                                             for r in v]
                                         for s, v in by_seed.items()}, np.mean)
    out["games"] = len(items)
    return out


def realized(side: dict, horizon: int, bar: float) -> float:
    return float(any(p <= horizon and s >= bar for p, s, *_ in side["clears"]))


def best(side: dict, horizon: int) -> float:
    return max([s for p, s, *_ in side["clears"] if p <= horizon], default=0.0)


def _rows(out: Path) -> list[dict]:
    return [json.loads(l) for l in (Path(out) / "games.jsonl").read_text().splitlines()]


def measures(r: dict, horizon: int) -> dict:
    # Mirror rows: both sides are the candidate from identical inputs; average them.
    sides = (r["side0"], r["side1"])
    return dict(T1=np.mean([realized(s, horizon, bc.TIERS[0][1]) for s in sides]),
                T2=np.mean([realized(s, horizon, bc.TIERS[1][1]) for s in sides]),
                best=np.mean([best(s, horizon) for s in sides]))


def summarize(out: Path) -> dict:
    out = Path(out)
    rows = _rows(out)
    report = {}
    lookbacks = sorted({r["lookback"] for r in rows if r.get("kind") == 1})
    for match in sorted({r["match"] for r in rows if r["seed_mode"] == "natural"}):
        report[match] = natural_summary([r for r in rows if r["match"] == match])
    for match in sorted({r["match"] for r in rows if r["seed_mode"] != "natural"}):
        for mode in SEED_MODES:
            items = [r for r in rows if r["match"] == match and r["seed_mode"] == mode]
            if not items:
                continue
            for L in lookbacks:
                horizon = L + SLACK
                setups = defaultdict(list)
                controls = defaultdict(list)
                for r in items:
                    if r["kind"] == 1 and r["lookback"] == L:
                        setups[r["group"]].append(measures(r, horizon))
                    elif r["kind"] == 2:
                        controls[r["group"]].append(measures(r, horizon))
                key = f"{match}|{mode}|L{L}"
                entry = dict(horizon=horizon, groups=len(setups))
                for name in ("T1", "T2", "best"):
                    entry[f"setup_{name}"] = _bootstrap({g: [m[name] for m in v] for g, v in setups.items()}, np.mean)
                    entry[f"control_{name}"] = _bootstrap({g: [m[name] for m in v] for g, v in controls.items()},
                                                          np.mean)
                    both = {g: [np.mean([m[name] for m in setups[g]]) - np.mean([m[name] for m in controls[g]])]
                            for g in setups if g in controls}
                    entry[f"lift_{name}"] = _bootstrap(both, np.mean)
                report[key] = entry
    summary = dict(schema=RESULT_SCHEMA, slack=SLACK, tiers=bc.TIERS,
                   created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                   identity=json.loads((out / "identity.json").read_text()), results=report)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def compare(candidate: Path, baseline: Path, *, candidate_id=None, baseline_id=None) -> dict:
    """Paired per-row differences (candidate minus baseline) on setups, bootstrap over groups."""
    def load(out, id):
        rows = _rows(out)
        ids = {r["candidate"] for r in rows}
        if id is None and len(ids) != 1:
            raise ValueError(f"{out} holds rows for {sorted(ids)}; name one")
        id = id or ids.pop()
        return json.loads((Path(out) / "identity.json").read_text()), {
            (r["pace"], r["seed_mode"], r["row"]): r for r in rows if r["candidate"] == id and r["seed_mode"] != "natural"}
    ic, cand = load(candidate, candidate_id)
    ib, base = load(baseline, baseline_id)
    if (ic["bank_sha256"], ic["seeds_sha256"]) != (ib["bank_sha256"], ib["seeds_sha256"]):
        raise ValueError("candidate and baseline used different banks or seed allocations")
    report = {}
    keys = sorted(set(cand) & set(base))
    for pace in sorted({k[0] for k in keys}) + ["pooled"]:
        for mode in SEED_MODES:
            diffs = defaultdict(lambda: defaultdict(list))
            for key in keys:
                p, m, _ = key
                if m != mode or (pace != "pooled" and p != pace):
                    continue
                c, b = cand[key], base[key]
                horizon = c["lookback"] + SLACK
                mc, mb = measures(c, horizon), measures(b, horizon)
                kind = {1: "setup", 2: "control"}[c["kind"]]
                for name in ("T1", "T2", "best"):
                    diffs[f"{kind}_{name}"][c["group"]].append(mc[name] - mb[name])
            if diffs:
                report[f"{pace}|{mode}"] = {k: _bootstrap(v, np.mean) for k, v in diffs.items()} | {
                    "rows": sum(len(v) for v in diffs["setup_T1"].values())}
    return dict(candidate=str(candidate), baseline=str(baseline), paired=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("CANDIDATE_OUT", "BASELINE_OUT"))
    parser.add_argument("--candidate-id")
    parser.add_argument("--baseline-id")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare, candidate_id=args.candidate_id, baseline_id=args.baseline_id),
                         indent=1))
    elif args.summarize:
        print(json.dumps(summarize(args.summarize)["results"], indent=1))
    else:
        run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
