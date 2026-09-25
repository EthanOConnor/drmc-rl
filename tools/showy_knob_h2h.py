"""Head-to-head measurement of the showy-setup knob: champion+knob(lambda) vs the unbiased champion.

Mirrors one rating-pool match (``drmc_rl.pool.worker.Runtimes.play``): events
backend, the pool's DEFAULT_RUNTIME, level 14, speed Hi, decision delay 4, both
sides of every seed. Entrant ``knob`` is the anchor checkpoint with
``showy_lambda``/``showy_model`` variant params (see ``variant_policy``);
entrant ``base`` is the bare anchor. Writes one small JSON summary (no traces).

    nice -n 10 python -m tools.showy_knob_h2h --tag "tools.trainer_planning_arena " \
        --model showy.json --lambdas 0.5 1 2 --paces normal top_humans --pairs 100 --out out.json

``--tag`` is ignored; it only makes this process visible to the pool workers'
foreign-arena check so one worker slot yields while it runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ANCHOR = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-retention-mixed-v2/core-final-inference.pt"
LIBS = "/Users/ethan/dev/drmario/drmc-rl/runs/review-20260909/controller-arena-0c76c0e-source/native-libraries"
STUB = dict(schema="drmc-showy-knob-v1", features=["h3s_sum", "surf_run3_sum", "threats"],
            mean=[0, 0, 0], scale=[1, 1, 1], coef=[0.5, 0.5, 0.2], intercept=-4.0)
STYLE_KEYS = ("placements", "clears", "lines", "combos", "chains", "t1", "t2", "t3", "horizontal", "horizontal_combo")


def side_counters(moves, physical):
    """Per-side counters from one game's move journal (same definitions as ``drmc_rl.pool.style``)."""
    from drmc_rl.eval import big_clear as bc
    t1, t2, t3 = (bar for _, bar in bc.TIERS)
    out = dict.fromkeys(STYLE_KEYS, 0)
    for move in moves:
        if int(move["side"]) != physical:
            continue
        out["placements"] += 1
        try:
            _, f, detail = bc.resolve(bc.place(bytes(move["board"]), move["pill"], int(move["placement"]["action"])))
        except (ValueError, KeyError, TypeError):
            continue
        if not f.rounds:
            continue
        score = f.score()
        out["clears"] += 1
        out["lines"] += f.lines
        combo = f.lines >= 2 or f.rounds >= 2
        out["combos"] += int(combo)
        out["chains"] += int(f.rounds >= 2)
        out["t1"] += int(score >= t1)
        out["t2"] += int(score >= t2)
        out["t3"] += int(score >= t3)
        horizontal = any(o == 0 for found, _, _ in detail for o, *_ in found)
        out["horizontal"] += int(horizontal)
        out["horizontal_combo"] += int(horizontal and f.lines >= 2)
        if score > out.get("best_score", -1):
            out["best_score"], out["best"] = score, dict(score=score, cells=f.cells, rounds=f.rounds, lines=f.lines,
                                                         viruses=f.viruses, horizontal=bool(horizontal))
    return out


def rates(total):
    p = max(total["placements"], 1)
    per = lambda k: round(100.0 * total[k] / p, 3)  # noqa: E731
    return dict(placements=total["placements"], counts={k: total[k] for k in ("clears", "t1", "t2", "t3", "horizontal",
                                                                             "horizontal_combo")},
                best=total.get("best"), clears=per("clears"), combos=per("combos"), chains=per("chains"),
                t1_plus=per("t1"), t2_plus=per("t2"), t3_plus=per("t3"),
                horizontal=per("horizontal"), horizontal_combo=per("horizontal_combo"),
                horizontal_share_of_clears=round(total["horizontal"] / total["clears"], 4) if total["clears"] else None,
                lines_per_clear=round(total["lines"] / total["clears"], 3) if total["clears"] else None)


def outcome(scores):
    """Win rate of the knob side (draws 0.5) with a normal-approx 95% CI and an Elo estimate."""
    scored = [s for s in scores if s is not None]
    n = len(scored)
    if not n:
        return dict(games=0)
    p = sum(scored) / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    elo = lambda q: None if q <= 0 or q >= 1 else round(-400 * math.log10(1 / q - 1), 1)  # noqa: E731
    lo, hi = max(p - 1.96 * se, 1e-4), min(p + 1.96 * se, 1 - 1e-4)
    return dict(games=n, timeouts=len(scores) - n, knob_score=round(p, 4), ci95=[round(lo, 4), round(hi, 4)],
                elo=elo(p), elo_ci95=[elo(lo), elo(hi)],
                wins=sum(s == 1 for s in scored), losses=sum(s == 0 for s in scored),
                draws=sum(s == 0.5 for s in scored))


def pick_seeds(count, start):
    """Deterministic seeds outside the evaluation reserve (s and s^0x100 are the same game)."""
    from drmc_rl.program.seed_reserve import load_reserve
    blocked = set(load_reserve().blocked)
    seeds, s, seen = [], start, set()
    while len(seeds) < count:
        s = s % 65535 + 1
        key = min(s, s ^ 0x100)
        if s not in blocked and (s ^ 0x100) not in blocked and key not in seen:
            seen.add(key)
            seeds.append(s)
    return seeds


class Runner:
    def __init__(self, args):
        import torch
        from drmc_rl.pool.coordinator import DEFAULT_RUNTIME
        from tools.trainer_planning_arena import ArenaRuntime
        os.environ.setdefault("DRMARIO_REACH_LIB", str(Path(args.reach_library).resolve()))
        self.args = args
        config = dict(DEFAULT_RUNTIME, rollout_backend="events", checkpoint=args.anchor, device=args.device,
                      threads=args.threads, native_library=args.native_library)
        if args.planner_workers:
            config["planner_workers"] = args.planner_workers
        config["variants"] = {"_anchor": dict(name="anchor", delay=4, checkpoint=args.anchor)}
        torch.manual_seed(0)
        self.runtime = ArenaRuntime(config)

    def variants(self, lam, model):
        base = dict(name="base", delay=4, checkpoint=self.args.base_checkpoint or self.args.checkpoint)
        knob = dict(base, name=f"knob{lam}", checkpoint=self.args.checkpoint)
        if lam is not None:
            knob.update(showy_lambda=float(lam), showy_model=model, showy_tier_bar=self.args.tier_bar)
            if self.args.extra_model and lam:
                from drmc_rl.style.showy_knob import ShowyModel
                knob["showy_terms"] = [dict(model=ShowyModel.load(self.args.extra_model).spec,
                                            **{"lambda": self.args.extra_lambda})]
        return dict(knob=knob, base=base)

    def play(self, variants, pace, seeds, *, policies=None):
        from drmc_rl.pool.conditions import pace_profile
        from tools.trainer_planning_arena import bind_execution_profiles, variant_policy
        rt = self.runtime
        rt.config["variants"] = variants
        rt.policies = policies or {e: variant_policy(rt.config, p, rt.policy) for e, p in variants.items()}
        jobs = [(seed, side, 2 * i + side) for i, seed in enumerate(seeds) for side in (0, 1)]
        match = dict(id=f"showy-{pace}", a="knob", b="base", games=len(jobs), level=14, pace=pace,
                     execution_profile=pace_profile(pace))
        check = dict(schedule=[match])
        bind_execution_profiles(check)
        return rt.play(match, jobs)


def decisions(batch):
    return sum(r["a_stats"].get("decisions", 0) + r["b_stats"].get("decisions", 0) for r, _, _ in batch)


def identity_check(runner, model, pace, seeds):
    """lambda=0 via the variant-param path must play byte-identical games to the bare anchor."""
    bare, _ = runner.play(runner.variants(None, model), pace, seeds)
    zero, _ = runner.play(runner.variants(0.0, model), pace, seeds)
    from drmc_rl.style.showy_knob import ShowyModel, ShowyPolicy
    wrapped0 = {"knob": ShowyPolicy(runner.runtime.policy, ShowyModel.load(model), 0.0), "base": runner.runtime.policy}
    forced, _ = runner.play(runner.variants(0.0, model), pace, seeds, policies=wrapped0)
    sig = lambda b: [(r["seed"], r["side"], r["score"], r["frames"], [m["placement"]["action"] for m in mv])  # noqa: E731
                     for r, mv, _ in b]
    return dict(param_lambda0_identical=sig(bare) == sig(zero), wrapped_lambda0_identical=sig(bare) == sig(forced),
                games=len(bare), placements=sum(len(mv) for _, mv, _ in bare))


def verify_inputs(runner, model, pace, seeds, lam=1.0):
    """Compare the wrapper's reconstructed bottle/pill with the engine's true decision state."""
    from drmc_rl.game.afterstate import planes_to_fields
    from drmc_rl.style.showy_knob import ShowyModel, ShowyPolicy
    seen = []

    class Probe(ShowyPolicy):
        def score(self, obs, infos):
            fields = planes_to_fields(np.asarray(obs)[:, :8])
            for i, info in enumerate(infos):
                seen.append((bytes(fields[i]), tuple(int(c) for c in np.asarray(info["next_pill_colors"]).reshape(-1))))
            return super().score(obs, infos)

    policies = {"knob": Probe(runner.runtime.policy, ShowyModel.load(model), lam), "base": runner.runtime.policy}
    batch, _ = runner.play(runner.variants(lam, model), pace, seeds, policies=policies)
    truth = set()
    for row, moves, _ in batch:
        a = int(row["side"])
        for m in moves:
            if int(m["side"]) == a:
                truth.add((bytes(m["board"]), tuple(int(c) for c in m["pill"])))
    boards = {b for b, _ in truth}
    return dict(probed=len(seen), exact=sum(s in truth for s in seen), board_match=sum(b in boards for b, _ in seen),
                knob_decisions=len(truth))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="ignored (process-visibility marker for pool workers)")
    ap.add_argument("--checkpoint", default=ANCHOR, help="knob side's checkpoint")
    ap.add_argument("--base-checkpoint", help="unbiased side's checkpoint (default: --checkpoint)")
    ap.add_argument("--anchor", default=ANCHOR, help="runtime anchor (the pool's; other checkpoints load as entrants)")
    ap.add_argument("--native-library", default=f"{LIBS}/libdrmario_pool.dylib")
    ap.add_argument("--reach-library", default=f"{LIBS}/libdrm_reach_full.dylib")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--planner-workers", type=int, default=0)
    ap.add_argument("--model", default=None, help="showy model JSON (default: built-in stub)")
    ap.add_argument("--tier-bar", type=float, default=30.0, help="own-clear score treated as showy (V ~ 1)")
    ap.add_argument("--extra-model", help="second knob term's model JSON (no own-clear override)")
    ap.add_argument("--extra-lambda", type=float, default=0.0)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--paces", nargs="+", default=["normal", "top_humans"])
    ap.add_argument("--pairs", type=int, default=4, help="seed pairs per (lambda, pace)")
    ap.add_argument("--batch-pairs", type=int, default=16)
    ap.add_argument("--seed-start", type=int, default=7919)
    ap.add_argument("--identity-check", action="store_true")
    ap.add_argument("--verify-inputs", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    model = args.model if args.model else STUB
    if args.model:
        from drmc_rl.style.showy_knob import ShowyModel
        model = ShowyModel.load(args.model).spec   # inline: the run is immune to the file changing later
    runner = Runner(args)
    seeds = pick_seeds(args.pairs, args.seed_start)
    report = dict(schema="drmc-showy-h2h-v1", checkpoint=args.checkpoint, base_checkpoint=args.base_checkpoint, device=args.device, level=14, speed="hi",
                  delay=4, backend="events", model=model if not args.model else args.model, model_spec=model,
                  tier_bar=args.tier_bar, extra_model=args.extra_model, extra_lambda=args.extra_lambda,
                  seeds=seeds, started=time.strftime("%Y-%m-%dT%H:%M:%S"), checks={}, results=[])

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(args.out)

    if args.identity_check:
        report["checks"]["identity"] = identity_check(runner, model, args.paces[0], seeds[:2])
        print(json.dumps(report["checks"]["identity"]), flush=True)
        save()
    if args.verify_inputs:
        report["checks"]["inputs"] = {p: verify_inputs(runner, model, p, seeds[:2]) for p in args.paces}
        print(json.dumps(report["checks"]["inputs"]), flush=True)
        save()
    for lam in args.lambdas:
        for pace in args.paces:
            scores, tot = [], {s: dict.fromkeys(STYLE_KEYS, 0) for s in ("knob", "base")}
            dec = wall = 0.0
            for start in range(0, len(seeds), args.batch_pairs):
                chunk = seeds[start:start + args.batch_pairs]
                batch, elapsed = runner.play(runner.variants(lam if lam else None, model), pace, chunk)
                wall += elapsed
                dec += decisions(batch)
                for row, moves, _ in batch:
                    scores.append(row["score"])
                    a = int(row["side"])
                    for name, phys in (("knob", a), ("base", 1 - a)):
                        c = side_counters(moves, phys)
                        for k in STYLE_KEYS:
                            tot[name][k] += c[k]
                        if c.get("best_score", -1) > tot[name].get("best_score", -1):
                            tot[name]["best_score"], tot[name]["best"] = c["best_score"], c["best"]
                entry = dict(lam=lam, pace=pace, pairs_done=start // args.batch_pairs * args.batch_pairs + len(chunk),
                             **outcome(scores), decisions_per_sec=round(dec / max(wall, 1e-9), 1),
                             wall_seconds=round(wall, 1), style={k: rates(v) for k, v in tot.items()})
                report["results"] = [r for r in report["results"] if (r["lam"], r["pace"]) != (lam, pace)] + [entry]
                save()
                print(json.dumps({k: entry[k] for k in ("lam", "pace", "pairs_done", "games", "knob_score", "ci95",
                                                        "elo", "decisions_per_sec")}), flush=True)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()
    runner.runtime.close()


if __name__ == "__main__":
    main()
