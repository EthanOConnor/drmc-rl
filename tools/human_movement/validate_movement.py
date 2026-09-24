"""Validate the human movement generator: statistics against held-out players, and exactness.

    python -m tools.human_movement.validate_movement stats FEATURES.parquet SLACK.parquet MODEL.json OUT.json \
        [--holdout-folds 0,1,2,3] [--per-band 6000] [--workers 4]
    python -m tools.human_movement.validate_movement correctness OUT.json [--cases 100000] [--workers 6]

``stats`` regenerates movement for held-out human placements: the same bottle, start state and
final pose, with each pace's calibrated band (MODEL.json must be fitted without those players).
Human and generated windows go through one metric function, so definitions match exactly.
Sloth and Super Human have no human band; they run on the Relaxed and Top Humans situations and
are reported against those bands. ``correctness`` checks random situations (high stacks, fast
gravity, random controller microstate, early-decision start delays) for an exact lock at the target
pose, the generated-script limits, and determinism.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

PACE_BANDS = {"sloth": "relaxed", "relaxed": "relaxed", "normal": "normal", "fast": "fast",
              "top_humans": "top_humans", "super_human": "top_humans"}
MOVE = 0xC3


def action_mask(action: int) -> int:
    d = action // 6
    return (2 if d == 1 else 1 if d == 2 else 0) | (4 if action % 6 >= 3 else 0) | (
        0x80 if action % 3 == 1 else 0x40 if action % 3 == 2 else 0)


def mask_action(mask: int) -> int:
    direction = 1 if mask & 2 else 2 if mask & 1 else 0
    rotation = 1 if mask & 0x80 else 2 if mask & 0x40 else 0
    return direction * 6 + (3 if mask & 4 else 0) + rotation


def window_metrics(cols, spawn, masks, threshold, initial, rot_need, target_x):
    """Reaction, rhythm and path metrics of one spawn-to-lock window (slack.py definitions)."""
    from drmc_rl.planning.fast_reach import simulate_frame
    state, xs, rots, first_down, late = spawn, [spawn.x], [spawn.rot], None, 0
    lock, states = None, [spawn]
    for i, m in enumerate(masks):
        state = simulate_frame(cols, state, mask_action(int(m)), speed_threshold=threshold)
        states.append(state)
        if m & 4 and first_down is None:
            first_down = i
        if first_down is not None and state.x != xs[-1]:
            late += 1
        xs.append(state.x)
        rots.append(state.rot)
        if state.locked:
            lock = i + 1
            break
    dx = np.diff(xs)
    moves = dx[dx != 0]
    xf = xs[-1]
    direction = np.sign(xf - xs[0])
    path = np.array(xs)
    over = float(np.max((path - xf) * direction)) if direction else float(np.max(np.abs(path - xf)))
    reversals = int(np.count_nonzero(np.diff(np.sign(moves)))) if len(moves) > 1 else 0
    rot_changes = int(np.count_nonzero(np.diff(rots)))
    window = list(masks[:lock])
    prev, presses, das, down = initial, [], 0, 0
    held = {1: None, 2: None}
    for i, b in enumerate(window):
        new = b & ~prev
        if new & MOVE:
            presses.append(i)
        for bit in (1, 2):
            if b & bit:
                held[bit] = i if held[bit] is None else held[bit]
                das |= i + 1 - held[bit] >= 16
            else:
                held[bit] = None
        down += bool(b & 4)
        prev = b
    active = [i for i, m in enumerate(window) if m & 0xC3 and not (i and window[i - 1] & 0xC3 == m & 0xC3)]
    idle = 0
    if len(active) > 1:
        run = 0
        for m in window[active[0]:active[-1] + 1]:
            run = 0 if m & 0xC7 else run + 1
            idle = max(idle, run)
    gaps = np.diff(presses)
    steer_end = max([i for i in range(1, len(states)) if (states[i].x, states[i].rot)
                     != (states[i - 1].x, states[i - 1].rot)], default=0)
    probe, min_drop = states[steer_end], 0
    while not probe.locked and min_drop < 600:
        probe = simulate_frame(cols, probe, 3, speed_threshold=threshold)
        min_drop += 1
    lateral = int(np.abs(dx).sum())
    corrected = int(reversals > 0 or max(0.0, over) > 0 or rot_changes > rot_need + 1 or late > 0)
    return {"tau": lock, "pose": (state.x, state.y, state.rot), "reaction": presses[0] if presses else -1,
            "gap": float(np.median(gaps)) if len(gaps) else -1.0, "das": int(das),
            "down_share": down / max(len(window), 1), "idle": idle, "reversals": reversals,
            "overshoot": max(0.0, over), "extra_rot": max(0, rot_changes - rot_need), "late": late,
            "extra_lateral": lateral - abs(target_x - 3), "corrected": corrected,
            "presses": len(presses), "steer_end": steer_end,
            "slack": (lock or 0) - steer_end - min_drop}


# ----------------------------------------------------------------------------- statistics
def _init_stats(model_path):
    global MODEL, runner
    from drmc_rl.human.movement import load_model
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    MODEL = load_model(model_path)
    runner = NativeReachabilityRunner()


def _stats_work(rows):
    from drmc_rl.human.movement import HumanMovement
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame
    out = []
    movers = {}
    for r in rows:
        pace = r["pace"]
        mv = movers.get(pace) or movers.setdefault(pace, HumanMovement(pace, MODEL))
        raw = bytearray()
        for n, b in struct.iter_unpack("<HB", r["rle"]):
            raw.extend(bytes((b,)) * n)
        human_masks = raw[:-1]
        board = np.frombuffer(bytes(r["field_bytes"]), dtype=np.uint8).reshape(16, 8)
        cols = np.zeros(8, dtype=np.uint16)
        for y in range(16):
            cols |= (board[y] != 0xFF).astype(np.uint16) << y
        initial = int(r["held_before_spawn"])
        spawn = FrameState(x=3, y=0, rot=0, speed_counter=int(r["speed_counter"]),
                           hor_velocity=int(r["horizontal_velocity"]) & 15,
                           hold_dir=HoldDir(1 if initial & 2 else 2 if initial & 1 else 0),
                           rot_hold=Rotation(1 if initial & 0x80 else 2 if initial & 0x40 else 0),
                           frame_parity=(int(r["frame_counter"]) & 1) ^ 1)
        threshold = compute_speed_threshold(int(r["speed"]), int(r["speed_ups"]))
        target = (int(r["lock_x"]), int(r["lock_y_top"]), int(r["lock_rotation"]) & 3)
        rot_need = {0: 0, 1: 1, 2: 2, 3: 1}[target[2]]
        human = window_metrics(cols, spawn, human_masks, threshold, initial, rot_need, target[0])
        if human["tau"] != int(r["tau_frames"]) or human["pose"] != target:
            out.append(None)
            continue
        decision = mv.decide(r["game_seed"], r["key"], threshold=threshold)
        delay, status = decision.reaction_frames, "ok"
        while True:
            start, locked = spawn, False
            for _ in range(delay):
                start = simulate_frame(cols, start, 0, speed_threshold=threshold)
                if start.locked:
                    locked = True
                    break
            reach = None if locked else runner.bfs_full(cols, start, speed_threshold=threshold,
                                                        **mv.planning.planner_args(0))
            cost = None if reach is None else reach.cost_for_pose(*target)
            if cost is None and reach is not None:
                reach = runner.bfs_full(cols, start, speed_threshold=threshold)
                cost = reach.cost_for_pose(*target)
                status = "outside_named_pace"
            if cost is not None:
                break
            status = "reaction_shortened"
            if delay == 0:
                break
            delay //= 2
        if cost is None:
            out.append({"band": r["band"], "pace": pace, "status": "unreachable", "human": human,
                        "rows_fallen": target[1] + 1, "fastest": r["fastest_frames"]})
            continue
        witness = reach.script_for_pose(*target).copy()
        started = time.perf_counter()
        script, info = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                                   witness=witness, execution_delay=delay)
        elapsed = time.perf_counter() - started
        masks = [0] * delay + [action_mask(int(a)) for a in script]
        gen = window_metrics(cols, spawn, masks, threshold, initial, rot_need, target[0])
        if gen["pose"] != target or gen["tau"] != len(masks):
            raise RuntimeError("generated script missed its target")
        out.append({"band": r["band"], "pace": pace, "status": status, "route": info["route"], "player": r["player"],
                    "human": human, "gen": gen, "rows_fallen": target[1] + 1,
                    "fastest": r["fastest_frames"], "ms": 1000 * elapsed, "threshold": threshold})
    return out


def _hash(text):
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=6).digest(), "little")


def _summ(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    return {"p10": round(float(np.percentile(v, 10)), 2), "p50": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "mean": round(float(v.mean()), 4), "n": int(len(v))}


def _ks(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return None
    grid = np.union1d(a, b)
    return round(float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                                      - np.searchsorted(b, grid, "right") / len(b)))), 4)


def compare(records):
    ok = [r for r in records if r and "gen" in r]
    result = {"placements": len(ok), "unreachable": sum(1 for r in records if r and r["status"] == "unreachable"),
              "status": dict(Counter(r["status"] for r in ok)), "routes": dict(Counter(r["route"] for r in ok)),
              "generate_ms": _summ([r["ms"] for r in ok])}
    for side in ("human", "gen"):
        m = [r[side] for r in ok]
        tau = np.array([x["tau"] for x in m], float)
        fastest = np.array([r["fastest"] for r in ok], float)
        result[side] = {
            "reaction": _summ([x["reaction"] for x in m if x["reaction"] >= 0]),
            "tau": _summ(tau), "lost": _summ(tau - fastest),
            "gap": _summ([x["gap"] for x in m if x["gap"] >= 0]),
            "das": round(float(np.mean([x["das"] for x in m])), 4),
            "down_share": round(float(np.mean([x["down_share"] for x in m])), 4),
            "no_down": round(float(np.mean([x["down_share"] == 0 for x in m])), 4),
            "down_share_quartiles": [round(float(q), 3) for q in np.percentile([x["down_share"] for x in m], [25, 50, 75])],
            "corrected": round(float(np.mean([x["corrected"] for x in m])), 4),
            "reversal": round(float(np.mean([x["reversals"] > 0 for x in m])), 4),
            "overshoot": round(float(np.mean([x["overshoot"] > 0 for x in m])), 4),
            "extra_rotation": round(float(np.mean([x["extra_rot"] >= 2 for x in m])), 4),
            "late_lateral": round(float(np.mean([x["late"] > 0 for x in m])), 4),
            "pause": round(float(np.mean([x["idle"] >= 10 for x in m])), 4),
            "idle": _summ([x["idle"] for x in m]),
        }
        rows = {}
        for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16)):
            sel = [(r[side]["tau"] - r["fastest"]) for r in ok if lo <= r["rows_fallen"] <= hi]
            rows[f"{lo}-{hi}"] = _summ(sel)
        result[side]["lost_by_rows_fallen"] = rows
        result[side]["steer_frames"] = _summ([x["steer_end"] for x in m])
        result[side]["descent_slack"] = _summ([x["slack"] for x in m])
        result[side]["slack_by_rows_fallen"] = {
            f"{lo}-{hi}": _summ([r[side]["slack"] for r in ok if lo <= r["rows_fallen"] <= hi])
            for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16))}
        for g, (lo, hi) in {"slow": (13, 99), "mid": (7, 12), "fast": (0, 6)}.items():
            sel = [(r[side]["tau"] - r["fastest"]) for r in ok if lo <= r["threshold"] <= hi]
            result[side].setdefault("lost_by_gravity", {})[g] = _summ(sel)
    spread = {}
    players = Counter(r["player"] for r in ok)
    for side in ("human", "gen"):
        per = {"reaction": [], "lost": [], "gap": []}
        for player, count in players.items():
            if count < 40:
                continue
            rows = [r for r in ok if r["player"] == player]
            per["reaction"].append(np.median([r[side]["reaction"] for r in rows if r[side]["reaction"] >= 0]))
            per["lost"].append(np.median([r[side]["tau"] - r["fastest"] for r in rows]))
            per["gap"].append(np.median([r[side]["gap"] for r in rows if r[side]["gap"] >= 0]))
        spread[side] = {k: round(float(np.std(v)), 2) for k, v in per.items() if len(v) >= 3}
    spread["players"] = sum(1 for c in players.values() if c >= 40)
    result["between_player_sd_of_medians"] = spread
    h, g = [r["human"] for r in ok], [r["gen"] for r in ok]
    fastest = [r["fastest"] for r in ok]
    result["ks"] = {
        "reaction": _ks([x["reaction"] for x in h if x["reaction"] >= 0], [x["reaction"] for x in g if x["reaction"] >= 0]),
        "lost": _ks([x["tau"] - f for x, f in zip(h, fastest)], [x["tau"] - f for x, f in zip(g, fastest)]),
        "gap": _ks([x["gap"] for x in h if x["gap"] >= 0], [x["gap"] for x in g if x["gap"] >= 0]),
        "down_share": _ks([x["down_share"] for x in h], [x["down_share"] for x in g]),
    }
    return result


def stats(args):
    import pyarrow.parquet as pq
    from tools.human_movement.fit_movement_model import BANDS
    feats = pq.read_table(args.features, columns=["rating", "rating_sd", "player_index", "player_fold", "day"]).to_pandas()
    players = pq.read_table(args.features + ".players.parquet").to_pandas()
    raw = pq.read_table(args.slack, columns=["player", "fastest_frames", "rle", "field_bytes", "held_before_spawn",
                                             "speed_counter", "horizontal_velocity", "frame_counter", "speed",
                                             "speed_ups", "lock_x", "lock_y_top", "lock_rotation", "tau_frames",
                                             "decision_id"]).to_pandas()
    raw = raw[raw.fastest_frames.notna() & raw.player.isin(set(players.player))].reset_index(drop=True)
    if len(raw) != len(feats):
        raise ValueError("features and slack rows are not aligned")
    held = {int(f) for f in args.holdout_folds.split(",")}
    rng = np.random.default_rng(args.seed)
    jobs = []
    for band, (lo, hi) in BANDS.items():
        sel = (feats.rating >= lo) & (feats.rating < hi) & (feats.rating_sd <= 150)
        if band == "top_humans":
            # The model was fitted on the even half of the top players.
            sel &= (feats.player_index % 2 == 0) if args.insample else (feats.player_index % 2 == 1)
        else:
            sel &= ~feats.player_fold.isin(held) if args.insample else feats.player_fold.isin(held)
        index = np.flatnonzero(sel.to_numpy())
        if len(index) > args.per_band:
            index = rng.choice(index, args.per_band, replace=False)
        for pace, base in PACE_BANDS.items():
            if base != band:
                continue
            for i in index:
                r = raw.iloc[int(i)].to_dict()
                # One style draw per human player, so between-player spread is comparable.
                r.update(band=band, pace=pace, key=_hash(r["decision_id"]), game_seed=_hash(r["player"]))
                jobs.append(r)
    chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
    with Pool(args.workers, initializer=_init_stats, initargs=(args.model,)) as pool:
        records = [x for part in pool.imap(_stats_work, chunks) for x in part]
    report = {"schema": "drmc-human-movement-validation-v1", "model": args.model,
              "holdout": {"folds": sorted(held), "top_humans": "odd player index half (model fitted on the even half)",
                          "evaluated": "training players (in-sample)" if args.insample else "held-out players"},
              "paces": {}}
    for pace, band in PACE_BANDS.items():
        report["paces"][pace] = {"band": band, **compare([r for r in records if r and r["pace"] == pace])}
    json.dump(report, open(args.output, "w"), indent=1)
    for pace, v in report["paces"].items():
        h, g = v["human"], v["gen"]
        print(f"{pace:12s} n={v['placements']:5d} react {h['reaction']['p50']:5.1f}/{g['reaction']['p50']:5.1f} "
              f"lost {h['lost']['p50']:5.1f}/{g['lost']['p50']:5.1f} gap {h['gap']['p50']:4.1f}/{g['gap']['p50']:4.1f} "
              f"das {h['das']:.2f}/{g['das']:.2f} down {h['down_share']:.2f}/{g['down_share']:.2f} "
              f"corr {h['corrected']:.2f}/{g['corrected']:.2f} pause {h['pause']:.2f}/{g['pause']:.2f} ks {v['ks']} routes {v['routes']}")


# ---------------------------------------------------------------------------- correctness
def _init_correct():
    global runner
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    runner = NativeReachabilityRunner()


def _correct_work(seeds):
    from drmc_rl.human.early_decision import early_start_delay
    from drmc_rl.human.movement import movement_for_pace
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame
    paces = ["sloth", "relaxed", "normal", "fast", "top_humans", "super_human"]
    stats, times = Counter(), []
    for seed in seeds:
        rng = np.random.default_rng([0xC0FFEE, seed])
        pace = paces[seed % len(paces)]
        mv = movement_for_pace(pace)
        cols = np.zeros(8, np.uint16)
        height = int(rng.choice([rng.integers(0, 9), rng.integers(8, 15)]))
        for c in range(8):
            hh = int(np.clip(height + rng.integers(-4, 3), 0, 15))
            for y in range(16 - hh, 16):
                if rng.random() < 0.8:
                    cols[c] |= 1 << y
        cols[3] &= ~np.uint16(3)
        cols[4] &= ~np.uint16(3)
        speed, ups = int(rng.integers(0, 3)), int(rng.choice([rng.integers(0, 12), rng.integers(12, 50)]))
        threshold = compute_speed_threshold(speed, ups)
        spawn = FrameState(x=3, y=0, rot=0, speed_counter=int(rng.integers(0, threshold + 1)),
                           hor_velocity=int(rng.integers(0, 16)), hold_dir=HoldDir(int(rng.integers(0, 3))),
                           rot_hold=Rotation(int(rng.integers(0, 3))), frame_parity=int(rng.integers(0, 2)))
        decision = mv.decide(seed, int(rng.integers(0, 1 << 30)), threshold=threshold)
        # Early-decision contract: execution starts spawn + max(reaction, request + compute - spawn, 0).
        lead = int(rng.choice([0, 0, rng.integers(1, 90)]))
        compute = int(rng.integers(0, 12))
        delay = early_start_delay(-lead, 0, compute, decision.reaction_frames)
        reach = None
        while True:
            start = spawn
            for _ in range(delay):
                start = simulate_frame(cols, start, 0, speed_threshold=threshold)
                if start.locked:
                    break
            if not start.locked:
                reach = runner.bfs_full(cols, start, speed_threshold=threshold, **mv.planning.planner_args(0))
                if np.any(reach.costs_u16 != 0xFFFF):
                    break
            if delay == 0:
                reach = None
                break
            delay //= 2
            stats["reaction_shortened"] += 1
        if reach is None:
            stats["no_reachable_placement"] += 1
            continue
        poses = np.flatnonzero(reach.costs_u16 != 0xFFFF)
        pose = int(rng.choice(poses))
        target = (pose & 7, (pose >> 3) & 15, (pose >> 7) & 3)
        witness = reach.script_for_pose(*target).copy()
        started = time.perf_counter()
        script, info = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                                   witness=witness, execution_delay=delay)
        times.append(time.perf_counter() - started)
        state = start
        for i, a in enumerate(script, 1):
            state = simulate_frame(cols, state, int(a), speed_threshold=threshold)
            if state.locked:
                break
        exact = state.locked and i == len(script) and (state.x, state.y, state.rot) == target
        try:
            mv.floor.validate(cols, start, script, speed_threshold=threshold, execution_delay=delay)
            stats["limits_valid"] += 1
        except ValueError:
            pass
        again, _ = mv.generate(decision, cols, start, target, speed_threshold=threshold,
                               witness=witness, execution_delay=delay)
        stats["cases"] += 1
        stats["exact"] += int(exact)
        stats["deterministic"] += int(np.array_equal(script, again))
        stats["route/" + info["route"]] += 1
        stats[f"route/{pace}/{info['route']}"] += 1
        stats["high_board"] += int(height >= 12)
        stats["fast_gravity"] += int(threshold <= 6)
        stats["early_start"] += int(lead > 0)
    return stats, times


def correctness(args):
    seeds = list(range(args.cases))
    chunks = [seeds[i:i + 1000] for i in range(0, len(seeds), 1000)]
    total, times = Counter(), []
    with Pool(args.workers, initializer=_init_correct) as pool:
        for part, t in pool.imap_unordered(_correct_work, chunks):
            total.update(part)
            times.extend(t)
    report = {"schema": "drmc-human-movement-correctness-v1", "cases_requested": args.cases,
              "counts": dict(sorted(total.items())),
              "generate_ms": {"p50": float(np.median(times) * 1000), "p99": float(np.percentile(times, 99) * 1000),
                              "max": float(np.max(times) * 1000)}}
    json.dump(report, open(args.output, "w"), indent=1)
    print(json.dumps(report, indent=1))
    if not total["exact"] == total["deterministic"] == total["limits_valid"] == total["cases"]:
        raise SystemExit("correctness failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stats")
    s.add_argument("features")
    s.add_argument("slack")
    s.add_argument("model")
    s.add_argument("output")
    s.add_argument("--holdout-folds", default="0,1,2,3")
    s.add_argument("--per-band", type=int, default=6000)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--insample", action="store_true", help="evaluate the fitted (training) players instead")
    c = sub.add_parser("correctness")
    c.add_argument("output")
    c.add_argument("--cases", type=int, default=100000)
    c.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    stats(args) if args.command == "stats" else correctness(args)


if __name__ == "__main__":
    main()
