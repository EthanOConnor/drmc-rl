"""Compare verified human execution with exact witnesses and planning latency.

This is a bounded motor-control diagnostic, not a promoted execution profile.
Only aggregate statistics leave the input corpus; scripts remain private.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from drmc_rl.execution.profile import GAMEPLAY_MASK, script_metrics
from drmc_rl.human.cadence import add_thinking_delay, hold_soft_drop_suffix, motor_parameters, shape_human_movement
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, compute_speed_threshold, simulate_frame
from drmc_rl.planning.native_reach import NativeReachabilityRunner, resolve_library_path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def action(mask):
    return (1 if mask & 2 else 2 if mask & 1 else 0) * 6 + (3 if mask & 4 else 0) + (1 if mask & 128 else 2 if mask & 64 else 0)


def buttons(index):
    return (0, 2, 1)[index // 6] | (4 if index % 6 >= 3 else 0) | (0, 128, 64)[index % 3]


def initial_state(row):
    initial = int(row["initial_buttons"])
    return FrameState(3, 0, 0, int(row["initial_speed_counter"]),
        int(row["initial_horizontal_velocity"]) & 15,
        HoldDir(1 if initial & 2 else 2 if initial & 1 else 0),
        int(row["initial_frame_parity"]),
        Rotation(1 if initial & 128 else 2 if initial & 64 else 0))


def columns(row):
    board = np.frombuffer(bytes.fromhex(row["field_hex"]), dtype=np.uint8).reshape(16, 8)
    return np.bitwise_or.reduce((board != 255).astype(np.uint16) << np.arange(16, dtype=np.uint16)[:, None], axis=0)


def metrics(script, initial):
    result = asdict(script_metrics(script, initial_buttons=initial))
    masks = np.asarray(script, dtype=np.uint8)
    previous = np.concatenate(([initial], masks[:-1]))
    changes = np.flatnonzero((masks ^ previous) & GAMEPLAY_MASK)
    result["first_change_frames"] = int(changes[0]) if changes.size else len(script)
    result["carried_input"] = float(bool(initial & GAMEPLAY_MASK))
    result["soft_drop_fraction"] = result["soft_drop_frames"] / max(1, len(script))
    return result


def weights(rows):
    """Equal players, equal games within player, equal sampled pills within game."""
    counts = Counter((row["player_id"], row["game_id"]) for row in rows)
    games = Counter(player for player, _ in counts)
    return np.asarray([1 / (games[row["player_id"]] * counts[row["player_id"], row["game_id"]]) for row in rows])


def summary(rows, key):
    if not rows:
        return {"scripts": 0}
    weight = weights(rows)
    values = {}
    for name in sorted({name for row in rows for name in row[key]}):
        keep = [i for i, row in enumerate(rows) if row[key].get(name) is not None]
        if not keep:
            continue
        array = np.asarray([rows[i][key][name] for i in keep], dtype=float)
        w = weight[keep]
        order = np.argsort(array, kind="stable")
        quantiles = np.interp([0.1, 0.5, 0.9, 0.99], np.cumsum(w[order]) / w.sum(), array[order])
        values[name] = {"mean": float(np.average(array, weights=w)),
                        **dict(zip(("p10", "p50", "p90", "p99"), quantiles.tolist(), strict=True))}
    return {"scripts": len(rows), "players": len({r["player_id"] for r in rows}),
            "games": len({r["game_id"] for r in rows}), "metrics": values}


def replay(cols, state, script, threshold):
    for frame, mask in enumerate(script, 1):
        state = simulate_frame(cols, state, action(int(mask)), speed_threshold=threshold)
        if state.locked:
            return state, frame
    return state, None


def audit_row(row, runner, leads):
    cols, spawn = columns(row), initial_state(row)
    threshold = compute_speed_threshold(row["speed"], row["speed_ups"])
    target = tuple(row["lock_pose"])
    human_lock, human_tau = replay(cols, spawn, row["script"], threshold)
    if not human_lock.locked or (human_lock.x, human_lock.y, human_lock.rot) != target or human_tau != len(row["script"]):
        raise ValueError("verified sample no longer replays with current stepper")
    result = {key: row[key] for key in ("player_id", "game_id", "cohort", "partition", "human", "high_board")}
    for lead in leads:
        state, _ = replay(cols, spawn, [0] * lead, threshold)
        if state.locked:
            costs = np.full(512, 65535, dtype=np.uint16)
            costs[state.rot * 128 + state.y * 8 + state.x] = lead
            cost = lead if (state.x, state.y, state.rot) == target else None
            witness = np.zeros(lead, dtype=np.uint8) if cost is not None else None
        else:
            reach = runner.bfs_full(cols, state, speed_threshold=threshold)
            costs = reach.costs_u16.copy()
            cost = reach.cost_for_pose(*target)
            raw = reach.script_for_pose(*target)
            witness = None if raw is None else np.asarray([0] * lead + [buttons(int(a)) for a in raw], dtype=np.uint8)
        legal = np.asarray([(rot % 2 == 0 and x < 7) or (rot % 2 == 1 and y >= 1)
                            for rot in range(4) for y in range(16) for x in range(8)])
        result[f"lead_{lead}"] = {"target_available": float(cost is not None),
                                  "candidates": int(((costs < 65535) & legal).sum())}
        if lead == 0:
            if witness is None or len(witness) > len(row["script"]):
                raise ValueError("exact planner omitted the recorded human target")
            lock, tau = replay(cols, spawn, witness, threshold)
            if not lock.locked or tau != len(witness) or (lock.x, lock.y, lock.rot) != target:
                raise ValueError("native witness replay mismatch")
            result["fastest"] = metrics(witness, row["initial_buttons"])
            result["difference"] = {name: result["human"][name] - value for name, value in result["fastest"].items()
                                    if value is not None and result["human"][name] is not None}
            for name, wanted in (("held_drop", 0), ("matched_duration", len(row["script"]) - len(witness))):
                shaped, _ = add_thinking_delay(cols, spawn, [action(int(b)) for b in witness],
                    speed_threshold=threshold, target=target, requested_frames=wanted)
                shaped, _ = hold_soft_drop_suffix(cols, spawn, shaped, speed_threshold=threshold, target=target)
                shaped_buttons = [buttons(int(a)) for a in shaped]
                lock, tau = replay(cols, spawn, shaped_buttons, threshold)
                if not lock.locked or (lock.x, lock.y, lock.rot) != target:
                    raise ValueError("cadence-shaped witness replay mismatch")
                result[name] = metrics(shaped_buttons[:tau], row["initial_buttons"])
            started = time.perf_counter()
            reaction, interval = motor_parameters(row["rating"])
            motor, info = shape_human_movement(cols, spawn, [action(int(b)) for b in witness],
                speed_threshold=threshold, target=target, requested_frames=len(row["script"]),
                reaction_frames=reaction, edge_interval=interval)
            if not info.get("retimed", False):
                motor = shaped
            motor_buttons = [buttons(int(a)) for a in motor]
            lock, tau = replay(cols, spawn, motor_buttons, threshold)
            if not lock.locked or (lock.x, lock.y, lock.rot) != target:
                raise ValueError("human motor witness replay mismatch")
            result["motor"] = metrics(motor_buttons[:tau], row["initial_buttons"])
            result["motor_fit"] = {"shaped": int(info["shaped"]), "retimed": int(info.get("retimed", False)),
                                   "elapsed_ms": (time.perf_counter() - started) * 1000}
            pace_frames = []
            for scale in (0.5, 1.0, 1.5):
                wanted = len(witness) + round((len(row["script"]) - len(witness)) * scale)
                paced, paced_info = shape_human_movement(cols, spawn, [action(int(b)) for b in witness],
                    speed_threshold=threshold, target=target, requested_frames=wanted,
                    reaction_frames=reaction, edge_interval=interval)
                if not paced_info.get("retimed", False):
                    paced, _ = add_thinking_delay(cols, spawn, paced, speed_threshold=threshold,
                        target=target, requested_frames=wanted - len(witness))
                    paced, _ = hold_soft_drop_suffix(cols, spawn, paced, speed_threshold=threshold, target=target)
                lock, tau = replay(cols, spawn, [buttons(int(a)) for a in paced], threshold)
                if not lock.locked or (lock.x, lock.y, lock.rot) != target:
                    raise ValueError("pace witness replay mismatch")
                pace_frames.append(tau)
            result["pace"] = {"inversions": sum(a > b for a, b in zip(pace_frames, pace_frames[1:])),
                              "quick_frames": pace_frames[0], "normal_frames": pace_frames[1], "relaxed_frames": pace_frames[2]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--heldout-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.max_states < 1 or 0 not in args.leads or min(args.leads) < 0:
        parser.error("use a new output, positive state count, and nonnegative leads including zero")
    manifest = json.loads(Path(str(args.input) + ".manifest.json").read_text())
    if manifest.get("schema") != "drmc-execution-corpus-sample-v2" or not manifest.get("recorded_lock_replay_verified"):
        parser.error("input must be a covered, aligned, recorded-lock-verified v2 sample")
    if digest(args.input) != manifest["output_sha256"]:
        parser.error("input hash mismatch")
    started = time.monotonic()
    rows, groups = [], defaultdict(list)
    games, players = {}, {}
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        if not row["recorded_lock_verified"]:
            raise ValueError("unverified sample row")
        game, player = row["game_id"], row["player_id"]
        if games.setdefault(game, row["split"]) != row["split"] or players.setdefault(player, row["player_fold"]) != row["player_fold"]:
            raise ValueError("holdout identity is inconsistent")
        row["partition"] = "heldout_player" if row["player_fold"] == 0 else row["split"]
        row["cohort"] = f'{int(row["rating"] // 400) * 400}/speed{row["speed"]}'
        row["human"] = metrics(row["script"], row["initial_buttons"])
        cols = columns(row)
        row["high_board"] = bool(np.any(cols & 15))
        rows.append(row)
        if not args.heldout_only or row["partition"] != "train":
            groups[row["cohort"], row["high_board"]].append(row)
    population = {}
    for cohort, partition in sorted({(r["cohort"], r["partition"]) for r in rows}):
        subset = [r for r in rows if r["cohort"] == cohort and r["partition"] == partition]
        population[f"{cohort}/{partition}"] = summary(subset, "human")
    # Equal strata round robin, then stable random rank. This deliberately
    # oversamples low-support/high-board strata; it is not population prevalence.
    rng = np.random.default_rng(args.seed)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < args.max_states and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < args.max_states:
                chosen.append(groups[key].pop())
    runner = NativeReachabilityRunner(max_frames=512)
    results = []
    for i, row in enumerate(chosen):
        results.append(audit_row(row, runner, args.leads))
        if (i + 1) % 16 == 0:
            print(json.dumps({"audited": i + 1, "total": len(chosen), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    comparisons = {}
    for name, subset in [("all", results), ("high_board", [r for r in results if r["high_board"]])]:
        comparisons[name] = {key: summary(subset, key) for key in ("human", "fastest", "held_drop", "matched_duration", "motor", "motor_fit", "pace", "difference", *(f"lead_{lead}" for lead in args.leads))}
    for cohort in sorted({r["cohort"] for r in results}):
        subset = [r for r in results if r["cohort"] == cohort]
        comparisons[cohort] = {key: summary(subset, key) for key in ("human", "fastest", "held_drop", "matched_duration", "motor", "motor_fit", "pace", "difference", *(f"lead_{lead}" for lead in args.leads))}
    report = {"schema": "drmc-execution-control-audit-v3", "diagnostic_only": True,
        "source_sha256": manifest["output_sha256"], "corpus_manifest_sha256": manifest["corpus_manifest_sha256"],
        "native_revision": subprocess.check_output(["git", "-C", "vendor/drmario_native", "rev-parse", "HEAD"], text=True).strip(),
        "native_library_sha256": digest(resolve_library_path()),
        "source_code_sha256": {p: digest(p) for p in ("tools/audit_execution_control.py", "drmc_rl/planning/fast_reach.py", "drmc_rl/execution/profile.py", "drmc_rl/human/cadence.py")},
        "weighting": "equal players, equal games within player, equal sampled pills within game",
        "holdout": "player_fold zero held out; other players retain whole-game train/validation/test splits",
        "selection": "rating/speed/high-board round robin; uniform seeded selection within strata",
        "high_board": "occupied cell in top four rows (height at least 13)",
        "seed": args.seed, "heldout_only": args.heldout_only, "leads": args.leads,
        "sample_scripts": len(rows), "audited_states": len(results),
        "pace_comparisons": 2 * len(results),
        "pace_inversions": sum(r["pace"]["inversions"] for r in results),
        "retimed_states": sum(r["motor_fit"]["retimed"] for r in results),
        "held_input_states": sum(r["motor_fit"]["shaped"] for r in results),
        "all_lock_replays_matched": True,
        "population": population, "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Short per-pill windows do not measure sustained inter-pill input rate.",
                         "First active input can be carried over; first change is reported separately.",
                         "Matched-duration shaping uses the recorded human tau to isolate path shape from timing prediction error.",
                         "No claim of full-pair timing value or an end-to-end emulator replay certificate."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "audited_states": len(results), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
