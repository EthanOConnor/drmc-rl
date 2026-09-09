"""Annotate real public controller replay with complete conditional geometry.

This CPU sidecar never chooses training actions or mutates the running model.
Whole-game samples span earlier and later decisions. It retains all root
candidates and splits by reset seed across updates/paces, so repeated games
cannot leak into validation. Original training holdout seeds remain excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.arena.experiment import dump
from drmc_rl.execution.pace import resolve_pace
from drmc_rl.human.motor_opportunity import (
    MotorOpportunityLabeler, OPPORTUNITY_CONDITION, OPPORTUNITY_SCHEMA,
    state_from_controller_replay,
)
from drmc_rl.planning.native_reach import NativeReachabilityRunner


def selected_rows(replay, *, per_game, limit, seed):
    groups = defaultdict(list)
    for i, (game_seed, side) in enumerate(zip(replay["game_seed"], replay["learner_port"], strict=True)):
        groups[int(game_seed), int(side)].append(i)
    # Randomize games, not the temporal coverage within each game. Limiting
    # work does not systematically discard late decisions or one player port.
    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    selected = []
    for key in keys:
        indices = groups[key]
        by_time = sorted(indices, key=lambda i: int(replay["observed_frame"][i]))
        positions = np.linspace(0, len(by_time) - 1, min(per_game, len(by_time))).astype(int)
        selected.extend(by_time[p] for p in positions)
        if len(selected) >= limit:
            break
    return selected[:limit]


def split_for_seed(game_seed, split_seed):
    key = f"motor-opportunity:{split_seed}:{game_seed}".encode()
    return "validation" if int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 5 == 0 else "train"


def annotate_row(replay, index, labeler):
    metadata = json.loads(str(replay["metadata"]))
    state = state_from_controller_replay(replay, index)
    labels = labeler.label(state, resolve_pace(metadata["pace"]), compute_frames=state["compute_frames"])
    if labels.delay != state["decision_delay_frames"]:
        raise ValueError("replay computation/reaction charge does not match conditional labels")
    lo, hi = map(int, replay["offsets"][index:index + 2])
    source_actions = replay["actions"][lo:hi]
    order = np.argsort(source_actions)
    if (not np.array_equal(source_actions[order], labels.actions) or
            not np.array_equal(replay["costs"][lo:hi][order], labels.root_costs)):
        raise ValueError("replayed complete root frontier differs from its original collection")
    payload = labels.arrays()
    payload.update({key: replay[key][index] for key in (
        "observation", "pill", "preview", "public_context", "controller_geometry",
    )})
    payload["base_logits"] = replay["base_logits"][lo:hi][order]
    payload["behavior_logp"] = replay["behavior_logp"][lo:hi][order]
    # Observed continuation outcomes are distinct from conditional own-board
    # effects; no unchosen move is assigned this natural-game result.
    payload["observed_action"] = replay["action"][index]
    payload["observed_return"] = replay["return"][index]
    return labels, payload


def run(config):
    source = Path(config["replay_directory"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "roots").mkdir(exist_ok=True)
    configuration = output / "config.json"
    if configuration.exists() and json.loads(configuration.read_text()) != config:
        raise ValueError("resume must preserve the exact opportunity-bank configuration")
    dump(configuration, config)
    journal_path = output / "roots.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()] if journal_path.exists() else []
    seen = {(r["source_sha256"], r["source_row"]) for r in rows}
    finished = set()
    legacy = set()
    maximum = int(config.get("max_roots", 2048))
    per_game = int(config.get("per_game", 3))
    per_update = int(config.get("per_update", 96))
    if min(maximum, per_game, per_update) < 1:
        raise ValueError("root and sampling budgets must be positive")
    watch_seconds = min(60., max(1., float(config.get("watch_seconds", 10))))
    holdout = set(map(int, config["holdout_seeds"]))
    planner = NativeReachabilityRunner()
    progress = dict(schema=OPPORTUNITY_SCHEMA, condition=OPPORTUNITY_CONDITION,
                    status="Running", target_roots=maximum)

    def report(status):
        progress.update(status=status, roots=len(rows),
                        candidates=sum(r["candidates"] for r in rows),
                        next_candidates=sum(r["next_candidates"] for r in rows),
                        splits=dict(Counter(r["split"] for r in rows)),
                        paces=dict(Counter(r["pace"] for r in rows)),
                        skipped_legacy_shards=sorted(legacy),
                        updated_at=datetime.now(UTC).isoformat())
        dump(output / "progress.json", progress)

    try:
        report("Complete" if len(rows) >= maximum else "Running")
        with MotorOpportunityLabeler(planner, lib_path=config.get("native_library")) as labeler:
            while len(rows) < maximum:
                work = False
                watermark = json.loads(Path(config.get(
                    "training_progress", source.parent / "training.json")).read_text())["updates"]
                for path in sorted(source.glob("update-*.npz")):
                    if path.name in finished:
                        continue
                    with np.load(path, allow_pickle=False) as data:
                        metadata = json.loads(str(data["metadata"]))
                        if int(metadata["update"]) > int(watermark):
                            # Replays are written before the optimizer audit.
                            # Wait until this update has a durable checkpoint.
                            continue
                        if metadata["schema"] != "drmc-public-controller-replay-v2":
                            legacy.add(path.name)
                            finished.add(path.name)
                            continue
                        # Materialize each compressed array once, never once
                        # per selected row on a remote filesystem.
                        replay = {key: data[key] for key in data.files}
                    if holdout.intersection(map(int, replay["game_seed"])):
                        raise ValueError("training replay overlaps reserved controller-evaluation seeds")
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    indices = selected_rows(replay, per_game=per_game, limit=per_update,
                                            seed=int(config["seed"]) + int(metadata["update"]))
                    for index in indices:
                        if (digest, index) in seen:
                            continue
                        started = time.perf_counter()
                        labels, payload = annotate_row(replay, index, labeler)
                        game_seed = int(replay["game_seed"][index])
                        split = split_for_seed(game_seed, int(config["seed"]))
                        identity = f"{digest[:16]}-{index:06d}"
                        record = dict(id=identity, source=path.name, source_sha256=digest,
                                      source_row=index, source_update=int(metadata["update"]),
                                      game_seed=game_seed, learner_port=int(replay["learner_port"][index]),
                                      frame=int(replay["observed_frame"][index]), split=split,
                                      pace=metadata["pace"], level=metadata["level"],
                                      **labels.summary())
                        record["seconds"] = time.perf_counter() - started
                        record["path"] = f"roots/{identity}.npz"
                        payload["metadata"] = np.asarray(json.dumps(record))
                        path_out = output / record["path"]
                        temporary = path_out.with_suffix(".npz.next")
                        with temporary.open("wb") as stream:
                            np.savez_compressed(stream, **payload)
                        temporary.replace(path_out)
                        with journal_path.open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        rows.append(record)
                        seen.add((digest, index))
                        work = True
                        report("Running")
                        if len(rows) >= maximum:
                            break
                    finished.add(path.name)
                    if len(rows) >= maximum:
                        break
                if len(rows) >= maximum:
                    report("Complete")
                    break
                if not config.get("watch", False):
                    report("Source exhausted")
                    break
                # Waiting is explicit, not presented as advancing annotation.
                if work or progress.get("status") != "Waiting for controller replay":
                    report("Waiting for controller replay")
                time.sleep(watch_seconds)
    except BaseException as error:
        progress["error"] = str(error)
        report("Failed")
        raise
    finally:
        planner.close()
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
