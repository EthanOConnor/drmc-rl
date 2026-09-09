"""Collect diverse complete public games with batched actors and durable roots.

This is native SMDP experience, not paced controller training. Source, anchor
and confirmation partitions use different reset seeds. Selection covers the
whole game while bounding retained snapshots; at most one loss predecessor
per game supplements temporal/tactical coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.envs.backends.drmario_pool import resolve_library_path
from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.search.native_pair import CAUSAL_PUBLIC_SCHEMA, capture_native_state, state_to_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.search.public_policy import PublicPolicyContinuation
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file
from drmc_rl.teachers.terminal_rollout import _native_executor
from tools.build_pair_state_pilot import (
    _OUTCOME_NAME, _atomic_gzip_jsonl, _candidate_bin, _condition_visible_reserve, _tactical_stratum,
)


def game_catalog(config):
    """Freeze conditions and unique seed assignments before asynchronous play."""
    partitions = config["partitions"]
    if not partitions or any(not re.fullmatch(r"[a-z][a-z0-9_-]*", name)
                             or int(count) != count or count < 1
                             for name, count in partitions.items()):
        raise ValueError("partitions require safe names and positive game counts")
    excluded = set()
    for pair in config.get("excluded_reset_seeds", []):
        if len(pair) != 2 or any(int(x) != x or not 0 <= x <= 255 for x in pair):
            raise ValueError("excluded reset seeds must be byte pairs")
        excluded.add(tuple(pair))
    conditions = config["conditions"]
    if not conditions or any(not 0 <= c["level"] <= 20 or c["speed"] not in (0, 1, 2)
                             or not np.isfinite(c["weight"]) or c["weight"] <= 0 for c in conditions):
        raise ValueError("invalid level/speed condition weights")
    matchups = config["matchups"]
    if not matchups or any(len(m) != 2 or any(x not in config["members"] for x in m) for m in matchups):
        raise ValueError("each matchup must name two frozen public members")
    rng = np.random.default_rng(config["seed"])
    # Exclude the all-zero register state, which is not a cycling RNG seed.
    seeds = [divmod(int(x), 256) for x in rng.permutation(np.arange(1, 65536))
             if divmod(int(x), 256) not in excluded]
    if sum(partitions.values()) > len(seeds):
        raise ValueError("not enough distinct reset seeds for disjoint partitions")
    weights = np.asarray([c["weight"] for c in conditions], dtype=np.float64)
    weights /= weights.sum()
    catalog = []
    for partition, count in partitions.items():
        exact = count * weights
        quotas = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - quotas), kind="stable")[:count - int(quotas.sum())]:
            quotas[i] += 1
        schedule = []
        for i, quota in enumerate(quotas):
            for j in range(int(quota)):
                c = conditions[i]
                schedule.append(dict(level=int(c["level"]), speed=int(c["speed"]),
                                     members=list(matchups[(j+i) % len(matchups)])))
        rng.shuffle(schedule)
        for assignment in schedule:
            index = len(catalog)
            catalog.append(dict(index=index, partition=partition, reset_seed=list(seeds[index]),
                                frame_counter_base=int(rng.integers(0, 256)), **assignment))
    return catalog


class RetainedPositions:
    """Bound memory independently of game length and retain early constructions."""
    def __init__(self, game_id, per_stratum=8):
        self.game_id = game_id
        self.per_stratum = per_stratum
        self.counts = [0, 0]
        self.first = [[], []]
        self.last = [deque(maxlen=16), deque(maxlen=16)]
        self.reservoir = defaultdict(list)

    def add(self, state, side, belief, action, event, speed_ups):
        identity = dict(checkpoint=hashlib.sha256(state.privileged.engine_checkpoint).hexdigest(),
                        root_side=side)
        row_id = hashlib.sha256(canonical_json(identity)).hexdigest()
        rank = hashlib.sha256(f"{self.game_id}:{row_id}".encode()).hexdigest()
        tactical = _tactical_stratum(state, side)
        row = dict(id=row_id, root_side=side, own_placement_index=self.counts[side],
                   decision_index=event, observed_action=action, tactical_stratum=tactical,
                   speed_ups=speed_ups, _rank=rank, _state=state, _belief=belief)
        self.counts[side] += 1
        if len(self.first[side]) < 4:
            self.first[side].append(row)
        self.last[side].append(row)
        reservoir = self.reservoir[side, tactical]
        reservoir.append(row)
        reservoir.sort(key=lambda r: r["_rank"])
        del reservoir[self.per_stratum:]

    def select(self, limit, outcomes):
        candidates = {}
        for values in (*self.first, *self.last, *self.reservoir.values()):
            for row in values:
                candidates[row["id"]] = dict(row)
        for row in candidates.values():
            total = self.counts[row["root_side"]]
            row["total_own_placements"] = total
            bin_index = min(2, row["own_placement_index"] * 3 // max(1, total))
            row["temporal_bin"] = ("opening", "middle", "late")[bin_index]
            row["sampling_reason"] = "temporal-tactical-reservoir"
        selected = []
        predecessors = []
        for side in (0, 1):
            if outcomes[side] != "loss":
                continue
            for offset in (4, 8, 16):
                if len(self.last[side]) >= offset:
                    row = candidates[self.last[side][-offset]["id"]]
                    predecessors.append({**row, "placements_before_failure": offset,
                                         "sampling_reason": "failure-predecessor"})
        if predecessors:
            index = int(self.game_id[:8], 16) % len(predecessors)
            selected.append(predecessors[index])
            candidates.pop(selected[0]["id"])
        phase_counts, tactical_counts, side_counts = Counter(), Counter(), Counter()
        for row in selected:
            phase_counts[row["temporal_bin"]] += 1
            tactical_counts[row["tactical_stratum"]] += 1
            side_counts[row["root_side"]] += 1
        while candidates and len(selected) < limit:
            row = min(candidates.values(), key=lambda r: (
                phase_counts[r["temporal_bin"]], tactical_counts[r["tactical_stratum"]],
                side_counts[r["root_side"]], r["_rank"]))
            selected.append(row)
            candidates.pop(row["id"])
            phase_counts[row["temporal_bin"]] += 1
            tactical_counts[row["tactical_stratum"]] += 1
            side_counts[row["root_side"]] += 1
        return selected


def _start_game(runner, spec, per_stratum):
    level, speed = spec["level"], spec["speed"]
    runner.reset(None, [build_vs_reset_spec(level=(level, level), speed_setting=(speed, speed),
        rng_override=True, rng_state=tuple(spec["reset_seed"]),
        frame_counter_base=spec["frame_counter_base"])])
    state = capture_native_state(runner, level=level, speed_setting=speed,
        viruses_initial=(min(84, 4*(level+1)),)*2, causal_public=True)
    game_id = hashlib.sha256(runner.snapshot(0)).hexdigest()
    belief = PillReserveBelief.from_initial_board(level=level,
                                                board=state.privileged.public.sides[0].board)
    return dict(runner=runner, spec=spec, state=state, belief=belief, game_id=game_id, events=0,
                retained=RetainedPositions(game_id, per_stratum))


def _advance_game(slot):
    runner = slot["runner"]
    runner.step_strict(slot["actions"])
    if np.any(runner.buffers.invalid_action >= 0):
        raise RuntimeError("source collector rejected a complete-frontier action")
    old = slot["state"]
    slot["state"] = capture_native_state(runner, level=old.level, speed_setting=old.speed_setting,
        viruses_initial=old.viruses_initial, previous=old, causal_public=True)
    slot["events"] += 1


def _finish_game(slot, states_per_game):
    state, spec = slot["state"], slot["spec"]
    outcomes = [_OUTCOME_NAME.get(x) for x in state.privileged.terminal_outcome]
    natural = all(x is not None for x in outcomes)
    if not natural:
        outcomes = [None, None]
    rows = []
    for selected in slot["retained"].select(states_per_game, outcomes):
        selected = dict(selected)
        root, belief = selected.pop("_state"), selected.pop("_belief")
        selected.pop("_rank")
        side = selected["root_side"]
        row = state_to_payload(root)
        legal = root.legal_actions_by_side[side]
        row.update(selected, game_id=slot["game_id"], game_index=spec["index"],
                   source_partition=spec["partition"], reset_seed=spec["reset_seed"],
                   source_members=spec["members"], level=spec["level"], speed=spec["speed"],
                   candidate_count=len(legal), candidate_count_bin=_candidate_bin(len(legal)),
                   clock_skew_bin=min(abs(root.privileged.pair_clocks[0]-root.privileged.pair_clocks[1])//30, 4),
                   reserve_belief=belief.to_dict(), reserve_seed_count=belief.seed_count,
                   rollout_policy="frozen-public-core-argmax", execution="native-smdp-v1",
                   natural_outcome_available=natural, outcome=outcomes[side])
        rows.append(row)
    return dict(schema="drmc-public-quality-source-game-v1", spec=spec, game_id=slot["game_id"],
                natural_outcome_available=natural, outcomes=outcomes, rows=rows,
                pair_events=slot["events"], policy_decisions=sum(slot["retained"].counts),
                console_frames=state.privileged.public.frame_id,
                public_observation_schema=CAUSAL_PUBLIC_SCHEMA)


def collect_games(catalog, actors, *, batch_size, max_events, states_per_game,
                  native_workers=1, per_stratum=8, on_game, progress=None, metrics=None):
    """Keep independent games in flight; actor batches never mix private inputs."""
    if min(batch_size, max_events, states_per_game, native_workers, per_stratum) < 1:
        raise ValueError("collection limits must be positive")
    executor = _native_executor(min(batch_size, native_workers, 32)) if native_workers > 1 else None
    pending = iter(catalog)
    runners, slots = [], []
    measured, batch_rows = Counter(), Counter()
    started = last_report = time.perf_counter()

    def fill():
        spec = next(pending, None)
        if spec is None:
            return None
        # Native round reset deliberately retains attack-color history, like
        # the cartridge. Independent source games require a cold pair, so
        # their physics and snapshot identity cannot depend on slot ordering.
        runner = DrMarioVsPoolRunner(num_pairs=1)
        runners.append(runner)
        return _start_game(runner, spec, per_stratum)

    try:
        for _ in range(min(batch_size, len(catalog))):
            slots.append(fill())
        while slots:
            measured["scheduler_iterations"] += 1
            measured["live_slot_iterations"] += len(slots)
            requests = defaultdict(list)
            for i, slot in enumerate(slots):
                state = slot["state"]
                slot["belief"] = _condition_visible_reserve(slot["belief"], slot["runner"], state=state)
                slot["actions"] = np.full(2, -2, np.int32)
                for side, need in enumerate(state.privileged.need_action):
                    if not need:
                        continue
                    if state.legal_actions_by_side[side]:
                        requests[slot["spec"]["members"][side]].append((i, side))
                    else:
                        slot["actions"][side] = -1
            before = time.perf_counter()
            for member, destinations in requests.items():
                predictions = actors[member].infer_batch([(slots[i]["state"], side) for i, side in destinations])
                batch_rows[len(destinations)] += 1
                for (i, side), (probability, _) in zip(destinations, predictions, strict=True):
                    slot = slots[i]
                    legal = slot["state"].legal_actions_by_side[side]
                    action = max(legal, key=lambda a: probability.get(a, 1e-8))
                    slot["actions"][side] = action
                    slot["retained"].add(slot["state"], side, slot["belief"], int(action),
                        slot["events"], min(int(slot["runner"].buffers.spawn_id[side])//10, 0x31))
                    measured["policy_decisions"] += 1
            measured["inference_seconds"] += time.perf_counter()-before
            before = time.perf_counter()
            if executor is None:
                for slot in slots:
                    _advance_game(slot)
            else:
                # Drain every future before closing handles, including on error.
                from concurrent.futures import wait
                futures = [executor.submit(_advance_game, slot) for slot in slots]
                wait(futures)
                for future in futures:
                    future.result()
            measured["native_seconds"] += time.perf_counter()-before
            active = []
            for slot in slots:
                runner = slot["runner"]
                done = (runner.buffers.terminated[0] or runner.buffers.truncated[0]
                        or slot["events"] >= max_events)
                if done:
                    result = _finish_game(slot, states_per_game)
                    on_game(result)
                    measured["completed_games"] += 1
                    measured["natural_games"] += int(result["natural_outcome_available"])
                    runner.close()
                    runners.remove(runner)
                    slot = fill()
                if slot is not None:
                    active.append(slot)
            slots = active
            now = time.perf_counter()
            if progress is not None and (now-last_report >= 5 or not slots):
                progress(dict(measured), len(slots))
                last_report = now
    finally:
        for runner in runners:
            runner.close()
        if metrics is not None:
            metrics.update(measured, wall_seconds=time.perf_counter()-started,
                           inference_batch_rows=dict(sorted(batch_rows.items())))


def game_path(output, spec):
    return output / "games" / f"{spec['index']:06d}.jsonl.gz"


def read_game(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def export_partitions(output, catalog, completed, identities):
    manifests = {}
    for partition in dict.fromkeys(s["partition"] for s in catalog):
        rows, game_ids, natural, censored = [], [], 0, 0
        for spec in catalog:
            if spec["partition"] != partition or spec["index"] not in completed:
                continue
            game = read_game(game_path(output, spec))
            rows.extend(game["rows"])
            game_ids.append(game["game_id"])
            natural += int(game["natural_outcome_available"])
            censored += int(not game["natural_outcome_available"])
        path = output / f"{partition}.jsonl.gz"
        _atomic_gzip_jsonl(path, rows)
        manifest = dict(schema="drmc-public-quality-bank-v1", partition=partition,
            artifact=str(path), sha256=sha256_file(path), states=len(rows), games=len(game_ids),
            natural_terminal_games=natural, censored_games=censored,
            member_sha256=identities, public_observation_schema=CAUSAL_PUBLIC_SCHEMA,
            execution="native-smdp-v1", chance_model=CHANCE_MODEL_ID,
            per_game_selection="bounded-temporal-tactical-reservoir-v1",
            temporal_counts=dict(Counter(r["temporal_bin"] for r in rows)),
            tactical_counts=dict(Counter(r["tactical_stratum"] for r in rows)),
            level_counts=dict(Counter(r["level"] for r in rows)),
            game_ids=game_ids, product_gates_passed=False)
        dump(Path(str(path)+".manifest.json"), manifest)
        manifests[partition] = {k: manifest[k] for k in
                               ("states", "games", "natural_terminal_games", "censored_games", "sha256")}
    return manifests


def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    catalog = game_catalog(config)
    identities = {name: sha256_file(Path(path)) for name, path in config["members"].items()}
    if len(set(identities.values())) != len(identities):
        raise ValueError("duplicate checkpoints do not add source-policy diversity")
    contract = dict(schema="drmc-public-quality-bank-job-v1", config=config, catalog=catalog,
                    member_sha256=identities, native_sha256=sha256_file(resolve_library_path()))
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("resume changed frozen source collection contract")
    if not contract_path.exists() and (output / "games").exists():
        raise ValueError("existing source games lack a frozen collection contract")
    dump(contract_path, contract)
    (output / "games").mkdir(exist_ok=True)
    completed, game_ids, measured = set(), set(), Counter()
    for spec in catalog:
        path = game_path(output, spec)
        if not path.exists():
            continue
        game = read_game(path)
        if (game["spec"] != spec or game["schema"] != "drmc-public-quality-source-game-v1"
                or game["public_observation_schema"] != CAUSAL_PUBLIC_SCHEMA
                or game["game_id"] in game_ids
                or any(row["game_id"] != game["game_id"] for row in game["rows"])):
            raise ValueError("committed source game has an inconsistent identity")
        completed.add(spec["index"])
        game_ids.add(game["game_id"])
        for key in ("pair_events", "policy_decisions", "console_frames"):
            measured[key] += game[key]
        measured["natural_games"] += int(game["natural_outcome_available"])
    previous = json.loads((output / "progress.json").read_text()) if (output / "progress.json").exists() else {}
    elapsed_before = float(previous.get("elapsed_seconds", 0))
    started = time.monotonic()
    progress = dict(schema="drmc-public-quality-bank-job-v1", status="Running", target_games=len(catalog), execution="native-smdp-v1",
                    product_gates_passed=False, batch_size=config.get("batch_size", 32))

    def report(**values):
        progress.update(values, games=len(completed), **dict(measured),
                        censored_games=len(completed)-measured["natural_games"],
                        elapsed_seconds=elapsed_before+time.monotonic()-started,
                        updated_at=datetime.now(UTC).isoformat())
        dump(output / "progress.json", progress)

    def receive(game):
        if game["game_id"] in game_ids:
            raise ValueError("two distinct reset seeds produced a duplicate initial game")
        _atomic_gzip_jsonl(game_path(output, game["spec"]), [game])
        completed.add(game["spec"]["index"])
        game_ids.add(game["game_id"])
        for key in ("pair_events", "policy_decisions", "console_frames"):
            measured[key] += game[key]
        measured["natural_games"] += int(game["natural_outcome_available"])
        report()

    report(phase="loading")
    try:
        remaining = [s for s in catalog if s["index"] not in completed]
        if remaining:
            actors = {name: PublicPolicyContinuation(Path(path), device=config.get("device", "cuda"))
                      for name, path in config["members"].items()}
            # A frame-trained history/motor core needs a matching controller
            # collector. Do not silently fabricate its missing execution inputs.
            if any(actor.policy.aux_spec != "zero_v1_vs" for actor in actors.values()):
                raise ValueError("native source bank requires legacy public input schemas; use controller replay for motor cores")
            metrics = {}
            report(phase="collecting")
            collect_games(remaining, actors, batch_size=config.get("batch_size", 32),
                max_events=config.get("max_events", 4096), states_per_game=config.get("states_per_game", 8),
                native_workers=config.get("native_workers", 1), per_stratum=config.get("reservoir_per_stratum", 8),
                on_game=receive, metrics=metrics,
                progress=lambda activity, live: report(activity={**activity, "live_games": live}))
            report(collection_metrics=metrics)
        if len(completed) != len(catalog):
            raise RuntimeError("source collection returned with unfinished games")
        report(phase="exporting")
        report(status="Complete", phase="complete", partitions=export_partitions(output, catalog, completed, identities))
    except BaseException as error:
        report(status="Failed", error=str(error))
        export_partitions(output, catalog, completed, identities)
        raise
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
