"""Initialise the afterstate core by imitating strong human players.

Arm B of the Stronger-3 afterstate experiment: the same
``drmc-afterstate-core-v1`` network and live ``public_pair_context_v3``
contract as the distilled arm, but trained on human-corpus placements
instead of the champion's choices.

``build`` streams an immutable corpus release game by game. For every
selected placement it reconstructs the causal public pair view a live actor
would have had at that spawn (``drmc_rl.human.corpus_public_state``), draws
one training pace uniformly among those whose motor frontier (after the live
``max(4, reaction)`` decision delay) contains the human's lock, plans that
exact frontier with the native planner, encodes the public context with that
pace's execution fields and stores exact afterstate deltas and facts. The
parts share the distillation dataset layout, so the same batcher rebuilds
model inputs.

Selection prefers strong players: rows need a WHR-C rating of at least
``--min-rating`` on the placement day and are kept with probability
proportional to ``exp((rating - pivot) / scale)``, with any single player's
share of the kept weight capped. Splits follow the release contracts: the
stable 80/10/10 replay split and 20 player folds; ``--holdout-folds`` are
never trained on and report held-out-player accuracy.

``train`` fits the policy to the human choice (all candidates whose settled
bottle equals the chosen one share the target, so same-colour rotation
aliases are not penalised) and the 51-atom value head to the game outcome,
from scratch: copying the V3 human model's 256x6 residual trunk (its
tile-embedding stem mapped onto the semantic planes by least squares, 72%
relative residual) learned no faster than a fresh trunk in a matched
400-step smoke run, so it is not used.
``evaluate`` reports held-out imitation accuracy and agreement with the
champion on a distillation part it labelled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import time

import numpy as np

DATASET_SCHEMA = "drmc-afterstate-human-imitation-v1"
PACES = ("sloth", "relaxed", "normal", "fast", "top_humans", "super_human", "frame_perfect")
SPLITS = ("train", "validation", "test", "heldout_players")
COMPUTE_FRAMES = 4  # the live reactive compute tier (arena and outcome PPO)
DECISION_COLUMNS = (
    "decision_id", "game_id", "day", "player", "player_slot", "player_fold", "random_split", "won",
    "spawn_frame", "lock_frame", "tau_frames", "field", "opp_field", "pill_left", "pill_right",
    "preview_left", "preview_right", "speed", "speed_ups", "frame_counter", "held_at_spawn",
    "held_before_spawn", "horizontal_velocity", "speed_counter", "lock_x", "lock_y_top",
    "lock_rotation", "lock_repaired", "input_frames", "input_rle_u16_u8",
)


def _stable_unit(text: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{text}".encode(), digest_size=8, person=b"drmc-imi").digest()
    return int.from_bytes(digest, "little") / 2.0**64


def _stable_key(text: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"))


def _split_of(row_split: str, fold: int, holdout_folds) -> int:
    if int(fold) in holdout_folds:
        return 3
    return {"train": 0, "validation": 1, "test": 2}[row_split]


# ---------------------------------------------------------------- selection
_SELECT_COLUMNS = ["decision_id", "player", "day", "random_split", "player_fold", "speed", "lock_repaired", "lock_x"]


class Selection:
    """Rating-weighted, player-capped, stable Bernoulli decision sample per split.

    Pass 1 fits one scale per split (and a factor per over-represented player)
    so the expected kept count meets the target; any later pass reapplies the
    identical rule to a row from its decision id alone.
    """

    def __init__(self, corpus, args, rule=None):
        self.corpus, self.args = corpus, args
        self.holdout = frozenset(args.holdout_folds)
        self.ratings = {}
        self.rule = rule

    def rating(self, player, day):
        key = (player, int(day))
        value = self.ratings.get(key)
        if value is None:
            value = self.ratings[key] = self.corpus.rating_at(*key)[0] or float("nan")
        return value

    def weight(self, data, i):
        if data["speed"][i] != 2 or data["lock_repaired"][i] or data["lock_x"][i] is None:
            return None
        rating = self.rating(data["player"][i], data["day"][i])
        if not rating >= self.args.min_rating:
            return None
        return rating, math.exp((rating - self.args.rating_pivot) / self.args.rating_scale)

    def fit(self, months):
        args = self.args
        weights, splits, players, units = [], [], [], []
        scanned = 0
        for month in months:
            for batch in self.corpus.batches("decisions", columns=_SELECT_COLUMNS, months=[month], batch_size=65536):
                data = batch.to_pydict()
                scanned += len(data["decision_id"])
                for i, decision in enumerate(data["decision_id"]):
                    got = self.weight(data, i)
                    if got is None:
                        continue
                    weights.append(got[1])
                    splits.append(_split_of(data["random_split"][i], data["player_fold"][i], self.holdout))
                    players.append(data["player"][i])
                    units.append(_stable_unit(decision, args.seed))
        weights = np.asarray(weights)
        splits = np.asarray(splits, dtype=np.int8)
        players = np.asarray(players, dtype=object)
        units = np.asarray(units)
        targets = {0: args.train_rows, 1: args.eval_rows, 2: args.eval_rows, 3: args.eval_rows}
        rule, report = {}, {}
        for split, target in targets.items():
            rows = np.flatnonzero(splits == split)
            if not len(rows):
                continue
            w = weights[rows]
            names = players[rows]
            factor = {}
            for _ in range(12):
                scaled = w * np.asarray([factor.get(p, 1.0) for p in names])
                total = scaled.sum()
                share = defaultdict(float)
                for p, value in zip(names, scaled):
                    share[p] += value
                over = {p: v for p, v in share.items() if v > args.player_cap * total * 1.0001}
                if not over:
                    break
                for p, v in over.items():
                    factor[p] = factor.get(p, 1.0) * args.player_cap * total / v
            scaled = w * np.asarray([factor.get(p, 1.0) for p in names])
            lo, hi = 0.0, 1e9
            for _ in range(100):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if np.minimum(1.0, mid * scaled).sum() < target else (lo, mid)
            kept = units[rows] < np.minimum(1.0, hi * scaled)
            counts = Counter(names[kept])
            rule[str(split)] = dict(scale=hi, player_factor=factor)
            report[SPLITS[split]] = dict(eligible=int(len(rows)), selected=int(kept.sum()), players=len(counts),
                                         top_player_share=max(counts.values(), default=0) / max(1, int(kept.sum())))
        self.rule = rule
        return dict(scanned=scanned, eligible=int(len(weights)), splits=report)

    def keep(self, data, i):
        """(rating, split) when row ``i`` of a column dict is selected, else None."""
        got = self.weight(data, i)
        if got is None:
            return None
        rating, weight = got
        split = _split_of(data["random_split"][i], data["player_fold"][i], self.holdout)
        rule = self.rule.get(str(split))
        if rule is None:
            return None
        probability = min(1.0, rule["scale"] * weight * rule["player_factor"].get(data["player"][i], 1.0))
        return (rating, split) if _stable_unit(data["decision_id"][i], self.args.seed) < probability else None


def corpus_months(corpus, months=None):
    found = sorted({(e.path.split("year=")[1][:4] + "-" + e.path.split("month=")[1][:2])
                    for e in corpus.files("decisions")})
    return [m for m in found if not months or m in months]


# ---------------------------------------------------------------- workers
_PLANNER = None


def _planner():
    global _PLANNER
    if _PLANNER is None:
        from drmc_rl.planning.native_reach import NativeReachabilityRunner

        _PLANNER = NativeReachabilityRunner()
    return _PLANNER


def _frontier(placement, pace, planner):
    from drmc_rl.game.observation import board_bytes_to_semantic_planes
    from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
    from drmc_rl.human.backend import plan_candidates
    from drmc_rl.human.corpus_public_state import spawn_frame_state

    row = placement.row
    start = spawn_frame_state(row)
    planes = board_bytes_to_semantic_planes(placement.board)
    state = {
        "board_planes": planes, "opponent_board_planes": planes,
        "pill": list(placement.pill), "preview": list(placement.preview),
        "speed": int(row["speed"]), "speed_ups": int(row["speed_ups"]),
        "public_context_schema": PUBLIC_CONTEXT_SCHEMA,
        "falling": {"x": 3, "y": 0, "rotation": 0, "speed_counter": start.speed_counter,
                    "horizontal_velocity": start.hor_velocity, "hold_dir": start.hold_dir.value,
                    "rotation_hold": start.rot_hold.value, "frame_parity": start.frame_parity},
    }
    delay = max(4, pace.reaction_frames)
    return plan_candidates(planner, state, delay, pace), delay


def _game_rows(task):
    """Worker: every selected placement of one chunk of games, as flat arrays."""
    from drmc_rl.execution.pace import resolve_pace
    from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA, PublicExecutionContext, encode_public_context
    from drmc_rl.human.backend import NoReachablePlacement
    from drmc_rl.human.corpus_public_state import CorpusGame
    from drmc_rl.planning.fast_reach import compute_speed_threshold
    from drmc_rl.search.public_policy import policy_request
    from tools.distill_afterstate_core import _plane_table

    games, selected, holdout, seed, check = task
    paces = [resolve_pace(p) for p in PACES]
    planner = _planner()
    table = _plane_table()
    out = defaultdict(list)
    reasons = Counter()
    for rows in games:
        try:
            game = CorpusGame(rows)
        except (ValueError, KeyError) as error:
            reasons[f"game:{type(error).__name__}"] += len([r for r in rows if r["decision_id"] in selected])
            continue
        checked = not check
        for side in game.sides.values():
            for p in side:
                row = p.row
                if row["decision_id"] not in selected:
                    continue
                if p.pose is None or p.action < 0:
                    reasons["invalid_lock"] += 1
                    continue
                rng = np.random.default_rng(int(_stable_unit(row["decision_id"], seed) * 2**62))
                frontiers = {}

                def feasible(index):
                    if index not in frontiers:
                        try:
                            candidate, delay = _frontier(p, paces[index], planner)
                            ok = candidate[-1][p.action] != 0xFFFF
                            frontiers[index] = (candidate, delay) if ok else None
                        except NoReachablePlacement:
                            frontiers[index] = None
                    return frontiers[index] is not None

                # Slower paces are strict subsets: find the slowest one that reaches the lock.
                if not feasible(len(paces) - 1):
                    reasons["unreachable_after_delay"] += 1
                    continue
                lo, hi = 0, len(paces) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    lo, hi = (lo, mid) if feasible(mid) else (mid + 1, hi)
                index = lo + int(rng.integers(len(paces) - lo))
                feasible(index)
                pace = paces[index]
                candidate, delay = frontiers[index]
                feasible_count = len(paces) - lo
                packed = candidate[8]
                actions = packed.actions[: packed.count].astype(np.int16)
                costs = packed.cost[: packed.count].astype(np.uint16)
                slot = np.flatnonzero(actions == p.action)
                public = game.public_state(p)
                execution = PublicExecutionContext(
                    reaction_frames=pace.reaction_frames, edge_interval=pace.edge_interval,
                    motion_interval=pace.motion_interval, max_buttons=pace.max_buttons,
                    gravity_frames=int(compute_speed_threshold(int(row["speed"]), int(row["speed_ups"]))) + 1,
                    speed_ups=int(row["speed_ups"]), decision_delay_frames=int(delay),
                    compute_frames=COMPUTE_FRAMES,
                )
                context = encode_public_context(public, p.side, execution)
                opponent_board = np.frombuffer(public.sides[1 - p.side].board, np.uint8)
                own_board = np.frombuffer(p.board, np.uint8)
                if not checked:
                    observation, _info = policy_request(public, p.side, actions.tolist(), costs.tolist(),
                                                        context_schema=PUBLIC_CONTEXT_SCHEMA, execution=execution)
                    rebuilt = np.concatenate((table[own_board].T, table[opponent_board].T)).reshape(16, 16, 8)
                    if not np.array_equal(rebuilt.astype(np.uint8), observation[:16].astype(np.uint8)):
                        raise ValueError("stored bottles do not rebuild the live policy observation")
                    checked = True
                out["own"].append(own_board)
                out["opp"].append(opponent_board)
                out["pill"].append(p.pill)
                out["preview"].append(p.preview)
                out["ctx"].append(context.astype(np.float16))
                out["counts"].append(len(actions))
                out["actions"].append(actions)
                out["costs"].append(costs)
                out["chosen"].append(int(slot[0]))
                out["won"].append(bool(row["won"]))
                out["pace"].append(PACES.index(pace.id))
                out["feasible_paces"].append(feasible_count)
                out["split"].append(_split_of(row["random_split"], row["player_fold"], holdout))
                out["player_fold"].append(int(row["player_fold"]))
                out["player_key"].append(_stable_key(str(row["player"])))
                out["game_key"].append(_stable_key(str(row["game_id"])))
                out["day"].append(int(row["day"]))
                out["speed_ups"].append(int(row["speed_ups"]))
                out["decision_index"].append(len(out["decision_index"]))
                out["rating"].append(float(row["_rating"]))
                reasons["kept"] += 1
    if not out:
        return None, reasons
    result = {
        "own": np.stack(out["own"]), "opp": np.stack(out["opp"]),
        "pill": np.asarray(out["pill"], np.int8), "preview": np.asarray(out["preview"], np.int8),
        "ctx": np.stack(out["ctx"]), "counts": np.asarray(out["counts"], np.int32),
        "actions": np.concatenate(out["actions"]), "costs": np.concatenate(out["costs"]),
        "chosen": np.asarray(out["chosen"], np.int16), "won": np.asarray(out["won"], np.int8),
        "pace": np.asarray(out["pace"], np.int8), "feasible_paces": np.asarray(out["feasible_paces"], np.int8),
        "split": np.asarray(out["split"], np.int8), "player_fold": np.asarray(out["player_fold"], np.int8),
        "player_key": np.asarray(out["player_key"], np.uint64), "game_key": np.asarray(out["game_key"], np.uint64),
        "day": np.asarray(out["day"], np.int32), "speed_ups": np.asarray(out["speed_ups"], np.int8),
        "rating": np.asarray(out["rating"], np.float32),
    }
    from tools.distill_afterstate_core import _afterstate_chunk

    offsets = np.concatenate(([0], np.cumsum(result["counts"]))).astype(np.int64)
    facts, delta_counts, cells, values = _afterstate_chunk(
        (result["own"], result["pill"].astype(np.int64), offsets, result["actions"]))
    result.update(facts=facts, delta_counts=delta_counts.astype(np.int32), delta_cells=cells, delta_values=values)
    return result, reasons


def _merge(chunks):
    data = {}
    for key in chunks[0]:
        data[key] = np.concatenate([c[key] for c in chunks])
    data["offsets"] = np.concatenate(([0], np.cumsum(data.pop("counts")))).astype(np.int64)
    data["delta_offsets"] = np.concatenate(([0], np.cumsum(data.pop("delta_counts")))).astype(np.int64)
    data["counts"] = np.diff(data["offsets"]).astype(np.int32)
    # distillation batcher compatibility: game identity as the grouping seed
    data["game_seed"] = (data["game_key"] % np.uint64(2**31)).astype(np.int64)
    return data


def build(args):
    import pyarrow as pa
    import pyarrow.compute as pc

    from drmc_rl.data.human_corpus import HumanCorpus

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = HumanCorpus(args.corpus_root, release=args.release)
    if args.release == "latest":
        raise SystemExit("use the immutable release id, not latest")
    started = time.monotonic()
    months = corpus_months(corpus, args.months)
    selection = Selection(corpus, args)
    selection_path = out / "selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text())
        selection.rule, selection_report = saved["rule"], saved["report"]
    else:
        selection_report = selection.fit(months)
        tmp = selection_path.with_suffix(".next")
        tmp.write_text(json.dumps(dict(report=selection_report, rule=selection.rule), indent=1))
        tmp.replace(selection_path)
    print(json.dumps(dict(selection=selection_report, seconds=round(time.monotonic() - started))), flush=True)
    holdout = frozenset(args.holdout_folds)
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(months=[])
    reasons = Counter(progress.get("reasons", {}))
    parts = list(progress.get("parts", []))
    pending = []
    part_index = int(progress.get("part_index", 0))

    def flush(final=False):
        nonlocal pending, part_index
        rows = sum(len(c["chosen"]) for c in pending)
        if not pending or (rows < args.rows_per_part and not final):
            return
        data = _merge(pending)
        path = out / f"part-{part_index:03d}.npz"
        tmp = out / f"part-{part_index:03d}.next.npz"
        np.savez(tmp, **data)
        tmp.replace(path)
        parts.append(path.name)
        print(json.dumps(dict(part=part_index, rows=int(len(data["chosen"])), candidates=int(data["offsets"][-1]),
                              seconds=round(time.monotonic() - started), reasons=dict(reasons))), flush=True)
        part_index += 1
        pending = []

    with get_context("spawn").Pool(args.workers) as pool:
        for month in months:
            if month in progress["months"]:
                continue
            table = corpus.dataset("decisions", months=[month]).to_table(columns=list(DECISION_COLUMNS))
            light = table.select(_SELECT_COLUMNS).to_pydict()
            wanted = {}
            for i in range(len(light["decision_id"])):
                got = selection.keep(light, i)
                if got is not None:
                    wanted[light["decision_id"][i]] = got[0]
            del light
            if not wanted:
                continue
            games_wanted = pc.unique(table.filter(pc.is_in(table["decision_id"], value_set=pa.array(sorted(wanted))))["game_id"])
            table = table.filter(pc.is_in(table["game_id"], value_set=games_wanted))
            table = table.sort_by([("game_id", "ascending"), ("spawn_frame", "ascending")])
            game_ids = table["game_id"].to_pylist()
            bounds = [0] + [i for i in range(1, len(game_ids)) if game_ids[i] != game_ids[i - 1]] + [len(game_ids)]
            spans = list(zip(bounds, bounds[1:]))
            per_slice = args.games_per_task * args.workers * 4
            tasks_done = 0
            for s0 in range(0, len(spans), per_slice):
                tasks, chunk = [], []
                for a, b in spans[s0:s0 + per_slice]:
                    rows = table.slice(a, b - a).to_pylist()
                    for row in rows:
                        if row["decision_id"] in wanted:
                            row["_rating"] = wanted[row["decision_id"]]
                    chunk.append(rows)
                    if len(chunk) >= args.games_per_task:
                        tasks.append(chunk)
                        chunk = []
                if chunk:
                    tasks.append(chunk)
                tasks = [(c, frozenset(r["decision_id"] for g in c for r in g if r["decision_id"] in wanted),
                          holdout, args.seed, i == 0) for i, c in enumerate(tasks)]
                for result, why in pool.imap(_game_rows, tasks, chunksize=1):
                    reasons.update(why)
                    if result is not None:
                        pending.append(result)
                        flush()
                tasks_done += len(tasks)
            del table
            flush(final=True)
            progress["months"].append(month)
            progress.update(parts=parts, reasons=dict(reasons), part_index=part_index)
            tmp = progress_path.with_suffix(".next")
            tmp.write_text(json.dumps(progress))
            tmp.replace(progress_path)
            print(json.dumps(dict(month=month, tasks=tasks_done, reasons=dict(reasons),
                                  seconds=round(time.monotonic() - started))), flush=True)
    flush(final=True)
    manifest = corpus.release_dir / "manifest.json"
    meta = dict(
        schema=DATASET_SCHEMA, created_at=datetime.now(UTC).isoformat(), corpus_release=corpus.release_id,
        corpus_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), months=args.months or "all",
        selection=selection_report, selection_scales=selection.rule, selection_rule=dict(
            min_rating=args.min_rating, pivot=args.rating_pivot, scale=args.rating_scale,
            player_cap=args.player_cap, train_rows=args.train_rows, eval_rows=args.eval_rows, seed=args.seed,
            speed="HI only", excluded="repaired or missing locks"),
        holdout_folds=sorted(holdout), splits=SPLITS, paces=PACES,
        pace_rule="uniform among paces whose max(4, reaction)-delayed frontier contains the human lock",
        compute_frames=COMPUTE_FRAMES, fact_offset=32, parts=parts, reasons=dict(reasons),
        public_view="drmc_rl.human.corpus_public_state (validated by tools.validate_corpus_public_state)",
        source_commit=os.popen("git rev-parse --short HEAD").read().strip(),
    )
    (out / "dataset.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(dict(done=True, parts=len(parts), reasons=dict(reasons))), flush=True)


# ---------------------------------------------------------------- training
class HostData:
    """Dataset parts in host memory; batches are built on the host and moved to the device."""

    def __init__(self, directory, keys_extra=("chosen", "won", "split", "player_fold", "rating")):
        import torch

        from tools.distill_afterstate_core import _Batcher, load_parts

        self.meta, data, self.host = load_parts(directory, "cpu")
        self.extra = {k: data[k].numpy() for k in keys_extra if k in data}
        self.pace = data["pace"].numpy()
        self.batcher = _Batcher(data, data["offsets"], "cpu")
        self.batcher.pace_np = self.pace
        self.rows = int(len(self.pace))
        self.torch = torch

    def batch(self, rows, device):
        inputs, aux, valid, (after, facts), index = self.batcher.inputs(np.asarray(rows), afterstate=True)
        move = lambda t: t.to(device, non_blocking=True)
        return (tuple(move(t) for t in inputs), move(aux), move(valid), (move(after), move(facts)))


def _targets(after, valid, chosen):
    """Uniform mass over every candidate whose settled bottle equals the chosen one."""
    import torch

    rows = torch.arange(after.shape[0], device=after.device)
    same = (after == after[rows, chosen].unsqueeze(1)).all(-1) & valid
    return same.float() / same.float().sum(-1, keepdim=True)


def _loss_terms(net, logits, value_logits, target, won):
    import torch

    logp = logits.float().log_softmax(-1)
    nll = -(target * logp.clamp_min(-1e4)).sum(-1)
    hit = (target.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1) > 0).float()
    outcome = torch.where(won > 0, 1.0, -1.0).float()
    value_ce = net.distributional_value_loss(value_logits.float(), outcome)
    return nll, hit, value_ce


def evaluate_imitation(net, data, rows, device, batch=256):
    import torch

    net.eval()
    sums = defaultdict(float)
    per_pace = defaultdict(lambda: [0, 0.0])
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        inputs, aux, valid, after = data.batch(part, device)
        chosen = torch.as_tensor(data.extra["chosen"][part].astype(np.int64), device=device)
        won = torch.as_tensor(data.extra["won"][part].astype(np.int64), device=device)
        with torch.inference_mode(), _autocast(device):
            logits, _v, extra = net(*inputs, aux=aux, afterstate=after, return_aux=True)
        target = _targets(after[0], valid, chosen)
        nll, hit, value_ce = _loss_terms(net, logits, extra["value_logits"], target, won)
        strict = (logits.argmax(-1) == chosen).float()
        sums["nll"] += float(nll.sum())
        sums["top1"] += float(hit.sum())
        sums["strict_top1"] += float(strict.sum())
        sums["value_ce"] += float(value_ce) * len(part)
        sums["candidates"] += float(valid.sum())
        for p, h in zip(data.pace[part], hit.cpu().numpy()):
            per_pace[PACES[int(p)]][0] += 1
            per_pace[PACES[int(p)]][1] += float(h)
    n = max(1, len(rows))
    result = {k: v / n for k, v in sums.items()}
    result["rows"] = int(len(rows))
    result["per_pace_top1"] = {p: v[1] / max(1, v[0]) for p, v in per_pace.items()}
    return result


def evaluate_champion(net, champion, device, batch=256, limit=None):
    """Agreement with the champion's argmax on live self-play states it labelled."""
    import torch

    net.eval()
    agree = strict = 0.0
    per_pace = defaultdict(lambda: [0, 0.0])
    b = champion.batcher
    rows = np.arange(champion.rows)
    if limit and limit < len(rows):
        rows = np.sort(np.random.default_rng(7).choice(rows, int(limit), replace=False))
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        inputs, aux, valid, (after, facts), index = b.inputs(part, afterstate=True)
        teacher = torch.where(valid, b.teacher_logits[index], torch.full(valid.shape, -1e9))
        with torch.inference_mode(), _autocast(device):
            logits, _v = net(*(t.to(device) for t in inputs), aux=aux.to(device),
                             afterstate=(after.to(device), facts.to(device)))
        student = logits.argmax(-1).cpu()
        champion_arg = teacher.argmax(-1)
        rows_t = torch.arange(len(part))
        same_after = (after[rows_t, student] == after[rows_t, champion_arg]).all(-1).float()
        exact = (student == champion_arg).float()
        agree += float(same_after.sum())
        strict += float(exact.sum())
        for p, h in zip(champion.pace[part], same_after.numpy()):
            per_pace[PACES[int(p)]][0] += 1
            per_pace[PACES[int(p)]][1] += float(h)
    n = max(1, len(rows))
    return dict(states=int(len(rows)), agree_same_afterstate=agree / n, agree_strict=strict / n,
                per_pace_agree=dict(sorted(((p, v[1] / max(1, v[0])) for p, v in per_pace.items()),
                                           key=lambda kv: PACES.index(kv[0]))))


def _autocast(device):
    import contextlib

    import torch

    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train(args):
    import torch

    from drmc_rl.models.policy.afterstate_core import afterstate_core_config
    from tools.eval_policy import _build_net_from_cfg

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = HostData(args.dataset)
    split = data.extra["split"]
    train_rows = np.flatnonzero(split == 0)
    eval_sets = {name: np.flatnonzero(split == i) for i, name in enumerate(SPLITS) if name != "train"}
    rng_eval = np.random.default_rng(1)
    eval_small = {k: (v if len(v) <= args.eval_rows else np.sort(rng_eval.choice(v, args.eval_rows, replace=False)))
                  for k, v in eval_sets.items() if k != "test" and len(v)}
    champion = None
    if args.champion_part:
        champion = HostData(args.champion_part, keys_extra=())
    cfg = afterstate_core_config()
    net, _aux, _ = _build_net_from_cfg(cfg, 20, args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        net.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
    params = sum(p.numel() for p in net.parameters())
    identity = dict(schema="drmc-afterstate-human-imitation-run-v1", dataset=str(args.dataset),
                    dataset_meta=data.meta, train_rows=int(len(train_rows)),
                    eval_rows={k: int(len(v)) for k, v in eval_sets.items()}, parameters=params,
                    champion_part=str(args.champion_part) if args.champion_part else None,
                    args=vars(args), source_commit=os.popen("git rev-parse --short HEAD").read().strip())
    (out / "run.json").write_text(json.dumps(identity, indent=1, default=str) + "\n")
    print(json.dumps(dict(parameters=params, train=len(train_rows))), flush=True)
    log = (out / "log.jsonl").open("a")
    rng = np.random.default_rng(args.seed + start_step)
    began = time.monotonic()

    def save(name, step, metrics):
        payload = dict(schema="drmc-afterstate-core-human-imitation-v1", cfg=cfg,
                       state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                       optimizer=optimizer.state_dict(), step=step, metrics=metrics,
                       observation_schema="public_pair_context_v3", imitation=identity)
        tmp = out / (name + ".next")
        torch.save(payload, tmp)
        tmp.replace(out / name)

    def evaluate(step):
        metrics = dict(step=step, seconds=time.monotonic() - began)
        for name, rows in eval_small.items():
            metrics[name] = evaluate_imitation(net, data, rows, args.device)
        if champion is not None:
            metrics["champion"] = evaluate_champion(net, champion, args.device, limit=args.champion_rows)
        return metrics

    metrics = None
    best = None
    for step in range(start_step + 1, args.steps + 1):
        net.train()
        lr = args.lr * min(1.0, step / args.warmup) * (
            0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))) * 0.95 + 0.05)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rows = np.sort(rng.choice(train_rows, args.batch, replace=False))
        inputs, aux, valid, after = data.batch(rows, args.device)
        chosen = torch.as_tensor(data.extra["chosen"][rows].astype(np.int64), device=args.device)
        won = torch.as_tensor(data.extra["won"][rows].astype(np.int64), device=args.device)
        with _autocast(args.device):
            logits, _value, extra = net(*inputs, aux=aux, afterstate=after, return_aux=True)
        target = _targets(after[0], valid, chosen)
        nll, hit, value_ce = _loss_terms(net, logits, extra["value_logits"], target, won)
        loss = nll.mean() + args.value_coefficient * value_ce
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        if step % args.log_every == 0:
            row = dict(step=step, lr=lr, loss=loss.item(), nll=nll.mean().item(), top1=hit.mean().item(),
                       value_ce=value_ce.item(), grad=float(grad), seconds=time.monotonic() - began,
                       rows_per_second=step * args.batch / max(1e-9, time.monotonic() - began))
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(step)
            log.write(json.dumps(metrics) + "\n"); log.flush()
            print(json.dumps(metrics), flush=True)
            save("imitation-latest.pt", step, metrics)
            score = metrics.get("validation", {}).get("nll")
            if score is not None and (best is None or score < best):
                best = score
                save("imitation-best.pt", step, metrics)
    final = dict(metrics or {}, test=evaluate_imitation(net, data, eval_sets["test"], args.device)
                 if len(eval_sets.get("test", [])) else None,
                 heldout_players_full=evaluate_imitation(net, data, eval_sets["heldout_players"], args.device)
                 if len(eval_sets.get("heldout_players", [])) else None)
    save("imitation-final.pt", args.steps, final)
    inference = torch.load(out / "imitation-final.pt", map_location="cpu", weights_only=False)
    inference.pop("optimizer")
    torch.save(inference, out / "imitation-final-inference.pt")
    (out / "final.json").write_text(json.dumps(final, indent=1) + "\n")
    print(json.dumps(dict(done=True, final=final)), flush=True)


def evaluate_command(args):
    import torch

    from tools.eval_policy import _build_net_from_cfg

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net, _aux, _ = _build_net_from_cfg(payload["cfg"], 20, args.device)
    net.load_state_dict(payload["state_dict"])
    report = dict(checkpoint=str(args.checkpoint),
                  sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    if args.dataset:
        data = HostData(args.dataset)
        split = data.extra["split"]
        for i, name in enumerate(SPLITS):
            rows = np.flatnonzero(split == i)
            if name in args.splits and len(rows):
                report[name] = evaluate_imitation(net, data, rows, args.device)
    if args.champion_part:
        report["champion"] = evaluate_champion(net, HostData(args.champion_part, keys_extra=()), args.device)
    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus-root", required=True)
    b.add_argument("--release", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--months", nargs="*")
    b.add_argument("--min-rating", type=float, default=1900.0)
    b.add_argument("--rating-pivot", type=float, default=2000.0)
    b.add_argument("--rating-scale", type=float, default=150.0)
    b.add_argument("--player-cap", type=float, default=0.06)
    b.add_argument("--train-rows", type=int, default=2_400_000)
    b.add_argument("--eval-rows", type=int, default=60_000)
    b.add_argument("--holdout-folds", type=int, nargs="*", default=[0, 1])
    b.add_argument("--seed", type=int, default=20260924)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--games-per-task", type=int, default=16)
    b.add_argument("--rows-per-part", type=int, default=200_000)
    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--value-coefficient", type=float, default=0.5)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--eval-rows", type=int, default=20000)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=20260924)
    t.add_argument("--champion-part", help="directory with a champion-labelled distillation part (dataset.json)")
    t.add_argument("--champion-rows", type=int)
    t.add_argument("--resume")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--dataset")
    e.add_argument("--splits", nargs="*", default=["validation", "heldout_players"])
    e.add_argument("--champion-part")
    e.add_argument("--device", default="cpu")
    e.add_argument("--output")
    args = parser.parse_args()
    {"build": build, "train": train, "evaluate": evaluate_command}[args.command](args)


if __name__ == "__main__":
    main()
