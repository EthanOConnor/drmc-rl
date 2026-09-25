"""Start banks of human big-clear setups (StartBank npz schema plus big-clear metadata).

Input: ``tools.mine_big_clears`` output (every human clear scoring at least the
T1 bar, with context and path facts). A *target* is one such clear; a *start*
is the acting side's spawn ``L`` placements earlier (``--lookbacks``), taken only
when the human's own ``L`` placements from there settled exactly, with no
garbage arriving (``clean_run >= L``), so the big clear is reachable from the
start under the same pills by the human's own line. Each start is rebuilt from
the corpus rows (``drmc_rl.human.corpus_public_state.CorpusGame``) and the
path is replayed exactly (every intermediate bottle and the final clear's
features must reproduce) before it is kept.

Rows per start (``kind``):

* ``real`` (0): the setup side against the opponent's causal public state at
  that spawn (settled bottle; a mid-resolution opponent is shown settled with
  its next pill), both on the source game's pill reserve;
* ``mirror`` (1): the setup in both bottles, a race to realise it.

Each row stores the source game's arena ``seed`` (-1 unless every pill and
preview of the side-game matched its reserve) with ``pill_counter`` = the
source reserve index, so a trainer can replay the human pills
(``start_mix.replay_share``) or play a fresh seed.

Splits: players are hashed into held-out players (``--holdout-players``), and
a share of the remaining games are held out (``--holdout-games``). ``training``
uses neither; ``benchmark`` uses only them and adds matched negative controls
(same held-out side-game, virus count within ``--control-virus-gap``, no T1+
clear by the human in the next ``--control-window`` placements, away from the
target's build-up).

Balancing (training): speed HI (weight 1) or MED (0.5), level band weights
(10-13: 0.5, 14+: 1); target sampling weight ``tier * type * level * speed /
sqrt(player targets)`` without replacement, then a hard cap of
``--player-cap`` of the targets per player.

  python -m tools.build_big_clear_bank select --mining DATA/mining --out DATA/bank/selection.json
  python -m tools.build_big_clear_bank extract --selection DATA/bank/selection.json \
      --remote mombox:fightcadeRatings/data/corpus/releases/RELEASE --out DATA/bank/starts.jsonl.gz
  python -m tools.build_big_clear_bank write --starts DATA/bank/starts.jsonl.gz --split training \
      --out DATA/bank/big-clear-train-v1.npz
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from drmc_rl.eval import big_clear as bc

BANK_SCHEMA = "drmc-big-clear-bank-v1"
KINDS = ("real", "mirror", "control")
LEVEL_WEIGHTS = {"<10": 0.0, "10-13": 0.5, "14-19": 1.0, "20+": 1.0}
TIER_WEIGHTS = {"T1": 1.0, "T2": 2.5, "T3": 6.0}
# Training resets play HI; MED setups are real boards but were built with more time.
SPEED_WEIGHTS = {2: 1.0, 1: 0.5}
TYPES = ("cascade", "multi_line", "long_line", "virus", "wide")
_CANONICAL_TO_NES = (1, 0, 2)


def _hash(text: str) -> float:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / float(1 << 48)


def split_of(player: str, game_id: str, args) -> str:
    if _hash("player:" + player) < args.holdout_players:
        return "heldout_player"
    if _hash("game:" + game_id) < args.holdout_games:
        return "heldout_game"
    return "train"


def clear_type(r) -> str:
    """Dominant showy feature of a clear (for balancing)."""
    if r["f_rounds"] >= 3:
        return "cascade"
    if r["f_max_round_lines"] >= 2 or r["f_cross"]:
        return "multi_line"
    if r["f_max_line"] >= 6:
        return "long_line"
    if r["f_viruses"] >= 5:
        return "virus"
    return "wide"


def level_band(level: int) -> str:
    return "20+" if level >= 20 else "14-19" if level >= 14 else "10-13" if level >= 10 else "<10"


# ----------------------------------------------------------------------------- select

def cmd_select(args) -> None:
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(Path(args.mining).glob("clears-*.parquet"))]
    clears = pd.concat(frames, ignore_index=True)
    t1 = bc.TIERS[0][1]
    clears = clears[clears.f_score >= t1].copy()
    clears["tier"] = clears.f_score.map(bc.tier)  # the current bars, not the mining-time labels
    clears["split"] = [split_of(p, g, args) for p, g in zip(clears.player, clears.game_id)]
    clears["type"] = [clear_type(r) for r in clears.to_dict("records")]
    clears["band"] = [level_band(int(v)) for v in clears.level]
    stats = dict(t1_clears=len(clears), players=int(clears.player.nunique()), by_split=clears.split.value_counts().to_dict(),
                 by_tier=clears.tier.value_counts().to_dict(), by_type=clears.type.value_counts().to_dict())
    eligible = clears[clears.speed.map(SPEED_WEIGHTS).fillna(0) > 0]
    eligible = eligible[(eligible.band.map(LEVEL_WEIGHTS) > 0) & (eligible.clean_run >= min(args.lookbacks))]
    stats["eligible"] = len(eligible)
    rng = np.random.default_rng(args.seed)

    train = eligible[eligible.split == "train"].copy()
    per_player = train.player.value_counts()
    type_share = train.type.value_counts(normalize=True)
    train["w"] = (train.tier.map(TIER_WEIGHTS) * train.band.map(LEVEL_WEIGHTS) * train.speed.map(SPEED_WEIGHTS)
                  * train.type.map(lambda t: min(3.0, (1 / len(TYPES)) / type_share[t]))
                  / np.sqrt(train.player.map(per_player)))
    keys = rng.random(len(train)) ** (1.0 / train.w.to_numpy())  # Efraimidis-Spirakis weighted sampling
    train = train.iloc[np.argsort(-keys)]
    cap = max(1, int(args.player_cap * args.targets))
    taken, counts = [], Counter()
    for r in train.itertuples():
        if counts[r.player] >= cap:
            continue
        counts[r.player] += 1
        taken.append(r.Index)
        if len(taken) >= args.targets:
            break
    train_sel = train.loc[taken]

    held = eligible[(eligible.split != "train") & (eligible.f_score >= bc.TIERS[1][1]) & (eligible.speed == 2)
                    & (eligible.level >= 14)
                    & (eligible.seed_ok) & (eligible.clean_run >= max(args.bench_lookbacks))]
    held = held.iloc[np.argsort(-(rng.random(len(held)) ** (1.0 / held.tier.map(TIER_WEIGHTS).to_numpy())))]
    # Half from held-out players (unseen people), half from held-out games of training players.
    bench, per_game, per_player_b = [], Counter(), Counter()
    for split in ("heldout_player", "heldout_game"):
        want = args.bench_targets // 2 if split == "heldout_player" else args.bench_targets - len(bench)
        taken = 0
        for r in held[held.split == split].itertuples():
            if per_game[r.game_id] >= 1 or per_player_b[r.player] >= args.bench_player_cap:
                continue
            per_game[r.game_id] += 1
            per_player_b[r.player] += 1
            bench.append(r.Index)
            taken += 1
            if taken >= want:
                break
    bench_sel = held.loc[bench]
    keep = ["month", "game_id", "slot", "ordinal", "spawn_frame", "player", "day", "level", "speed", "won", "seed", "seed_ok",
            "clean_run", "tier", "type", "band", "split", "f_score", "f_cells", "f_rounds", "f_viruses",
            "f_max_round_lines", "f_max_line", "f_garbage", "f_cross", "f_colors", "f_span_rows"]
    out = dict(schema=BANK_SCHEMA + "-selection", tiers=bc.TIERS, weights=bc.WEIGHTS,
               args={k: v for k, v in vars(args).items() if k != "func"}, stats=stats,
               training=dict(targets=len(train_sel), players=int(train_sel.player.nunique()),
                             eligible_players=int(train.player.nunique()),
                             speeds=train_sel.speed.value_counts().to_dict(),
                             tiers=train_sel.tier.value_counts().to_dict(), types=train_sel.type.value_counts().to_dict(),
                             bands=train_sel.band.value_counts().to_dict(),
                             top_player_share=float(train_sel.player.value_counts().iloc[0] / len(train_sel))),
               benchmark=dict(targets=len(bench_sel), players=int(bench_sel.player.nunique()),
                              tiers=bench_sel.tier.value_counts().to_dict(), types=bench_sel.type.value_counts().to_dict(),
                              splits=bench_sel.split.value_counts().to_dict()),
               rows=dict(training=train_sel[keep].to_dict("records"), benchmark=bench_sel[keep].to_dict("records")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    print(json.dumps({k: out[k] for k in ("stats", "training", "benchmark")}, indent=1, default=str))


# ---------------------------------------------------------------------------- extract

def _stream_games(args, months: dict[str, set[str]]):
    """Yield (game_id, rows) for the requested games, one month file at a time."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    host, root = args.remote.split(":", 1) if args.remote else (None, args.release)
    for month, games in sorted(months.items()):
        year, mm = month.split("-")
        relative = f"decisions/year={year}/month={mm}/part-00000.parquet"
        if host:
            local = scratch / f"{month}.parquet"
            subprocess.run(["rsync", "-a", f"{host}:{root}/{relative}", str(local)], check=True)
        else:
            local = Path(root) / relative
        table = pq.read_table(local)
        table = table.filter(pc.is_in(table.column("game_id"), value_set=__import__("pyarrow").array(sorted(games))))
        if host:
            local.unlink()
        by_game = defaultdict(list)
        for row in table.to_pylist():
            by_game[row["game_id"]].append(row)
        for gid, rows in by_game.items():
            yield gid, rows


def _settled_opponent(game, side: int, frame: int):
    """Opponent's causal public state at ``frame``, as a settled checkpoint side."""
    placements = game.sides[1 - side]
    current = None
    for i, p in enumerate(placements):
        if p.spawn <= frame:
            current = i
    if current is None:
        p = placements[0]
        return p.board, p.pill, p.preview, 0, int(p.row["speed_ups"]), "not_started"
    p = placements[current]
    falling = p.lock is None or frame < p.lock
    if falling or p.settled is None:
        return p.board, p.pill, p.preview, current, int(p.row["speed_ups"]), "falling"
    if current + 1 < len(placements):
        q = placements[current + 1]
        return p.settled, q.pill, q.preview, current + 1, int(q.row["speed_ups"]), "resolving"
    return p.settled, p.preview, p.preview, current + 1, int(p.row["speed_ups"]), "resolving_last"


def _path_ok(placements, start: int, target: int, want: dict) -> tuple[bool, str]:
    """Replay the human's own placements start..target exactly (no garbage, same features)."""
    for k in range(start, target + 1):
        p = placements[k]
        if p.action < 0 or p.placed is None:
            return False, "pose"
        settled, features, _ = bc.resolve(p.placed)
        if k < target:
            if placements[k + 1].board != settled:
                return False, "diverged"
        else:
            if abs(features.score() - want["f_score"]) > 1e-6 or features.cells != want["f_cells"]:
                return False, "target_mismatch"
    return True, "ok"


def _max_score(placements, start: int, stop: int) -> float:
    best = 0.0
    for p in placements[start:stop]:
        if p.placed is not None:
            best = max(best, bc.resolve(p.placed)[1].score())
    return best


def cmd_extract(args) -> None:
    from drmc_rl.human.corpus_public_state import CorpusGame, count_viruses

    selection = json.loads(Path(args.selection).read_text())
    targets = [dict(t, role=role) for role in args.roles for t in selection["rows"][role]]
    months, by_game = defaultdict(set), defaultdict(list)
    for t in targets:
        months[t["month"]].add(t["game_id"])
        by_game[t["game_id"]].append(t)
    lookbacks = {"training": args.lookbacks, "benchmark": args.bench_lookbacks}
    counts = Counter()
    out = gzip.open(args.out, "wt")
    for gid, rows in _stream_games(args, months):
        try:
            game = CorpusGame(rows)
        except ValueError:
            counts["game_error"] += len(by_game[gid])
            continue
        for t in by_game[gid]:
            side = int(t["slot"]) - 1
            placements = game.sides[side]
            c = int(t["ordinal"])
            if c >= len(placements) or placements[c].spawn != t["spawn_frame"]:
                counts["ordinal_mismatch"] += 1
                continue
            controls_done = False
            for L in lookbacks[t["role"]]:
                s = c - L
                if s < 0 or t["clean_run"] < L:
                    counts[f"skip_L{L}"] += 1
                    continue
                ok, why = _path_ok(placements, s, c, t)
                counts[f"path_{why}"] += 1
                if not ok:
                    continue
                p = placements[s]
                opp = _settled_opponent(game, side, p.spawn)
                start = dict(role=t["role"], game_id=gid, slot=t["slot"], player=t["player"], split=t["split"],
                             month=t["month"], level=t["level"], seed=t["seed"] if t["seed_ok"] else -1,
                             lookback=L, ordinal=s, target_ordinal=c, target=t, board=list(p.board),
                             pill=list(p.pill), preview=list(p.preview), speed_ups=int(p.row["speed_ups"]),
                             viruses=count_viruses(p.board),
                             opponent=dict(board=list(opp[0]), pill=list(opp[1]), preview=list(opp[2]),
                                           ordinal=opp[3], speed_ups=opp[4], phase=opp[5]))
                out.write(json.dumps(start) + "\n")
                counts[f"start_{t['role']}"] += 1
                if t["role"] == "benchmark" and not controls_done:
                    control = _control(placements, s, c, start, args)
                    if control is not None:
                        opp = _settled_opponent(game, side, placements[control].spawn)
                        q = placements[control]
                        out.write(json.dumps(dict(start, role="control", ordinal=control, lookback=L,
                                                  board=list(q.board), pill=list(q.pill), preview=list(q.preview),
                                                  speed_ups=int(q.row["speed_ups"]), viruses=count_viruses(q.board),
                                                  opponent=dict(board=list(opp[0]), pill=list(opp[1]),
                                                                preview=list(opp[2]), ordinal=opp[3],
                                                                speed_ups=opp[4], phase=opp[5]))) + "\n")
                        counts["start_control"] += 1
                        controls_done = True
                    else:
                        counts["control_missing"] += 1
    out.close()
    print(json.dumps(dict(counts), indent=1))
    Path(args.out).with_suffix(".counts.json").write_text(json.dumps(dict(counts), indent=1))


def _control(placements, s: int, c: int, start: dict, args):
    """A same-side-game spawn with similar viruses where the human made no T1+ clear in the window."""
    from drmc_rl.human.corpus_public_state import count_viruses

    want = start["viruses"]
    best, best_gap = None, None
    t1 = bc.TIERS[0][1]
    for j in range(0, len(placements) - args.control_window):
        if abs(j - c) <= args.control_window:  # away from the target's build-up and aftermath
            continue
        if _max_score(placements, j, j + args.control_window) >= t1:
            continue
        gap = abs(count_viruses(placements[j].board) - want)
        if best_gap is None or gap < best_gap:
            best, best_gap = j, gap
    return best if best_gap is not None and best_gap <= args.control_virus_gap else None


# ------------------------------------------------------------------------------ write

def to_arrays(starts: list[dict]) -> dict[str, np.ndarray]:
    boards, falling, preview, counter, speed_ups, kind = [], [], [], [], [], []
    meta = defaultdict(list)
    for st in starts:
        own = np.asarray(st["board"], np.uint8)
        nes = lambda pair: [_CANONICAL_TO_NES[int(c)] for c in pair]
        own_counter = (int(st["ordinal"]) + 1) & 127  # corpus placement k falls reserve[k + 1]
        variants = []
        if st["role"] == "training":
            opp = st["opponent"]
            slot = int(st["slot"]) - 1
            sides = [(own, st["pill"], st["preview"], own_counter, st["speed_ups"]),
                     (np.asarray(opp["board"], np.uint8), opp["pill"], opp["preview"], (int(opp["ordinal"]) + 1) & 127,
                      opp["speed_ups"])]
            if slot == 1:
                sides = sides[::-1]
            variants.append((0, sides))
        mirror = [(own, st["pill"], st["preview"], own_counter, st["speed_ups"])] * 2
        variants.append((2 if st["role"] == "control" else 1, mirror))
        for k, sides in variants:
            boards.append(np.stack([s[0] for s in sides]).reshape(2, 16, 8))
            falling.append([nes(s[1]) for s in sides])
            preview.append([nes(s[2]) for s in sides])
            counter.append([s[3] for s in sides])
            speed_ups.append([s[4] for s in sides])
            kind.append(k)
            t = st["target"]
            meta["seed"].append(int(st["seed"]))
            meta["lookback"].append(int(st["lookback"]))
            meta["setup_side"].append(int(st["slot"]) - 1 if k == 0 else -1)
            meta["target_score"].append(float(t["f_score"]))
            meta["target_tier"].append(int(bc.tier_index(float(t["f_score"]))))
            meta["target_cells"].append(int(t["f_cells"]))
            meta["target_rounds"].append(int(t["f_rounds"]))
            meta["target_viruses"].append(int(t["f_viruses"]))
            meta["target_type"].append(TYPES.index(t["type"]))
            meta["level"].append(int(st["level"]))
            meta["viruses"].append(int(st["viruses"]))
            meta["player"].append(hashlib.sha256(st["player"].encode()).hexdigest()[:12])
            meta["origin"].append(f"corpus:{st['game_id']}:{st['slot']}:{st['ordinal']}:{KINDS[k]}")
            meta["split"].append(st["split"])
            meta["group_key"].append(f"{st['game_id']}:{st['slot']}:{t['ordinal']}")
    arrays = dict(boards=np.asarray(boards, np.uint8), falling=np.asarray(falling, np.uint8),
                  preview=np.asarray(preview, np.uint8), pill_counter=np.asarray(counter, np.uint8),
                  speed_ups=np.asarray(speed_ups, np.uint8), kind=np.asarray(kind, np.uint8))
    for key, values in meta.items():
        dtype = np.float32 if key == "target_score" else None
        arrays[key] = np.asarray(values, dtype=dtype) if dtype else np.asarray(values)
    arrays["stratum"] = (arrays["target_tier"] * 4 + arrays["kind"]).astype(np.uint8)
    keys = arrays.pop("group_key")
    arrays["group"] = np.unique(keys, return_inverse=True)[1].astype(np.int32)
    return arrays


def cmd_write(args) -> None:
    from drmc_rl.program.seed_reserve import load_reserve

    starts = [json.loads(l) for l in gzip.open(args.starts, "rt")]
    roles = {"training": ("training",), "benchmark": ("benchmark", "control")}[args.split]
    starts = [s for s in starts if s["role"] in roles]
    if args.split == "benchmark":
        # Keep a target's setups only with its matched control (paired lift).
        with_control = {(s["game_id"], s["slot"], s["target"]["ordinal"]) for s in starts if s["role"] == "control"}
        starts = [s for s in starts if (s["game_id"], s["slot"], s["target"]["ordinal"]) in with_control]
    arrays = to_arrays(starts)
    blocked = load_reserve().blocked
    reserved = np.isin(arrays["seed"], list(blocked))
    if args.split == "training":
        arrays["seed"] = np.where(reserved, -1, arrays["seed"])  # never replay an evaluation-reserve seed
    if not args.skip_pool_check:
        from tools.build_clear_endgame_bank import pool_load_mask

        ok = pool_load_mask(dict(arrays, spawn_f=np.zeros(len(arrays["boards"]), np.int64),
                                 levels=np.full((len(arrays["boards"]), 2), 14, np.uint8),
                                 speeds=np.full((len(arrays["boards"]), 2), 2, np.uint8)))
        if args.split == "benchmark":
            bad = set(arrays["group"][~ok].tolist())
            ok = ~np.isin(arrays["group"], list(bad))
        print(f"pool check dropped {int((~ok).sum())} rows")
        arrays = {k: v[ok] for k, v in arrays.items()}
        arrays["group"] = np.unique(arrays["group"], return_inverse=True)[1].astype(np.int32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    kinds = Counter(KINDS[k] for k in arrays["kind"])
    meta = dict(schema=BANK_SCHEMA, score=bc.SCHEMA, tiers=bc.TIERS, weights=bc.WEIGHTS, split=args.split,
                rows=int(len(arrays["boards"])), groups=int(arrays["group"].max() + 1), kinds=dict(kinds),
                lookbacks={str(k): int(v) for k, v in zip(*np.unique(arrays["lookback"], return_counts=True))},
                tiers_rows={f"T{int(k) }": int(v) for k, v in zip(*np.unique(arrays["target_tier"], return_counts=True))},
                types={TYPES[int(k)]: int(v) for k, v in zip(*np.unique(arrays["target_type"], return_counts=True))},
                players=int(len(set(arrays["player"].tolist()))),
                replayable_rows=int((arrays["seed"] > 0).sum()), reserved_seed_rows=int(reserved.sum()),
                levels=dict(Counter(level_band(int(v)) for v in arrays["level"])),
                starts=str(args.starts), sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select")
    s.add_argument("--mining", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--targets", type=int, default=9000)
    s.add_argument("--bench-targets", type=int, default=160)
    s.add_argument("--bench-player-cap", type=int, default=6)
    s.add_argument("--player-cap", type=float, default=0.015)
    s.add_argument("--holdout-players", type=float, default=0.12)
    s.add_argument("--holdout-games", type=float, default=0.08)
    s.add_argument("--seed", type=int, default=20260925)
    e = sub.add_parser("extract")
    e.add_argument("--selection", required=True)
    e.add_argument("--remote")
    e.add_argument("--release")
    e.add_argument("--scratch", default="/tmp/big-clear-extract")
    e.add_argument("--out", required=True)
    e.add_argument("--control-window", type=int, default=16, help="no T1+ human clear in this many placements")
    e.add_argument("--control-virus-gap", type=int, default=5)
    e.add_argument("--roles", nargs="+", default=["training", "benchmark"], choices=("training", "benchmark"))
    w = sub.add_parser("write")
    w.add_argument("--starts", required=True)
    w.add_argument("--split", choices=("training", "benchmark"), required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--skip-pool-check", action="store_true")
    for p in (s, e):
        p.add_argument("--lookbacks", type=int, nargs="+", default=[3, 6, 10, 20])
        p.add_argument("--bench-lookbacks", type=int, nargs="+", default=[6, 10])
    args = parser.parse_args()
    {"select": cmd_select, "extract": cmd_extract, "write": cmd_write}[args.command](args)


if __name__ == "__main__":
    main()
