"""Placement rows for the skill-gap comparison: strong humans (Fightcade corpus) and pool AI traces.

    python -m tools.skill_gaps.extract human --corpus RELEASE_DIR --players players.json \
        --months 2026-01,2026-02 --min-rating 2000 --out human.npz
    python -m tools.skill_gaps.extract pool --traces ~/drmc-rl-pool/data/traces \
        --conditions c4f340433f101=fast,... --entrants E0,E1 --out pool.npz

Both sources give the same row layout, ordered by (sequence, placement index), where a
sequence is one side of one game: the root bottle at the decision (128 NES bytes), the
settled afterstate, the opponent's bottle, the canonical pill and macro action, the
placement's exact resolution (lines, rounds, cells, viruses cleared, showiness score),
the decision frame, the game outcome for this side (1 win, 0 loss, -1 draw/unknown),
and a group code (human: player hash; pool: entrant index) plus a pace/rating field.
Read-only on its sources; the output is a local dataset and must not be committed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from drmc_rl.eval import big_clear as bc

RAW_TO_CANON = (1, 0, 2)
ROT_OFFSETS = ((0, 0, 0, 1), (0, 0, -1, 0), (0, 1, 0, 0), (-1, 0, 0, 0))
ORIENT_INDEX = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}


def pose_to_action() -> np.ndarray:
    """Corpus pose (rot*128 + y*8 + x, bottom-left of the 2x2 box) -> macro action (``tools.annotate_replay_events``)."""
    out = np.full(512, -1, np.int32)
    for rot in range(4):
        dr1, dc1, dr2, dc2 = ROT_OFFSETS[rot]
        for y in range(16):
            for x in range(8):
                r1, c1, r2, c2 = y + dr1, x + dc1, y + dr2, x + dc2
                if 0 <= r1 < 16 and 0 <= c1 < 8 and 0 <= r2 < 16 and 0 <= c2 < 8:
                    o = ORIENT_INDEX.get((r2 - r1, c2 - c1))
                    if o is not None:
                        out[rot * 128 + y * 8 + x] = o * 128 + r1 * 8 + c1
    return out


class Rows:
    FIELDS = dict(root=(np.uint8, 128), after=(np.uint8, 128), opp=(np.uint8, 128), pill=(np.int8, 2),
                  action=(np.int16, 0), lines=(np.int8, 0), rounds=(np.int8, 0), cells=(np.int8, 0),
                  vcleared=(np.int8, 0), score=(np.float32, 0), hlines=(np.int8, 0), frame=(np.int32, 0),
                  seq=(np.int32, 0), game=(np.int32, 0), t=(np.int16, 0), won=(np.int8, 0), group=(np.int64, 0), rating=(np.float32, 0),
                  pace=(np.int8, 0))

    def __init__(self):
        self.cols = {k: [] for k in self.FIELDS}

    def add(self, root, pill, action, opp, frame, seq, game, t, won, group, rating, pace) -> bool:
        try:
            placed = bc.place(root, pill, action)
        except ValueError:
            return False
        settled, f, _ = bc.resolve(placed)
        c = self.cols
        c["root"].append(np.frombuffer(root, np.uint8))
        c["after"].append(np.frombuffer(bytes(settled), np.uint8))
        c["opp"].append(np.frombuffer(opp, np.uint8) if opp is not None else np.full(128, 0xFF, np.uint8))
        c["pill"].append(pill)
        for k, v in (("action", action), ("lines", f.lines), ("rounds", f.rounds), ("cells", f.cells),
                     ("vcleared", f.viruses), ("score", f.score() if f.rounds else -1.0), ("hlines", f.horizontal_lines),
                     ("frame", frame), ("seq", seq), ("game", game), ("t", t), ("won", won), ("group", group), ("rating", rating),
                     ("pace", pace)):
            c[k].append(v)
        return True

    def save(self, path, **meta):
        out = {k: np.asarray(v, self.FIELDS[k][0]) for k, v in self.cols.items()}
        np.savez_compressed(path, meta=json.dumps(meta), **out)
        return len(out["seq"])


def human(args):
    import pyarrow.parquet as pq
    poses = pose_to_action()
    traj = defaultdict(list)
    for pl in json.loads(Path(args.players).read_text()):
        for pt in pl.get("traj") or []:
            traj[pl["name"]].append((pt["day"], pt["eC"]))
    traj = {p: (np.array([d for d, _ in v]), np.array([e for _, e in v])) for p, v in ((p, sorted(v)) for p, v in traj.items())}
    g = pq.ParquetDataset(str(Path(args.corpus) / "games")).read(
        columns=["game_id", "level_p1", "level_p2", "speed_p1", "speed_p2"]).to_pydict()
    std = {gid for gid, a, b, c, d in zip(g["game_id"], g["level_p1"], g["level_p2"], g["speed_p1"], g["speed_p2"])
           if a == b == 14 and c == d == 2}
    rows, seq_ids, game_ids, started = Rows(), {}, {}, time.time()
    cols = ["game_id", "player", "player_slot", "day", "won", "spawn_frame", "field", "opp_field", "pill_left",
            "pill_right", "lock_x", "lock_y_top", "lock_rotation"]
    for month in args.months.split(","):
        year, mon = month.split("-")
        t = pq.read_table(Path(args.corpus) / f"decisions/year={year}/month={mon}/part-00000.parquet", columns=cols)
        n = t.num_rows
        field = np.frombuffer(t.column("field").combine_chunks().buffers()[1], np.uint8).reshape(n, 128)
        opp = np.frombuffer(t.column("opp_field").combine_chunks().buffers()[1], np.uint8).reshape(n, 128)
        c = {k: t.column(k).to_numpy(zero_copy_only=False) for k in cols if k not in ("field", "opp_field")}
        ratings, keep = {}, []
        for i in range(n):
            if c["game_id"][i] not in std:
                continue
            key = (c["player"][i], int(c["day"][i]))
            if key not in ratings:
                tr = traj.get(key[0])
                ratings[key] = np.nan if tr is None else float(np.interp(key[1], tr[0], tr[1]))
            if ratings[key] >= args.min_rating:
                keep.append(i)
        keep = np.asarray(keep)
        order = np.lexsort((c["spawn_frame"][keep], c["player_slot"][keep], c["game_id"][keep]))
        keep = keep[order]
        prev, t_in = None, 0
        for i in keep:
            skey = (c["game_id"][i], int(c["player_slot"][i]))
            if skey != prev:
                prev, t_in = skey, 0
            seq = seq_ids.setdefault(skey, len(seq_ids))
            game = game_ids.setdefault(skey[0], len(game_ids))
            x, y, rot = int(c["lock_x"][i]), int(c["lock_y_top"][i]), int(c["lock_rotation"][i])
            if not (0 <= x < 8 and 0 <= y < 16):
                t_in += 1
                continue
            action = int(poses[(rot & 3) * 128 + y * 8 + x])
            if action >= 0:
                pill = (RAW_TO_CANON[int(c["pill_left"][i]) & 3], RAW_TO_CANON[int(c["pill_right"][i]) & 3])
                rows.add(field[i].tobytes(), pill, action, opp[i].tobytes(), int(c["spawn_frame"][i]), seq, game, t_in,
                         int(bool(c["won"][i])), zlib.crc32(c["player"][i].encode()),
                         ratings[(c["player"][i], int(c["day"][i]))], 0)
            t_in += 1
        print(json.dumps(dict(month=month, rows=n, kept=len(keep), total=len(rows.cols["seq"]),
                              seconds=round(time.time() - started))), flush=True)
    names = {zlib.crc32(p.encode()): p for p in traj}
    total = rows.save(args.out, source="human", months=args.months, min_rating=args.min_rating,
                      players={str(k): v for k, v in names.items()})
    print(json.dumps(dict(out=str(args.out), rows=total)))


def pool(args):
    conds = dict(kv.split("=") for kv in args.conditions.split(","))
    paces = {name: i for i, name in enumerate(dict.fromkeys(conds.values()))}
    entrants = args.entrants.split(",")
    rows, seq, games = Rows(), 0, 0
    root = Path(args.traces).expanduser()
    for cond, pace in conds.items():
        for path in sorted((root / cond).glob("*.json.gz")):
            d = json.load(gzip.open(path))
            parts = d["id"].split("/")
            names = (parts[1], parts[2])
            if not any(n in entrants for n in names):
                continue
            g = d["game"]
            games += 1
            for side in (0, 1):
                name = names[side]
                if name not in entrants:
                    continue
                won = -1 if g["winner"] not in ("a", "b") else int(g["winner"] == "ab"[side])
                t_in = 0
                for m in d["moves"]:
                    if m["side"] != side:
                        continue
                    rows.add(bytes(m["board"]), tuple(m["pill"]), int(m["placement"]["action"]), bytes(m["opponent"]),
                             int(m["frame"]), seq, games - 1, t_in, won, entrants.index(name), np.nan, paces[pace])
                    t_in += 1
                seq += 1
    total = rows.save(args.out, source="pool", entrants=entrants, paces=list(paces), games=games)
    print(json.dumps(dict(out=str(args.out), rows=total, games=games, sequences=seq)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("human")
    h.add_argument("--corpus", required=True)
    h.add_argument("--players", required=True)
    h.add_argument("--months", required=True)
    h.add_argument("--min-rating", type=float, default=2000.0)
    h.add_argument("--out", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--traces", required=True)
    p.add_argument("--conditions", required=True, help="hash=pace,...")
    p.add_argument("--entrants", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    (human if args.cmd == "human" else pool)(args)


if __name__ == "__main__":
    main()
