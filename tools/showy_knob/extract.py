"""Human afterstates and showy-clear events from one Fightcade corpus month.

    python -m tools.showy_knob.extract --corpus RELEASE_DIR --players players.json \
        --month 2026-03 --min-rating 2000 --out ex/2026-03.npz

Keeps 14-Hi games (level 14 and speed Hi on both sides) and placements by
players whose 14-Hi rating (``players.json`` trajectory ``eC``, interpolated at
the match day) is at least ``--min-rating``. Rows are ordered by (game, slot,
spawn frame). Per row: the exact settled afterstate (128 NES bytes), the
placement's own showiness score (``drmc_rl.eval.big_clear``; -1 when nothing
clears), horizontal flag, lines, rounds, rating, a player hash (for the
player-held-out split) and a sequence id (for window labels). Read-only on the
corpus; the output is a local dataset and must not be committed.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from drmc_rl.eval import big_clear as bc

RAW_TO_CANON = (1, 0, 2)


def pose_table() -> np.ndarray:
    from drmc_rl.human.backend import ACTION_TO_POSE
    table = np.full(512, -1, dtype=np.int64)
    for action, pose in enumerate(ACTION_TO_POSE):
        if pose >= 0 and table[pose] < 0:
            table[pose] = action
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--players", type=Path, required=True)
    ap.add_argument("--month", required=True)
    ap.add_argument("--min-rating", type=float, default=2000.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    poses = pose_table()
    traj = defaultdict(list)
    for pl in json.loads(args.players.read_text()):
        for pt in pl.get("traj") or []:
            traj[pl["name"]].append((pt["day"], pt["eC"]))
    traj = {p: (np.array([d for d, _ in v]), np.array([e for _, e in v]))
            for p, v in ((p, sorted(v)) for p, v in traj.items())}
    g = pq.ParquetDataset(str(args.corpus / "games")).read(
        columns=["game_id", "level_p1", "level_p2", "speed_p1", "speed_p2"]).to_pydict()
    std = {gid for gid, a, b, c, d in zip(g["game_id"], g["level_p1"], g["level_p2"], g["speed_p1"], g["speed_p2"])
           if a == b == 14 and c == d == 2}
    year, month = args.month.split("-")
    cols = ["game_id", "player", "player_slot", "day", "spawn_frame", "field", "pill_left", "pill_right",
            "lock_x", "lock_y_top", "lock_rotation"]
    t = pq.read_table(args.corpus / f"decisions/year={year}/month={month}/part-00000.parquet", columns=cols)
    n = t.num_rows
    field = np.frombuffer(t.column("field").combine_chunks().buffers()[1], dtype=np.uint8).reshape(n, 128)
    c = {k: t.column(k).to_numpy(zero_copy_only=False) for k in cols if k != "field"}
    started = time.time()
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
    seqkey = np.array([f"{c['game_id'][i]}|{c['player_slot'][i]}" for i in keep])
    order = np.lexsort((c["spawn_frame"][keep], seqkey))
    keep, seqkey = keep[order], seqkey[order]
    _, seq = np.unique(seqkey, return_inverse=True)
    N = len(keep)
    after = np.full((N, 128), 0xFF, np.uint8)
    score = np.full(N, -1.0, np.float32)
    horiz, lines, rounds = (np.zeros(N, np.int8) for _ in range(3))
    valid = np.zeros(N, bool)
    for j, i in enumerate(keep):
        x, y, rot = c["lock_x"][i], c["lock_y_top"][i], c["lock_rotation"][i]
        if not (0 <= x < 8 and 0 <= y < 16):
            continue
        action = int(poses[(int(rot) & 3) * 128 + int(y) * 8 + int(x)])
        if action < 0:
            continue
        pill = (RAW_TO_CANON[int(c["pill_left"][i]) & 3], RAW_TO_CANON[int(c["pill_right"][i]) & 3])
        try:
            placed = bc.place(field[i].tobytes(), pill, action)
        except ValueError:
            continue
        valid[j] = True
        settled, f, _ = bc.resolve(placed)
        after[j] = np.frombuffer(settled, np.uint8)
        if f.rounds:
            score[j], horiz[j], lines[j], rounds[j] = f.score(), f.horizontal_lines > 0, f.lines, f.rounds
    rating = np.array([ratings[(c["player"][i], int(c["day"][i]))] for i in keep], np.float32)
    pid = np.array([zlib.crc32(c["player"][i].encode()) for i in keep], np.uint32)
    np.savez_compressed(args.out, after=after, score=score, horiz=horiz, lines=lines, rounds=rounds, valid=valid,
                        rating=rating, pid=pid, seq=seq.astype(np.int32))
    print(json.dumps(dict(month=args.month, rows=n, kept=N, valid=int(valid.sum()), t2=int((score >= 30).sum()),
                          seconds=round(time.time() - started, 1))), flush=True)


if __name__ == "__main__":
    main()
