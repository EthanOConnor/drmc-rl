"""Corpus-wide audit of drm_replay lock records vs field-diff ground truth.

No re-replay, no emulation: for each move, the next same-player spawn field
diff pins where the pill actually landed (see derive_lock_poses in
tools.annotate_replay_events and
fightcadeRatings/docs/drm-replay-lock-record-issues.md). This tool scans
quarks and reports:

  verified        lock event matches the field diff
  mismatch        lock event disagrees (dx/dy/rot histograms reported)
  underivable     clear/garbage in the window, game end, non-domino diff

With --planner cuda, mismatches are further split into CAUGHT (recorded pose
infeasible -> annotation would have flagged it) vs SILENT (recorded pose
feasible -> annotation would have been quietly wrong without repair).

Shardable and planner/torch-free by default:
  .venv/bin/python -m tools.audit_lock_records --limit 300 --planner cuda
  for i in 0 1 2 3: ... --shard $i/4 &
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FCR_ROOT = REPO_ROOT.parent / "fightcadeRatings"

from envs.retro.fast_reach import compute_speed_threshold
from tools.annotate_replay_events import (
    GRID_H, GRID_W, decode_field, derive_lock_poses, occupancy_cols,
    pair_moves, parse_quark_events)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcr-root", type=str, default=str(FCR_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="max quarks")
    ap.add_argument("--shard", type=str, default=None, metavar="I/M")
    ap.add_argument("--planner", choices=("none", "cuda"), default="none",
                    help="classify mismatches into caught vs silent")
    args = ap.parse_args()

    import sqlite3
    fcr = Path(args.fcr_root)
    db = sqlite3.connect(fcr / "data" / "drmario.sqlite")
    rows = db.execute(
        "SELECT quarkid, sha256 FROM processed_replay "
        "WHERE sha256 IS NOT NULL ORDER BY quarkid").fetchall()
    if args.shard:
        i, m = (int(x) for x in args.shard.split("/"))
        rows = rows[i::m]
    if args.limit:
        rows = rows[: args.limit]

    ctx = None
    if args.planner == "cuda":
        from reach_cuda import CudaReach
        ctx = CudaReach(max_batch=8192)

    counters: dict = collections.defaultdict(int)
    dxy = collections.Counter()
    n_moves = 0
    n_quarks = 0
    silent = caught = oob = 0
    t0 = time.time()

    for quarkid, sha in rows:
        blob = fcr / "data" / "blobs" / sha
        if not blob.exists():
            counters["blob_missing"] += 1
            continue
        events = parse_quark_events(blob.read_bytes())
        moves = pair_moves(events, counters)
        if not moves:
            continue
        derived = derive_lock_poses(events, moves, counters)
        n_quarks += 1
        n_moves += len(moves)

        mismatched = []
        for j, (mv, d) in enumerate(zip(moves, derived)):
            if d is None:
                counters["underivable"] += 1
                continue
            lk = mv["lock"]
            rec_x, rec_rot = int(lk["x"]), int(lk["rot"]) & 3
            rec_y = (GRID_H - 1) - int(lk["y"])
            if (rec_x, rec_y) == (d["x"], d["y_top"]) and rec_rot in d["rot_choices"]:
                counters["verified"] += 1
            else:
                counters["mismatch"] += 1
                key = (d["x"] - rec_x, d["y_top"] - rec_y,
                       "rot" if (rec_x, rec_y) == (d["x"], d["y_top"]) else
                       ("v" if d["rot_choices"][0] & 1 else "h"))
                dxy[key] += 1
                mismatched.append((j, mv, d, (rec_x, rec_y, rec_rot)))

        if ctx is not None and mismatched:
            n = len(mismatched)
            cols = np.zeros((n, 8), np.uint16)
            par = np.zeros(n, np.uint8)
            thr = np.zeros(n, np.uint8)
            for k, (j, mv, d, rec) in enumerate(mismatched):
                sp = mv["spawn"]
                cols[k] = occupancy_cols(decode_field(sp["field"]))
                par[k] = int(sp["f"]) & 1
                thr[k] = compute_speed_threshold(int(sp["spd"]), int(sp["spdups"]))
            costs = ctx.solve_costs(cols, par, thr)
            for k, (j, mv, d, (rx, ry, rr)) in enumerate(mismatched):
                if not (0 <= rx < GRID_W and 0 <= ry < GRID_H):
                    oob += 1
                elif costs[k][rr * 128 + ry * 8 + rx] != 0xFFFF:
                    silent += 1
                else:
                    caught += 1

    dt = time.time() - t0
    print(f"quarks {n_quarks}  moves {n_moves}  wall {dt:.1f}s "
          f"({n_moves / max(dt, 1e-9):.0f} moves/s)")
    tot_deriv = counters["verified"] + counters["mismatch"]
    print(f"derivable {tot_deriv} ({tot_deriv / max(n_moves, 1):.4f}) | "
          f"verified {counters['verified']} | mismatch {counters['mismatch']} "
          f"({counters['mismatch'] / max(tot_deriv, 1):.5f} of derivable)")
    print("underivable reasons:",
          {k: v for k, v in sorted(counters.items()) if k.startswith('fielddiff')})
    print("mismatch (dx, dy, kind) histogram:")
    for k, v in dxy.most_common(12):
        print(f"  {v:6d}  {k}")
    if ctx is not None:
        print(f"mismatch classification: SILENT (recorded pose feasible) {silent} | "
              f"caught-by-infeasibility {caught} | recorded OOB {oob}")
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
