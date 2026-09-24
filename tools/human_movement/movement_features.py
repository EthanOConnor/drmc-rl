"""Per-placement movement features for fitting the human movement model.

Input is the output of ``slack.py`` (exact-replay rows with the fastest route) plus the
ratings parquet. Each placement is replayed again with the audited FBNeo alignment
(held byte before spawn, ``raw[:-1]`` drives movement, parity xor 1) to recover the frame at
which steering ended and the minimum remaining drop from there, which splits a placement's
time into reaction, steering and descent.

    python -m tools.human_movement.movement_features SLACK.parquet RATINGS.parquet OUT.parquet WORKERS
"""
from __future__ import annotations

import hashlib
import struct
import sys
from multiprocessing import Pool

import numpy as np

MOVE = 0xC3
COLUMNS = (
    "rating", "rating_sd", "player_index", "player_fold", "day", "speed", "speed_ups", "threshold",
    "height", "rows_fallen", "dx", "rot_need", "lock_rot", "tau", "fastest", "lost",
    "reaction", "presses_lat", "presses_rot", "gap_median", "gap_first", "das", "rot_first",
    "first_rot_offset", "steer_end", "steer_y", "min_drop", "descent_slack", "down_first",
    "down_frames", "down_share", "idle_max", "reversals", "overshoot", "extra_rot", "late_lateral",
    "extra_lateral", "corrected",
)
GAP_LIMIT = 24  # individual press gaps kept per placement for the gap distribution


def fold(player: str) -> int:
    return int.from_bytes(hashlib.blake2b(player.encode(), digest_size=8).digest(), "little") % 20


def _init():
    global simulate_frame, FrameState, HoldDir, Rotation, threshold_for
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation, simulate_frame
    from drmc_rl.planning.fast_reach import compute_speed_threshold as threshold_for


def _action(mask: int) -> int:
    direction = 1 if mask & 2 else 2 if mask & 1 else 0
    rotation = 1 if mask & 0x80 else 2 if mask & 0x40 else 0
    return direction * 6 + (3 if mask & 4 else 0) + rotation


def row_features(r):
    raw = bytearray()
    for n, b in struct.iter_unpack("<HB", r["rle"]):
        raw.extend(bytes((b,)) * n)
    tau = int(r["tau_frames"])
    window = raw[:-1]
    if len(window) != tau:
        return None
    field = r["field_bytes"]
    cols = np.zeros(8, dtype=np.uint16)
    board = np.frombuffer(bytes(field), dtype=np.uint8).reshape(16, 8)
    for y in range(16):
        cols |= (board[y] != 0xFF).astype(np.uint16) << y
    initial = int(r["held_before_spawn"])
    state = FrameState(x=3, y=0, rot=0, speed_counter=int(r["speed_counter"]),
                       hor_velocity=int(r["horizontal_velocity"]) & 15,
                       hold_dir=HoldDir(1 if initial & 2 else 2 if initial & 1 else 0),
                       rot_hold=Rotation(1 if initial & 0x80 else 2 if initial & 0x40 else 0),
                       frame_parity=(int(r["frame_counter"]) & 1) ^ 1)
    threshold = threshold_for(int(r["speed"]), int(r["speed_ups"]))
    states = [state]
    for mask in window:
        state = simulate_frame(cols, state, _action(mask), speed_threshold=threshold)
        states.append(state)
        if state.locked:
            break
    if not state.locked or len(states) - 1 != tau:
        return None
    steer_end = 0
    for i in range(1, len(states)):
        if (states[i].x, states[i].rot) != (states[i - 1].x, states[i - 1].rot):
            steer_end = i
    probe, min_drop = states[steer_end], 0
    while not probe.locked and min_drop < 600:
        probe = simulate_frame(cols, probe, 3, speed_threshold=threshold)
        min_drop += 1
    prev, presses, kinds, das, down_first, down_frames = initial, [], [], 0, -1, 0
    held_since = {1: None, 2: None}
    for i, b in enumerate(window):
        new = b & ~prev
        if new & MOVE:
            presses.append(i)
            kinds.append(1 if new & 3 else 0)
        for bit in (1, 2):
            if b & bit:
                held_since[bit] = i if held_since[bit] is None else held_since[bit]
                das |= i + 1 - held_since[bit] >= 16
            else:
                held_since[bit] = None
        if b & 4:
            down_frames += 1
            if down_first < 0:
                down_first = i
        prev = b
    rot_presses = [p for p, k in zip(presses, kinds) if k == 0]
    gaps = np.diff(presses)
    height = next((16 - y for y in range(16) if any(field[y * 8 + c] != 0xFF for c in range(8))), 0)
    lock_rot = int(r["lock_rotation"]) & 3
    rot_need = {0: 0, 1: 1, 2: 2, 3: 1}[lock_rot]
    dx = int(r["lock_x"]) - 3
    extra_rot = max(0.0, r["h_rot_changes"] - rot_need)
    corrected = int(r["h_reversals"] > 0 or r["h_overshoot"] > 0 or r["h_rot_changes"] > rot_need + 1
                    or r["h_late_lateral"] > 0)
    return [
        r["rating"], r["rating_sd"], r["player_index"], fold(r["player"]), r["day"], r["speed"],
        r["speed_ups"], threshold, height, int(r["lock_y_top"]) + 1, dx, rot_need, lock_rot, tau,
        r["fastest_frames"], tau - r["fastest_frames"],
        presses[0] if presses else -1, sum(kinds), len(kinds) - sum(kinds),
        float(np.median(gaps)) if len(gaps) else -1, float(gaps[0]) if len(gaps) else -1, int(das),
        int(bool(kinds) and kinds[0] == 0), (rot_presses[0] - presses[0]) if rot_presses else -1,
        steer_end, states[steer_end].y, min_drop, tau - steer_end - min_drop, down_first,
        down_frames, down_frames / max(tau, 1), r["h_idle_max"], r["h_reversals"], r["h_overshoot"],
        extra_rot, r["h_late_lateral"], r["h_lateral"] - abs(dx), corrected,
    ], [min(int(g), 255) for g in gaps[:GAP_LIMIT]]


def _work(rows):
    out = []
    for r in rows:
        try:
            out.append(row_features(r))
        except Exception:  # noqa: BLE001 - malformed rows are excluded and counted
            out.append(None)
    return out


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    src, ratings_src, dest, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    curves = {}
    for r in pq.read_table(ratings_src).to_pylist():
        curves.setdefault(r["player"], []).append((r["day"], r["skill_elo"], r["skill_sd"]))
    curves = {p: np.array(sorted(v), dtype=float).T for p, v in curves.items()}
    table = pq.read_table(src)
    rows, players = [], {}
    for r in table.to_pylist():
        if r["fastest_frames"] is None or r["player"] not in curves:
            continue
        days, elo, sd = curves[r["player"]]
        r["rating"], r["rating_sd"] = float(np.interp(r["day"], days, elo)), float(np.interp(r["day"], days, sd))
        r["player_index"] = players.setdefault(r["player"], len(players))
        rows.append(r)
    del table
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    with Pool(workers, initializer=_init) as pool:
        results = [x for part in pool.imap(_work, chunks) for x in part]
    kept = [x for x in results if x is not None]
    data = np.asarray([x[0] for x in kept], dtype=np.float32)
    columns = {name: data[:, i] for i, name in enumerate(COLUMNS)}
    columns["gaps"] = pa.array([x[1] for x in kept], type=pa.list_(pa.uint8()))
    pq.write_table(pa.table(columns), dest)
    names = sorted(players, key=players.get)
    pq.write_table(pa.table({"player": names, "player_fold": [fold(p) for p in names]}), dest + ".players.parquet")
    print("rows", len(rows), "kept", len(kept))


if __name__ == "__main__":
    main()
