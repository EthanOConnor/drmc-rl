"""Fastest possible time for each recorded placement (drmc-rl native planner), and the human's time lost.

Uses the audited FBNeo boundary from tools/audit_execution_replay.py: the start state takes the
held byte before spawn, the frame-counter parity is inverted, and raw[:-1] drives movement. A row
is kept only when that replay reproduces the recorded lock pose AND lock frame, so the planner's
cost from the same state is directly comparable: slack = recorded frames - fastest frames.
"""
import os, sys
from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '/Users/ethan/dev/drmario/drmc-rl')
src, dest, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])

def init():
    global replay_row, runner, FrameState, HoldDir, Rotation, threshold_for
    from tools.audit_execution_replay import replay_row
    from drmc_rl.planning.native_reach import NativeReachabilityRunner
    from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation
    from drmc_rl.planning.fast_reach import compute_speed_threshold as threshold_for
    global simulate_frame, decode_input_rle
    from drmc_rl.planning.fast_reach import simulate_frame
    from drmc_rl.data.human_corpus import decode_input_rle
    runner = NativeReachabilityRunner()


def action_of(mask):
    direction = 1 if mask & 2 else 2 if mask & 1 else 0
    rotation = 1 if mask & 0x80 else 2 if mask & 0x40 else 0
    return direction * 6 + (3 if mask & 4 else 0) + rotation


def path_stats(cols, state, actions, threshold, masks=None):
    xs, rots, first_down, late = [state.x], [state.rot], None, 0
    for i, action in enumerate(actions):
        state = simulate_frame(cols, state, int(action), speed_threshold=threshold)
        if masks is not None and masks[i] & 4 and first_down is None:
            first_down = i
        if first_down is not None and state.x != xs[-1]:
            late += 1
        xs.append(state.x); rots.append(state.rot)
        if state.locked:
            break
    dx = np.diff(xs)
    moves = dx[dx != 0]
    xf = xs[-1]
    direction = np.sign(xf - xs[0])
    path = np.array(xs)
    over = float(np.max((path - xf) * direction)) if direction else float(np.max(np.abs(path - xf)))
    return {'lateral': int(np.abs(dx).sum()), 'reversals': int(np.count_nonzero(np.diff(np.sign(moves)))) if len(moves) > 1 else 0,
            'overshoot': max(0.0, over), 'rot_changes': int(np.count_nonzero(np.diff(rots))), 'late_lateral': late}

def work(rows):
    out = []
    for r in rows:
        try:
            rep = replay_row(r, parity_xor=1, input_delay_frames=1)
        except ValueError:
            out.append(None); continue
        if rep['status'] != 'match':
            out.append(None); continue
        board = np.frombuffer(bytes(r['field']), dtype=np.uint8).reshape(16, 8)
        cols = np.zeros(8, dtype=np.uint16)
        for y in range(16):
            cols |= (board[y] != 0xFF).astype(np.uint16) << y
        initial = int(r['held_before_spawn'])
        direction = 1 if initial & 2 else 2 if initial & 1 else 0
        rotation = 1 if initial & 0x80 else 2 if initial & 0x40 else 0
        state = FrameState(x=3, y=0, rot=0, speed_counter=int(r['speed_counter']),
                           hor_velocity=int(r['horizontal_velocity']) & 15, hold_dir=HoldDir(direction),
                           rot_hold=Rotation(rotation), frame_parity=(int(r['frame_counter']) & 1) ^ 1)
        threshold = threshold_for(int(r['speed']), int(r['speed_ups']))
        reach = runner.bfs_full(cols, state, speed_threshold=threshold)
        x, y, rot = rep['expected_pose']
        cost = int(reach.costs_u16[reach.pose_index(x, y, rot)])
        if cost == 0xFFFF:
            out.append(None); continue
        raw = decode_input_rle(r['input_rle_u16_u8'], r['input_frames'])
        window = raw[:-1]
        human = path_stats(cols, state, [action_of(m) for m in window], threshold, masks=window)
        fast_script = reach.script_for_pose(x, y, rot)
        fast = path_stats(cols, state, list(fast_script), threshold) if fast_script is not None else None
        seq = [initial] + list(window)
        edges = sum(bin((a ^ b) & 0xC7).count('1') for a, b in zip(seq, seq[1:]))
        active = [i for i, m in enumerate(window) if m & 0xC3 and not (i and window[i - 1] & 0xC3 == m & 0xC3)]
        idle = 0
        if len(active) > 1:
            busy = np.array([bool(m & 0xC7) for m in window[active[0]:active[-1] + 1]])
            run = best = 0
            for b in busy:
                run = 0 if b else run + 1; best = max(best, run)
            idle = best
        fast_edges = None
        if fast_script is not None:
            fseq = [0] + [((2 if a // 6 == 1 else 1 if a // 6 == 2 else 0) | (4 if (a % 6) >= 3 else 0)
                           | (0x80 if a % 3 == 1 else 0x40 if a % 3 == 2 else 0)) for a in fast_script]
            fast_edges = sum(bin((p ^ q) & 0xC7).count('1') for p, q in zip(fseq, fseq[1:]))
        out.append({'fastest_frames': cost, 'h_lateral': human['lateral'], 'h_reversals': human['reversals'],
                    'h_overshoot': human['overshoot'], 'h_rot_changes': human['rot_changes'],
                    'h_late_lateral': human['late_lateral'], 'h_edges': edges, 'h_idle_max': idle,
                    'f_lateral': None if fast is None else fast['lateral'],
                    'f_rot_changes': None if fast is None else fast['rot_changes'], 'f_edges': fast_edges})
    return out

if __name__ == '__main__':
    t = pq.read_table(src)
    rows = t.to_pylist()
    chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
    with Pool(workers, initializer=init) as pool:
        costs = [c for part in pool.imap(work, chunks) for c in part]
    import pyarrow as pa
    keys = ['fastest_frames', 'h_lateral', 'h_reversals', 'h_overshoot', 'h_rot_changes', 'h_late_lateral', 'h_edges',
            'h_idle_max', 'f_lateral', 'f_rot_changes', 'f_edges']
    for k in keys:
        t = t.append_column(k, pa.array([None if c is None else c[k] for c in costs], type=pa.float64()))
    pq.write_table(t.drop(['input_rle_u16_u8', 'field']).append_column('rle', t['input_rle_u16_u8']).append_column('field_bytes', t['field']), dest)
    ok = sum(c is not None for c in costs)
    print('rows', len(costs), 'with fastest', ok)
