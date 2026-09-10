"""Read-only exact human sequence extraction for the diagnostic proposer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

from drmc_rl.human.expressive_sequences import GOALS, SCHEMA, construction_windows, replay_sequences


def write_progress(output, report):
    path = output/'progress.json'
    temporary = path.with_suffix('.json.next')
    temporary.write_text(json.dumps({**report, 'updated_at':datetime.now(timezone.utc).isoformat()}, indent=2)+'\n')
    temporary.replace(path)


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(config['fcr_root'])))
    import store
    connection = sqlite3.connect('file:'+str(Path(config['db']).resolve())+'?mode=ro', uri=True)
    sessions = connection.execute('SELECT quarkid,sha256 FROM processed_replay WHERE sha256 IS NOT NULL ORDER BY quarkid').fetchall()
    connection.close()
    rng = np.random.default_rng(int(config.get('seed', 20260910)))
    maximum = int(config.get('max_sessions', 512))
    cap = int(config.get('max_windows_per_session', 96))
    if min(maximum, cap) < 1:
        raise ValueError('positive source limits required')
    rows, windows, identities = [], [], []
    counts, goal_counts = Counter(), Counter()
    started = time.monotonic()
    report = dict(schema=SCHEMA, status='Running', phase='verifying_replay', target_sessions=maximum,
                  sessions=0, windows=0, state_presentations=0, config=config)
    write_progress(output, report)
    for index in rng.permutation(len(sessions))[:maximum]:
        session, sha = sessions[int(index)]
        try:
            raw = store.get_blob(sha)
        except FileNotFoundError:
            counts['missing_blob'] += 1
            continue
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('archive blob differs from its content hash')
        segments, counters = replay_sequences(raw)
        counts.update(counters)
        candidates = [(segment, start, length, goal) for segment in segments
                      for start, length, goal in construction_windows(segment)]
        selected = rng.permutation(len(candidates))[:cap]
        source_id = len(identities)
        for slot in selected:
            segment, start, length, goal = candidates[int(slot)]
            sequence = segment[start:start+length]
            windows.append((len(rows), length, goal, source_id))
            rows.extend(sequence)
            goal_counts[GOALS[goal]] += 1
        identities.append(dict(session=str(session), sha256=sha, windows=len(selected)))
        report.update(sessions=len(identities), windows=len(windows), state_presentations=len(rows),
                      counters=dict(counts), goals=dict(goal_counts), elapsed_seconds=time.monotonic()-started)
        write_progress(output, report)
    if not windows:
        raise ValueError('no verified multi-placement constructions')
    arrays = {key:np.asarray([r[key] for r in rows]) for key in
              ('board','pill','preview','action','goals','frame','lock_frame','game','player','level','speed')}
    arrays.update(windows=np.asarray(windows, np.int64),
                  sessions=np.asarray([r['session'] for r in identities]),
                  metadata=np.asarray(json.dumps(dict(schema=SCHEMA, goals=GOALS, sources=identities,
                      scope='own-board transition verification; no motor, quality or commentary preference labels'))))
    np.savez_compressed(output/'sequences.npz', **arrays)
    report.update(status='Complete', phase='complete', source_sha256=hashlib.sha256((output/'sequences.npz').read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started)
    write_progress(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        run(config)
    except BaseException as exc:
        output = Path(config['output'])
        if output.is_dir() and not isinstance(exc, FileExistsError):
            prior = json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
            write_progress(output, {**prior,'status':'Failed','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
