#!/usr/bin/env python3
import csv, json, argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('--mmc1-log', required=True)
ap.add_argument('--ines', required=False)
ap.add_argument('--out', required=True)
args = ap.parse_args()
writes = []
with open(args.mmc1_log, 'r') as f:
    r = csv.DictReader(f)
    for row in r:
        addr = row.get('addr') or row.get('address')
        if not addr:
            continue
        if not addr.startswith('0x'):
            try:
                addr = hex(int(addr, 16))
            except Exception:
                continue
        if int(addr, 16) >= 0x8000:
            val_s = row.get('val') or row.get('value') or row.get('data')
            try:
                val = int(val_s, 0) if isinstance(val_s, str) else int(val_s)
            except Exception:
                val = 0
            writes.append({
                'frame': int(row.get('frame', 0)),
                'addr': int(addr, 16),
                'val': val
            })
segments = []
last = None
for w in writes:
    if not segments or w['frame'] != last:
        segments.append({'start_frame': w['frame'], 'end_frame': w['frame']+1, 'events': []})
    segments[-1]['events'].append(w); last = w['frame']
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f: json.dump({'segments': segments}, f, indent=2)
print(f"Wrote {args.out} with {len(segments)} segments")
