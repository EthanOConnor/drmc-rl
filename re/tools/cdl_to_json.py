#!/usr/bin/env python3
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument('--cdl', required=True)
ap.add_argument('--bank-bytes', type=int, default=16384)
ap.add_argument('--out', required=True)
args = ap.parse_args()
with open(args.cdl, 'rb') as f: data = f.read()
n = len(data); bank = args.bank_bytes; banks = (n + bank - 1) // bank
cov = []
for b in range(banks):
    chunk = data[b*bank:(b+1)*bank]
    marked = sum(1 for x in chunk if x != 0)
    cov.append({'bank': b, 'marked_bytes': marked, 'total': len(chunk)})
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f: json.dump({'banks': cov, 'total_bytes': n}, f, indent=2)
print(f"Wrote {args.out}")
