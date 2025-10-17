#!/usr/bin/env python3
import re, sys, csv, os

in_path = sys.argv[1] if len(sys.argv) > 1 else 're/out/mmc1_writes.csv'
out_path = sys.argv[2] if len(sys.argv) > 2 else 're/out/mmc1_writes_norm.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

pat = re.compile(r'^f(?P<frame>\d+)\s+.*?\$..:(?P<pc>[0-9A-F]{4}):\s+([0-9A-F]{2}\s+){0,2}(?P<mn>\w+)\s+\$(?P<addr>[0-9A-F]{4})(?:.*?=\s*#\$(?P<lit>[0-9A-F]{2}))?', re.I)

rows = []
with open(in_path, 'r', errors='ignore') as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        if m.group('mn').upper() != 'STA':
            continue
        addr = int(m.group('addr'), 16)
        if addr < 0x8000:
            continue
        frame = int(m.group('frame'))
        pc = int(m.group('pc'), 16)
        lit = m.group('lit')
        # value heuristic: literal if present, else parse A register from line
        if lit:
            val = int(lit, 16)
        else:
            ma = re.search(r'A:([0-9A-F]{2})', line)
            val = int(ma.group(1), 16) if ma else 0
        rows.append({'frame': frame, 'pc': f"0x{pc:04X}", 'addr': f"0x{addr:04X}", 'val': val})

with open(out_path, 'w', newline='') as out:
    w = csv.DictWriter(out, fieldnames=['frame','pc','addr','val'])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out_path} with {len(rows)} MMC1 write rows")
