#!/usr/bin/env python3
import argparse, json, os
TEMPLATE = '''from dataclasses import dataclass
@dataclass(frozen=True)
class BottleSpec:
    base_addr: int; width: int; height: int; stride: int; encoding: str
BOTTLE = BottleSpec(base_addr={bbase}, width={bwidth}, height={bheight}, stride={bstride}, encoding="{benc}")
FALLING_PILL = {falling}
PREVIEW_PILL = {preview}
GRAVITY_LOCK = {gravlock}
TIMERS = {timers}
LEVEL = {level}
'''
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='inp', required=True)
ap.add_argument('--out', dest='outp', required=True)
a = ap.parse_args()
m = json.load(open(a.inp,'r'))
hx = lambda x: int(x,16) if isinstance(x,str) and x.startswith('0x') else int(x)
content = TEMPLATE.format(
  bbase=hx(m['bottle']['base_addr']), bwidth=int(m['bottle']['width']),
  bheight=int(m['bottle']['height']), bstride=int(m['bottle']['stride']),
  benc=m['bottle']['encoding'],
  falling={k: hx(v) for k,v in m['falling_pill'].items()},
  preview={k: hx(v) for k,v in m['preview_pill'].items()},
  gravlock={k: hx(v) for k,v in m['gravity_lock'].items()},
  timers={k: hx(v) for k,v in m['timers'].items()},
  level={k: hx(v) for k,v in m['level'].items()},
)
os.makedirs(os.path.dirname(a.outp), exist_ok=True)
open(a.outp,'w').write(content)
print(f"Wrote {a.outp}")
