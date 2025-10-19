#!/usr/bin/env python3
"""Dump Dr. Mario state planes from a raw NES RAM snapshot.

Usage:
  python tools/ram_planes_dump.py --ram path/to/ram.bin \
      [--offsets envs/specs/ram_offsets.json]

The RAM snapshot must be exactly 0x800 bytes (CPU RAM $0000-$07FF).
"""
import argparse, json, sys, os
import numpy as np

import envs.specs.ram_to_state as ram_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ram', required=True, help='Path to 0x800-byte RAM file')
    ap.add_argument('--offsets', default='envs/specs/ram_offsets.json', help='Offsets JSON')
    ap.add_argument('--state-repr', choices=['extended', 'bitplane'], default='extended')
    args = ap.parse_args()

    ram_specs.set_state_representation(args.state_repr)

    data = open(args.ram, 'rb').read()
    if len(data) != 0x800:
        print(f'Error: RAM file must be 2048 bytes, got {len(data)}', file=sys.stderr)
        sys.exit(2)
    offsets = json.load(open(args.offsets, 'r'))
    planes = ram_specs.ram_to_state(data, offsets)

    # Print per-channel nonzero counts on latest planes
    if ram_specs.STATE_USE_BITPLANES:
        names = [
            'color_R', 'color_Y', 'color_B',
            'virus_mask', 'locked_mask', 'falling_mask',
            'preview_mask', 'clearing_mask', 'empty_mask',
            'gravity', 'lock', 'level',
        ]
    else:
        names = [
            'virus_R', 'virus_Y', 'virus_B',
            'fixed_R', 'fixed_Y', 'fixed_B',
            'fall_R', 'fall_Y', 'fall_B',
            'orientation', 'gravity', 'settle', 'level', 'preview_A', 'preview_B', 'preview_rot',
        ]

    for i, name in enumerate(names[:planes.shape[0]]):
        nz = int(planes[i].sum())
        if name in {'orientation', 'gravity', 'settle', 'level', 'preview_A', 'preview_B', 'preview_rot', 'lock'}:
            val = float(planes[i, 0, 0])
            print(f'{i:02d} {name:12s} nz={nz:3d} val={val:.3f}')
        else:
            print(f'{i:02d} {name:12s} nz={nz:3d}')


if __name__ == '__main__':
    main()
