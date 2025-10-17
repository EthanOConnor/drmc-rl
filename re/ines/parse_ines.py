#!/usr/bin/env python3
import argparse, json, sys, hashlib, os, struct


def parse_ines_header(data: bytes):
    if data[:4] != b'NES\x1a':
        raise ValueError('Not an iNES file')
    prg_rom_16k = data[4]
    chr_rom_8k = data[5]
    flag6 = data[6]
    flag7 = data[7]
    mapper_low = (flag6 >> 4) & 0x0F
    mapper_high = (flag7 >> 4) & 0x0F
    mapper = mapper_low | (mapper_high << 4)
    mirroring = 'vertical' if (flag6 & 1) else 'horizontal'
    battery = bool(flag6 & 2)
    trainer = bool(flag6 & 4)
    four_screen = bool(flag6 & 8)
    ines2 = (flag7 & 0x0C) == 0x08
    return {
        'mapper': mapper,
        'prg_rom_bytes': prg_rom_16k * 16384,
        'chr_rom_bytes': chr_rom_8k * 8192,
        'mirroring': mirroring, 'battery': battery, 'trainer': trainer, 'four_screen': four_screen,
        'ines2': ines2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rom', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.rom, 'rb') as f:
        data = f.read()
    header = parse_ines_header(data)
    header['sha1'] = hashlib.sha1(data).hexdigest()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(header, f, indent=2)
    print(json.dumps(header, indent=2))


if __name__ == '__main__':
    main()

