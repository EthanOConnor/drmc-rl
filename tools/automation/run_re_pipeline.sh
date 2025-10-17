#!/usr/bin/env bash
set -euo pipefail

cmd=${1:-}

case "$cmd" in
  ines)
    : "${DRMARIO_ROM:?Set DRMARIO_ROM to path of Dr. Mario ROM}"
    mkdir -p re/out
    python3 re/ines/parse_ines.py --rom "$DRMARIO_ROM" --out re/out/ines.json
    echo "Wrote re/out/ines.json"
    ;;
  cdl)
    cdl_file=$(ls re/fceux/out/*.cdl 2>/dev/null | head -n1 || true)
    if [[ -z "${cdl_file}" ]]; then
      echo "No CDL file found in re/fceux/out/. Run FCEUX Code/Data Logger and save .cdl first." >&2
      exit 1
    fi
    mkdir -p re/out
    python3 re/tools/cdl_to_json.py --cdl "$cdl_file" --out re/out/cdl_summary.json
    echo "Wrote re/out/cdl_summary.json from ${cdl_file}"
    ;;
  bankmap)
    if [[ ! -f re/out/mmc1_writes.csv ]]; then
      echo "Missing re/out/mmc1_writes.csv. Generate via emulator trace or BizHawk Lua." >&2
      exit 1
    fi
    python3 re/tools/build_bank_map.py --mmc1-log re/out/mmc1_writes.csv --ines re/out/ines.json --out re/out/bankmap.json
    echo "Wrote re/out/bankmap.json"
    ;;
  *)
    echo "Usage: $0 {ines|cdl|bankmap}" >&2
    exit 2
    ;;
esac

