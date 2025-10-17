#!/usr/bin/env bash
set -euo pipefail

ROM=${1:-legal_ROMs/Dr. Mario (Japan, USA) (rev0).nes}
OUT_JSON=${2:-re/ghidra/out/symbols.json}

HEADLESS="/opt/homebrew/Caskroom/ghidra/11.4.2-20250826/ghidra_11.4.2_PUBLIC/support/analyzeHeadless"
PROJ_DIR="$(pwd)/re/ghidra/out/ghidra_proj"
rm -rf "$PROJ_DIR" && mkdir -p "$PROJ_DIR"

# Try to use NES loader if extension is present; otherwise Raw Binary with 6502
ROM_BASENAME=$(basename "$ROM")
DOMAIN_PATH="/$ROM_BASENAME"

"$HEADLESS" "$PROJ_DIR" DrMario \
  -import "$ROM" \
  -loader "NES ROM" \
  -max-cpu 2 \
  -overwrite

"$HEADLESS" "$PROJ_DIR" DrMario \
  -process "$DOMAIN_PATH" \
  -scriptPath re/ghidra/scripts \
  -postScript dump_symbols.py "$OUT_JSON" \
  -max-cpu 2 \
  -deleteProject

ls -l "$OUT_JSON"
