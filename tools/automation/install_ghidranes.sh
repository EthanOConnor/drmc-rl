#!/usr/bin/env bash
set -euo pipefail

# Install a Ghidra NES/iNES loader extension from a ZIP.
# Usage: bash tools/automation/install_ghidranes.sh /path/to/GhidraNES.zip

ZIP_PATH=${1:-}
if [[ -z "$ZIP_PATH" || ! -f "$ZIP_PATH" ]]; then
  echo "Usage: $0 /path/to/GhidraNES.zip" >&2
  exit 2
fi

GHIDRA_BASE=$(readlink /opt/homebrew/bin/ghidraRun || true)
if [[ -z "$GHIDRA_BASE" ]]; then
  echo "ghidraRun not found; install ghidra cask first." >&2
  exit 1
fi
GHIDRA_DIR=$(dirname "$GHIDRA_BASE")
GHIDRA_DIR=$(dirname "$GHIDRA_DIR")
EXT_DIR="$GHIDRA_DIR/Ghidra/Extensions"
mkdir -p "$EXT_DIR"

unzip -q "$ZIP_PATH" -d "$EXT_DIR"
echo "Installed extension into $EXT_DIR"
echo "You can verify in Ghidra: File -> Install Extensions, or run headless import again."

