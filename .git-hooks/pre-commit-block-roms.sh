#!/usr/bin/env bash
set -euo pipefail

# Block committing NES ROMs or legal_ROMs directory content
staged=$(git diff --cached --name-only)

if echo "$staged" | grep -E '\.nes$|^legal_ROMs/'; then
  echo "Error: Attempting to commit ROM files or legal_ROMs content is blocked." >&2
  echo "Please keep ROMs out of git. See docs/LEGAL.md." >&2
  exit 1
fi

exit 0

