#!/usr/bin/env bash
# Run the drmc-rl live agent against a REAL local fcadefbneo instance.
#
# STATUS: the Python<->Lua bridge is tested locally (tests/test_live_bridge.py
# simulates the Lua side). Running inside Fightcade itself is blocked on the
# fightcadeRatings agent's answers to our live-Lua questions, posted at the
# bottom of ../fightcadeRatings/COORDINATION.md ("drmc-rl ask: live-play
# bridge for a Fightcade exhibition", 2026-06-11 -- no reply yet as of
# 2026-06-10 late):
#   1. Is Lua enabled for LIVE netplay matches (vs spectator/replay streams,
#      the proven replay_dump.lua path)? If not: patched local build or
#      training-room mode?
#   2. Does joypad.set on the local side enter the GGPO input stream like
#      controller presses (input-delay frames? NetworkGetInput 5-byte block
#      caveats)?
#   3. Do Lua frame hooks run during rollback re-execution, and is a
#      rollback/replay flag exposed? (Input application is frame-indexed and
#      rollback-safe either way; state export may double-emit.)
#   4. ToS/etiquette for a disclosed bot exhibition match.
# Placeholders below are marked TODO(confirm).
#
# Launch order (server first -- it waits for state.jsonl to appear):
#
#   terminal 1:  tools/fc_local_match.sh server
#   terminal 2:  tools/fc_local_match.sh emulator
#
# The known-good invocation pattern for fcadefbneo+Lua (from
# ../fightcadeRatings/replay_extract.py, spectator path) is:
#   [wine] fcadefbneo.exe <script.lua> quark:stream,nes_drmario,<quarkid>,7001
# For a live match the Fightcade client launches fcadefbneo with a
# quark:served,... argument; TODO(confirm) how to attach a Lua script there
# (CLI arg as above, or the client's lua autoload dir).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
export FC_DRMC_DIR="${FC_DRMC_DIR:-/tmp/drmc_live}"   # IPC dir (state.jsonl/plan.json)
export FC_DRMC_SIDE="${FC_DRMC_SIDE:-1}"              # which player we control
export FC_DRMC_LOG="${FC_DRMC_LOG:-$FC_DRMC_DIR/lua.log}"

# TODO(confirm): path to the Fightcade-bundled emulator. On this machine the
# source tree is at ../fc-fbneo; Fightcade installs its own binary under the
# Fightcade app dir (e.g. ~/Fightcade/emulator/fbneo/fcadefbneo.exe, run via
# the bundled wine on macOS/Linux).
FBNEO="${FBNEO:-$HOME/Fightcade/emulator/fbneo/fcadefbneo.exe}"
WINE="${WINE:-$HOME/Fightcade/wine/bin/wine}"         # TODO(confirm) macOS path
LUA="$REPO/tools/fc_live_agent.lua"

POLICY="${POLICY:-checkpoint}"                        # checkpoint|greedy-cost|random
MARGIN="${MARGIN:-6}"                                 # frames of plan latency margin

mkdir -p "$FC_DRMC_DIR"

case "${1:-help}" in
  server)
    # Start FIRST. Tails state.jsonl (waits for it to exist), plans per pill
    # spawn, writes plan.json. Add --dry-run to observe without injecting.
    exec "$REPO/.venv/bin/python" -m tools.live_agent_server \
      --ipc-dir "$FC_DRMC_DIR" \
      --side "$FC_DRMC_SIDE" \
      --policy "$POLICY" \
      --margin "$MARGIN" \
      --log "$FC_DRMC_DIR/server.log" \
      "${@:2}"
    ;;

  emulator)
    # Start SECOND. For a LIVE match: start the challenge from the Fightcade
    # client UI as usual; the client spawns fcadefbneo with a quark:served,...
    # argument. TODO(confirm): how to make that spawn load $LUA (CLI arg vs
    # autoload). For a LOCAL offline sanity run (no netplay; 1P game where the
    # agent plays by itself), the spectator-style direct invocation is:
    echo "TODO(confirm): exact live-match invocation; offline pattern:"
    echo "  FC_DRMC_DIR=$FC_DRMC_DIR FC_DRMC_SIDE=$FC_DRMC_SIDE \\"
    echo "    \"$WINE\" \"$FBNEO\" \"$LUA\" nes_drmario"
    exit 1
    ;;

  watch)
    # Tail both halves of the bridge.
    exec tail -f "$FC_DRMC_DIR/server.log" "$FC_DRMC_LOG"
    ;;

  *)
    sed -n '2,30p' "$0"
    exit 1
    ;;
esac
