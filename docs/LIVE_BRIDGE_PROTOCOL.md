# Live-Play Bridge Protocol (agent server ↔ any frontend)

The drmc-rl live agent is frontend-agnostic: anything that can export Dr.
Mario RAM state per frame and apply per-frame NES buttons can host it —
the fcadefbneo Lua client (`tools/fc_live_agent.lua`), a patched build, or
a native client (e.g. the macOS Fightcade client in development). This
file is the contract. Server side: `tools/live_agent_server.py` (planning
p50 17 ms, p95 33 ms spawn→plan).

## Transport

File-based IPC in `FC_DRMC_DIR` (default `/tmp/drmc_live`). Two files:

- `state.jsonl` — frontend appends one JSON line per export (see cadence).
- `plan.json` — server writes atomically (tmp+rename); frontend re-reads
  cheaply each frame and applies.

A native client may prefer a socket; the server accepts `--socket
unix:<path>` as a drop-in alternative carrying the same JSON messages
(state lines in, plan objects out). [If not yet present, request it — the
file path is the reference implementation.]

## State line (frontend → server)

```json
{"f": 12345,            // emulated frame counter ($0043 full or low byte ok; monotonic preferred)
 "mode": 4,             // $0046; server only acts when 4 (gameplay)
 "side": 1,             // which player the bot controls (1|2)
 "na": 0,               // p{side}_nextAction ($0317/$0397): 0 = pill falling
 "pc": 137,             // pill spawn counter ($0310 BCD low or any per-spawn-changing id)
 "pill": [c1, c2],      // $0301/$0302 (+0x80 for P2): raw NES colors 0..2
 "prev": [c1, c2],      // $031A/$031B (+0x80)
 "x": 3, "y": 12,       // falling pill col $0305, row-from-bottom $0306 (+0x80)
 "rot": 0,              // $0325 (+0x80)
 "sc": 5, "hv": 0,      // speed counter / hor velocity ($0312/$0313 family, +0x80)
 "spd": 2, "spdups": 4, // speed setting $030B, speedups $030A (+0x80)
 "field": "<hex 128B>", // own bottle $0400 (P1) / $0500 (P2), row 0 = top
 "nesf": 102           // NES frameCounter low byte $0043 (parity source)
}
```

Export cadence: on every spawn (`na` enters 0 with a new `spawn` id), then
every ~4 frames while no active plan covers the current frame, then every
~8 frames as verification while a plan runs. Duplicate/rolled-back frames
are fine — the server collapses to the freshest line per spawn id.

## Plan object (server → frontend)

```json
{"plan_id": 42,
 "spawn": 137,                // pc of the spawn this plan answers
 "start_frame": 12351,        // first frame the script applies to
 "buttons": [b0, b1, ...]}    // one button byte per frame (mask layout below)
```

Button byte = DrMarioPool mask layout (NOT the NES standard order):
Right=0x01, Left=0x02, Down=0x04, B=0x40, A=0x80 — matches
`DrMarioPool::buttons_mask_from_reach_action` and `GameLogic` button
constants; frontends map these to their own pad representation. Plan
objects also carry "spawn": <pc> identifying which spawn the plan is for.

Frontend behavior:
- Apply `buttons[current_frame - start_frame]` when in range; neutral
  otherwise. Frame-indexed lookup makes rollback re-execution consistent
  (re-executed frame F re-applies the same byte).
- Under GGPO, inject only on non-rollback frames (rollback frames re-read
  confirmed inputs from the GGPO stream); `bSkipPerfmonUpdates` is the
  C-side rollback marker in fcadefbneo.
- GGPO input delay applies after injection: if the client knows `iDelay`,
  report it once as `{"delay": n}` in a state line; the server shifts
  `start_frame` accordingly.
- Plans whose `start_frame` already passed must be discarded (the server
  detects the miss from subsequent state lines and replans from the fresh
  micro-state — mid-fall replanning is exact).

## Safety rules

- Only ever write the bot side's buttons.
- Stop injecting when `mode != 4` (menus/level select are the human's).
- The server logs every decision (pose, cost, options, latency) to
  `FC_DRMC_DIR/server.log`.

## Known constraints from the Fightcade side

(Verified from fcadefbneo source by the fightcadeRatings project — see
`../fightcadeRatings/COORDINATION.md`, 2026-06-11 answers.)
- Live-netplay Lua is disabled in stock Fightcade (`kNetLua=0` per quark);
  spectator Lua works. Hence: native client / virtual controller is the
  recommended injection path; a patched build needs the `kNetLua` flag AND
  moving the Lua joypad hook before `NetworkGetInput` (else desync).
- Etiquette bar for any online match: Fightcade staff blessing, private
  casual room, disclosed bot, human supervising.
