-- Live-play bridge (Lua side) for the drmc-rl Dr. Mario agent in Fightcade
-- fcadefbneo. Counterpart: tools/live_agent_server.py (protocol documented
-- there).
--
-- Each frame (registerbefore, so injected inputs apply to the frame about to
-- execute) this script:
--   1. reads our side's Dr. Mario state from NES RAM,
--   2. appends a state line to <FC_DRMC_DIR>/state.jsonl when a new pill
--      spawns, every 4 frames while awaiting a plan, and every 8 frames while
--      a plan is active (server-side desync verification),
--   3. re-reads <FC_DRMC_DIR>/plan.json (tiny file; written atomically by the
--      server) and applies buttons[frame - start_frame] via joypad.set.
--
-- The button script is FRAME-INDEXED by fba.framecount(): if GGPO rollback
-- re-executes a frame, the same buttons are applied again, so input
-- application is deterministic under rollback. (State *export* may double-emit
-- during rollback re-execution; the server tolerates duplicate lines.)
--
-- A plan is only "armed" if it is seen at or before its start_frame; a plan
-- that arrives late is ignored (the server notices the continuing "awaiting"
-- state lines and re-plans from fresh micro-state).
--
-- Env config:
--   FC_DRMC_DIR   IPC directory shared with the server (default ".")
--   FC_DRMC_SIDE  1 or 2: which player this agent controls (default 1)
--   FC_DRMC_LOG   optional log file path
--
-- Conventions mirror fightcadeRatings/replay_dump.lua, which is proven to run
-- under Fightcade's FBNeo Lua host (fba.* API, memory.readbyte, io, os.getenv).

local DIR = os.getenv("FC_DRMC_DIR") or "."
local SIDE = tonumber(os.getenv("FC_DRMC_SIDE") or "1")
local LOGPATH = os.getenv("FC_DRMC_LOG")

local STATE_PATH = DIR .. "/state.jsonl"
local PLAN_PATH = DIR .. "/plan.json"

local AWAIT_EMIT_EVERY = 4  -- frames between state lines while awaiting a plan
local VERIFY_EMIT_EVERY = 8 -- frames between state lines while a plan is active

local EMU = fba or emu

-- RAM addresses (dr-mario-disassembly defines/drmario_ram*.asm).
local ADDR_FRAMECOUNTER = 0x0043
local ADDR_MODE = 0x0046 -- 4 = gameplay
local BASE = (SIDE == 2) and 0x0380 or 0x0300
local OPP = (SIDE == 2) and 0x0300 or 0x0380
local FIELD = (SIDE == 2) and 0x0500 or 0x0400

-- Offsets into the per-player $30-byte block.
local O_PILL1, O_PILL2 = 0x01, 0x02       -- fallingPill colors
local O_X, O_Y = 0x05, 0x06               -- fallingPill col / row-from-bottom
local O_SPEEDUPS, O_SPEEDSETTING = 0x0A, 0x0B
local O_PILLS_LO, O_PILLS_HI = 0x10, 0x11 -- pillsCounter BCD
local O_SC, O_HV = 0x12, 0x13             -- speedCounter / horVelocity
local O_LEVEL = 0x16
local O_NEXTACTION = 0x17                 -- 0 = pillFalling
local O_ATTACK = 0x18
local O_NEXT1, O_NEXT2 = 0x1A, 0x1B       -- preview colors
local O_ROT = 0x25                        -- fallingPillRotation

local out = assert(io.open(STATE_PATH, "a"))
local logf = nil
if LOGPATH then logf = io.open(LOGPATH, "a") end

local function log(msg)
  if logf then
    logf:write(string.format("[%d] %s\n", EMU.framecount(), msg))
    logf:flush()
  end
end

local function rb(addr)
  local ok, v = pcall(memory.readbyte, addr)
  if ok and v ~= nil then return v end
  return 0
end

local function bcd(v)
  return math.floor(v / 16) * 10 + (v % 16)
end

local function field_hex()
  local parts = {}
  for i = 0, 127 do
    parts[#parts + 1] = string.format("%02x", rb(FIELD + i))
  end
  return table.concat(parts, "")
end

-- ---------------------------------------------------------------------------
-- Input injection. FBNeo Lua: joypad.set(table) keyed by full input names
-- ("P1 Up", "P1 Button A", ...; src/burner/luaengine.cpp joypad_set). false
-- forces release, so we own all six buttons on OUR side every frame and never
-- touch the other side's inputs.
-- Button byte bits (DrMarioPool.cpp): 0x01 Right, 0x02 Left, 0x04 Down,
-- 0x40 B, 0x80 A.
-- ---------------------------------------------------------------------------

local P = "P" .. SIDE .. " "
local K_RIGHT, K_LEFT, K_DOWN, K_UP = P .. "Right", P .. "Left", P .. "Down", P .. "Up"
local K_B, K_A = P .. "Button B", P .. "Button A"

local joypad_warned = false

local function bit_set(mask, bit)
  return math.floor(mask / bit) % 2 == 1
end

local function apply_buttons(mask)
  local t = {}
  t[K_RIGHT] = bit_set(mask, 0x01)
  t[K_LEFT] = bit_set(mask, 0x02)
  t[K_DOWN] = bit_set(mask, 0x04)
  t[K_UP] = false
  t[K_B] = bit_set(mask, 0x40)
  t[K_A] = bit_set(mask, 0x80)
  local ok, err = pcall(joypad.set, t)
  if not ok and not joypad_warned then
    joypad_warned = true
    log("joypad.set failed: " .. tostring(err))
  end
end

-- ---------------------------------------------------------------------------
-- Plan file polling/parsing. Re-read every frame (a few KB at most; cheap at
-- 60fps -- replay_dump.lua proves per-frame io is tolerable). Parsed only when
-- the raw content changes.
-- ---------------------------------------------------------------------------

local plan = nil       -- {id, spawn, start, n, buttons}
local plan_raw = nil
local armed_id = nil   -- plan ids seen at/before their start_frame
local dead_id = nil    -- plan ids that arrived too late to start

local function read_plan()
  local f = io.open(PLAN_PATH, "r")
  if not f then return end
  local raw = f:read("*a")
  f:close()
  if raw == nil or raw == plan_raw then return end
  local id = string.match(raw, '"plan_id"%s*:%s*(%d+)')
  local spawn = string.match(raw, '"spawn"%s*:%s*(%d+)')
  local start = string.match(raw, '"start_frame"%s*:%s*(%d+)')
  local btn_str = string.match(raw, '"buttons"%s*:%s*%[([%d,%s]*)%]')
  if not (id and spawn and start and btn_str) then return end -- partial; retry
  plan_raw = raw
  local buttons = {}
  for b in string.gmatch(btn_str, "%d+") do
    buttons[#buttons + 1] = tonumber(b)
  end
  plan = {
    id = tonumber(id),
    spawn = tonumber(spawn),
    start = tonumber(start),
    n = #buttons,
    buttons = buttons,
  }
end

-- ---------------------------------------------------------------------------
-- State export.
-- ---------------------------------------------------------------------------

local function emit_state(frame, pc, na)
  out:write(string.format(
    '{"f":%d,"nesf":%d,"mode":4,"side":%d,"pc":%d,"na":%d,'
      .. '"pill":[%d,%d],"prev":[%d,%d],"x":%d,"y":%d,"rot":%d,'
      .. '"sc":%d,"hv":%d,"spd":%d,"spdups":%d,"level":%d,'
      .. '"atk_us":%d,"atk_opp":%d,"field":"%s"}\n',
    frame, rb(ADDR_FRAMECOUNTER), SIDE, pc, na,
    rb(BASE + O_PILL1), rb(BASE + O_PILL2),
    rb(BASE + O_NEXT1), rb(BASE + O_NEXT2),
    rb(BASE + O_X), rb(BASE + O_Y), rb(BASE + O_ROT),
    rb(BASE + O_SC), rb(BASE + O_HV),
    rb(BASE + O_SPEEDSETTING), rb(BASE + O_SPEEDUPS), rb(BASE + O_LEVEL),
    rb(BASE + O_ATTACK), rb(OPP + O_ATTACK),
    field_hex()))
  out:flush()
end

-- ---------------------------------------------------------------------------
-- Per-frame hook.
-- ---------------------------------------------------------------------------

log(string.format("fc_live_agent start: side=%d dir=%s rom=%s",
  SIDE, DIR, tostring(EMU.romname and EMU.romname() or "?")))

local prev_na = nil

EMU.registerbefore(function()
  local frame = EMU.framecount()
  local mode = rb(ADDR_MODE)
  if mode ~= 0x04 then
    -- Not in gameplay (menus, level intro, ...): leave inputs to the human.
    prev_na = nil
    return
  end

  read_plan()

  local na = rb(BASE + O_NEXTACTION)
  local pc = bcd(rb(BASE + O_PILLS_LO)) + 100 * bcd(rb(BASE + O_PILLS_HI))

  -- Arm a freshly seen plan only if its start frame is still reachable.
  if plan and plan.id ~= armed_id and plan.id ~= dead_id then
    if frame <= plan.start then
      armed_id = plan.id
      log(string.format("plan %d armed: spawn=%d start=%d n=%d",
        plan.id, plan.spawn, plan.start, plan.n))
    else
      dead_id = plan.id
      log(string.format("plan %d arrived late (frame=%d > start=%d); ignored",
        plan.id, frame, plan.start))
    end
  end

  local active = plan ~= nil and plan.id == armed_id and plan.spawn == pc
    and frame < plan.start + plan.n
  local covers = active and frame >= plan.start

  if na == 0 and covers then
    apply_buttons(plan.buttons[frame - plan.start + 1])
  else
    apply_buttons(0) -- neutral; we still own our side's buttons
  end

  -- State export: new decision, awaiting cadence, or verification cadence.
  local spawn_edge = (na == 0 and prev_na ~= 0)
  local awaiting = (na == 0) and not active
  local verify = (na == 0) and covers and frame > plan.start
    and (frame - plan.start) % VERIFY_EMIT_EVERY == 0
  if spawn_edge or (awaiting and frame % AWAIT_EMIT_EVERY == 0) or verify then
    emit_state(frame, pc, na)
  end

  prev_na = na
end)
