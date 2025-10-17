-- FCEUX Lua script: trace writes to MMC1 registers ($8000-$FFFF) to CSV
-- Usage: FCEUX -> Tools -> Lua Console -> Load this script
-- Output: re/out/mmc1_writes.csv (relative to ROM working dir)

-- Resolve output path relative to this script, and ensure directory exists
local function script_dir()
  local src = debug.getinfo(1, "S").source or "@."
  local path = src:match("^@(.+)$") or "."
  return path:match("(.*/)") or "./"
end

local base = script_dir()  -- e.g., .../re/fceux/
local outdir = base:gsub("/fceux/?$", "/out")
local outfile_path = outdir .. "/mmc1_writes.csv"
local outfile = io.open(outfile_path, "w")
if not outfile then
  -- Fallback: try current working directory
  outfile_path = "mmc1_writes.csv"
  outfile = io.open(outfile_path, "w")
end
assert(outfile, "failed to open mmc1_writes.csv for writing")
outfile:write("frame,pc,addr,value\n")
outfile:flush()

local function on_write(addr, value)
    local frame = emu.framecount()
    local regs = emu.getregisters()
    local pc = regs["pc"] or 0
    outfile:write(string.format("%d,0x%04X,0x%04X,0x%02X\n", frame, pc, addr, value))
end

-- Register callbacks for full MMC1 range ($8000-$FFFF)
memory.registerwrite(0x8000, 0xFFFF, on_write)

emu.addgamegenie(" ") -- no-op to keep script resident
