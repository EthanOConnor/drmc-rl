local out = io.open("re/out/mmc1_io_trace.csv", "w")
out:write("frame,pc,op,addr,val,kind\n")
memory.usememorydomain("System Bus")
local function hex(n, w) return string.format("0x%0"..(w or 4).."X", n) end
while true do
  local frame = emu.framecount()
  local joy1 = memory.read_u8(0x4016)
  local joy2 = memory.read_u8(0x4017)
  out:write(string.format("%d,,READ,0x4016,%d,io\n", frame, joy1))
  out:write(string.format("%d,,READ,0x4017,%d,io\n", frame, joy2))
  for addr=0x8000,0x8000+0x20 do
    local v = memory.read_u8(addr)
    out:write(string.format("%d,,PEEK,%s,%d,prg\n", frame, hex(addr,4), v))
  end
  emu.frameadvance()
end
