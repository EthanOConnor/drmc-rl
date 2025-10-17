# Mesen workflow (preferred for interactive RE)

1. Open your **legally owned** ROM in Mesen (NES).
2. **Debugger → Code/Data Logger**: Enable, then play through:
   - Reset → Title → Level Select → Start Level
   - Let at least 1 pill spawn, move/rotate, let it lock, trigger a clear
3. **Breakpoints**:
   - Add write breakpoints on **$8000/$A000/$C000/$E000** (MMC1 registers) to capture bank switching.
4. **Export** (Debugger → File → Export):
   - **Disassembly** (with control flow)
   - **Labels/Comments**
   - **Trace log** around level start & pill spawn
5. Save all dumps under `re/mesen/out/` and commit **only** the metadata, not ROM.
6. Optional: Use the **Event Viewer** to catch NMI/IRQ and controller reads ($4016).
