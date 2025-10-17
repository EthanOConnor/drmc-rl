# Ghidra setup for NES (6502) with banking

1. Install Ghidra and the **GhidraNes** extension to load iNES ROMs.
2. Import the `.nes` file (File → New Project → drag ROM → choose **NES ROM**).
3. Run **Auto Analyze**.
4. Import labels exported from Mesen/FCEUX to improve symbolization.
5. Start at vectors ($FFFA-$FFFF) to locate **RESET**, **NMI**, and **IRQ**; build call graphs from there.
6. Identify MMC1 writes at $8000/$A000/$C000/$E000 and annotate bank switching.
