# RE Automation

Run the scripted parts of the RE pipeline from here. GUI steps (FCEUX/Mesen/Ghidra) are minimized but may be required for CDL and label exports.

## Parse iNES header
- `DRMARIO_ROM=legal_ROMs/Dr.\ Mario\ (Japan,\ USA)\ (rev0).nes bash tools/automation/run_re_pipeline.sh ines`

## CDL summary (after you save `re/fceux/out/*.cdl`)
- `bash tools/automation/run_re_pipeline.sh cdl`

## Bank map (after you have `re/out/mmc1_writes.csv`)
- `bash tools/automation/run_re_pipeline.sh bankmap`

Outputs go under `re/out/`.
