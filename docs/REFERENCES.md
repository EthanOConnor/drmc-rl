# References

## Core Libraries
- Stable-Retro (Farama): https://github.com/Farama-Foundation/stable-retro
- Sample Factory 2.x: https://github.com/alex-petrenko/sample-factory
- TorchRL: https://github.com/pytorch/rl
- PettingZoo (Parallel API): https://pettingzoo.farama.org/
- EnvPool: https://github.com/sail-sg/envpool
- Rich (TUI): https://github.com/Textualize/rich
- Textual (TUI framework): https://github.com/Textualize/textual
- Weights & Biases: https://wandb.ai/ (free tier available)

## Emulation & Cores
- Gym Retro (original): https://github.com/openai/retro
- Libretro cores (Mesen/Nestopia): https://docs.libretro.com/
- FCEUX (Trace/CDL): http://www.fceux.com/
- Mesen: https://www.mesen.ca/
- NesDev wiki (MMC1, controllers): https://www.nesdev.org/wiki/Nesdev_Wiki
- BizHawk (Lua/TAS tooling): https://tasvideos.org/BizHawk

## Dr. Mario Specific
- **Dr. Mario AI (meatfighter)**: https://meatfighter.com/drmarioai/
  - Java bot for Nintaco emulator
  - Source archived in `archive/drmarioai/`
  - Notable: `Searcher.java` (BFS reachability), `DefaultEvaluator.java` (heuristic scoring)
  - Note: Uses memory writes for fast mode (not controller input)
- Dr. Mario Disassembly: submodule at `dr-mario-disassembly/`
