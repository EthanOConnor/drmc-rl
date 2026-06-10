# References

## Local Umbrella Workspace

- `/Users/ethan/dev/drmario/README.md`: local map of sibling Dr. Mario projects.
- `game_engine/`: submodule mount for the standalone `drmario-native` project.
- `dr-mario-disassembly/`: pinned repo-local disassembly submodule.
- `../disassembly/`: shared local reference checkouts for ad hoc cross-project
  work when available.

## Dr. Mario Specific

- Dr. Mario Disassembly: pinned in `dr-mario-disassembly/`; shared copies may be
  available under the umbrella workspace.
- Dr. Mario AI (meatfighter): https://meatfighter.com/drmarioai/
  - Useful references: `Searcher.java` for BFS reachability and
    `DefaultEvaluator.java` for heuristic scoring.
  - Note: that bot uses memory writes for fast mode, not controller input.

## Current Training Libraries

- Gymnasium: https://gymnasium.farama.org/
- PyTorch: https://pytorch.org/
- TorchRL: https://github.com/pytorch/rl
- Rich: https://github.com/Textualize/rich
- Weights & Biases: https://wandb.ai/

## Emulation and Parity

- Libretro cores: https://docs.libretro.com/
- Mesen: https://www.mesen.ca/
- FCEUX: http://www.fceux.com/
- NesDev wiki: https://www.nesdev.org/wiki/Nesdev_Wiki
- BizHawk: https://tasvideos.org/BizHawk
- Stable-Retro: https://github.com/Farama-Foundation/stable-retro
- Gym Retro: https://github.com/openai/retro

## Historical or Non-Default Directions

- PettingZoo: https://pettingzoo.farama.org/
- EnvPool: https://github.com/sail-sg/envpool
- Textual: https://github.com/Textualize/textual

Keep these out of default setup docs unless they become active project scope
again.
