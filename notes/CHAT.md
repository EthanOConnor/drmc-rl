# CHAT.md — drmc-rl

Scratchpad for ideas, hypotheses, sketches—things that might become real work later.
Keep it readable and dated.

---

## 2025-12-16 – Initial thoughts from codebase review

- **speedrun_experiment.py refactoring approach**:
  - The file is 5436 lines. Could split into:
    - `training/speedrun/runner.py` – main training loop
    - `training/speedrun/viewer.py` – TUI/visualization (the `_monitor_worker` is ~800 lines alone)
    - `training/speedrun/devices.py` – MLX device introspection utilities
    - `training/speedrun/metrics.py` – reward component tracking
  - Would need careful import management to avoid circular deps.

- **MLX vs PyTorch unification**:
  - Code supports both MLX and PyTorch backends with parallel implementations.
  - Consider: is this worth maintaining long-term, or should we pick one?
  - MLX is Apple-only but very fast on M-series. PyTorch is universal.
  - Possible: keep MLX for local dev on Mac, PyTorch for cluster training.

- **Placement policy replanning**:
  - Current: single inference per spawn, reuse logits for replanning.
  - Alternative: could do lightweight re-inference on significant board changes during execution.
  - Trade-off: accuracy vs latency. Current approach is probably fine for speedrunning.

- **Risk-aware play for 2-player vs mode**:
  - The risk_tau conditioning is designed for single-player speedrun.
  - For 2-player: might want opponent-aware risk adjustment.
  - Example: if opponent is about to clear, take higher risk to catch up.
  - Would need opponent state in observation space.

---

## Ideas to explore

- **Imitation learning from TAS recordings**:
  - TAS inputs exist for Dr. Mario speedruns.
  - Could use for behavior cloning warm-start before RL.
  - Need to parse .fm2/.bk2 movie files.

- **Curriculum via virus count**:
  - Start with 4 viruses (trivial), increase as success rate improves.
  - Could tie to evaluator confidence: if predicted clear time variance is low, increase difficulty.

- **Ensemble of policy heads**:
  - Train all 3 heads (dense, shift_score, factorized) and ensemble at inference.
  - Might improve robustness vs single head.
