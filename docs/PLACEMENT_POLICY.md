# Placement Policy & SMDP-PPO Training

This document describes the single-inference placement policy system and SMDP-PPO training implementation.

## Overview

The placement policy system replaces per-frame REINFORCE training with:
- **One forward pass per spawn**: Policy inference occurs once per pill spawn, not every frame
- **Macro-action learning**: Each decision spans τ frames until pill locks
- **SMDP-PPO**: Credit assignment uses γ^τ discounting for temporally extended actions
- **Masked actions**: Invalid placements are masked at inference time

## Components

### 1. Policy Heads

Three interchangeable architectures are provided in `models/policy/placement_heads.py`:

#### A. Dense Conv Heatmaps (Recommended)
```python
head_type = "dense"
```
- Simplest strong baseline
- Direct 4×16×8 heatmap generation
- FiLM conditioning on current + preview pill colors
- Fast parallel inference

#### B. Shift-and-Score
```python
head_type = "shift_score"
```
- Local feature scorer
- Pairs anchor features with partner-shifted features
- Color-order symmetric by design
- Compact representation

#### C. Factorized (anchor → orientation)
```python
head_type = "factorized"
```
- Hierarchical selection: pick cell (i,j) then orientation o
- Smallest parameter count
- Slightly more complex credit assignment

All heads:
- Accept `[B, C, 16, 8]` board state
- Embed current + preview pill colors via Deep-Sets (order-invariant per pill)
- Apply CoordConv for spatial awareness
- Output `[B, 4, 16, 8]` logit maps + value estimates

### 2. Masked Distribution

`models/policy/placement_dist.py` provides:
- `MaskedPlacementDist`: Categorical distribution over 512 placement actions
- Invalid action masking (sets logits to -∞)
- Safe softmax with numerical guards
- Gumbel-Top-k sampling for exploration

### 3. Decision-Level Rollout Buffer

`training/rollout/decision_buffer.py` stores:
- One transition per **decision** (not frame)
- Each decision includes:
  - `obs`: Board state at decision time
  - `mask`: Feasibility mask [4, 16, 8]
  - `pill_colors`: Current pill [2]
  - `preview_pill_colors`: Preview pill [2]
  - `action`: Selected placement
  - `log_prob`, `value`: Policy outputs
  - `tau`: Frame duration
  - `reward`: Cumulative reward over τ frames
  - `obs_next`: State after placement

### 4. SMDP-PPO Trainer

`training/algo/ppo_smdp.py` implements:
- PPO with SMDP discounting
- GAE with Γ_t = γ^τ_t
- Entropy annealing
- Scripted curriculum support (optional; configured via `curriculum.*` in the runner config)

## Mathematical Formulation

### SMDP Discounting
For a decision at time t spanning τ_t frames:
```
Γ_t = γ^{τ_t}
```

### GAE Backward Pass
```python
δ_t = R_t + Γ_t * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = δ_t + Γ_t * λ * (1 - done_t) * A_{t+1}
ret_t = R_t + Γ_t * ret_{t+1} * (1 - done_t)
```

### PPO Loss
```python
ratio = exp(log_π(a_t|s_t) - log_π_old(a_t|s_t))
L_policy = -min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t)
L_value = MSE(V(s_t), ret_t)
L_total = L_policy + β_v * L_value - β_h * H[π(·|s_t)]
```

## Usage

### Quick Start

```python
from models.policy.placement_heads import PlacementPolicyNet
from models.policy.placement_dist import MaskedPlacementDist

# Create policy
net = PlacementPolicyNet(
    in_channels=4,  # `bitplane_bottle`; see `env.state_repr`
    head_type="dense",
    pill_embed_dim=32,
)

# Forward pass
board = ...  # [B, C, 16, 8]
pill_colors = ...  # [B, 2] current pill color indices
preview_pill_colors = ...  # [B, 2] preview pill color indices
mask = ...  # [B, 4, 16, 8] boolean

logits_map, value = net(board, pill_colors, preview_pill_colors, mask)

# Sample action
dist = MaskedPlacementDist(logits_map, mask)
action_idx, log_prob = dist.sample()
```

### Training

```bash
# Unified runner (recommended)
# Note: requires `--core` (QuickNES/Mesen) and a local ROM path for libretro.
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui headless \
  --backend libretro --core quicknes --rom-path legal_ROMs/DrMario.nes
```

Interactive board visualization + speed control:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml --ui debug \
  --backend libretro --core quicknes --rom-path legal_ROMs/DrMario.nes \
  --env-id DrMarioPlacementEnv-v0 --num_envs 1
```

### Configuration

Edit `training/configs/smdp_ppo.yaml`:

```yaml
smdp_ppo:
  lr: 3.0e-4
  gamma: 0.995
  gae_lambda: 0.95
  clip_epsilon: 0.2
  
  # Policy head
  head_type: "dense"  # or "shift_score", "factorized"
  pill_embed_dim: 32
  
  # Rollout
  decisions_per_update: 512
  num_epochs: 4
  minibatch_size: 128
  
  # Exploration
  entropy_coef: 0.01
  entropy_schedule_end: 0.003
  entropy_schedule_steps: 1000000
```

## Testing

Run the unit tests:

```bash
# Core placement policy tests
pytest tests/test_placement_policy.py -v

# Smoke tests
pytest tests/test_placement_smoke.py -v
```

Tests cover:
1. **Mask & numerics**: Softmax normalization, invalid action handling
2. **Color invariance**: Deep-Sets embedding symmetry
3. **Geometry alignment**: Sensible placement targeting
4. **SMDP math**: GAE-SMDP correctness, τ=1 recovery
5. **Distribution API**: Log-prob and entropy consistency

## Performance Metrics

Track these during training:

### Decision-Level
- `decisions/sec`: Decisions made per second
- `policy/entropy`: Policy exploration level
- `policy/kl`: KL divergence from old policy
- `policy/clip_frac`: Fraction of updates clipped

### Episode-Level
- `drm/viruses_per_ep`: Viruses cleared
- `drm/lines_per_ep`: Lines cleared
- `train/return_mean`: Average episode return

### Placement-Specific
- `placements/options`: Mean feasible actions per spawn
- `placements/tau_mean`: Mean frames per placement
- `placements/mask_sparsity`: Fraction of invalid actions

## Integration Points

### Placement Wrapper

The placement macro-environment (`envs/retro/placement_env.py`) provides:
- `info["placements/feasible_mask"]`: Boolean mask [4, 16, 8]
- `info["placements/legal_mask"]`: Boolean mask [4, 16, 8] (in-bounds-only)
- `info["placements/costs"]`: Float costs [4, 16, 8] (frames to lock; `inf` if unreachable)
- `info["next_pill_colors"]`: Current pill color indices [2]
- `info["preview_pill"]`: HUD preview pill metadata (colors/rotation; raw NES encoding)
- `info["placements/spawn_id"]`: Pill spawn counter for cache invalidation
- `info["placements/tau"]`: Frames consumed by the macro step (SMDP duration)

### Single Inference per Spawn

The runner/agent must:
1. Cache logits keyed by `spawn_id`
2. Reuse cached logits if execution diverges
3. Apply updated feasibility mask on replanning

Example pattern:
```python
# Decision time
spawn_id = info.get("placements/spawn_id")
if spawn_id != last_spawn_id:
    # New spawn: run inference
    logits, value = policy(obs, colors, preview_colors, mask)
    cache[spawn_id] = logits
    last_spawn_id = spawn_id
else:
    # Replan: reuse cached logits
    logits = cache[spawn_id]
    # Apply updated mask
    dist = MaskedPlacementDist(logits, updated_mask)
```

## Hyperparameter Tuning

### Learning Rate
- Start: `3e-4`
- Reduce if policy KL > 0.05 consistently
- Increase if learning stalls

### Entropy Coefficient
- Initial: `0.01`
- Anneal to: `0.003`
- Lower values → more deterministic policy

### Gamma (Discount)
- Standard: `0.995`
- Higher for longer episodes
- Lower if credit assignment is noisy

### GAE Lambda
- Standard: `0.95`
- Higher → more Monte Carlo (high variance)
- Lower → more TD (low variance, high bias)

### Minibatch Size
- Smaller → more updates, noisier gradients
- Larger → fewer updates, more stable
- Balance with GPU memory

## Curriculum Learning

The unified runner supports a simple scripted curriculum (enabled in
`training/configs/smdp_ppo.yaml` by default):

- **Synthetic levels**: `-10..0` are represented by setting `env.level` negative.
- **Match-count staging** (0 viruses; applied by patching the bottle RAM at reset time):
  - `-10`: 1 match (“any match”)
  - `-9`: 2 matches
  - …
  - `-4`: 7 matches
- **Virus-count staging** (applied by patching the bottle RAM at reset time):
  - `-3`: 1 virus
  - `-2`: 2 viruses
  - `-1`: 3 viruses
  - `0`: 4 viruses (vanilla level 0)
- **Advancement**: stay on the current curriculum level until the rolling clear
  rate over `window_episodes` reaches `success_threshold`, then advance by +1.
- **Rehearsal**: after advancing, sample a lower level with probability
  `rehearsal_prob` to maintain performance.

Disable the curriculum with:

```bash
python -m training.run --cfg training/configs/smdp_ppo.yaml --override curriculum.enabled=false
```

## Troubleshooting

### Policy Collapse
**Symptom**: Entropy → 0 too quickly
**Fix**: Increase `entropy_coef`, reduce `entropy_schedule_steps`

### Value Drift
**Symptom**: Value loss explodes
**Fix**: Clip value loss, reduce `gamma`, normalize rewards

### Mask Errors
**Symptom**: Sampled invalid actions
**Fix**: Check wrapper mask generation, verify masking logic

### Slow Training
**Symptom**: Low decisions/sec
**Fix**:
- Ensure the native reachability helper is built: `python -m tools.build_reach_native` (required for practical speed)
- Increase `num_envs`, reduce `decisions_per_update`

## References

- **SMDP-PPO**: Extends PPO to Semi-Markov Decision Processes
- **Deep Sets**: Zaheer et al., "Deep Sets", NeurIPS 2017
- **FiLM**: Perez et al., "FiLM: Visual Reasoning with Feature-wise Linear Modulation", AAAI 2018
- **CoordConv**: Liu et al., "An Intriguing Failing of Convolutional Neural Networks", NeurIPS 2018
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using GAE", ICLR 2016

## License

Same as parent project.
