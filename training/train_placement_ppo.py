#!/usr/bin/env python3
"""Simplified placement policy training with SMDP-PPO.

Integrates with existing placement environment and visualization.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import envs.specs.ram_to_state as ram_specs
import torch

# Import existing components
from models.policy.placement_heads import PlacementPolicyNet
from models.policy.placement_dist import MaskedPlacementDist
from training.rollout.decision_buffer import DecisionRolloutBuffer, DecisionStep
from training.utils.checkpoint_io import checkpoint_path, save_checkpoint

# Import environment registration
from envs.retro.register_env import register_placement_env_id
from envs.retro.placement_actions import action_count as get_env_action_count


def main():
    parser = argparse.ArgumentParser(description="Train placement policy with SMDP-PPO")
    
    # Environment
    parser.add_argument("--rom-path", type=str, default=None, help="Path to Dr. Mario ROM")
    parser.add_argument("--core-path", type=str, default=None, help="Path to libretro core")
    parser.add_argument("--backend", type=str, default="libretro", choices=["libretro", "mock"])
    
    # Training
    parser.add_argument("--total-decisions", type=int, default=10000, help="Total decisions to collect")
    parser.add_argument("--decisions-per-update", type=int, default=256, help="Decisions before update")
    parser.add_argument("--num-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size")
    
    # Policy
    parser.add_argument("--head", type=str, default="dense", 
                       choices=["dense", "shift_score", "factorized"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--entropy-decay", type=float, default=0.999)
    parser.add_argument("--value-coef", type=float, default=0.5)
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N updates")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save every N decisions")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.rom_path:
        os.environ["DRMARIO_ROM_PATH"] = args.rom_path
    if args.core_path:
        os.environ["DRMARIO_CORE_PATH"] = args.core_path
        
    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    # Register and create environment
    register_placement_env_id()
    
    import gymnasium as gym
    
    # Create placement environment by wrapping base env
    from envs.retro.drmario_env import DrMarioRetroEnv
    from envs.retro.placement_wrapper import DrMarioPlacementEnv
    
    base_env = DrMarioRetroEnv(backend=args.backend, obs_mode="state")
    env = DrMarioPlacementEnv(base_env)
    
    # Get observation shape from env
    obs, info = env.reset()
    obs_arr = _obs_to_array(obs)
    obs_shape = obs_arr.shape if hasattr(obs_arr, 'shape') else (12, 16, 8)
    in_channels = obs_shape[0] if len(obs_shape) == 3 else 12
    
    print(f"Observation shape: {obs_shape}, in_channels: {in_channels}")
    
    # Create policy network
    policy = PlacementPolicyNet(
        in_channels=in_channels,
        head_type=args.head,
        pill_embed_dim=32,
    ).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # Create rollout buffer
    buffer = DecisionRolloutBuffer(
        capacity=args.decisions_per_update * 2,
        obs_shape=obs_shape,
        num_envs=1,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\\nTraining Configuration:")
    print(f"  Policy head: {args.head}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Decisions per update: {args.decisions_per_update}")
    print(f"  Total decisions: {args.total_decisions}")
    print(f"  Checkpoint dir: {checkpoint_dir}\\n")
    
    # Training loop
    obs, info = env.reset()
    decision_count = 0
    update_count = 0
    episode_count = 0
    episode_reward = 0.0
    episode_viruses = 0
    start_time = time.time()
    current_entropy_coef = args.entropy_coef
    
    # Spawn cache to ensure ~1 forward per spawn
    spawn_cache = {}  # spawn_id -> (logits, value)
    last_spawn_id = None
    
    while decision_count < args.total_decisions:
        # Extract info
        mask = _extract_mask(info)
        pill_colors = _extract_pill_colors(info)
        preview_pill_colors = _extract_preview_pill_colors(info)
        spawn_id = info.get("placements/spawn_id", decision_count)  # Fallback if not available
        
        # Get raw 464-element mask for action conversion
        mask_464_raw = info.get("placements/feasible_mask", np.ones(464, dtype=bool))
        
        # Check spawn cache to ensure ~1 forward per spawn
        if spawn_id != last_spawn_id or spawn_id not in spawn_cache:
            # Select action (policy outputs 512-space action)
            with torch.no_grad():
                action_512, log_prob, value = _select_action(
                    policy, obs, mask, pill_colors, preview_pill_colors, device, deterministic=False
                )
                # Cache for this spawn
                spawn_cache[spawn_id] = (action_512, log_prob, value)
        else:
            # Use cached action
            action_512, log_prob, value = spawn_cache[spawn_id]
            
        last_spawn_id = spawn_id
        
        # Convert 512-action to 464-action for environment
        action_464 = _convert_512_action_to_464(action_512, mask_464_raw)
            
        # Execute action (use 464-space action)
        obs_next, reward, terminated, truncated, info_next = env.step(action_464)
        
        # Get tau (frames consumed) from info
        tau = info_next.get("placements/tau", 1)
        if isinstance(tau, np.ndarray):
            tau = int(tau.item())
        else:
            tau = int(tau) if tau else 1
            
        # Track episode stats
        episode_reward += reward
        episode_viruses = info_next.get("drm", {}).get("viruses_cleared", episode_viruses)
        
        # Store decision (use 512-space action for learning)
        step = DecisionStep(
            obs=_obs_to_array(obs),
            mask=mask,
            pill_colors=pill_colors,
            preview_pill_colors=preview_pill_colors,
            action=action_512,  # Store policy's native action
            log_prob=log_prob,
            value=value,
            tau=tau,
            reward=float(reward),
            obs_next=_obs_to_array(obs_next),
            done=terminated or truncated,
        )
        buffer.add(step)
        decision_count += 1
        
        # Update state
        obs = obs_next
        info = info_next
        
        # Handle episode end
        if terminated or truncated:
            # Extract more detailed episode stats
            viruses_cleared = info_next.get("drm", {}).get("viruses_cleared", 0)
            survival_time = info_next.get("drm", {}).get("t", 0)
            print(f"Episode {episode_count}: reward={episode_reward:.1f}, viruses={viruses_cleared}, decisions={decision_count}, survival={survival_time}")
            episode_count += 1
            episode_reward = 0.0
            episode_viruses = 0
            obs, info = env.reset()
            # Clear spawn cache for new episode
            spawn_cache.clear()
            last_spawn_id = None
            
        # Update policy
        if len(buffer) >= args.decisions_per_update:
            # Bootstrap value
            with torch.no_grad():
                mask_boot = _extract_mask(info)
                colors_boot = _extract_pill_colors(info)
                preview_boot = _extract_preview_pill_colors(info)
                _, value_boot = _forward_policy(
                    policy, obs, mask_boot, colors_boot, preview_boot, device
                )
                
            # Get batch with advantages
            batch = buffer.get_batch(bootstrap_value=value_boot)
            
            # PPO update
            metrics = _update_ppo(
                policy,
                optimizer,
                batch,
                device,
                args.num_epochs,
                args.minibatch_size,
                args.clip_epsilon,
                args.value_coef,
                current_entropy_coef,  # Use current (decayed) entropy coef
            )
            
            # Decay entropy coefficient
            current_entropy_coef *= args.entropy_decay
            
            update_count += 1
            buffer.clear()
            
            # Log
            if update_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                dps = decision_count / max(elapsed, 1e-6)
                print(f"[Update {update_count}] "
                      f"Decisions: {decision_count}/{args.total_decisions}, "
                      f"π_loss: {metrics['policy_loss']:.4f}, "
                      f"v_loss: {metrics['value_loss']:.4f}, "
                      f"H: {metrics['entropy']:.4f}, "
                      f"KL: {metrics['kl_per_epoch']:.5f}, "
                      f"Adv: μ={metrics['adv_mean']:.3f} σ={metrics['adv_std']:.3f}, "
                      f"{dps:.1f} dec/s")
                      
        # Save checkpoint
        if decision_count % args.save_interval == 0 and decision_count > 0:
            ckpt_path = checkpoint_path(checkpoint_dir, "policy", decision_count, compress=True)
            save_checkpoint(
                {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "decision_count": decision_count,
                    "update_count": update_count,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")
            
    # Final save
    final_path = checkpoint_dir / f"policy_final_{decision_count}.pt.gz"
    save_checkpoint(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "decision_count": decision_count,
            "update_count": update_count,
            "args": vars(args),
        },
        final_path,
    )
    
    print(f"\\nTraining complete! Final checkpoint: {final_path}")
    env.close()


def _convert_512_action_to_464(action_512: int, mask_464: np.ndarray) -> int:
    """Convert policy's 512-action to environment's 464-action.
    
    Args:
        action_512: Flat index in [0, 511] from policy
        mask_464: The 464-element feasible mask from environment
        
    Returns:
        Action index in [0, 463] for the environment
    """
    from envs.retro.placement_actions import iter_edges
    from models.policy.placement_dist import unflatten_placement
    
    # Convert flat 512-action to (o, i, j)
    o, i, j = unflatten_placement(action_512)
    
    # Find matching 464-action
    # Look for an edge that places a pill at position (i, j) with orientation o
    for action_idx, edge in enumerate(iter_edges()):
        if not mask_464[action_idx]:
            continue
            
        origin_row, origin_col = edge.origin
        dest_row, dest_col = edge.dest
        
        # Check if this edge matches our desired placement
        match = False
        if o == 0 and edge.direction == "right" and origin_row == i and origin_col == j:
            match = True  # H+ horizontal, left cell at (i,j)
        elif o == 1 and edge.direction == "left" and origin_row == i and origin_col == j:
            match = True  # H- horizontal, right cell at (i,j)
        elif o == 2 and edge.direction == "down" and origin_row == i and origin_col == j:
            match = True  # V+ vertical, top cell at (i,j)
        elif o == 3 and edge.direction == "up" and origin_row == i and origin_col == j:
            match = True  # V- vertical, bottom cell at (i,j)
            
        if match:
            return action_idx
            
    # No exact match found - find closest valid action
    # Fall back to first valid action in mask
    valid_actions = np.where(mask_464)[0]
    if len(valid_actions) > 0:
        return int(valid_actions[0])
    
    # Last resort
    return 0


def _convert_464_mask_to_512(mask_464: np.ndarray) -> np.ndarray:
    """Convert 464-action mask to [4, 16, 8] mask for policy.
    
    The environment uses 464 actions (directed edges between adjacent cells).
    The policy uses 512 actions (4 orientations × 16 rows × 8 cols).
    
    We create a conservative mask that marks (o, i, j) as valid if ANY
    464-action involves placing a pill at that position.
    """
    from envs.retro.placement_actions import iter_edges
    
    # Create empty [4, 16, 8] mask
    mask_512 = np.zeros((4, 16, 8), dtype=bool)
    
    # Map each valid 464-action to its corresponding cells
    for action_idx, edge in enumerate(iter_edges()):
        if not mask_464[action_idx]:
            continue
            
        # Each edge has origin and dest cells
        # origin = (row1, col1), dest = (row2, col2)
        # direction tells us the orientation
        origin_row, origin_col = edge.origin
        dest_row, dest_col = edge.dest
        
        # Determine orientation from direction
        # H+ (0): horizontal, left cell is primary
        # H- (1): horizontal, right cell is primary  
        # V+ (2): vertical, top cell is primary
        # V- (3): vertical, bottom cell is primary
        
        if edge.direction == "right":
            # Horizontal pill, origin is left
            o = 0  # H+
            i, j = origin_row, origin_col
            mask_512[o, i, j] = True
        elif edge.direction == "left":
            # Horizontal pill, origin is right
            o = 1  # H-
            i, j = origin_row, origin_col
            mask_512[o, i, j] = True
        elif edge.direction == "down":
            # Vertical pill, origin is top
            o = 2  # V+
            i, j = origin_row, origin_col
            mask_512[o, i, j] = True
        elif edge.direction == "up":
            # Vertical pill, origin is bottom
            o = 3  # V-
            i, j = origin_row, origin_col
            mask_512[o, i, j] = True
            
    return mask_512


def _extract_mask(info: dict) -> np.ndarray:
    """Extract action mask from info.
    
    The environment provides a mask for the 464-action space,
    but the policy uses 512 actions (4×16×8). We convert the
    464-action mask to the 512-action format.
    """
    for key in ("placements/feasible_mask", "placements/legal_mask", "mask"):
        mask = info.get(key)
        if mask is not None and isinstance(mask, np.ndarray):
            if mask.shape == (4, 16, 8):
                # Already in correct format
                return mask.astype(bool)
            elif len(mask) == 464:
                # Convert from 464-action space to [4, 16, 8]
                return _convert_464_mask_to_512(mask.astype(bool))
    # Fallback: all valid
    return np.ones((4, 16, 8), dtype=bool)


def _extract_pill_colors(info: dict) -> np.ndarray:
    """Extract next pill colors."""
    colors = info.get("next_pill_colors")
    if colors is not None:
        arr = np.asarray(colors, dtype=np.int64)
        if arr.shape == (2,):
            return arr
    # Fallback
    return np.array([0, 0], dtype=np.int64)


def _extract_preview_pill_colors(info: dict) -> np.ndarray:
    """Extract preview pill colors (canonical indices 0=R,1=Y,2=B)."""

    raw_left = None
    raw_right = None
    raw_ram = info.get("raw_ram")
    try:
        if isinstance(raw_ram, (bytes, bytearray, memoryview)) and len(raw_ram) > 0x031B:
            raw_left = int(raw_ram[0x031A]) & 0x03
            raw_right = int(raw_ram[0x031B]) & 0x03
    except Exception:
        raw_left = None
        raw_right = None

    if raw_left is None or raw_right is None:
        preview = info.get("preview_pill")
        if isinstance(preview, dict):
            try:
                raw_left = int(preview.get("first_color", 0)) & 0x03
                raw_right = int(preview.get("second_color", 0)) & 0x03
            except Exception:
                raw_left = None
                raw_right = None
        elif isinstance(preview, (list, tuple)) and len(preview) >= 2:
            try:
                raw_left = int(preview[0]) & 0x03
                raw_right = int(preview[1]) & 0x03
            except Exception:
                raw_left = None
                raw_right = None

    if raw_left is None or raw_right is None:
        return np.array([0, 0], dtype=np.int64)

    def _map_color(raw: int) -> int:
        return int(ram_specs.COLOR_VALUE_TO_INDEX.get(int(raw) & 0x03, 0))

    return np.array([_map_color(raw_left), _map_color(raw_right)], dtype=np.int64)


def _obs_to_array(obs) -> np.ndarray:
    """Convert observation (dict or array) to numpy array."""
    if isinstance(obs, np.ndarray):
        arr = obs
    elif isinstance(obs, dict):
        # State-based observation has 'obs' key
        if 'obs' in obs:
            arr = obs['obs']
        else:
            # Try other common keys
            for key in ['state', 'observation', 'board', 'grid']:
                if key in obs and isinstance(obs[key], np.ndarray):
                    arr = obs[key]
                    break
            else:
                # Fallback: get first array
                arrays = [v for v in obs.values() if isinstance(v, np.ndarray)]
                if arrays:
                    arr = arrays[0]
                else:
                    # Last resort
                    return np.zeros((12, 16, 8), dtype=np.float32)
    else:
        return np.zeros((12, 16, 8), dtype=np.float32)
    
    # Handle stacked observations [4, C, 16, 8] -> [C, 16, 8] (use most recent)
    if arr.ndim == 4 and arr.shape[0] == 4:
        arr = arr[-1]  # Take most recent frame
    
    # Copy to ensure writable array for PyTorch
    return np.array(arr, dtype=np.float32)


def _forward_policy(policy, obs, mask, colors, preview_colors, device):
    """Forward pass through policy."""
    obs_arr = _obs_to_array(obs)
    obs_t = torch.from_numpy(obs_arr).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    colors_t = torch.from_numpy(colors).unsqueeze(0).to(device)
    preview_t = torch.from_numpy(preview_colors).unsqueeze(0).to(device)
    
    logits, value = policy(obs_t, colors_t, preview_t, mask_t)
    return logits.squeeze(0), float(value.squeeze().item())


def _select_action(policy, obs, mask, colors, preview_colors, device, deterministic=False):
    """Select action using masked distribution."""
    policy.eval()
    logits, value = _forward_policy(policy, obs, mask, colors, preview_colors, device)
    
    dist = MaskedPlacementDist(logits, torch.from_numpy(mask).to(device))
    action_idx, log_prob = dist.sample(deterministic=deterministic)
    
    return int(action_idx.item()), float(log_prob.item()), value


def _update_ppo(policy, optimizer, batch, device, num_epochs, minibatch_size, clip_eps, value_coef, entropy_coef):
    """PPO update with enhanced diagnostics."""
    policy.train()
    
    T = len(batch.actions)
    
    # Convert to tensors
    obs = torch.from_numpy(batch.observations).to(device)
    masks = torch.from_numpy(batch.masks).to(device)
    colors = torch.from_numpy(batch.pill_colors).to(device)
    preview_colors = torch.from_numpy(batch.preview_pill_colors).to(device)
    actions = torch.from_numpy(batch.actions).to(device)
    log_probs_old = torch.from_numpy(batch.log_probs).to(device)
    returns = torch.from_numpy(batch.returns).to(device)
    advantages = torch.from_numpy(batch.advantages).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std().item()),
    }
    
    num_updates = 0
    total_kl = 0.0
    
    for epoch in range(num_epochs):
        indices = torch.randperm(T, device=device)
        
        for start in range(0, T, minibatch_size):
            end = min(start + minibatch_size, T)
            mb_idx = indices[start:end]
            
            # Minibatch
            mb_obs = obs[mb_idx]
            mb_masks = masks[mb_idx]
            mb_colors = colors[mb_idx]
            mb_preview = preview_colors[mb_idx]
            mb_actions = actions[mb_idx]
            mb_log_probs_old = log_probs_old[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]
            
            # Forward
            logits, values = policy(mb_obs, mb_colors, mb_preview, mb_masks)
            
            # Compute log probs and entropy
            log_probs_list = []
            entropy_list = []
            
            for i in range(len(mb_actions)):
                dist = MaskedPlacementDist(logits[i], mb_masks[i])
                log_prob = dist.log_prob(mb_actions[i])
                entropy = dist.entropy()
                log_probs_list.append(log_prob)
                entropy_list.append(entropy)
                
            log_probs = torch.stack(log_probs_list)
            entropy = torch.stack(entropy_list).mean()
            
            # PPO loss
            ratio = torch.exp(log_probs - mb_log_probs_old)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = (values.squeeze(-1) - mb_returns).pow(2).mean()
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            # Track
            with torch.no_grad():
                kl = (mb_log_probs_old - log_probs).pow(2).mean()  # KL divergence approximation
                
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["kl"] += kl.item()
            total_kl += kl.item()
            num_updates += 1
            
    # Average
    for key in metrics:
        if key not in ["adv_mean", "adv_std"]:  # Don't average these
            metrics[key] /= max(num_updates, 1)
            
    # Add KL per epoch
    metrics["kl_per_epoch"] = total_kl / max(num_updates, 1)
        
    return metrics


if __name__ == "__main__":
    main()
