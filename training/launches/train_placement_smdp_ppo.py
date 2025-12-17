#!/usr/bin/env python3
"""Launch script for SMDP-PPO placement policy training.

Example usage:
    python training/launches/train_placement_smdp_ppo.py --num-envs 16 --head dense
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dr. Mario placement policy with SMDP-PPO")
    
    # Environment
    parser.add_argument("--env-id", type=str, default="DrMario-Placement-v0")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)
    
    # Training
    parser.add_argument("--total-steps", type=int, default=5000000, help="Total environment steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    
    # PPO
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    
    # Rollout
    parser.add_argument("--decisions-per-update", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    
    # Policy
    parser.add_argument("--head", type=str, default="dense", 
                       choices=["dense", "shift_score", "factorized"],
                       help="Policy head architecture")
    parser.add_argument("--pill-embed-dim", type=int, default=32)
    
    # Device
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device (auto, cpu, cuda, mps)")
    
    # Logging
    parser.add_argument("--logdir", type=str, default="runs/placement_smdp_ppo")
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=100)
    
    return parser.parse_args()


def select_device(device_spec: str) -> str:
    """Select compute device."""
    if device_spec != "auto":
        return device_spec
        
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Select device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Import training components
    from training.algo.ppo_smdp import SMDPPPOAdapter, SMDPPPOConfig
    from training.utils.logger import ConsoleLogger
    from training.utils.events import EventBus
    
    # Register environment
    from envs.retro.register_env import register_placement_env_id
    register_placement_env_id()
    
    # Create vectorized environment
    import gymnasium as gym
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    
    def make_env(rank: int):
        def _init():
            env = gym.make(args.env_id)
            env.reset(seed=args.seed + rank)
            return env
        return _init
    
    # Use async for I/O-bound envs (emulation), sync for CPU-bound
    env = AsyncVectorEnv([make_env(i) for i in range(args.num_envs)])
    
    # Create config object
    class SimpleNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
        def to_dict(self):
            return self.__dict__
    
    cfg = SimpleNamespace(
        seed=args.seed,
        logdir=args.logdir,
        smdp_ppo=SimpleNamespace(
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            decisions_per_update=args.decisions_per_update,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            head_type=args.head,
            pill_embed_dim=args.pill_embed_dim,
        ),
        train=SimpleNamespace(
            total_steps=args.total_steps,
            checkpoint_interval=args.checkpoint_interval,
        ),
    )
    
    # Create logger and event bus
    logger = ConsoleLogger()
    event_bus = EventBus()
    
    # Add progress logging
    @event_bus.on("episode_end")
    def log_episode(step, ret, len, **kwargs):
        if step % args.log_interval == 0:
            viruses = kwargs.get("drm/viruses_cleared", 0)
            print(f"[{step:>8}] Episode: reward={ret:>6.1f}, len={len:>4}, viruses={viruses:>2.0f}")
    
    @event_bus.on("update_end")
    def log_update(step, **metrics):
        if step % (args.log_interval * 10) == 0:
            policy_loss = metrics.get("loss/policy", 0.0)
            value_loss = metrics.get("loss/value", 0.0)
            entropy = metrics.get("policy/entropy", 0.0)
            sps = metrics.get("perf/sps", 0.0)
            dps = metrics.get("perf/dps", 0.0)
            print(f"[{step:>8}] Update: π_loss={policy_loss:.4f}, v_loss={value_loss:.4f}, "
                  f"H={entropy:.4f}, {sps:.0f} steps/s, {dps:.1f} dec/s")
    
    @event_bus.on("checkpoint")
    def log_checkpoint(step, path, **kwargs):
        print(f"[{step:>8}] Checkpoint saved: {path}")
    
    # Create trainer
    trainer = SMDPPPOAdapter(
        cfg=cfg,
        env=env,
        logger=logger,
        event_bus=event_bus,
        device=device,
    )
    
    # Train
    print(f"\nStarting SMDP-PPO training with {args.head} head...")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Environments: {args.num_envs}")
    print(f"Decisions per update: {args.decisions_per_update}")
    print(f"Log dir: {args.logdir}\n")
    
    try:
        trainer.train_forever()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        env.close()
        print("Training complete.")


if __name__ == "__main__":
    main()
