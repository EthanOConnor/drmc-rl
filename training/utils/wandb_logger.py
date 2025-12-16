"""WandB integration for experiment tracking.

Provides optional Weights & Biases logging with graceful fallback.

WandB offers a free tier for personal use:
https://wandb.ai/pricing

This module stubs the interface so training works without WandB,
but enables seamless integration when available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


@dataclass
class WandBConfig:
    """Configuration for WandB logging."""
    
    enabled: bool = False
    project: str = "drmc-rl"
    entity: Optional[str] = None  # Username or team
    name: Optional[str] = None  # Run name
    tags: list = field(default_factory=list)
    notes: Optional[str] = None
    
    # Logging settings
    log_freq: int = 100  # Log every N steps
    save_code: bool = True
    
    # Video/media settings
    log_video: bool = False
    video_freq: int = 10000


class WandBLogger:
    """WandB experiment logger with graceful fallback.
    
    If WandB is not installed or disabled, logs are silently dropped.
    
    Usage:
        logger = WandBLogger(WandBConfig(enabled=True))
        logger.init(config={"lr": 3e-4, "algo": "ppo"})
        logger.log({"loss": 0.5, "reward": 100}, step=1000)
        logger.finish()
    """
    
    def __init__(self, config: Optional[WandBConfig] = None):
        self.config = config or WandBConfig()
        self._run = None
        self._enabled = False
    
    @property
    def enabled(self) -> bool:
        """Check if WandB is active."""
        return self._enabled and self._run is not None
    
    def init(
        self,
        config: Optional[Dict[str, Any]] = None,
        resume: Optional[str] = None,
    ) -> bool:
        """Initialize WandB run.
        
        Args:
            config: Hyperparameters to log
            resume: Resume mode ("allow", "must", "never", or run ID)
        
        Returns:
            True if WandB was successfully initialized
        """
        if not WANDB_AVAILABLE:
            return False
        
        if not self.config.enabled:
            return False
        
        try:
            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=config,
                resume=resume,
                save_code=self.config.save_code,
            )
            self._enabled = True
            return True
        except Exception as e:
            print(f"WandB init failed: {e}")
            self._enabled = False
            return False
    
    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log metrics.
        
        Args:
            data: Dictionary of metric name -> value
            step: Global step (optional, WandB auto-increments)
            commit: Whether to commit this log entry
        """
        if not self.enabled:
            return
        
        try:
            wandb.log(data, step=step, commit=commit)
        except Exception:
            pass  # Silently fail
    
    def log_summary(self, data: Dict[str, Any]) -> None:
        """Log summary metrics (shown on run page)."""
        if not self.enabled:
            return
        
        try:
            for key, value in data.items():
                wandb.run.summary[key] = value
        except Exception:
            pass
    
    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact (checkpoint, dataset, etc.)."""
        if not self.enabled:
            return
        
        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
        except Exception:
            pass
    
    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch a PyTorch model for gradient logging."""
        if not self.enabled:
            return
        
        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception:
            pass
    
    def finish(self) -> None:
        """Finish the WandB run."""
        if self._run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
            self._run = None
            self._enabled = False
    
    def __enter__(self) -> "WandBLogger":
        return self
    
    def __exit__(self, *args) -> None:
        self.finish()


# Convenience functions
def is_wandb_available() -> bool:
    """Check if WandB is installed."""
    return WANDB_AVAILABLE


def quick_init(
    project: str = "drmc-rl",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> WandBLogger:
    """Quick initialization of WandB logger.
    
    Args:
        project: WandB project name
        config: Hyperparameters to log
        **kwargs: Additional WandBConfig fields
    
    Returns:
        Initialized WandBLogger (may be disabled if WandB unavailable)
    """
    cfg = WandBConfig(enabled=True, project=project, **kwargs)
    logger = WandBLogger(cfg)
    logger.init(config=config)
    return logger
