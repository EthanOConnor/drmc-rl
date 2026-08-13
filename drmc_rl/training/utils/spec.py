from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass(slots=True)
class EpisodeSummary:
    """Episode statistics emitted by the adapters."""

    step: int
    reward: float
    length: int
    info: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutBatch:
    """Batch of trajectories collected during training."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns": self.returns,
        }


def stack_rollouts(chunks: Sequence[RolloutBatch]) -> RolloutBatch:
    obs = np.concatenate([c.observations for c in chunks], axis=0)
    act = np.concatenate([c.actions for c in chunks], axis=0)
    rew = np.concatenate([c.rewards for c in chunks], axis=0)
    don = np.concatenate([c.dones for c in chunks], axis=0)
    val = np.concatenate([c.values for c in chunks], axis=0)
    logp = np.concatenate([c.log_probs for c in chunks], axis=0)
    adv = np.concatenate([c.advantages for c in chunks], axis=0)
    ret = np.concatenate([c.returns for c in chunks], axis=0)
    return RolloutBatch(obs, act, rew, don, val, logp, adv, ret)
