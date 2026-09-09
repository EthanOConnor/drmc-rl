"""Predict exact effects and conditional future access from candidate tokens.

These heads supervise the shared representation. Ordinary policy inference
does not execute them or use an unvalidated geometric preference as a reward.
"""

from __future__ import annotations

import torch.nn as nn
import torch

from drmc_rl.models.policy.effect_tokens import EFFECT_TOKEN_DIM

MOTOR_AUXILIARY_SCHEMA = "drmc-motor-auxiliary-v1"


class MotorAuxiliaryHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.features = nn.Sequential(nn.Linear(d_model + 13, d_model), nn.SiLU())
        self.effects = nn.Linear(d_model, EFFECT_TOKEN_DIM)
        self.opportunity = nn.Linear(d_model, 3 * 2 * 128)

    def forward(self, candidates, geometry):
        if geometry.shape != (candidates.shape[0], 13):
            raise ValueError("motor prediction requires the exact observed controller geometry")
        context = geometry.to(candidates.dtype).unsqueeze(1).expand(-1, candidates.shape[1], -1)
        features = self.features(torch.cat((self.norm(candidates), context), -1))
        output = self.opportunity(features).reshape(*features.shape[:2], 3, 2, 128)
        return dict(effect_predictions=self.effects(features),
                    motor_reach_logits=output[:, :, 0],
                    motor_clear_logits=output[:, :, 1],
                    motor_log_cost=output[:, :, 2])
