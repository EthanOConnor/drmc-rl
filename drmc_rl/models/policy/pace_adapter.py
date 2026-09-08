"""Small outcome-trained strategy residual on a frozen public G5 core.

The final context bit explicitly gates the first experiment to Sloth through
Top Humans. The two fastest presets retain the exact parent output, regardless
of learned weights. No pace-dependent strength rating enters this model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PaceAdapter(nn.Module):
    def __init__(self, width=320, hidden=128, residual_limit=4.0):
        super().__init__()
        self.width, self.hidden, self.residual_limit = width, hidden, residual_limit
        self.candidate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU())
        self.context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU())
        self.motor = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.actor = nn.Sequential(nn.Linear(2*hidden+32, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(2*hidden+32, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for head in (self.actor, self.critic):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, candidates, context, motor, base_logits, base_value, mask):
        c, g, p = self.candidate(candidates), self.context(context), self.motor(motor)
        joined = torch.cat((c, g[:, None].expand_as(c), p[:, None].expand(-1,c.shape[1],-1)), -1)
        delta = self.residual_limit * torch.tanh(self.actor(joined).squeeze(-1))
        pooled = (c * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        dv = self.critic(torch.cat((pooled,g,p),-1)).squeeze(-1)
        enabled = motor[:, -1] > .5
        logits = torch.where(enabled[:, None], base_logits + delta, base_logits)
        value = torch.where(enabled, base_value + dv, base_value)
        return logits.masked_fill(~mask, -1e9), value


class _FeatureCore(nn.Module):
    def __init__(self, base, adapter):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapter = adapter
        self.motor = None
        self.features = None

    def forward(self, *args, **kwargs):
        logits, value, features = self.base(*args, **kwargs, return_aux=True)
        motor = self.motor.to(logits.device)
        self.features = (features["candidate_context"], features["global_context"],
                         motor, logits, value.reshape(-1), args[5].bool())
        return self.adapter(*self.features)


class PacePolicy:
    """Arena-compatible policy; optional on-policy records stay in memory only."""
    def __init__(self, parent, device="cpu", *, adapter_path=None, training=False, seed=0):
        from tools.vs_head_to_head import PlainPolicy
        self.parent = Path(parent)
        self.parent_sha256 = hashlib.sha256(self.parent.read_bytes()).hexdigest()
        self.plain = PlainPolicy(self.parent, device, public_only=True)
        payload = None
        if adapter_path is not None:
            payload = torch.load(adapter_path, map_location=device, weights_only=True)
            if (payload["schema"] != "drmc-pace-adapter-v1" or payload["parent_sha256"] != self.parent_sha256
                    or payload["context_schema"] != "own-motor-gravity-v1"):
                raise ValueError("pace adapter belongs to a different parent or schema")
        adapter_config = {} if payload is None else payload.get("adapter_config", {})
        self.adapter = PaceAdapter(self.plain.net.d_model, **adapter_config).to(device)
        if payload is not None:
            self.adapter.load_state_dict(payload["state_dict"])
        self.core = _FeatureCore(self.plain.net, self.adapter)
        self.plain.net = self.core
        self.device, self.training = device, training
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.learning_records = None

    def score(self, obs, infos):
        return self.score_mixed(obs, infos, np.ones(len(infos),dtype=bool))

    def score_mixed(self, obs, infos, adapted):
        """One shared core pass for learner and frozen-parent decision requests."""
        adapted = np.asarray(adapted,dtype=bool)
        if adapted.shape != (len(infos),):
            raise ValueError("one adapter role per decision required")
        motor = np.asarray([i["pace/context"] for i in infos],dtype=np.float32)
        motor[~adapted,-1] = 0
        self.core.motor = torch.as_tensor(motor)
        actions, masks, logits, values = self.plain.score_and_value(obs, infos)
        self.learning_records = None
        if not self.training or not adapted.any():
            return actions, masks, logits
        # Record the exact distribution that chose the controller target.
        # Force score_public_inputs/argmax to select that sampled target; the
        # original logits and likelihood are retained for PPO, never the marker.
        indices = np.flatnonzero(adapted)
        selected = torch.as_tensor(indices,device=self.core.features[0].device)
        # Retain only learner rows. Parent features and padding otherwise stay
        # alive through NumPy views for an entire full-game PPO collection.
        features = tuple(t.detach().index_select(0,selected).cpu().numpy()
                         for t in self.core.features)
        probabilities = torch.softmax(torch.as_tensor(logits[indices]), -1)
        slots = torch.multinomial(probabilities, 1, generator=self.rng).squeeze(1).cpu().numpy()
        self.learning_records = [None] * len(infos)
        for row,(i, slot) in enumerate(zip(indices,slots)):
            n = int(masks[i].sum())
            if not np.array_equal(masks[i], np.arange(len(masks[i])) < n) or not masks[i,slot]:
                raise RuntimeError("pace rollout requires complete contiguous packed candidates")
            record = {"candidate":features[0][row,:n], "context":features[1][row],
                "motor":features[2][row], "base_logits":features[3][row,:n],
                "base_value":float(features[4][row]), "slot":int(slot), "action":int(actions[i,slot]),
                "old_logprob":float(np.log(probabilities[row,int(slot)].item())),
                "old_value":float(values[i])}
            self.learning_records[i] = record
            logits[i,int(slot)] = np.max(logits[i,masks[i]]) + 1
        return actions, masks, logits

    def save(self, path, **metadata):
        path = Path(path)
        payload = {"schema":"drmc-pace-adapter-v1", "parent_sha256":self.parent_sha256,
            "state_dict":{k:v.detach().cpu() for k,v in self.adapter.state_dict().items()},
            "adapter_config":{"hidden":self.adapter.hidden,"residual_limit":self.adapter.residual_limit},
            "context_schema":"own-motor-gravity-v1", **metadata}
        temporary = path.with_suffix(path.suffix+".next")
        torch.save(payload, temporary)
        temporary.replace(path)
