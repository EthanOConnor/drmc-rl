"""Full-network outcome learning on the live public controller input contract.

Rollouts retain model inputs, never frozen trunk features or native restore
bytes. The fixed post-migration initial policy supplies the regularization
reference. Natural terminal results are labels, never actor inputs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import torch

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.training.quality_supervision import upgrade_public_model
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from tools.eval_policy import _make_aux_builder
from tools.vs_head_to_head import PlainPolicy

CORE_SCHEMA = "drmc-public-controller-core-v1"
INPUT_FIELDS = ("observation", "pill", "preview", "actions", "costs", "mask", "public_context")
CONTROLLER_GEOMETRY_FIELDS = (
    "speed", "speed_ups", "pill_counter_total", "decision_delay_frames", "compute_frames",
    "x", "y", "rotation", "speed_counter", "horizontal_velocity", "hold_dir",
    "rotation_hold", "frame_parity",
)


class ControllerCorePolicy(PlainPolicy):
    def __init__(self, parent, device="cpu", *, resume=None, training=True, seed=0):
        super().__init__(Path(parent), device, public_only=True)
        self.parent_sha256 = hashlib.sha256(Path(parent).read_bytes()).hexdigest()
        payload = load_checkpoint(Path(parent), map_location=device)
        saved = load_checkpoint(Path(resume), map_location=device) if resume is not None else None
        if self.aux_spec == PUBLIC_CONTEXT_SCHEMA:
            self.cfg = deepcopy(payload["cfg"])
        else:
            # Reconstruct the original regularization reference when resuming
            # a pre-residual study; a new migration must not rewrite its run.
            preserve_policy = True
            if saved is not None:
                resume_cfg = saved["cfg"].get("smdp_ppo", saved["cfg"])
                preserve_policy = resume_cfg.get("candidate_context_residual", False)
            self.net, self.cfg = upgrade_public_model(
                payload, mode="context", device=device, preserve_policy=preserve_policy,
            )
        self.cfg.setdefault("env", {})["public_observations"] = True
        self.aux_spec = PUBLIC_CONTEXT_SCHEMA
        self.aux_shim = _make_aux_builder(self.net.aux_dim, aux_spec=self.aux_spec)
        self.aux_dim = self.net.aux_dim
        self.requires_causal_observations = True
        self.net.eval()
        self.reference = deepcopy(self.net).requires_grad_(False).eval()
        if saved is not None:
            if (saved.get("schema") != CORE_SCHEMA
                    or saved.get("parent_sha256") != self.parent_sha256
                    or saved.get("observation_schema") != PUBLIC_CONTEXT_SCHEMA
                    or saved["cfg"] != self.cfg):
                raise ValueError("controller-core resume changed its parent or input/model contract")
            self.net.load_state_dict(saved["state_dict"], strict=True)
        self.training_module = self.net
        self.training = training
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.learning_records = None
        self._collection_id = 0
        self._collection_versions = None

    def _parameter_versions(self):
        # PyTorch increments these counters for optimizer and load_state_dict
        # writes. Check identity directly rather than trying to infer a weight
        # change from a numerically approximate GPU probability comparison.
        return tuple((name, value._version) for name, value in
                     (*self.net.named_parameters(), *self.net.named_buffers()))

    def finish_collection(self, records):
        if (self.net.training or self._collection_versions != self._parameter_versions()
                or any(r.get("collection_id") != self._collection_id for r in records)):
            raise RuntimeError("controller network changed or collection versions were mixed")
        self._collection_versions = None

    def score(self, obs, infos):
        if not self.training:
            self.learning_records = None
            return super().score(obs, infos)
        versions = self._parameter_versions()
        if self.net.training:
            raise RuntimeError("controller collection requires deterministic evaluation mode")
        if self._collection_versions is None:
            self._collection_versions = versions
            self._collection_id += 1
        elif self._collection_versions != versions:
            raise RuntimeError("controller network changed during collection")
        inputs, aux, actions, masks = self.model_inputs(obs, infos)
        with torch.inference_mode():
            logits, values = self.net(*inputs, aux=aux)
            reference, _ = self.reference(*inputs, aux=aux)
            logs = logits.float().log_softmax(-1).cpu()
            slots = torch.multinomial(logs.exp(), 1, generator=self.rng).squeeze(1)
            arrays = [t.detach().cpu().numpy() for t in (*inputs, aux)]
            reference = reference.float().cpu().numpy()
            values = values.reshape(-1).float().cpu().numpy()
            scores = logits.float().cpu().numpy().copy()
        self.learning_records = []
        for i, slot in enumerate(slots.tolist()):
            n = int(masks[i].sum())
            if not np.array_equal(masks[i], np.arange(masks.shape[1]) < n) or not masks[i, slot]:
                raise RuntimeError("controller learning requires a complete contiguous frontier")
            # Own detached arrays prevent a row retaining the entire batch or
            # inference tensor storage throughout a long natural game.
            row = dict(
                observation=arrays[0][i].astype(np.uint8),
                pill=arrays[1][i].astype(np.int8),
                preview=arrays[2][i].astype(np.int8),
                actions=arrays[3][i, :n].astype(np.int16),
                costs=arrays[4][i, :n].astype(np.uint16),
                mask=arrays[5][i, :n].copy(),
                public_context=arrays[6][i].copy(),
                base_logits=reference[i, :n].copy(),
                behavior_logp=logs[i, :n].numpy().copy(),
                slot=slot, action=int(actions[i, slot]),
                old_logprob=float(logs[i, slot]), old_value=float(values[i]),
                observed_frame=int(infos[i]["public_pair_state"].frame_id),
                viewer_side=int(infos[i]["public_acting_side"]),
                collection_id=self._collection_id,
                controller_geometry=np.asarray([
                    infos[i]["public_controller_geometry"][key]
                    for key in CONTROLLER_GEOMETRY_FIELDS
                ], dtype=np.int32),
            )
            self.learning_records.append(row)
            scores[i, slot] = scores[i, masks[i]].max() + 1
        scores[~masks] = -np.inf
        return actions, masks, scores

    def training_batch(self, records):
        count = len(records)
        width = max(32, max(len(r["actions"]) for r in records))
        arrays = dict(
            observation=np.stack([r["observation"] for r in records]).astype(np.float32),
            pill=np.stack([r["pill"] for r in records]).astype(np.int64),
            preview=np.stack([r["preview"] for r in records]).astype(np.int64),
            actions=np.full((count, width), -1, np.int64),
            costs=np.zeros((count, width), np.float32),
            mask=np.zeros((count, width), bool),
            public_context=np.stack([r["public_context"] for r in records]),
        )
        parent = np.full((count, width), -1e9, np.float32)
        for i, row in enumerate(records):
            n = len(row["actions"])
            for key in ("actions", "costs", "mask"):
                arrays[key][i, :n] = row[key]
            parent[i, :n] = row["base_logits"]
        features = tuple(torch.as_tensor(arrays[key], device=self.device) for key in INPUT_FIELDS)
        keys = ("slot", "old_logprob", "old_value", "return", "weight", "advantage",
                "actor_weight", "value_weight", "entropy_weight", "parent_kl_weight")
        data = {key: torch.as_tensor([r[key] for r in records], device=self.device,
                                    dtype=torch.long if key == "slot" else torch.float32)
                for key in keys if key in records[0]}
        data["parent_logp"] = torch.as_tensor(parent, device=self.device).log_softmax(-1)
        return features, data

    def training_forward(self, features):
        logits, value = self.net(*features[:6], aux=features[6])
        return logits, value.reshape(-1)

    @torch.no_grad()
    def precise_behavior_logp(self, record):
        """Independent FP64 reference for a rare FP32 collection-audit outlier.

        Copy the unchanged model to CPU instead of changing the actor, its
        optimizer tensors, or the recorded behavior distribution. CPU supports
        FP64 on every training host, including those using Metal inference.
        This is verification only; PPO still uses the original collection logs.
        """
        features, _ = self.training_batch([record])
        inputs = tuple(value.detach().to(device="cpu", dtype=torch.float64)
                       if value.is_floating_point() else value.detach().cpu()
                       for value in features)
        reference = deepcopy(self.net).to(device="cpu", dtype=torch.float64).eval()
        logits, _ = reference(*inputs[:6], aux=inputs[6])
        return logits[0, :len(record["actions"])].log_softmax(-1).numpy().copy()

    def save(self, path, **metadata):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".next")
        torch.save(dict(
            schema=CORE_SCHEMA, cfg=self.cfg,
            state_dict={k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            parent_sha256=self.parent_sha256, observation_schema=PUBLIC_CONTEXT_SCHEMA,
            regularization_reference="fixed-post-migration-initial-policy",
            calibrated=False, diagnostic_only=True, **metadata,
        ), temporary)
        temporary.replace(path)


def write_public_replay(path, records, games, *, update, pace, level):
    """A pickle-free public input/terminal-label shard for independent teachers.

    Candidate arrays are packed with offsets; full legal inventories survive.
    Outcome labels apply to the observed continuation, not every alternative.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([len(r["actions"]) for r in records], np.int32)
    payload = {
        key: np.stack([r[key] for r in records])
        for key in ("observation", "pill", "preview", "public_context")
    }
    payload.update({key: np.concatenate([r[key] for r in records])
                    for key in ("actions", "costs", "base_logits", "behavior_logp")})
    payload["offsets"] = np.concatenate(([0], np.cumsum(counts)))
    for key in ("action", "slot", "return", "old_logprob", "old_value", "observed_frame", "viewer_side"):
        payload[key] = np.asarray([r[key] for r in records])
    payload["game_seed"] = np.asarray([games[r["game_id"]]["seed"] for r in records], np.int32)
    payload["learner_port"] = np.asarray([games[r["game_id"]]["side"] for r in records], np.int8)
    payload["controller_geometry"] = np.stack([r["controller_geometry"] for r in records])
    payload["metadata"] = np.asarray(json.dumps(dict(
        schema="drmc-public-controller-replay-v2", update=update, behavior_update=update - 1,
        pace=pace, level=level,
        observation_schema=PUBLIC_CONTEXT_SCHEMA, actor_inputs=INPUT_FIELDS,
        controller_geometry_fields=CONTROLLER_GEOMETRY_FIELDS,
        controller_geometry_scope="observed-own-controller-boundary-for-conditional-labels",
        outcome_scope="natural-terminal-observed-policy-continuation",
        reference_scope="fixed-post-migration-initial-policy",
    )))
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)
