"""Independent public policy anchors and per-pace retention for outcome PPO.

Anchor actions are deterministic teacher choices. They never enter the PPO
likelihood or terminal-return objective. Each teacher receives its original
controller input encoding over the same complete paced frontier.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import torch

from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.models.policy.controller_core import INPUT_FIELDS
from drmc_rl.training.episodic_objective import categorical_kl


class RetentionRecorder:
    def __init__(self, encoder, planner, sides):
        if encoder.aux_spec != PUBLIC_CONTEXT_SCHEMA:
            raise ValueError("retention collection requires the live public encoder")
        self.encoder,self.planner,self.sides=encoder,planner,set(sides)

    def record(self,state,pace,delay,compute_frames,reference_scores):
        from drmc_rl.human.backend import plan_candidates
        from drmc_rl.human.controller_context import controller_policy_inputs
        candidate=plan_candidates(self.planner,state,delay,pace)
        observations,infos=controller_policy_inputs(self.encoder,candidate,state,pace,delay,compute_frames)
        inputs,aux,actions,masks=self.encoder.model_inputs(observations,infos)
        n=int(masks[0].sum())
        ours=actions[0,:n]
        reference_legal=set(np.flatnonzero(np.isfinite(reference_scores)))
        if not reference_legal or not reference_legal <= set(ours):
            raise ValueError("reference frontier is not contained in the complete public frontier")
        arrays=[t.detach().cpu().numpy() for t in (*inputs,aux)]
        row={key:arrays[j][0].copy() for j,key in enumerate(INPUT_FIELDS)}
        for key in ("actions","costs","mask"):
            row[key]=row[key][:n].copy()
        for key,dtype in (("observation",np.uint8),("pill",np.int8),("preview",np.int8),
                          ("actions",np.int16),("costs",np.uint16)):
            row[key]=row[key].astype(dtype)
        # The old teacher can suppress same-color rotation aliases. Preserve
        # its exact distribution with zero mass there, while retaining every
        # physically feasible student candidate and its actual cost.
        target=reference_scores[ours].astype(np.float32)
        target[~np.isfinite(target)]=-1e9
        row.update(base_logits=target,anchor_only=True,action=int(reference_scores.argmax()),
                   reference_candidates=len(reference_legal))
        return row


def select_game_anchors(batch, *, pace, seed, rows_per_game=4):
    """Sample across each natural game's duration, including early preparation."""
    rng = np.random.default_rng(seed)
    records = []
    for game, moves, _ in batch:
        if game["reason"] == "timeout":
            continue
        available = [m["anchor"] for m in moves if "anchor" in m]
        if not available:
            continue
        indices = [int(rng.choice(part)) for part in np.array_split(
            np.arange(len(available)), min(rows_per_game,len(available)))]
        for i in indices:
            row = dict(available[i],pace=pace,game_seed=game["seed"],learner_port=game["side"])
            assert row["anchor_only"]
            records.append(row)
    return records


def save_anchor_bank(path, records, metadata):
    path = Path(path)
    payload = dict(schema="drmc-controller-retention-v1",metadata=metadata,
        records=[{k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else v for k,v in r.items()}
                 for r in records])
    tmp=path.with_suffix(".next")
    torch.save(payload,tmp)
    tmp.replace(path)


class PaceRetention:
    def __init__(self, actor, bank_paths, *, excluded_seeds, paces, max_kl_increase=.03,
                 coefficient=.1, batch_size=64, pressure_strength=0.):
        if (not np.isfinite(max_kl_increase) or max_kl_increase < 0
                or not np.isfinite(coefficient) or coefficient <= 0 or batch_size < 1):
            raise ValueError("invalid retention budget")
        self.actor,self.coefficient,self.batch_size = actor,float(coefficient),int(batch_size)
        self.max_kl_increase=float(max_kl_increase)
        if not np.isfinite(pressure_strength) or pressure_strength < 0:
            raise ValueError("invalid retention pressure")
        self.pressure_strength=float(pressure_strength)
        self.records,self.identities,self.by_pace=[],{},{}
        excluded=set(map(int,excluded_seeds))
        for name in bank_paths:
            path=Path(name)
            self.identities[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            payload=torch.load(path,map_location="cpu",weights_only=True)
            if payload["schema"] != "drmc-controller-retention-v1":
                raise ValueError("unsupported retention bank")
            for stored in payload["records"]:
                row={k:v.numpy().copy() if isinstance(v,torch.Tensor) else v for k,v in stored.items()}
                if not row.get("anchor_only") or row["game_seed"] in excluded:
                    raise ValueError("retention bank contains PPO or evaluation data")
                if row["pace"] not in paces:
                    raise ValueError("retention bank has an unexpected pace")
                if not len(row["actions"]) or not row["mask"].all():
                    raise ValueError("retention bank must preserve complete nonempty frontiers")
                self.by_pace.setdefault(row["pace"],[]).append(row)
                self.records.append(row)
        if set(self.by_pace) != set(paces):
            raise ValueError("retention requires every training pace")
        self.paces=tuple(paces)
        self.weights={}
        self.seeds={r["game_seed"] for r in self.records}
        for pace,rows in self.by_pace.items():
            counts=Counter((r["game_seed"],r["learner_port"]) for r in rows)
            self.weights[pace]=np.asarray([1/(len(counts)*counts[r["game_seed"],r["learner_port"]]) for r in rows])
        self.baseline=self.measure()
        self.set_pressure(self.baseline)

    def set_pressure(self, measured):
        # Increase the soft restoring force before a pace reaches the unchanged
        # hard limit. Use accepted weights only, never a rejected trial.
        self.pressure={p:1+self.pressure_strength*float(np.clip(
            (measured[p]-self.baseline[p])/max(self.max_kl_increase,1e-12),0,1))**2
            for p in self.paces}

    def _kl(self, rows):
        features,data=self.actor.training_batch(rows)
        logits,_=self.actor.training_forward(features)
        return categorical_kl(data["parent_logp"],logits.log_softmax(-1))

    def loss(self, rng):
        # Each minibatch visits every pace; game-normalized sampling prevents
        # longer teacher games from acquiring more retention weight.
        selected=[]
        per_pace=max(1,self.batch_size//len(self.paces))
        for pace in self.paces:
            rows=self.by_pace[pace]
            ids=rng.choice(len(rows),per_pace,replace=True,p=self.weights[pace])
            selected.extend(rows[i] for i in ids)
        values=self._kl(selected)
        if self.pressure_strength == 0:
            return self.coefficient*values.mean()
        values=values.reshape(len(self.paces),per_pace).mean(-1)
        weights=values.new_tensor([self.pressure[p] for p in self.paces])
        return self.coefficient*(values*weights).mean()

    @torch.no_grad()
    def measure(self):
        result={}
        for pace,rows in self.by_pace.items():
            total=0.
            for start in range(0,len(rows),self.batch_size):
                values=self._kl(rows[start:start+self.batch_size]).double().cpu().numpy()
                total+=float(values @ self.weights[pace][start:start+len(values)])
            if not np.isfinite(total):
                raise ValueError("nonfinite per-pace retention KL")
            result[pace]=max(0.,total)
        return result

    def accepts(self, measured):
        return set(measured)==set(self.baseline) and all(
            np.isfinite(measured[p]) and measured[p]<=self.baseline[p]+self.max_kl_increase
            for p in self.paces)


def balance_pace_credit(records, completed_games):
    """Equal-pace episode SUM objective, with one common collection scale.

    All decisions in a pace receive the same factor. Natural games with no
    controllable decision count in its denominator and contribute zero gradient.
    Inverse trajectory lengths never enter this actor factor.
    """
    if not completed_games or any(n<=0 for n in completed_games.values()):
        raise ValueError("every selected pace needs natural completed games")
    total=sum(completed_games.values())
    for row in records:
        if row.get("anchor_only"):
            raise ValueError("teacher anchors are not PPO behavior samples")
        factor=total/(len(completed_games)*completed_games[row["pace"]])
        for key in ("actor_weight","value_weight","entropy_weight","parent_kl_weight"):
            row[key]*=factor
