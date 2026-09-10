"""Measure shadow constructions in natural autonomous controller games.

The frozen competitive actor chooses every action. Proposal scores cannot
change controls, supply quality labels, or establish style noninferiority.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.execution.pace import PACES
from drmc_rl.human.spatial_execution import ConstructionObserver
from drmc_rl.human.spatial_proposer import (
    RECURRENT_PUBLIC, SCHEMA, FrozenConstructionEncoder, SpatialProposer,
)
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from drmc_rl.planning.native_reach import resolve_library_path
from tools.build_expressive_sequences import write_progress
from tools.eval_policy import _build_net_from_cfg
from tools.fit_spatial_expressive import sha256
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import paired_jobs, variant_policy
from tools.vs_head_to_head import PlainPolicy


def load_models(study_path, feature_device):
    study_path = Path(study_path)
    study = json.loads(study_path.read_text())
    if study.get("status") != "Complete" or study.get("schema") != SCHEMA:
        raise ValueError("a completed fixed-final construction study is required")
    if study["config"].get("plan_update_schema") != RECURRENT_PUBLIC:
        raise ValueError("actual event comparison requires recurrent and history-reset controls")
    checkpoint = Path(study["config"]["checkpoint"])
    if sha256(checkpoint) != study["competitive_sha256"]:
        raise ValueError("construction feature checkpoint changed")
    saved = load_checkpoint(checkpoint, map_location="cpu")
    cfg = saved["cfg"]
    sp = cfg.get("smdp_ppo", cfg)
    core, _, _ = _build_net_from_cfg(cfg, int(sp["candidate_board_channels"])+4, feature_device)
    core.load_state_dict(saved.get("ema_state_dict") or saved["state_dict"], strict=True)
    encoder = FrozenConstructionEncoder(core, zero_auxiliary=sp.get("aux_spec") == "zero_v1_vs")
    models = {}
    for name in ("persistent", "stateless"):
        path = study_path.parent / (name+"-final.pt")
        if sha256(path) != study["arms"][name]["checkpoint_sha256"]:
            raise ValueError("fixed-final proposer checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["schema"] != SCHEMA or payload["competitive_sha256"] != study["competitive_sha256"]
                or payload["source_sha256"] != study["source_sha256"]
                or payload["plan_update_schema"] != RECURRENT_PUBLIC
                or payload["persistent"] != (name == "persistent")
                or payload["feature_dim"] != encoder.feature_dim):
            raise ValueError("construction checkpoint contract mismatch")
        model = SpatialProposer(payload["feature_dim"], payload["width"],
                                persistent=payload["persistent"], plan_update_schema=RECURRENT_PUBLIC)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[name] = model.eval().requires_grad_(False)
    return models, encoder, study


def summarize(rows):
    result = {}
    for name in ("persistent", "stateless"):
        plans = [p for row in rows for p in row["plans"] if p["arm"] == name]
        decisions = [d for row in rows for d in row["decisions"] if d["arm"] == name]
        result[name] = dict(plans=len(plans), decisions=len(decisions),
            terminations=dict(Counter(p["reason"] for p in plans)),
            goals=dict(Counter(p["goal"] for p in plans)),
            root_goals_observed=sum(p["root_goal_observed"] for p in plans),
            anchor_revisions=sum(p["anchor_revisions"] for p in plans),
            mean_budget=float(np.mean([p["budget"] for p in plans])) if plans else None,
            raw_preferences_reachable=sum(d["raw_preference_reachable"] for d in decisions),
            actor_agreements=sum(d["agrees_with_actor"] for d in decisions))
    return result


def comparisons(rows):
    """Descriptive paired reset-seed uncertainty, never an adoption gate."""
    output = []
    rng = np.random.default_rng(41973)
    for condition in sorted({r["condition"] for r in rows}):
        selected = [r for r in rows if r["condition"] == condition]
        seeds = sorted({r["game"]["seed"] for r in selected})
        values = []
        for seed in seeds:
            games = [r for r in selected if r["game"]["seed"] == seed]
            if len(games) != 2 or {r["game"]["side"] for r in games} != {0, 1}:
                raise ValueError("whole side-swapped seed pairs required")
            per_game = []
            for row in games:
                rates = []
                for arm in ("persistent", "stateless"):
                    plans = [p for p in row["plans"] if p["arm"] == arm]
                    decisions = [d for d in row["decisions"] if d["arm"] == arm]
                    rates.append([sum(p["root_goal_observed"] for p in plans)/max(len(plans), 1),
                                  sum(d["raw_preference_reachable"] for d in decisions)/max(len(decisions), 1),
                                  sum(d["agrees_with_actor"] for d in decisions)/max(len(decisions), 1)])
                per_game.append(np.asarray(rates[0])-rates[1])
            values.append(np.mean(per_game, axis=0))
        values = np.asarray(values)
        boot = values[rng.integers(len(seeds), size=(20000, len(seeds)))].mean(1)
        output.append(dict(condition=condition, reset_seeds=len(seeds), games=len(selected),
            metrics={name:dict(persistent_minus_reset=float(values[:, i].mean()),
                              individual_ci95=np.quantile(boot[:, i], [.025, .975]).tolist())
                     for i, name in enumerate(("root_goal_rate", "raw_preference_reachable_rate", "actor_agreement_rate"))}))
    return output


def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    arena = config["arena"]
    if arena.get("async_planning") or arena.get("replay_games", 0):
        raise ValueError("use deterministic synchronous batches with separate move journals")
    identities = {name:sha256(params.get("checkpoint", arena["checkpoint"]))
                  for name, params in arena["variants"].items()}
    report = dict(schema="drmc-spatial-execution-audit-v1", status="Running", config=config,
        actor_sha256=identities, native_sha256=sha256(arena["native_library"]),
        reach_library=str(resolve_library_path()), reach_sha256=sha256(resolve_library_path()),
        study_sha256=sha256(config["study"]), motor_profiles=[p.to_dict() for p in PACES],
        games=0, target_games=sum(m["games"] for m in arena["schedule"]),
        simulated_console_frames=0, outcome_training_frames=0, optimizer_updates=0,
        quality_admission=False, modifies_actor_choices=False, censored_games=0,
        scope="Shadow proposals on autonomous full games. Conditional verified payoff geometry; no quality, style-strength or preference claim.")
    write_progress(output, report)
    planner = None
    rows = []
    try:
        models, encoder, study = load_models(config["study"], config.get("feature_device", "mps"))
        report["proposal_sha256"] = {k:v["checkpoint_sha256"] for k, v in study["arms"].items()}
        actor = PlainPolicy(Path(arena["checkpoint"]), arena.get("device", "mps"), public_only=True)
        policies = {name:variant_policy(arena, params, actor) for name, params in arena["variants"].items()}
        planner = ParallelPlanning(arena.get("planner_workers", 3))
        for match in arena["schedule"]:
            jobs = paired_jobs(arena, match)
            for start in range(0, len(jobs), arena.get("pairs", 16)):
                batch = jobs[start:start+arena.get("pairs", 16)]
                observer = ConstructionObserver(models, encoder, batch, feature_device=config.get("feature_device", "mps"))
                report.update(phase="playing", current_condition=match["id"])
                write_progress(output, report)

                def activity(value):
                    report.update(current_batch=value, elapsed_seconds=time.monotonic()-started)
                    write_progress(output, report)

                result, seconds = run_event_batch(arena, match, batch, actor, planner, None,
                                                  policies=policies, observer=observer, activity=activity)
                observer.close()
                for pair, (game, moves, _) in enumerate(result):
                    side = 2*pair+game["side"]
                    row = dict(condition=match["id"], level=match["level"], pace=match["pace"], game=game,
                        plans=[p for p in observer.plans if p["side"] == side],
                        decisions=[d for d in observer.decisions if d["side"] == side],
                        transitions=[t for t in observer.transitions if t["side"] == side],
                        observation_counts=dict(observer.states[side]["counters"]))
                    rows.append(row)
                    with gzip.open(output/f"{match['id']}-{game['index']:04d}.json.gz", "wt") as handle:
                        json.dump(dict(**row, moves=moves), handle)
                report.update(games=len(rows), simulated_console_frames=sum(r["game"]["frames"] for r in rows),
                    censored_games=sum(r["game"]["reason"] == "timeout" for r in rows), arms=summarize(rows),
                    last_batch_seconds=seconds, elapsed_seconds=time.monotonic()-started)
                report.pop("current_batch", None)
                write_progress(output, report)
                if report["censored_games"]:
                    raise ValueError("censored games cannot establish construction realization")
        report.update(status="Complete", phase="complete", paired_comparisons=comparisons(rows),
            uncertainty="20,000 whole reset-seed paired resamples; individual descriptive intervals, no multiplicity-based selection.",
            observation_counts=dict(sum((Counter(r["observation_counts"]) for r in rows), Counter())))
    except BaseException as error:
        report.update(status="Failed", error=str(error))
        raise
    finally:
        if planner is not None:
            planner.close()
        report["elapsed_seconds"] = time.monotonic()-started
        write_progress(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
