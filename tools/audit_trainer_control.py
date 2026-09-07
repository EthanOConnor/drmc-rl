"""Measure fixed-state strength ordering independently of match outcomes.

This diagnostic uses the frozen V3 score, not a promoted competitive-regret
oracle. It records model scores so alternative decoders can be checked without
repeating inference or silently changing the source bank.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from drmc_rl.game.observation import board_bytes_to_semantic_planes
from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
from drmc_rl.human.strength import RegretStrengthController
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.search.native_pair import state_from_payload


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def measure_legacy(calibration, ratings, quality, styles, mask, *, quantiles=99):
    """Reproduce the former overlapping tolerance decoder for comparison."""

    probabilities = np.linspace(*calibration.quantile_levels[[0, -1]], quantiles)
    selected = np.zeros((len(quality), len(ratings), quantiles))
    fixed_style = np.zeros_like(selected)
    targets = np.zeros_like(selected)
    center = int(np.argmin(np.abs(np.asarray(ratings) - 1600)))
    for i in range(len(quality)):
        scores = quality[i, mask[i]]
        regret = scores.max() - scores
        opportunity = float(scores.std())
        for r, rating in enumerate(ratings):
            target = np.interp(probabilities, calibration.quantile_levels,
                               calibration.curve(float(rating), opportunity))
            _, tolerance = calibration.parameters(float(rating), opportunity)
            distance = np.abs(np.log1p(regret)[None] - np.log1p(target)[:, None])
            plausible = distance <= distance.min(axis=1, keepdims=True) + tolerance
            for output, preference in ((selected, styles[i, r, mask[i]]),
                                       (fixed_style, styles[i, center, mask[i]])):
                slots = np.where(plausible, preference[None], -np.inf).argmax(axis=1)
                output[i, r] = regret[slots]
            targets[i, r] = target
    return {"chosen_regret": selected, "fixed_style_regret": fixed_style, "target_regret": targets}


def measure(calibration, ratings, quality, styles, mask, *, quantiles=99, reference_rating=None):
    """Exercise the production decoder at paired quantiles on frozen scores."""

    probabilities = np.linspace(*calibration.quantile_levels[[0, -1]], quantiles)
    values = {key: np.zeros((len(quality), len(ratings), quantiles))
              for key in ("chosen_regret", "fixed_style_regret", "target_regret")}
    controller = RegretStrengthController(calibration, reference_rating=reference_rating)
    center = int(np.argmin(np.abs(np.asarray(ratings) - 1600)))
    for i in range(len(quality)):
        for r, rating in enumerate(ratings):
            for q, quantile in enumerate(probabilities):
                _, selected = controller.choose(quality[i], styles[i, r], mask[i],
                                                rating=rating, quantile=quantile)
                _, fixed = controller.choose(quality[i], styles[i, center], mask[i],
                                             rating=rating, quantile=quantile)
                values["chosen_regret"][i, r, q] = selected["chosen_regret"]
                values["target_regret"][i, r, q] = selected["target_regret"]
                values["fixed_style_regret"][i, r, q] = fixed["chosen_regret"]
    return values


def summarize(values, ratings):
    out = {}
    for key, value in values.items():
        delta = np.diff(value, axis=1)
        out[key] = {
            "mean_by_rating": value.mean(axis=(0, 2)).tolist(),
            "adjacent_inversion_fraction": (delta > 1e-8).mean(axis=(0, 2)).tolist(),
            "states_with_mean_inversion": int((delta.mean(axis=2) > 1e-8).any(axis=1).sum()),
            "maximum_inversion": float(max(0, delta.max())),
        }
    return {"ratings": list(ratings), **out}


def audit(args):
    import torch

    torch.set_num_threads(2)
    runtime = AfterstatePolicyRuntime(args.checkpoint, device=args.device)
    ratings = np.asarray([float(value) for value in args.ratings.split(",")])
    if len(ratings) < 2 or not np.all(np.diff(ratings) > 0):
        raise ValueError("ratings must increase strictly")
    cells = Counter()
    records = []
    with gzip.open(args.state_bank, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            cell = (row["level"], row["speed_setting"], row["tactical_stratum"])
            if cells[cell] >= args.per_cell:
                continue
            cells[cell] += 1
            records.append(row)
    quality = np.full((len(records), 512), -np.inf, dtype=np.float32)
    styles = np.full((len(records), len(ratings), 512), -np.inf, dtype=np.float32)
    mask = np.zeros((len(records), 512), dtype=bool)
    try:
        for index, row in enumerate(records):
            state = state_from_payload(row)
            side = int(row["root_side"])
            own, opponent = state.privileged.public.sides[side], state.privileged.public.sides[1-side]
            feasible = np.zeros(512, dtype=bool)
            costs = np.full(512, 0xFFFF, dtype=np.uint16)
            actions = np.asarray(state.legal_actions_by_side[side])
            feasible[actions] = True
            costs[actions] = state.action_costs_by_side[side]
            if own.pill[0] == own.pill[1]:
                feasible[256:], costs[256:] = False, 0xFFFF
            packed = pack_feasible_candidates(feasible.reshape(4, 16, 8),
                costs.reshape(4, 16, 8), max_candidates=max(128, int(feasible.sum())),
                sort_by_cost=True)
            request = dict(
                board_planes=board_bytes_to_semantic_planes(own.board),
                opponent_board_planes=board_bytes_to_semantic_planes(opponent.board),
                opponent_state_age_frames=0, pill=own.pill, preview=own.preview,
                game_phase=min(float(row["decision_index"])/100, 1),
                candidate_actions=packed.actions, candidate_costs=packed.cost,
                candidate_mask=packed.mask, speed=row["speed_setting"],
                speed_ups=row["speed_ups"],
            )
            results = runtime.score_batch([{**request, "rating": rating,
                "style_rating": runtime.condition.mean if args.style_conditioning == "population" else None,
                "opponent_rating": None} for rating in ratings])
            width = len(packed.mask)
            mask[index, :width] = packed.mask
            quality[index, :width] = results[0]["competitive_score"]
            for r, result in enumerate(results):
                np.testing.assert_allclose(result["competitive_score"],
                                           results[0]["competitive_score"], rtol=1e-5, atol=1e-5)
                styles[index, r, :width] = result["human_logits"]
            if index % 20 == 0:
                print(f"scored {index+1}/{len(records)} states", flush=True)
        measured = measure(runtime.calibration, ratings, quality, styles, mask,
                           reference_rating=runtime.condition.mean)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        score_path = args.output.with_suffix(".npz")
        np.savez_compressed(score_path, ratings=ratings, quality=quality, styles=styles,
            mask=mask, source_ids=np.asarray([str(row["id"]) for row in records]), **measured)
        result = {"schema": "drmc-trainer-control-audit-v1", "diagnostic_only": True,
            "checkpoint_sha256": sha256(args.checkpoint), "state_bank_sha256": sha256(args.state_bank),
            "score_archive_sha256": sha256(score_path), "states": len(records),
            "cells": len(cells), "per_cell": args.per_cell,
            "decoder": "ordered-log-regret-bands-v1",
            "style_conditioning": args.style_conditioning,
            "style_reference_rating": runtime.condition.mean,
            "calibration": runtime.calibration.to_dict(), **summarize(measured, ratings)}
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k: v for k, v in result.items() if k != "calibration"}, indent=2))
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ratings", default="800,1200,1600,2000,2400")
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--style-conditioning", choices=("population", "requested"), default="population")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
