"""Replace a V3 checkpoint's global regret knob with conditional tail calibration."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

from drmc_rl.human.afterstate_model import HUMAN_AFTERSTATE_SCHEMA, build_afterstate_policy
from drmc_rl.human.conditioning import HumanSkillCondition
from drmc_rl.human.model import build_timing_model
from drmc_rl.training.utils.checkpoint_io import load_checkpoint, save_checkpoint
from tools.train_afterstate_policy import (
    DEFAULT_AFTERSTATES,
    DEFAULT_DATASET,
    _evaluate,
    _source_paths,
)


def recalibrate(args: argparse.Namespace) -> dict:
    import torch

    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    if payload.get("schema") != HUMAN_AFTERSTATE_SCHEMA:
        raise ValueError(f"not a {HUMAN_AFTERSTATE_SCHEMA} checkpoint: {args.checkpoint}")
    cfg = payload["cfg"]
    meta = payload["human_meta"]
    condition = HumanSkillCondition.from_dict(meta["skill_condition"])
    model = build_afterstate_policy(
        cfg,
        condition_dim=int(cfg.get("condition_dim", 40)),
        device=args.device,
    )
    model.load_state_dict(payload["state_dict"])
    timing = build_timing_model(device=args.device)
    timing.load_state_dict(payload["timing_state_dict"])
    paths = _source_paths(args.dataset, args.afterstates, args.max_shards)
    use_bf16 = str(args.device).startswith("cuda") and torch.cuda.is_bf16_supported()

    def autocast():
        if use_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    metrics, calibration = _evaluate(
        model,
        timing,
        paths,
        afterstates=args.afterstates,
        condition=condition,
        device=args.device,
        batch_size=int(args.batch_size),
        max_rows=int(args.calibration_rows),
        autocast=autocast,
    )
    payload["human_meta"] = {
        **meta,
        "regret_calibration": calibration.to_dict(),
        "metrics": metrics,
        "strength_calibrated_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(payload, args.output)
    result = {
        "checkpoint": str(args.output),
        "source_checkpoint": str(args.checkpoint),
        "metrics": metrics,
        "regret_calibration": calibration.to_dict(),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--afterstates", type=Path, default=DEFAULT_AFTERSTATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calibration-rows", type=int, default=100_000)
    parser.add_argument("--max-shards", type=int)
    recalibrate(parser.parse_args())


if __name__ == "__main__":
    main()
