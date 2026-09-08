"""Measured pace throughput with exact planner and game trajectory comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.models.policy.pace_adapter import PacePolicy
from drmc_rl.planning.fast_reach import FrameState, HoldDir, Rotation
from drmc_rl.planning.native_reach import NativeReachabilityRunner
from tools.trainer_arena_cache import MemoPlanner
from tools.trainer_event_rollout import ParallelPlanning, run_event_batch
from tools.trainer_planning_arena import paired_jobs, run_batch
from tools.vs_head_to_head import PlainPolicy


def capture_policy_input(path, target, score):
    """Capture a public query to diagnose batch-sensitive numerical ties."""
    from drmc_rl.game.observation import board_bytes_to_semantic_planes
    own = board_bytes_to_semantic_planes(target["board"])
    if target["pill"][0] == target["pill"][1]:
        own[6:8] = 0

    def observed(policy, observations, infos):
        scores = score(policy, observations, infos)
        for row, info in enumerate(infos):
            if not path.exists() and np.array_equal(observations[row,:8],own):
                payload = dict(row=row,observations=observations.tolist(),infos=infos,
                    scores={str(a):float(scores[row,a]) for a in np.flatnonzero(np.isfinite(scores[row]))})
                path.write_text(json.dumps(payload,default=lambda a:a.tolist()))
        return scores

    return observed


def benchmark_planner(config, output):
    """Compare retained real roots against a separately built old library."""
    roots = json.loads(Path(config["planner_corpus"]).read_text())
    results, reference = [], None
    for library in config["planner_libraries"]:
        local = threading.local()

        def solve(root):
            if not hasattr(local, "runner"):
                local.runner = NativeReachabilityRunner(lib_path=library)
            x, y, rot, sc, hv, hd, rh, parity, locked = root["micro"]
            state = FrameState(x, y, rot, sc, hv, HoldDir(hd), parity, Rotation(rh), bool(locked))
            return local.runner.bfs_full(root["columns"], state, **root["kwargs"]).copy()

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=config.get("planner_workers", 4)) as executor:
            actual = list(executor.map(solve, roots))
        seconds = time.perf_counter()-started
        if reference is None:
            reference = actual
        for i, (before, after) in enumerate(zip(reference, actual)):
            for field in ("costs_u16", "offsets_u16", "lengths_u16", "script_buf"):
                if not np.array_equal(getattr(before, field), getattr(after, field)):
                    dump(output/"planner-mismatch.json", dict(root=roots[i], field=field))
                    raise RuntimeError(f"planner changed {field} for retained root {i}")
        result = dict(library=library, roots=len(roots), seconds=seconds,
                      roots_per_second=len(roots)/seconds, identical_outputs=True)
        results.append(result)
        dump(output/"planner-throughput.json", results)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if config.get("planner_corpus"):
        benchmark_planner(config, output)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if config.get("strict_fp32",False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    parent = PlainPolicy(Path(config["checkpoint"]), config["device"], public_only=True)
    actor = PacePolicy(config["checkpoint"], config["device"], adapter_path=config["adapter_checkpoint"])
    config["variants"] = {"adapter": {"delay": 4}, "parent": {"delay": 4}}
    config["replay_games"] = 0
    results = []
    for match in config["schedule"]:
        jobs = paired_jobs(config, match)
        reference = None
        if config.get("reference_directory"):
            with gzip.open(Path(config["reference_directory"])/f"{match['id']}.json.gz", "rt") as stream:
                saved = json.load(stream)
            if saved["match"] != match or saved["checkpoint"] != config["adapter_checkpoint"]:
                raise ValueError("reference trajectory belongs to a different benchmark")
            reference = saved["batch"]
        for mode in config.get("modes", ["reference", "events"]):
            if config.get("capture_policy_input"):
                import tools.trainer_event_rollout as event_module
                from drmc_rl.human.anticipation import score_public_inputs
                target = json.loads(Path(config["capture_policy_input"]).read_text())
                event_module.score_public_inputs = capture_policy_input(
                    output/f"policy-input-{mode}.json",target,score_public_inputs)
            if mode not in ("reference", "events", "async"):
                raise ValueError(f"unknown rollout mode: {mode}")
            config["async_planning"] = mode == "async"
            if config.get("mixed_core", True) and mode != "reference":
                config["mixed_core_actor"] = "adapter"
            else:
                config.pop("mixed_core_actor", None)
            planner = (MemoPlanner(NativeReachabilityRunner(lib_path=config.get("reference_planner_library"))) if mode == "reference"
                       else ParallelPlanning(config.get("planner_workers", 4), config.get("planner_roots")))
            runner = run_batch if mode == "reference" else run_event_batch
            try:
                metrics = {}
                batch, seconds = runner(config, match, jobs, None, planner, None,
                    policies={"adapter": actor, "parent": parent},
                    **({"metrics": metrics} if mode != "reference" else {}))
                if reference is None:
                    reference = batch
                    with gzip.open(output/f"{match['id']}.json.gz", "wt") as stream:
                        json.dump(dict(match=match, checkpoint=config["adapter_checkpoint"], batch=batch), stream)
                same_moves = len(reference) == len(batch) and all(a[1] == b[1] for a, b in zip(reference, batch))
                same_outcomes = len(reference) == len(batch) and all(
                    all(a[0][k] == b[0][k] for k in ("frames", "score", "reason")) for a, b in zip(reference, batch))
                frames = sum(b[0]["frames"] for b in batch)
                decisions = sum(b[0][s].get("decisions", 0) for b in batch for s in ("a_stats", "b_stats"))
                result = dict(mode=mode, pace=match["pace"], level=match["level"], games=len(batch),
                    frames=frames, decisions=decisions, seconds=seconds, fps=frames/seconds,
                    same_moves=same_moves, same_outcomes=same_outcomes,
                    validated_frames=sum(b[0][s].get("validated_input_frames", 0)
                        for b in batch for s in ("a_stats", "b_stats")), breakdown=metrics)
                results.append(result)
                dump(output/"throughput.json", results)
                print(json.dumps(result), flush=True)
                if not same_moves or not same_outcomes:
                    dump(output/f"mismatch-{match['pace']}-{mode}.json",
                         {"reference": [r[0:2] for r in reference], "actual": [r[0:2] for r in batch]})
                    raise RuntimeError("batched rollout differs from the reference trajectory")
            finally:
                planner.close()


if __name__ == "__main__":
    main()
