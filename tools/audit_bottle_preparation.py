"""Measure exact reusable bottle features, not the quality of an untrained migration."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from drmc_rl.arena.experiment import dump
from drmc_rl.game.public_context import PUBLIC_CONTEXT_SCHEMA
from drmc_rl.models.policy.bottle_preparation import prepare_bottle
from drmc_rl.training.utils.checkpoint_io import load_checkpoint
from drmc_rl.teachers.counterfactual_release import sha256_file
from tools.eval_policy import _build_net_from_cfg


def audit(config):
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("neural preparation audit output already exists")
    device = config.get("device", "cpu")
    torch.set_num_threads(int(config.get("threads", 1)))
    payload = load_checkpoint(Path(config["checkpoint"]), map_location="cpu")
    original_config = payload["cfg"]
    settings = original_config.get("smdp_ppo", original_config)
    if settings.get("aux_spec") != PUBLIC_CONTEXT_SCHEMA:
        raise ValueError("audit needs the actual public-history controller core")
    original, _, _ = _build_net_from_cfg(original_config, 20, device)
    state = payload.get("ema_state_dict") or payload["state_dict"]
    original.load_state_dict(state, strict=True)
    changed_config = deepcopy(original_config)
    changed = changed_config.get("smdp_ppo", changed_config)
    changed["candidate_conditioned_trunk"] = False
    changed["candidate_context_residual"] = False
    reusable, _, _ = _build_net_from_cfg(changed_config, 20, device)
    removed = [k for k in state if k == "side_condition_scale"]
    reusable.load_state_dict({k:v for k,v in state.items() if k not in removed}, strict=True)
    paths = [Path(p) for p in config["replay_shards"]]
    report = dict(schema="drmc-neural-preparation-audit-v1", status="Running", config=config,
        checkpoint_sha256=sha256_file(Path(config["checkpoint"])),
        source_sha256={str(p):sha256_file(p) for p in paths},
        changed_config=changed_config, removed_parameters=removed,
        trained_examples=0, outcome_frames=0, product_gates_passed=False, records=[],
        scope="Exact reuse within an untrained unconditioned-encoder variant. All nine previews are conditional inputs, not equiprobable chance outcomes. Preparation and remaining decision work are charged separately. No strength or migration-parity claim.")
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, report)

    def synchronize():
        if device.startswith("mps"):
            torch.mps.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()

    def measure(call):
        synchronize()
        start = time.perf_counter()
        value = call()
        synchronize()
        return value, 1000*(time.perf_counter()-start)

    seen = set()
    repeats = int(config.get("repeats", 3))
    if repeats < 1:
        raise ValueError("repeats must be positive")
    try:
        for path in paths:
            source_count = 0
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                if metadata.get("observation_schema") != PUBLIC_CONTEXT_SCHEMA:
                    raise ValueError("replay does not carry the public context contract")
                for index, seed in enumerate(data["game_seed"]):
                    if int(seed) in seen:
                        continue
                    seen.add(int(seed))
                    start, end = map(int, data["offsets"][index:index+2])
                    count = end-start
                    if count < 1:
                        raise ValueError("recorded decision has no legal action")
                    width = max(32, 1 << (count-1).bit_length())
                    actions = np.full((1, width), -1, np.int64)
                    costs = np.zeros((1, width), np.float32)
                    mask = np.arange(width)[None, :] < count
                    actions[0, :count] = data["actions"][start:end]
                    costs[0, :count] = data["costs"][start:end]
                    single = tuple(torch.as_tensor(x, device=device) for x in (
                        data["observation"][index:index+1].astype(np.float32),
                        data["pill"][index:index+1].astype(np.int64),
                        data["preview"][index:index+1].astype(np.int64), actions, costs, mask,
                        data["public_context"][index:index+1].astype(np.float32)))
                    nine = [x.expand(9, *x.shape[1:]) for x in single]
                    nine[2] = torch.cartesian_prod(torch.arange(3, device=device), torch.arange(3, device=device))
                    nine[6] = nine[6].clone()
                    nine[6][:, 6:12] = torch.nn.functional.one_hot(nine[2], 3).flatten(1).to(nine[6].dtype)

                    def forward(net, x, prepared=None):
                        return net(*x[:6], aux=x[6], prepared_bottles=prepared)

                    with torch.inference_mode():
                        # Warm the exact shapes used below; cold preparation is
                        # separately recorded, never hidden in the tail budget.
                        forward(original, single)
                        forward(reusable, single)
                        full_nine = forward(reusable, nine)
                        prepared, prep_ms = measure(lambda: (
                            prepare_bottle(reusable, single[0][:, :8]),
                            prepare_bottle(reusable, single[0][:, 8:16])))
                        for ready in (prepared, (prepared[0], None)):
                            forward(reusable, nine, ready)
                            forward(reusable, single, ready)
                        warm, original_output = forward(reusable, single), forward(original, single)
                        original_p = original_output[0][:, :count].softmax(-1)
                        migrated_logp = warm[0][:, :count].log_softmax(-1)
                        drift = dict(policy_kl=float((original_p*(original_p.clamp_min(1e-30).log()-migrated_logp)).sum()),
                            same_choice=bool(original_output[0].argmax(-1)==warm[0].argmax(-1)),
                            value_abs_error=float((original_output[1]-warm[1]).abs().max()))
                        times = {k:[] for k in ("full_one_ms", "full_nine_ms", "prepared_one_ms",
                            "prepared_nine_ms", "own_prepared_fresh_opponent_one_ms", "preparation_ms")}
                        calls = [lambda: forward(reusable, single), lambda: forward(reusable, nine),
                            lambda: forward(reusable, single, prepared), lambda: forward(reusable, nine, prepared),
                            lambda: forward(reusable, single, (prepared[0], None)),
                            lambda: (prepare_bottle(reusable, single[0][:, :8]),
                                     prepare_bottle(reusable, single[0][:, 8:16]))]
                        names = list(times)
                        for repeat in range(repeats):
                            # Rotate order across both roots and repetitions.
                            rotation = (len(report["records"])+repeat) % len(calls)
                            for pos in list(range(rotation, len(calls)))+list(range(rotation)):
                                _, elapsed = measure(calls[pos])
                                times[names[pos]].append(elapsed)
                        actual = forward(reusable, nine, prepared)
                        probability_error = float((actual[0].softmax(-1)-full_nine[0].softmax(-1)).abs().max())
                        value_error = float((actual[1]-full_nine[1]).abs().max())
                        choices_equal = bool((actual[0].argmax(-1)==full_nine[0].argmax(-1)).all())
                    record = dict(source=str(path), source_row=index, game_seed=int(seed),
                        pace=metadata["pace"], level=metadata["level"], candidates=count,
                        timing=times, first_preparation_ms=prep_ms, migration_drift=drift,
                        prepared_probability_max_error=probability_error,
                        prepared_value_max_error=value_error, prepared_choices_equal=choices_equal,
                        bottle_rows_full_nine=18, bottle_rows_prepared_nine=2)
                    report["records"].append(record)
                    source_count += 1
                    dump(output, report)
                    if probability_error > 1e-5 or value_error > 1e-5 or not choices_equal:
                        raise RuntimeError("prepared evaluation failed its fixed numerical/choice comparison")
                    if (len(report["records"]) >= int(config.get("states", 16))
                            or source_count >= int(config.get("states_per_shard", 4))):
                        break
            if len(report["records"]) >= int(config.get("states", 16)):
                break
        if len(report["records"]) != int(config.get("states", 16)):
            raise ValueError("not enough distinct recorded reset seeds")
        report["status"] = "Complete"
    except BaseException as error:
        report.update(status="Failed", error=str(error))
        raise
    finally:
        dump(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    audit(json.loads(parser.parse_args().config.read_text()))
