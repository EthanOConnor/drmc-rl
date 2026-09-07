"""Sample recorded held-button traces from an immutable human corpus release.

This prepares profile-fitting data; it does not open the human-execution gate.
Every accepted script reproduces the recorded lock pose and frame under the
audited FBNeo boundary. This is not a new end-to-end emulator replay certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from drmc_rl.data.human_corpus import HumanCorpus, decode_input_rle
from tools.audit_execution_replay import FIELDS, replay_row

COLUMNS = tuple(dict.fromkeys((*FIELDS,
    "decision_id", "game_id", "source_blob_sha256", "player", "day",
    "random_split", "player_fold", "held_before_spawn", "spawn_frame", "lock_frame",
)))


def execution_row(row: dict, *, rating: float, rating_sd: float) -> dict:
    """Keep only covered, aligned scripts matching recorded lock pose and time."""
    if row["lock_repaired"]:
        raise ValueError("repaired_lock")
    if not np.isfinite(rating) or not np.isfinite(rating_sd):
        raise ValueError("nonfinite_rating")
    if row.get("held_before_spawn") is None:
        raise ValueError("missing_prior_held")
    initial = int(row["held_before_spawn"])
    if not 0 <= initial <= 255 or initial & 3 == 3 or initial & 0xC0 == 0xC0:
        raise ValueError("unsupported_prior_held")
    raw = decode_input_rle(row["input_rle_u16_u8"], row["input_frames"])
    if len(raw) < 2 or len(raw) != int(row["tau_frames"]) + 1:
        raise ValueError("input_window_mismatch")
    if raw[0] != int(row["held_at_spawn"]):
        raise ValueError("spawn_held_mismatch")
    replay = replay_row(row, parity_xor=1, input_delay_frames=1)
    if replay["status"] == "excluded":
        raise ValueError(replay["reason"])
    if replay["status"] != "match":
        raise ValueError("recorded_lock_mismatch")
    return {
        "decision_id": row["decision_id"], "game_id": row["game_id"],
        "source_blob_sha256": row["source_blob_sha256"],
        "player_id": hashlib.sha256(str(row["player"]).encode()).hexdigest(),
        "day": int(row["day"]), "split": row["random_split"],
        "player_fold": int(row["player_fold"]),
        "rating": float(rating), "rating_sd": float(rating_sd),
        "speed": int(row["speed"]), "speed_ups": int(row["speed_ups"]),
        "initial_buttons": initial,
        "initial_horizontal_velocity": int(row["horizontal_velocity"]),
        "spawn_frame": int(row["spawn_frame"]), "lock_frame": int(row["lock_frame"]),
        "field_hex": bytes(row["field"]).hex(),
        "initial_speed_counter": int(row["speed_counter"]),
        "initial_frame_parity": (int(row["frame_counter"]) & 1) ^ 1,
        "lock_pose": list(replay["expected_pose"]),
        "recorded_lock_verified": True,
        "script": list(raw[:-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", required=True, help="immutable release path, not latest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.release == "latest" or not 0 < args.fraction <= 1:
        parser.error("use an immutable release and fraction in (0,1]")
    if args.output.exists():
        parser.error("output already exists; use a new sample identity")
    corpus = HumanCorpus(args.root, release=args.release)
    coverage_verified = corpus.manifest.get("contracts", {}).get("input_coverage") == "complete-spawn-lock-window-v1"
    prior_verified = corpus.manifest.get("contracts", {}).get("held_before_spawn") == "previous-recorded-frame-or-null-v1"
    if not coverage_verified or not prior_verified:
        parser.error("corpus must declare complete input coverage and recorded prior-held state")
    corpus.verify()
    rng = np.random.default_rng(args.seed)
    accepted, excluded, strata = 0, Counter(), Counter()
    scanned = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    # Never advance latest mid-scan. HumanCorpus has resolved the immutable
    # directory, and the output sidecar binds the exact producer manifest.
    with temporary.open("x") as stream:
        for batch in corpus.batches("decisions", columns=COLUMNS):
            scanned += batch.num_rows
            selected = np.flatnonzero(rng.random(batch.num_rows) < args.fraction)
            for row in batch.take(selected).to_pylist():
                rating, sd = corpus.rating_at(row["player"], int(row["day"]))
                if rating is None or sd is None:
                    excluded["missing_rating"] += 1
                    continue
                try:
                    output = execution_row(row, rating=rating, rating_sd=sd)
                except (ValueError, TypeError) as error:
                    excluded[str(error)] += 1
                    continue
                stream.write(json.dumps(output, separators=(",", ":")) + "\n")
                accepted += 1
                strata[f'{int(rating // 400) * 400}/{output["speed"]}/{output["split"]}'] += 1
    temporary.replace(args.output)
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest_bytes = (corpus.release_dir / "manifest.json").read_bytes()
    result = {
        "schema": "drmc-execution-corpus-sample-v2", "corpus_release": corpus.release_id,
        "corpus_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "output_sha256": digest, "seed": args.seed, "fraction": args.fraction,
        "scanned_decisions": scanned, "accepted": accepted, "excluded": dict(excluded),
        "strata": dict(sorted(strata.items())),
        "input_contract": "fbneo-processed-before-motion-v1: prior held byte, raw[:-1], recorded parity xor 1",
        "source_input_coverage_verified": coverage_verified,
        "source_prior_held_verified": prior_verified,
        "recording_alignment_verified": True,
        "recorded_lock_replay_verified": True,
        "rom_replay_verified": False,
        "sampling": "uniform decision Bernoulli sample; retain whole-game and player holdout ids",
        "gate_pass": False,
        "source_code_sha256": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
            "tools/extract_execution_corpus.py", "tools/audit_execution_replay.py",
            "drmc_rl/planning/fast_reach.py", "drmc_rl/data/human_corpus.py")},
    }
    Path(str(args.output) + ".manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
