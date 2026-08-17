"""Build a deterministic stratified bank of exact native pair snapshots."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.search.native_pair import capture_native_state, state_to_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file


def _candidate_bin(count: int) -> str:
    if count <= 16:
        return "01-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    return "65-plus"


def _tactical_stratum(state, root_side: int) -> str:
    public = state.privileged.public
    own = public.sides[root_side]
    if state.privileged.pending_attacks[root_side] > 0:
        return "incoming-garbage"
    if (own.viruses_remaining or 0) <= 4:
        return "race-finish"
    if any(tile != 0xFF for tile in own.board[: 3 * 8]):
        return "topout-defense"
    if any(tile != 0xFF for tile in own.board[3 * 8 : 7 * 8]):
        return "high-pressure"
    return "midgame"


def _condition_visible_reserve(
    belief: PillReserveBelief, runner: DrMarioVsPoolRunner
) -> PillReserveBelief:
    result = belief
    for side in range(2):
        result = result.condition_visible(
            reserve_counter=int(runner.buffers.spawn_id[side]),
            falling_colors=tuple(
                int(value) for value in runner.buffers.pill_colors[side]
            ),
            preview_colors=tuple(
                int(value) for value in runner.buffers.preview_colors[side]
            ),
        )
    return result


def _atomic_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as target:
                for row in rows:
                    target.write(canonical_json(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-decisions-per-game", type=int, default=96)
    parser.add_argument(
        "--states-per-game",
        type=int,
        default=8,
        help="cap captured acting-side states before rotating level/speed/seed",
    )
    parser.add_argument("--level", type=int, action="append", default=[])
    parser.add_argument("--speed", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.states < 1 or args.max_decisions_per_game < 1 or args.states_per_game < 1:
        parser.error("state and decision counts must be positive")
    levels = tuple(args.level or (5, 10, 15, 20))
    speeds = tuple(args.speed or (0, 1, 2))
    if any(not 0 <= value <= 20 for value in levels):
        parser.error("levels must be in [0,20]")
    if any(value not in (0, 1, 2) for value in speeds):
        parser.error("speeds must be 0, 1, or 2")

    rng = np.random.default_rng(args.seed)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    rows: list[dict[str, object]] = []
    game_index = 0
    posterior_seed_counts: list[int] = []
    try:
        while len(rows) < args.states:
            game_start_rows = len(rows)
            level = levels[game_index % len(levels)]
            speed = speeds[(game_index // len(levels)) % len(speeds)]
            seed = int(rng.integers(0, 65536))
            spec = build_vs_reset_spec(
                level=(level, level),
                speed_setting=(speed, speed),
                rng_state=(seed & 0xFF, (seed >> 8) & 0xFF),
                rng_override=True,
                frame_counter_base=int(rng.integers(0, 256)),
            )
            runner.reset(None, [spec])
            reserve_belief = PillReserveBelief()
            for decision in range(args.max_decisions_per_game):
                reserve_belief = _condition_visible_reserve(reserve_belief, runner)
                initial_viruses = min(84, 4 * (level + 1))
                state = capture_native_state(
                    runner,
                    level=level,
                    speed_setting=speed,
                    viruses_initial=(initial_viruses, initial_viruses),
                )
                if state.privileged.decision_boundary.value == "terminal":
                    break
                acting = [side for side, flag in enumerate(state.privileged.need_action) if flag]
                for root_side in acting:
                    legal = state.legal_actions_by_side[root_side]
                    if not legal:
                        continue
                    payload = state_to_payload(state)
                    identity_payload = {
                        "checkpoint": hashlib.sha256(
                            state.privileged.engine_checkpoint
                        ).hexdigest(),
                        "root_side": root_side,
                    }
                    payload.update(
                        {
                            "id": hashlib.sha256(canonical_json(identity_payload)).hexdigest(),
                            "root_side": root_side,
                            "game_index": game_index,
                            "decision_index": decision,
                            "level": level,
                            "speed": speed,
                            "candidate_count": len(legal),
                            "candidate_count_bin": _candidate_bin(len(legal)),
                            "tactical_stratum": _tactical_stratum(state, root_side),
                            "clock_skew_bin": min(
                                abs(
                                    state.privileged.pair_clocks[0]
                                    - state.privileged.pair_clocks[1]
                                )
                                // 30,
                                4,
                            ),
                            "reserve_belief": reserve_belief.to_dict(),
                        }
                    )
                    posterior_seed_counts.append(reserve_belief.seed_count)
                    rows.append(payload)
                    if len(rows) >= args.states:
                        break
                if len(rows) >= args.states or len(rows) - game_start_rows >= args.states_per_game:
                    break
                actions = np.full(2, -2, dtype=np.int32)
                for side in acting:
                    legal = state.legal_actions_by_side[side]
                    actions[side] = int(rng.choice(legal)) if legal else -1
                runner.step_strict(actions)
                if int(runner.buffers.terminated[0]) or int(runner.buffers.truncated[0]):
                    break
            game_index += 1
    finally:
        runner.close()

    _atomic_gzip_jsonl(args.output, rows)
    seed_counts = np.asarray(posterior_seed_counts, dtype=np.int64)
    manifest = {
        "schema": "drmc-pair-state-pilot-bank-v2",
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "states": len(rows),
        "seed": args.seed,
        "levels": list(levels),
        "speeds": list(speeds),
        "pair_state_schema": "drmc-pair-state-v2",
        "native_checkpoint_schema": "drm-vspool-snapshot-v1",
        "chance_model": CHANCE_MODEL_ID,
        "reserve_belief_history": "all visible falling/preview entries at captured boundaries",
        "posterior_seed_count": {
            "min": int(seed_counts.min(initial=0)),
            "median": float(np.median(seed_counts)) if seed_counts.size else 0.0,
            "max": int(seed_counts.max(initial=0)),
        },
        "diagnostic_only": True,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
