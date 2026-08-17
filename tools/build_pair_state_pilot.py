"""Build a deterministic bank of exact native pair snapshots.

The production-shaped path rolls both sides with the frozen Strong League
continuation mixture, preserves public reserve-belief history, and samples
throughout complete games. Random actions remain available only when no mixture
is supplied and are explicitly marked diagnostic in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import (
    VS_OUTCOME_DRAW,
    VS_OUTCOME_LOSS,
    VS_OUTCOME_WIN,
    DrMarioVsPoolRunner,
    build_vs_reset_spec,
)
from drmc_rl.search.native_pair import capture_native_state, state_to_payload
from drmc_rl.search.pill_belief import CHANCE_MODEL_ID, PillReserveBelief
from drmc_rl.search.strong_league import FrozenStrongLeagueMixture
from drmc_rl.search.strong_league_memberwise import (
    read_davidson_calibration,
    read_mixture_members,
)
from drmc_rl.teachers.counterfactual_release import canonical_json, sha256_file
from drmc_rl.teachers.state_bank import select_game_rows

PILL_TYPES = frozenset((0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0))
_OUTCOME_NAME = {
    VS_OUTCOME_WIN: "win",
    VS_OUTCOME_DRAW: "draw",
    VS_OUTCOME_LOSS: "loss",
}


def _candidate_bin(count: int) -> str:
    if count <= 16:
        return "01-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    return "65-plus"


def _pill_occupied(tile: int) -> bool:
    """Treat player-created pill material, not initial viruses, as stack pressure."""

    return (int(tile) & 0xF0) in PILL_TYPES


def _tactical_stratum(state, root_side: int) -> str:
    public = state.privileged.public
    own = public.sides[root_side]
    if state.privileged.pending_attacks[root_side] > 0:
        return "incoming-garbage"
    if (own.viruses_remaining or 0) <= 4:
        return "race-finish"
    if any(_pill_occupied(tile) for tile in own.board[: 3 * 8]):
        return "topout-defense"
    if any(_pill_occupied(tile) for tile in own.board[3 * 8 : 7 * 8]):
        return "high-pressure"
    return "midgame"


def _condition_visible_reserve(
    belief: PillReserveBelief, runner: DrMarioVsPoolRunner
) -> PillReserveBelief:
    result = belief
    for side in range(2):
        result = result.condition_visible(
            reserve_counter=int(runner.buffers.spawn_id[side]),
            falling_colors=tuple(int(value) for value in runner.buffers.pill_colors[side]),
            preview_colors=tuple(int(value) for value in runner.buffers.preview_colors[side]),
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


def _choose_action(
    mixture: FrozenStrongLeagueMixture | None,
    state,
    side: int,
    legal: tuple[int, ...],
    rng: np.random.Generator,
) -> int:
    if not legal:
        return -1
    if mixture is None:
        return int(rng.choice(legal))
    probability = np.asarray(mixture.prior(state, side, legal), dtype=np.float64)
    if probability.shape != (len(legal),) or not np.isfinite(probability).all():
        raise RuntimeError("frozen rollout policy returned invalid action probabilities")
    return int(legal[int(np.argmax(probability))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-decisions-per-game", type=int, default=320)
    parser.add_argument(
        "--states-per-game",
        type=int,
        default=16,
        help=(
            "select at most this many rows after rolling the whole game; "
            "selection favors globally underrepresented tactical strata"
        ),
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100000,
        help="fail closed rather than loop forever when a requested bank cannot be filled",
    )
    parser.add_argument("--level", type=int, action="append", default=[])
    parser.add_argument("--speed", type=int, action="append", default=[])
    parser.add_argument("--mixture-manifest", type=Path)
    parser.add_argument("--wdl-calibration", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (
        args.states < 1
        or args.max_decisions_per_game < 1
        or args.states_per_game < 1
        or args.max_games < 1
    ):
        parser.error("state, game, and decision counts must be positive")
    if (args.mixture_manifest is None) != (args.wdl_calibration is None):
        parser.error("--mixture-manifest and --wdl-calibration must be supplied together")
    levels = tuple(args.level or (5, 10, 15, 20))
    speeds = tuple(args.speed or (0, 1, 2))
    if any(not 0 <= value <= 20 for value in levels):
        parser.error("levels must be in [0,20]")
    if any(value not in (0, 1, 2) for value in speeds):
        parser.error("speeds must be 0, 1, or 2")

    mixture = None
    rollout_policy = "diagnostic-random-actions"
    rollout_manifest_sha256 = None
    if args.mixture_manifest is not None:
        members = read_mixture_members(args.mixture_manifest)
        calibration = read_davidson_calibration(args.wdl_calibration)
        mixture = FrozenStrongLeagueMixture(members, calibration, device=args.device)
        rollout_policy = "frozen-strong-league-mixture-argmax"
        rollout_manifest_sha256 = sha256_file(args.mixture_manifest)

    rng = np.random.default_rng(args.seed)
    runner = DrMarioVsPoolRunner(num_pairs=1)
    rows: list[dict[str, object]] = []
    game_index = 0
    posterior_seed_counts: list[int] = []
    tactical_counts: dict[str, int] = {}
    candidate_states_considered = 0
    natural_terminal_games = 0
    horizon_incomplete_games = 0
    try:
        while len(rows) < args.states:
            if game_index >= args.max_games:
                raise RuntimeError(
                    f"failed to collect {args.states} states within {args.max_games} games"
                )
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
            game_id = hashlib.sha256(runner.snapshot(0)).hexdigest()
            initial_board = np.asarray(
                runner.buffers.board_bytes.reshape(2, 128)[0],
                dtype=np.uint8,
            ).copy()
            reserve_belief = PillReserveBelief.from_initial_board(
                level=level,
                board=initial_board,
            )
            game_candidates: list[dict[str, object]] = []
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
                decision_rows: dict[int, dict[str, object]] = {}
                for root_side in acting:
                    legal = state.legal_actions_by_side[root_side]
                    if not legal:
                        continue
                    tactical = _tactical_stratum(state, root_side)
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
                            "game_id": game_id,
                            "decision_index": decision,
                            "level": level,
                            "speed": speed,
                            "candidate_count": len(legal),
                            "candidate_count_bin": _candidate_bin(len(legal)),
                            "tactical_stratum": tactical,
                            "clock_skew_bin": min(
                                abs(
                                    state.privileged.pair_clocks[0]
                                    - state.privileged.pair_clocks[1]
                                )
                                // 30,
                                4,
                            ),
                            "reserve_belief": reserve_belief.to_dict(),
                            "reserve_seed_count": reserve_belief.seed_count,
                            "rollout_policy": rollout_policy,
                            "speed_ups": min(
                                int(runner.buffers.spawn_id[root_side]) // 10,
                                0x31,
                            ),
                        }
                    )
                    game_candidates.append(payload)
                    decision_rows[root_side] = payload
                    candidate_states_considered += 1
                actions = np.full(2, -2, dtype=np.int32)
                for side in acting:
                    actions[side] = _choose_action(
                        mixture,
                        state,
                        side,
                        state.legal_actions_by_side[side],
                        rng,
                    )
                    if side in decision_rows:
                        decision_rows[side]["observed_action"] = int(actions[side])
                runner.step_strict(actions)
                if int(runner.buffers.terminated[0]) or int(runner.buffers.truncated[0]):
                    break

            terminal_outcomes = tuple(int(item) for item in runner.buffers.outcome[:2])
            natural_terminal = all(item in _OUTCOME_NAME for item in terminal_outcomes)
            if natural_terminal:
                natural_terminal_games += 1
            else:
                horizon_incomplete_games += 1
            for row in game_candidates:
                side = int(row["root_side"])
                row["natural_outcome_available"] = natural_terminal
                row["outcome"] = (
                    _OUTCOME_NAME[terminal_outcomes[side]] if natural_terminal else None
                )

            remaining = args.states - len(rows)
            selected = select_game_rows(
                game_candidates,
                limit=min(args.states_per_game, remaining),
                global_tactical_counts=tactical_counts,
                seed=args.seed + game_index,
            )
            rows.extend(selected)
            posterior_seed_counts.extend(int(row["reserve_seed_count"]) for row in selected)
            game_index += 1
    finally:
        runner.close()

    _atomic_gzip_jsonl(args.output, rows)
    seed_counts = np.asarray(posterior_seed_counts, dtype=np.int64)
    manifest = {
        "schema": "drmc-pair-state-candidate-bank-v4",
        "artifact": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "states": len(rows),
        "candidate_states_considered": candidate_states_considered,
        "games_rolled": game_index,
        "natural_terminal_games": natural_terminal_games,
        "horizon_incomplete_games": horizon_incomplete_games,
        "game_identity": "sha256-initial-native-pair-snapshot",
        "observed_action_provenance": "frozen-rollout-action-at-captured-boundary",
        "natural_outcome_provenance": "authoritative-native-terminal-vs-outcome",
        "seed": args.seed,
        "levels": list(levels),
        "speeds": list(speeds),
        "pair_state_schema": "drmc-pair-state-v2",
        "native_checkpoint_schema": "drm-vspool-snapshot-v1",
        "chance_model": CHANCE_MODEL_ID,
        "reserve_belief_history": (
            "public initial virus bottle plus all visible falling/preview "
            "entries at captured boundaries"
        ),
        "reserve_initial_board_conditioned": True,
        "per_game_selection": "whole-game-global-tactical-round-robin-v1",
        "posterior_seed_count": {
            "min": int(seed_counts.min()) if seed_counts.size else 0,
            "median": float(np.median(seed_counts)) if seed_counts.size else 0.0,
            "max": int(seed_counts.max()) if seed_counts.size else 0,
        },
        "tactical_counts": dict(sorted(tactical_counts.items())),
        "rollout_policy": rollout_policy,
        "rollout_policy_manifest_sha256": rollout_manifest_sha256,
        "diagnostic_only": mixture is None,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
