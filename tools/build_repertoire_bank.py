"""Select exact tactical opportunities from a synthetic human VS start bank.

The source bank's asynchronous boards and randomized future remain curriculum
approximations, never replay-certified trajectories or counterfactual labels.
Geometry changes reset sampling only; no action target or reward is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from drmc_rl.envs.backends.drmario_vs_pool import DrMarioVsPoolRunner, build_vs_reset_spec
from drmc_rl.game.cascade import resolve_cascade
from drmc_rl.human.afterstate_sim import NativeAfterstateSimulator
from drmc_rl.human.repertoire import placement_geometry
from drmc_rl.training.envs.start_bank import StartBank
from tools.audit_trainer_repertoire import digest


def held_out(game: str, seed: int) -> bool:
    """All positions from one source replay stay on the same side of the split."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{game}".encode()).digest()[:8], "big") % 5 == 0


def opportunities(board, canonical_pill, raw_preview, actions, costs, speed, speed_ups, simulator):
    geometry = [placement_geometry(board, canonical_pill, int(a)) for a in actions]
    raw_pill = np.asarray((1, 0, 2), np.uint8)[canonical_pill]
    effects = simulator.simulate_packed(fields=board[None], pills=raw_pill[None],
        previews=np.asarray(raw_preview)[None], candidate_actions=actions[None],
        candidate_costs=costs[None], candidate_count=np.asarray([len(actions)]),
        speed=np.asarray([speed]), speed_ups=np.asarray([speed_ups]))
    if effects.invalid.any():
        raise ValueError("native start-bank candidate was invalid")
    for i, g in enumerate(geometry):
        if bool(g["first_wave_cells"]) != bool(effects.clear_events[i]):
            raise ValueError(f"geometry/native clear mismatch: action={actions[i]} pill={raw_pill.tolist()} "
                f"geometry={g} events={effects.clear_events[i]} terminal={effects.terminal_reason[i]} "
                f"board={board.tobytes().hex()}")
    return {
        "horizontal": any(g["horizontal_lines"] for g in geometry),
        "long_horizontal": any(g["longest_horizontal"] >= 5 for g in geometry),
        "crossing": any(g["crossing_cells"] for g in geometry),
        "large_first_wave": any(g["first_wave_cells"] >= 8 for g in geometry),
        "cascade": bool(np.any(effects.clear_events >= 2)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-source-rows", type=int, default=4096)
    ap.add_argument("--max-per-game", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()
    if args.output_dir.exists() or min(args.max_source_rows, args.max_per_game) < 1:
        ap.error("use a fresh output directory and positive bounds")
    started = time.monotonic()
    source = np.load(args.source, allow_pickle=False)
    bank = StartBank(args.source)
    # A row-only split would leak positions from the same match into validation.
    games = source["quark_names"][source["quark_idx"]]
    rng = np.random.default_rng(args.seed)
    selected, labels, per_game = [], [], Counter()
    runner = DrMarioVsPoolRunner(num_pairs=1)
    examined, terminal_resets, unsettled = 0, 0, 0
    try:
        with NativeAfterstateSimulator(num_envs=128) as simulator:
            for index in rng.permutation(len(bank))[:args.max_source_rows]:
                game = str(games[index])
                if per_game[game] >= args.max_per_game:
                    continue
                examined += 1
                if any(resolve_cascade(board).settled_field != board.tobytes() for board in bank.boards[index]):
                    unsettled += 1
                    continue
                levels, speeds = source["levels"][index], source["speeds"][index]
                runner.reset(None, [build_vs_reset_spec(level=tuple(map(int, levels)),
                    speed_setting=tuple(map(int, speeds)), rng_override=True,
                    rng_state=(1, 1), **bank.spec_kwargs(int(index)))])
                buf = runner.buffers
                if buf.terminated[0]:
                    terminal_resets += 1
                    continue
                combined = Counter()
                for side in (0, 1):
                    actions = np.flatnonzero(buf.feasible_mask[side]).astype(np.int32)
                    if not len(actions):
                        continue
                    result = opportunities(buf.board_bytes[side], buf.pill_colors[side],
                        source["preview"][index, side], actions, buf.cost_to_lock[side, actions],
                        int(speeds[side]), int(bank.speed_ups[index, side]), simulator)
                    combined.update({key: int(value) for key, value in result.items()})
                if any(combined.values()):
                    selected.append(int(index))
                    labels.append({key: bool(value) for key, value in combined.items()})
                    per_game[game] += 1
                if examined % 128 == 0:
                    print(json.dumps({"examined": examined, "selected": len(selected),
                        "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
    finally:
        runner.close()
    if not selected:
        raise ValueError("no repertoire opportunities found")
    args.output_dir.mkdir(parents=True)
    records, summaries = [], {}
    for split in ("train", "heldout"):
        slots = [j for j, i in enumerate(selected) if held_out(str(games[i]), args.seed) == (split == "heldout")]
        indices = np.asarray([selected[j] for j in slots], np.int64)
        if not len(indices):
            raise ValueError(f"empty {split} split")
        arrays = {key: source[key][indices] for key in source.files if key != "quark_names"}
        arrays.update(quark_names=source["quark_names"], source_row=indices)
        path = args.output_dir / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        counts = Counter(key for j in slots for key, value in labels[j].items() if value)
        summaries[split] = {"rows": len(indices), "games": len(set(games[indices])),
            "opportunities": dict(counts), "sha256": digest(path)}
        records.extend({"source_row": int(selected[j]), "game": str(games[selected[j]]),
            "split": split, "opportunities": labels[j]} for j in slots)
    report = {"schema": "drmc-repertoire-curriculum-v1", "diagnostic_only": True,
        "source_sha256": digest(args.source), "source_path": str(args.source), "seed": args.seed,
        "examined": examined, "terminal_resets": terminal_resets, "unsettled_roots_excluded": unsettled,
        "max_per_game": args.max_per_game,
        "summary": summaries, "records": records,
        "source_code_sha256": {p: digest(p) for p in (str(Path(__file__)),
            "drmc_rl/human/repertoire.py", "drmc_rl/human/afterstate_sim.py")},
        "limitations": ["Synthetic two-board resets from asynchronous human snapshots.",
            "Randomized future; not a reconstructed historical continuation or public-posterior label.",
            "Opportunity labels do not say the move is best or should be imitated.",
            "No named community motif labels without verified event-sequence definitions.",
            "Clean-start paired full-game outcomes govern adoption."],
        "elapsed_seconds": time.monotonic()-started}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
