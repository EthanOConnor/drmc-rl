"""Measure corpus public-view reconstruction against the native engine.

Plays native frame-VS games with a simple exact-afterstate heuristic, writes
each side's placements in the corpus row schema (bottle at spawn, raw pills,
spawn/lock frames, lock pose), rebuilds every decision's public view with
``drmc_rl.human.corpus_public_state.CorpusGame`` and compares the encoded
``public_pair_context_v3`` vector with the one the engine produced.

The falling opponent pose is taken from the native trace here: its corpus
reconstruction (recorded inputs under the FBNeo contract) is audited
separately by ``tools/audit_execution_replay.py``. Everything else (boards,
pills, phases, virus counts, event history and ages) is reconstructed.

    DRMARIO_REACH_LIB=.../libdrm_reach_full.dylib python -m tools.validate_corpus_public_state \
        --native-library .../libdrmario_pool.dylib --games 8 --output report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from drmc_rl.game.afterstate import FACT_NAMES, resolve_placement
from drmc_rl.game.public_context import CONTEXT_FEATURE_NAMES, encode_public_context
from drmc_rl.human.corpus_public_state import CorpusGame

_F = {name: i for i, name in enumerate(FACT_NAMES)}


def _heuristic(field, pill, legal, rng):
    best, score = None, None
    for action in legal:
        _after, facts = resolve_placement(field, pill, int(action))
        value = (3 * facts[_F["tiles_cleared"]] + 5 * facts[_F["viruses_cleared"]]
                 + 12 * max(0.0, facts[_F["lines"]] - 1) + 2 * facts[_F["lines"]]
                 - 1.5 * facts[_F["height_after"]] - 30 * facts[_F["spawn_blocked"]] + rng.normal() * 0.8)
        if score is None or value > score:
            best, score = int(action), value
    return best


def play(seed, *, native_library, pace_name="top_humans", max_frames=40000):
    """One native game: per-frame side states and every decision's native public view."""
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.execution.pace import resolve_pace
    from drmc_rl.human.anticipation import execution_for_action
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    pace = resolve_pace(pace_name)
    rng = np.random.default_rng(seed)
    planner = NativeReachabilityRunner()
    frames, decisions = [], []
    controllers, last_spawn = [None, None], [None, None]
    with FrameVsPool(1, lib_path=native_library) as pool:
        pool.reset([seed], level=14)
        for frame in range(max_frames):
            states = pool.states
            if states[0].terminal:
                break
            frames.append([dict(frame=int(x.frame), falling=bool(x.falling), pill=tuple(x.pill),
                                preview=tuple(x.preview), pose=(int(x.x), int(np.int8(x.y_top)), int(x.rotation)),
                                board=bytes(x.board)) for x in states])
            for side in (0, 1):
                current = states[side]
                key = (current.spawn_id, current.pill_counter_total)
                if not current.falling or last_spawn[side] == key:
                    continue
                last_spawn[side] = key
                state = pool.semantic(side, public_context=True)
                public = state["public_pair_state"]
                decisions.append(dict(frame=int(current.frame), side=side,
                                      context=encode_public_context(public, side, None)))
                delay = max(4, pace.reaction_frames)
                try:
                    candidate = plan_candidates(planner, state, delay, pace)
                except NoReachablePlacement:
                    controllers[side] = None
                    continue
                legal = np.flatnonzero(candidate[-1] != 0xFFFF)
                action = _heuristic(np.frombuffer(bytes(current.board), np.uint8), tuple(state["pill"]), legal, rng)
                controllers[side] = (frame + delay, execution_for_action(candidate, action, pace, delay=delay))
            buttons = [0, 0]
            for side, controller in enumerate(controllers):
                if controller is None or not states[side].falling:
                    continue
                start, move = controller
                index = frame - start
                if 0 <= index < len(move["controller_frames"]):
                    buttons[side] = move["controller_frames"][index]
            pool.step(buttons)
    return frames, decisions


def corpus_rows(frames):
    """Placements in the corpus decision schema (slot 1/2, raw NES pill colors)."""
    rows, traces = [], {}
    for side in (0, 1):
        spawn = None
        for i, record in enumerate(frames):
            now, previous = record[side], frames[i - 1][side] if i else None
            if now["falling"] and (previous is None or not previous["falling"]):
                spawn = i
            if spawn is not None and previous is not None and previous["falling"] and not now["falling"]:
                start = frames[spawn][side]
                x, y, rot = now["pose"]
                row = dict(player_slot=side + 1, spawn_frame=start["frame"], lock_frame=now["frame"],
                           tau_frames=now["frame"] - start["frame"], field=start["board"],
                           pill_left=start["pill"][0], pill_right=start["pill"][1],
                           preview_left=start["preview"][0], preview_right=start["preview"][1],
                           lock_x=x, lock_y_top=y, lock_rotation=rot, opp_field=frames[spawn][1 - side]["board"])
                traces[(side, start["frame"])] = [frames[j][side]["pose"] for j in range(spawn, i)]
                rows.append(row)
                spawn = None
    return rows, traces


GROUPS = (
    ("own", lambda n: n.startswith("own.")),
    ("opponent.pill_preview", lambda n: n.startswith("opponent.p")),
    ("opponent.active", lambda n: n.startswith("opponent.") and n.split(".")[1] in (
        "active_known", "controllable", "active_age") or n.startswith("opponent.active_")),
    ("opponent.pose", lambda n: n.startswith("opponent.") and n.split(".")[1] in (
        "column", "row_top", "rotation_0", "rotation_1", "rotation_2", "rotation_3")),
    ("opponent.viruses", lambda n: n.startswith("opponent.viruses")),
    ("opponent.phase", lambda n: n.startswith("opponent.phase")),
    ("boundary_clock", lambda n: n in ("own_decision", "opponent_decision", "terminal", "clock_delta_known",
                                       "clock_delta", "game_age")),
    ("event.kind_side_known", lambda n: n.startswith("event_") and (".kind_" in n or ".side_" in n or n.endswith(".known"))),
    ("event.age", lambda n: n.startswith("event_") and n.endswith("age_frames_log")),
    ("event.payload", lambda n: n.startswith("event_") and n.split(".")[1] in (
        "garbage_size", "tiles_cleared", "viruses_cleared", "lock_row", "lock_column", "rotation", "terminal_outcome")),
)


def compare(frames, decisions, *, native_pose=True):
    rows, traces = corpus_rows(frames)

    def pose_at(placement, age):
        trace = traces.get((placement.side, placement.spawn))
        return trace[min(age, len(trace) - 1)] if trace else (3, 0, 0)

    game = CorpusGame(rows, pose_at=pose_at if native_pose else None)
    lookup = {(p.side, p.spawn): p for side in game.sides.values() for p in side}
    names = np.asarray(CONTEXT_FEATURE_NAMES)
    masks = {g: np.asarray([f(n) for n in names]) for g, f in GROUPS}
    diff = defaultdict(list)
    exact = 0
    worst = defaultdict(float)
    compared = 0
    for decision in decisions:
        placement = lookup.get((decision["side"], decision["frame"]))
        if placement is None:
            continue
        rebuilt = encode_public_context(game.public_state(placement), decision["side"], None)
        delta = np.abs(rebuilt - decision["context"])
        compared += 1
        exact += bool((delta < 1e-5).all())
        for group, mask in masks.items():
            diff[group].append(float(delta[mask].max()))
        for i in np.flatnonzero(delta > 1e-5):
            worst[names[i]] = max(worst[names[i]], float(delta[i]))
    summary = {g: dict(exact_rate=float(np.mean(np.asarray(v) < 1e-5)), mean_max_abs=float(np.mean(v)))
               for g, v in diff.items()}
    return dict(decisions=compared, exact_vectors=exact, groups=summary,
                largest_feature_errors=dict(sorted(worst.items(), key=lambda kv: -kv[1])[:25]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = []
    totals = defaultdict(list)
    for g in range(args.games):
        frames, decisions = play(args.seed + g, native_library=args.native_library)
        report = compare(frames, decisions)
        report["seed"] = args.seed + g
        reports.append(report)
        for group, value in report["groups"].items():
            totals[group].append((value["exact_rate"], report["decisions"]))
    pooled = {g: sum(r * n for r, n in v) / max(1, sum(n for _, n in v)) for g, v in totals.items()}
    result = dict(schema="drmc-corpus-public-state-validation-v1", games=args.games,
                  decisions=sum(r["decisions"] for r in reports),
                  exact_vectors=sum(r["exact_vectors"] for r in reports),
                  pooled_group_exact_rate=pooled, per_game=reports)
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.write_text(text + "\n")
    print(json.dumps(dict(decisions=result["decisions"], exact_vectors=result["exact_vectors"],
                          pooled_group_exact_rate=pooled), indent=1))
    print(json.dumps(reports[0]["largest_feature_errors"], indent=1))


if __name__ == "__main__":
    main()
