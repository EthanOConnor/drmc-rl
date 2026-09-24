"""Engine + planner parity for a new arena host, without checkpoints or torch.

Plays fixed-seed frame-by-frame games whose placements are chosen by a
deterministic rule over the complete native candidate inventory (the network is
replaced, everything else is the arena's own planning, controller-script and
validation path). Every frame's native state, every candidate inventory and
every controller tape is hashed. The digests must equal the reference recorded
on the Mac for the same native engine commit; a mismatch means this host's
compiler/platform build of ``libdrmario_pool`` or ``libdrm_reach_full`` differs.

    python -m tools.arena_host_selftest --native-library PATH [--reach-library PATH]
    python -m tools.arena_host_selftest ... --record   # add this build to the reference file

Network numerics are validated separately by the study coordinator's
``--calibration-games`` replay of journaled games (docs/ARENA_HOSTS.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

REFERENCE = Path(__file__).with_name("arena_host_selftest.json")
SEEDS = (11, 222, 3333, 44444)
CASES = (("frame_perfect", 14), ("top_humans", 20), ("normal", 14))
FIELDS = ("frame", "garbage_sent_total", "pill_counter_total", "mode", "phase", "subphase", "spawn_id",
          "level", "speed", "speed_ups", "x", "y_top", "rotation", "speed_counter", "horizontal_velocity",
          "held_buttons", "frame_parity", "terminal", "outcome", "event_type")


def _update(digest, value):
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode() + repr(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(b"[%d" % len(value))
        for item in value:
            _update(digest, item)
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(repr(key).encode())
            _update(digest, value[key])
    else:
        digest.update(repr(value).encode())


def play(native_library, pace_name, level, frames=20000):
    from drmc_rl.envs.backends.vs_frames import FrameVsPool
    from drmc_rl.execution.pace import resolve_pace
    from drmc_rl.human.anticipation import execution_for_action
    from drmc_rl.human.backend import NoReachablePlacement, plan_candidates
    from drmc_rl.planning.native_reach import NativeReachabilityRunner

    pace = resolve_pace(pace_name)
    planner = NativeReachabilityRunner()
    engine, plans = hashlib.sha256(), hashlib.sha256()
    count, decisions = len(SEEDS), 0
    controllers, last = [None] * (2 * count), [None] * (2 * count)
    try:
        with FrameVsPool(count, lib_path=native_library) as pool:
            pool.reset(list(SEEDS), level=level)
            for frame in range(frames):
                states = pool.states
                for state in states:
                    engine.update(bytes(state.board) + bytes(state.pill) + bytes(state.preview))
                    engine.update(repr(tuple(int(getattr(state, f)) for f in FIELDS)).encode())
                if all(states[2 * p].terminal for p in range(count)):
                    break
                buttons = [0] * (2 * count)
                for side, current in enumerate(states):
                    if states[2 * (side // 2)].terminal:
                        continue
                    if not current.falling:
                        controllers[side] = None
                        continue
                    key = (current.spawn_id, current.pill_counter_total)
                    if last[side] != key:
                        last[side] = key
                        delay = max(4, pace.reaction_frames)
                        try:
                            candidate = plan_candidates(planner, pool.semantic(side), delay, pace)
                        except NoReachablePlacement:
                            plans.update(b"none")
                            continue
                        _update(plans, candidate)
                        feasible = np.flatnonzero(np.asarray(candidate[-1]) != 65535)
                        # Deepest rows first keeps games long; the rotation among
                        # equally deep placements varies which scripts are used.
                        rows = (feasible % 128) // 8
                        deepest = feasible[rows == rows.max()]
                        action = int(deepest[(decisions * 7919 + side) % len(deepest)])
                        move = execution_for_action(candidate, action, pace, delay=delay)
                        _update(plans, [move["placement"], move["controller_frames"]])
                        controllers[side] = (frame + delay, move)
                        decisions += 1
                    if controllers[side] is None:
                        continue
                    start, move = controllers[side]
                    index = frame - start
                    if 0 <= index < len(move["controller_frames"]):
                        buttons[side] = move["controller_frames"][index]
                pool.step(buttons)
    finally:
        planner.close()
    return dict(engine=engine.hexdigest(), planner=plans.hexdigest(), decisions=decisions, frames=frame)


def native_commit(root=Path(__file__).resolve().parents[1]):
    out = subprocess.run(["git", "-C", str(root / "vendor" / "drmario_native"), "rev-parse", "--short=7", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return out or "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--reach-library")
    parser.add_argument("--native-commit", help="engine commit the library was built from (default: submodule HEAD)")
    parser.add_argument("--record", action="store_true", help="store this result as the reference")
    args = parser.parse_args(argv)
    if args.reach_library:
        os.environ["DRMARIO_REACH_LIB"] = str(Path(args.reach_library).resolve())
    commit = (args.native_commit or native_commit())[:7]
    result = {f"{pace}@{level}": play(args.native_library, pace, level) for pace, level in CASES}
    host = dict(system=platform.system(), machine=platform.machine(), python=platform.python_version())
    reference = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    if args.record:
        reference[commit] = dict(result=result, recorded_on=host)
        REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    expected = reference.get(commit, {}).get("result")
    verdict = "no reference for this native commit" if expected is None else (
        "match" if expected == result else "MISMATCH")
    print(json.dumps(dict(native_commit=commit, host=host, verdict=verdict, result=result), indent=1))
    sys.exit(0 if verdict == "match" else 1)


if __name__ == "__main__":
    main()
