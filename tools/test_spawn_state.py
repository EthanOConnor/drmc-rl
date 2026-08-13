"""Tests for spawn-state derivation in tools.annotate_replay_events.

Covers:
  1. held-byte -> (hold_dir, rot_hold) mapping, incl. the documented edge
     cases (L+R both held -> right; A+B both held -> A; down ignored).
  2. derive_spawn_state: old-format record -> assumed, new-format -> events,
     --spawn-state assumed forcing.
  3. New-format corpus check: every spawn in the sample v2.1 events file
     derives an "events" state with in-range values consistent with the raw
     held/hv/scnt/fc bytes (file path via DRMC_NEW_EVENTS_JSONL, skipped if
     unset/missing).
  4. CPU vs CUDA batch-planner parity on ~50 real moves from that file using
     the derived true spawn state (skipped without the file or a GPU).

Run: .venv/bin/python -m tools.test_spawn_state   (or pytest tools/test_spawn_state.py)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from drmc_rl.planning.fast_reach import compute_speed_threshold
from tools.annotate_replay_events import (
    BTN_A,
    BTN_B,
    BTN_DOWN,
    BTN_LEFT,
    BTN_RIGHT,
    SPAWN_STATE_KEYS,
    decode_field,
    derive_spawn_state,
    held_to_hd_rh,
    make_batch_planner,
    occupancy_cols,
)

NEW_EVENTS_JSONL = os.environ.get(
    "DRMC_NEW_EVENTS_JSONL",
    "/tmp/claude-1000/-home-ethan-drmario/8c504688-db1f-4e06-ad26-323f92ff6b95/"
    "scratchpad/new_4995.jsonl",
)


def test_held_to_hd_rh():
    # Neutral.
    assert held_to_hd_rh(0x00) == (0, 0)
    # Single directions (drmario_constants.asm: right=$01 left=$02).
    assert held_to_hd_rh(BTN_LEFT) == (1, 0)
    assert held_to_hd_rh(BTN_RIGHT) == (2, 0)
    # Rotation buttons (a=$80 -> CW=1, b=$40 -> CCW=2).
    assert held_to_hd_rh(BTN_A) == (0, 1)
    assert held_to_hd_rh(BTN_B) == (0, 2)
    # Combined direction + rotation.
    assert held_to_hd_rh(BTN_LEFT | BTN_B) == (1, 2)
    assert held_to_hd_rh(BTN_RIGHT | BTN_A) == (2, 1)
    # Edge case: L+R both held -> RIGHT (fallingPill_checkXMove checks
    # btn_right first).
    assert held_to_hd_rh(BTN_LEFT | BTN_RIGHT) == (2, 0)
    # Edge case: A+B both held -> A (checked first in fallingPill_checkRotate).
    assert held_to_hd_rh(BTN_A | BTN_B) == (0, 1)
    # Down / up / start / select carry no spawn-state dimension.
    assert held_to_hd_rh(BTN_DOWN) == (0, 0)
    assert held_to_hd_rh(0x08 | 0x10 | 0x20) == (0, 0)
    assert held_to_hd_rh(BTN_DOWN | BTN_LEFT | BTN_A) == (1, 1)
    # Only the low byte matters.
    assert held_to_hd_rh(0x100 | BTN_RIGHT) == (2, 0)


def test_derive_spawn_state_old_format():
    # Pre-upgrade record: no fc/held/hv/scnt keys -> phase-1 assumptions.
    sp = {"v": 2, "t": "spawn", "p": 1, "f": 1235, "spd": 1, "spdups": 0}
    ss = derive_spawn_state(sp)
    assert ss == {"parity": 1, "sc": 0, "hv": 0, "hd": 0, "rh": 0,
                  "src": "assumed"}
    # Partial keys (defensive: never mix real and assumed dimensions).
    ss = derive_spawn_state({"f": 10, "fc": 3, "held": 0x01})
    assert ss["src"] == "assumed" and ss["hd"] == 0 and ss["parity"] == 0


def test_derive_spawn_state_new_format():
    sp = {"f": 537, "fc": 202, "held": BTN_LEFT | BTN_A, "hv": 0x13, "scnt": 5}
    ss = derive_spawn_state(sp)
    assert ss["src"] == "events"
    assert ss["parity"] == 0          # fc & 1, NOT f & 1 (537 & 1 == 1)
    assert ss["sc"] == 5
    assert ss["hv"] == 0x03           # planner stores 4 bits (hv & 0x0F)
    assert (ss["hd"], ss["rh"]) == (1, 1)
    # --spawn-state assumed forces the old behavior even on new records.
    ss = derive_spawn_state(sp, mode="assumed")
    assert ss == {"parity": 1, "sc": 0, "hv": 0, "hd": 0, "rh": 0,
                  "src": "assumed"}


def _load_new_spawns():
    path = Path(NEW_EVENTS_JSONL)
    if not path.exists():
        return None
    spawns = []
    for line in path.read_text().splitlines():
        if '"t":"spawn"' in line:
            spawns.append(json.loads(line))
    return spawns


def test_new_format_corpus():
    spawns = _load_new_spawns()
    if spawns is None:
        print(f"SKIP: {NEW_EVENTS_JSONL} not found")
        return
    assert len(spawns) > 0
    n_events = 0
    for sp in spawns:
        assert all(k in sp for k in SPAWN_STATE_KEYS), sp
        ss = derive_spawn_state(sp)
        assert ss["src"] == "events"
        n_events += 1
        # Ranges the planner state space accepts.
        assert ss["parity"] in (0, 1)
        assert 0 <= ss["sc"] <= 0xFF
        assert 0 <= ss["hv"] <= 0x0F
        assert ss["hd"] in (0, 1, 2)
        assert ss["rh"] in (0, 1, 2)
        # Consistency with the raw bytes.
        held = int(sp["held"]) & 0xFF
        assert ss["parity"] == (int(sp["fc"]) & 1)
        assert ss["hv"] == (int(sp["hv"]) & 0x0F)
        assert ss["sc"] == (int(sp["scnt"]) & 0xFF)
        if held & BTN_RIGHT:
            assert ss["hd"] == 2
        elif held & BTN_LEFT:
            assert ss["hd"] == 1
        else:
            assert ss["hd"] == 0
        if held & BTN_A:
            assert ss["rh"] == 1
        elif held & BTN_B:
            assert ss["rh"] == 2
        else:
            assert ss["rh"] == 0
    print(f"new-format corpus: {n_events}/{len(spawns)} spawns derived as 'events'")


def test_cpu_cuda_parity_true_spawn_state():
    spawns = _load_new_spawns()
    if spawns is None:
        print(f"SKIP: {NEW_EVENTS_JSONL} not found")
        return
    try:
        solve_cuda = make_batch_planner("cuda")
    except Exception as exc:  # no GPU / driver: skip, the CPU path is exact
        print(f"SKIP: cuda planner unavailable ({exc})")
        return
    solve_cpu = make_batch_planner("cpu")

    # ~50 real moves, biased toward non-neutral spawn states.
    picked = [sp for sp in spawns if int(sp["held"]) or int(sp["hv"])][:40]
    picked += [sp for sp in spawns if not (int(sp["held"]) or int(sp["hv"]))][:10]
    n = len(picked)
    cols = np.zeros((n, 8), dtype=np.uint16)
    par = np.zeros(n, dtype=np.uint8)
    thr = np.zeros(n, dtype=np.uint8)
    sc = np.zeros(n, dtype=np.uint8)
    hv = np.zeros(n, dtype=np.uint8)
    hd = np.zeros(n, dtype=np.uint8)
    rh = np.zeros(n, dtype=np.uint8)
    for j, sp in enumerate(picked):
        cols[j] = occupancy_cols(decode_field(sp["field"]))
        thr[j] = compute_speed_threshold(int(sp["spd"]), int(sp["spdups"]))
        ss = derive_spawn_state(sp)
        assert ss["src"] == "events"
        par[j], sc[j], hv[j], hd[j], rh[j] = (
            ss["parity"], ss["sc"], ss["hv"], ss["hd"], ss["rh"])
    c_cpu = solve_cpu(cols, par, thr, sc=sc, hv=hv, hd=hd, rh=rh)
    c_gpu = solve_cuda(cols, par, thr, sc=sc, hv=hv, hd=hd, rh=rh)
    assert np.array_equal(c_cpu, c_gpu), (
        f"CPU/CUDA mismatch on {int((c_cpu != c_gpu).any(axis=1).sum())}/{n} moves")
    nonneutral = int(((hd != 0) | (rh != 0) | (hv != 0) | (sc != 0)).sum())
    print(f"cpu/cuda parity: {n} real moves identical "
          f"({nonneutral} with non-neutral spawn state)")


def main() -> None:
    for fn in (test_held_to_hd_rh, test_derive_spawn_state_old_format,
               test_derive_spawn_state_new_format, test_new_format_corpus,
               test_cpu_cuda_parity_true_spawn_state):
        fn()
        print(f"PASS {fn.__name__}")
    print("all spawn-state tests passed")


if __name__ == "__main__":
    main()
