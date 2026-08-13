"""Bit-exact parity tests: CUDA planner vs CPU drm_reach_bfs_v4.

Stages are tested as they land:
  phase12 : wanted set, greedy UB table, gd fields (vs drm_reach_v4_debug_tables)
  costs   : out_costs[512] (vs drm_reach_bfs_v4)  [once the BFS stage lands]

Board sources: real corpus spawn fields (from fightcadeRatings blobs) and
random fuzz boards; spawn states cover the annotation shape plus randomized
mid-fall states and the full threshold range.

Usage:
  .venv/bin/python -m tools.test_reach_cuda_parity [--quarks N] [--fuzz N] [--seed S]
"""

from __future__ import annotations

import argparse
import ctypes as C
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FCR_ROOT = REPO_ROOT.parent / "fightcadeRatings"

from drmc_rl.envs.backends.drmario_pool import default_library_path
from drmc_rl.planning.fast_reach import compute_speed_threshold
from tools.annotate_replay_events import (
    decode_field,
    occupancy_cols,
    pair_moves,
    parse_quark_events,
)


def load_cpu_debug():
    lib = C.CDLL(str(default_library_path()))
    fn = lib.drm_reach_v4_debug_tables
    fn.restype = C.c_int
    fn.argtypes = [C.POINTER(C.c_uint16)] + [C.c_int] * 9 + [
        C.POINTER(C.c_uint8), C.POINTER(C.c_uint16), C.POINTER(C.c_uint8), C.c_int]
    return fn


def load_cpu_v4():
    lib = C.CDLL(str(default_library_path()))
    fn = lib.drm_reach_bfs_v4
    fn.restype = C.c_int
    fn.argtypes = [C.POINTER(C.c_uint16)] + [C.c_int] * 10 + [C.POINTER(C.c_uint16)]
    return fn


def corpus_boards(n_quarks: int, rng) -> list[tuple[np.ndarray, int, int]]:
    """(cols, parity, thr) from real spawn fields across random quarks."""
    db = sqlite3.connect(FCR_ROOT / "data" / "drmario.sqlite")
    rows = db.execute(
        "SELECT quarkid, sha256 FROM processed_replay WHERE sha256 IS NOT NULL"
    ).fetchall()
    picks = [rows[i] for i in rng.choice(len(rows), size=n_quarks, replace=False)]
    out = []
    for qid, sha in picks:
        blob = FCR_ROOT / "data" / "blobs" / sha
        if not blob.exists():
            continue
        ev = parse_quark_events(blob.read_bytes())
        moves = pair_moves(ev, {"spawn_without_lock": 0, "lock_without_spawn": 0})
        for mv in moves:
            sp = mv["spawn"]
            cols = occupancy_cols(decode_field(sp["field"]))
            thr = compute_speed_threshold(int(sp["spd"]), int(sp["spdups"]))
            out.append((cols, int(sp["f"]) & 1, thr))
    return out


def fuzz_cases(n: int, rng) -> list[dict]:
    """Random boards + randomized full spawn state, incl. degenerate ones."""
    cases = []
    for _ in range(n):
        density = rng.uniform(0.0, 0.75)
        occ = rng.random((16, 8)) < density
        # keep the spawn row often free so many cases are non-trivial
        if rng.random() < 0.8:
            occ[0, :] = False
        cols = np.zeros(8, dtype=np.uint16)
        for r in range(16):
            cols |= occ[r].astype(np.uint16) << r
        cases.append(dict(
            cols=cols,
            sx=int(rng.integers(0, 8)) if rng.random() < 0.3 else 3,
            sy=int(rng.integers(0, 16)) if rng.random() < 0.3 else 0,
            srot=int(rng.integers(0, 4)),
            sc=int(rng.integers(0, 128)),
            hv=int(rng.integers(0, 16)),
            hd=int(rng.integers(0, 3)),
            parity=int(rng.integers(0, 2)),
            rh=int(rng.integers(0, 3)),
            thr=int(rng.choice([0, 1, 2, 5, 11, 17, 25, 39, 69, 127])),
        ))
    return cases


def check_phase12(ctx, cases: list[dict], label: str, gd_cap: int = 128) -> int:
    cpu_dbg = load_cpu_debug()
    cols = np.stack([c["cols"] for c in cases])
    kw = {}
    for f in ("sx", "sy", "srot", "sc", "hv", "hd", "rh"):
        kw[f] = np.array([c.get(f, {"sx": 3}.get(f, 0)) for c in cases], dtype=np.uint8)
    parity = np.array([c["parity"] for c in cases], dtype=np.uint8)
    thr = np.array([c["thr"] for c in cases], dtype=np.uint8)

    g_wanted, g_ub, g_gd, g_nw = ctx.debug_phase12(cols, parity, thr, gd_cap=gd_cap, **kw)

    c_wanted = np.zeros(512, dtype=np.uint8)
    c_ub = np.zeros(512, dtype=np.uint16)
    c_gd = np.zeros((512, 512), dtype=np.uint8)
    bad = 0
    for i, c in enumerate(cases):
        nw = cpu_dbg(
            c["cols"].ctypes.data_as(C.POINTER(C.c_uint16)),
            int(kw["sx"][i]), int(kw["sy"][i]), int(kw["srot"][i]),
            int(kw["sc"][i]), int(kw["hv"][i]), int(kw["hd"][i]),
            int(parity[i]), int(kw["rh"][i]), int(thr[i]),
            c_wanted.ctypes.data_as(C.POINTER(C.c_uint8)),
            c_ub.ctypes.data_as(C.POINTER(C.c_uint16)),
            c_gd.ctypes.data_as(C.POINTER(C.c_uint8)), 512)
        ok = (nw == g_nw[i]
              and (c_wanted == g_wanted[i]).all()
              and (c_ub == g_ub[i]).all()
              and (c_gd[:min(nw, gd_cap)] == g_gd[i, :min(nw, gd_cap)]).all())
        if not ok:
            bad += 1
            if bad <= 3:
                dw = int((c_wanted != g_wanted[i]).sum())
                du = int((c_ub != g_ub[i]).sum())
                dg = int((c_gd[:min(nw, gd_cap)] != g_gd[i, :min(nw, gd_cap)]).sum())
                print(f"  MISMATCH case {i}: nw cpu={nw} gpu={g_nw[i]} "
                      f"wanted!={dw} ub!={du} gd!={dg}")
    print(f"[phase12/{label}] {len(cases) - bad}/{len(cases)} exact")
    return bad


def check_costs(ctx, cases: list[dict], label: str) -> int:
    """Bit-exact on every macro-legal pose (POSE_TO_ACTION >= 0) — the entire
    consumer-visible surface. Macro-illegal poses (vertical y==0, horizontal
    x==7; never mapped to actions) are don't-care: CPU v4's mid-depth early
    exit truncates their recording at an enumeration-order-dependent point
    that a parallel implementation cannot (and should not) reproduce.
    """
    from tools.annotate_replay_events import POSE_TO_ACTION
    legal = POSE_TO_ACTION >= 0
    cpu_v4 = load_cpu_v4()
    cols = np.stack([c["cols"] for c in cases])
    kw = {}
    for f in ("sx", "sy", "srot", "sc", "hv", "hd", "rh"):
        kw[f] = np.array([c.get(f, 3 if f == "sx" else 0) for c in cases], dtype=np.uint8)
    parity = np.array([c["parity"] for c in cases], dtype=np.uint8)
    thr = np.array([c["thr"] for c in cases], dtype=np.uint8)

    g_costs = ctx.solve_costs(cols, parity, thr, **kw)

    c_costs = np.zeros(512, dtype=np.uint16)
    bad = 0
    for i, c in enumerate(cases):
        c_costs.fill(0xFFFF)
        cpu_v4(
            c["cols"].ctypes.data_as(C.POINTER(C.c_uint16)),
            int(kw["sx"][i]), int(kw["sy"][i]), int(kw["srot"][i]),
            int(kw["sc"][i]), int(kw["hv"][i]), int(kw["hd"][i]),
            int(parity[i]), int(kw["rh"][i]), int(thr[i]), 2048,
            c_costs.ctypes.data_as(C.POINTER(C.c_uint16)))
        if not (c_costs[legal] == g_costs[i][legal]).all():
            bad += 1
            if bad <= 5:
                d = np.flatnonzero((c_costs != g_costs[i]) & legal)
                print(f"  COST MISMATCH case {i} (thr={thr[i]} p={parity[i]}): "
                      f"{len(d)} poses, first {[(int(p), int(c_costs[p]), int(g_costs[i][p])) for p in d[:4]]}")
    print(f"[costs/{label}] {len(cases) - bad}/{len(cases)} exact")
    return bad


def _py_v4_step(fm_h, fm_v, thr, s, act):
    """Python mirror of the C v4_step (exact single-state stepper) for
    independent script replay verification."""
    dir_, dn, rot = act // 6, (act % 6) // 3, act % 3
    def fits(x, y, r):
        if not (0 <= x < 8 and 0 <= y < 16):
            return False
        m = fm_v[y] if (r & 1) else fm_h[y]
        return bool((m >> x) & 1)
    press = (dir_ == 1 and s["hd"] != 1) or (dir_ == 2 and s["hd"] != 2)
    down_only = dn and dir_ == 0
    drop = False
    if (s["p"] & 1) == 0 and down_only:
        drop = True; s["sc"] = 0
    else:
        s["sc"] += 1
        if s["sc"] > thr:
            drop = True; s["sc"] = 0
    if drop:
        ny = s["y"] + 1
        if ny >= 16 or not fits(s["x"], ny, s["rot"]):
            s["locked"] = True; s["hd"] = dir_; s["rh"] = rot; s["p"] ^= 1
            return
        s["y"] = ny
    allow = False
    if press:
        s["hv"] = 0; allow = True
    elif dir_ != 0:
        s["hv"] += 1
        if s["hv"] >= 0x10:
            s["hv"] = 0x0A; allow = True
    if allow and dir_ != 0:
        nx = s["x"] + (1 if dir_ == 2 else -1)
        if fits(nx, s["y"], s["rot"]):
            s["x"] = nx
        else:
            s["hv"] = 0x0F
    if rot != 0 and rot != s["rh"]:
        r0 = s["rot"] & 3
        r1 = (r0 - 1) & 3 if rot == 1 else (r0 + 1) & 3
        if (r1 & 1) == 0:
            if fits(s["x"], s["y"], r1):
                if dir_ == 1 and fits(s["x"] - 1, s["y"], r1):
                    s["x"] -= 1
                s["rot"] = r1
            elif fits(s["x"] - 1, s["y"], r1):
                s["x"] -= 1; s["rot"] = r1
        else:
            if fits(s["x"], s["y"], r1):
                s["rot"] = r1
    s["hd"] = dir_; s["rh"] = rot; s["p"] ^= 1


def _fit_masks(cols):
    occ = np.zeros(16, dtype=np.uint16)
    for x in range(8):
        for y in range(16):
            if cols[x] & (1 << y):
                occ[y] |= 1 << x
    empty = (~occ) & 0xFF
    fm_h = [(empty[y] & (empty[y] >> 1)) & 0xFF for y in range(16)]
    fm_v = [empty[0]] + [(empty[y] & empty[y - 1]) & 0xFF for y in range(1, 16)]
    return fm_h, fm_v


def check_scripts(ctx, cases: list[dict], label: str) -> int:
    """Every emitted script must replay to a lock at its pose in exactly
    cost frames (independent Python stepper). Also reports the GPU greedy
    match rate (status==0 fraction)."""
    from tools.annotate_replay_events import POSE_TO_ACTION
    legal = POSE_TO_ACTION >= 0
    cols = np.stack([c["cols"] for c in cases])
    parity = np.array([c["parity"] for c in cases], dtype=np.uint8)
    thr = np.array([c["thr"] for c in cases], dtype=np.uint8)
    kw = {}
    for f in ("sx", "sy", "srot", "sc", "hv", "hd", "rh"):
        kw[f] = np.array([c.get(f, 3 if f == "sx" else 0) for c in cases], dtype=np.uint8)

    costs, off, length, scr, st = ctx.solve_scripts(cols, parity, thr, **kw)
    bad = 0
    n_replayed = 0
    for i, c in enumerate(cases):
        if st[i] & 4:
            print(f"  PARITY ALARM case {i}: greedy beat exact cost")
            bad += 1
            continue
        fm_h, fm_v = _fit_masks(c["cols"])
        t = int(thr[i])
        for pose in np.flatnonzero((length[i] > 0) & legal):
            script = scr[i, off[i, pose]: off[i, pose] + length[i, pose]]
            s = dict(x=int(kw["sx"][i]), y=int(kw["sy"][i]),
                     rot=int(kw["srot"][i]), sc=min(int(kw["sc"][i]), t),
                     hv=int(kw["hv"][i]) & 0x0F, hd=int(kw["hd"][i]),
                     p=int(parity[i]), rh=int(kw["rh"][i]), locked=False)
            for act in script:
                if s["locked"]:
                    break
                _py_v4_step(fm_h, fm_v, min(t, 127), s, int(act))
            ok = (s["locked"] and len(script) == costs[i, pose]
                  and (s["rot"] & 3) * 128 + s["y"] * 8 + s["x"] == pose)
            n_replayed += 1
            if not ok:
                bad += 1
                if bad <= 3:
                    print(f"  SCRIPT BAD case {i} pose {pose}: locked={s['locked']} "
                          f"end=({s['x']},{s['y']},{s['rot'] & 3}) len={len(script)} "
                          f"cost={costs[i, pose]}")
    matched = float((st == 0).mean())
    print(f"[scripts/{label}] {n_replayed} scripts replayed, {bad} bad; "
          f"greedy-matched instances: {matched:.4f}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarks", type=int, default=3)
    ap.add_argument("--fuzz", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    from drmc_rl.planning.cuda import CudaReach
    ctx = CudaReach(max_batch=16384)

    bad = 0
    boards = corpus_boards(args.quarks, rng)
    corpus_cases = [dict(cols=c, parity=p, thr=t) for c, p, t in boards]
    print(f"corpus cases: {len(corpus_cases)}")
    bad += check_phase12(ctx, corpus_cases, "corpus")
    bad += check_costs(ctx, corpus_cases, "corpus")
    bad += check_scripts(ctx, corpus_cases[:600], "corpus")

    fz = fuzz_cases(args.fuzz, rng)
    bad += check_phase12(ctx, fz, "fuzz")
    bad += check_costs(ctx, fz, "fuzz")
    bad += check_scripts(ctx, fz, "fuzz")

    ctx.close()
    if bad:
        print(f"FAIL: {bad} mismatching cases")
        return 1
    print("OK: all stages bit-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
