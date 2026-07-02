"""Annotate fightcadeRatings v2 replay events with planner/policy move metrics.

Cross-project deliverable agreed in ../fightcadeRatings/COORDINATION.md
("Ask from fightcadeRatings: annotation schema for placement playback" +
"drmc-rl: schema accepted"). Reads event blobs from their sqlite store,
computes per-move annotations, and writes them back to *their* store
(table `drmc_annotation` + per-quark JSONL blob via `store.put_blob`).
The boundary stays data-only: no code dependency in either direction
beyond importing their `store` module for connect/put_blob.

Join key: (quarkid, p, spawn_f). One row per spawn->lock pair.

Fields (per the accepted schema):
  opt_frames_chosen  exact minimal input frames to reach the chosen lock pose
                     (drm_reach_bfs_v4; planner-exact). NULL if infeasible.
  opt_frames_best    min cost over all feasible placements for that spawn.
  rank               1 + number of feasible candidates whose policy logit is
                     strictly greater than the chosen placement's logit.
  n_options          number of feasible macro placements (distinct actions).
  value_gap          best_logit - chosen_logit, in champion *policy-logit*
                     units (dimensionless; chosen==best -> 0.0). Comparable
                     only within one model_ver. NOTE: the accepted schema said
                     "value head"; we rank with the candidate policy logits
                     instead because that is the quantity the policy actually
                     argmaxes over -- documented here so consumers know.
  tuck               heuristic: 1 if some occupied cell sits strictly above
                     one of the chosen pose's cells (i.e. the pose is not
                     reachable by a straight vertical drop). Not an exact
                     maneuver classifier.
  kick               always 0 in phase 1 (not derived yet; reserved).
  feasible           1 if the observed lock pose has a finite planner cost
                     from the spawn snapshot (desync canary when 0).
  interrupted        1 if a garbage volley landed on this player's board
                     strictly between spawn_f and lock_f (static-board fields
                     are still emitted; reachability may be off for these).
  planner_ver        'v4' (drm_reach_bfs_v4).
  model_ver          champion checkpoint filename.
  blob_sha           sha256 of this quark's annotation JSONL blob.

Lock-record verification/repair: drm_replay sometimes samples the lock
event's x/y/rot one frame or pipeline stage off the actual placement (see
fightcadeRatings/docs/drm-replay-lock-record-issues.md). Every move's pose is
therefore checked against the next same-player spawn-field diff (authoritative
when clean) and repaired when the event disagrees; per-move provenance lands
in the blob records as `lock_verified` (1 verified / 0 repaired / null
underivable) + `lock_rec` (the original event coords). The sqlite schema is
unchanged. Standalone corpus scan: tools/audit_lock_records.py.

Known approximations (documented, deliberate for phase 1):
  - OLD-format blobs only (records without the fc/held/hv/scnt spawn-state
    keys, i.e. the pre-upgrade corpus): the planner spawn state assumes a
    neutral controller at spawn (hold_dir=0, rot_hold=0, hv=0, sc=0) and
    parity = spawn_f & 1 from the harness frame index, not NES $0043 (a
    constant parity flip costs at most ~1 frame on soft-drop timing).
    NEW-format blobs carry the exact NES state at end of the spawn frame
    (fc = frameCounter $0043, held = buttons-held byte, hv = horVelocity,
    scnt = speedCounter) and the planner is seeded with the true state; see
    derive_spawn_state(). Per-move provenance lands in the blob records as
    `spawn_src` ("events" / "assumed"); --spawn-state assumed forces the old
    behavior everywhere (reproducibility escape hatch).
  - Collision occupancy follows the engine (DrMarioPool::build_cols_u16):
    occupied = tile != 0x00 and tile < 0xF0 (mid-clear $Bx counts occupied).
  - aux_v1 inputs mirror training semantics where derivable from the replay
    (level/speed from init/spawn records, frames_used = spawn_f - game init
    frame, viruses from field snapshots); task budget fields use the
    no-max-frames tanh fallback, same as budget-less training envs.

Usage (single-threaded on purpose -- a training run shares this box):
  nice -n 19 .venv/bin/python -m tools.annotate_replay_events [--limit N]
      [--requark QUARKID] [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import ctypes as C
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FCR_ROOT = REPO_ROOT.parent / "fightcadeRatings"

from envs.backends.drmario_pool import default_library_path
from envs.retro.fast_reach import compute_speed_threshold
from envs.specs.ram_to_state import COLOR_VALUE_TO_INDEX
from models.policy.candidate_packing import pack_feasible_candidates
from tools.eval_policy import _build_net_from_cfg, _make_aux_builder
from training.utils.checkpoint_io import load_checkpoint

PLANNER_VER = "v4"
CHECKPOINT = REPO_ROOT / "runs" / "best_agents" / "smdp_ppo_step535164979.pt.gz"
GRID_W, GRID_H = 8, 16
OBS_CHANNELS = 12  # bitplane_bottle_conn_mask
BATCH = 256

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS drmc_annotation (
    quarkid TEXT, p INTEGER, spawn_f INTEGER,
    opt_frames_chosen INTEGER, opt_frames_best INTEGER,
    rank INTEGER, n_options INTEGER, value_gap REAL,
    tuck INTEGER, kick INTEGER, feasible INTEGER, interrupted INTEGER,
    planner_ver TEXT, model_ver TEXT, blob_sha TEXT, annotated_at REAL,
    PRIMARY KEY (quarkid, p, spawn_f)
)
"""

# Pose geometry (DrMarioPool.cpp ROT_OFFSETS / orient_index). Pose base =
# bottom-left of the 2x2 box, y=0 top. Macro action = o*128 + row*8 + col.
ROT_OFFSETS = ((0, 0, 0, 1), (0, 0, -1, 0), (0, 1, 0, 0), (-1, 0, 0, 0))
ORIENT_INDEX = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}


def _pose_to_action_map() -> np.ndarray:
    """Map pose idx (rot*128 + y*8 + x) -> macro action idx, -1 if oob."""
    out = np.full(512, -1, dtype=np.int32)
    for rot in range(4):
        dr1, dc1, dr2, dc2 = ROT_OFFSETS[rot]
        for y in range(GRID_H):
            for x in range(GRID_W):
                r1, c1 = y + dr1, x + dc1
                r2, c2 = y + dr2, x + dc2
                if not (0 <= r1 < GRID_H and 0 <= c1 < GRID_W):
                    continue
                if not (0 <= r2 < GRID_H and 0 <= c2 < GRID_W):
                    continue
                o = ORIENT_INDEX.get((r2 - r1, c2 - c1))
                if o is None:
                    continue
                out[rot * 128 + y * 8 + x] = o * 128 + r1 * 8 + c1
    return out


POSE_TO_ACTION = _pose_to_action_map()

# NES controller bit layout, verified against the disassembly
# (dr-mario-disassembly/defines/drmario_constants.asm:145-152):
#   btn_right=$01 btn_left=$02 btn_down=$04 btn_up=$08
#   btn_start=$10 btn_select=$20 btn_b=$40 btn_a=$80
BTN_RIGHT, BTN_LEFT, BTN_DOWN = 0x01, 0x02, 0x04
BTN_B, BTN_A = 0x40, 0x80

# v2.1 spawn records carry the exact NES state at end of the spawn frame.
SPAWN_STATE_KEYS = ("fc", "held", "hv", "scnt")


def held_to_hd_rh(held: int) -> tuple:
    """Buttons-held byte -> planner (hold_dir, rot_hold).

    hold_dir: 0 neutral, 1 left, 2 right (fast_reach.HoldDir values).
    rot_hold: 0 none, 1 A/CW, 2 B/CCW (fast_reach.Rotation values).

    Edge cases (deliberate rules):
      - L+R both held: impossible on a real NES pad, representable in replay
        bytes. Resolve to RIGHT, mirroring fallingPill_checkXMove, which
        checks btn_right before btn_left once a move fires
        (drmario_prg_game_logic.asm @checkInput_right, ~line 311) -- the
        game's own held-repeat would move right.
      - A+B both held: fallingPill_checkRotate applies A then B independently
        on the same frame (drmario_prg_game_logic.asm lines 368-382, "if both
        A and B are pressed ... both will be taken into account"), so neither
        wins in-game and rot_hold has no "both" value. Pick A (checked first);
        worst case the planner models a same-frame B edge as available when
        it is not (<= 1 rotation-retry frame of cost error).
      - btn_down held maps to NO spawn-state dimension: the BFS chooses every
        future frame's inputs freely (including down), so a currently-held
        down button carries no state. Ignored on purpose.
    """
    held &= 0xFF
    if held & BTN_RIGHT:
        hd = 2
    elif held & BTN_LEFT:
        hd = 1
    else:
        hd = 0
    if held & BTN_A:
        rh = 1
    elif held & BTN_B:
        rh = 2
    else:
        rh = 0
    return hd, rh


def derive_spawn_state(sp: Dict[str, Any], mode: str = "events") -> Dict[str, Any]:
    """Planner spawn state for one spawn record.

    Returns {parity, sc, hv, hd, rh, src}. src == "events" when mode is
    "events" AND the record carries the v2.1 spawn-state keys
    (fc/held/hv/scnt); otherwise the phase-1 assumptions (neutral controller,
    parity from the harness frame index) with src == "assumed".
    """
    if mode == "events" and all(k in sp for k in SPAWN_STATE_KEYS):
        hd, rh = held_to_hd_rh(int(sp["held"]))
        return {
            "parity": int(sp["fc"]) & 1,   # NES $0043 bit0 (exact soft-drop gate)
            "sc": int(sp["scnt"]) & 0xFF,  # planner clamps to speed threshold
            # planner state stores 4 bits; drm_reach_full.c sanitizes with
            # hor_velocity &= 0x0F ("state space stores 4 bits in the Python
            # packer") -- match that here so both backends see the same value.
            "hv": int(sp["hv"]) & 0x0F,
            "hd": hd, "rh": rh, "src": "events",
        }
    return {"parity": int(sp["f"]) & 1, "sc": 0, "hv": 0, "hd": 0, "rh": 0,
            "src": "assumed"}


def load_planner():
    lib = C.CDLL(str(default_library_path()))
    fn = lib.drm_reach_bfs_v4
    fn.restype = C.c_int
    fn.argtypes = [C.POINTER(C.c_uint16)] + [C.c_int] * 10 + [C.POINTER(C.c_uint16)]
    return fn


def make_batch_planner(backend: str):
    """Batched planner: solve(cols (n,8)u16, parity (n,)u8, thr (n,)u8,
    sc=None, hv=None, hd=None, rh=None) -> (n,512)u16.

    The optional kwargs are per-move (n,)u8 spawn-state arrays (speed_counter,
    hor_velocity, hold_dir, rot_hold, from derive_spawn_state); None means
    all-zero, i.e. the phase-1 neutral assumption. The spawn pose is fixed at
    (x=3, y=0, rot=0) for annotation.

    'cuda' runs the bit-exact GPU port (reach_cuda; same costs as CPU v4 on
    every macro-legal pose and arbitrary spawn states, verified by
    tools/test_reach_cuda_parity).
    """
    if backend == "cuda":
        from reach_cuda import CudaReach
        ctx = CudaReach(max_batch=8192)

        def solve_cuda(cols, parity, thr, sc=None, hv=None, hd=None, rh=None):
            n = len(cols)
            zeros = np.zeros(n, dtype=np.uint8)
            sc_, hv_, hd_, rh_ = (zeros if a is None else np.asarray(a, dtype=np.uint8)
                                  for a in (sc, hv, hd, rh))
            out = np.empty((n, 512), dtype=np.uint16)
            for s in range(0, n, ctx.max_batch):
                e = min(s + ctx.max_batch, n)
                out[s:e] = ctx.solve_costs(cols[s:e], parity[s:e], thr[s:e],
                                           sc=sc_[s:e], hv=hv_[s:e],
                                           hd=hd_[s:e], rh=rh_[s:e])
            return out

        return solve_cuda

    fn = load_planner()

    def solve_cpu(cols, parity, thr, sc=None, hv=None, hd=None, rh=None):
        out = np.empty((len(cols), 512), dtype=np.uint16)
        buf = np.empty(512, dtype=np.uint16)
        for i in range(len(cols)):
            row = np.ascontiguousarray(cols[i])
            rc = fn(row.ctypes.data_as(C.POINTER(C.c_uint16)),
                    3, 0, 0,
                    0 if sc is None else int(sc[i]),
                    0 if hv is None else int(hv[i]),
                    0 if hd is None else int(hd[i]),
                    int(parity[i]),
                    0 if rh is None else int(rh[i]),
                    int(thr[i]), 2048,
                    buf.ctypes.data_as(C.POINTER(C.c_uint16)))
            if rc != 0:
                raise RuntimeError(f"drm_reach_bfs_v4 rc={rc}")
            out[i] = buf
        return out

    return solve_cpu


def decode_field(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(GRID_H, GRID_W)


def occupancy_cols(field: np.ndarray) -> np.ndarray:
    """uint16[8] column masks, bit y = occupied (engine collision semantics)."""
    occ = (field != 0x00) & (field < 0xF0)
    cols = np.zeros(8, dtype=np.uint16)
    for r in range(GRID_H):
        cols |= occ[r].astype(np.uint16) << r
    return cols


def canonical_color(raw: int) -> int:
    return int(COLOR_VALUE_TO_INDEX.get(int(raw) & 0x03, 0))


def build_obs(field: np.ndarray, feasible_512: np.ndarray) -> np.ndarray:
    """12-channel bitplane_bottle_conn_mask obs, mirrors DrMarioPool::build_obs."""
    obs = np.zeros((OBS_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    type_hi = field & 0xF0
    color_lo = field & 0x03
    is_empty = field == 0xFF
    is_zero = field == 0x00
    is_just_emptied = (type_hi == 0xF0) & ~is_empty
    is_clearing = (type_hi == 0xB0) | is_just_emptied
    color_valid = ~(is_empty | is_zero | is_clearing)
    for raw_value, ch in ((1, 0), (0, 1), (2, 2)):  # red, yellow, blue
        obs[ch] = ((color_lo == raw_value) & color_valid).astype(np.float32)
    obs[3] = (type_hi == 0xD0).astype(np.float32)  # virus
    for code, ch in ((0x50, 4), (0x40, 5), (0x70, 6), (0x60, 7)):
        # connected_up / down / left / right
        obs[ch] = (type_hi == code).astype(np.float32)
    feas = feasible_512.reshape(4, GRID_H, GRID_W)
    obs[8:12] = feas.astype(np.float32)
    return obs


def parse_quark_events(raw: bytes) -> Dict[str, list]:
    """Split a JSONL event blob into init/spawn/lock/grb records (in order)."""
    inits, spawns, locks, grbs = [], [], [], []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if '"t":"spawn"' in line:
            spawns.append(json.loads(line))
        elif '"t":"lock"' in line:
            locks.append(json.loads(line))
        elif '"t":"init"' in line:
            inits.append(json.loads(line))
        elif '"grb"' in line:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "grb" in ev and "f" in ev:
                grbs.append(ev)
    return {"init": inits, "spawn": spawns, "lock": locks, "grb": grbs}


PILL_HALF_TYPES = {0x40, 0x50, 0x60, 0x70}  # connected pill-half tile codes


def derive_lock_poses(events: Dict[str, list], moves: List[Dict[str, Any]],
                      counters: Dict[str, int]) -> List[Optional[Dict[str, Any]]]:
    """Field-diff ground truth for each move's lock pose.

    The next same-player spawn field is authoritative: when nothing vanished
    (no clear resolved) and exactly two cells were added forming a domino of
    pill-half tiles whose colors match the spawn's `pill` record, those cells
    ARE the placement — independent of the lock event's x/y/rot, which
    drm_replay sometimes samples one frame/stage off (see
    fightcadeRatings/docs/drm-replay-lock-record-issues.md).

    Returns a list aligned with `moves`: None when underivable (clear/garbage
    in the window, game end, non-domino diff), else
    {"x", "y_top", "rot_choices": (rot,) or (rotA, rotB) when the pill is
    monochrome and color order cannot disambiguate}.
    """
    init_fs = sorted(int(e["f"]) for e in events["init"])
    spawns_by_p: Dict[int, List[tuple]] = {1: [], 2: []}
    for e in events["spawn"]:
        spawns_by_p[int(e["p"])].append((int(e["f"]), e))
    for p in spawns_by_p:
        spawns_by_p[p].sort(key=lambda t: t[0])

    import bisect
    out: List[Optional[Dict[str, Any]]] = []
    for mv in moves:
        sp, lk, p = mv["spawn"], mv["lock"], mv["p"]
        lock_f = int(lk["f"])
        sp_list = spawns_by_p[p]
        k = bisect.bisect_right([t[0] for t in sp_list], lock_f)
        if k >= len(sp_list):
            counters["fielddiff_no_next_spawn"] += 1
            out.append(None)
            continue
        next_f, next_sp = sp_list[k]
        j = bisect.bisect_right(init_fs, lock_f)
        if j < len(init_fs) and init_fs[j] <= next_f:
            counters["fielddiff_game_boundary"] += 1
            out.append(None)
            continue
        f0 = decode_field(sp["field"])
        f1 = decode_field(next_sp["field"])
        occ0 = (f0 != 0x00) & (f0 < 0xF0)
        occ1 = (f1 != 0x00) & (f1 < 0xF0)
        if (occ0 & ~occ1).any():
            counters["fielddiff_cells_vanished"] += 1
            out.append(None)
            continue
        added = np.argwhere(~occ0 & occ1)
        if len(added) != 2:
            counters["fielddiff_not_two_cells"] += 1
            out.append(None)
            continue
        (r1, c1), (r2, c2) = sorted(map(tuple, added.tolist()))
        t1, t2 = int(f1[r1, c1]) & 0xF0, int(f1[r2, c2]) & 0xF0
        if t1 not in PILL_HALF_TYPES or t2 not in PILL_HALF_TYPES:
            counters["fielddiff_not_pill_tiles"] += 1
            out.append(None)
            continue
        col1, col2 = int(f1[r1, c1]) & 0x03, int(f1[r2, c2]) & 0x03
        pill = [int(sp["pill"][0]) & 0x03, int(sp["pill"][1]) & 0x03]
        if r1 == r2 and c2 == c1 + 1:
            # horizontal; ROT_OFFSETS: rot0 half1 = left cell, rot2 = swapped
            if pill[0] == pill[1]:
                rots = (0, 2)
            elif (col1, col2) == (pill[0], pill[1]):
                rots = (0,)
            elif (col1, col2) == (pill[1], pill[0]):
                rots = (2,)
            else:
                counters["fielddiff_color_mismatch"] += 1
                out.append(None)
                continue
            out.append({"x": c1, "y_top": r1, "rot_choices": rots})
        elif c1 == c2 and r2 == r1 + 1:
            # vertical; base = bottom cell. rot1 half1 = bottom, rot3 = top.
            if pill[0] == pill[1]:
                rots = (1, 3)
            elif (col2, col1) == (pill[0], pill[1]):   # bottom, top
                rots = (1,)
            elif (col1, col2) == (pill[0], pill[1]):
                rots = (3,)
            else:
                counters["fielddiff_color_mismatch"] += 1
                out.append(None)
                continue
            out.append({"x": c1, "y_top": r2, "rot_choices": rots})
        else:
            counters["fielddiff_not_domino"] += 1
            out.append(None)
    return out


def resolve_lock_pose(lk: Dict[str, Any], d: Optional[Dict[str, Any]],
                      out_costs: np.ndarray,
                      counters: Dict[str, int]) -> tuple:
    """Chosen pose from a lock record, verified/repaired against the
    field-diff ground truth `d` (one entry of derive_lock_poses; drm_replay
    sometimes samples lock coords one frame/stage off — see
    fightcadeRatings/docs/drm-replay-lock-record-issues.md).

    Returns (x, y_top, rot, lock_verified, (rec_x, rec_y_top, rec_rot)) where
    lock_verified is 1 = record matched the field diff, 0 = repaired from the
    field diff, None = underivable (record used as-is).
    """
    rec_x, rec_rot = int(lk["x"]), int(lk["rot"]) & 3
    rec_y_top = (GRID_H - 1) - int(lk["y"])
    x, y_top, rot = rec_x, rec_y_top, rec_rot
    lock_verified: Optional[int] = None
    if d is not None:
        rots = d["rot_choices"]
        if len(rots) > 1:
            # monochrome pill: color order can't disambiguate; prefer the
            # recorded rot, else the planner-feasible one.
            if rec_rot in rots and (rec_x, rec_y_top) == (d["x"], d["y_top"]):
                use_rot = rec_rot
            else:
                use_rot = rots[0]
                for rr in rots:
                    if out_costs[rr * 128 + d["y_top"] * 8 + d["x"]] != 0xFFFF:
                        use_rot = rr
                        break
        else:
            use_rot = rots[0]
        lock_verified = int((rec_x, rec_y_top) == (d["x"], d["y_top"])
                            and rec_rot in rots)
        x, y_top, rot = d["x"], d["y_top"], use_rot
        if not lock_verified:
            counters["lock_repaired"] += 1
    return x, y_top, rot, lock_verified, (rec_x, rec_y_top, rec_rot)


def pair_moves(events: Dict[str, list], counters: Dict[str, int]) -> List[Dict[str, Any]]:
    """Pair each spawn with the next same-player lock within the same game."""
    timeline: List[tuple] = []
    for kind in ("init", "spawn", "lock"):
        for ev in events[kind]:
            timeline.append((int(ev["f"]), {"init": 0, "spawn": 1, "lock": 2}[kind], ev))
    timeline.sort(key=lambda t: (t[0], t[1]))

    moves: List[Dict[str, Any]] = []
    pending: Dict[int, Optional[dict]] = {1: None, 2: None}
    game_ctx: Dict[int, dict] = {}
    for _f, kind, ev in timeline:
        if kind == 0:  # init: new game, drop unpaired spawns
            for p in (1, 2):
                if pending[p] is not None:
                    counters["spawn_without_lock"] += 1
                pending[p] = None
                fld = decode_field(ev["field1"] if p == 1 else ev["field2"])
                game_ctx[p] = {
                    "init_f": int(ev["f"]),
                    "level": int(ev["lvl"][p - 1]),
                    "viruses_initial": int(((fld & 0xF0) == 0xD0).sum()),
                }
        elif kind == 1:  # spawn
            p = int(ev["p"])
            if pending[p] is not None:
                counters["spawn_without_lock"] += 1
            pending[p] = ev
        else:  # lock
            p = int(ev["p"])
            sp = pending[p]
            if sp is None:
                counters["lock_without_spawn"] += 1
                continue
            pending[p] = None
            ctx = game_ctx.get(p, {"init_f": int(sp["f"]), "level": 0, "viruses_initial": 0})
            moves.append({"p": p, "spawn": sp, "lock": ev, "ctx": ctx})
    for p in (1, 2):
        if pending[p] is not None:
            counters["spawn_without_lock"] += 1
    return moves


def annotate_quark(quarkid: str, raw: bytes, planner, net, aux_shim,
                   candidate_max: int, counters: Dict[str, int],
                   device: str = "cpu",
                   spawn_mode: str = "events") -> List[Dict[str, Any]]:
    events = parse_quark_events(raw)
    if not events["spawn"]:
        counters["quarks_no_v2"] += 1
        return []
    moves = pair_moves(events, counters)
    grbs = events["grb"]

    recs: List[Dict[str, Any]] = []
    batch_items: List[dict] = []  # net inputs for rank/value_gap

    # Planner pre-pass: batch all spawns of the quark through one solver call
    # (the CUDA backend grinds them in parallel; CPU loops).
    n_mv = len(moves)
    work_cols = np.zeros((n_mv, 8), dtype=np.uint16)
    work_par = np.zeros(n_mv, dtype=np.uint8)
    work_thr = np.zeros(n_mv, dtype=np.uint8)
    work_sc = np.zeros(n_mv, dtype=np.uint8)
    work_hv = np.zeros(n_mv, dtype=np.uint8)
    work_hd = np.zeros(n_mv, dtype=np.uint8)
    work_rh = np.zeros(n_mv, dtype=np.uint8)
    spawn_srcs: List[str] = []
    fields: List[np.ndarray] = []
    for j, mv in enumerate(moves):
        sp = mv["spawn"]
        field = decode_field(sp["field"])
        fields.append(field)
        work_cols[j] = occupancy_cols(field)
        ss = derive_spawn_state(sp, spawn_mode)
        work_par[j] = ss["parity"]
        work_sc[j] = ss["sc"]
        work_hv[j] = ss["hv"]
        work_hd[j] = ss["hd"]
        work_rh[j] = ss["rh"]
        spawn_srcs.append(ss["src"])
        work_thr[j] = compute_speed_threshold(int(sp["spd"]), int(sp["spdups"]))
    all_costs = (planner(work_cols, work_par, work_thr,
                         sc=work_sc, hv=work_hv, hd=work_hd, rh=work_rh)
                 if n_mv else None)
    derived = derive_lock_poses(events, moves, counters)

    for j, mv in enumerate(moves):
        sp, lk, p = mv["spawn"], mv["lock"], mv["p"]
        spawn_f, lock_f = int(sp["f"]), int(lk["f"])
        field = fields[j]
        out_costs = all_costs[j]

        # Pose costs -> macro action costs/feasibility (as in ensure_planner).
        action_cost = np.full(512, 0xFFFF, dtype=np.uint16)
        finite = np.flatnonzero(out_costs != 0xFFFF)
        for pose in finite:
            a = POSE_TO_ACTION[pose]
            if a >= 0:
                action_cost[a] = out_costs[pose]
        feasible_512 = action_cost != 0xFFFF
        n_options = int(feasible_512.sum())
        opt_best = int(action_cost[feasible_512].min()) if n_options else None

        # Chosen pose: lock record, verified/repaired against the field-diff
        # ground truth.
        x, y_top, rot, lock_verified, (rec_x, rec_y_top, rec_rot) = \
            resolve_lock_pose(lk, derived[j], out_costs, counters)
        chosen_action = -1
        chosen_cost: Optional[int] = None
        if 0 <= x < GRID_W and 0 <= y_top < GRID_H:
            pose = rot * 128 + y_top * 8 + x
            chosen_action = int(POSE_TO_ACTION[pose])
            if chosen_action >= 0 and out_costs[pose] != 0xFFFF:
                chosen_cost = int(out_costs[pose])
        feasible = chosen_cost is not None

        interrupted = any(
            int(g["grb"]) == p and spawn_f < int(g["f"]) < lock_f for g in grbs
        )

        # Tuck heuristic: occupied cell strictly above any chosen pose cell.
        tuck: Optional[int] = None
        if 0 <= x < GRID_W and 0 <= y_top < GRID_H:
            dr1, dc1, dr2, dc2 = ROT_OFFSETS[rot]
            occ = (field != 0x00) & (field < 0xF0)
            tuck = 0
            for (cy, cx) in ((y_top + dr1, x + dc1), (y_top + dr2, x + dc2)):
                if 0 <= cy < GRID_H and 0 <= cx < GRID_W and occ[:cy, cx].any():
                    tuck = 1

        rec = {
            "p": p, "spawn_f": spawn_f, "lock_f": lock_f,
            "lock_x": x, "lock_y_top": y_top, "lock_rot": rot,
            # provenance: 1 = lock event matched field diff, 0 = repaired from
            # field diff, None = underivable (clear/garbage window, game end)
            "lock_verified": lock_verified,
            "lock_rec": [rec_x, rec_y_top, rec_rot],
            "opt_frames_chosen": chosen_cost,
            "opt_frames_best": opt_best,
            "slack": (lock_f - spawn_f - chosen_cost) if feasible else None,
            "rank": None, "n_options": n_options, "value_gap": None,
            "tuck": tuck, "kick": 0,
            "feasible": int(feasible), "interrupted": int(interrupted),
            # provenance: "events" = planner seeded with the exact NES spawn
            # state from the v2.1 record, "assumed" = phase-1 neutral state.
            "spawn_src": spawn_srcs[j],
        }
        recs.append(rec)
        counters["moves"] += 1
        counters["feasible"] += int(feasible)
        counters["interrupted"] += int(interrupted)
        if spawn_srcs[j] == "events":
            counters["spawn_state_events"] += 1
        if not feasible:
            continue

        # Defer policy scoring so we can batch the net forward.
        packed = pack_feasible_candidates(
            feasible_512.reshape(4, GRID_H, GRID_W),
            action_cost.reshape(4, GRID_H, GRID_W),
            max_candidates=candidate_max, sort_by_cost=True,
        )
        slot = np.flatnonzero(packed.actions[: packed.count] == chosen_action)
        if slot.size == 0:
            counters["chosen_not_packed"] += 1
            continue
        info = {
            "pill/speed_setting": int(sp["spd"]),
            "speed_setting": int(sp["spd"]),
            "level": mv["ctx"]["level"],
            "task_mode": "viruses",
            "task/frames_used": spawn_f - mv["ctx"]["init_f"],
            "drm/viruses_initial": mv["ctx"]["viruses_initial"],
            "viruses_remaining": int(((field & 0xF0) == 0xD0).sum()),
            "placements/options": n_options,
        }
        batch_items.append({
            "rec": rec,
            "obs": build_obs(field, feasible_512),
            "pill": [canonical_color(sp["pill"][0]), canonical_color(sp["pill"][1])],
            "prev": [canonical_color(sp["prev"][0]), canonical_color(sp["prev"][1])],
            "packed": packed,
            "slot": int(slot[0]),
            "info": info,
        })

    # Batched policy forward for rank / value_gap.
    for start in range(0, len(batch_items), BATCH):
        chunk = batch_items[start : start + BATCH]
        obs = np.stack([c["obs"] for c in chunk])
        pills = np.array([c["pill"] for c in chunk], dtype=np.int64)
        prevs = np.array([c["prev"] for c in chunk], dtype=np.int64)
        cand_a = np.stack([c["packed"].actions for c in chunk])
        cand_m = np.stack([c["packed"].mask for c in chunk])
        cand_c = np.stack([c["packed"].cost for c in chunk])
        aux = None
        if aux_shim is not None:
            aux = aux_shim._build_aux_batch(obs, [c["info"] for c in chunk])
        with torch.inference_mode():
            logits, _values = net(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(pills).to(device),
                torch.from_numpy(prevs).to(device),
                torch.from_numpy(cand_a.astype(np.int32)).to(device),
                torch.from_numpy(cand_c.astype(np.float32)).to(device),
                torch.from_numpy(cand_m).to(device),
                aux=None if aux is None else torch.from_numpy(aux).to(device),
            )
            lg = logits.float().cpu().numpy()
        lg[~cand_m] = -np.inf
        for i, c in enumerate(chunk):
            chosen_logit = float(lg[i, c["slot"]])
            best_logit = float(lg[i].max())
            c["rec"]["rank"] = 1 + int((lg[i] > chosen_logit).sum())
            c["rec"]["value_gap"] = best_logit - chosen_logit

    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate v2 replay events (see module docstring)")
    ap.add_argument("--limit", type=int, default=None, help="annotate at most N quarks")
    ap.add_argument("--requark", type=str, default=None,
                    help="(re-)annotate this quark only, replacing existing rows")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, no write")
    ap.add_argument("--fcr-root", type=str, default=str(FCR_ROOT))
    ap.add_argument("--shard", type=str, default=None, metavar="I/M",
                    help="process only every M-th quark starting at I (parallel workers)")
    ap.add_argument("--planner", choices=("cpu", "cuda"), default="cpu",
                    help="reach planner backend (cuda = reach_cuda GPU port, "
                         "bit-exact with cpu v4 on all macro-legal poses)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the champion net (e.g. cuda)")
    ap.add_argument("--spawn-state", choices=("events", "assumed"), default="events",
                    help="'events' seeds the planner with the exact NES spawn "
                         "state when the record has the v2.1 fc/held/hv/scnt "
                         "keys (falls back to assumptions otherwise); "
                         "'assumed' forces the phase-1 neutral-state behavior "
                         "everywhere (reproducibility escape hatch)")
    args = ap.parse_args()

    torch.set_num_threads(1)

    fcr_root = Path(args.fcr_root)
    sys.path.insert(0, str(fcr_root))
    import store  # fightcadeRatings/store.py

    payload = load_checkpoint(CHECKPOINT, map_location="cpu")
    net, aux_dim, candidate_max = _build_net_from_cfg(payload["cfg"], OBS_CHANNELS, args.device)
    net.load_state_dict(payload.get("ema_state_dict") or payload["state_dict"])
    net.to(args.device)
    aux_shim = _make_aux_builder(aux_dim)
    model_ver = CHECKPOINT.name
    print(f"model={model_ver} step={payload.get('step')} planner={PLANNER_VER} "
          f"backend={args.planner}")

    planner = make_batch_planner(args.planner)
    con = store.connect()
    con.execute(CREATE_TABLE)
    con.commit()

    if args.requark:
        rows = con.execute(
            "SELECT quarkid, sha256 FROM processed_replay WHERE quarkid=?",
            (args.requark,),
        ).fetchall()
        if not rows:
            raise SystemExit(f"quark {args.requark!r} not in processed_replay")
    else:
        rows = con.execute(
            "SELECT quarkid, sha256 FROM processed_replay ORDER BY quarkid"
        ).fetchall()
        if args.shard:
            i, m = (int(x) for x in args.shard.split("/"))
            rows = rows[i::m]

    counters: Dict[str, int] = defaultdict(int)
    all_slacks: List[int] = []
    all_ranks: List[int] = []
    quarks_done = 0
    t0 = time.time()

    for quarkid, sha in rows:
        if args.limit is not None and quarks_done >= args.limit:
            break
        if not args.requark:
            done = con.execute(
                "SELECT 1 FROM drmc_annotation WHERE quarkid=? AND planner_ver=? "
                "AND model_ver=? LIMIT 1",
                (quarkid, PLANNER_VER, model_ver),
            ).fetchone()
            if done:
                counters["quarks_skipped_done"] += 1
                continue
        try:
            raw = store.get_blob(sha)
        except FileNotFoundError:
            counters["quarks_blob_missing"] += 1
            continue

        recs = annotate_quark(quarkid, raw, planner, net, aux_shim, candidate_max,
                              counters, device=args.device,
                              spawn_mode=args.spawn_state)
        if not recs:
            continue
        quarks_done += 1

        for r in recs:
            if r["slack"] is not None:
                all_slacks.append(r["slack"])
            if r["rank"] is not None:
                all_ranks.append(r["rank"])

        now = time.time()
        header = {"quarkid": quarkid, "planner_ver": PLANNER_VER,
                  "model_ver": model_ver, "annotated_at": now, "moves": len(recs)}
        blob = "\n".join(json.dumps(x) for x in [header] + recs) + "\n"

        if args.dry_run:
            feas = sum(r["feasible"] for r in recs)
            print(f"[dry-run] {quarkid}: {len(recs)} moves, {feas} feasible, "
                  f"blob {len(blob)} bytes")
            continue

        blob_sha = store.put_blob(blob.encode())
        con.execute("DELETE FROM drmc_annotation WHERE quarkid=?", (quarkid,))
        con.executemany(
            "INSERT OR REPLACE INTO drmc_annotation "
            "(quarkid,p,spawn_f,opt_frames_chosen,opt_frames_best,rank,n_options,"
            "value_gap,tuck,kick,feasible,interrupted,planner_ver,model_ver,"
            "blob_sha,annotated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (quarkid, r["p"], r["spawn_f"], r["opt_frames_chosen"],
                 r["opt_frames_best"], r["rank"], r["n_options"], r["value_gap"],
                 r["tuck"], r["kick"], r["feasible"], r["interrupted"],
                 PLANNER_VER, model_ver, blob_sha, now)
                for r in recs
            ],
        )
        con.commit()
        print(f"{quarkid}: {len(recs)} moves -> blob {blob_sha[:12]}")

    moves = counters["moves"]
    feas_frac = counters["feasible"] / moves if moves else 0.0
    slacks = np.array(all_slacks) if all_slacks else np.array([0])
    ranks = np.array(all_ranks) if all_ranks else np.array([0])
    print("--- summary ---")
    print(f"quarks annotated: {quarks_done}  moves: {moves}")
    print(f"feasible: {counters['feasible']}/{moves} = {feas_frac:.4f}")
    print(f"spawn_state: events {counters.get('spawn_state_events', 0)}/{moves} "
          f"(rest assumed)")
    print(f"interrupted: {counters['interrupted']}")
    print(f"slack: median {np.median(slacks):.0f}  p10 {np.percentile(slacks,10):.0f}  "
          f"p90 {np.percentile(slacks,90):.0f}")
    if all_ranks:
        print(f"rank: median {np.median(ranks):.0f}  rank1 {(ranks==1).mean():.3f}  "
              f"top3 {(ranks<=3).mean():.3f}  p90 {np.percentile(ranks,90):.0f}")
    print(f"counters: {dict(counters)}")
    print(f"wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
