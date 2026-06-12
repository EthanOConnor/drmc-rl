"""Behavior-cloned human-style VS opponents from the fightcadeRatings corpus.

Builds rating-banded (state -> placement) supervised datasets from the v2
replay events in ``../fightcadeRatings/data/drmario.sqlite`` (read-only; see
their COORDINATION.md) and trains one small candidate-policy net per WHR band.
The checkpoints embed a ``cfg`` dict compatible with
``tools.eval_policy._build_net_from_cfg``, so `OpponentPool.ensure_loaded`
(training/envs/vs_opponents.py) loads them with zero changes — drop them into
a pool dir / manifest like any frozen snapshot.

Extraction reuses the spawn/lock pairing + planner machinery from
``tools.annotate_replay_events`` (the annotation work already solved spawn
alignment). Per move we store the decision-time field, pill/preview colors,
the planner's feasible candidate set, and the slot of the placement the human
actually chose. Same-color pills mirror the training envs' symmetry
reduction: orientations 2/3 are masked and the chosen action is canonicalized
onto its o0/o1 equivalent.

WHR bands (player's WHR-C ``eC`` at the game's day, nearest-day join like
tools/skill_grade.py): lt1600 (<1600), 1600to2000, gt2000 (>=2000).

Usage (single-threaded + nice on purpose; training runs share this box):
  nice -n 19 .venv/bin/python -m tools.train_bc_opponent extract \
      [--out data/human_vs/bc_dataset_v1.npz] [--max-moves-per-band 50000]
  nice -n 19 .venv/bin/python -m tools.train_bc_opponent train \
      [--dataset data/human_vs/bc_dataset_v1.npz] [--epochs 4] \
      [--out-dir runs/bc_opponents]
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FCR_ROOT = REPO_ROOT.parent / "fightcadeRatings"

from envs.retro.fast_reach import compute_speed_threshold
from models.policy.candidate_packing import pack_feasible_candidates
from tools.annotate_replay_events import (
    GRID_H,
    GRID_W,
    POSE_TO_ACTION,
    build_obs,
    decode_field,
    load_planner,
    occupancy_cols,
    pair_moves,
    parse_quark_events,
)

BANDS = ("lt1600", "1600to2000", "gt2000")
KMAX = 128  # candidate_max_candidates (matches training envs)

# Net config embedded in the checkpoints; `_build_net_from_cfg(cfg, 12, dev)`
# reconstructs the exact architecture (OpponentPool.ensure_loaded path).
BC_NET_CFG: Dict[str, Any] = {
    "smdp_ppo": {
        "policy_type": "candidate",
        "aux_spec": "none",
        "pill_embed_dim": 96,
        "pill_embed_type": "ordered_pair",
        "encoder_blocks": 2,
        "candidate_max_candidates": KMAX,
        "candidate_d_model": 96,
        "candidate_pos_embed_dim": 32,
        "candidate_cost_embed_dim": 32,
        "candidate_hidden_dim": 192,
        "candidate_board_encoder": "cnn",
        "candidate_board_channels": 8,
        "candidate_transformer_layers": 2,
        "candidate_transformer_heads": 4,
        "candidate_transformer_ff_mult": 4,
        "candidate_patch_kernel": 9,
    }
}


def band_of(whr: float) -> str:
    if whr < 1600.0:
        return "lt1600"
    if whr < 2000.0:
        return "1600to2000"
    return "gt2000"


def rating_lookup(players_path: Path):
    """name -> (sorted days, eC list) from WHR-C trajectories (players.json)."""

    players = json.loads(players_path.read_text())
    table: Dict[str, tuple] = {}
    for p in players:
        traj = p.get("traj") or []
        if traj:
            table[p["name"]] = ([t["day"] for t in traj], [t["eC"] for t in traj])

    def rate(name: str, day: int) -> Optional[float]:
        entry = table.get(name)
        if entry is None:
            return None
        days, ecs = entry
        i = bisect.bisect_left(days, day)
        cands = [j for j in (i - 1, i) if 0 <= j < len(days)]
        if not cands:
            return None
        j = min(cands, key=lambda j: abs(days[j] - day))
        return float(ecs[j])

    return rate


# Macro-action geometry: second-cell offset per orientation (see
# annotate_replay_events.ORIENT_INDEX / DrMarioPool ROT_OFFSETS).
_SECOND_CELL = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}


def canonicalize_same_color(action: int) -> int:
    """Map an o>=2 macro action to its o0/o1 same-cells equivalent."""

    o, rc = divmod(int(action), 128)
    r, c = divmod(rc, 8)
    if o == 2:  # cells {(r,c),(r,c-1)} == o0 at (r, c-1)
        return 0 * 128 + r * 8 + (c - 1)
    if o == 3:  # cells {(r,c),(r-1,c)} == o1 at (r-1, c)
        return 1 * 128 + (r - 1) * 8 + c
    return int(action)


def extract_moves_from_events(
    raw: bytes,
    planner,
    *,
    sides: Optional[set] = None,
    max_lock_frames: int = 2048,
) -> List[Dict[str, Any]]:
    """(state -> placement) rows from one quark's v2 event blob.

    Returns one dict per feasible human move whose chosen placement survives
    candidate packing. Deterministic for a fixed blob.
    """

    from collections import defaultdict

    counters: Dict[str, int] = defaultdict(int)
    events = parse_quark_events(raw)
    if not events["spawn"]:
        return []
    moves = pair_moves(events, counters)

    out_costs = np.empty(512, dtype=np.uint16)
    rows: List[Dict[str, Any]] = []
    import ctypes as C

    for mv in moves:
        sp, lk, p = mv["spawn"], mv["lock"], mv["p"]
        if sides is not None and p not in sides:
            continue
        spawn_f = int(sp["f"])
        field = decode_field(sp["field"])
        cols = occupancy_cols(field)
        thr = compute_speed_threshold(int(sp["spd"]), int(sp["spdups"]))
        out_costs.fill(0xFFFF)
        rc = planner(
            cols.ctypes.data_as(C.POINTER(C.c_uint16)),
            3, 0, 0, 0, 0, 0, spawn_f & 1, 0, thr, int(max_lock_frames),
            out_costs.ctypes.data_as(C.POINTER(C.c_uint16)),
        )
        if rc != 0:
            continue

        action_cost = np.full(512, 0xFFFF, dtype=np.uint16)
        for pose in np.flatnonzero(out_costs != 0xFFFF):
            a = POSE_TO_ACTION[pose]
            if a >= 0:
                action_cost[a] = out_costs[pose]

        # Chosen pose from the lock record (NES y counts from the bottom).
        x, rot = int(lk["x"]), int(lk["rot"]) & 3
        y_top = (GRID_H - 1) - int(lk["y"])
        if not (0 <= x < GRID_W and 0 <= y_top < GRID_H):
            continue
        pose = rot * 128 + y_top * 8 + x
        chosen_action = int(POSE_TO_ACTION[pose])
        if chosen_action < 0 or out_costs[pose] == 0xFFFF:
            continue  # infeasible under our movement model (desync canary)

        pill = (int(sp["pill"][0]) & 0x03, int(sp["pill"][1]) & 0x03)
        prev = (int(sp["prev"][0]) & 0x03, int(sp["prev"][1]) & 0x03)
        same_color = pill[0] == pill[1]
        if same_color:
            # Mirror the envs' symmetry reduction: drop o2/o3 duplicates.
            action_cost[2 * 128 :] = 0xFFFF
            chosen_action = canonicalize_same_color(chosen_action)
            if action_cost[chosen_action] == 0xFFFF:
                continue  # canonical variant unreachable (rare edge)

        feasible_512 = action_cost != 0xFFFF
        packed = pack_feasible_candidates(
            feasible_512.reshape(4, GRID_H, GRID_W),
            action_cost.reshape(4, GRID_H, GRID_W),
            max_candidates=KMAX,
            sort_by_cost=True,
        )
        slot = np.flatnonzero(packed.actions[: packed.count] == chosen_action)
        if slot.size == 0:
            continue  # chosen action not packed (overflow beyond KMAX)

        rows.append(
            {
                "p": p,
                "spawn_f": spawn_f,
                "field": field.reshape(-1).copy(),
                "pill": pill,
                "prev": prev,
                "spd": int(sp["spd"]),
                "spdups": int(sp["spdups"]),
                "parity": spawn_f & 1,
                "chosen_action": chosen_action,
                "chosen_slot": int(slot[0]),
                "cand_actions": packed.actions.astype(np.int16).copy(),
                "cand_costs": packed.cost.astype(np.uint16).copy(),
                "cand_count": int(packed.count),
            }
        )
    return rows


def rows_to_arrays(rows: List[Dict[str, Any]], extra: Optional[Dict[str, list]] = None) -> Dict[str, np.ndarray]:
    """Stack extraction rows into the flat npz array layout."""

    n = len(rows)
    arrays = {
        "field": np.zeros((n, 128), dtype=np.uint8),
        "pill": np.zeros((n, 2), dtype=np.uint8),
        "prev": np.zeros((n, 2), dtype=np.uint8),
        "spd": np.zeros((n,), dtype=np.uint8),
        "spdups": np.zeros((n,), dtype=np.uint8),
        "parity": np.zeros((n,), dtype=np.uint8),
        "chosen_action": np.zeros((n,), dtype=np.int16),
        "chosen_slot": np.zeros((n,), dtype=np.int16),
        "cand_actions": np.zeros((n, KMAX), dtype=np.int16),
        "cand_costs": np.zeros((n, KMAX), dtype=np.uint16),
        "cand_count": np.zeros((n,), dtype=np.int16),
    }
    for i, r in enumerate(rows):
        arrays["field"][i] = r["field"]
        arrays["pill"][i] = r["pill"]
        arrays["prev"][i] = r["prev"]
        arrays["spd"][i] = r["spd"]
        arrays["spdups"][i] = r["spdups"]
        arrays["parity"][i] = r["parity"]
        arrays["chosen_action"][i] = r["chosen_action"]
        arrays["chosen_slot"][i] = r["chosen_slot"]
        arrays["cand_actions"][i] = r["cand_actions"]
        arrays["cand_costs"][i] = r["cand_costs"]
        arrays["cand_count"][i] = r["cand_count"]
    for key, values in (extra or {}).items():
        arrays[key] = np.asarray(values)
    return arrays


def batch_inputs(arrays: Dict[str, np.ndarray], idx: np.ndarray):
    """Net inputs (numpy) for a batch of dataset rows.

    Rebuilds the 12-channel obs (board planes + feasible planes) and the
    candidate mask from the stored packed candidates; replicates the envs'
    same-color channel-6/7 zeroing.
    """

    from envs.specs.ram_to_state import COLOR_VALUE_TO_INDEX

    B = idx.shape[0]
    obs = np.zeros((B, 12, GRID_H, GRID_W), dtype=np.float32)
    cand_actions = arrays["cand_actions"][idx].astype(np.int32)
    cand_count = arrays["cand_count"][idx]
    cand_mask = np.arange(KMAX)[None, :] < cand_count[:, None]
    cand_costs = arrays["cand_costs"][idx].astype(np.float32)
    pills_raw = arrays["pill"][idx]
    prevs_raw = arrays["prev"][idx]

    color_map = np.zeros(4, dtype=np.int64)
    for raw_value, canonical in COLOR_VALUE_TO_INDEX.items():
        color_map[raw_value & 0x03] = canonical
    pills = color_map[pills_raw.astype(np.int64)]
    prevs = color_map[prevs_raw.astype(np.int64)]

    for k in range(B):
        i = int(idx[k])
        feasible_512 = np.zeros(512, dtype=bool)
        acts = cand_actions[k][: int(cand_count[k])]
        feasible_512[acts[acts >= 0]] = True
        obs[k] = build_obs(arrays["field"][i].reshape(GRID_H, GRID_W), feasible_512)
        if pills_raw[k, 0] == pills_raw[k, 1]:
            obs[k, 6] = 0.0
            obs[k, 7] = 0.0

    chosen_slot = arrays["chosen_slot"][idx].astype(np.int64)
    return obs, pills, prevs, cand_actions, cand_costs, cand_mask, chosen_slot


def build_bc_net(device: str = "cpu"):
    from tools.eval_policy import _build_net_from_cfg

    net, aux_dim, _kmax = _build_net_from_cfg(BC_NET_CFG, 12, device)
    assert aux_dim == 0
    return net


def train_band(
    arrays: Dict[str, np.ndarray],
    *,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
    val_mask: Optional[np.ndarray] = None,
    device: str = "cpu",
    log_every: int = 200,
) -> tuple:
    """Train one BC net on `arrays`; returns (net, metrics dict)."""

    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    n = arrays["field"].shape[0]
    if val_mask is None:
        val_mask = np.zeros((n,), dtype=bool)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)

    net = build_bc_net(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))

    losses: List[float] = []
    t0 = time.time()
    for epoch in range(int(epochs)):
        order = rng.permutation(train_idx)
        for start in range(0, order.shape[0], batch_size):
            idx = order[start : start + batch_size]
            obs, pills, prevs, ca, cc, cm, slot = batch_inputs(arrays, idx)
            logits, _values = net(
                torch.from_numpy(obs),
                torch.from_numpy(pills),
                torch.from_numpy(prevs),
                torch.from_numpy(ca),
                torch.from_numpy(cc),
                torch.from_numpy(cm),
                aux=None,
            )
            logits = logits.masked_fill(~torch.from_numpy(cm), -1e9)
            loss = F.cross_entropy(logits, torch.from_numpy(slot))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if log_every and len(losses) % log_every == 0:
                print(
                    f"  epoch {epoch} step {len(losses)} loss {np.mean(losses[-log_every:]):.4f} "
                    f"({time.time()-t0:.0f}s)"
                )

    net.eval()
    metrics: Dict[str, float] = {"train_loss_final": float(np.mean(losses[-50:])) if losses else float("nan")}
    for name, idx_all in (("train", train_idx[:20000]), ("val", val_idx)):
        if idx_all.size == 0:
            continue
        top1 = top3 = total = 0
        with torch.inference_mode():
            for start in range(0, idx_all.shape[0], 512):
                idx = idx_all[start : start + 512]
                obs, pills, prevs, ca, cc, cm, slot = batch_inputs(arrays, idx)
                logits, _ = net(
                    torch.from_numpy(obs),
                    torch.from_numpy(pills),
                    torch.from_numpy(prevs),
                    torch.from_numpy(ca),
                    torch.from_numpy(cc),
                    torch.from_numpy(cm),
                    aux=None,
                )
                lg = logits.masked_fill(~torch.from_numpy(cm), -1e9).numpy()
                ranks = (lg > lg[np.arange(len(idx)), slot][:, None]).sum(axis=1)
                top1 += int((ranks == 0).sum())
                top3 += int((ranks <= 2).sum())
                total += len(idx)
        metrics[f"{name}_top1"] = top1 / total
        metrics[f"{name}_top3"] = top3 / total
        metrics[f"{name}_n"] = total
    return net, metrics


# ------------------------------------------------------------------ commands

def cmd_extract(args) -> None:
    import sqlite3

    sys.path.insert(0, str(Path(args.fcr_root)))
    import store  # fightcadeRatings/store.py

    rate = rating_lookup(Path(args.players))
    planner = load_planner()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    quarks = con.execute(
        "SELECT p.quarkid, p.sha256, q.p1, q.p2, q.day FROM processed_replay p "
        "JOIN raw_quark q USING(quarkid) ORDER BY p.quarkid"
    ).fetchall()
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(len(quarks))

    cap = int(args.max_moves_per_band)
    per_band: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BANDS}
    per_band_extra: Dict[str, Dict[str, list]] = {
        b: {"whr": [], "quark_idx": [], "side": []} for b in BANDS
    }
    quark_names: List[str] = []
    t0 = time.time()
    n_quarks = 0

    for oi in order:
        if all(len(per_band[b]) >= cap for b in BANDS):
            break
        quarkid, sha, p1, p2, day = quarks[oi]
        side_band: Dict[int, tuple] = {}
        for p, name in ((1, p1), (2, p2)):
            whr = rate(str(name), int(day))
            if whr is None:
                continue
            b = band_of(whr)
            if len(per_band[b]) < cap:
                side_band[p] = (b, whr)
        if not side_band:
            continue
        try:
            raw = store.get_blob(sha)
        except FileNotFoundError:
            continue
        rows = extract_moves_from_events(raw, planner, sides=set(side_band))
        if not rows:
            continue
        quark_names.append(str(quarkid))
        qidx = len(quark_names) - 1
        n_quarks += 1
        for r in rows:
            b, whr = side_band[r["p"]]
            if len(per_band[b]) >= cap:
                continue
            per_band[b].append(r)
            per_band_extra[b]["whr"].append(whr)
            per_band_extra[b]["quark_idx"].append(qidx)
            per_band_extra[b]["side"].append(r["p"])
        if n_quarks % 25 == 0:
            sizes = {b: len(per_band[b]) for b in BANDS}
            print(f"{n_quarks} quarks, moves {sizes} ({time.time()-t0:.0f}s)")

    out: Dict[str, np.ndarray] = {"quark_names": np.asarray(quark_names)}
    for b in BANDS:
        arrays = rows_to_arrays(per_band[b], per_band_extra[b])
        for key, arr in arrays.items():
            out[f"{b}/{key}"] = arr
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    sizes = {b: len(per_band[b]) for b in BANDS}
    print(f"wrote {out_path} moves={sizes} quarks={n_quarks} wall={time.time()-t0:.0f}s")


def cmd_train(args) -> None:
    import torch

    from training.utils.checkpoint_io import save_checkpoint

    torch.set_num_threads(int(args.threads))
    data = np.load(args.dataset, allow_pickle=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = BANDS if args.band == "all" else (args.band,)
    summary = {}
    for b in bands:
        keys = [k[len(b) + 1 :] for k in data.files if k.startswith(f"{b}/")]
        arrays = {k: data[f"{b}/{k}"] for k in keys}
        n = arrays["field"].shape[0]
        if n == 0:
            print(f"[{b}] empty band, skipped")
            continue
        # Hold out ~5% of quarks (not moves) for validation.
        val_mask = (arrays["quark_idx"] % 20) == 0
        print(f"[{b}] {n} moves ({int(val_mask.sum())} val), training...")
        net, metrics = train_band(
            arrays,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            seed=int(args.seed),
            val_mask=val_mask,
        )
        path = out_dir / f"bc_{b}.pt.gz"
        save_checkpoint(
            {
                "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "cfg": BC_NET_CFG,
                "step": 0,
                "bc_meta": {
                    "band": b,
                    "moves": int(n),
                    "dataset": str(args.dataset),
                    "metrics": metrics,
                    "whr_mean": float(arrays["whr"].mean()),
                    "trained_at": time.time(),
                },
            },
            path,
        )
        print(f"[{b}] saved {path}  metrics={ {k: round(v,4) if isinstance(v,float) else v for k,v in metrics.items()} }")
        summary[b] = metrics
    (out_dir / "bc_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="extract banded BC dataset from the corpus")
    ex.add_argument("--db", default=str(FCR_ROOT / "data" / "drmario.sqlite"))
    ex.add_argument("--players", default=str(FCR_ROOT / "data" / "out" / "players.json"))
    ex.add_argument("--fcr-root", default=str(FCR_ROOT))
    ex.add_argument("--out", default=str(REPO_ROOT / "data" / "human_vs" / "bc_dataset_v1.npz"))
    ex.add_argument("--max-moves-per-band", type=int, default=50000)
    ex.add_argument("--seed", type=int, default=0)
    ex.set_defaults(fn=cmd_extract)

    tr = sub.add_parser("train", help="train one BC net per band")
    tr.add_argument("--dataset", default=str(REPO_ROOT / "data" / "human_vs" / "bc_dataset_v1.npz"))
    tr.add_argument("--band", default="all", choices=("all",) + BANDS)
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--threads", type=int, default=2)
    tr.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "bc_opponents"))
    tr.set_defaults(fn=cmd_train)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
