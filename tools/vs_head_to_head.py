"""VS-pool head-to-head probe: search policy vs plain policy, same checkpoint.

Side 0 of every pair is driven by the depth-2 search
(`models/policy/search_policy.SearchPolicy`); side 1 by the plain
deterministic argmax of the same checkpoint. Reports the search side's win
rate with a Wilson 95% CI (draws reported separately).

Usage:
  .venv/bin/python -m tools.vs_head_to_head --matches 60 --level 14 --pairs 6 \
      [--checkpoint PATH] [--beam 8] [--search-deadline-ms 60] [--device mps] \
      [--a-plain]   # A/A control: side 0 also plain argmax
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.policy.candidate_packing import pack_feasible_candidates
from models.policy.search_policy import SearchPolicy
from training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

REPO_ROOT = Path(__file__).resolve().parents[1]

_CANON_TO_RAW = (1, 0, 2)
_SPEEDUPS_MAX = 0x31


def default_checkpoint() -> Path:
    cks = sorted(
        (REPO_ROOT / "runs" / "vs2_02" / "checkpoints").glob("smdp_ppo_*.pt.gz"),
        key=lambda p: p.stat().st_mtime,
    )
    if not cks:
        raise SystemExit("no checkpoints under runs/vs2_02/checkpoints")
    return cks[-1]


def wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class PlainPolicy:
    """Deterministic argmax over packed candidates (the live-bridge default)."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        from tools.eval_policy import _build_net_from_cfg, _make_aux_builder
        from training.utils.checkpoint_io import load_checkpoint

        payload = load_checkpoint(checkpoint, map_location="cpu")
        self.net, aux_dim, self.candidate_max = _build_net_from_cfg(
            payload.get("cfg", {}), 12, device
        )
        self.net.load_state_dict(payload.get("ema_state_dict") or payload["state_dict"])
        self.aux_shim = _make_aux_builder(aux_dim)
        self.device = device

    def act(self, obs: np.ndarray, infos: List[Dict[str, Any]]) -> np.ndarray:
        B = int(obs.shape[0])
        ca = np.full((B, self.candidate_max), -1, dtype=np.int32)
        cm = np.zeros((B, self.candidate_max), dtype=np.bool_)
        cc = np.zeros((B, self.candidate_max), dtype=np.float32)
        pills = np.zeros((B, 2), dtype=np.int64)
        prevs = np.zeros((B, 2), dtype=np.int64)
        for i, info in enumerate(infos):
            pk = pack_feasible_candidates(
                np.asarray(info["placements/feasible_mask"], dtype=bool),
                info["placements/cost_to_lock"],
                max_candidates=self.candidate_max,
                sort_by_cost=True,
            )
            ca[i], cm[i], cc[i] = pk.actions, pk.mask, pk.cost
            pills[i] = np.asarray(info["next_pill_colors"], dtype=np.int64)
            pv = info["preview_pill"]
            # canonical for the net (preview_pill dict is raw NES; the
            # raw<->canonical map is its own inverse)
            prevs[i] = (_CANON_TO_RAW[int(pv["first_color"])],
                        _CANON_TO_RAW[int(pv["second_color"])])
        aux = None
        if self.aux_shim is not None:
            aux = self.aux_shim._build_aux_batch(obs.astype(np.float32), infos)
        with torch.inference_mode():
            logits, _v = self.net(
                torch.from_numpy(obs.astype(np.float32)).to(self.device),
                torch.from_numpy(pills).to(self.device),
                torch.from_numpy(prevs).to(self.device),
                torch.from_numpy(ca).to(self.device),
                torch.from_numpy(cc).to(self.device),
                torch.from_numpy(cm).to(self.device),
                aux=None if aux is None else torch.from_numpy(aux).to(self.device),
            )
            lg = logits.float().cpu().numpy()
        lg[~cm] = -np.inf
        slots = np.argmax(lg, axis=1)
        acts = ca[np.arange(B), slots]
        acts[~cm.any(axis=1)] = -1  # empty mask: noop-fall escape hatch
        return acts.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="default: newest runs/vs2_02/checkpoints/*.pt.gz")
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--level", type=int, default=14)
    ap.add_argument("--speed-setting", type=int, default=2, help="2 = HI")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--search-deadline-ms", type=float, default=60.0)
    ap.add_argument("--device", type=str, default="auto",
                    help="search leaf device (auto = mps if available)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--a-plain", action="store_true",
                    help="A/A control: side 0 plain argmax instead of search")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    print(f"checkpoint: {ckpt}")
    plain = PlainPolicy(ckpt, device="cpu")
    search: Optional[SearchPolicy] = None
    if not args.a_plain:
        search = SearchPolicy(
            ckpt,
            beam=int(args.beam),
            deadline_ms=float(args.search_deadline_ms),
            device=device,
            seed=int(args.seed),
        )
        print(f"search: beam={args.beam} deadline_ms={args.search_deadline_ms} device={device}")

    env = DrMarioVsPoolVecEnv(
        num_pairs=int(args.pairs),
        level=int(args.level),
        speed_setting=int(args.speed_setting),
        randomize_rng=True,
    )
    obs, infos = env.reset(seed=int(args.seed))
    S = env.num_sides
    base_speed_ups = max(0, int(args.level) - 20)
    ep_decisions = np.zeros(S, dtype=np.int64)

    wins = losses = draws = 0
    search_ms: List[float] = []
    agreed = 0
    search_decisions = 0
    t_start = time.perf_counter()

    while wins + losses + draws < int(args.matches):
        need = np.array([bool(i.get("placements/needs_action", False)) for i in infos])
        acts = np.full(S, -1, dtype=np.int32)

        # Side 1 (odd): plain argmax, batched.
        odd = [i for i in range(1, S, 2) if need[i]]
        if odd:
            acts[odd] = plain.act(obs[odd], [infos[i] for i in odd])

        # Side 0 (even): search (or plain for the A/A control).
        even = [i for i in range(0, S, 2) if need[i]]
        if even and search is None:
            acts[even] = plain.act(obs[even], [infos[i] for i in even])
        elif even:
            boards = env._runner.buffers.board_bytes
            pills = env._runner.buffers.pill_colors
            for i in even:
                info = infos[i]
                mask = np.asarray(info["placements/feasible_mask"], dtype=bool)
                if not mask.any():
                    continue  # noop-fall
                pv = info["preview_pill"]
                spd_ups = min(_SPEEDUPS_MAX, base_speed_ups + int(ep_decisions[i]) // 10)
                a, sinfo = search.decide(
                    boards[i],
                    (_CANON_TO_RAW[int(pills[i, 0])], _CANON_TO_RAW[int(pills[i, 1])]),
                    (int(pv["first_color"]), int(pv["second_color"])),
                    mask,
                    info["placements/cost_to_lock"],
                    int(args.speed_setting),
                    spd_ups,
                    int(args.level),
                    viruses_initial=int(info.get("drm/viruses_initial", 0)),
                    frames_used=int(info.get("task/frames_used", 0)),
                )
                if a >= 0:
                    acts[i] = int(a)
                search_ms.append(float(sinfo["elapsed_ms"]))
                agreed += int(bool(sinfo["agreed_with_policy"]))
                search_decisions += 1

        ep_decisions[need] += 1
        obs, _r, term, trunc, infos = env.step(acts)

        for i in range(0, S, 2):
            if not (term[i] or trunc[i]):
                continue
            oc = str(infos[i].get("vs/outcome", ""))
            if oc == "win":
                wins += 1
            elif oc == "loss":
                losses += 1
            else:
                draws += 1
            ep_decisions[i] = 0
            ep_decisions[i + 1] = 0
            done = wins + losses + draws
            n_dec = wins + losses
            wr = wins / max(1, n_dec)
            print(
                f"[{done}/{args.matches}] {oc:7s} | W{wins} L{losses} D{draws} "
                f"win_rate={wr:.3f} ({time.perf_counter() - t_start:.0f}s)",
                flush=True,
            )

    env.close()
    if search is not None:
        search.close()

    n_dec = wins + losses
    wr = wins / max(1, n_dec)
    lo, hi = wilson_ci(wins, max(1, n_dec))
    result = {
        "checkpoint": str(ckpt),
        "level": int(args.level),
        "speed_setting": int(args.speed_setting),
        "beam": int(args.beam) if search is not None else 0,
        "search_deadline_ms": float(args.search_deadline_ms),
        "matches": wins + losses + draws,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wr,
        "win_rate_ci95": [lo, hi],
        "wall_sec": time.perf_counter() - t_start,
    }
    if search_ms:
        arr = np.asarray(search_ms)
        result["search_ms_p50"] = float(np.percentile(arr, 50))
        result["search_ms_p95"] = float(np.percentile(arr, 95))
        result["search_agreement"] = agreed / max(1, search_decisions)
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
