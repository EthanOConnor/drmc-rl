"""Deterministic checkpoint evaluation over a level sweep.

Runs a trained placement policy (or a random/greedy baseline) on the cpp-pool
backend across fixed levels and reports clear rate, frames-to-clear
distribution, and decision counts. Designed for apples-to-apples comparisons
between checkpoints: fixed RNG seeds, no curriculum, no reward shaping in the
reported numbers.

Examples:
    python -m tools.eval_policy --checkpoint runs/<id>/checkpoints/smdp_ppo_*.pt.gz \
        --levels 0,5,10,15,20 --episodes 32
    python -m tools.eval_policy --policy random --levels 0,10,20 --episodes 32
    python -m tools.eval_policy --policy greedy-cost --levels 0 --episodes 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch

from envs.specs import ram_to_state as ram_specs  # noqa: F401  (layout side effects)
from models.policy.candidate_packing import pack_feasible_candidates
from models.policy.candidate_policy import CandidatePlacementPolicyNet
from models.policy.placement_dist import MaskedPlacementDist
from training.envs.drmario_pool_vec import DrMarioPoolVecEnv
from training.utils.checkpoint_io import load_checkpoint


def _build_net_from_cfg(cfg: Dict[str, Any], in_channels: int, device: str):
    sp = cfg.get("smdp_ppo", cfg)

    def g(key: str, default):
        return sp.get(key, default)

    from training.algo.ppo_smdp import _AUX_DIM_BY_SPEC

    aux_dim = int(_AUX_DIM_BY_SPEC.get(str(g("aux_spec", "none")).strip().lower(), 0))
    if str(g("policy_type", "candidate")).lower() != "candidate":
        raise SystemExit("eval_policy currently supports policy_type=candidate checkpoints")
    net = CandidatePlacementPolicyNet(
        in_channels=int(in_channels),
        board_channels=int(g("candidate_board_channels", 8)),
        board_encoder=str(g("candidate_board_encoder", "cnn")),
        encoder_blocks=int(g("encoder_blocks", 0)),
        d_model=int(g("candidate_d_model", 128)),
        pill_embed_dim=int(g("pill_embed_dim", 128)),
        pill_embed_type=str(g("pill_embed_type", "ordered_pair")),
        num_colors=3,
        aux_dim=aux_dim,
        pos_embed_dim=int(g("candidate_pos_embed_dim", 32)),
        cost_embed_dim=int(g("candidate_cost_embed_dim", 32)),
        cand_hidden_dim=int(g("candidate_hidden_dim", 256)),
        transformer_layers=int(g("candidate_transformer_layers", 4)),
        transformer_heads=int(g("candidate_transformer_heads", 4)),
        transformer_ff_mult=int(g("candidate_transformer_ff_mult", 4)),
        patch_kernel=int(g("candidate_patch_kernel", 9)),
    ).to(device)
    net.eval()
    return net, aux_dim, int(g("candidate_max_candidates", 128))


def _make_aux_builder(aux_dim: int):
    if aux_dim <= 0:
        return None
    from training.algo.ppo_smdp import _AUX_DIM_BY_SPEC, SMDPPPOAdapter

    shim = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    spec = next((s for s, d in _AUX_DIM_BY_SPEC.items() if int(d) == int(aux_dim)), "v1")
    shim.aux_spec = spec
    shim.aux_dim = aux_dim
    return shim


def evaluate_level(
    *,
    level: int,
    episodes: int,
    num_envs: int,
    speed_setting: int,
    state_repr: str,
    policy: str,
    net,
    aux_shim,
    candidate_max: int,
    device: str,
    temperature: float,
    seed: int,
    strength: Optional[float] = None,
    max_decisions_per_episode: int = 2000,
    search=None,
) -> Dict[str, Any]:
    env = DrMarioPoolVecEnv(
        num_envs=num_envs,
        state_repr=state_repr,
        level=level,
        speed_setting=speed_setting,
        randomize_rng=True,
        emit_board=search is not None,
    )
    rng = np.random.default_rng(seed)
    obs, infos = env.reset(seed=seed)

    done_eps: List[Dict[str, Any]] = []
    ep_frames = np.zeros(num_envs, dtype=np.int64)
    ep_decisions = np.zeros(num_envs, dtype=np.int64)

    t_start = time.perf_counter()
    while len(done_eps) < episodes:
        B = num_envs
        masks = np.stack([i["placements/feasible_mask"] for i in infos])
        actions = np.zeros(B, dtype=np.int64)

        if search is not None:
            # Depth-2 search per env (models/policy/search_policy.py). The
            # search takes raw NES colors; pool infos are canonical for the
            # falling pill (swap 0<->1) and raw already for the preview dict.
            base_speed_ups = max(0, level - 20)
            for i in range(B):
                info = infos[i]
                pc = info["next_pill_colors"]
                pv = info["preview_pill"]
                spd_ups = min(0x31, base_speed_ups + int(ep_decisions[i]) // 10)
                a, _sinfo = search.decide(
                    info["board"],
                    ((1, 0, 2)[int(pc[0])], (1, 0, 2)[int(pc[1])]),
                    (int(pv["first_color"]), int(pv["second_color"])),
                    masks[i],
                    info["placements/cost_to_lock"],
                    speed_setting,
                    spd_ups,
                    level,
                    viruses_initial=int(info.get("drm/viruses_initial", 0)),
                    frames_used=int(ep_frames[i]),
                )
                actions[i] = int(a) if int(a) >= 0 else 0
        elif policy == "random":
            for i in range(B):
                feas = np.flatnonzero(masks[i].reshape(-1))
                actions[i] = int(rng.choice(feas)) if feas.size else 0
        elif policy == "greedy-cost":
            for i in range(B):
                cost = infos[i]["placements/cost_to_lock"].reshape(-1).astype(np.float64)
                flat = masks[i].reshape(-1)
                cost[~flat.astype(bool)] = np.inf
                actions[i] = int(np.argmin(cost))
        else:  # checkpoint policy
            costs = np.stack([i["placements/cost_to_lock"] for i in infos]).astype(np.float32)
            pills = np.stack([i["next_pill_colors"] for i in infos]).astype(np.int64)
            previews = np.stack(
                [
                    (
                        [i["preview_pill"]["first_color"], i["preview_pill"]["second_color"]]
                        if isinstance(i.get("preview_pill"), dict)
                        else i["preview_pill"]
                    )
                    for i in infos
                ]
            ).astype(np.int64)
            aux = None
            if aux_shim is not None:
                aux = aux_shim._build_aux_batch(obs.astype(np.float32), list(infos))

            cand_actions = np.full((B, candidate_max), -1, dtype=np.int32)
            cand_mask = np.zeros((B, candidate_max), dtype=np.bool_)
            cand_cost = np.zeros((B, candidate_max), dtype=np.float32)
            for i in range(B):
                packed = pack_feasible_candidates(
                    masks[i].astype(bool), costs[i], max_candidates=candidate_max, sort_by_cost=True
                )
                cand_actions[i] = packed.actions
                cand_mask[i] = packed.mask
                cand_cost[i] = packed.cost

            with torch.inference_mode():
                logits, _values = net(
                    torch.from_numpy(obs.astype(np.float32)).to(device),
                    torch.from_numpy(pills).to(device),
                    torch.from_numpy(previews).to(device),
                    torch.from_numpy(cand_actions).to(device),
                    torch.from_numpy(cand_cost).to(device),
                    torch.from_numpy(cand_mask).to(device),
                    aux=None if aux is None else torch.from_numpy(aux).to(device),
                )
                logits_cpu = logits.float().cpu()
            if strength is not None and strength < 1.0:
                # Value-gap strength dial: sample uniformly among candidates
                # whose logit is within delta of the best, where delta grows as
                # strength drops. Degrades believably (plausible-but-worse
                # placements) unlike epsilon-random. Optionally filters
                # frame-perfect-only candidates (cost outliers) at low strength.
                lg = logits_cpu.numpy().copy()
                lg[~cand_mask] = -np.inf
                best = lg.max(axis=1, keepdims=True)
                # strength 1.0 -> delta 0 (argmax); strength 0.0 -> delta ~ full range
                spread = np.where(np.isfinite(lg), lg, np.nan)
                rng_span = np.nanmax(spread, axis=1, keepdims=True) - np.nanmin(
                    spread, axis=1, keepdims=True
                )
                delta = (1.0 - float(strength)) * np.maximum(rng_span, 1e-6)
                eligible = (lg >= best - delta) & cand_mask
                slot_np = np.zeros(B, dtype=np.int64)
                for i in range(B):
                    idx = np.flatnonzero(eligible[i])
                    slot_np[i] = int(rng.choice(idx)) if idx.size else 0
            elif temperature > 0:
                dist = MaskedPlacementDist(logits_cpu / temperature, torch.from_numpy(cand_mask))
                slot, _lp = dist.sample(deterministic=False)
                slot_np = slot.numpy().reshape(-1).astype(np.int64)
            else:
                dist = MaskedPlacementDist(logits_cpu, torch.from_numpy(cand_mask))
                slot = dist.mode()
                slot_np = slot.numpy().reshape(-1).astype(np.int64)
            actions = cand_actions[np.arange(B), slot_np].astype(np.int64)

        obs, _rewards, terminated, truncated, infos = env.step(actions)
        for i in range(B):
            tau = int(infos[i].get("placements/tau", 0))
            ep_frames[i] += tau
            ep_decisions[i] += 1
            over_budget = ep_decisions[i] >= max_decisions_per_episode
            if terminated[i] or truncated[i] or over_budget:
                drm = infos[i].get("drm", {}) if isinstance(infos[i].get("drm"), dict) else {}
                done_eps.append(
                    {
                        "cleared": bool(drm.get("cleared", False)),
                        "frames": int(ep_frames[i]),
                        "decisions": int(ep_decisions[i]),
                        "viruses_cleared": int(drm.get("viruses_cleared", 0)),
                    }
                )
                ep_frames[i] = 0
                ep_decisions[i] = 0

    env.close()
    eps = done_eps[:episodes]
    cleared = [e for e in eps if e["cleared"]]
    frames_clear = np.array([e["frames"] for e in cleared], dtype=np.float64)
    return {
        "level": level,
        "episodes": len(eps),
        "clear_rate": len(cleared) / max(1, len(eps)),
        "frames_to_clear_p50": float(np.median(frames_clear)) if frames_clear.size else None,
        "frames_to_clear_p90": float(np.percentile(frames_clear, 90)) if frames_clear.size else None,
        "viruses_cleared_mean": float(np.mean([e["viruses_cleared"] for e in eps])),
        "decisions_mean": float(np.mean([e["decisions"] for e in eps])),
        "wall_sec": time.perf_counter() - t_start,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--policy", choices=["checkpoint", "random", "greedy-cost"], default="checkpoint")
    ap.add_argument("--levels", type=str, default="0,5,10,15,20")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--speed-setting", type=int, default=2)
    ap.add_argument("--state-repr", type=str, default="bitplane_bottle_conn_mask")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic argmax")
    ap.add_argument("--strength", type=float, default=None,
                    help="strength dial in [0,1]; <1 samples among near-best candidates")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--search", type=int, nargs="?", const=8, default=None, metavar="BEAM",
                    help="enable depth-2 policy-guided search (docs/SEARCH_DESIGN.md)")
    ap.add_argument("--search-deadline-ms", type=float, default=60.0)
    args = ap.parse_args()

    search = None
    if args.search is not None:
        if args.policy != "checkpoint" or not args.checkpoint:
            raise SystemExit("--search requires --policy checkpoint and --checkpoint")
        from models.policy.search_policy import SearchPolicy

        search = SearchPolicy(
            args.checkpoint,
            beam=int(args.search),
            deadline_ms=float(args.search_deadline_ms),
            device=args.device,
            seed=args.seed,
        )
        print(f"search enabled: beam={args.search} deadline_ms={args.search_deadline_ms}")

    net = None
    aux_shim = None
    candidate_max = 128
    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --policy checkpoint")
        payload = load_checkpoint(Path(args.checkpoint), map_location="cpu")
        cfg = payload.get("cfg", {})
        channels = {"bitplane_bottle": 4, "bitplane_bottle_mask": 8,
                    "bitplane_bottle_conn": 8, "bitplane_bottle_conn_mask": 12}
        in_ch = channels.get(args.state_repr, 12)
        net, aux_dim, candidate_max = _build_net_from_cfg(cfg, in_ch, args.device)
        sd = payload.get("ema_state_dict") or payload["state_dict"]
        net.load_state_dict(sd)
        aux_shim = _make_aux_builder(aux_dim)
        print(f"loaded checkpoint step={payload.get('step')} sha={payload.get('sha')}")

    results = []
    for level in [int(x) for x in args.levels.split(",") if x.strip()]:
        res = evaluate_level(
            level=level,
            episodes=args.episodes,
            num_envs=args.num_envs,
            speed_setting=args.speed_setting,
            state_repr=args.state_repr,
            policy=args.policy,
            net=net,
            aux_shim=aux_shim,
            candidate_max=candidate_max,
            device=args.device,
            temperature=args.temperature,
            seed=args.seed,
            strength=args.strength,
            search=search,
        )
        results.append(res)
        p50 = res["frames_to_clear_p50"]
        print(
            f"level={level:3d} clear={res['clear_rate']*100:5.1f}% "
            f"p50_frames={p50 if p50 is None else int(p50)} "
            f"p90={res['frames_to_clear_p90']} viruses={res['viruses_cleared_mean']:.1f} "
            f"({res['wall_sec']:.1f}s)"
        )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
