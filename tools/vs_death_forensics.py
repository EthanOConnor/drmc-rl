"""Match-ending forensics for VS self-play: pressure kill vs self-burial.

Plays plain-vs-plain matches (same checkpoint, deterministic argmax both
sides, fresh seeds) and records, per side at every decision: stack height
(max column height from ``board_bytes``), ``viruses_rem``, and cumulative
garbage received (volley release events with pair-clock frame stamps).

Per finished match it emits one JSON line: winner side, final viruses_rem
for both sides, match length, the loser's garbage received in the final
15 s vs its per-15 s match average, the loser's stack-height trajectory
over its final 10 decisions, and whether ANY garbage landed on the loser
in the final 15 s. A summary distinguishes pressure kills (final-15 s
garbage >= 2x match average) from pure self-burial (zero garbage in the
final 15 s).

Timing caveat: volley events are *release* events (checkReleaseAttack on
the receiver's board), i.e. the moment garbage starts falling from the
receiver's top row — effectively "landed", not "sent". A volley released
just before the window can still be settling inside it.

Usage:
  .venv/bin/python -m tools.vs_death_forensics --matches 40 --pairs 4 \
      --checkpoint runs/best_agents/vs_champion_smdp_ppo_step530046434.pt.gz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from drmc_rl.envs.backends.drmario_vs_pool import GRID_H, GRID_W
from tools.vs_head_to_head import PlainPolicy
from drmc_rl.training.envs.drmario_vs_vec import NES_FPS, DrMarioVsPoolVecEnv

WINDOW_S = 15.0
WINDOW_FRAMES = int(round(WINDOW_S * NES_FPS))
TAIL_DECISIONS = 10


def stack_heights(boards: np.ndarray) -> np.ndarray:
    """Max column height per side from NES tile bytes (row 0 = top)."""

    b = boards.reshape(-1, GRID_H, GRID_W)
    occ = b != 0xFF
    any_row = occ.any(axis=2)  # (S, H)
    # First occupied row from the top; H if the board is empty.
    first = np.where(any_row.any(axis=1), np.argmax(any_row, axis=1), GRID_H)
    return (GRID_H - first).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str,
                    default="runs/best_agents/vs_champion_smdp_ppo_step530046434.pt.gz")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--level", type=int, default=14)
    ap.add_argument("--speed-setting", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    policy = PlainPolicy(Path(args.checkpoint), device="cpu")
    env = DrMarioVsPoolVecEnv(
        num_pairs=int(args.pairs),
        level=int(args.level),
        speed_setting=int(args.speed_setting),
        randomize_rng=True,
    )
    obs, infos = env.reset(seed=int(args.seed))
    S = env.num_sides

    # Per-side, per-match logs (cleared when the pair's match is recorded).
    dec_log: List[List[tuple]] = [[] for _ in range(S)]  # (pair_frame, height, viruses)
    recv_log: List[List[tuple]] = [[] for _ in range(S)]  # (pair_frame, half_pills)

    records: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    def pair_clocks() -> np.ndarray:
        sf = env._runner.buffers.side_frames.reshape(env.num_pairs, 2)
        return sf.max(axis=1).astype(np.int64)

    while len(records) < int(args.matches):
        need = [bool(i.get("placements/needs_action", False)) for i in infos]
        idx = [i for i in range(S) if need[i]]
        acts = np.full(S, -1, dtype=np.int32)
        if idx:
            acts[idx] = policy.act(obs[idx], [infos[i] for i in idx])

        # Decision-time snapshot (settled board, new pill spawned).
        clocks = pair_clocks()
        hts = stack_heights(env._runner.buffers.board_bytes)
        vir = env._runner.buffers.viruses_rem
        for i in idx:
            dec_log[i].append((int(clocks[i // 2]), int(hts[i]), int(vir[i])))

        obs, _r, term, trunc, infos = env.step(acts)

        # Volley release events from this step (re-reading the runner's event
        # buffer is safe; env.step does not clear it).
        for v in env._runner.volleys():
            recv_log[v.pair * 2 + v.receiver].append((int(v.frame), int(v.size)))

        clocks = pair_clocks()
        for pair_i in range(env.num_pairs):
            i0, i1 = pair_i * 2, pair_i * 2 + 1
            if not (term[i0] or trunc[i0] or term[i1] or trunc[i1]):
                continue
            oc0 = str(infos[i0].get("vs/outcome", ""))
            end_frame = int(clocks[pair_i])
            rec: Dict[str, Any] = {
                "match": len(records),
                "outcome_p1": oc0,
                "len_s": round(end_frame / NES_FPS, 1),
            }
            if oc0 in ("win", "loss"):
                w = i0 if oc0 == "win" else i1
                l = i0 ^ w ^ i1  # the other side
                recv_l = recv_log[l]
                total_l = sum(s for _, s in recv_l)
                final15 = sum(s for f, s in recv_l if f >= end_frame - WINDOW_FRAMES)
                per15_avg = total_l * WINDOW_FRAMES / max(1, end_frame)
                tail = dec_log[l][-TAIL_DECISIONS:]
                tail_h = [h for _, h, _ in tail]
                slope = (
                    float(np.polyfit(np.arange(len(tail_h)), tail_h, 1)[0])
                    if len(tail_h) >= 2 else float("nan")
                )
                rec.update(
                    winner_side=w % 2,
                    loser_viruses_rem=int(infos[l]["drm"]["viruses_remaining"]),
                    winner_viruses_rem=int(infos[w]["drm"]["viruses_remaining"]),
                    loser_recv_total=total_l,
                    winner_recv_total=sum(s for _, s in recv_log[w]),
                    loser_recv_final15s=final15,
                    loser_recv_per15s_avg=round(per15_avg, 2),
                    loser_garbage_final15s_any=bool(final15 > 0),
                    loser_final10_heights=tail_h,
                    loser_final10_slope=round(slope, 3),
                )
            records.append(rec)
            print(json.dumps(rec), flush=True)
            for i in (i0, i1):
                dec_log[i] = []
                recv_log[i] = []
            if len(records) >= int(args.matches):
                break

    env.close()

    dec = [r for r in records if "winner_side" in r]
    n = len(dec)
    print(f"\n=== forensics over {len(records)} matches "
          f"({n} decisive, {len(records) - n} draw/timeout, "
          f"{time.perf_counter() - t0:.0f}s wall) ===")
    summary: Dict[str, Any] = {"matches": len(records), "decisive": n}
    if n:
        zero15 = [r for r in dec if not r["loser_garbage_final15s_any"]]
        pressure = [
            r for r in dec
            if r["loser_recv_final15s"] > 0
            and r["loser_recv_final15s"] >= 2.0 * max(r["loser_recv_per15s_avg"], 1e-9)
        ]
        zero_slopes = [r["loser_final10_slope"] for r in zero15
                       if np.isfinite(r["loser_final10_slope"])]
        summary.update(
            frac_loss_zero_garbage_final15s=len(zero15) / n,
            frac_loss_pressure_kill_2x=len(pressure) / n,
            mean_loser_viruses_rem=float(np.mean([r["loser_viruses_rem"] for r in dec])),
            mean_winner_viruses_rem=float(np.mean([r["winner_viruses_rem"] for r in dec])),
            mean_final10_slope_no_garbage=(
                float(np.mean(zero_slopes)) if zero_slopes else None
            ),
            mean_garbage_recv_losers=float(np.mean([r["loser_recv_total"] for r in dec])),
            mean_garbage_recv_winners=float(np.mean([r["winner_recv_total"] for r in dec])),
            mean_len_s=float(np.mean([r["len_s"] for r in dec])),
        )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        out = {"summary": summary, "matches": records}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
