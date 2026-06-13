"""Live-play agent server for the Fightcade Dr. Mario bridge.

Python half of the live bridge (Lua half: tools/fc_live_agent.lua). The Lua
script running inside Fightcade's fcadefbneo appends per-frame game state to
``<ipc_dir>/state.jsonl``; this server tails it, plans once per pill spawn,
and writes back a FRAME-INDEXED button script to ``<ipc_dir>/plan.json``
(atomic tmp+rename). Lua applies ``buttons[frame - start_frame]`` each frame,
which is deterministic under GGPO rollback re-execution.

state.jsonl line schema (written by fc_live_agent.lua, one JSON object/line):
  f        emulator frame index (fba.framecount()) at the before-frame hook
  nesf     NES $0043 frameCounter (parity source for the planner)
  mode     NES $0046 (4 = gameplay)
  side     1|2 (which player the Lua controls)
  pc       pill spawn counter (BCD $0310/$0311 decoded; per-game, resets to 1)
  na       our side's nextAction ($0317/$0397); 0 = pill falling (decision)
  pill     [c1, c2] raw NES falling pill colors ($0301/$0302)
  prev     [c1, c2] raw NES preview colors ($031A/$031B)
  x, y     falling pill base col / row-from-bottom ($0305/$0306)
  rot      falling pill rotation ($0325)
  sc, hv   speedCounter / horVelocity ($0312/$0313)
  spd      speedSetting ($030B), spdups speedUps ($030A), level ($0316)
  atk_us / atk_opp   attackSize for our slot / the opponent slot
  field    our 128-byte playfield as 256 hex chars (row 0 = top)

plan.json schema (written here):
  {"plan_id": n, "spawn": pc, "start_frame": F, "buttons": [b0, b1, ...]}
where buttons[i] is the NES button byte for emulated frame F+i
(bit0=Right, bit1=Left, bit2=Down, bit6=B, bit7=A — DrMarioPool.cpp BTN_*).

Planning: the planner is seeded from the *latest* state line's falling-pill
micro-state, rolled forward ``--margin`` frames of neutral input (the frames
that elapse before the plan can land), then ``drm_reach_bfs_full`` enumerates
all reachable lock poses with frame-exact controller scripts. The policy
(checkpoint / greedy-cost / random) picks the pose; the chosen pose's script
is converted to button bytes. If the plan misses its start window, or a
verification state line disagrees with the predicted trajectory, the server
re-plans from the fresh micro-state (exact: the planner accepts mid-fall
state). Known caveat: re-plans assume a neutral controller at the seed frame;
the 1-2 frame plan pickup delay in Lua can briefly violate that — desyncs are
logged and corrected by the next verification line.

Modes:
  real (default)  tail state.jsonl forever
  --dry-run       plan + log decisions, never write plan.json
  --bench N       measure spawn->plan-written latency over N synthetic spawns

Search: ``--search [BEAM]`` replaces the plain argmax with the depth-2
policy-guided search (`models/policy/search_policy.SearchPolicy`,
docs/SEARCH_DESIGN.md). The search deadline (``--search-deadline-ms``,
default 60) must leave room inside the plan margin (6 frames ~= 100 ms) for
the v1 planner + script generation; verify with ``--bench``. Everything else
(v1 script generation, plan.json, replan loop) is unchanged.

Ponder: ``--ponder`` (implies ``--search``) additionally searches the *next*
decision in the background while the committed script executes
(`PonderingSearchPolicy`). After every plan write the predicted next root is
pondered; at the next spawn a cache hit commits instantly with a shrunken
plan margin (``PONDER_MARGIN`` = 2 frames instead of 6 — planner + script
p95 is ~10 ms, well inside 33 ms), which preserves placements that a 6-frame
roll-forward would lose; the decision log reports ``n_options`` at both
margins on hits. A miss (garbage, desync, replan) falls back to the normal
deadline search at the normal margin.

Usage:
  .venv/bin/python -m tools.live_agent_server --ipc-dir /tmp/drmc_live \
      [--side 1] [--policy checkpoint|greedy-cost|random] [--checkpoint PATH]
      [--temperature 0.0] [--margin 6] [--dry-run] [--bench 100]
      [--search 8] [--search-deadline-ms 60] [--device auto] [--ponder]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from envs.backends.drmario_pool import default_library_path as pool_library_path
from envs.retro.fast_reach import (
    FrameState,
    HoldDir,
    compute_speed_threshold,
    simulate_frame,
)
from envs.retro.reach_native import NativeReachabilityRunner
from tools.annotate_replay_events import POSE_TO_ACTION, build_obs, canonical_color

GRID_W, GRID_H = 8, 16
NEUTRAL_ACTION = 0  # 18-action index: neutral dir, no down, no rotation
DEFAULT_MARGIN = 6  # frames between a state line and its plan's start_frame
PONDER_MARGIN = 2  # margin on a ponder cache hit (decision precomputed)

# NES button bits (DrMarioPool.cpp).
BTN_RIGHT, BTN_LEFT, BTN_DOWN, BTN_B, BTN_A = 0x01, 0x02, 0x04, 0x40, 0x80


def _buttons_table() -> List[int]:
    """18-action index -> NES button byte (DrMarioPool::buttons_mask_from_reach_action)."""
    out = []
    for idx in range(18):
        hold_dir = idx // 6
        sub = idx % 6
        rot = sub % 3
        mask = 0
        if hold_dir == 1:
            mask |= BTN_LEFT
        elif hold_dir == 2:
            mask |= BTN_RIGHT
        if sub >= 3:
            mask |= BTN_DOWN
        if rot == 1:
            mask |= BTN_A
        elif rot == 2:
            mask |= BTN_B
        out.append(mask)
    return out


ACTION_TO_BUTTONS: List[int] = _buttons_table()


def _action_to_pose_map() -> np.ndarray:
    """Inverse of POSE_TO_ACTION: macro action -> pose idx (rot*128 + y*8 + x)."""
    out = np.full(512, -1, dtype=np.int32)
    for pose, action in enumerate(POSE_TO_ACTION):
        if action >= 0:
            out[action] = pose
    return out


ACTION_TO_POSE = _action_to_pose_map()


def field_from_hex(s: str) -> np.ndarray:
    field = np.frombuffer(bytes.fromhex(s), dtype=np.uint8)
    if field.shape[0] != GRID_H * GRID_W:
        raise ValueError(f"field hex decodes to {field.shape[0]} bytes, want 128")
    return field.reshape(GRID_H, GRID_W)


def cols_from_field(field: np.ndarray) -> np.ndarray:
    """uint16[8] column masks, bit r (row 0 = top) = occupied.

    Matches harness.cpp cell_empty: empty when v >= 0xF0 or mid-clear ($Bx).
    0x00 (never present during play) is also treated as empty for safety.
    """
    occ = (field < 0xF0) & ((field & 0xF0) != 0xB0) & (field != 0x00)
    cols = np.zeros(GRID_W, dtype=np.uint16)
    for r in range(GRID_H):
        cols |= occ[r].astype(np.uint16) << r
    return cols


def default_checkpoint() -> Path:
    cand_dir = REPO_ROOT / "runs" / "vs_push_01" / "checkpoints"
    cks = sorted(cand_dir.glob("smdp_ppo_*.pt.gz"), key=lambda p: p.stat().st_mtime)
    if cks:
        return cks[-1]
    return REPO_ROOT / "runs" / "best_agents" / "smdp_ppo_step535164979.pt.gz"


@dataclass
class ActivePlan:
    plan_id: int
    spawn: int
    start_frame: int
    buttons: List[int]
    script: np.ndarray  # 18-action indices
    states: List[FrameState]  # states[i] = predicted state at start of frame start_frame+i
    pose: tuple  # (x, y_top, rot)
    cost: int


class LiveAgentServer:
    def __init__(
        self,
        ipc_dir: str | Path,
        *,
        side: int = 1,
        policy: str = "checkpoint",
        checkpoint: Optional[str] = None,
        temperature: float = 0.0,
        margin: int = DEFAULT_MARGIN,
        max_frames: int = 2048,
        dry_run: bool = False,
        log_path: Optional[str] = None,
        seed: int = 0,
        search_beam: Optional[int] = None,
        search_deadline_ms: float = 60.0,
        device: str = "auto",
        ponder: bool = False,
        ponder_beam: int | None = None,
    ) -> None:
        self.dir = Path(ipc_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.side = int(side)
        self.policy = str(policy)
        self.temperature = float(temperature)
        self.margin = int(margin)
        self.ponder = bool(ponder)
        self.dry_run = bool(dry_run)
        self.rng = np.random.default_rng(seed)
        self._log_fh = open(log_path, "a") if log_path else None

        self.runner = NativeReachabilityRunner(
            max_frames=int(max_frames), lib_path=str(pool_library_path())
        )

        # Per-spawn decision state.
        self.cur_pc: Optional[int] = None
        self.plan: Optional[ActivePlan] = None
        self.plan_seq = 0
        self.game_start_f = 0
        self.viruses_initial = 0
        self.last_decision: Optional[Dict[str, Any]] = None

        self._net = None
        self._aux_shim = None
        self._candidate_max = 128
        self._search = None
        self._search_info: Optional[Dict[str, Any]] = None
        self.checkpoint_name: Optional[str] = None
        if self.ponder and search_beam is None:
            search_beam = 8  # --ponder implies --search
        if search_beam is not None:
            if self.policy != "checkpoint":
                raise ValueError("--search requires --policy checkpoint")
            self._load_search(checkpoint, int(search_beam), float(search_deadline_ms), device,
                              ponder_beam=ponder_beam)
        elif self.policy == "checkpoint":
            self._load_net(checkpoint)
        elif self.policy not in ("greedy-cost", "random"):
            raise ValueError(f"unknown policy {policy!r}")

    # ------------------------------------------------------------- policy
    def _load_net(self, checkpoint: Optional[str]) -> None:
        import torch

        from tools.eval_policy import _build_net_from_cfg, _make_aux_builder
        from training.utils.checkpoint_io import load_checkpoint

        torch.set_num_threads(1)  # training runs share this box
        path = Path(checkpoint) if checkpoint else default_checkpoint()
        payload = load_checkpoint(path, map_location="cpu")
        net, aux_dim, candidate_max = _build_net_from_cfg(payload.get("cfg", {}), 12, "cpu")
        net.load_state_dict(payload.get("ema_state_dict") or payload["state_dict"])
        self._net = net
        self._aux_shim = _make_aux_builder(aux_dim)
        self._candidate_max = int(candidate_max)
        self.checkpoint_name = path.name
        self._log(
            {
                "type": "model",
                "checkpoint": str(path),
                "step": payload.get("step"),
                "aux_dim": aux_dim,
            }
        )

    def _load_search(
        self, checkpoint: Optional[str], beam: int, deadline_ms: float, device: str,
        *, ponder_beam: Optional[int] = None,
    ) -> None:
        import torch

        from models.policy.search_policy import PonderingSearchPolicy, SearchPolicy

        torch.set_num_threads(1)  # training runs share this box
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        path = Path(checkpoint) if checkpoint else default_checkpoint()
        cls = PonderingSearchPolicy if self.ponder else SearchPolicy
        kw = {}
        if self.ponder and ponder_beam:
            kw["ponder_beam"] = int(ponder_beam)
        self._search = cls(
            path,
            beam=beam,
            deadline_ms=deadline_ms,
            device=device,
            seed=int(self.rng.integers(2**31)),
            **kw,
        )
        self.checkpoint_name = path.name
        self._log(
            {
                "type": "model",
                "checkpoint": str(path),
                "step": self._search.checkpoint_step,
                "search_beam": beam,
                "search_deadline_ms": deadline_ms,
                "device": device,
                "ponder": self.ponder,
            }
        )

    def _choose(
        self,
        d: Dict[str, Any],
        field: np.ndarray,
        action_cost: np.ndarray,
        feasible: np.ndarray,
    ) -> int:
        feas_idx = np.flatnonzero(feasible)
        self._search_info = None
        if self._search is not None:
            action, info = self._search.decide(
                field.reshape(-1),
                (int(d["pill"][0]), int(d["pill"][1])),
                (int(d["prev"][0]), int(d["prev"][1])),
                feasible,
                action_cost,
                int(d.get("spd", 2)),
                int(d.get("spdups", 0)),
                int(d.get("level", 0)),
                viruses_initial=int(self.viruses_initial),
                frames_used=int(d["f"]) - int(self.game_start_f),
            )
            self._search_info = info
            if action < 0:  # search saw no candidates; fall back to cheapest
                cost = action_cost.astype(np.float64).copy()
                cost[~feasible] = np.inf
                return int(np.argmin(cost))
            return int(action)
        if self.policy == "random":
            return int(self.rng.choice(feas_idx))
        if self.policy == "greedy-cost":
            cost = action_cost.astype(np.float64).copy()
            cost[~feasible] = np.inf
            return int(np.argmin(cost))

        # Checkpoint policy: candidate packing + masked choice as in
        # tools/eval_policy.py's checkpoint branch.
        import torch

        from models.policy.candidate_packing import pack_feasible_candidates
        from models.policy.placement_dist import MaskedPlacementDist

        obs = build_obs(field, feasible)[None]
        pills = np.array(
            [[canonical_color(d["pill"][0]), canonical_color(d["pill"][1])]], dtype=np.int64
        )
        prevs = np.array(
            [[canonical_color(d["prev"][0]), canonical_color(d["prev"][1])]], dtype=np.int64
        )
        packed = pack_feasible_candidates(
            feasible.reshape(4, GRID_H, GRID_W),
            action_cost.reshape(4, GRID_H, GRID_W).astype(np.float32),
            max_candidates=self._candidate_max,
            sort_by_cost=True,
        )
        aux = None
        if self._aux_shim is not None:
            spd = int(d.get("spd", 2))
            info = {
                "pill/speed_setting": spd,
                "speed_setting": spd,
                "level": int(d.get("level", 0)),
                "task_mode": "viruses",
                "task/frames_used": int(d["f"]) - int(self.game_start_f),
                "drm/viruses_initial": int(self.viruses_initial),
                "viruses_remaining": int(((field & 0xF0) == 0xD0).sum()),
                "placements/options": int(feas_idx.size),
            }
            aux = self._aux_shim._build_aux_batch(obs, [info])
        with torch.inference_mode():
            logits, _values = self._net(
                torch.from_numpy(obs),
                torch.from_numpy(pills),
                torch.from_numpy(prevs),
                torch.from_numpy(packed.actions[None].astype(np.int32)),
                torch.from_numpy(packed.cost[None].astype(np.float32)),
                torch.from_numpy(packed.mask[None]),
                aux=None if aux is None else torch.from_numpy(aux),
            )
            logits_cpu = logits.float().cpu()
        mask_t = torch.from_numpy(packed.mask[None])
        if self.temperature > 0:
            dist = MaskedPlacementDist(logits_cpu / self.temperature, mask_t)
            slot, _lp = dist.sample(deterministic=False)
        else:
            dist = MaskedPlacementDist(logits_cpu, mask_t)
            slot = dist.mode()
        return int(packed.actions[int(slot.reshape(-1)[0])])

    # ------------------------------------------------------------- planning
    def process_line(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one state line; returns the plan dict if one was written."""
        if int(d.get("na", 1)) != 0:
            return None  # not in pillFalling; nothing to decide
        if "side" in d and int(d["side"]) != self.side:
            self._log({"type": "warn", "msg": f"state line for side {d['side']}, we are {self.side}"})
            return None
        pc = int(d["pc"])
        f = int(d["f"])

        if self.cur_pc is None or pc != self.cur_pc:
            if self.cur_pc is None or pc <= self.cur_pc:
                # Counter reset (or first sight): new game.
                self.game_start_f = f
                self.viruses_initial = int(
                    ((field_from_hex(d["field"]) & 0xF0) == 0xD0).sum()
                )
            self.cur_pc = pc
            self.plan = None
            return self._plan_from(d, reason="spawn")

        if self.plan is None:
            return self._plan_from(d, reason="replan-awaiting")

        idx = f - self.plan.start_frame
        if idx < 0:
            return None  # plan pending start; stale awaiting line
        if idx >= len(self.plan.buttons):
            # Script should have locked by now but the pill is still falling.
            self.plan = None
            self._ponder_invalidate()
            return self._plan_from(d, reason="replan-late")

        # Verification line while the plan is (supposedly) being applied.
        pred = self.plan.states[idx]
        obs_pose = (int(d["x"]), (GRID_H - 1) - int(d["y"]), int(d["rot"]) & 3)
        if (pred.x, pred.y, pred.rot & 3) != obs_pose:
            self._log(
                {
                    "type": "desync",
                    "f": f,
                    "plan_id": self.plan.plan_id,
                    "idx": idx,
                    "predicted": [pred.x, pred.y, pred.rot & 3],
                    "observed": list(obs_pose),
                }
            )
            self.plan = None
            self._ponder_invalidate()
            return self._plan_from(d, reason="replan-desync")
        return None

    def _ponder_invalidate(self) -> None:
        if self.ponder and self._search is not None:
            self._search.ponder_invalidate()

    def _plan_from(self, d: Dict[str, Any], *, reason: str) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        f = int(d["f"])
        field = field_from_hex(d["field"])
        cols = cols_from_field(field)
        thr = compute_speed_threshold(int(d.get("spd", 2)), int(d.get("spdups", 0)))

        # Instant-commit margin: when the next decision was pondered to
        # completion, the search is a ~1 ms cache lookup, so only the planner
        # + script generation (~10 ms) needs to fit inside the margin.
        margin = self.margin
        if (
            self.ponder
            and self._search is not None
            and self._search.has_ponder_for(
                field.reshape(-1), (int(d["pill"][0]), int(d["pill"][1]))
            )
        ):
            margin = min(margin, PONDER_MARGIN)

        st = FrameState(
            x=int(d["x"]),
            y=(GRID_H - 1) - int(d["y"]),
            rot=int(d["rot"]) & 3,
            speed_counter=int(d.get("sc", 0)),
            hor_velocity=int(d.get("hv", 0)),
            hold_dir=HoldDir.NEUTRAL,
            frame_parity=int(d.get("nesf", f)) & 1,
        )
        # Roll forward the latency margin with neutral input: the plan starts
        # at f + margin, and Lua holds neutral until then.
        for _ in range(margin):
            st = simulate_frame(cols, st, NEUTRAL_ACTION, speed_threshold=thr)
            if st.locked:
                self._log({"type": "skip", "f": f, "pc": self.cur_pc,
                           "msg": "pill locks within latency margin"})
                return None

        # Feasibility gain measurement: on an early commit, also count the
        # options that would have survived the default margin.
        n_options_default_margin: Optional[int] = None
        if margin < self.margin:
            st_full = st
            locked_full = False
            for _ in range(self.margin - margin):
                st_full = simulate_frame(cols, st_full, NEUTRAL_ACTION, speed_threshold=thr)
                if st_full.locked:
                    locked_full = True
                    break
            if locked_full:
                n_options_default_margin = 0
            else:
                reach_full = self.runner.bfs_full(cols, st_full, speed_threshold=thr)
                feas_full = np.zeros(512, dtype=bool)
                for pose in np.flatnonzero(reach_full.costs_u16 != 0xFFFF):
                    a = int(POSE_TO_ACTION[pose])
                    if a >= 0:
                        feas_full[a] = True
                n_options_default_margin = int(feas_full.sum())

        reach = self.runner.bfs_full(cols, st, speed_threshold=thr)

        action_cost = np.full(512, 0xFFFF, dtype=np.uint16)
        finite = np.flatnonzero(reach.costs_u16 != 0xFFFF)
        for pose in finite:
            a = int(POSE_TO_ACTION[pose])
            if a >= 0:
                action_cost[a] = reach.costs_u16[pose]
        feasible = action_cost != 0xFFFF
        n_options = int(feasible.sum())
        if n_options == 0:
            self._log({"type": "skip", "f": f, "pc": self.cur_pc, "msg": "no feasible placements"})
            return None

        action = self._choose(d, field, action_cost, feasible)
        pose_idx = int(ACTION_TO_POSE[action])
        px, py, prot = pose_idx & 7, (pose_idx >> 3) & 15, (pose_idx >> 7) & 3
        script = reach.script_for_pose(px, py, prot)
        if script is None:
            self._log({"type": "error", "f": f, "msg": f"no script for chosen pose {pose_idx}"})
            return None
        script = np.array(script, dtype=np.uint8)  # copy: runner reuses buffers
        buttons = [ACTION_TO_BUTTONS[int(a)] for a in script]

        # Predicted per-frame states for desync verification.
        states = [st]
        s = st
        for a in script:
            s = simulate_frame(cols, s, int(a), speed_threshold=thr)
            states.append(s)
        if not (s.locked and (s.x, s.y, s.rot & 3) == (px, py, prot)):
            self._log(
                {
                    "type": "warn",
                    "msg": "script re-simulation does not lock at chosen pose",
                    "pose": [px, py, prot],
                    "sim": [s.x, s.y, s.rot & 3, bool(s.locked)],
                }
            )

        self.plan_seq += 1
        start_frame = f + margin
        plan = {
            "plan_id": self.plan_seq,
            "spawn": int(d["pc"]),
            "start_frame": start_frame,
            "buttons": buttons,
        }
        self._write_plan(plan)
        latency_ms = (time.perf_counter() - t0) * 1e3
        self.plan = ActivePlan(
            plan_id=self.plan_seq,
            spawn=int(d["pc"]),
            start_frame=start_frame,
            buttons=buttons,
            script=script,
            states=states,
            pose=(px, py, prot),
            cost=int(action_cost[action]),
        )
        self.last_decision = {
            "type": "decision",
            "reason": reason,
            "f": f,
            "pc": int(d["pc"]),
            "plan_id": self.plan_seq,
            "pose": [px, py, prot],
            "action": int(action),
            "cost": int(action_cost[action]),
            "n_options": n_options,
            "script_len": len(buttons),
            "start_frame": start_frame,
            "margin": margin,
            "latency_ms": round(latency_ms, 3),
            "script": script.tolist(),
        }
        if n_options_default_margin is not None:
            self.last_decision["n_options_default_margin"] = n_options_default_margin
        if self._search_info is not None:
            si = self._search_info
            self.last_decision["search"] = {
                "stage": si.get("stage"),
                "agreed_with_policy": bool(si.get("agreed_with_policy", True)),
                "fallback_action": si.get("fallback_action"),
                "nodes_expanded": si.get("nodes_expanded"),
                "value_best": si.get("value_best"),
                "value_fallback": si.get("value_fallback"),
                "search_ms": round(float(si.get("elapsed_ms", 0.0)), 3),
            }
            if self.ponder:
                self.last_decision["search"]["ponder_hit"] = bool(si.get("ponder_hit", False))
                if si.get("ponder_depth") is not None:
                    self.last_decision["search"]["ponder_depth"] = si.get("ponder_depth")
        self._log({k: v for k, v in self.last_decision.items() if k != "script"})
        # Kick the next-decision ponder while the committed script executes
        # (newest job wins, so replans automatically supersede stale jobs).
        if self.ponder and self._search is not None:
            self._search.start_ponder(
                field.reshape(-1),
                int(action),
                (int(d["pill"][0]), int(d["pill"][1])),
                (int(d["prev"][0]), int(d["prev"][1])),
                feasible_mask512=feasible,
                cost_to_lock512=action_cost,
                speed_setting=int(d.get("spd", 2)),
                speed_ups=int(d.get("spdups", 0)),
                level=int(d.get("level", 0)),
                viruses_initial=int(self.viruses_initial),
                frames_used=f - int(self.game_start_f),
            )
        return plan

    # ------------------------------------------------------------- IO
    def _write_plan(self, plan: Dict[str, Any]) -> None:
        if self.dry_run:
            return
        tmp = self.dir / "plan.json.tmp"
        tmp.write_text(json.dumps(plan, separators=(",", ":")))
        os.replace(tmp, self.dir / "plan.json")

    def _log(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, separators=(",", ":"))
        print(line, flush=True)
        if self._log_fh is not None:
            self._log_fh.write(line + "\n")
            self._log_fh.flush()

    def run(self, *, once: bool = False, from_start: bool = False) -> None:
        """Tail state.jsonl. ``once`` processes available lines then returns."""
        state_path = self.dir / "state.jsonl"
        while not state_path.exists():
            if once:
                return
            time.sleep(0.05)
        with state_path.open("r") as fh:
            if not from_start and not once:
                fh.seek(0, os.SEEK_END)
            while True:
                lines = self._drain(fh, once=once)
                if lines is None:
                    return
                for d in lines:
                    try:
                        self.process_line(d)
                    except Exception as exc:  # keep the bridge alive
                        self._log({"type": "error", "msg": repr(exc)})

    def _drain(self, fh, *, once: bool) -> Optional[List[Dict[str, Any]]]:
        """Read all complete lines; collapse runs of the same spawn to the latest."""
        raw: List[Dict[str, Any]] = []
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line or not line.endswith("\n"):
                fh.seek(pos)
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                self._log({"type": "warn", "msg": f"bad state line: {line[:120]}"})
        if not raw:
            if once:
                return None
            time.sleep(0.002)
            return []
        out: List[Dict[str, Any]] = []
        for d in raw:
            if out and out[-1].get("pc") == d.get("pc"):
                out[-1] = d  # same spawn: only the freshest micro-state matters
            else:
                out.append(d)
        return out


# ----------------------------------------------------------------- bench
def _synthetic_field(rng: np.ndarray) -> np.ndarray:
    """Empty bottle with a deterministic scattering of viruses in the bottom rows."""
    field = np.full((GRID_H, GRID_W), 0xFF, dtype=np.uint8)
    virus_colors = (0xD0, 0xD1, 0xD2)  # type nibble $D = virus
    g = np.random.default_rng(1234)
    for _ in range(12):
        r = int(g.integers(10, GRID_H))
        c = int(g.integers(0, GRID_W))
        field[r, c] = virus_colors[int(g.integers(0, 3))]
    return field


def bench(server: LiveAgentServer, n: int) -> None:
    field_hex = _synthetic_field(None).tobytes().hex()
    lat: List[float] = []
    for i in range(n):
        line = {
            "f": 1000 + 100 * i,
            "nesf": (1000 + 100 * i) & 0xFF,
            "mode": 4,
            "side": server.side,
            "pc": i + 1,
            "na": 0,
            "pill": [i % 3, (i + 1) % 3],
            "prev": [(i + 2) % 3, i % 3],
            "x": 3,
            "y": 15,
            "rot": 0,
            "sc": 0,
            "hv": 0,
            "spd": 2,
            "spdups": 0,
            "level": 10,
            "atk_us": 0,
            "atk_opp": 0,
            "field": field_hex,
        }
        plan = server.process_line(line)
        if plan is not None and server.last_decision is not None:
            lat.append(float(server.last_decision["latency_ms"]))
    arr = np.array(lat)
    print(
        f"bench: n={arr.size} spawn->plan-written latency ms "
        f"p50={np.percentile(arr, 50):.1f} p95={np.percentile(arr, 95):.1f} "
        f"max={arr.max():.1f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipc-dir", type=str, default=None,
                    help="IPC dir shared with fc_live_agent.lua (FC_DRMC_DIR)")
    ap.add_argument("--side", type=int, choices=(1, 2), default=1)
    ap.add_argument("--policy", choices=("checkpoint", "greedy-cost", "random"),
                    default="checkpoint")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="default: newest runs/vs_push_01/checkpoints/*.pt.gz, "
                         "else runs/best_agents/smdp_ppo_step535164979.pt.gz")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic argmax")
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN,
                    help="frames between a state line and the plan's start_frame")
    ap.add_argument("--dry-run", action="store_true", help="log decisions, never write plan.json")
    ap.add_argument("--bench", type=int, default=None, metavar="N",
                    help="measure spawn->plan latency over N synthetic spawns, then exit")
    ap.add_argument("--once", action="store_true", help="process available lines, then exit")
    ap.add_argument("--from-start", action="store_true",
                    help="process state.jsonl from the beginning (default: tail from end)")
    ap.add_argument("--log", type=str, default=None, help="decision log (jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search", type=int, nargs="?", const=8, default=None, metavar="BEAM",
                    help="enable depth-2 policy-guided search with this beam width "
                         "(default 8 when given without a value)")
    ap.add_argument("--search-deadline-ms", type=float, default=60.0,
                    help="search anytime deadline; must leave plan-margin room "
                         "for the planner + script generation")
    ap.add_argument("--ponder", action="store_true",
                    help="search the next decision during script-execution dead "
                         "time; cache hits commit instantly with a 2-frame plan "
                         "margin (implies --search)")
    ap.add_argument("--ponder-beam", type=int, default=None,
                    help="wider ply-2 beam for ponder jobs (default: same as --search beam)")
    ap.add_argument("--device", type=str, default="auto",
                    help="torch device for the search nets (auto = mps if available)")
    args = ap.parse_args()

    ipc_dir = args.ipc_dir
    if ipc_dir is None:
        if args.bench is None:
            raise SystemExit("--ipc-dir is required (except for --bench)")
        import tempfile

        ipc_dir = tempfile.mkdtemp(prefix="drmc_live_bench_")

    server = LiveAgentServer(
        ipc_dir,
        side=args.side,
        policy=args.policy,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        margin=args.margin,
        dry_run=args.dry_run,
        log_path=args.log,
        seed=args.seed,
        search_beam=args.search,
        search_deadline_ms=args.search_deadline_ms,
        device=args.device,
        ponder=args.ponder,
        ponder_beam=args.ponder_beam,
    )
    if args.bench is not None:
        bench(server, int(args.bench))
        return
    server.run(once=args.once, from_start=args.from_start)


if __name__ == "__main__":
    main()
