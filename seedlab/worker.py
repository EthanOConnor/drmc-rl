"""Prime95-style catalog search worker.

Claims work units from the catalog queue, runs every seed in the unit through
K attempts on a `DrMarioPoolVecEnv` (exact per-env engine seeds via
`seed_provider`), and folds results into `seed_stats`/`solutions`.

Frame counts are the planner's exact NES frame costs accumulated per decision
(`max(1, tau)`, matching `tools/eval_policy.py` and the trainer).
"""

from __future__ import annotations

import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from seedlab import rng as slrng
from seedlab.db import Attempt, CatalogDB, WorkUnit, pack_actions


@dataclass(slots=True)
class _Job:
    seed: int
    deterministic: bool


@dataclass(slots=True)
class _EpState:
    job: Optional[_Job] = None
    frames: int = 0
    decisions: int = 0
    actions: List[int] = field(default_factory=list)


_STATE_REPR_CHANNELS = {
    "bitplane_bottle": 4,
    "bitplane_bottle_mask": 8,
    "bitplane_bottle_conn": 8,
    "bitplane_bottle_conn_mask": 12,
}


class Solver:
    """Pluggable per-decision action selection (checkpoint / greedy-cost / random)."""

    def __init__(
        self,
        *,
        policy: str,
        checkpoint: Optional[str],
        device: str,
        temperature: float,
        rng: np.random.Generator,
        state_repr: str = "bitplane_bottle_conn_mask",
    ) -> None:
        self.policy = str(policy)
        self.device = str(device)
        self.temperature = float(temperature)
        self.rng = rng
        self.net = None
        self.aux_shim = None
        self.candidate_max = 128
        self.solver_id = self.policy
        if self.policy == "checkpoint":
            if not checkpoint:
                raise ValueError("--checkpoint required for --policy checkpoint")
            from pathlib import Path

            from tools.eval_policy import _build_net_from_cfg, _make_aux_builder
            from training.utils.checkpoint_io import load_checkpoint

            payload = load_checkpoint(Path(checkpoint), map_location="cpu")
            cfg = payload.get("cfg", {})
            in_ch = _STATE_REPR_CHANNELS.get(str(state_repr), 12)
            self.net, aux_dim, self.candidate_max = _build_net_from_cfg(cfg, in_ch, self.device)
            sd = payload.get("ema_state_dict") or payload["state_dict"]
            self.net.load_state_dict(sd)
            self.aux_shim = _make_aux_builder(aux_dim)
            step = payload.get("step")
            self.solver_id = f"ckpt:{Path(checkpoint).stem}@{step}"

    def act(self, obs: np.ndarray, infos: Sequence[dict], deterministic: np.ndarray) -> np.ndarray:
        B = len(infos)
        masks = np.stack([i["placements/feasible_mask"] for i in infos])
        actions = np.zeros(B, dtype=np.int64)

        if self.policy == "random":
            for i in range(B):
                feas = np.flatnonzero(masks[i].reshape(-1))
                actions[i] = int(self.rng.choice(feas)) if feas.size else 0
            return actions

        if self.policy == "greedy-cost":
            for i in range(B):
                cost = infos[i]["placements/cost_to_lock"].reshape(-1).astype(np.float64)
                flat = masks[i].reshape(-1).astype(bool)
                cost[~flat] = np.inf
                if deterministic[i]:
                    actions[i] = int(np.argmin(cost))
                else:
                    order = np.argsort(cost)[:3]
                    order = order[np.isfinite(cost[order])]
                    actions[i] = int(self.rng.choice(order)) if order.size else 0
            return actions

        # checkpoint policy
        import torch

        from models.policy.candidate_packing import pack_feasible_candidates
        from models.policy.placement_dist import MaskedPlacementDist

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
        if self.aux_shim is not None:
            aux = self.aux_shim._build_aux_batch(obs.astype(np.float32), list(infos))

        cand_actions = np.full((B, self.candidate_max), -1, dtype=np.int32)
        cand_mask = np.zeros((B, self.candidate_max), dtype=np.bool_)
        cand_cost = np.zeros((B, self.candidate_max), dtype=np.float32)
        for i in range(B):
            packed = pack_feasible_candidates(
                masks[i].astype(bool), costs[i], max_candidates=self.candidate_max,
                sort_by_cost=True,
            )
            cand_actions[i] = packed.actions
            cand_mask[i] = packed.mask
            cand_cost[i] = packed.cost

        with torch.inference_mode():
            logits, _values = self.net(
                torch.from_numpy(obs.astype(np.float32)).to(self.device),
                torch.from_numpy(pills).to(self.device),
                torch.from_numpy(previews).to(self.device),
                torch.from_numpy(cand_actions).to(self.device),
                torch.from_numpy(cand_cost).to(self.device),
                torch.from_numpy(cand_mask).to(self.device),
                aux=None if aux is None else torch.from_numpy(aux).to(self.device),
            )
            logits_cpu = logits.float().cpu()

        mask_t = torch.from_numpy(cand_mask)
        det_dist = MaskedPlacementDist(logits_cpu, mask_t)
        det_slots = det_dist.mode().numpy().reshape(-1).astype(np.int64)
        if self.temperature > 0:
            samp_dist = MaskedPlacementDist(logits_cpu / self.temperature, mask_t)
            samp_slots, _lp = samp_dist.sample(deterministic=False)
            samp_slots = samp_slots.numpy().reshape(-1).astype(np.int64)
        else:
            samp_slots = det_slots
        slots = np.where(deterministic, det_slots, samp_slots)
        return cand_actions[np.arange(B), slots].astype(np.int64)


class CatalogWorker:
    def __init__(
        self,
        *,
        db: CatalogDB,
        worker_id: str,
        policy: str = "greedy-cost",
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.6,
        attempts_per_seed: int = 1,
        num_envs: int = 32,
        state_repr: str = "bitplane_bottle_conn_mask",
        max_decisions: int = 600,
        levels: Optional[Sequence[int]] = None,
        flush_seconds: float = 30.0,
        seed: int = 0,
        max_units: Optional[int] = None,
    ) -> None:
        self.db = db
        self.worker_id = str(worker_id)
        self.attempts_per_seed = int(max(1, attempts_per_seed))
        self.num_envs = int(max(1, num_envs))
        self.state_repr = str(state_repr)
        self.max_decisions = int(max(1, max_decisions))
        self.levels = list(levels) if levels else None
        self.flush_seconds = float(flush_seconds)
        self.max_units = int(max_units) if max_units else None
        self.rng = np.random.default_rng(seed)
        self.solver = Solver(
            policy=policy, checkpoint=checkpoint, device=device,
            temperature=temperature, rng=self.rng, state_repr=self.state_repr,
        )
        self.stop_requested = False
        # Lifetime counters (for status lines).
        self.total_attempts = 0
        self.total_clears = 0
        self.total_new_bests = 0

    def install_signal_handlers(self) -> None:
        def _handler(_sig, _frame):
            self.stop_requested = True
            print("[seedlab] stop requested; finishing current flush...", flush=True)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    # ------------------------------------------------------------------ unit run
    def run(self) -> None:
        units_done = 0
        while not self.stop_requested:
            if self.max_units is not None and units_done >= self.max_units:
                print(f"[seedlab] max units reached ({units_done}); exiting.", flush=True)
                return
            unit = self.db.claim_unit(worker_id=self.worker_id, levels=self.levels)
            if unit is None:
                print("[seedlab] no todo units left; exiting.", flush=True)
                return
            try:
                completed = self._run_unit(unit)
            except Exception:
                self.db.release_unit(unit.id)
                raise
            if completed:
                self.db.complete_unit(unit.id)
                units_done += 1
            else:
                self.db.release_unit(unit.id)
                return

    def _run_unit(self, unit: WorkUnit) -> bool:
        """Run one work unit; returns True if it finished, False if interrupted."""

        from training.envs.drmario_pool_vec import DrMarioPoolVecEnv

        orbit = slrng.orbit()
        seeds = orbit[unit.seed_lo : unit.seed_hi]
        jobs: Deque[_Job] = deque()
        for s in seeds:
            for a in range(self.attempts_per_seed):
                jobs.append(_Job(seed=s, deterministic=(unit.pass_idx == 0 and a == 0)))
        n_jobs = len(jobs)
        if n_jobs == 0:
            return True

        eps = [_EpState() for _ in range(self.num_envs)]

        def provider(i: int) -> Optional[Tuple[int, int]]:
            job = jobs.popleft() if jobs else None
            eps[i].job = job
            eps[i].frames = 0
            eps[i].decisions = 0
            eps[i].actions = []
            if job is None:
                return None  # idle tail env; episode ignored
            return slrng.seed_to_bytes(job.seed)

        env = DrMarioPoolVecEnv(
            num_envs=self.num_envs,
            state_repr=self.state_repr,
            level=int(unit.level),
            speed_setting=int(unit.speed),
            randomize_rng=True,  # idle tail envs fall back to random seeds
            seed_provider=provider,
        )
        results: List[Attempt] = []
        done_jobs = 0
        unit_clears = 0
        unit_bests = 0
        pending = np.zeros(self.num_envs, dtype=bool)
        t0 = time.perf_counter()
        last_flush = t0

        try:
            obs, infos = env.reset(seed=int(self.rng.integers(0, 2**31)))
            deterministic = np.array(
                [bool(eps[i].job.deterministic) if eps[i].job else False
                 for i in range(self.num_envs)],
                dtype=bool,
            )
            while done_jobs < n_jobs:
                actions = self.solver.act(obs, infos, deterministic)
                obs, _rewards, terminated, truncated, infos = env.step(actions)

                for i in range(self.num_envs):
                    if pending[i]:
                        # This step was env i's autoreset; action was ignored.
                        pending[i] = False
                        deterministic[i] = bool(eps[i].job.deterministic) if eps[i].job else False
                        continue
                    st = eps[i]
                    if st.job is None:
                        # Idle tail env: force respawn attempts to pick up jobs
                        # if any reappear (they don't today); just recycle.
                        if terminated[i] or truncated[i]:
                            pending[i] = True
                        continue
                    tau = int(infos[i].get("placements/tau", 1))
                    st.frames += max(1, tau)
                    st.decisions += 1
                    st.actions.append(int(actions[i]))

                    over_budget = st.decisions >= self.max_decisions
                    if terminated[i] or truncated[i] or over_budget:
                        drm = infos[i].get("drm", {}) if isinstance(infos[i].get("drm"), dict) else {}
                        cleared = bool(drm.get("cleared", False)) and not over_budget
                        results.append(
                            Attempt(
                                level=int(unit.level),
                                speed=int(unit.speed),
                                seed=int(st.job.seed),
                                cleared=cleared,
                                frames=int(st.frames),
                                spawns=int(st.decisions),
                                solver=self.solver.solver_id,
                                actions=pack_actions(st.actions) if cleared else None,
                            )
                        )
                        done_jobs += 1
                        if cleared:
                            unit_clears += 1
                        if over_budget and not (terminated[i] or truncated[i]):
                            env.request_reset(i)
                        pending[i] = True

                now = time.perf_counter()
                if results and (now - last_flush) >= self.flush_seconds:
                    unit_bests += self.db.record_attempts(results)
                    self.total_attempts += len(results)
                    results.clear()
                    last_flush = now
                if self.stop_requested:
                    break
        finally:
            env.close()

        if results:
            unit_bests += self.db.record_attempts(results)
            self.total_attempts += len(results)
            results.clear()

        dt = max(1e-9, time.perf_counter() - t0)
        self.total_clears += unit_clears
        self.total_new_bests += unit_bests
        finished = done_jobs >= n_jobs
        print(
            f"[seedlab] unit={unit.id} level={unit.level} speed={unit.speed} "
            f"pass={unit.pass_idx} seeds=[{unit.seed_lo},{unit.seed_hi}) "
            f"jobs={done_jobs}/{n_jobs} clears={unit_clears} new_bests={unit_bests} "
            f"({done_jobs / dt:.1f} eps/s){'' if finished else ' [interrupted]'}",
            flush=True,
        )
        return finished
