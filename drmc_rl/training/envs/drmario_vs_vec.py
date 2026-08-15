from __future__ import annotations

"""Batched, in-process 2-player VS pool vector environment.

Two modes:

- Pure self-play (default, ``opponent_pool_cfg=None``): both sides of every
  pair are flattened into ``num_envs = 2 * num_pairs`` "envs" so the existing
  SMDP-PPO adapter can drive them with a single learning policy.
- Opponent pool (``opponent_pool_cfg={"enabled": True, ...}``): only the N
  learner sides (side 0 of each pair) are exposed as envs
  (``num_envs == num_pairs``). The P2 sides are driven internally by frozen
  policy nets sampled from a PFSP `OpponentPool`
  (`drmc_rl/training/envs/vs_opponents.py`); the trainer sees a plain 1P vector env.

The API mirrors `DrMarioPoolVecEnv` (info dicts with
``placements/feasible_mask``, ``placements/cost_to_lock``, ``placements/tau``,
``next_pill_colors``, ``preview_pill``, ``episode``/``drm`` dicts on terminal,
NEXT_STEP autoreset) so `SMDPPPOAdapter` works unchanged.

Observation parity matters: the loaded 1P champion was trained on the native
`DrMarioPool::build_obs` 12-channel ``bitplane_bottle_conn_mask`` planes plus
the Python-side symmetry reduction in `DrMarioPoolVecEnv`. The VS pool does
not build observations natively, so this module replicates both, exactly, in
numpy from ``board_bytes`` + ``feasible_mask`` (see `_build_obs_in_place`).

Opponent-board observability (``state_repr=bitplane_bottle_conn_mask_vs``,
default OFF): channels 0..7 stay the own-bottle planes, channels 8..15 are the
opponent's bottle planes (same scheme, as the opponent observes them), and the
feasible planes move to 16..19. Opponent scalars (virus count, garbage pending
both directions, opponent pill/preview colors) are exposed via ``vs/*`` info
keys for ``aux_spec=v1_vs``. Old 12-channel policies keep working on the
20-channel obs because the candidate net only reads its ``board_channels``
prefix.

Reward: terminal +1 win / -1 loss / 0 draw, plus a small shaping term
``garbage_reward_coef * garbage_sent_delta`` (half pills sent this decision;
anneal via ``set_attr("garbage_reward_coef", ...)``). Optional
``clear_win_bonus``: extra terminal reward on wins earned by clearing all
own viruses (vs opponent topout) — reorders HOW to win, never whether
winning beats losing.
"""

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_vs_pool import (
    GRID_H,
    GRID_W,
    MACRO_ACTIONS,
    VS_OUTCOME_DRAW,
    VS_OUTCOME_LOSS,
    VS_OUTCOME_WIN,
    DrMarioVsPoolRunner,
    build_vs_reset_spec,
)
from drmc_rl.training.envs.policy_batch import PlacementPolicyBatch
from drmc_rl.models.policy.candidate_packing import candidate_bucket_width

NES_FPS = 60.1

# NES tile encoding (see vendor/drmario_native/DrMarioPool.cpp / drmc_rl/game/specs/ram_to_state.py).
_TILE_EMPTY = 0xFF
_TILE_CLEARED = 0xB0
_TILE_JUST_EMPTIED = 0xF0
_MASK_TYPE = 0xF0
_MASK_COLOR = 0x03


def _canonical_to_raw_color(canonical: int) -> int:
    """Map canonical color idx (0=R,1=Y,2=B) -> NES raw bits (Y=0,R=1,B=2)."""

    c = int(canonical) & 0x03
    if c == 0:
        return 1
    if c == 1:
        return 0
    if c == 2:
        return 2
    return 0


def _make_gpu_plan_solver(max_lock_frames: int):
    """Batched CUDA solver shared by native pool consumers."""

    from drmc_rl.envs.backends.gpu_plan_solver import make_gpu_plan_solver

    return make_gpu_plan_solver(int(max_lock_frames))


class DrMarioVsPoolVecEnv:
    """Self-play VS vector environment backed by the native VS pool library.

    Sides are flattened: env index ``i`` is pair ``i // 2``, side ``i % 2``
    (side 0 = P1). Episodes are one VS game; on terminal both sides report
    ``episode``/``drm`` info and the pair autoresets on the next step
    (NEXT_STEP semantics, matching `DrMarioPoolVecEnv`).
    """

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        *,
        num_pairs: int,
        state_repr: str = "bitplane_bottle_conn_mask",
        level: int = 10,
        speed_setting: int = 2,
        randomize_rng: bool = True,
        garbage_reward_coef: float = 0.05,
        clear_win_bonus: float = 0.0,
        virus_progress_reward_coef: float = 0.0,
        decision_penalty: float = 0.0,
        match_horizon_pills: int = 0,
        horizon_penalty: float = 0.0,
        max_lock_frames: int = 2048,
        max_wait_frames: int = 6000,
        lib_path: Optional[str] = None,
        seed_provider: Optional[Callable[[int], Optional[Tuple[int, int]]]] = None,
        frame_counter_provider: Optional[Callable[[int], Optional[int]]] = None,
        opponent_pool_cfg: Optional[Dict[str, Any]] = None,
        start_bank_cfg: Optional[Dict[str, Any]] = None,
        gpu_planner: bool = False,
        direct_policy_batch: bool = False,
    ) -> None:
        self.num_pairs = int(max(1, int(num_pairs)))
        self.num_sides = 2 * self.num_pairs
        self._opp_pool = None
        opp_cfg = dict(opponent_pool_cfg or {})
        self._opp_enabled = bool(opp_cfg.get("enabled", False))
        # Pool mode exposes only the learner sides as envs.
        self.num_envs = self.num_pairs if self._opp_enabled else self.num_sides
        self.level = int(max(0, min(int(level), 20)))
        self.speed_setting = int(max(0, min(int(speed_setting), 2)))
        self.rng_randomize = bool(randomize_rng)
        self.garbage_reward_coef = float(garbage_reward_coef)
        self.clear_win_bonus = float(clear_win_bonus)
        self.virus_progress_reward_coef = max(0.0, float(virus_progress_reward_coef))
        self.decision_penalty = max(0.0, float(decision_penalty))
        self.match_horizon_pills = max(0, int(match_horizon_pills))
        self.horizon_penalty = max(0.0, float(horizon_penalty))
        self.seed_provider = seed_provider
        self.frame_counter_provider = frame_counter_provider
        self.direct_policy_batch = bool(direct_policy_batch)

        state_repr_norm = str(state_repr or "").strip().lower().replace("-", "_")
        if state_repr_norm not in {"bitplane_bottle_conn_mask", "bitplane_bottle_conn_mask_vs"}:
            raise ValueError(
                "cpp-vs-pool backend only supports state_repr="
                "bitplane_bottle_conn_mask[_vs]."
            )
        # Opponent-board observability gate: the `_vs` representation appends
        # the opponent bottle planes (8..15) ahead of the feasible planes.
        self.opponent_obs = state_repr_norm == "bitplane_bottle_conn_mask_vs"
        human_opponents = list(opp_cfg.get("human_opponents") or [])
        afterstate_opponents = list(opp_cfg.get("afterstate_opponents") or [])
        if (human_opponents or afterstate_opponents) and not self.opponent_obs:
            raise ValueError(
                "semantic human/afterstate opponents require state_repr="
                "bitplane_bottle_conn_mask_vs"
            )
        ram_specs.set_state_representation(state_repr_norm)
        obs_channels = 20 if self.opponent_obs else 12
        self._feas_ch = 16 if self.opponent_obs else 8

        self.single_action_space = spaces.Discrete(MACRO_ACTIONS)
        self.single_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_channels, GRID_H, GRID_W), dtype=np.float32
        )
        self.observation_space = self.single_observation_space

        self._rng = np.random.default_rng(None)

        # Start-state bank (Go-Exploit): a fraction of episode resets start
        # from mid-game human-corpus positions instead of clean level starts.
        # Disabled (None) leaves the reset path bit-identical: no extra RNG
        # draws happen unless a bank is loaded.
        self._start_bank = None
        self._start_bank_fraction = 0.0
        bank_cfg = dict(start_bank_cfg or {})
        if bool(bank_cfg.get("enabled", False)):
            from drmc_rl.training.envs.start_bank import StartBank

            path = bank_cfg.get("path") or "runs/start_bank/start_bank_v1.npz"
            self._start_bank = StartBank(path)
            self._start_bank_fraction = float(bank_cfg.get("fraction", 0.25))

        # Deferred planning: rollout reachability solved on the GPU in one
        # batch per decision wave (bit-exact vs the in-pool CPU planner; see
        # tests/test_vs_gpu_planner.py) instead of per-side CPU BFS inside the
        # native step. On CPU-lean hosts the CPU planner dominates training
        # cost, so this frees most of the pool's compute for frame stepping.
        plan_solver = None
        if bool(gpu_planner):
            plan_solver = _make_gpu_plan_solver(int(max_lock_frames))

        self._runner = DrMarioVsPoolRunner(
            num_pairs=self.num_pairs,
            max_lock_frames=int(max_lock_frames),
            max_wait_frames=int(max_wait_frames),
            volley_capacity=max(64, 8 * self.num_pairs),
            lib_path=lib_path,
            plan_solver=plan_solver,
        )

        # Views (no copies) into runner-owned buffers.
        self._mask_1d = self._runner.buffers.feasible_mask
        self._mask = self._mask_1d.reshape(self.num_sides, 4, GRID_H, GRID_W)
        self._cost_1d = self._runner.buffers.cost_to_lock
        self._cost = self._cost_1d.reshape(self.num_sides, 4, GRID_H, GRID_W)

        # Python-built observations (the VS pool does not emit obs natively).
        self._obs = np.zeros((self.num_sides, obs_channels, GRID_H, GRID_W), dtype=np.float32)
        # Side index of each side's opponent (pair partner).
        self._opp_idx = np.arange(self.num_sides, dtype=np.int64) ^ 1

        # Persistent per-env infos (updated in-place).
        self._infos: List[Dict[str, Any]] = [dict() for _ in range(self.num_sides)]

        # Autoreset (NEXT_STEP semantics; per pair).
        self._pending_reset = np.zeros((self.num_pairs,), dtype=bool)

        # Per-side episode stats.
        self._ep_return = np.zeros((self.num_sides,), dtype=np.float64)
        self._ep_frames = np.zeros((self.num_sides,), dtype=np.int64)
        self._ep_decisions = np.zeros((self.num_sides,), dtype=np.int64)
        self._ep_pills = np.zeros((self.num_sides,), dtype=np.int64)
        self._ep_viruses_cleared = np.zeros((self.num_sides,), dtype=np.int64)
        self._human_recent_actions = np.full((self.num_sides, 4), -1, dtype=np.int32)
        self._human_recent_tau = np.zeros((self.num_sides, 4), dtype=np.float32)
        self._viruses_prev = np.full((self.num_sides,), -1, dtype=np.int32)
        self._viruses_initial = np.zeros((self.num_sides,), dtype=np.int32)
        self._garbage_sent_prev = np.zeros((self.num_sides,), dtype=np.int64)
        self._volleys_sent_round = np.zeros((self.num_sides,), dtype=np.int64)
        self._volleys_recv_round = np.zeros((self.num_sides,), dtype=np.int64)
        self._garbage_recv_round = np.zeros((self.num_sides,), dtype=np.int64)

        # Pair-clock tracking (emulated time, for vs/cpm).
        self._pair_clock_prev = np.zeros((self.num_pairs,), dtype=np.int64)

        # Need-action snapshot from the last reset/step call.
        self._need_action = np.zeros((self.num_sides,), dtype=np.uint8)
        self._last_tau = np.ones((self.num_sides,), dtype=np.int32)

        # Rolling VS metrics (dashboard contract: tools/vs_dashboard.py).
        self._matches_total = 0
        self._win_p1 = deque(maxlen=100)  # 1.0 P1 win, 0.0 P1 loss, 0.5 draw
        self._draws = deque(maxlen=100)
        self._match_len_sec = deque(maxlen=100)
        self._clear_wins = deque(maxlen=100)
        self._topout_wins = deque(maxlen=100)
        self._horizon_matches = deque(maxlen=100)
        self._pills_per_match = deque(maxlen=100)
        self._garbage_per_match = deque(maxlen=100)
        self._viruses_per_match = deque(maxlen=100)
        self._volleys_total = 0
        self._emu_frames_total = 0
        # Per-side completed-game feature records for the skill grader.
        self._skill_games: List[Dict[str, float]] = []

        # Frozen-opponent pool (PFSP) state.
        self._pair_opponents: List[Optional[Any]] = [None] * self.num_pairs
        self._aux_shim: Optional[Any] = None
        self._snapshot_every = 0
        self._matches_since_snapshot = 0
        # Gate-best snapshot protection (anti-collapse): a snapshot only
        # enters the pool while the policy is healthy, so a collapsing
        # learner cannot poison its own opponent distribution (the lock-in
        # mechanism of the 2026-07-02 vs6tf_03 mutual-fast-topout collapse:
        # match_len_p50 224s -> 14s, then the pool filled with collapsed
        # selves and nothing punished the exploit). Blocked snapshots retry
        # at the next hook, so recovery resumes snapshotting automatically.
        self._snapshot_gate_cfg: Dict[str, float] = {}
        self._league_wr_best = 0.0
        self._snapshots_blocked = 0
        if self._opp_enabled:
            from drmc_rl.training.envs.vs_opponents import (
                OpponentPool,
                default_seed_paths,
                parse_league_config,
            )

            league = parse_league_config(opp_cfg.get("league"))
            # Frozen-opponent action temperature. 0 = argmax (eval-matched
            # strength). The historical default was full sampling (temp 1.0),
            # which makes soft cross-entropy BC nets play far below their
            # argmax strength — the learner then trains against weak BC but is
            # graded against argmax BC. Default to argmax.
            self._opp_temperature = float(opp_cfg.get("opponent_temperature", 0.0))
            self._snapshot_every = int(opp_cfg.get("snapshot_every_matches", 400))
            gate = opp_cfg.get("snapshot_gate")
            if isinstance(gate, dict) and bool(gate.get("enabled", False)):
                self._snapshot_gate_cfg = {
                    "min_match_len_sec": float(gate.get("min_match_len_sec", 60.0)),
                    "league_wr_frac": float(gate.get("league_wr_frac", 0.5)),
                    "league_wr_min_games": float(gate.get("league_wr_min_games", 50)),
                }
            if league.mode == "exploiter":
                # The exploiter trains exclusively vs the listed main agents;
                # it never freezes snapshots of itself into the pool.
                self._snapshot_every = 0
            pool = opp_cfg.get("pool")  # injected pool (tests)
            if pool is None:
                pool = OpponentPool(
                    opp_cfg.get("dir") or "runs/opponent_pool",
                    max_pool=int(opp_cfg.get("max_pool", 12)),
                    league=league,
                    device=str(opp_cfg.get("device", "auto")),
                )
                if human_opponents:
                    pool.seed_humans(human_opponents)
                if afterstate_opponents:
                    pool.seed_afterstates(afterstate_opponents)
                if league.mode != "exploiter" and not pool.entries:
                    seed_paths = opp_cfg.get("seed_paths") or default_seed_paths()
                    if not seed_paths:
                        raise RuntimeError(
                            "opponent_pool enabled but no seed checkpoints found "
                            "(runs/vs2_*/checkpoints, runs/best_agents)."
                        )
                    from drmc_rl.training.envs.vs_opponents import CHAMPION_CHECKPOINT

                    protected = [Path(p).name == CHAMPION_CHECKPOINT.name for p in seed_paths]
                    pool.seed(seed_paths, protected=protected)
                if league.main_agents:
                    pool.seed_league_targets(league.main_agents)
            elif "league" in opp_cfg:
                pool.league = league
            if not pool.entries:
                raise RuntimeError("opponent_pool enabled but the pool is empty.")
            self._opp_pool = pool

    # ------------------------------------------------------------------ vector API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        _ = options

        specs = self._build_reset_specs(reset_mask=None)
        self._runner.reset(None, specs)

        self._pending_reset.fill(False)
        self._ep_return.fill(0.0)
        self._ep_frames.fill(0)
        self._ep_decisions.fill(0)
        self._ep_pills.fill(0)
        self._ep_viruses_cleared.fill(0)
        self._human_recent_actions.fill(-1)
        self._human_recent_tau.fill(0.0)
        self._garbage_sent_prev.fill(0)
        self._volleys_sent_round.fill(0)
        self._volleys_recv_round.fill(0)
        self._garbage_recv_round.fill(0)
        self._pair_clock_prev = self._pair_clocks().astype(np.int64)
        self._emu_frames_total += int(self._pair_clock_prev.sum())

        v_now = self._runner.buffers.viruses_rem.astype(np.int32, copy=False)
        self._viruses_prev = v_now.copy()
        self._viruses_initial = v_now.copy()
        self._need_action = self._runner.buffers.need_action.copy()
        self._last_tau.fill(1)

        self._build_obs_in_place()
        self._apply_symmetry_reduction_in_place()
        if self.opponent_obs:
            self._fill_opponent_planes_in_place()
        if self.direct_policy_batch:
            for info in self._infos:
                info.clear()
        else:
            self._refresh_infos(step_tau_raw=np.zeros((self.num_sides,), dtype=np.uint32))

        if self._opp_pool is not None:
            for pair_i in range(self.num_pairs):
                self._pair_opponents[pair_i] = self._opp_pool.sample(self._rng)
            return self._obs[0::2], [self._infos[i] for i in range(0, self.num_sides, 2)]
        return self._obs, list(self._infos)

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self._opp_pool is not None:
            learner_acts = np.asarray(list(actions), dtype=np.int32).reshape(self.num_pairs)
            acts = np.empty((self.num_sides,), dtype=np.int32)
            acts[0::2] = learner_acts
            acts[1::2] = self._opponent_actions()
        else:
            acts = np.asarray(list(actions), dtype=np.int32).reshape(self.num_sides)

        # Only act for sides parked at a decision boundary; leave others parked.
        # (-1 from the policy's empty-mask fallback becomes noop-fall, which is
        # exactly the intended escape hatch when the feasible mask is empty.)
        send = acts.copy()
        send[self._need_action == 0] = -2
        was_need_action = self._need_action.astype(bool)

        reset_mask = self._pending_reset.astype(np.uint8, copy=True)
        reset_specs = None
        if bool(reset_mask.any()):
            reset_specs = self._build_reset_specs(reset_mask=reset_mask)

        self._runner.step(send, reset_mask if reset_specs is not None else None, reset_specs)

        buf = self._runner.buffers
        tau_raw = buf.tau_frames.astype(np.uint32, copy=False)
        np.maximum(tau_raw, 1, out=self._last_tau, casting="unsafe")
        invalid = buf.invalid_action.astype(np.int32, copy=False)
        term_pair = buf.terminated.astype(np.uint8, copy=False)
        trunc_pair = buf.truncated.astype(np.uint8, copy=False)
        outcome = buf.outcome.astype(np.uint8, copy=False)
        garbage_sent = buf.garbage_sent_total.astype(np.int64, copy=False)
        v_now = buf.viruses_rem.astype(np.int32, copy=False)
        self._need_action = buf.need_action.copy()

        # Volley events from this call.
        volleys = self._runner.volleys()
        for v in volleys:
            recv_gi = v.pair * 2 + v.receiver
            sent_gi = v.pair * 2 + (1 - v.receiver)
            self._volleys_recv_round[recv_gi] += 1
            self._garbage_recv_round[recv_gi] += int(v.size)
            self._volleys_sent_round[sent_gi] += 1
            self._volleys_total += 1

        # Emulated (pair-clock) time accounting.
        pair_clocks = self._pair_clocks().astype(np.int64)
        was_reset_pair = reset_mask.astype(bool)
        clock_delta = pair_clocks - self._pair_clock_prev
        clock_delta[was_reset_pair] = pair_clocks[was_reset_pair]  # clock restarts at 0 on reset
        self._emu_frames_total += int(np.maximum(clock_delta, 0).sum())
        self._pair_clock_prev = pair_clocks

        rewards = np.zeros((self.num_sides,), dtype=np.float32)
        terminated = np.zeros((self.num_sides,), dtype=bool)
        truncated = np.zeros((self.num_sides,), dtype=bool)
        garbage_delta_arr = np.zeros((self.num_sides,), dtype=np.int64)
        virus_delta_arr = np.zeros((self.num_sides,), dtype=np.int64)
        accepted_arr = np.zeros((self.num_sides,), dtype=bool)
        outcome_str: List[str] = ["" for _ in range(self.num_sides)]
        horizon_pair = np.zeros(self.num_pairs, dtype=bool)

        for i in range(self.num_sides):
            pair_i = i // 2
            is_reset = bool(reset_mask[pair_i] != 0)

            tau_i_raw = int(tau_raw[i])
            self._ep_decisions[i] += 1
            self._ep_frames[i] += int(max(1, tau_i_raw))

            v_prev = int(self._viruses_prev[i])
            v_i = int(v_now[i])
            if v_prev >= 0:
                self._ep_viruses_cleared[i] += max(0, v_prev - v_i)
            self._viruses_prev[i] = int(v_i)

            if is_reset:
                # NEXT_STEP autoreset: action ignored, return reset state.
                if i % 2 == 0:
                    self._pending_reset[pair_i] = False
                self._garbage_sent_prev[i] = 0
                self._volleys_sent_round[i] = 0
                self._volleys_recv_round[i] = 0
                self._garbage_recv_round[i] = 0
                self._viruses_initial[i] = int(v_i)
                self._ep_viruses_cleared[i] = 0
                self._human_recent_actions[i].fill(-1)
                self._human_recent_tau[i].fill(0.0)
                continue

            accepted = bool(was_need_action[i]) and int(invalid[i]) == -1 and int(acts[i]) >= -1
            accepted_arr[i] = accepted
            if accepted:
                self._ep_pills[i] += 1
                self._human_recent_actions[i, 1:] = self._human_recent_actions[i, :-1]
                self._human_recent_tau[i, 1:] = self._human_recent_tau[i, :-1]
                self._human_recent_actions[i, 0] = int(acts[i])
                self._human_recent_tau[i, 0] = float(tau_i_raw)

            g_delta = int(garbage_sent[i]) - int(self._garbage_sent_prev[i])
            g_delta = max(0, g_delta)
            self._garbage_sent_prev[i] = int(garbage_sent[i])
            garbage_delta_arr[i] = int(g_delta)

            delta_v = max(0, v_prev - v_i) if v_prev >= 0 else 0
            virus_delta_arr[i] = int(delta_v)
            r = (
                float(self.garbage_reward_coef) * float(g_delta)
                + float(self.virus_progress_reward_coef) * float(delta_v)
                - (float(self.decision_penalty) if accepted else 0.0)
            )

            pair_done = bool(int(term_pair[pair_i]) != 0)
            if pair_done:
                oc = int(outcome[i])
                if oc == VS_OUTCOME_WIN:
                    r += 1.0
                    # Clear-win incentive: extra terminal bonus when the win
                    # came from clearing all own viruses (not opponent topout).
                    # Keeps the win > loss ordering for any bonus >= 0; only
                    # reorders HOW to win.
                    if self.clear_win_bonus and int(v_now[i]) == 0:
                        r += float(self.clear_win_bonus)
                    outcome_str[i] = "win"
                elif oc == VS_OUTCOME_LOSS:
                    r -= 1.0
                    outcome_str[i] = "loss"
                elif oc == VS_OUTCOME_DRAW:
                    outcome_str[i] = "draw"
                if int(trunc_pair[pair_i]) != 0 and oc == 0:
                    truncated[i] = True
                    outcome_str[i] = "timeout"
                else:
                    terminated[i] = True
            rewards[i] = r

        # A human-calibrated unresolved-game horizon prevents survival loops
        # from occupying rollout lanes indefinitely. It is a truncation (not a
        # fabricated win/loss); both sides receive the same optional urgency
        # penalty and the value function may bootstrap normally.
        if self.match_horizon_pills:
            for pair_i in range(self.num_pairs):
                i0, i1 = pair_i * 2, pair_i * 2 + 1
                if int(term_pair[pair_i]) != 0:
                    continue
                if max(int(self._ep_pills[i0]), int(self._ep_pills[i1])) < self.match_horizon_pills:
                    continue
                horizon_pair[pair_i] = True
                truncated[i0] = truncated[i1] = True
                outcome_str[i0] = outcome_str[i1] = "horizon"
                rewards[i0] -= float(self.horizon_penalty)
                rewards[i1] -= float(self.horizon_penalty)

        self._build_obs_in_place()
        self._apply_symmetry_reduction_in_place()
        if self.opponent_obs:
            self._fill_opponent_planes_in_place()
        if self.direct_policy_batch:
            for info in self._infos:
                info.clear()
        else:
            self._refresh_infos(step_tau_raw=tau_raw)
            for i in range(self.num_sides):
                info_i = self._infos[i]
                r_env = float(rewards[i])
                info_i["vs/garbage_sent_delta"] = int(garbage_delta_arr[i])
                info_i["vs/outcome"] = str(outcome_str[i])
                info_i["r_env"] = r_env
                info_i["r_shape"] = float(self.garbage_reward_coef) * float(
                    garbage_delta_arr[i]
                )
                info_i["r_total"] = r_env
                info_i["reward/r_env"] = r_env
                info_i["reward/r_shape"] = info_i["r_shape"]
                info_i["reward/r_total"] = r_env
                info_i["reward/virus_progress"] = float(
                    self.virus_progress_reward_coef * int(virus_delta_arr[i])
                )
                info_i["reward/decision_penalty"] = (
                    -float(self.decision_penalty) if bool(accepted_arr[i]) else 0.0
                )
                opponent_i = (i // 2) * 2 + (1 - i % 2)
                info_i["cleared"] = bool(
                    outcome_str[i] == "win" and int(v_now[i]) == 0
                )
                info_i["topout"] = bool(
                    outcome_str[i] == "loss" and int(v_now[opponent_i]) != 0
                )
                info_i["goal_achieved"] = bool(outcome_str[i] == "win")
                info_i["terminal_reason"] = str(outcome_str[i])

        self._ep_return += rewards.astype(np.float64)

        # Finalize episodes + match-level metrics; mark pending resets.
        for pair_i in range(self.num_pairs):
            i0, i1 = pair_i * 2, pair_i * 2 + 1
            if not bool(terminated[i0] or truncated[i0] or terminated[i1] or truncated[i1]):
                continue
            self._record_match(
                pair_i, outcome, pair_clocks[pair_i], horizon=bool(horizon_pair[pair_i])
            )
            for i in (i0, i1):
                info_i = self._infos[i]
                info_i["episode"] = {
                    "r": float(self._ep_return[i]),
                    "l": int(self._ep_frames[i]),
                    "decisions": int(self._ep_decisions[i]),
                }
                info_i["drm"] = {
                    "viruses_cleared": int(self._ep_viruses_cleared[i]),
                    "viruses_remaining": int(v_now[i]),
                    "viruses_initial": int(self._viruses_initial[i]),
                    "top_out": bool(info_i.get("topout", False)),
                    "cleared": bool(info_i.get("cleared", False)),
                    "level": int(self.level),
                    "speed_setting": int(self.speed_setting),
                    "vs_outcome": str(outcome_str[i]),
                    "garbage_sent": int(self._garbage_sent_prev[i]),
                }
                self._ep_return[i] = 0.0
                self._ep_frames[i] = 0
                self._ep_decisions[i] = 0
                self._ep_pills[i] = 0
                self._ep_viruses_cleared[i] = 0
            self._pending_reset[pair_i] = True

            if self._opp_pool is not None:
                entry = self._pair_opponents[pair_i]
                if entry is not None:
                    oc0 = int(outcome[i0])
                    result = 1.0 if oc0 == VS_OUTCOME_WIN else (0.0 if oc0 == VS_OUTCOME_LOSS else 0.5)
                    self._opp_pool.record(entry.id, result)
                    self._infos[i0]["vs/opponent_id"] = str(entry.id)
                self._matches_since_snapshot += 1
                # PFSP resample for the pair's next episode.
                self._pair_opponents[pair_i] = self._opp_pool.sample(self._rng)

        if self._opp_pool is not None:
            sel = np.arange(0, self.num_sides, 2)
            return (
                self._obs[0::2],
                rewards[sel].astype(np.float32),
                terminated[sel],
                truncated[sel],
                [self._infos[i] for i in sel],
            )
        return self._obs, rewards.astype(np.float32), terminated, truncated, list(self._infos)

    def close(self) -> None:
        if self._opp_pool is not None:
            self._opp_pool.close()
        if hasattr(self._runner, "close"):
            self._runner.close()

    def transition_tau(self) -> np.ndarray:
        idx = slice(0, None, 2) if self._opp_pool is not None else slice(None)
        return self._last_tau[idx]

    def policy_batch(self, aux_spec: str = "none") -> PlacementPolicyBatch:
        """Return policy inputs as arrays, bypassing per-environment dictionaries."""

        idx = np.arange(0, self.num_sides, 2) if self._opp_pool is not None else np.arange(
            self.num_sides
        )
        aux = None if aux_spec == "none" else self._build_direct_aux(idx, aux_spec)
        buf = self._runner.buffers
        return PlacementPolicyBatch(
            feasible_mask=self._mask[idx],
            cost_to_lock=self._cost[idx],
            pill_colors=buf.pill_colors[idx].astype(np.int64),
            preview_pill_colors=buf.preview_colors[idx].astype(np.int64),
            aux=aux,
        )

    def _build_direct_aux(self, side_idxs: np.ndarray, aux_spec: str) -> np.ndarray:
        """Build aux_v1[_vs] directly from vector state, without info dictionaries."""

        spec = str(aux_spec).strip().lower()
        if spec not in {"v1", "v1_vs"}:
            raise ValueError(f"Unsupported direct aux spec: {aux_spec!r}")
        obs = self._obs[side_idxs]
        batch = int(len(side_idxs))
        dim = 72 if spec == "v1_vs" else 57
        out = np.zeros((batch, dim), dtype=np.float32)
        virus = obs[:, 3] > 0.5
        colors = obs[:, :3] > 0.5
        occupancy = colors.any(axis=1)
        virus_by_color = (colors & virus[:, None]).reshape(batch, 3, -1).sum(axis=2)

        def heights(mask: np.ndarray) -> np.ndarray:
            occupied = mask.any(axis=1)
            first = mask.argmax(axis=1)
            return np.where(occupied, 16 - first, 0).astype(np.float32)

        column_heights = heights(occupancy)
        virus_heights = heights(virus)
        viruses = virus.reshape(batch, -1).sum(axis=1).astype(np.float32)
        options = self._mask_1d[side_idxs].sum(axis=1).astype(np.float32)
        initial = self._viruses_initial[side_idxs].astype(np.float32)
        progress = np.divide(
            initial - viruses,
            initial,
            out=np.zeros_like(initial),
            where=initial > 0,
        )

        k = 0
        out[:, k + self.speed_setting] = 1.0
        k += 3
        out[:, k] = np.clip(viruses / 84.0, 0.0, 1.0)
        k += 1
        out[:, k : k + 3] = np.clip(virus_by_color / 84.0, 0.0, 1.0)
        k += 3
        out[:, k + (self.level + 15)] = 1.0
        k += 36
        out[:, k] = np.tanh(self._ep_frames[side_idxs] / 8000.0)
        k += 1
        out[:, k] = np.clip(column_heights.max(axis=1) / 16.0, 0.0, 1.0)
        k += 1
        out[:, k : k + 8] = np.clip(column_heights / 16.0, 0.0, 1.0)
        k += 8
        out[:, k] = np.clip(progress, 0.0, 1.0)
        k += 1
        out[:, k] = np.clip(options / 512.0, 0.0, 1.0)
        k += 1
        out[:, k] = np.clip(occupancy.reshape(batch, -1).sum(axis=1) / 128.0, 0.0, 1.0)
        k += 1
        out[:, k] = np.clip(virus_heights.max(axis=1) / 16.0, 0.0, 1.0)
        k += 1

        if spec == "v1_vs":
            buf = self._runner.buffers
            opp = side_idxs ^ 1
            out[:, k] = np.clip(buf.viruses_rem[opp].astype(np.float32) / 84.0, 0.0, 1.0)
            k += 1
            out[:, k] = np.clip(buf.garbage_pending[side_idxs].astype(np.float32) / 4.0, 0.0, 1.0)
            k += 1
            out[:, k] = np.clip(buf.garbage_pending[opp].astype(np.float32) / 4.0, 0.0, 1.0)
            k += 1
            for values in (buf.pill_colors[opp], buf.preview_colors[opp]):
                for half in range(2):
                    color = values[:, half].astype(np.int64)
                    valid = (color >= 0) & (color < 3)
                    out[np.nonzero(valid)[0], k + color[valid]] = 1.0
                    k += 3
        if k != dim:
            raise RuntimeError(f"direct aux packing mismatch: {k} != {dim}")
        return out

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        return None

    def set_attr(self, name: str, values: Any, indices: Optional[Sequence[int]] = None) -> None:
        _ = indices
        if isinstance(values, (list, tuple, np.ndarray)) and not isinstance(values, (str, bytes, bytearray)):
            value = values[0] if len(values) else None
        else:
            value = values
        if name == "level":
            self.level = int(max(0, min(int(value), 20)))
            return
        if name == "garbage_reward_coef":
            self.garbage_reward_coef = float(value)
            return
        if name == "virus_progress_reward_coef":
            self.virus_progress_reward_coef = max(0.0, float(value))
            return
        if name in {"rng_randomize", "randomize_rng"}:
            self.rng_randomize = bool(value)
            return
        try:
            setattr(self, name, value)
        except Exception:
            return

    # ------------------------------------------------------------------ VS metrics
    def get_vs_metrics(self) -> Dict[str, float]:
        """Scalar metrics for the dashboard (tools/vs_dashboard.py)."""

        minutes = float(self._emu_frames_total) / NES_FPS / 60.0
        cpm = float(self._volleys_total) / minutes if minutes > 0 else 0.0
        out: Dict[str, float] = {
            "vs/matches_total": float(self._matches_total),
            "vs/opponent_pool": 1.0,
            # Pooled across both sides every sent volley is received by the
            # opponent, so the pooled sent/received rates coincide.
            "vs/cpm": cpm,
            "vs/cpm_received": cpm,
        }
        if self._win_p1:
            # Pool mode: side 0 is always the learner, so this is the learner's
            # rolling win rate vs the frozen pool (meaningful, unlike pure
            # self-play where it hovers at 0.5 by construction).
            out["vs/win_rate"] = float(np.mean(self._win_p1))
        if self._draws:
            out["vs/draw_rate"] = float(np.mean(self._draws))
        if self._match_len_sec:
            out["vs/match_len_p50_sec"] = float(np.median(self._match_len_sec))
        if self._clear_wins:
            out["vs/clear_win_rate"] = float(np.mean(self._clear_wins))
            out["vs/topout_win_rate"] = float(np.mean(self._topout_wins))
            out["vs/horizon_rate"] = float(np.mean(self._horizon_matches))
            out["vs/pills_per_match"] = float(np.mean(self._pills_per_match))
            out["vs/garbage_per_match"] = float(np.mean(self._garbage_per_match))
            out["vs/viruses_cleared_per_match"] = float(np.mean(self._viruses_per_match))
        if self._snapshot_gate_cfg:
            out["vs/snapshots_blocked"] = float(self._snapshots_blocked)
        if self._opp_pool is not None:
            out["vs/opponent_pool"] = float(len(self._opp_pool.entries))
            rates = [float(e.wins) / float(e.games) for e in self._opp_pool.entries if int(e.games) > 0]
            if rates:
                out["vs/pool_winrate_min"] = float(min(rates))
                out["vs/pool_winrate_max"] = float(max(rates))
            league = self._opp_pool.league
            if league.mode != "pfsp":
                targets = self._opp_pool.league_targets()
                out["vs/league_targets"] = float(len(targets))
                played = [t for t in targets if int(t.games) > 0]
                if played:
                    # Cumulative learner win rate vs the fixed targets
                    # (per-target counts persist in the pool manifest).
                    t_rates = [float(t.wins) / float(t.games) for t in played]
                    total_games = sum(int(t.games) for t in played)
                    out["vs/league_win_rate"] = float(
                        sum(float(t.wins) for t in played) / float(total_games)
                    )
                    out["vs/league_win_rate_min"] = float(min(t_rates))
                    out["vs/league_win_rate_max"] = float(max(t_rates))
                    for t, r in zip(played, t_rates):
                        out[f"vs/league_wr_{t.id}"] = float(r)
        return out

    def maybe_snapshot(
        self,
        state_dict_getter: Callable[[], Dict[str, Any]],
        *,
        cfg: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> bool:
        """Freeze the current (EMA) weights into the opponent pool when due.

        Called from the trainer's metrics hook; a snapshot is taken every
        ``snapshot_every_matches`` completed learner matches.
        """

        if self._opp_pool is None or self._snapshot_every <= 0:
            return False
        if self._matches_since_snapshot < self._snapshot_every:
            return False
        if not self._snapshot_healthy():
            self._snapshots_blocked += 1
            return False
        self._opp_pool.snapshot(state_dict_getter(), dict(cfg or {}), int(step))
        self._matches_since_snapshot = 0
        return True

    def _snapshot_healthy(self) -> bool:
        """Gate-best check: block pool admission while the policy looks
        collapsed. Fast trigger: rolling median match length under the floor
        (the topout equilibrium shows up here within one window). Slow
        backstop: cumulative league-anchor win rate falling below a fraction
        of its own best."""

        gate = self._snapshot_gate_cfg
        if not gate:
            return True
        m = self.get_vs_metrics()
        ml = m.get("vs/match_len_p50_sec")
        if ml is not None and float(ml) < gate["min_match_len_sec"]:
            return False
        lw = m.get("vs/league_win_rate")
        games = sum(
            int(t.games) for t in self._opp_pool.league_targets()
        ) if self._opp_pool is not None else 0
        if lw is not None and games >= gate["league_wr_min_games"]:
            self._league_wr_best = max(self._league_wr_best, float(lw))
            if float(lw) < gate["league_wr_frac"] * self._league_wr_best:
                return False
        return True

    def skill_games_pending(self) -> int:
        return len(self._skill_games)

    def pop_skill_games(self) -> List[Dict[str, float]]:
        games = self._skill_games
        self._skill_games = []
        return games

    # ------------------------------------------------------------------ internals
    def _pair_clocks(self) -> np.ndarray:
        sf = self._runner.buffers.side_frames.reshape(self.num_pairs, 2)
        return sf.max(axis=1)

    # ------------------------------------------------------------ frozen opponents
    def _get_aux_shim(self, aux_spec: str = "v1") -> Any:
        """Aux builder shim per spec (same trick as tools/eval_policy)."""

        if self._aux_shim is None:
            self._aux_shim = {}
        spec = str(aux_spec or "v1")
        shim = self._aux_shim.get(spec)
        if shim is None:
            from drmc_rl.training.algo.ppo_smdp import _AUX_DIM_BY_SPEC, SMDPPPOAdapter

            shim = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
            shim.aux_spec = spec
            shim.aux_dim = int(_AUX_DIM_BY_SPEC[spec])
            self._aux_shim[spec] = shim
        return shim

    def _human_opponent_aux(self, entry: Any, side_idxs: np.ndarray) -> np.ndarray:
        """Semantic v2 conditioning for fixed-rating human pool entries."""

        from drmc_rl.human.model import POLICY_CONDITION_DIM, policy_condition_features

        rating = float(entry.rating if entry.rating is not None else entry.condition.mean)
        out = np.zeros((len(side_idxs), POLICY_CONDITION_DIM), dtype=np.float32)
        for row, side in enumerate(side_idxs):
            recent = [
                {"action": int(action), "tau_frames": float(tau)}
                for action, tau in zip(
                    self._human_recent_actions[int(side)],
                    self._human_recent_tau[int(side)],
                )
                if int(action) >= 0
            ]
            out[row] = policy_condition_features(
                entry.condition,
                rating=rating,
                rating_sd=float(entry.rating_sd),
                opponent_rating=rating,
                opponent_rating_sd=float(entry.rating_sd),
                opponent_state_age_frames=0,
                game_phase=min(float(self._ep_pills[int(side)]) / 100.0, 1.0),
                recent_decisions=recent,
            )
        return out

    def _opponent_aux(self, entry: Any, side_idxs: np.ndarray) -> np.ndarray:
        if getattr(entry, "kind", "rl") == "human":
            return self._human_opponent_aux(entry, side_idxs)
        return self._build_direct_aux(side_idxs, getattr(entry, "aux_spec", "v1"))

    def _opponent_actions(self) -> np.ndarray:
        """Macro actions for the P2 sides, from their assigned frozen nets.

        Sides not parked at a decision (or pending reset) stay parked (-2).
        Forwards are batched per opponent entry.
        """

        acts = np.full((self.num_pairs,), -2, dtype=np.int32)
        groups: Dict[str, List[int]] = {}
        entries: Dict[str, Any] = {}
        for pair_i in range(self.num_pairs):
            gi = pair_i * 2 + 1
            if bool(self._pending_reset[pair_i]) or int(self._need_action[gi]) == 0:
                continue
            entry = self._pair_opponents[pair_i]
            if entry is None:
                acts[pair_i] = -1  # noop-fall escape hatch
                continue
            groups.setdefault(entry.id, []).append(gi)
            entries[entry.id] = entry
        if not groups:
            return acts
        # V3 owns an exact native afterstate simulator and exposes multiple
        # semantically distinct heads.  Score every parked side for one entry
        # in a single simulation/network batch, then leave ordinary frozen
        # candidate nets on the existing optimized path.
        afterstate_ids = [
            entry_id
            for entry_id in groups
            if getattr(entries[entry_id], "kind", "rl") == "afterstate"
        ]
        for entry_id in afterstate_ids:
            side_idxs = groups.pop(entry_id)
            chosen = self._afterstate_opponent_actions(entries[entry_id], side_idxs)
            for gi, action in zip(side_idxs, chosen):
                acts[gi // 2] = int(action)
        if not groups:
            return acts
        # Fast path (all-CUDA pool): ONE host->device upload for every parked
        # opponent side, per-entry forwards on device-side slices, GPU argmax,
        # and one tiny device->host copy of the chosen slots. The old
        # per-group tensor builds + copy-backs serialized the wave on GPU
        # round trips (the dominant wave cost by far).
        devices = {getattr(entries[eid], "device", "cpu") for eid in groups}
        temp = float(getattr(self, "_opp_temperature", 0.0))
        if devices == {"cuda"} and temp <= 0.0:
            for gi, action in self._opponent_actions_cuda(groups, entries):
                acts[gi // 2] = int(action)
            return acts
        launched = [
            (side_idxs, self._forward_opponent_launch(entries[entry_id], side_idxs))
            for entry_id, side_idxs in groups.items()
        ]
        for side_idxs, pending in launched:
            chosen = self._sample_opponent_actions(*pending)
            for k, gi in enumerate(side_idxs):
                acts[gi // 2] = int(chosen[k])
        return acts

    def _afterstate_opponent_actions(self, entry: Any, side_idxs: List[int]) -> np.ndarray:
        """Exact V3 actions for one opponent entry, losslessly batched."""

        from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates_batch

        sides = np.asarray(side_idxs, dtype=np.int64)
        packed = pack_feasible_candidates_batch(
            self._mask[sides].astype(bool, copy=False),
            self._cost[sides],
            max_candidates=int(entry.candidate_max),
            sort_by_cost=True,
        )
        rating = float(
            entry.rating if entry.rating is not None else entry.runtime.condition.mean
        )
        requests = []
        for row, side in enumerate(sides):
            recent = [
                {"action": int(action), "tau_frames": float(tau)}
                for action, tau in zip(
                    self._human_recent_actions[int(side)],
                    self._human_recent_tau[int(side)],
                )
                if int(action) >= 0
            ]
            requests.append(
                {
                    "board_planes": self._obs[int(side), :8],
                    "opponent_board_planes": self._obs[int(side), 8:16],
                    "opponent_state_age_frames": 0,
                    "rating_sd": float(entry.rating_sd),
                    "opponent_rating": rating,
                    "opponent_rating_sd": float(entry.rating_sd),
                    "game_phase": min(float(self._ep_pills[int(side)]) / 100.0, 1.0),
                    "recent_decisions": recent,
                    "pill": self._runner.buffers.pill_colors[int(side)],
                    "preview": self._runner.buffers.preview_colors[int(side)],
                    "candidate_actions": packed.actions[row],
                    "candidate_costs": packed.cost[row],
                    "candidate_mask": packed.mask[row],
                    "rating": rating,
                    "speed": self.speed_setting,
                    # Native play increments speed every ten placed pills.
                    "speed_ups": int(self._ep_pills[int(side)]) // 10,
                }
            )
        results = entry.runtime.score_batch(requests)
        selected = np.empty(len(sides), dtype=np.int32)
        for row, result in enumerate(results):
            if entry.selection == "style":
                slot = entry.runtime.choose_style(
                    result["human_logits"], packed.mask[row], temperature=0.0
                )
            else:
                slot = entry.runtime.choose_quality(
                    result["competitive_score"], packed.mask[row]
                )
            selected[row] = int(packed.actions[row, slot])
        return selected

    def _opponent_actions_cuda(self, groups: Dict[str, List[int]], entries: Dict[str, Any]):
        """Deterministic opponent actions with a single H2D and a single D2H.

        Numerically identical to the per-group path (same nets, same inputs,
        same argmax over the same -1e9-masked logits); only the transfer
        schedule differs.
        """

        import torch

        from drmc_rl.models.policy.candidate_packing import (
            pack_feasible_candidates_tensor_batch,
        )

        order: List[int] = []
        spans: List[Tuple[Any, int, int]] = []
        for entry_id, side_idxs in groups.items():
            spans.append((entries[entry_id], len(order), len(order) + len(side_idxs)))
            order.extend(side_idxs)
        n = len(order)

        buf = self._opp_staging(n)
        buf["obs"][:n] = self._obs[order]
        buf["pills"][:n] = self._runner.buffers.pill_colors[order]
        buf["previews"][:n] = self._runner.buffers.preview_colors[order]
        buf["feasible"][:n] = self._mask[order]
        buf["raw_cost"][:n] = self._cost[order]
        aux_rows = [
            (k, gi, entry)
            for entry, lo, hi in spans
            if int(entry.aux_dim) > 0
            for k, gi in zip(range(lo, hi), order[lo:hi])
        ]
        if aux_rows:
            for entry, lo, hi in spans:
                if int(entry.aux_dim) > 0:
                    rows = self._opponent_aux(
                        entry, np.asarray(order[lo:hi], dtype=np.int64)
                    )
                    buf["aux"][lo:hi, : rows.shape[1]] = rows

        candidate_cap = max(int(entry.candidate_max) for entry, _lo, _hi in spans)
        packed_width = candidate_bucket_width(
            buf["feasible"][:n], max_candidates=candidate_cap
        )
        dev = {}
        for name in ("obs", "pills", "previews", "feasible", "raw_cost", "aux"):
            dev[name] = buf[f"{name}_t"][:n].to("cuda", non_blocking=True)
        packed = pack_feasible_candidates_tensor_batch(
            dev["feasible"], dev["raw_cost"], max_candidates=packed_width
        )

        slots = torch.empty(n, dtype=torch.int64, device="cuda")
        with torch.inference_mode():
            for entry, lo, hi in spans:
                aux = dev["aux"][lo:hi, : int(entry.aux_dim)] if int(entry.aux_dim) > 0 else None
                logits, _values = entry.net(
                    dev["obs"][lo:hi],
                    dev["pills"][lo:hi],
                    dev["previews"][lo:hi],
                    packed.actions[lo:hi, : int(entry.candidate_max)],
                    packed.cost[lo:hi, : int(entry.candidate_max)],
                    packed.mask[lo:hi, : int(entry.candidate_max)],
                    aux=aux,
                )
                # logits are already -1e9 on masked slots -> plain argmax.
                slots[lo:hi] = logits.argmax(dim=1)
        chosen = packed.actions.gather(1, slots.unsqueeze(1)).squeeze(1).cpu().numpy()
        # Empty mask -> slot 0 -> padding action -1 (noop-fall), as before.
        return [(gi, int(chosen[k])) for k, gi in enumerate(order)]

    def _opp_staging(self, n: int) -> Dict[str, Any]:
        """Pinned staging buffers (numpy views + torch pinned tensors)."""

        import torch

        cur = getattr(self, "_opp_staging_buf", None)
        aux_max = 72
        if cur is not None and cur["obs"].shape[0] >= n:
            return cur
        cap = max(self.num_sides, n)
        C = self._obs.shape[1]
        t = {
            "obs_t": torch.empty((cap, C, GRID_H, GRID_W), dtype=torch.float32).pin_memory(),
            "pills_t": torch.empty((cap, 2), dtype=torch.int64).pin_memory(),
            "previews_t": torch.empty((cap, 2), dtype=torch.int64).pin_memory(),
            "feasible_t": torch.empty(
                (cap, 4, GRID_H, GRID_W), dtype=torch.bool
            ).pin_memory(),
            "raw_cost_t": torch.empty(
                (cap, 4, GRID_H, GRID_W), dtype=torch.float32
            ).pin_memory(),
            "aux_t": torch.zeros((cap, aux_max), dtype=torch.float32).pin_memory(),
        }
        buf = {k[:-2]: v.numpy() for k, v in t.items()}
        buf.update(t)
        self._opp_staging_buf = buf
        return buf

    def _forward_opponent(self, entry: Any, side_idxs: List[int]) -> np.ndarray:
        """Sample macro actions for `side_idxs` from one frozen opponent net."""

        return self._sample_opponent_actions(*self._forward_opponent_launch(entry, side_idxs))

    def _forward_opponent_launch(self, entry: Any, side_idxs: List[int]):
        """Queue one entry-group forward; no host synchronization."""

        import torch

        from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates_batch

        obs = np.ascontiguousarray(self._obs[side_idxs], dtype=np.float32)
        pills = self._runner.buffers.pill_colors[side_idxs].astype(np.int64)
        previews = self._runner.buffers.preview_colors[side_idxs].astype(np.int64)

        kmax = int(entry.candidate_max)
        packed = pack_feasible_candidates_batch(
            self._mask[side_idxs].astype(bool, copy=False),
            self._cost[side_idxs],
            max_candidates=kmax,
            sort_by_cost=True,
        )
        cand_actions = packed.actions
        cand_mask = packed.mask
        cand_cost = packed.cost

        aux = None
        if int(entry.aux_dim) > 0:
            aux = self._opponent_aux(entry, np.asarray(side_idxs, dtype=np.int64))

        dev = getattr(entry, "device", "cpu")
        with torch.inference_mode():
            logits, _values = entry.net(
                torch.from_numpy(obs).to(dev),
                torch.from_numpy(pills).to(dev),
                torch.from_numpy(previews).to(dev),
                torch.from_numpy(cand_actions).to(dev),
                torch.from_numpy(cand_cost).to(dev),
                torch.from_numpy(cand_mask).to(dev),
                aux=None if aux is None else torch.from_numpy(aux).to(dev),
            )
        # No .cpu() here: the copy-back in _sample_opponent_actions is the
        # only synchronization point, after every group's forward is queued.
        return logits, cand_actions, cand_mask

    def _sample_opponent_actions(
        self, logits: Any, cand_actions: np.ndarray, cand_mask: np.ndarray
    ) -> np.ndarray:
        import torch

        from drmc_rl.models.policy.placement_dist import MaskedPlacementDist

        logits_cpu = logits.float().cpu()
        temp = getattr(self, "_opp_temperature", 0.0)
        if temp > 0:
            dist = MaskedPlacementDist(logits_cpu / temp, torch.from_numpy(cand_mask))
            slot, _lp = dist.sample(deterministic=False)
        else:
            dist = MaskedPlacementDist(logits_cpu, torch.from_numpy(cand_mask))
            slot, _lp = dist.sample(deterministic=True)  # argmax: eval-matched
        slot_np = slot.numpy().reshape(-1).astype(np.int64)
        B = cand_actions.shape[0]
        # Empty mask -> fallback slot 0 -> padding action -1 (noop-fall).
        return cand_actions[np.arange(B), slot_np].astype(np.int32)

    def _record_match(
        self, pair_i: int, outcome: np.ndarray, pair_clock: int, *, horizon: bool = False
    ) -> None:
        i0 = pair_i * 2
        oc0 = int(outcome[i0])
        self._matches_total += 1
        if oc0 == VS_OUTCOME_WIN:
            self._win_p1.append(1.0)
            self._draws.append(0.0)
        elif oc0 == VS_OUTCOME_LOSS:
            self._win_p1.append(0.0)
            self._draws.append(0.0)
        else:
            self._win_p1.append(0.5)
            self._draws.append(1.0)

        length_s = float(max(1, int(pair_clock))) / NES_FPS
        self._match_len_sec.append(length_s)
        viruses_rem = self._runner.buffers.viruses_rem
        learner_clear = bool(not horizon and oc0 == VS_OUTCOME_WIN and int(viruses_rem[i0]) == 0)
        learner_topout_win = bool(
            not horizon and oc0 == VS_OUTCOME_WIN and int(viruses_rem[i0]) != 0
        )
        self._clear_wins.append(float(learner_clear))
        self._topout_wins.append(float(learner_topout_win))
        self._horizon_matches.append(float(horizon))
        self._pills_per_match.append(float(self._ep_pills[i0]))
        self._garbage_per_match.append(float(self._garbage_sent_prev[i0]))
        self._viruses_per_match.append(float(self._ep_viruses_cleared[i0]))
        minutes = max(length_s / 60.0, 1e-6)

        # DrMC-style per-game features for tools/skill_grade.py (one record per
        # side). Approximations (no per-frame stats from the pool):
        #   cpm           = garbage volleys sent / min
        #   cur           = cures: 1 if this side won by clearing all viruses
        #                   (the crown-table cur is a small per-set count;
        #                   human feature range is [0, 3.33])
        #   spd           = 31 + 0.5 * (pills // 10): HI base speed plus the
        #                   time-average of the every-10-pills speedup ramp
        #   salt_per_min  = garbage half pills received / min (true SALT
        #                   weights received garbage by stack depth)
        #   pills_per_min = pills locked / min
        #   garbage_per_min = garbage half pills sent / min
        # Pool mode: only the learner side is graded (P2 is a frozen net).
        sides = (i0,) if self._opp_pool is not None else (i0, i0 + 1)
        for i in sides:
            pills = int(self._ep_pills[i])
            won = int(outcome[i]) == VS_OUTCOME_WIN
            cured = bool(won and int(viruses_rem[i]) == 0)
            self._skill_games.append(
                {
                    "length_s": length_s,
                    "cpm": float(self._volleys_sent_round[i]) / minutes,
                    "cur": 1.0 if cured else 0.0,
                    "spd": 31.0 + 0.5 * float(pills // 10),
                    "salt_per_min": float(self._garbage_recv_round[i]) / minutes,
                    "pills_per_min": float(pills) / minutes,
                    "garbage_per_min": float(self._garbage_sent_prev[i]) / minutes,
                    "won": 1.0 if won else 0.0,
                }
            )

    def _build_reset_specs(self, *, reset_mask: Optional[np.ndarray]) -> List[object]:
        specs: List[object] = []
        for pair_i in range(self.num_pairs):
            if reset_mask is not None and int(reset_mask[pair_i]) == 0:
                specs.append(build_vs_reset_spec())  # placeholder; ignored natively
                continue

            provided = self.seed_provider(pair_i) if self.seed_provider is not None else None
            if provided is not None:
                seed_bytes = (int(provided[0]) & 0xFF, int(provided[1]) & 0xFF)
                rng_override = True
            elif self.rng_randomize:
                seed_bytes = (
                    int(self._rng.integers(0, 256)) & 0xFF,
                    int(self._rng.integers(0, 256)) & 0xFF,
                )
                rng_override = True
            else:
                seed_bytes = (0, 0)
                rng_override = False

            provided_frame = (
                self.frame_counter_provider(pair_i)
                if self.frame_counter_provider is not None
                else None
            )
            frame_counter_base = (
                int(provided_frame) & 0xFFFF
                if provided_frame is not None
                else int(self._rng.integers(0, 2**16))
                if self.rng_randomize
                else 0
            )

            # Go-Exploit: draw a fraction of resets from the start bank
            # (mid-game checkpoint overlay; RNG stays randomized above).
            bank_kwargs: Dict[str, Any] = {}
            if (
                self._start_bank is not None
                and float(self._rng.random()) < self._start_bank_fraction
            ):
                bank_kwargs = self._start_bank.sample_spec_kwargs(self._rng)

            specs.append(
                build_vs_reset_spec(
                    level=(self.level, self.level),
                    speed_setting=(self.speed_setting, self.speed_setting),
                    rng_state=seed_bytes,
                    rng_override=rng_override,
                    frame_counter_base=frame_counter_base,
                    **bank_kwargs,
                )
            )
        return specs

    def _build_obs_in_place(self) -> None:
        """Replicate `DrMarioPool::build_obs` (bitplane_bottle_conn_mask, 12ch).

        Channels: 0..2 color_{r,y,b}, 3 virus_mask, 4..7 connection edges
        (connected_up/down/left/right from the tile-type nibble), then the
        feasible mask planes (one per orientation) at ``self._feas_ch``.
        With ``opponent_obs`` the opponent bottle planes occupy 8..15 and are
        filled by `_fill_opponent_planes_in_place` after symmetry reduction.
        """

        board = self._runner.buffers.board_bytes.reshape(self.num_sides, GRID_H, GRID_W)
        type_hi = board & _MASK_TYPE
        color_lo = board & _MASK_COLOR

        is_empty = board == _TILE_EMPTY
        is_zero = board == 0x00
        just_emptied = (type_hi == _TILE_JUST_EMPTIED) & ~is_empty
        clearing = (type_hi == _TILE_CLEARED) | just_emptied
        color_valid = ~(is_empty | is_zero | clearing)

        obs = self._obs
        # Raw NES color bits: 0=Y,1=R,2=B -> canonical planes R,Y,B.
        obs[:, 0] = (color_valid & (color_lo == 1)).astype(np.float32)
        obs[:, 1] = (color_valid & (color_lo == 0)).astype(np.float32)
        obs[:, 2] = (color_valid & (color_lo == 2)).astype(np.float32)
        obs[:, 3] = (type_hi == ram_specs.T_VIRUS).astype(np.float32)
        obs[:, 4] = (type_hi == ram_specs.T_BOTTOM).astype(np.float32)  # connected_up
        obs[:, 5] = (type_hi == ram_specs.T_TOP).astype(np.float32)  # connected_down
        obs[:, 6] = (type_hi == ram_specs.T_RIGHT).astype(np.float32)  # connected_left
        obs[:, 7] = (type_hi == ram_specs.T_LEFT).astype(np.float32)  # connected_right
        obs[:, self._feas_ch : self._feas_ch + 4] = self._mask.astype(np.float32)

    def _fill_opponent_planes_in_place(self) -> None:
        """Copy each side's opponent bottle planes into channels 8..15.

        Called after `_apply_symmetry_reduction_in_place`, so side i's
        opponent planes are exactly side i^1's own-board planes as that side
        observes them (including the channel-6/7 zeroing quirk for same-color
        pills).
        """

        self._obs[:, 8:16] = self._obs[self._opp_idx, 0:8]

    def _apply_symmetry_reduction_in_place(self) -> None:
        """Same-color pills: drop H-/V- duplicate orientations (o=2,3).

        Replicates `DrMarioPoolVecEnv._apply_symmetry_reduction_in_place`
        exactly — including zeroing obs channels 6 and 7 (which for the
        12-channel conn_mask layout are the connected_left/right planes, not
        the feasibility planes). The 1P champion was trained with that
        behavior, so obs parity requires reproducing it as-is.
        """

        colors = self._runner.buffers.pill_colors
        if colors.shape != (self.num_sides, 2):
            return
        same = colors[:, 0] == colors[:, 1]
        if not bool(np.any(same)):
            return
        idxs = np.nonzero(same)[0]
        if idxs.size <= 0:
            return
        self._mask[idxs, 2, :, :] = 0
        self._mask[idxs, 3, :, :] = 0
        self._cost[idxs, 2, :, :] = np.uint16(0xFFFF)
        self._cost[idxs, 3, :, :] = np.uint16(0xFFFF)
        self._obs[idxs, 6, :, :] = 0.0
        self._obs[idxs, 7, :, :] = 0.0

    def _refresh_infos(self, *, step_tau_raw: np.ndarray) -> None:
        buf = self._runner.buffers
        pill_colors = buf.pill_colors
        preview_colors = buf.preview_colors
        spawn_ids = buf.spawn_id
        viruses = buf.viruses_rem
        invalid = buf.invalid_action
        garbage_pending = buf.garbage_pending
        garbage_sent = buf.garbage_sent_total
        need_action = self._need_action

        options_count = self._mask_1d.sum(axis=1).astype(np.int32, copy=False)

        for i in range(self.num_sides):
            info = self._infos[i]
            info.clear()

            info["placements/feasible_mask"] = self._mask[i]
            info["placements/cost_to_lock"] = self._cost[i]
            info["placements/options"] = int(options_count[i])
            info["placements/reach_backend"] = "cpp-vs-pool"
            info["placements/needs_action"] = bool(need_action[i])
            info["placements/spawn_id"] = int(spawn_ids[i])
            info["pill/spawn_id"] = int(spawn_ids[i])

            info["next_pill_colors"] = np.asarray(pill_colors[i], dtype=np.int64)
            p_left = _canonical_to_raw_color(int(preview_colors[i, 0]))
            p_right = _canonical_to_raw_color(int(preview_colors[i, 1]))
            info["preview_pill"] = {"first_color": int(p_left), "second_color": int(p_right), "rotation": 0}
            # Own-board NES tile bytes (view into the runner buffer; consumed at
            # decision time, e.g. by search distillation). Same contract as the
            # 1P pool env's `info["board"]`.
            info["board"] = buf.board_bytes[i]

            info["level"] = int(self.level)
            info["task_mode"] = "viruses"
            info["rng_randomize"] = bool(self.rng_randomize)
            info["viruses_remaining"] = int(viruses[i])
            info["drm/viruses_initial"] = int(self._viruses_initial[i])
            info["pill/speed_setting"] = int(self.speed_setting)
            info["speed_setting"] = int(self.speed_setting)
            info["task/frames_used"] = int(self._ep_frames[i])

            info["vs/pair"] = int(i // 2)
            info["vs/side"] = int(i % 2)
            info["vs/garbage_pending"] = int(garbage_pending[i])
            info["vs/garbage_sent_total"] = int(garbage_sent[i])

            # Opponent state (pair partner) for opponent-aware policies
            # (aux_spec=v1_vs); everything here is already exported per-side
            # by the native vspool.
            opp = i ^ 1
            info["vs/opponent_viruses_remaining"] = int(viruses[opp])
            info["vs/garbage_pending_opp"] = int(garbage_pending[opp])
            info["vs/opponent_pill_colors"] = np.asarray(pill_colors[opp], dtype=np.int64)
            info["vs/opponent_preview_colors"] = np.asarray(preview_colors[opp], dtype=np.int64)
            # Opponent NES tile bytes (view, like info["board"]) — consumed at
            # decision time by search_distill's opponent_model=self.
            info["vs/opponent_board"] = buf.board_bytes[opp]

            tau_i_raw = int(step_tau_raw[i]) if i < int(step_tau_raw.shape[0]) else 0
            info["placements/tau"] = int(max(1, tau_i_raw))
            if int(invalid[i]) != -1:
                info["placements/invalid_action"] = int(invalid[i])

            info["delta_v"] = 0
            info["r_env"] = 0.0
            info["r_shape"] = 0.0
            info["r_total"] = 0.0
            info["goal_achieved"] = False
            info["cleared"] = False
            info["topout"] = False
            info["terminal_reason"] = ""


__all__ = ["DrMarioVsPoolVecEnv"]
