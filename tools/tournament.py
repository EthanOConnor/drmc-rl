"""Round-robin VS tournaments and SPRT change gates over the native VS pool.

Subcommands (`python -m tools.tournament <cmd>`):

  run    — full round-robin over a roster yaml; every game is a row in a
           sqlite store (default runs/tournaments/tournaments.sqlite), so an
           interrupted tournament resumes where it left off. Game seeds and
           side assignments are derived deterministically from the tournament
           seed, so re-runs replay the same schedule.
  report — pairwise W-L-D matrix with Wilson 95% CIs plus maximum-likelihood
           Elo ratings (Bradley-Terry logistic MLE, draws = half points,
           anchor mean=0) with 95% CIs from the observed Fisher information.
  sprt   — sequential A-vs-B change gate: trinomial (BayesElo) SPRT log-
           likelihood ratio after every game, stopping on accept-H1 /
           accept-H0 / max-games (fishtest-style).

Roster yaml:

  entries:
    - name: tip
      checkpoint: runs/vs2_02/checkpoints/smdp_ppo_stepNNN.pt.gz
      mode: plain            # plain | search | ponder
    - name: tip-search
      checkpoint: runs/vs2_02/checkpoints/smdp_ppo_stepNNN.pt.gz
      mode: search
      params: {beam: 8, deadline_ms: 60}

Match running reuses the head-to-head machinery (`tools/vs_head_to_head.py`
PlainPolicy + per-env SearchPolicy.decide) over `DrMarioVsPoolVecEnv`.
See docs/TOURNAMENTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "runs" / "tournaments" / "tournaments.sqlite"
NES_FPS = 60.1
_ELO_K = math.log(10.0) / 400.0
_CANON_TO_RAW = (1, 0, 2)
_SPEEDUPS_MAX = 0x31


# ---------------------------------------------------------------------- stats
def wilson_ci(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% interval (same convention as tools/vs_head_to_head)."""

    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def elo_mle(
    num_entries: int,
    games: Sequence[Tuple[int, int, float]],
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bradley-Terry / logistic Elo MLE with mean-zero anchor.

    ``games`` is (i, j, score_i) with score in {0, 0.5, 1} (draw = 0.5; the
    quasi-likelihood score equation matches mean score to the logistic
    expectation). Returns (ratings, standard_errors); SEs come from the
    pseudo-inverse of the observed Fisher information (a graph Laplacian,
    singular along the all-ones translation direction the anchor removes).
    Entries that never lose (or never win) diverge: ratings are then capped
    by iteration count and the CIs blow up, which is the honest answer.
    """

    n = int(num_entries)
    # Aggregate per ordered pair: total games + total score (same p per pair).
    agg: Dict[Tuple[int, int], List[float]] = {}
    for i, j, s in games:
        key = (int(i), int(j))
        ent = agg.setdefault(key, [0.0, 0.0])
        ent[0] += 1.0
        ent[1] += float(s)

    r = np.zeros(n, dtype=np.float64)
    fisher = np.zeros((n, n), dtype=np.float64)
    for _ in range(max_iter):
        grad = np.zeros(n, dtype=np.float64)
        fisher.fill(0.0)
        for (i, j), (n_ij, s_sum) in agg.items():
            p = 1.0 / (1.0 + math.exp(-_ELO_K * (r[i] - r[j])))
            g = _ELO_K * (s_sum - n_ij * p)
            grad[i] += g
            grad[j] -= g
            w = _ELO_K * _ELO_K * n_ij * p * (1.0 - p)
            fisher[i, i] += w
            fisher[j, j] += w
            fisher[i, j] -= w
            fisher[j, i] -= w
        step = np.linalg.pinv(fisher) @ grad
        step = np.clip(step, -200.0, 200.0)  # damp degenerate sweeps
        r += step
        r -= r.mean()
        if float(np.max(np.abs(step))) < tol:
            break
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(fisher)), 0.0, None))
    return r, se


def _bayeselo_to_proba(elo: float, drawelo: float) -> Tuple[float, float, float]:
    pw = 1.0 / (1.0 + 10.0 ** ((-elo + drawelo) / 400.0))
    pl = 1.0 / (1.0 + 10.0 ** ((elo + drawelo) / 400.0))
    return pw, 1.0 - pw - pl, pl


def sprt_llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Trinomial (BayesElo-model) SPRT log-likelihood ratio, fishtest-style.

    elo0/elo1 are logistic Elo hypotheses; the draw rate is estimated from the
    observed W/D/L (zero cells are regularized at 0.5 games for that estimate
    only — Dr. Mario VS draws are rare, so D=0 must not stall the test).
    """

    if wins + draws + losses <= 0 or (wins <= 0 and losses <= 0):
        return 0.0
    w, d, lo = max(wins, 0.5), max(draws, 0.5), max(losses, 0.5)
    n = w + d + lo
    pw, pl = w / n, lo / n
    drawelo = 200.0 * math.log10((1.0 - pl) / pl * (1.0 - pw) / pw)
    x = 10.0 ** (-drawelo / 400.0)
    scale = 4.0 * x / (1.0 + x) ** 2  # bayeselo -> logistic elo derivative
    p0 = _bayeselo_to_proba(elo0 / scale, drawelo)
    p1 = _bayeselo_to_proba(elo1 / scale, drawelo)
    return (
        wins * math.log(p1[0] / p0[0])
        + draws * math.log(p1[1] / p0[1])
        + losses * math.log(p1[2] / p0[2])
    )


def sprt_bounds(alpha: float, beta: float) -> Tuple[float, float]:
    """(lower, upper) LLR stopping bounds: accept H0 at lower, H1 at upper."""

    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


# ------------------------------------------------------------------ scheduling
@dataclass(frozen=True)
class GameSpec:
    game_idx: int  # index within the pair's series (0..games_per_pair-1)
    seed: int  # 16-bit NES rng state
    a_side: int  # which physical side entry A plays (0 or 1)


@dataclass(frozen=True)
class GameResult:
    spec: GameSpec
    winner: str  # 'a' | 'b' | 'draw'
    match_len_sec: float
    decisions: int
    terminal_reason: str = "unknown"  # clear | topout | horizon | simultaneous | unknown
    replay: Optional[List[Dict[str, Any]]] = None


def game_seed(tournament_seed: int, name_a: str, name_b: str, game_idx: int) -> int:
    h = hashlib.sha256(f"{tournament_seed}:{name_a}:{name_b}:{game_idx}".encode()).digest()
    s = int.from_bytes(h[:2], "little")
    return s if s != 0 else 0x55AA  # avoid the degenerate all-zero LFSR state


def make_specs(
    tournament_seed: int, name_a: str, name_b: str, game_idxs: Iterable[int]
) -> List[GameSpec]:
    return [
        GameSpec(game_idx=k, seed=game_seed(tournament_seed, name_a, name_b, k), a_side=k % 2)
        for k in game_idxs
    ]


# ----------------------------------------------------------------------- store
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  created TEXT NOT NULL,
  roster TEXT NOT NULL,
  level INTEGER NOT NULL,
  games_per_pair INTEGER NOT NULL,
  seed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tournament_id INTEGER NOT NULL REFERENCES tournaments(id),
  entry_a TEXT NOT NULL,
  entry_b TEXT NOT NULL,
  game_idx INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  side_assignment INTEGER NOT NULL,
  winner TEXT NOT NULL,
  match_len_sec REAL,
  decisions INTEGER,
  created TEXT NOT NULL,
  UNIQUE (tournament_id, entry_a, entry_b, game_idx)
);
"""


class TournamentStore:
    """Minimal sqlite results store; one row per completed game."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_or_create_tournament(
        self, name: str, roster: Dict[str, Any], level: int, games_per_pair: int, seed: int
    ) -> int:
        row = self.conn.execute("SELECT * FROM tournaments WHERE name = ?", (name,)).fetchone()
        if row is not None:
            old_names = [e["name"] for e in json.loads(row["roster"]).get("entries", [])]
            new_names = [e["name"] for e in roster.get("entries", [])]
            if sorted(old_names) != sorted(new_names):
                raise SystemExit(
                    f"tournament '{name}' exists with entries {old_names}; "
                    f"roster now has {new_names}. Pick a new --name."
                )
            if int(row["seed"]) != int(seed):
                raise SystemExit(
                    f"tournament '{name}' was created with seed {row['seed']}; "
                    f"resume with the same --seed (got {seed})."
                )
            self.conn.execute(
                "UPDATE tournaments SET games_per_pair = ? WHERE id = ?",
                (int(games_per_pair), int(row["id"])),
            )
            self.conn.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO tournaments (name, created, roster, level, games_per_pair, seed)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                name,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(roster),
                int(level),
                int(games_per_pair),
                int(seed),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def tournament_by_name(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM tournaments WHERE name = ?", (name,)).fetchone()

    def recorded_game_idxs(self, tournament_id: int, entry_a: str, entry_b: str) -> set:
        rows = self.conn.execute(
            "SELECT game_idx FROM games WHERE tournament_id = ? AND entry_a = ? AND entry_b = ?",
            (int(tournament_id), entry_a, entry_b),
        ).fetchall()
        return {int(r["game_idx"]) for r in rows}

    def insert_game(
        self, tournament_id: int, entry_a: str, entry_b: str, result: GameResult
    ) -> None:
        self.conn.execute(
            "INSERT INTO games (tournament_id, entry_a, entry_b, game_idx, seed,"
            " side_assignment, winner, match_len_sec, decisions, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(tournament_id),
                entry_a,
                entry_b,
                int(result.spec.game_idx),
                int(result.spec.seed),
                int(result.spec.a_side),
                str(result.winner),
                float(result.match_len_sec),
                int(result.decisions),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def games_for(self, tournament_id: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM games WHERE tournament_id = ? ORDER BY entry_a, entry_b, game_idx",
            (int(tournament_id),),
        ).fetchall()


# ------------------------------------------------------------------ tournament
def run_tournament(
    store: TournamentStore,
    name: str,
    roster: Dict[str, Any],
    *,
    games_per_pair: int,
    level: int,
    seed: int,
    runner: Any,
    verbose: bool = True,
) -> int:
    """Round-robin over roster entries; resumable (skips recorded game_idxs).

    ``runner`` provides ``play(entry_a, entry_b, specs) -> Iterable[GameResult]``
    (the real one is `VsMatchRunner`; tests inject a stub).
    """

    entries = list(roster["entries"])
    if len(entries) < 2:
        raise SystemExit("roster needs at least 2 entries")
    names = [str(e["name"]) for e in entries]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate entry names in roster: {names}")

    tid = store.get_or_create_tournament(name, roster, level, games_per_pair, seed)
    for ia in range(len(entries)):
        for ib in range(ia + 1, len(entries)):
            ea, eb = entries[ia], entries[ib]
            done = store.recorded_game_idxs(tid, ea["name"], eb["name"])
            todo = [k for k in range(int(games_per_pair)) if k not in done]
            if not todo:
                continue
            specs = make_specs(seed, ea["name"], eb["name"], todo)
            if verbose:
                print(
                    f"pair {ea['name']} vs {eb['name']}: {len(done)} recorded, "
                    f"{len(specs)} to play",
                    flush=True,
                )
            for res in runner.play(ea, eb, specs):
                store.insert_game(tid, ea["name"], eb["name"], res)
                if verbose:
                    print(
                        f"  game {res.spec.game_idx:3d} seed={res.spec.seed:5d} "
                        f"a_side={res.spec.a_side} -> {res.winner:4s} "
                        f"({res.match_len_sec:.0f}s, {res.decisions} decisions)",
                        flush=True,
                    )
    return tid


# ---------------------------------------------------------------- match runner
def _entry_board_channels(entry: Dict[str, Any]) -> int:
    """Board-channel count from the entry checkpoint's embedded cfg."""
    from drmc_rl.training.utils.checkpoint_io import load_checkpoint

    ckpt = REPO_ROOT / str(entry["checkpoint"])
    if not ckpt.exists():
        ckpt = Path(str(entry["checkpoint"]))
    payload = load_checkpoint(ckpt, map_location="cpu")
    if payload.get("schema") == "drmc-human-afterstate-v3":
        return 16
    cfg = payload.get("cfg", {})
    sp = cfg.get("smdp_ppo", cfg)
    return int(sp.get("candidate_board_channels", 8))


def pick_state_repr(entries: Sequence[Dict[str, Any]]) -> str:
    """Opponent-obs (20-ch) env when any entry's net needs it, else legacy.

    Legacy 8-board-channel nets run on the 20-ch env via the slice in
    `PlainPolicy.act`; search/ponder entries are not yet wired for the
    opponent-obs obs assembly and are rejected with a clear error.
    """
    needs_vs = [e for e in entries if _entry_board_channels(e) >= 16]
    if not needs_vs:
        return "bitplane_bottle_conn_mask"
    bad = [
        e["name"] for e in needs_vs if str(e.get("mode", "plain")).lower() not in {"plain", "human"}
    ]
    if bad:
        raise SystemExit(
            "search/ponder modes are not yet supported for opponent-obs "
            f"checkpoints (entries: {', '.join(bad)})"
        )
    return "bitplane_bottle_conn_mask_vs"


class VsMatchRunner:
    """Plays scheduled VS games between two roster entries on the native pool.

    Pairs in the vector env each work through a shared queue of `GameSpec`s;
    per-game NES rng seeds go in through the env's ``seed_provider`` hook and
    entry A's physical side comes from ``spec.a_side``. Policy wrappers are
    cached per entry config, so the same checkpoint loads once per process.
    """

    def __init__(
        self,
        *,
        level: int = 14,
        speed_setting: int = 2,
        num_pairs: int = 4,
        device: str = "auto",
        threads: int = 4,
        run_seed: int = 0,
        state_repr: str = "bitplane_bottle_conn_mask",
        replay_sample_rate: float = 0.0,
        max_decisions_per_side: int = 500,
    ) -> None:
        import torch

        from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

        torch.set_num_threads(int(threads))
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.level = int(level)
        self.speed_setting = int(speed_setting)
        self.num_pairs = int(num_pairs)
        self.run_seed = int(run_seed)
        self.replay_sample_rate = min(max(float(replay_sample_rate), 0.0), 1.0)
        self.max_decisions_per_side = max(0, int(max_decisions_per_side))
        self._assigned: List[Optional[GameSpec]] = [None] * self.num_pairs
        self.env = DrMarioVsPoolVecEnv(
            num_pairs=self.num_pairs,
            level=self.level,
            speed_setting=self.speed_setting,
            randomize_rng=True,
            seed_provider=self._seed_for_pair,
            state_repr=state_repr,
        )
        self._policies: Dict[str, Any] = {}

    def close(self) -> None:
        self.env.close()
        for pol in self._policies.values():
            pol.close()

    def _seed_for_pair(self, pair_i: int) -> Optional[Tuple[int, int]]:
        spec = self._assigned[pair_i]
        if spec is None:
            return None  # idle pair: randomized junk game, results ignored
        return (spec.seed & 0xFF, (spec.seed >> 8) & 0xFF)

    def _policy_for(self, entry: Dict[str, Any]) -> "_EntryPolicy":
        key = json.dumps(entry, sort_keys=True)
        if key not in self._policies:
            self._policies[key] = _EntryPolicy(entry, runner=self)
        return self._policies[key]

    def play(
        self, entry_a: Dict[str, Any], entry_b: Dict[str, Any], specs: Sequence[GameSpec]
    ) -> Iterator[GameResult]:
        pol_a = self._policy_for(entry_a)
        pol_b = self._policy_for(entry_b)
        queue = deque(specs)
        self._assigned = [queue.popleft() if queue else None for _ in range(self.num_pairs)]
        series_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.run_seed}:{entry_a['name']}:{entry_b['name']}".encode()
            ).digest()[:4],
            "little",
        )
        obs, infos = self.env.reset(seed=series_seed)
        S = self.env.num_sides
        ep_decisions = np.zeros(S, dtype=np.int64)
        replaying = [
            spec is not None
            and random.Random(spec.seed ^ self.run_seed).random() < self.replay_sample_rate
            for spec in self._assigned
        ]
        replays: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_pairs)]
        remaining = len(specs)

        while remaining > 0:
            need = np.array(
                [bool(infos[i].get("placements/needs_action", False)) for i in range(S)]
            )
            acts = np.full(S, -1, dtype=np.int32)  # -1 = noop-fall (idle pairs too)
            idxs_a: List[int] = []
            idxs_b: List[int] = []
            for pair_i in range(self.num_pairs):
                spec = self._assigned[pair_i]
                if spec is None:
                    continue
                ga = pair_i * 2 + spec.a_side
                gb = pair_i * 2 + (1 - spec.a_side)
                if need[ga]:
                    idxs_a.append(ga)
                if need[gb]:
                    idxs_b.append(gb)
            if idxs_a:
                acts[idxs_a] = pol_a.decide(idxs_a, obs, infos, ep_decisions)
            if idxs_b:
                acts[idxs_b] = pol_b.decide(idxs_b, obs, infos, ep_decisions)
            boards = self.env._runner.buffers.board_bytes
            pills = self.env._runner.buffers.pill_colors
            for pair_i, capture in enumerate(replaying):
                if not capture or self._assigned[pair_i] is None:
                    continue
                i0 = pair_i * 2
                if not bool(need[i0] or need[i0 + 1]):
                    continue
                replays[pair_i].append(
                    {
                        "boards": [boards[i0].tolist(), boards[i0 + 1].tolist()],
                        "pills": [pills[i0].tolist(), pills[i0 + 1].tolist()],
                        "actions": [int(acts[i0]), int(acts[i0 + 1])],
                        "decision": int(max(ep_decisions[i0], ep_decisions[i0 + 1])),
                    }
                )
            ep_decisions[need] += 1
            obs, _r, term, trunc, infos = self.env.step(acts)

            for pair_i in range(self.num_pairs):
                i0 = pair_i * 2
                ended = bool(term[i0] or trunc[i0] or term[i0 + 1] or trunc[i0 + 1])
                timed_out = bool(
                    self.max_decisions_per_side
                    and max(ep_decisions[i0], ep_decisions[i0 + 1]) >= self.max_decisions_per_side
                )
                if not (ended or timed_out):
                    continue
                decision_count = int(ep_decisions[i0] + ep_decisions[i0 + 1])
                frame_count = int(max(self.env._ep_frames[i0], self.env._ep_frames[i0 + 1]))
                if timed_out and not ended:
                    # Arena adjudication only: a policy pair that survives without
                    # resolving the board has reached the match horizon. Reset this
                    # vector lane on its next step, just like a native terminal.
                    self.env._pending_reset[pair_i] = True
                    for side_i in (i0, i0 + 1):
                        self.env._ep_return[side_i] = 0.0
                        self.env._ep_frames[side_i] = 0
                        self.env._ep_decisions[side_i] = 0
                        self.env._ep_pills[side_i] = 0
                        self.env._ep_viruses_cleared[side_i] = 0
                ep_decisions[i0] = 0
                ep_decisions[i0 + 1] = 0
                spec = self._assigned[pair_i]
                self._assigned[pair_i] = queue.popleft() if queue else None
                if spec is None:
                    continue
                ga = i0 + spec.a_side
                gb = i0 + (1 - spec.a_side)
                oc = str(infos[ga].get("vs/outcome", "")) if ended else ""
                winner = "a" if oc == "win" else ("b" if oc == "loss" else "draw")
                terminal_reason = "horizon" if timed_out and not ended else "simultaneous"
                if ended:
                    ep_a = infos[ga].get("episode", {})
                    frame_count = int(ep_a.get("l", frame_count))
                    decision_count = int(infos[i0].get("episode", {}).get("decisions", 0)) + int(
                        infos[i0 + 1].get("episode", {}).get("decisions", 0)
                    )
                    if winner != "draw":
                        winner_i = ga if winner == "a" else gb
                        loser_i = gb if winner == "a" else ga
                        if bool(infos[winner_i].get("drm", {}).get("cleared", False)):
                            terminal_reason = "clear"
                        elif bool(infos[loser_i].get("drm", {}).get("top_out", False)):
                            terminal_reason = "topout"
                        else:
                            terminal_reason = "unknown"
                remaining -= 1
                yield GameResult(
                    spec=spec,
                    winner=winner,
                    match_len_sec=float(frame_count) / NES_FPS,
                    decisions=decision_count,
                    terminal_reason=terminal_reason,
                    replay=replays[pair_i] or None,
                )
                replays[pair_i] = []
                next_spec = self._assigned[pair_i]
                replaying[pair_i] = bool(
                    next_spec is not None
                    and random.Random(next_spec.seed ^ self.run_seed).random()
                    < self.replay_sample_rate
                )


class _EntryPolicy:
    """One roster entry's acting policy (plain argmax / search / ponder)."""

    def __init__(self, entry: Dict[str, Any], *, runner: VsMatchRunner) -> None:
        from tools.vs_head_to_head import PlainPolicy

        self.runner = runner
        self.mode = str(entry.get("mode", "plain")).lower()
        params = dict(entry.get("params") or {})
        ckpt = REPO_ROOT / str(entry["checkpoint"])
        if not ckpt.exists():
            ckpt = Path(str(entry["checkpoint"]))
        if not ckpt.exists():
            raise SystemExit(f"entry '{entry.get('name')}': checkpoint not found: {ckpt}")
        self.search = None
        self.human = None
        self.human_v3 = False
        # Reliance probe: zero the opponent planes (ch 8-15) and the v1_vs aux
        # tail before the net sees them — measures how much a 20-ch checkpoint
        # actually uses opponent information.
        self.mask_opponent = bool(params.get("mask_opponent", False))
        if self.mode == "plain":
            self.plain = PlainPolicy(ckpt, device=str(params.get("device", "cpu")))
            if self.mask_opponent and self.plain.in_channels != 20:
                raise SystemExit(
                    f"entry '{entry.get('name')}': mask_opponent requires an "
                    "opponent-obs (20-channel) checkpoint"
                )
            return
        if self.mode == "human":
            from drmc_rl.human.afterstate_model import HUMAN_AFTERSTATE_SCHEMA
            from drmc_rl.human.afterstate_runtime import AfterstatePolicyRuntime
            from drmc_rl.human.runtime import HumanPolicyRuntime
            from drmc_rl.training.utils.checkpoint_io import load_checkpoint

            runtime_args = {
                "device": str(params.get("device", "cpu")),
                "seed": int(params.get("seed", runner.run_seed + 1)),
            }
            self.human_v3 = (
                load_checkpoint(ckpt, map_location="cpu").get("schema") == HUMAN_AFTERSTATE_SCHEMA
            )
            runtime = AfterstatePolicyRuntime if self.human_v3 else HumanPolicyRuntime
            self.human = runtime(ckpt, **runtime_args)
            self.human_rating = float(params["rating"])
            self.human_rating_sd = float(params.get("rating_sd", 0.0))
            self.human_opponent_rating = float(params.get("opponent_rating", self.human_rating))
            self.human_temperature = float(params.get("temperature", 1.0))
            self.human_strength_control = str(params.get("strength_control", "regret")).lower()
            if self.human_strength_control not in {"regret", "style"}:
                raise SystemExit(
                    f"entry '{entry.get('name')}': strength_control must be regret or style"
                )
            if self.human_strength_control == "style" and not self.human_v3:
                raise SystemExit(
                    f"entry '{entry.get('name')}': pure style control requires a V3 checkpoint"
                )
            return
        if self.mask_opponent:
            raise SystemExit("mask_opponent is only supported for mode: plain")
        if self.mode not in {"search", "ponder"}:
            raise SystemExit(f"entry '{entry.get('name')}': unknown mode '{self.mode}'")
        from drmc_rl.models.policy.search_policy import PonderingSearchPolicy, SearchPolicy

        kwargs = dict(
            beam=int(params.get("beam", 8)),
            deadline_ms=float(params.get("deadline_ms", 60.0)),
            device=str(params.get("device", runner.device)),
            seed=int(params.get("seed", runner.run_seed + 1)),
        )
        if self.mode == "ponder":
            self.search = PonderingSearchPolicy(
                ckpt, ponder_budget_s=float(params.get("ponder_budget_s", 1.0)), **kwargs
            )
        else:
            self.search = SearchPolicy(ckpt, **kwargs)

    def close(self) -> None:
        if self.search is not None:
            self.search.close()
        if self.human is not None:
            close = getattr(self.human, "close", None)
            if close is not None:
                close()

    def decide(
        self,
        idxs: List[int],
        obs: np.ndarray,
        infos: List[Dict[str, Any]],
        ep_decisions: np.ndarray,
    ) -> np.ndarray:
        if self.mode == "plain":
            sub_obs = obs[idxs]
            sub_infos = [infos[i] for i in idxs]
            if self.mask_opponent:
                sub_obs = sub_obs.copy()
                sub_obs[:, 8:16] = 0
                sub_infos = [
                    {k: v for k, v in info.items() if not k.startswith("vs/")} for info in sub_infos
                ]
            return self.plain.act(sub_obs, sub_infos)
        if self.mode == "human":
            from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates

            if obs.shape[1] != 20:
                raise ValueError("human arena entrants require the 20-channel VS observation")
            acts = np.full(len(idxs), -1, dtype=np.int32)
            env = self.runner.env
            for k, i in enumerate(idxs):
                info = infos[i]
                packed = pack_feasible_candidates(
                    np.asarray(info["placements/feasible_mask"], dtype=bool),
                    info["placements/cost_to_lock"],
                    max_candidates=(
                        128
                        if self.human_v3
                        else int(self.human.cfg["smdp_ppo"]["candidate_max_candidates"])
                    ),
                    sort_by_cost=True,
                )
                if not packed.mask.any():
                    continue
                recent = [
                    {"action": int(action), "tau_frames": float(tau)}
                    for action, tau in zip(env._human_recent_actions[i], env._human_recent_tau[i])
                    if int(action) >= 0
                ]
                score_args = dict(
                    board_planes=obs[i, :8],
                    opponent_board_planes=obs[i, 8:16],
                    opponent_state_age_frames=0,
                    rating_sd=self.human_rating_sd,
                    opponent_rating=self.human_opponent_rating,
                    game_phase=min(float(ep_decisions[i]) / 100.0, 1.0),
                    recent_decisions=recent,
                    pill=env._runner.buffers.pill_colors[i],
                    preview=env._runner.buffers.preview_colors[i],
                    candidate_actions=packed.actions,
                    candidate_costs=packed.cost,
                    candidate_mask=packed.mask,
                    rating=self.human_rating,
                )
                if self.human_v3:
                    speed_ups = min(
                        _SPEEDUPS_MAX,
                        max(0, self.runner.level - 20) + int(ep_decisions[i]) // 10,
                    )
                    details = self.human.score(
                        **score_args,
                        speed=self.runner.speed_setting,
                        speed_ups=speed_ups,
                    )
                    if self.human_strength_control == "style":
                        slot = self.human.choose_style(
                            details["human_logits"],
                            packed.mask,
                            temperature=self.human_temperature,
                        )
                    else:
                        slot, _strength = self.human.choose_strength(
                            details["competitive_score"],
                            details["human_logits"],
                            packed.mask,
                            rating=float(details["resolved_rating"]),
                            temperature=self.human_temperature,
                        )
                else:
                    logits, _value, _rating, _clamped = self.human.score(**score_args)
                    slot = self.human.choose(
                        logits, packed.mask, temperature=self.human_temperature
                    )
                acts[k] = int(packed.actions[slot])
            return acts
        acts = np.full(len(idxs), -1, dtype=np.int32)
        env = self.runner.env
        base_speed_ups = max(0, self.runner.level - 20)
        for k, i in enumerate(idxs):
            info = infos[i]
            mask = np.asarray(info["placements/feasible_mask"], dtype=bool)
            if not mask.any():
                continue  # noop-fall
            boards = env._runner.buffers.board_bytes
            pills = env._runner.buffers.pill_colors
            pv = info["preview_pill"]
            spd_ups = min(_SPEEDUPS_MAX, base_speed_ups + int(ep_decisions[i]) // 10)
            a, _sinfo = self.search.decide(
                boards[i],
                (_CANON_TO_RAW[int(pills[i, 0])], _CANON_TO_RAW[int(pills[i, 1])]),
                (int(pv["first_color"]), int(pv["second_color"])),
                mask,
                info["placements/cost_to_lock"],
                self.runner.speed_setting,
                spd_ups,
                self.runner.level,
                viruses_initial=int(info.get("drm/viruses_initial", 0)),
                frames_used=int(info.get("task/frames_used", 0)),
            )
            if a >= 0:
                acts[k] = int(a)
            if self.mode == "ponder" and a >= 0:
                # Offline dead time is simulated (same convention as
                # tools/vs_head_to_head --a-ponder): run the next-decision
                # ponder to completion before the env advances.
                self.search.start_ponder(
                    boards[i].copy(),
                    int(a),
                    (_CANON_TO_RAW[int(pills[i, 0])], _CANON_TO_RAW[int(pills[i, 1])]),
                    (int(pv["first_color"]), int(pv["second_color"])),
                    feasible_mask512=mask,
                    cost_to_lock512=info["placements/cost_to_lock"],
                    speed_setting=self.runner.speed_setting,
                    speed_ups=spd_ups,
                    level=self.runner.level,
                    viruses_initial=int(info.get("drm/viruses_initial", 0)),
                    frames_used=int(info.get("task/frames_used", 0)),
                )
                self.search.wait_ponder_idle(timeout=60.0)
        return acts


# --------------------------------------------------------------------- report
def _pair_records(rows: Sequence[sqlite3.Row]) -> Dict[Tuple[str, str], List[int]]:
    """(entry_a, entry_b) -> [wins_a, losses_a, draws] over recorded games."""

    rec: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        key = (str(r["entry_a"]), str(r["entry_b"]))
        wld = rec.setdefault(key, [0, 0, 0])
        if r["winner"] == "a":
            wld[0] += 1
        elif r["winner"] == "b":
            wld[1] += 1
        else:
            wld[2] += 1
    return rec


def render_report(store: TournamentStore, name: str) -> str:
    t = store.tournament_by_name(name)
    if t is None:
        raise SystemExit(f"no tournament named '{name}' in {store.path}")
    rows = store.games_for(int(t["id"]))
    roster = json.loads(t["roster"])
    names = [str(e["name"]) for e in roster["entries"]]
    rec = _pair_records(rows)

    lines: List[str] = []
    lines.append(
        f"tournament '{name}'  level={t['level']}  games_per_pair={t['games_per_pair']}  "
        f"entries={len(names)}  games_recorded={len(rows)}"
    )

    # Pairwise W-L-D matrix (row perspective).
    width = max([len(n) for n in names] + [10])
    header = " " * (width + 2) + "  ".join(n.ljust(width) for n in names)
    lines.append("")
    lines.append("pairwise W-L-D (row vs col):")
    lines.append(header)
    for na in names:
        cells = []
        for nb in names:
            if na == nb:
                cells.append(".".ljust(width))
                continue
            if (na, nb) in rec:
                w, lo, d = rec[(na, nb)]
            elif (nb, na) in rec:
                lo, w, d = rec[(nb, na)]
            else:
                cells.append("-".ljust(width))
                continue
            cells.append(f"{w}-{lo}-{d}".ljust(width))
        lines.append(na.ljust(width + 2) + "  ".join(cells))

    lines.append("")
    lines.append("pairwise win rate over decisive games (Wilson 95%):")
    for (na, nb), (w, lo_n, d) in sorted(rec.items()):
        nd = w + lo_n
        wr = w / nd if nd else float("nan")
        ci_lo, ci_hi = wilson_ci(w, nd) if nd else (0.0, 1.0)
        lines.append(
            f"  {na} vs {nb}: {w}-{lo_n}-{d}  win_rate={wr:.3f}  CI95=[{ci_lo:.3f}, {ci_hi:.3f}]"
        )

    # Elo MLE (draws = half points).
    idx = {n: i for i, n in enumerate(names)}
    games: List[Tuple[int, int, float]] = []
    totals: Dict[str, List[int]] = {n: [0, 0, 0] for n in names}  # W, D, L
    for r in rows:
        i, j = idx[str(r["entry_a"])], idx[str(r["entry_b"])]
        s = 1.0 if r["winner"] == "a" else (0.0 if r["winner"] == "b" else 0.5)
        games.append((i, j, s))
        wa, wb = str(r["entry_a"]), str(r["entry_b"])
        if r["winner"] == "a":
            totals[wa][0] += 1
            totals[wb][2] += 1
        elif r["winner"] == "b":
            totals[wa][2] += 1
            totals[wb][0] += 1
        else:
            totals[wa][1] += 1
            totals[wb][1] += 1
    lines.append("")
    lines.append("Elo (logistic MLE, draws=0.5, anchor mean=0, ±95% from Fisher info):")
    if games:
        ratings, ses = elo_mle(len(names), games)
        order = np.argsort(-ratings)
        lines.append(f"  {'name'.ljust(width)}  {'elo':>8}  {'±95':>7}  {'games':>5}  W-D-L")
        for o in order:
            n = names[int(o)]
            w, d, lo = totals[n]
            lines.append(
                f"  {n.ljust(width)}  {ratings[o]:>+8.1f}  {1.96 * ses[o]:>7.1f}  "
                f"{w + d + lo:>5d}  {w}-{d}-{lo}"
            )
    else:
        lines.append("  (no games recorded)")
    return "\n".join(lines)


# ----------------------------------------------------------------------- sprt
def run_sprt(
    store: TournamentStore,
    name: str,
    entry_a: Dict[str, Any],
    entry_b: Dict[str, Any],
    *,
    elo0: float,
    elo1: float,
    alpha: float,
    beta: float,
    max_games: int,
    level: int,
    seed: int,
    runner: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Sequential A-vs-B SPRT; records every game; resumes by tournament name."""

    roster = {"entries": [entry_a, entry_b]}
    tid = store.get_or_create_tournament(name, roster, level, max_games, seed)
    na, nb = str(entry_a["name"]), str(entry_b["name"])
    lower, upper = sprt_bounds(alpha, beta)

    W = D = L = 0
    for r in store.games_for(tid):
        if r["winner"] == "a":
            W += 1
        elif r["winner"] == "b":
            L += 1
        else:
            D += 1
    done = store.recorded_game_idxs(tid, na, nb)

    def _status() -> Tuple[float, Optional[str]]:
        llr = sprt_llr(W, D, L, elo0, elo1)
        if llr >= upper:
            return llr, "H1"
        if llr <= lower:
            return llr, "H0"
        return llr, None

    llr, verdict = _status()
    if verdict is None and len(done) < max_games:
        specs = make_specs(seed, na, nb, [k for k in range(max_games) if k not in done])
        series = runner.play(entry_a, entry_b, specs)
        for res in series:
            store.insert_game(tid, na, nb, res)
            if res.winner == "a":
                W += 1
            elif res.winner == "b":
                L += 1
            else:
                D += 1
            llr, verdict = _status()
            if verbose:
                print(
                    f"[{W + D + L}/{max_games}] W{W} D{D} L{L}  llr={llr:+.3f}  "
                    f"bounds=[{lower:+.3f}, {upper:+.3f}]",
                    flush=True,
                )
            if verdict is not None:
                series.close()
                break

    out = {
        "tournament": name,
        "a": na,
        "b": nb,
        "elo0": elo0,
        "elo1": elo1,
        "alpha": alpha,
        "beta": beta,
        "games": W + D + L,
        "W": W,
        "D": D,
        "L": L,
        "llr": llr,
        "lower": lower,
        "upper": upper,
        "verdict": verdict or "inconclusive",
    }
    if verbose:
        meaning = {
            "H1": f"accept H1: elo(A-B) >= {elo1}",
            "H0": f"accept H0: elo(A-B) <= {elo0}",
            "inconclusive": "inconclusive (max games reached)",
        }[out["verdict"]]
        print(f"verdict: {meaning}  after {out['games']} games  llr={llr:+.3f}")
    return out


# ------------------------------------------------------------------------ cli
def parse_entry_spec(spec: str) -> Dict[str, Any]:
    """`NAME=CHECKPOINT[,mode=plain|search|ponder][,beam=8][,deadline_ms=60]...`"""

    if "=" not in spec:
        raise SystemExit(f"entry spec '{spec}' must look like NAME=CHECKPOINT[,k=v...]")
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    entry: Dict[str, Any] = {"name": name.strip(), "checkpoint": parts[0].strip()}
    params: Dict[str, Any] = {}
    for kv in parts[1:]:
        if "=" not in kv:
            raise SystemExit(f"bad entry param '{kv}' in '{spec}' (expected k=v)")
        k, v = kv.split("=", 1)
        k = k.strip()
        if k == "mode":
            entry["mode"] = v.strip()
        elif k in {"beam", "seed"}:
            params[k] = int(v)
        elif k in {"deadline_ms", "ponder_budget_s"}:
            params[k] = float(v)
        else:
            params[k] = v.strip()
    entry.setdefault("mode", "plain")
    if params:
        entry["params"] = params
    return entry


def load_roster(path: Path) -> Dict[str, Any]:
    import yaml

    roster = yaml.safe_load(path.read_text())
    if not isinstance(roster, dict) or "entries" not in roster:
        raise SystemExit(f"roster {path} must be a mapping with an 'entries' list")
    return roster


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="tools.tournament",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", type=str, default=str(DEFAULT_DB))

    def add_run_common(p: argparse.ArgumentParser) -> None:
        add_common(p)
        p.add_argument("--level", type=int, default=14)
        p.add_argument("--speed-setting", type=int, default=2, help="2 = HI")
        p.add_argument("--seed", type=int, default=12345, help="tournament seed (schedule)")
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--threads", type=int, default=4)

    p_run = sub.add_parser("run", help="round-robin tournament over a roster yaml")
    p_run.add_argument("--roster", type=str, required=True)
    p_run.add_argument("--games-per-pair", type=int, required=True)
    p_run.add_argument("--name", type=str, default=None, help="default: roster file stem")
    p_run.add_argument("--pairs", type=int, default=4, help="parallel env pairs (ponder: use 1)")
    add_run_common(p_run)

    p_rep = sub.add_parser("report", help="pairwise matrix + Elo MLE table")
    p_rep.add_argument("--tournament", type=str, required=True)
    add_common(p_rep)

    p_sprt = sub.add_parser("sprt", help="sequential A-vs-B SPRT change gate")
    p_sprt.add_argument("--a", type=str, required=True, help="NAME=CHECKPOINT[,k=v...]")
    p_sprt.add_argument("--b", type=str, required=True, help="NAME=CHECKPOINT[,k=v...]")
    p_sprt.add_argument("--elo0", type=float, default=0.0)
    p_sprt.add_argument("--elo1", type=float, default=5.0)
    p_sprt.add_argument("--alpha", type=float, default=0.05)
    p_sprt.add_argument("--beta", type=float, default=0.05)
    p_sprt.add_argument("--max-games", type=int, default=400)
    p_sprt.add_argument("--name", type=str, default=None, help="default: sprt_<a>_vs_<b>_<date>")
    add_run_common(p_sprt)

    args = ap.parse_args(argv)
    store = TournamentStore(args.db)

    if args.cmd == "report":
        print(render_report(store, args.tournament))
        return

    if args.cmd == "run":
        roster_path = Path(args.roster)
        roster = load_roster(roster_path)
        name = args.name or roster_path.stem
        runner = VsMatchRunner(
            level=args.level,
            speed_setting=args.speed_setting,
            num_pairs=args.pairs,
            device=args.device,
            threads=args.threads,
            run_seed=args.seed,
            state_repr=pick_state_repr(roster["entries"]),
        )
        t0 = time.perf_counter()
        try:
            run_tournament(
                store,
                name,
                roster,
                games_per_pair=args.games_per_pair,
                level=args.level,
                seed=args.seed,
                runner=runner,
            )
        finally:
            runner.close()
        print(f"done in {time.perf_counter() - t0:.0f}s")
        print()
        print(render_report(store, name))
        return

    if args.cmd == "sprt":
        entry_a = parse_entry_spec(args.a)
        entry_b = parse_entry_spec(args.b)
        name = args.name or (
            f"sprt_{entry_a['name']}_vs_{entry_b['name']}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        )
        runner = VsMatchRunner(
            level=args.level,
            speed_setting=args.speed_setting,
            num_pairs=1,  # strictly sequential: test after every game
            device=args.device,
            threads=args.threads,
            run_seed=args.seed,
            state_repr=pick_state_repr([entry_a, entry_b]),
        )
        try:
            out = run_sprt(
                store,
                name,
                entry_a,
                entry_b,
                elo0=args.elo0,
                elo1=args.elo1,
                alpha=args.alpha,
                beta=args.beta,
                max_games=args.max_games,
                level=args.level,
                seed=args.seed,
                runner=runner,
            )
        finally:
            runner.close()
        print(json.dumps(out, indent=2))
        return


if __name__ == "__main__":
    main()
