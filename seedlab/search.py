"""Per-seed solution search: policy-guided beam (T2) and exact B&B (T3).

Built directly on `DrMarioPoolRunner` so arbitrary tree nodes restore in O(1)
via the pool's extraction-checkpoint reset (board + reserve index + counters).
See docs/SEEDLAB_SEARCH.md for the design and the verification invariant:
search-internal frame accounting ranks nodes, but anything recorded into the
catalog is first replayed from a true level reset and stored with the replay
frame count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import envs.specs.ram_to_state as ram_specs
from envs.backends.drmario_pool import DrMarioPoolRunner, build_reset_spec

from seedlab import rng as slrng

_OBS_SPECS = {
    "bitplane_bottle": (1, 4),
    "bitplane_bottle_mask": (2, 8),
    "bitplane_bottle_conn": (3, 8),
    "bitplane_bottle_conn_mask": (4, 12),
}

ACTION_SPACE = 4 * 16 * 8  # orient-major flat placement index


@dataclass(slots=True)
class Node:
    """A search node: `depth` pills locked, pill `depth` now falling."""

    depth: int
    g: int                      # exact frames from reset (sum of step taus)
    v_rem: int
    board: bytes                # post-settle board (engine layout)
    trace: Tuple[int, ...]      # actions taken from the root
    # Decision context captured when the node was created:
    obs: np.ndarray             # (C,16,8) float32
    mask: np.ndarray            # (512,) bool
    cost: np.ndarray            # (512,) uint16
    pills: np.ndarray           # (2,) canonical colors of falling pill
    preview: np.ndarray         # (2,) canonical colors of preview pill
    spec: object = None         # cached ctypes reset spec (built lazily)
    neg_logp: float = 0.0       # cumulative -log pi(action) along the trace


@dataclass(slots=True)
class _LightChild:
    """Phase-1 beam child: enough to rank/dedup, no planner context yet."""

    depth: int
    g: int
    v_rem: int
    board: bytes
    trace: Tuple[int, ...]
    neg_logp: float


@dataclass(slots=True)
class SearchResult:
    cleared: bool
    frames: Optional[int]
    trace: Tuple[int, ...]
    nodes: int
    wall_sec: float
    certified: bool = False
    note: str = ""


class SearchEngine:
    """Small batch engine for tree search over one (level, speed, seed)."""

    def __init__(
        self,
        *,
        num_envs: int = 64,
        state_repr: str = "bitplane_bottle_conn_mask",
        lazy: bool = False,
    ) -> None:
        if state_repr not in _OBS_SPECS:
            raise ValueError(f"unknown state_repr {state_repr!r}")
        obs_spec, channels = _OBS_SPECS[state_repr]
        ram_specs.set_state_representation(state_repr)
        self.num_envs = int(num_envs)
        self.channels = int(channels)
        # lazy: step() skips the post-action re-plan (state-only outputs).
        # Use only for phase-1 beam expansion; replay/rollouts need full steps.
        self.lazy = bool(lazy)
        self.runner = DrMarioPoolRunner(
            num_envs=self.num_envs,
            obs_spec=obs_spec,
            obs_channels=channels,
            emit_board=True,
            lazy_decision_outputs=self.lazy,
        )
        self._noop_spec = build_reset_spec()

    # ------------------------------------------------------------------ specs
    def root_spec(self, *, level: int, speed: int, seed: int):
        return build_reset_spec(
            level=int(level),
            speed_setting=int(speed),
            rng_state=slrng.seed_to_bytes(int(seed)),
            rng_override=True,
        )

    def checkpoint_spec(
        self,
        *,
        level: int,
        speed: int,
        seed: int,
        board: bytes,
        depth: int,
        mask: Optional[np.ndarray] = None,
        cost: Optional[np.ndarray] = None,
    ):
        """Checkpoint reset spec; injects the node's own planner outputs when
        mask/cost are provided so the restore cache-hits instead of replanning.

        speed_ups: base for high levels plus one bump per 10 pills played
        (approximate at the boundary; the final replay from reset is
        authoritative for anything recorded).
        """

        base_ups = max(0, int(level) - 20)
        inject = mask is not None and cost is not None
        return build_reset_spec(
            level=int(level),
            speed_setting=int(speed),
            rng_state=slrng.seed_to_bytes(int(seed)),
            rng_override=True,
            checkpoint_enabled=True,
            checkpoint_board=np.frombuffer(board, dtype=np.uint8),
            checkpoint_pill_counter=int(depth) & 0x7F,
            checkpoint_speed_ups=min(0x31, base_ups + int(depth) // 10),
            inject_plan=inject,
            inject_feasible=None if not inject else mask.astype(np.uint8),
            inject_costs=None if not inject else cost,
        )

    def node_spec(self, *, level: int, speed: int, seed: int, node: Node):
        if node.spec is not None:
            return node.spec
        node.spec = self.checkpoint_spec(
            level=level, speed=speed, seed=seed, board=node.board,
            depth=node.depth, mask=node.mask, cost=node.cost,
        )
        return node.spec

    # ---------------------------------------------------------------- restore
    def restore(self, specs: Sequence[object], mask: np.ndarray) -> None:
        """Reset envs where mask!=0 to the given specs (others untouched)."""

        full = [
            specs[i] if mask[i] else self._noop_spec for i in range(self.num_envs)
        ]
        self.runner.reset(np.asarray(mask, dtype=np.uint8), full)

    def step(self, actions: np.ndarray) -> None:
        self.runner.step(np.asarray(actions, dtype=np.int32), None, None)

    # ----------------------------------------------------------------- reads
    def read_node(self, i: int, *, depth: int, g: int, trace: Tuple[int, ...]) -> Node:
        b = self.runner.buffers
        mask = b.feasible_mask[i].astype(bool).reshape(-1).copy()
        cost = b.cost_to_lock[i].reshape(-1).copy()
        pills = b.pill_colors[i].copy()
        if pills[0] == pills[1]:
            # Same-color pill: orientations 2/3 duplicate 0/1.
            mask = mask.copy()
            mask[2 * 128 : 4 * 128] = False
        return Node(
            depth=depth,
            g=g,
            v_rem=int(b.viruses_rem[i]),
            board=bytes(b.board_bytes[i].tobytes()),
            trace=trace,
            obs=b.obs[i].copy(),
            mask=mask,
            cost=cost,
            pills=pills,
            preview=b.preview_colors[i].copy(),
        )

    def root(self, *, level: int, speed: int, seed: int) -> Node:
        mask = np.zeros(self.num_envs, dtype=np.uint8)
        mask[0] = 1
        self.restore([self.root_spec(level=level, speed=speed, seed=seed)]
                     + [self._noop_spec] * (self.num_envs - 1), mask)
        return self.read_node(0, depth=0, g=0, trace=())

    def close(self) -> None:
        self.runner.close()

    # ---------------------------------------------------------------- replay
    def replay(
        self, *, level: int, speed: int, seed: int, trace: Sequence[int]
    ) -> Tuple[bool, int, int]:
        """Replay a trace from a true reset. Returns (cleared, frames, spawns)."""

        mask = np.zeros(self.num_envs, dtype=np.uint8)
        mask[0] = 1
        self.restore([self.root_spec(level=level, speed=speed, seed=seed)]
                     + [self._noop_spec] * (self.num_envs - 1), mask)
        b = self.runner.buffers
        frames = 0
        actions = np.zeros(self.num_envs, dtype=np.int32)
        for k, action in enumerate(trace):
            if not bool(b.feasible_mask[0].reshape(-1)[int(action)]):
                return False, frames, k
            actions[0] = int(action)
            self.step(actions)
            if int(b.invalid_action[0]) != -1:
                return False, frames, k
            frames += max(1, int(b.tau_frames[0]))
            if bool(b.terminated[0]) or bool(b.truncated[0]):
                cleared = int(b.viruses_rem[0]) == 0
                return cleared and k == len(trace) - 1, frames, k + 1
        return False, frames, len(trace)


_CANONICAL_TO_RAW = np.array([1, 0, 2, 0], dtype=np.int64)  # R,Y,B -> NES raw


def _aux_infos(nodes: Sequence[Node], *, level: int, speed: int, v_initial: int) -> List[dict]:
    infos = []
    for n in nodes:
        infos.append(
            {
                "pill/speed_setting": int(speed),
                "level": int(level),
                "task/frames_used": int(n.g),
                "task_mode": "viruses",
                "drm/viruses_initial": int(v_initial),
                "viruses_remaining": int(n.v_rem),
                "placements/options": int(n.mask.sum()),
                "placements/feasible_mask": n.mask.reshape(4, 16, 8),
                "placements/cost_to_lock": n.cost.reshape(4, 16, 8),
                "next_pill_colors": n.pills.astype(np.int64),
                # Vec env exposes preview in raw NES colors; match it.
                "preview_pill": _CANONICAL_TO_RAW[n.preview.astype(np.int64) & 0x03],
            }
        )
    return infos


def _select_children(
    nodes: Sequence[Node], *, top_m: int, solver=None, level: int, speed: int,
    v_initial: int, prior_noise: float = 0.0, noise_rng=None,
) -> List[List[Tuple[int, float]]]:
    """Per node, the top-M candidate (action, -log pi) pairs.

    Falls back to cost-to-lock ordering (with -log pi = 0) without a
    checkpoint policy; that mode is for debugging only — it shares
    greedy-cost's blindness to virus progress.
    """

    if solver is not None and solver.policy == "checkpoint":
        obs = np.stack([n.obs for n in nodes]).astype(np.float32)
        infos = _aux_infos(nodes, level=level, speed=speed, v_initial=v_initial)
        cand_actions, cand_mask, logits = solver.candidate_logits(obs, infos)
        lg = logits.numpy().astype(np.float64)
        lg[~cand_mask] = -np.inf
        # Masked log-softmax per node for -log pi(action).
        mx = np.max(lg, axis=1, keepdims=True)
        z = np.exp(lg - mx)
        logp = lg - (mx + np.log(np.maximum(z.sum(axis=1, keepdims=True), 1e-300)))
        # Diversification (record polish): Gumbel-perturbed prior ordering so
        # repeat visits explore different subtrees; -log pi stays unperturbed.
        order_scores = lg
        if prior_noise > 0.0 and noise_rng is not None:
            u = noise_rng.random(lg.shape)
            gumbel = -np.log(-np.log(np.clip(u, 1e-12, 1.0 - 1e-12)))
            order_scores = lg + float(prior_noise) * gumbel
            order_scores[~cand_mask] = -np.inf
        out: List[List[Tuple[int, float]]] = []
        for i, n in enumerate(nodes):
            order = np.argsort(-order_scores[i])
            acts: List[Tuple[int, float]] = []
            for slot in order:
                if not cand_mask[i, slot]:
                    break
                a = int(cand_actions[i, slot])
                if n.mask[a]:  # respect symmetry reduction
                    acts.append((a, float(-logp[i, slot])))
                if len(acts) >= top_m:
                    break
            out.append(acts)
        return out

    out = []
    for n in nodes:
        feas = np.flatnonzero(n.mask)
        order = feas[np.argsort(n.cost[feas].astype(np.int64))]
        out.append([(int(a), 0.0) for a in order[:top_m]])
    return out


def beam_search(
    engine: SearchEngine,
    *,
    level: int,
    speed: int,
    seed: int,
    width: int = 32,
    top_m: int = 8,
    solver=None,
    max_depth: int = 160,
    lambda_frames: float = 150.0,
    kappa_frames: float = 35.0,
    node_budget: int = 200_000,
    prior_noise: float = 0.0,
    noise_rng=None,
) -> SearchResult:
    """Anytime policy-guided beam over (board, pill_idx) with exact g-costs.

    Ranking: g + λ·viruses_remaining + κ·Σ(-log π). The κ term keeps
    policy-favored paths competitive against frame-cheap junk placements —
    without it the constant-virus-count frontier degenerates into tau-greedy
    stacking that never clears.

    Two-phase expansion (fast path on a `lazy=True` engine): phase 1 steps
    every child with planning skipped (state-only outputs: tau, viruses,
    board, terminal); phase 2 plans only the ≤W frontier survivors via an
    injected-checkpoint restore. On a non-lazy engine the same code runs
    correctly, just without the deferred-planning savings.
    """

    t0 = time.perf_counter()
    root = engine.root(level=level, speed=speed, seed=seed)
    v_initial = root.v_rem
    frontier = [root]
    best: Optional[Tuple[int, Tuple[int, ...]]] = None
    nodes_expanded = 0
    E = engine.num_envs
    b = engine.runner.buffers

    for _depth in range(max_depth):
        if not frontier or nodes_expanded >= node_budget:
            break
        child_lists = _select_children(
            frontier, top_m=top_m, solver=solver, level=level, speed=speed,
            v_initial=v_initial, prior_noise=prior_noise, noise_rng=noise_rng,
        )
        jobs: List[Tuple[Node, int, float]] = [
            (node, a, nlp) for node, acts in zip(frontier, child_lists) for a, nlp in acts
        ]
        # Phase 1: cheap expansion — collect (g, v_rem, board, trace) per child.
        lights: List[_LightChild] = []
        seen: Dict[bytes, int] = {}

        for ofs in range(0, len(jobs), E):
            batch = jobs[ofs : ofs + E]
            mask = np.zeros(E, dtype=np.uint8)
            specs: List[object] = []
            actions = np.zeros(E, dtype=np.int32)
            for i, (node, action, _nlp) in enumerate(batch):
                mask[i] = 1
                specs.append(engine.node_spec(level=level, speed=speed, seed=seed, node=node))
                actions[i] = int(action)
            specs.extend([engine._noop_spec] * (E - len(batch)))
            engine.restore(specs, mask)
            engine.step(actions)
            nodes_expanded += len(batch)

            for i, (node, action, nlp) in enumerate(batch):
                if int(b.invalid_action[i]) != -1:
                    continue
                tau = max(1, int(b.tau_frames[i]))
                g = node.g + tau
                trace = node.trace + (int(action),)
                if bool(b.terminated[i]) or bool(b.truncated[i]):
                    if int(b.viruses_rem[i]) == 0:
                        if best is None or g < best[0]:
                            best = (g, trace)
                    continue
                if best is not None and g >= best[0]:
                    continue  # cannot beat the incumbent; prune
                child = _LightChild(
                    depth=node.depth + 1,
                    g=g,
                    v_rem=int(b.viruses_rem[i]),
                    board=bytes(b.board_bytes[i].tobytes()),
                    trace=trace,
                    neg_logp=node.neg_logp + float(nlp),
                )
                key = child.board
                prev = seen.get(key)
                if prev is not None and lights[prev].g <= g:
                    continue
                if prev is not None:
                    lights[prev] = child
                else:
                    seen[key] = len(lights)
                    lights.append(child)

        lights.sort(
            key=lambda n: n.g + lambda_frames * n.v_rem + kappa_frames * n.neg_logp
        )
        survivors = lights[: int(width)]

        # Phase 2: plan only the survivors (checkpoint restore replans).
        frontier = []
        for ofs in range(0, len(survivors), E):
            batch = survivors[ofs : ofs + E]
            mask = np.zeros(E, dtype=np.uint8)
            specs = []
            for i, lc in enumerate(batch):
                mask[i] = 1
                specs.append(
                    engine.checkpoint_spec(
                        level=level, speed=speed, seed=seed,
                        board=lc.board, depth=lc.depth,
                    )
                )
            specs.extend([engine._noop_spec] * (E - len(batch)))
            engine.restore(specs, mask)
            for i, lc in enumerate(batch):
                node = engine.read_node(i, depth=lc.depth, g=lc.g, trace=lc.trace)
                node.neg_logp = lc.neg_logp
                if node.mask.any():
                    frontier.append(node)

    return SearchResult(
        cleared=best is not None,
        frames=None if best is None else int(best[0]),
        trace=() if best is None else best[1],
        nodes=nodes_expanded,
        wall_sec=time.perf_counter() - t0,
    )


def _pills_lower_bound(v_rem: int) -> int:
    # One pill = 2 halves; one half can participate in clearing ≤3 viruses.
    return max(1, -(-int(v_rem) // 6))


_TILE_VIRUS_TYPE = 0xD0
_MASK_TYPE = 0xF0


def _pills_lower_bound_board(board: bytes, v_rem: int) -> int:
    """Board-aware admissible pills bound.

    Per color: group viruses into "line components" — two same-color viruses
    can only share a clear event if they are colinear within a span of ≤3
    cells; each component needs ≥1 clear event, and every clear line has 4
    cells of one color, so a component with k viruses in its clearing line
    consumes ≥(4−min(k,3)) placed halves per event. Non-virus line cells are
    always caller-placed (the board starts virus-only; clears only remove),
    and one half can serve at most 2 crossing lines, so
    halves ≥ Σ_events (4 − viruses_in_line) / 2, pills ≥ ceil(halves/2).
    """

    pos = [[], [], []]
    for idx, t in enumerate(board):
        if (t & _MASK_TYPE) == _TILE_VIRUS_TYPE:
            pos[t & 0x03].append((idx >> 3, idx & 7))  # (row_idx, col)

    deficit_halves = 0
    for cells in pos:
        n = len(cells)
        if n == 0:
            continue
        # Union-find over could-share-a-line pairs.
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                (r1, c1), (r2, c2) = cells[i], cells[j]
                if (r1 == r2 and abs(c1 - c2) <= 3) or (c1 == c2 and abs(r1 - r2) <= 3):
                    parent[find(i)] = find(j)
        comp_sizes: Dict[int, int] = {}
        for i in range(n):
            root = find(i)
            comp_sizes[root] = comp_sizes.get(root, 0) + 1
        for size in comp_sizes.values():
            deficit_halves += 4 - min(size, 3)

    pills_from_lines = -(-deficit_halves // 4)  # /2 for cross-serving, /2 halves->pills
    return max(_pills_lower_bound(v_rem), pills_from_lines)


def exact_search(
    engine: SearchEngine,
    *,
    level: int,
    speed: int,
    seed: int,
    incumbent_frames: Optional[int] = None,
    node_budget: int = 500_000,
    bounds=None,
    lambda_frames: float = 150.0,
) -> SearchResult:
    """DFS branch-and-bound. Certified optimal iff the search closes in budget.

    The pruning bound uses engine-measured minimal step frames
    (seedlab.bounds.StepBounds): the immediate step takes
    max(exact min cost_to_lock at this node, continuing minimum); the
    remaining ≥ ceil(viruses/6)−1 pills each cost at least the continuing
    minimum at their exact speed-ups schedule, except the final one, which
    may end the stage at the (smaller) terminal minimum.
    """

    from seedlab.bounds import SPEEDUPS_MAX, StepBounds

    t0 = time.perf_counter()
    if bounds is None:
        bounds = StepBounds(engine, speed_setting=speed)
    base_ups = max(0, int(level) - 20)

    def ups_at(depth: int) -> int:
        return min(SPEEDUPS_MAX, base_ups + int(depth) // 10)

    def lower_bound(node: Node, min_cost_now: int) -> int:
        L = _pills_lower_bound_board(node.board, node.v_rem)
        if L <= 1:
            return node.g + min_cost_now
        future = 0
        for i in range(1, L):
            u = ups_at(node.depth + i)
            future += bounds.terminal(u) if i == L - 1 else bounds.continuing(u)
        immediate = max(min_cost_now, bounds.continuing(ups_at(node.depth)))
        return node.g + immediate + future

    root = engine.root(level=level, speed=speed, seed=seed)
    E = engine.num_envs
    b = engine.runner.buffers

    best_frames = incumbent_frames
    best_trace: Tuple[int, ...] = ()
    nodes = 0
    exhausted = True
    # Transposition table: board+depth -> best g seen.
    tt: Dict[bytes, int] = {}

    stack: List[Node] = [root]
    while stack:
        if nodes >= node_budget:
            exhausted = False
            break
        node = stack.pop()
        if node.v_rem <= 0:
            continue
        if best_frames is not None:
            min_step = int(node.cost[node.mask].min()) if node.mask.any() else 0
            if lower_bound(node, min_step) >= best_frames:
                continue
        key = node.board + bytes([node.depth & 0xFF])
        prev = tt.get(key)
        if prev is not None and prev <= node.g:
            continue
        tt[key] = node.g

        feas = np.flatnonzero(node.mask)
        if feas.size == 0:
            continue
        order = feas[np.argsort(node.cost[feas].astype(np.int64))]
        children: List[Node] = []
        for ofs in range(0, len(order), E):
            batch_actions = order[ofs : ofs + E]
            mask = np.zeros(E, dtype=np.uint8)
            specs: List[object] = []
            actions = np.zeros(E, dtype=np.int32)
            for i, action in enumerate(batch_actions):
                mask[i] = 1
                specs.append(engine.node_spec(level=level, speed=speed, seed=seed, node=node))
                actions[i] = int(action)
            specs.extend([engine._noop_spec] * (E - len(batch_actions)))
            engine.restore(specs, mask)
            engine.step(actions)
            nodes += len(batch_actions)

            for i, action in enumerate(batch_actions):
                if int(b.invalid_action[i]) != -1:
                    continue
                tau = max(1, int(b.tau_frames[i]))
                g = node.g + tau
                if best_frames is not None and g >= best_frames:
                    continue
                trace = node.trace + (int(action),)
                if bool(b.terminated[i]) or bool(b.truncated[i]):
                    if int(b.viruses_rem[i]) == 0:
                        best_frames = g
                        best_trace = trace
                    continue
                children.append(
                    engine.read_node(i, depth=node.depth + 1, g=g, trace=trace)
                )
        # Exploration order is free (soundness rests on the bound alone):
        # favor virus progress so the DFS finds clears early and tightens the
        # incumbent — cheapest-g-first degenerates into never-clearing
        # tau-greedy wandering. Stack pops the last element, so push worst
        # first.
        children.sort(key=lambda n: n.g + lambda_frames * n.v_rem, reverse=True)
        stack.extend(children)

    return SearchResult(
        cleared=best_trace != (),
        frames=best_frames if best_trace != () else None,
        trace=best_trace,
        nodes=nodes,
        wall_sec=time.perf_counter() - t0,
        certified=exhausted,
        note="closed" if exhausted else "budget exhausted",
    )
