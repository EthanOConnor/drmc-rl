from __future__ import annotations

import glob

import numpy as np
import pytest

from seedlab.db import CatalogDB


def _is_pool_present() -> bool:
    try:
        from envs.backends.drmario_pool import is_library_present

        return bool(is_library_present())
    except Exception:
        return False


def _latest_checkpoint() -> str | None:
    paths = sorted(glob.glob("runs/best_agents/*.pt.gz"))
    return paths[-1] if paths else None


pool_required = pytest.mark.skipif(
    not _is_pool_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)
ckpt_required = pytest.mark.skipif(
    not _is_pool_present() or _latest_checkpoint() is None,
    reason="needs cpp-pool library and a runs/best_agents checkpoint",
)


def test_level_weights_width_first() -> None:
    import numpy as np

    from seedlab.explore import level_weights

    levels = [0, 1, 2, 3, 4]
    w = level_weights(levels, 0)
    assert abs(w.sum() - 1.0) < 1e-9
    # Bulk on the frontier; strictly decaying above it.
    assert w[0] > 0.6
    assert all(w[i] > w[i + 1] for i in range(len(levels) - 1))

    # Frontier advanced to level 2: finished levels keep a uniform residue.
    w2 = level_weights(levels, 2)
    assert np.argmax(w2) == 2
    assert w2[0] > 0 and w2[1] > 0
    assert abs(float(w2[0]) - float(w2[1])) < 1e-9

    # Everything finished: uniform deepening.
    w3 = level_weights(levels, len(levels))
    assert np.allclose(w3, 1.0 / len(levels))


def test_fmt_frames_seconds() -> None:
    from seedlab.report import fmt_frames

    assert fmt_frames(None) == "-"
    assert fmt_frames(1313) == "1313 (21.8s)"


def test_pick_tier_share_weighted_wall_time() -> None:
    from seedlab.explore import DEFAULT_TIERS, TIER_SHARES, pick_tier

    spent: dict = {}
    # Simulate iterations: greedy 0.3s, rollouts 2s, beams 3/10/40s, exact 60s.
    cost = {"greedy x1": 300, "rollout x4": 2000, "beam w8": 3000,
            "beam w32": 10000, "beam w128": 40000, "exact": 60000,
            "polish": 20000}
    picks = []
    for _ in range(600):
        t = pick_tier(DEFAULT_TIERS, spent, level=0, exact_max_level=2)
        picks.append(t)
        spent[t] = spent.get(t, 0.0) + cost[t]
    total = sum(spent.values())
    sum_shares = sum(TIER_SHARES.get(t, 1.0) for t in DEFAULT_TIERS)
    # Wall-time share converges to share_t/Σshares (polish gets 3x).
    for t in DEFAULT_TIERS:
        expected = TIER_SHARES.get(t, 1.0) / sum_shares
        assert abs(spent[t] / total - expected) < 0.07, (t, spent[t] / total, expected)
    # Cheap tier runs far more often than the deep ones.
    assert picks.count("greedy x1") > 20 * picks.count("exact")

    # exact excluded above its level cap.
    assert pick_tier(DEFAULT_TIERS, {}, level=10, exact_max_level=2) != "exact"
    spent_hi = {t: 0.0 for t in DEFAULT_TIERS}
    spent_hi["exact"] = -1.0  # would win argmin if not excluded
    assert pick_tier(DEFAULT_TIERS, spent_hi, level=10, exact_max_level=2) != "exact"


def test_frontier_index_priority_floor() -> None:
    from seedlab.explore import frontier_index

    levels = list(range(21))
    total = 100
    nothing_covered: dict = {}
    # Priority 4: build-up anchors at 4 even though 0-3 are unfinished.
    assert frontier_index(levels, nothing_covered, total_seeds=total, priority_level=4) == 4
    # 4-6 finished -> frontier moves up to 7.
    covered = {4: 100, 5: 100, 6: 100}
    assert frontier_index(levels, covered, total_seeds=total, priority_level=4) == 7
    # Everything >= 4 finished -> falls back to the unfinished low levels.
    covered_hi = {lvl: 100 for lvl in range(4, 21)}
    covered_hi[0] = 100
    assert frontier_index(levels, covered_hi, total_seeds=total, priority_level=4) == 1
    # All finished -> len(levels) (uniform deepening).
    all_done = {lvl: 100 for lvl in levels}
    assert frontier_index(levels, all_done, total_seeds=total, priority_level=4) == len(levels)
    # No floor -> plain lowest-unfinished.
    assert frontier_index(levels, covered, total_seeds=total, priority_level=None) == 0


@ckpt_required
def test_step_bounds_admissible() -> None:
    """Engine-measured step minima must lower-bound every observed step tau."""

    import torch

    torch.set_num_threads(2)
    import numpy as np

    from seedlab.bounds import StepBounds
    from seedlab.search import SearchEngine, _aux_infos
    from seedlab.worker import Solver

    solver = Solver(
        policy="checkpoint", checkpoint=_latest_checkpoint(), device="cpu",
        temperature=0.8, rng=np.random.default_rng(3),
    )
    eng = SearchEngine(num_envs=4)
    sb = StepBounds(eng, speed_setting=2)
    b = eng.runner.buffers

    violations = []
    for level, seed_idx in ((0, 11), (2, 222), (4, 4444)):
        from seedlab import rng as slrng

        seed = slrng.orbit()[seed_idx]
        node = eng.root(level=level, speed=2, seed=seed)
        for depth in range(80):
            infos = _aux_infos([node], level=level, speed=2, v_initial=(level + 1) * 4)
            acts = solver.act(np.stack([node.obs]).astype(np.float32), infos,
                              np.zeros(1, dtype=bool))
            actions = np.zeros(4, dtype=np.int32)
            actions[0] = acts[0]
            eng.step(actions)
            if int(b.invalid_action[0]) != -1:
                break
            tau = max(1, int(b.tau_frames[0]))
            terminal = bool(b.terminated[0]) or bool(b.truncated[0])
            bound = sb.terminal(depth // 10) if terminal else sb.continuing(depth // 10)
            if tau < bound:
                violations.append((level, f"{seed:04x}", depth, tau, bound, terminal))
            if terminal:
                break
            node = eng.read_node(0, depth=node.depth + 1, g=0, trace=())
    eng.close()
    assert not violations, violations


@pool_required
def test_checkpoint_restore_matches_reset() -> None:
    from seedlab.search import SearchEngine

    eng = SearchEngine(num_envs=2)
    try:
        b = eng.runner.buffers
        root = eng.root(level=3, speed=2, seed=0x8988)
        spec = eng.node_spec(level=3, speed=2, seed=0x8988, node=root)
        mask = np.array([0, 1], dtype=np.uint8)
        eng.restore([eng._noop_spec, spec], mask)
        np.testing.assert_array_equal(
            b.board_bytes[1], np.frombuffer(root.board, dtype=np.uint8)
        )
        assert int(b.viruses_rem[1]) == root.v_rem
        np.testing.assert_array_equal(b.pill_colors[1], root.pills)
        np.testing.assert_array_equal(b.preview_colors[1], root.preview)
    finally:
        eng.close()


@ckpt_required
def test_beam_improves_and_replays() -> None:
    import torch

    torch.set_num_threads(2)
    from seedlab.search import SearchEngine, beam_search
    from seedlab.worker import Solver

    solver = Solver(
        policy="checkpoint", checkpoint=_latest_checkpoint(), device="cpu",
        temperature=0.0, rng=np.random.default_rng(0),
    )
    eng = SearchEngine(num_envs=16)
    try:
        res = beam_search(
            eng, level=0, speed=2, seed=0x8988, width=8, top_m=6,
            solver=solver, max_depth=60,
        )
        assert res.cleared and res.trace
        ok, frames, spawns = eng.replay(level=0, speed=2, seed=0x8988, trace=res.trace)
        assert ok, "beam trace must replay from a true reset"
        assert spawns == len(res.trace)
        # Replay is authoritative; search-internal frames may drift slightly.
        assert abs(frames - res.frames) <= 32
        # The trained-policy greedy rollout on this seed takes ~1600+ frames;
        # beam should land well under it.
        assert frames < 1500
    finally:
        eng.close()


@ckpt_required
def test_exact_search_respects_incumbent() -> None:
    import torch

    torch.set_num_threads(2)
    from seedlab.search import SearchEngine, beam_search, exact_search
    from seedlab.worker import Solver

    solver = Solver(
        policy="checkpoint", checkpoint=_latest_checkpoint(), device="cpu",
        temperature=0.0, rng=np.random.default_rng(0),
    )
    eng = SearchEngine(num_envs=16)
    try:
        seed = 0x8988
        beam = beam_search(
            eng, level=0, speed=2, seed=seed, width=8, top_m=6,
            solver=solver, max_depth=60,
        )
        assert beam.cleared
        res = exact_search(
            eng, level=0, speed=2, seed=seed,
            incumbent_frames=beam.frames, node_budget=4_000,
        )
        # Budget is checked per node pop; the final expansion may overshoot by
        # one node's full child count.
        assert res.nodes <= 4_000 + 128
        if res.cleared:
            assert res.frames is not None and res.frames < beam.frames
    finally:
        eng.close()


@ckpt_required
def test_explorer_iteration_smoke(tmp_path) -> None:
    import torch

    torch.set_num_threads(2)
    from seedlab.explore import Explorer

    db = CatalogDB(tmp_path / "catalog.sqlite3")
    explorer = Explorer(
        db=db, levels=[0], speed=2, checkpoint=_latest_checkpoint(),
        num_envs=8, tiers=(("rollout x4", 0.5), ("beam w8", 0.5)), seed=5,
    )
    try:
        for _ in range(2):
            r = explorer.run_iteration()
            assert r.nodes > 0
        rows = db._conn.execute("SELECT COUNT(*) FROM search_log;").fetchone()[0]
        assert int(rows) == 2

        # Dashboard renders the search-activity panel from this data.
        from rich.console import Console

        from seedlab.dashboard import build_view

        console = Console(width=140, height=40, file=open("/dev/null", "w"))
        console.print(build_view(db, 2))
    finally:
        explorer.close()
        db.close()
