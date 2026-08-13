"""Search-amplified training targets (docs/SEARCH_DISTILL.md).

Unit tests for the improved-policy math, KL masking, value blending, and the
flag-off bit-identity regression guard; plus tiny end-to-end rollout+update
smokes on the 1P pool env and the VS pool env (skipped without the native
library).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gymnasium import spaces

import drmc_rl.game.specs.ram_to_state as ram_specs
from drmc_rl.envs.backends.drmario_pool import is_library_present
from drmc_rl.models.policy.candidate_packing import pack_feasible_candidates
from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter
from drmc_rl.training.algo.search_distill import (
    SearchDistillConfig,
    blend_value_targets,
    improved_policy_target,
    masked_distill_kl,
)
from drmc_rl.training.rollout.decision_buffer import DecisionBatch
from drmc_rl.training.utils.cfg import to_config_node

needs_pool = pytest.mark.skipif(
    not is_library_present(),
    reason="cpp-pool library missing (build with: python -m tools.build_drmario_pool)",
)


def _has_vspool_symbols() -> bool:
    if not is_library_present():
        return False
    try:
        import ctypes

        from drmc_rl.envs.backends.drmario_pool import resolve_library_path

        lib = ctypes.CDLL(str(resolve_library_path()))
        return hasattr(lib, "drm_vspool_create")
    except Exception:
        return False


needs_vspool = pytest.mark.skipif(
    not _has_vspool_symbols(),
    reason="cpp-vs-pool library missing (build with: make -C vendor/drmario_native libdrmario_pool)",
)


# ------------------------------------------------------ improved-policy math
def test_improved_policy_monotone_in_q() -> None:
    K = 8
    prior = np.zeros(K)
    mask = np.ones(K, dtype=bool)
    q = np.linspace(0.0, 1.0, K)
    evaluated = np.ones(K, dtype=bool)
    target, qc = improved_policy_target(prior, mask, q, evaluated, baseline=0.0, sigma_scale=2.0)
    assert target.shape == (K,)
    assert np.all(np.diff(target) > 0)  # equal priors: higher Q => higher prob
    np.testing.assert_allclose(qc, q)
    assert target.sum() == pytest.approx(1.0, abs=1e-6)


def test_improved_policy_mask_and_normalization() -> None:
    K = 6
    rng = np.random.default_rng(0)
    prior = rng.normal(size=K)
    mask = np.array([True, True, False, True, False, True])
    prior[~mask] = -np.inf  # padding slots, as packed
    q = rng.normal(size=K)
    evaluated = np.array([True, False, False, True, False, False])
    target, _ = improved_policy_target(prior, mask, q, evaluated, baseline=0.1, sigma_scale=1.0)
    assert np.all(target[~mask] == 0.0)
    assert target.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(target[mask] > 0.0)


def test_improved_policy_baseline_fills_unevaluated() -> None:
    K = 4
    prior = np.zeros(K)
    mask = np.ones(K, dtype=bool)
    q = np.array([5.0, 0.0, 0.0, 0.0])
    evaluated = np.array([True, False, False, False])
    _, qc = improved_policy_target(prior, mask, q, evaluated, baseline=-2.0, sigma_scale=1.0)
    np.testing.assert_allclose(qc, [5.0, -2.0, -2.0, -2.0])


def test_improved_policy_sigma_zero_or_flat_q_recovers_prior() -> None:
    K = 5
    rng = np.random.default_rng(1)
    prior = rng.normal(size=K)
    mask = np.ones(K, dtype=bool)
    q = rng.normal(size=K)
    evaluated = np.ones(K, dtype=bool)
    prior_sm = np.exp(prior - prior.max())
    prior_sm /= prior_sm.sum()
    # sigma_scale = 0: Q is ignored.
    t0, _ = improved_policy_target(prior, mask, q, evaluated, baseline=0.0, sigma_scale=0.0)
    np.testing.assert_allclose(t0, prior_sm, atol=1e-6)
    # Flat Q: normalization is degenerate, target falls back to the prior.
    t1, _ = improved_policy_target(
        prior, mask, np.full(K, 3.3), evaluated, baseline=3.3, sigma_scale=10.0
    )
    np.testing.assert_allclose(t1, prior_sm, atol=1e-6)


def test_improved_policy_evaluated_beats_prior_order() -> None:
    # A low-prior candidate with a much higher Q should overtake at large c.
    prior = np.array([2.0, 0.0])
    mask = np.ones(2, dtype=bool)
    q = np.array([0.0, 1.0])
    evaluated = np.ones(2, dtype=bool)
    t, _ = improved_policy_target(prior, mask, q, evaluated, baseline=0.0, sigma_scale=4.0)
    assert t[1] > t[0]


# --------------------------------------------------------------- KL masking
def test_masked_kl_unsearched_rows_contribute_zero() -> None:
    B, K = 4, 6
    rng = np.random.default_rng(2)
    logits = torch.from_numpy(rng.normal(size=(B, K))).float()
    log_p = torch.log_softmax(logits, dim=-1)
    target = torch.zeros((B, K))
    target[0] = torch.softmax(torch.from_numpy(rng.normal(size=K)).float(), dim=-1)
    target[2] = torch.softmax(torch.from_numpy(rng.normal(size=K)).float(), dim=-1)
    searched = torch.tensor([True, False, True, False])

    kl_sum, n = masked_distill_kl(target, log_p, searched)
    assert float(n) == 2.0
    # Reference: row-wise KL on the searched rows only.
    ref = 0.0
    for b in (0, 2):
        t = target[b]
        ref += float((t * (torch.log(t) - log_p[b])).sum())
    assert float(kl_sum) == pytest.approx(ref, abs=1e-5)
    assert float(kl_sum) >= 0.0

    # Garbage rows with searched=False never leak in.
    target[1] = 99.0
    kl_sum2, n2 = masked_distill_kl(target, log_p, searched)
    assert float(kl_sum2) == pytest.approx(float(kl_sum), abs=1e-6)
    assert float(n2) == 2.0


def test_masked_kl_zero_when_target_matches_net() -> None:
    B, K = 2, 5
    logits = torch.randn(B, K)
    log_p = torch.log_softmax(logits, dim=-1)
    target = torch.softmax(logits, dim=-1)
    kl_sum, n = masked_distill_kl(target, log_p, torch.ones(B, dtype=torch.bool))
    assert float(n) == float(B)
    assert float(kl_sum) == pytest.approx(0.0, abs=1e-5)


# ----------------------------------------------------------- value blending
def test_blend_value_targets() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0])
    search_vals = torch.tensor([5.0, 5.0, 5.0])
    searched = torch.tensor([True, False, True])
    out = blend_value_targets(returns, search_vals, searched, 0.5)
    np.testing.assert_allclose(out.numpy(), [3.0, 2.0, 4.0])
    out0 = blend_value_targets(returns, search_vals, searched, 0.0)
    np.testing.assert_allclose(out0.numpy(), returns.numpy())


# --------------------------------------------------------------- config gate
def test_config_validation() -> None:
    assert SearchDistillConfig.from_dict(None).enabled is False
    assert SearchDistillConfig.from_dict({}).enabled is False
    # Invalid values only rejected when enabled.
    SearchDistillConfig.from_dict({"decision_fraction": 0.0})
    with pytest.raises(ValueError):
        SearchDistillConfig.from_dict({"enabled": True, "decision_fraction": 0.0})
    with pytest.raises(ValueError):
        SearchDistillConfig.from_dict({"enabled": True, "value_mix": 1.5})
    with pytest.raises(ValueError):
        SearchDistillConfig.from_dict({"enabled": True, "sims": 0})


# ------------------------------------------------- flag-off regression guard
class _StubEnv:
    num_envs = 2
    single_observation_space = spaces.Box(low=0.0, high=1.0, shape=(12, 16, 8), dtype=np.float32)
    observation_space = single_observation_space


class _StubLogger:
    def log_scalar(self, *a, **k):
        pass

    def log_scalars(self, *a, **k):
        pass

    def flush(self):
        pass


class _StubBus:
    def __init__(self):
        self.events = []

    def emit(self, name, **payload):
        self.events.append((name, payload))


def _adapter_cfg(tmp_path, extra_smdp=None) -> object:
    smdp = {
        "policy_type": "candidate",
        "aux_spec": "v1",
        "lr": 1e-3,
        "gamma": 0.99,
        "decisions_per_update": 8,
        "num_epochs": 2,
        "minibatch_size": 4,
        "pill_embed_dim": 16,
        "pill_embed_type": "ordered_pair",
        "candidate_max_candidates": 32,
        "candidate_d_model": 32,
        "candidate_pos_embed_dim": 8,
        "candidate_cost_embed_dim": 8,
        "candidate_hidden_dim": 32,
        "candidate_board_channels": 8,
        "candidate_transformer_layers": 1,
        "candidate_transformer_heads": 2,
        "candidate_patch_kernel": 3,
    }
    smdp.update(extra_smdp or {})
    return to_config_node(
        {
            "seed": 0,
            "logdir": str(tmp_path),
            "train": {"total_steps": 1, "checkpoint_interval": 10**9},
            "smdp_ppo": smdp,
            "env": {"state_repr": "bitplane_bottle_conn_mask"},
        }
    )


def _synthetic_batch(T: int, kmax: int, seed: int = 0) -> DecisionBatch:
    rng = np.random.default_rng(seed)
    obs = (rng.random((T, 12, 16, 8)) < 0.15).astype(np.float32)
    masks = np.zeros((T, 4, 16, 8), dtype=bool)
    costs = np.full((T, 4, 16, 8), np.inf, dtype=np.float32)
    actions = np.zeros(T, dtype=np.int64)
    for t in range(T):
        n = int(rng.integers(3, 12))
        flat = rng.choice(512, size=n, replace=False)
        m = np.zeros(512, dtype=bool)
        m[flat] = True
        masks[t] = m.reshape(4, 16, 8)
        c = np.full(512, np.inf, dtype=np.float32)
        c[flat] = rng.integers(1, 200, size=n).astype(np.float32)
        costs[t] = c.reshape(4, 16, 8)
        packed = pack_feasible_candidates(masks[t], costs[t], max_candidates=kmax)
        actions[t] = int(packed.actions[int(rng.integers(0, packed.count))])
    return DecisionBatch(
        observations=obs,
        masks=masks,
        costs_to_lock=costs,
        pill_colors=rng.integers(0, 3, size=(T, 2)).astype(np.int64),
        preview_pill_colors=rng.integers(0, 3, size=(T, 2)).astype(np.int64),
        aux=rng.random((T, 57)).astype(np.float32),
        actions=actions,
        log_probs=rng.normal(-2.0, 0.3, size=T).astype(np.float32),
        values=rng.normal(size=T).astype(np.float32),
        taus=np.full(T, 20, dtype=np.int32),
        rewards=rng.normal(size=T).astype(np.float32),
        observations_next=obs.copy(),
        dones=np.zeros(T, dtype=bool),
        env_ids=np.arange(T, dtype=np.int32) % 2,
        advantages=rng.normal(size=T).astype(np.float32),
        returns=rng.normal(size=T).astype(np.float32),
        gammas=np.full(T, 0.99**20, dtype=np.float32),
    )


def _make_adapter(tmp_path, extra_smdp=None) -> SMDPPPOAdapter:
    prev_repr = ram_specs.get_state_representation()
    ram_specs.set_state_representation("bitplane_bottle_conn_mask")
    try:
        torch.manual_seed(1234)
        return SMDPPPOAdapter(
            _adapter_cfg(tmp_path, extra_smdp), _StubEnv(), _StubLogger(), _StubBus(), device="cpu"
        )
    finally:
        ram_specs.set_state_representation(prev_repr)


def test_flag_off_update_is_bit_identical(tmp_path) -> None:
    """enabled:false (and no config key at all) must match plain PPO exactly."""

    kmax = 32
    variants = [
        None,  # no search_distill key at all (pre-feature behavior)
        {"search_distill": {"enabled": False}},
        {"search_distill": {"enabled": False, "beta": 7.0, "value_mix": 0.9}},
    ]
    results = []
    for extra in variants:
        adapter = _make_adapter(tmp_path, extra)
        batch = _synthetic_batch(T=8, kmax=kmax, seed=3)
        torch.manual_seed(99)  # randperm stream inside _update_policy
        metrics = adapter._update_policy(batch)
        results.append((metrics, {k: v.clone() for k, v in adapter.net.state_dict().items()}))

    m0, sd0 = results[0]
    for m, sd in results[1:]:
        for key in ("loss/policy", "loss/value", "loss/total", "policy/entropy"):
            assert m[key] == m0[key], key
        for k in sd0:
            assert torch.equal(sd0[k], sd[k]), k
        assert not any(k.startswith("search_distill/") for k in m)


def test_enabled_with_zero_searched_matches_disabled(tmp_path) -> None:
    """A batch with no searched decisions must not perturb the update."""

    kmax = 32
    adapter_off = _make_adapter(tmp_path, None)
    batch_off = _synthetic_batch(T=8, kmax=kmax, seed=3)
    torch.manual_seed(99)
    m_off = adapter_off._update_policy(batch_off)

    adapter_on = _make_adapter(tmp_path, None)
    # Enable the loss path without building the rollout search runner.
    adapter_on.search_distill_cfg = SearchDistillConfig.from_dict({"enabled": True})
    batch_on = _synthetic_batch(T=8, kmax=kmax, seed=3)
    batch_on.search_targets = np.zeros((8, kmax), dtype=np.float32)
    batch_on.search_values = np.zeros(8, dtype=np.float32)
    batch_on.search_mask = np.zeros(8, dtype=bool)
    torch.manual_seed(99)
    m_on = adapter_on._update_policy(batch_on)

    assert m_on["loss/total"] == m_off["loss/total"]
    for k in adapter_off.net.state_dict():
        assert torch.equal(adapter_off.net.state_dict()[k], adapter_on.net.state_dict()[k]), k


def test_enabled_searched_batch_adds_kl_term(tmp_path) -> None:
    """Searched decisions add beta*KL and emit the kl metric."""

    kmax = 32
    adapter = _make_adapter(tmp_path, None)
    adapter.search_distill_cfg = SearchDistillConfig.from_dict(
        {"enabled": True, "beta": 1.0, "value_mix": 0.0}
    )
    batch = _synthetic_batch(T=8, kmax=kmax, seed=3)
    targets = np.zeros((8, kmax), dtype=np.float32)
    searched = np.zeros(8, dtype=bool)
    rng = np.random.default_rng(7)
    for t in (0, 3, 5):
        packed = pack_feasible_candidates(
            batch.masks[t], batch.costs_to_lock[t], max_candidates=kmax
        )
        w = rng.random(kmax) * packed.mask
        targets[t] = (w / w.sum()).astype(np.float32)
        searched[t] = True
    batch.search_targets = targets
    batch.search_values = np.zeros(8, dtype=np.float32)
    batch.search_mask = searched
    torch.manual_seed(99)
    metrics = adapter._update_policy(batch)
    assert "search_distill/kl_target_net" in metrics
    assert metrics["search_distill/kl_target_net"] > 0.0


# ------------------------------------------------------- integration smokes
def _smoke_cfg(tmp_path, *, gamma: float) -> object:
    return _adapter_cfg(
        tmp_path,
        {
            "gamma": gamma,
            "decisions_per_update": 16,
            "minibatch_size": 8,
            "num_epochs": 1,
            "search_distill": {
                "enabled": True,
                "sims": 4,
                "beam": 4,
                "decision_fraction": 0.5,
                "beta": 1.0,
                "value_mix": 0.5,
            },
        },
    )


def _run_one_update(cfg, env):
    bus = _StubBus()
    adapter = SMDPPPOAdapter(cfg, env, _StubLogger(), bus, device="cpu")
    try:
        adapter.train_forever()
    finally:
        if adapter._sd_runner is not None:
            adapter._sd_runner.close()
    updates = [p for name, p in bus.events if name == "update_end"]
    assert updates, "no update_end emitted"
    return updates[-1]


@needs_pool
def test_search_distill_smoke_1p(tmp_path) -> None:
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_pool_vec import DrMarioPoolVecEnv

    env = DrMarioPoolVecEnv(
        num_envs=4,
        state_repr="bitplane_bottle_conn_mask",
        level=2,
        speed_setting=2,
        randomize_rng=True,
        emit_board=True,
    )
    try:
        metrics = _run_one_update(_smoke_cfg(tmp_path / "p1", gamma=0.99), env)
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)

    assert "search_distill/searched_fraction_actual" in metrics
    assert metrics["search_distill/searched_fraction_actual"] > 0.0
    assert "search_distill/kl_target_net" in metrics
    assert "search_distill/search_ms_p50" in metrics
    assert "search_distill/q_gap_mean" in metrics
    assert metrics["search_distill/q_gap_mean"] >= 0.0


@needs_vspool
def test_search_distill_smoke_vs(tmp_path) -> None:
    prev_repr = ram_specs.get_state_representation()
    from drmc_rl.training.envs.drmario_vs_vec import DrMarioVsPoolVecEnv

    env = DrMarioVsPoolVecEnv(
        num_pairs=2,
        state_repr="bitplane_bottle_conn_mask",
        level=5,
        speed_setting=2,
        randomize_rng=True,
    )
    try:
        metrics = _run_one_update(_smoke_cfg(tmp_path / "vs", gamma=1.0), env)
    finally:
        env.close()
        ram_specs.set_state_representation(prev_repr)

    assert metrics["search_distill/searched_fraction_actual"] > 0.0
    assert "search_distill/kl_target_net" in metrics
    assert "search_distill/search_ms_p50" in metrics
