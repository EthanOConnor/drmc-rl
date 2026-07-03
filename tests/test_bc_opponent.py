from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from envs.backends.drmario_pool import is_library_present

pytestmark = pytest.mark.skipif(
    not is_library_present(),
    reason="native pool library missing (build with: make -C game_engine libdrmario_pool)",
)

FIXTURE = Path(__file__).parent / "fixtures" / "fc_v2_events_small.jsonl"


def _fixture_rows():
    from tools.train_bc_opponent import extract_moves_from_events
    from tools.annotate_replay_events import make_batch_planner

    raw = FIXTURE.read_bytes()
    planner = make_batch_planner("cpu")
    return extract_moves_from_events(raw, planner)


def test_extraction_deterministic_on_fixture():
    from tools.train_bc_opponent import rows_to_arrays

    rows_a = _fixture_rows()
    rows_b = _fixture_rows()
    assert len(rows_a) > 10
    arrays_a = rows_to_arrays(rows_a)
    arrays_b = rows_to_arrays(rows_b)
    for key in arrays_a:
        assert np.array_equal(arrays_a[key], arrays_b[key]), key


def test_extraction_fields_sane():
    rows = _fixture_rows()
    for r in rows:
        assert 0 <= r["chosen_action"] < 512
        assert 0 <= r["chosen_slot"] < r["cand_count"]
        # Chosen slot points at the chosen action inside the packed list.
        assert r["cand_actions"][r["chosen_slot"]] == r["chosen_action"]
        # Same-color pills must not choose o>=2 actions (symmetry reduction).
        if r["pill"][0] == r["pill"][1]:
            assert r["chosen_action"] < 256
            assert (r["cand_actions"][: r["cand_count"]] < 256).all()


def test_bc_training_loss_decreases():
    import torch.nn.functional as F

    from tools.train_bc_opponent import batch_inputs, build_bc_net, rows_to_arrays

    arrays = rows_to_arrays(_fixture_rows())
    n = arrays["field"].shape[0]
    torch.manual_seed(0)
    net = build_bc_net()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    idx = np.arange(n)

    losses = []
    for _step in range(30):
        obs, pills, prevs, ca, cc, cm, slot = batch_inputs(arrays, idx)
        logits, _values = net(
            torch.from_numpy(obs),
            torch.from_numpy(pills),
            torch.from_numpy(prevs),
            torch.from_numpy(ca),
            torch.from_numpy(cc),
            torch.from_numpy(cm),
            aux=None,
        )
        logits = logits.masked_fill(~torch.from_numpy(cm), -1e9)
        loss = F.cross_entropy(logits, torch.from_numpy(slot))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]


def test_bc_checkpoint_loads_through_opponent_pool(tmp_path):
    from tools.train_bc_opponent import BC_NET_CFG, build_bc_net
    from training.envs.vs_opponents import OpponentPool
    from training.utils.checkpoint_io import save_checkpoint

    net = build_bc_net()
    ckpt = tmp_path / "bc_test_band.pt.gz"
    save_checkpoint(
        {"state_dict": net.state_dict(), "cfg": BC_NET_CFG, "step": 0},
        ckpt,
    )

    pool = OpponentPool(tmp_path / "pool")
    pool.seed([ckpt])
    entry = pool.entries[-1]
    pool.ensure_loaded(entry)
    assert entry.net is not None
    assert entry.aux_dim == 0
    assert entry.candidate_max == 128

    # Forward smoke through the loaded entry on a fixture state.
    from tools.train_bc_opponent import batch_inputs, rows_to_arrays

    arrays = rows_to_arrays(_fixture_rows())
    obs, pills, prevs, ca, cc, cm, _slot = batch_inputs(arrays, np.arange(4))
    dev = entry.device  # pool device is "auto" (cuda when available)
    with torch.inference_mode():
        logits, values = entry.net(
            torch.from_numpy(obs).to(dev),
            torch.from_numpy(pills).to(dev),
            torch.from_numpy(prevs).to(dev),
            torch.from_numpy(ca).to(dev),
            torch.from_numpy(cc).to(dev),
            torch.from_numpy(cm).to(dev),
            aux=None,
        )
    assert logits.shape == (4, 128)
    assert values.shape[0] == 4
