from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from drmc_rl.training.algo.ppo_smdp import SMDPPPOAdapter
from drmc_rl.training.utils.cfg import to_config_node
from drmc_rl.training.utils.checkpoint_io import load_checkpoint


@pytest.mark.parametrize("compress", [False, True])
def test_training_checkpoint_restores_weights_and_public_contract(tmp_path, compress):
    adapter = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    adapter.global_step = 100
    adapter.decision_step = 10
    adapter._next_checkpoint = 100
    adapter.checkpoint_interval = 100
    adapter.checkpoint_keep_last = 1
    adapter.checkpoint_compress = compress
    adapter.checkpoint_dir = tmp_path
    adapter.net = torch.nn.Linear(2, 1)
    adapter._ema_state = adapter.net.state_dict()
    adapter.optimizer = torch.optim.Adam(adapter.net.parameters())
    adapter.cfg = to_config_node({"smdp_ppo": {"aux_spec": "zero_v1_vs"}})
    events = []
    adapter.event_bus = SimpleNamespace(emit=lambda event, **fields: events.append((event, fields)))
    adapter._maybe_checkpoint()

    path = tmp_path / ("smdp_ppo_step100.pt.gz" if compress else "smdp_ppo_step100.pt")
    restored = load_checkpoint(path, map_location="cpu")
    assert restored["step"] == 100
    assert restored["decision_step"] == 10
    assert restored["cfg"]["smdp_ppo"]["aux_spec"] == "zero_v1_vs"
    assert all(torch.equal(value, restored["state_dict"][key]) for key, value in adapter.net.state_dict().items())
    assert events[0][1]["path"] == str(path)
    adapter.global_step = 200
    adapter._maybe_checkpoint()
    assert not path.exists()
    assert len(list(tmp_path.iterdir())) == 1


def test_causal_migration_requires_fresh_run_and_uses_effective_inference_weights(tmp_path):
    adapter = SMDPPPOAdapter.__new__(SMDPPPOAdapter)
    adapter.device = "cpu"
    adapter.env = SimpleNamespace(public_observations=True)
    adapter.net = torch.nn.Linear(2, 1)
    state = adapter.net.state_dict()
    ema = {key: value + 0.5 for key, value in state.items()}
    path = tmp_path / "legacy.pt"
    torch.save(
        dict(cfg={"env": {"public_observations": False}}, state_dict=state, ema_state_dict=ema),
        path,
    )
    for resume_optimizer, resume_step in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="fresh weights-only"):
            adapter._load_checkpoint(
                path,
                resume_optimizer=resume_optimizer,
                resume_step=resume_step,
                override_optimizer_lr=False,
                strict=True,
            )
    adapter._load_checkpoint(
        path, resume_optimizer=False, resume_step=False, override_optimizer_lr=False, strict=True
    )
    for name, value in adapter.net.state_dict().items():
        torch.testing.assert_close(value, ema[name])
