from __future__ import annotations

from pathlib import Path

import pytest

from training.algo.simple_pg import SimplePGAdapter
from training.algo.sf2_adapter import SampleFactoryAdapter
from training.diagnostics.event_bus import EventBus
from training.diagnostics.logger import DiagLogger
from training.diagnostics.video import VideoEventHandler
from training.envs import make_vec_env
from training.utils.cfg import load_and_merge_cfg, to_config_node

try:  # pragma: no cover - optional dependency for smoke tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="simple_pg smoke test requires PyTorch")
def test_simple_pg_smoke(tmp_path) -> None:
    cfg_dict = load_and_merge_cfg("training/configs/base.yaml", "training/configs/simple_pg.yaml")
    cfg_dict["logdir"] = str(tmp_path / "run")
    cfg_dict["viz"] = []
    cfg_dict.setdefault("env", {})["num_envs"] = 2
    cfg_dict.setdefault("env", {})["episode_length"] = 8
    cfg_dict.setdefault("train", {})["total_steps"] = 32
    cfg_dict["train"]["checkpoint_interval"] = 16
    cfg_dict.setdefault("simple_pg", {})["batch_steps"] = 16
    cfg = to_config_node(cfg_dict)

    logger = DiagLogger(cfg)
    event_bus = EventBus()
    VideoEventHandler(event_bus, logger, Path(cfg.logdir) / "videos", interval=4)
    env = make_vec_env(cfg)

    updates = []
    event_bus.on("update_end", lambda payload: updates.append(payload))

    adapter = SimplePGAdapter(cfg, env, logger, event_bus, device="cpu")
    adapter.train_forever()

    logger.close()
    if hasattr(env, "close"):
        env.close()

    assert updates, "Adapter failed to emit update_end events"
    metrics = updates[-1]
    assert "loss/policy" in metrics
    assert "adv/mean" in metrics
    checkpoints = list((Path(cfg.logdir) / "checkpoints").glob("*.pt"))
    assert checkpoints, "Expected checkpoint artifact to be created"


@pytest.mark.skipif(torch is None, reason="PPO smoke test requires PyTorch")
def test_sf2_smoke_train(tmp_path) -> None:
    cfg_dict = load_and_merge_cfg("training/configs/base.yaml", "training/configs/ppo.yaml")
    cfg_dict["logdir"] = str(tmp_path / "run")
    cfg_dict["viz"] = []
    cfg_dict.setdefault("env", {})["num_envs"] = 2
    cfg_dict["env"]["episode_length"] = 8
    cfg_dict.setdefault("train", {})["total_steps"] = 64
    cfg_dict["train"]["checkpoint_interval"] = 32
    cfg_dict.setdefault("ppo", {})["rollout"] = 4
    cfg_dict["ppo"]["batch_size"] = 16
    cfg_dict["ppo"]["minibatch_size"] = 8
    cfg_dict["ppo"]["mini_epochs"] = 2

    cfg = to_config_node(cfg_dict)

    logger = DiagLogger(cfg)
    event_bus = EventBus()
    VideoEventHandler(event_bus, logger, Path(cfg.logdir) / "videos", interval=4)
    env = make_vec_env(cfg)

    updates = []
    event_bus.on("update_end", lambda payload: updates.append(payload))

    adapter = SampleFactoryAdapter(cfg, env, logger, event_bus, device="cpu")
    adapter.train_forever()

    logger.close()
    if hasattr(env, "close"):
        env.close()

    assert updates, "Adapter failed to emit update_end events"
    metrics = updates[-1]
    assert "loss/policy" in metrics
    checkpoints = list((Path(cfg.logdir) / "checkpoints").glob("*.pt"))
    assert checkpoints, "Expected checkpoint artifact to be created"
