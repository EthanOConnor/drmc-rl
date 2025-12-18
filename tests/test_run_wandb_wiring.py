from __future__ import annotations

from training.run import load_config, parse_args


def test_wandb_flag_enables_wandb_viz_and_project() -> None:
    args = parse_args(
        [
            "--cfg",
            "training/configs/base.yaml",
            "--wandb",
            "--wandb-project",
            "unit-test-project",
            "--ui",
            "none",
        ]
    )
    cfg = load_config(args)
    viz = list(getattr(cfg, "viz", []))
    assert "wandb" in viz
    assert getattr(cfg, "wandb_project") == "unit-test-project"

