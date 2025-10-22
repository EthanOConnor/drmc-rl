from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from training.algo.base import AlgoAdapter
from training.diagnostics.event_bus import EventBus
from training.diagnostics.logger import DiagLogger
from training.diagnostics.video import VideoEventHandler
from training.envs import make_vec_env
from training.utils.cfg import apply_dot_overrides, load_and_merge_cfg, to_config_node
from training.utils.reproducibility import pick_device, set_reproducibility, write_environment_file


DEFAULT_BASE_CFG = Path("training/configs/base.yaml")


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Dr. Mario training entrypoint")
    parser.add_argument("--algo", choices=["simple_pg", "ppo", "appo"], default="simple_pg")
    parser.add_argument("--engine", choices=["builtin", "sf2"], default=None)
    parser.add_argument("--cfg", type=str, default=str(DEFAULT_BASE_CFG))
    parser.add_argument("--override", type=str, default=None, help="Comma separated key=value overrides")
    parser.add_argument("--viz", nargs="*", default=None, help="Override diagnostics backends")
    parser.add_argument("--video_interval", type=int, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--obs_mode", type=str, default=None)
    parser.add_argument("--total_steps", type=float, default=None)
    parser.add_argument("--dry_run", type=str, default="false")
    parser.add_argument("--device", type=str, default=None)
    # Compatibility knobs retained from historical scripts
    parser.add_argument("--timeout", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--state-viz-interval", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> Any:
    cfg_dict = load_and_merge_cfg(DEFAULT_BASE_CFG, args.cfg if args.cfg != str(DEFAULT_BASE_CFG) else None)
    cfg_dict["algo"] = args.algo
    if args.engine:
        cfg_dict["engine"] = args.engine
    else:
        cfg_dict["engine"] = "builtin" if args.algo == "simple_pg" else "sf2"
    if args.viz is not None:
        cfg_dict["viz"] = args.viz if isinstance(args.viz, list) else [args.viz]
    if args.video_interval is not None:
        cfg_dict["video_interval"] = int(args.video_interval)
    if args.logdir is not None:
        cfg_dict["logdir"] = args.logdir
    if args.seed is not None:
        cfg_dict["seed"] = int(args.seed)
    if args.num_envs is not None:
        cfg_dict.setdefault("env", {})["num_envs"] = int(args.num_envs)
    if args.obs_mode is not None:
        cfg_dict.setdefault("env", {})["obs_mode"] = args.obs_mode
    if args.total_steps is not None:
        cfg_dict.setdefault("train", {})["total_steps"] = int(args.total_steps)
    if args.device is not None:
        cfg_dict["device"] = args.device

    apply_dot_overrides(cfg_dict, args.override)

    # Normalise viz configuration
    viz = cfg_dict.get("viz", [])
    if isinstance(viz, str):
        cfg_dict["viz"] = [viz]

    cfg = to_config_node(cfg_dict)
    cfg.dry_run = str(args.dry_run).lower() in {"1", "true", "yes"}
    return cfg


def build_adapter(cfg: Any, env: Any, logger: DiagLogger, event_bus: EventBus, device: str) -> AlgoAdapter:
    if cfg.algo == "simple_pg":
        from training.algo.simple_pg import SimplePGAdapter as Adapter
    elif cfg.algo in {"ppo", "appo"}:
        from training.algo.sf2_adapter import SampleFactoryAdapter as Adapter
    else:  # pragma: no cover - guardrail
        raise ValueError(f"Unknown algorithm: {cfg.algo}")
    return Adapter(cfg, env, logger, event_bus, device=device)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)
    ctx = set_reproducibility(int(getattr(cfg, "seed", 42)))
    device = pick_device(getattr(cfg, "device", "auto"))

    logger = DiagLogger(cfg)
    event_bus = EventBus()
    VideoEventHandler(event_bus, logger, Path(cfg.logdir) / "videos", interval=int(getattr(cfg, "video_interval", 5000)))
    env = make_vec_env(cfg)

    metadata: Dict[str, Any] = {
        "algo": cfg.algo,
        "engine": cfg.engine,
        "seed": ctx.seed,
        "device": device,
    }
    logger.write_metadata(metadata)
    write_environment_file(Path(cfg.logdir) / "env.txt")

    if cfg.dry_run:
        logger.log_text("run/dry_run", "Dry run completed", step=0)
        logger.flush()
        logger.close()
        if hasattr(env, "close"):
            env.close()
        return

    adapter = build_adapter(cfg, env, logger, event_bus, device)
    try:
        adapter.train_forever()
    finally:
        if hasattr(adapter, "close"):
            adapter.close()
        logger.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
