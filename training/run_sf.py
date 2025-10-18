from __future__ import annotations

"""Sample Factory launcher for Dr. Mario RL.

Registers the env and starts PPO training using a YAML config.

Usage:
    python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml
"""

import argparse
import os
import sys


def main() -> None:
    try:
        from sample_factory.launcher.run import run as sf_run  # type: ignore
    except Exception as e:  # pragma: no cover
        print("Sample Factory is not installed. Install extras: pip install '.[rl]'\n", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--state-viz-interval', type=int, default=None, help='Emit state RGB frames every N steps (optional)')
    args = ap.parse_args()

    try:
        from sample_factory.runner.run_description import RunDescription  # noqa: F401
        from sample_factory.runner.run_description import Experiment, ParamGrid  # noqa: F401
        from sample_factory.launcher.run import run  # noqa: F401
    except Exception:
        print("[INFO] Sample Factory not installed. Please install extra 'rl' and retry.")
        print("pip install sample-factory wandb")
        sys.exit(1)

    # Register env id
    try:
        from envs.retro.register_env import register_env_id
        register_env_id()
    except Exception:
        pass

    if args.state_viz_interval is not None:
        os.environ["DRMARIO_STATE_VIZ_INTERVAL"] = str(max(1, int(args.state_viz_interval)))

    # Defer to SF CLI by shelling out for simplicity:
    os.system(f"python -m sample_factory.launcher.run --run {args.cfg}")


if __name__ == "__main__":
    main()
