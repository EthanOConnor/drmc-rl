from __future__ import annotations

"""Sample Factory launcher for Dr. Mario RL.

Registers the env and starts PPO training using a YAML config.

Usage:
    python training/run_sf.py --cfg training/sf_configs/state_baseline.yaml
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

from envs.retro.core_utils import resolve_libretro_core


def _extract_core_settings(config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], bool]:
    core_name: Optional[str] = None
    core_path: Optional[str] = None
    modified = False
    containers = [config]
    for key in ("env", "env_args", "retro", "libretro"):
        value = config.get(key)
        if isinstance(value, dict):
            containers.append(value)

    for container in containers:
        if core_name is None:
            value = container.pop("core", None)
            if isinstance(value, str) and value:
                core_name = value
                modified = True
        if core_path is None:
            value = container.pop("core_path", None)
            if isinstance(value, str) and value:
                core_path = value
                modified = True

    return core_name, core_path, modified


def _cleanup_temp_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _prepare_sf_config(cfg_path: Path) -> Tuple[Path, Optional[str], Optional[str], Optional[Path]]:
    with cfg_path.open("r", encoding="utf-8") as fp:
        try:
            raw_cfg: Any = yaml.safe_load(fp) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config: {exc}") from exc

    if not isinstance(raw_cfg, dict):
        return cfg_path, None, None, None

    config_data = dict(raw_cfg)
    core_name, core_path, modified = _extract_core_settings(config_data)

    if not modified:
        return cfg_path, core_name, core_path, None

    with tempfile.NamedTemporaryFile("w", suffix=cfg_path.suffix or ".yaml", delete=False) as tmp_file:
        yaml.safe_dump(config_data, tmp_file, sort_keys=False)
        temp_path = Path(tmp_file.name)

    return temp_path, core_name, core_path, temp_path


def main() -> None:
    try:
        from sample_factory.launcher.run import run as sf_run  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover
        print("Sample Factory is not installed. Install extras: pip install '.[rl]'\n", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--timeout', type=int, default=None, help='Override episode timeout in frames (optional)')
    ap.add_argument('--state-viz-interval', type=int, default=None, help='Emit state RGB frames every N steps (optional)')
    args = ap.parse_args()

    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.is_file():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg_for_sf, core_name, core_path_str, temp_cfg_path = _prepare_sf_config(cfg_path)
    except (OSError, ValueError) as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    cleanup_paths: list[Path] = [temp_cfg_path] if temp_cfg_path is not None else []
    resolved_core: Optional[Path] = None
    if core_name:
        try:
            resolved_core = resolve_libretro_core(core_name)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            _cleanup_temp_files(cleanup_paths)
            sys.exit(1)
    if core_path_str:
        candidate = Path(core_path_str).expanduser()
        if not candidate.is_absolute():
            candidate = (cfg_path.parent / core_path_str).resolve()
        if not candidate.is_file():
            print(f"Libretro core not found at {candidate}", file=sys.stderr)
            _cleanup_temp_files(cleanup_paths)
            sys.exit(1)
        resolved_core = candidate

    if resolved_core is not None:
        os.environ["DRMARIO_CORE_PATH"] = str(resolved_core)
        os.environ["LIBRETRO_CORE"] = str(resolved_core)

    try:
        from sample_factory.runner.run_description import RunDescription  # noqa: F401
        from sample_factory.runner.run_description import Experiment, ParamGrid  # noqa: F401
        from sample_factory.launcher.run import run  # noqa: F401
    except Exception:
        print("[INFO] Sample Factory not installed. Please install extra 'rl' and retry.")
        print("pip install sample-factory wandb")
        _cleanup_temp_files(cleanup_paths)
        sys.exit(1)

    # Register env id
    try:
        from envs.retro.register_env import register_env_id
        register_env_id()
    except Exception:
        pass

    if args.state_viz_interval is not None:
        os.environ["DRMARIO_STATE_VIZ_INTERVAL"] = str(max(1, int(args.state_viz_interval)))

    if args.timeout is not None:
        os.environ["DRMARIO_TIMEOUT"] = str(int(args.timeout))

    try:
        os.system(f"python -m sample_factory.launcher.run --run {cfg_for_sf}")
    finally:
        _cleanup_temp_files(cleanup_paths)


if __name__ == "__main__":
    main()
