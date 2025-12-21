from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import uuid

from envs.retro.core_utils import resolve_libretro_core
from training.algo.base import AlgoAdapter
from training.diagnostics.event_bus import EventBus
from training.diagnostics.logger import DiagLogger
from training.diagnostics.video import VideoEventHandler
from training.envs import make_vec_env
from training.utils.cfg import apply_dot_overrides, load_and_merge_cfg, to_config_node
from training.utils.reproducibility import pick_device, set_reproducibility, write_environment_file


DEFAULT_BASE_CFG = Path("training/configs/base.yaml")


def _generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _cleanup_engine_pid_dir(pid_dir: Path) -> None:
    """Best-effort cleanup for orphaned `drmario_engine` subprocesses.

    The cpp-engine backend writes a pidfile per engine instance under a run-local
    directory (see `DRMARIO_ENGINE_PID_DIR`). Under crashes or forced worker
    termination, those processes can be orphaned; we attempt to reap them here.
    """

    try:
        pid_dir = Path(pid_dir)
    except Exception:
        return
    if not pid_dir.is_dir():
        return

    for pid_file in sorted(pid_dir.glob("*.pid")):
        try:
            text = pid_file.read_text(encoding="utf-8")
        except Exception:
            continue

        pid: int | None = None
        for line in text.splitlines():
            if line.startswith("pid="):
                try:
                    pid = int(line.split("=", 1)[1].strip())
                except Exception:
                    pid = None
                break
        if pid is None or pid <= 1:
            continue

        # Skip already-dead pids.
        try:
            os.kill(int(pid), 0)
        except Exception:
            try:
                pid_file.unlink()
            except Exception:
                pass
            continue

        # Guardrail: only kill processes that look like our engine binary.
        try:
            comm = (
                subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="], stderr=subprocess.DEVNULL)
                .decode("utf-8", errors="replace")
                .strip()
            )
        except Exception:
            comm = ""
        if "drmario_engine" not in comm:
            continue

        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            continue

        # Give it a moment to exit cleanly.
        for _ in range(20):
            try:
                os.kill(int(pid), 0)
            except Exception:
                break
            try:
                threading.Event().wait(0.05)
            except Exception:
                break
        else:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass

        try:
            pid_file.unlink()
        except Exception:
            pass

    try:
        if not any(pid_dir.iterdir()):
            pid_dir.rmdir()
    except Exception:
        pass


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Dr. Mario RL training entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m training.run --algo simple_pg --ui tui
  python -m training.run --algo ppo_smdp --ui debug --env-id DrMarioPlacementEnv-v0 --core quicknes --rom-path legal_ROMs/DrMario.nes
  python -m training.run --algo ppo --cfg training/configs/ppo.yaml
  python -m training.run --algo simple_pg --wandb --total_steps 1e6
""",
    )
    parser.add_argument(
        "--algo",
        choices=["simple_pg", "ppo", "appo", "ppo_smdp"],
        default=None,
        help="Algorithm to run (defaults to cfg.algo or 'simple_pg').",
    )
    parser.add_argument(
        "--engine",
        choices=["builtin", "sf2"],
        default=None,
        help="Training engine ('builtin' or 'sf2'); defaults to cfg.engine or algo-based default.",
    )
    parser.add_argument("--cfg", type=str, default=str(DEFAULT_BASE_CFG))
    parser.add_argument("--override", type=str, default=None, help="Comma separated key=value overrides")

    # UI options
    parser.add_argument(
        "--ui",
        choices=["tui", "debug", "headless", "none"],
        default="headless",
        help="UI mode: tui (metrics), debug (board + playback controls), headless (logging only), none",
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="drmc-rl", help="WandB project name")
    parser.add_argument("--viz", nargs="*", default=None, help="Override diagnostics backends")
    parser.add_argument("--video_interval", type=int, default=None)
    parser.add_argument("--logdir", type=str, default=None)

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--obs_mode", type=str, default=None)
    parser.add_argument("--env-id", type=str, default=None, help="Gym env id (e.g. DrMarioRetroEnv-v0)")
    parser.add_argument(
        "--action_space",
        type=str,
        choices=["controller", "intent", "placement"],
        default=None,
        help="Select action space translation layer for environment wrappers.",
    )
    parser.add_argument("--total_steps", type=float, default=None)
    parser.add_argument("--dry_run", type=str, default="false")
    parser.add_argument("--device", type=str, default=None)

    # Retro backend convenience flags (also available via --override env.*)
    parser.add_argument(
        "--backend", type=str, default=None, help="Backend: libretro|stable-retro|cpp-engine|mock"
    )
    parser.add_argument("--core", type=str, default=None, help="Libretro core name (e.g. quicknes) or path")
    parser.add_argument("--core-path", type=str, default=None, help="Path to libretro core file")
    parser.add_argument("--rom-path", type=str, default=None, help="Path to Dr. Mario ROM")
    parser.add_argument("--level", type=int, default=None, help="Starting level (0..20)")
    parser.add_argument("--vectorization", type=str, default=None, help="Vector env mode: auto|sync|async")
    parser.add_argument(
        "--randomize-rng",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Randomize the ROM RNG state on each env reset (episode).",
    )

    # Compatibility knobs retained from historical scripts
    parser.add_argument("--timeout", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--state-viz-interval", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> Any:
    cfg_dict = load_and_merge_cfg(DEFAULT_BASE_CFG, args.cfg if args.cfg != str(DEFAULT_BASE_CFG) else None)
    algo = args.algo or cfg_dict.get("algo") or "simple_pg"
    # Backwards-compatible alias.
    if str(algo).lower() == "smdp_ppo":
        algo = "ppo_smdp"
    cfg_dict["algo"] = algo

    if args.engine:
        cfg_dict["engine"] = args.engine
    else:
        cfg_engine = cfg_dict.get("engine")
        if cfg_engine:
            cfg_dict["engine"] = cfg_engine
        else:
            cfg_dict["engine"] = "builtin" if algo in {"simple_pg", "ppo_smdp"} else "sf2"
    if args.viz is not None:
        cfg_dict["viz"] = args.viz if isinstance(args.viz, list) else [args.viz]
    if args.wandb:
        viz = cfg_dict.get("viz", [])
        if isinstance(viz, str):
            viz = [viz]
        if not isinstance(viz, list):
            viz = []
        if "wandb" not in viz:
            viz.append("wandb")
        cfg_dict["viz"] = viz
        cfg_dict["wandb_project"] = str(args.wandb_project)
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
    if args.env_id is not None:
        cfg_dict.setdefault("env", {})["id"] = str(args.env_id)
    if args.action_space is not None:
        cfg_dict.setdefault("env", {})["action_space"] = args.action_space
    if args.backend is not None:
        cfg_dict.setdefault("env", {})["backend"] = str(args.backend)
    if args.level is not None:
        cfg_dict.setdefault("env", {})["level"] = int(args.level)
    if args.vectorization is not None:
        cfg_dict.setdefault("env", {})["vectorization"] = str(args.vectorization)
    if args.randomize_rng is not None:
        cfg_dict.setdefault("env", {})["randomize_rng"] = bool(args.randomize_rng)
    if args.rom_path is not None:
        cfg_dict.setdefault("env", {})["rom_path"] = str(args.rom_path)
    if args.core is not None:
        cfg_dict.setdefault("env", {})["core"] = str(args.core)
    if args.core_path is not None:
        cfg_dict.setdefault("env", {})["core_path"] = str(args.core_path)
    if args.total_steps is not None:
        cfg_dict.setdefault("train", {})["total_steps"] = int(args.total_steps)
    if args.device is not None:
        cfg_dict["device"] = args.device

    apply_dot_overrides(cfg_dict, args.override)

    env_cfg = cfg_dict.setdefault("env", {})
    core_name = env_cfg.get("core")
    core_path_value = env_cfg.get("core_path")
    resolved_core_path: Path | None = None
    if isinstance(core_name, str) and core_name:
        resolved_core_path = resolve_libretro_core(core_name)
        env_cfg["core_path"] = str(resolved_core_path)
    elif isinstance(core_path_value, str) and core_path_value:
        candidate = Path(core_path_value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path(args.cfg).expanduser().parent / candidate).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Libretro core not found at {candidate}")
        resolved_core_path = candidate
        env_cfg["core_path"] = str(resolved_core_path)

    if resolved_core_path is not None:
        os.environ["DRMARIO_CORE_PATH"] = str(resolved_core_path)
        os.environ["LIBRETRO_CORE"] = str(resolved_core_path)

    rom_path_value = env_cfg.get("rom_path")
    if isinstance(rom_path_value, str) and rom_path_value:
        os.environ["DRMARIO_ROM_PATH"] = str(Path(rom_path_value).expanduser())

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
    elif cfg.algo == "ppo_smdp":
        from training.algo.ppo_smdp import SMDPPPOAdapter as Adapter
    else:  # pragma: no cover - guardrail
        raise ValueError(f"Unknown algorithm: {cfg.algo}")
    return Adapter(cfg, env, logger, event_bus, device=device)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)
    # UI-driven defaults for vectorization + debug info payloads.
    try:
        env_cfg = getattr(cfg, "env", None)
        env_data = env_cfg.data if hasattr(env_cfg, "data") else {}
        if args.vectorization is None:
            vec_mode = env_data.get("vectorization", "auto")
            if vec_mode is None or str(vec_mode).lower() == "auto":
                env_data["vectorization"] = "sync" if args.ui == "debug" else "async"
        if "emit_raw_ram" not in env_data:
            env_data["emit_raw_ram"] = bool(args.ui == "debug")
    except Exception:
        pass

    # Ensure each run writes to its own subdirectory unless the user explicitly
    # overrides `--logdir`.
    if args.logdir is None:
        base_logdir = Path(str(getattr(cfg, "logdir", "runs/auto"))).expanduser()
        run_id = str(os.environ.get("DRMARIO_RUN_ID") or _generate_run_id())
        cfg.run_id = run_id
        cfg.logdir = str(base_logdir / run_id)
    else:
        # Best-effort: derive run id from explicit logdir for metadata/UI.
        try:
            cfg.run_id = str(getattr(cfg, "run_id", Path(str(getattr(cfg, "logdir"))).name))
        except Exception:
            pass
    engine_pid_dir: Path | None = None
    try:
        engine_pid_dir = Path(str(cfg.logdir)) / "engine_pids"
        os.environ["DRMARIO_ENGINE_PID_DIR"] = str(engine_pid_dir)
    except Exception:
        engine_pid_dir = None
    ctx = set_reproducibility(int(getattr(cfg, "seed", 42)))
    device = pick_device(getattr(cfg, "device", "auto"))

    logger = DiagLogger(cfg)
    event_bus = EventBus()
    VideoEventHandler(event_bus, logger, Path(cfg.logdir) / "videos", interval=int(getattr(cfg, "video_interval", 5000)))
    metadata: Dict[str, Any] = {
        "run_id": getattr(cfg, "run_id", None),
        "logdir": str(getattr(cfg, "logdir", "")),
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
        return

    adapter: AlgoAdapter | None = None
    try:
        if args.ui == "debug":
            from training.envs.interactive import PlaybackControl, RateLimitedVecEnv, StopTraining
            from training.ui.runner_debug_tui import RunnerDebugTUI

            class _DebugSession:
                def __init__(self) -> None:
                    self.control = PlaybackControl()
                    try:
                        self.control.set_rng_randomize(
                            bool(getattr(getattr(cfg, "env", object()), "randomize_rng", False))
                        )
                    except Exception:
                        pass
                    self.env: RateLimitedVecEnv | None = None
                    self.adapter: AlgoAdapter | None = None
                    self.training_thread: threading.Thread | None = None
                    self.exc: list[BaseException] = []

                def start(self) -> None:
                    self.exc = []
                    base_env = make_vec_env(cfg)
                    self.env = RateLimitedVecEnv(base_env, self.control)
                    self.adapter = build_adapter(cfg, self.env, logger, event_bus, device)

                    def _train() -> None:
                        try:
                            if self.adapter is not None:
                                self.adapter.train_forever()
                        except StopTraining:
                            pass
                        except BaseException as e:  # noqa: BLE001
                            self.exc.append(e)

                    self.training_thread = threading.Thread(target=_train, name="training", daemon=True)
                    self.training_thread.start()

                def stop(self) -> None:
                    if self.control is not None:
                        self.control.request_stop()
                    if self.training_thread is not None:
                        self.training_thread.join(timeout=5.0)
                        if self.training_thread.is_alive():
                            # Best-effort cleanup: force-close the env even if
                            # the training thread is stuck in a pending async step.
                            try:
                                if self.env is not None and hasattr(self.env, "close"):
                                    self.env.close()
                            except Exception:
                                pass
                            raise RuntimeError("Training thread did not exit after stop request.")
                    if self.adapter is not None and hasattr(self.adapter, "close"):
                        self.adapter.close()
                    if self.env is not None and hasattr(self.env, "close"):
                        self.env.close()
                    if self.exc:
                        raise self.exc[0]

                def restart(self, num_envs: int) -> None:
                    self.stop()
                    try:
                        if hasattr(self.control, "reset_for_restart"):
                            self.control.reset_for_restart()
                    except Exception:
                        pass
                    try:
                        cfg.env.num_envs = int(num_envs)
                        # Debug UI should default to sync vectorization + full debug info.
                        cfg.env.vectorization = "sync"
                        cfg.env.emit_raw_ram = True
                    except Exception:
                        pass
                    self.start()

            session = _DebugSession()
            session.start()
            if session.env is None or session.training_thread is None:
                raise RuntimeError("Debug session failed to start.")

            ui = RunnerDebugTUI(
                env=session.env,
                control=session.control,
                event_bus=event_bus,
                title=f"DrMC-RL: {cfg.algo}",
            )
            try:
                ui.run(session)
            finally:
                # Ensure training terminates if UI exits early or errors out.
                session.stop()
        elif args.ui == "tui":
            try:
                from training.ui.tui import TrainingTUI, RICH_AVAILABLE
                from training.ui.event_handler import TUIEventHandler

                if RICH_AVAILABLE:
                    total_steps = int(getattr(cfg.train, "total_steps", 2000000))
                    tui = TrainingTUI(experiment_name=f"DrMC-RL: {cfg.algo}")
                    tui.set_hyperparams(
                        {
                            "algo": cfg.algo,
                            "seed": ctx.seed,
                            "device": device,
                            "run_id": getattr(cfg, "run_id", ""),
                            "total_steps": total_steps,
                        }
                    )
                    TUIEventHandler(event_bus, tui)
                    env = make_vec_env(cfg)
                    adapter = build_adapter(cfg, env, logger, event_bus, device)
                    with tui:
                        try:
                            adapter.train_forever()
                        except KeyboardInterrupt:
                            tui.set_status("Stopping (Ctrl+C)...")
                else:
                    env = make_vec_env(cfg)
                    adapter = build_adapter(cfg, env, logger, event_bus, device)
                    try:
                        adapter.train_forever()
                    except KeyboardInterrupt:
                        pass
            except ImportError as e:
                print(f"Warning: TUI import failed ({e}), falling back to headless mode")
                env = make_vec_env(cfg)
                adapter = build_adapter(cfg, env, logger, event_bus, device)
                try:
                    adapter.train_forever()
                except KeyboardInterrupt:
                    pass
        else:
            env = make_vec_env(cfg)
            adapter = build_adapter(cfg, env, logger, event_bus, device)
            try:
                adapter.train_forever()
            except KeyboardInterrupt:
                pass
    finally:
        if adapter is not None and hasattr(adapter, "close"):
            adapter.close()
        logger.close()
        if "env" in locals() and env is not None and hasattr(env, "close"):
            env.close()
        if engine_pid_dir is not None:
            _cleanup_engine_pid_dir(engine_pid_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
