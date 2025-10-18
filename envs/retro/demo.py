"""Smoke test for DrMarioRetroEnv with configurable emulator backend.

Usage:
    python envs/retro/demo.py --mode pixel --level 0 --steps 1000 --risk-tau 0.5 --backend libretro
"""
import argparse
import time
import sys
from pathlib import Path

from typing import Any, Dict, Optional

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None
else:
    try:
        from PIL import ImageTk
    except Exception:  # pragma: no cover - optional dependency
        ImageTk = None

# Ensure repo root is on sys.path when running as a script
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.retro.register_env import register_env_id
from envs.retro.state_viz import state_to_rgb
from gymnasium import make
_state_to_rgb = state_to_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pixel', 'state'], default='pixel')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--risk-tau', type=float, default=0.5)
    ap.add_argument('--dump-state', action='store_true', help='Print state-plane stats (state mode)')
    ap.add_argument('--frame-offset', type=int, default=0, help='Advance N frames with NOOP at reset (influences ROM RNG)')
    ap.add_argument('--start-presses', type=int, default=None, help='Override auto start presses at reset (default: 2 on first boot, 1 afterwards)')
    ap.add_argument('--start-hold-frames', type=int, default=None, help='Frames to hold START during auto-start (default 6)')
    ap.add_argument('--start-gap-frames', type=int, default=None, help='Frames between START taps (default 40)')
    ap.add_argument('--start-level-taps', type=int, default=None, help='Number of LEFT taps on level select before starting (default 12)')
    ap.add_argument('--start-settle-frames', type=int, default=None, help='Frames to run NOOP after auto-start before gameplay (default 180)')
    ap.add_argument('--start-wait-frames', type=int, default=None, help='Additional NOOP frames while waiting for virus count to become positive (default 600)')
    ap.add_argument('--backend', type=str, default=None, help='libretro (default), stable-retro, or mock')
    ap.add_argument('--core-path', type=str, default=None, help='Override DRMARIO_CORE_PATH for libretro backend')
    ap.add_argument('--rom-path', type=str, default=None, help='Override DRMARIO_ROM_PATH for libretro backend')
    ap.add_argument('--no-auto-start', action='store_true', help='Disable automatic START presses at reset')
    ap.add_argument('--save-frames', type=str, default=None, help='Directory to dump rendered frames (PNG if Pillow installed, else .npy)')
    ap.add_argument('--show-window', action='store_true', help='Open an OpenCV window showing live frames (press q to exit early)')
    ap.add_argument('--display-scale', type=float, default=2.0, help='Scale factor for the preview window (requires --show-window)')
    args = ap.parse_args()

    register_env_id()
    env_kwargs = {"obs_mode": args.mode, "level": args.level, "risk_tau": args.risk_tau, "render_mode": "rgb_array"}
    if args.backend:
        env_kwargs["backend"] = args.backend
    if args.core_path:
        env_kwargs["core_path"] = Path(args.core_path).expanduser()
    if args.rom_path:
        env_kwargs["rom_path"] = Path(args.rom_path).expanduser()
    if args.no_auto_start:
        env_kwargs["auto_start"] = False
    env = make("DrMarioRetroEnv-v0", **env_kwargs)
    reset_options: Dict[str, Any] = {'frame_offset': args.frame_offset}
    if args.start_presses is not None:
        reset_options['start_presses'] = int(args.start_presses)
    if args.start_hold_frames is not None:
        reset_options['start_hold_frames'] = int(args.start_hold_frames)
    if args.start_gap_frames is not None:
        reset_options['start_gap_frames'] = int(args.start_gap_frames)
    if args.start_level_taps is not None:
        reset_options['start_level_taps'] = int(args.start_level_taps)
    if args.start_settle_frames is not None:
        reset_options['start_settle_frames'] = int(args.start_settle_frames)
    if args.start_wait_frames is not None:
        reset_options['start_wait_viruses'] = int(args.start_wait_frames)
    obs, info = env.reset(options=reset_options)
    t0 = time.time()
    steps, reward_sum = 0, 0.0
    save_dir = None
    if args.save_frames:
        save_dir = Path(args.save_frames).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        if Image is None:
            print("Pillow not installed; frames will be saved as .npy arrays.")

    show_window = bool(args.show_window)
    viewer = None
    if show_window:
        if Image is None or ImageTk is None:
            print("Pillow (with ImageTk) is required for the live window; skipping preview.")
            show_window = False
        else:
            try:
                import tkinter as tk
            except Exception as exc:
                print(f"Tkinter preview unavailable ({exc}); skipping live window.")
                show_window = False
            else:
                class _TkViewer:
                    def __init__(self, title: str, scale: float, with_stats: bool) -> None:
                        self._scale = max(0.5, float(scale))
                        self._tk = tk.Tk()
                        self._tk.title(title)
                        self._closed = False
                        self._img_label = tk.Label(self._tk)
                        self._img_label.pack()
                        self._text_var: Optional[tk.StringVar] = None
                        if with_stats:
                            self._text_var = tk.StringVar(value="")
                            self._text_label = tk.Label(
                                self._tk, textvariable=self._text_var, anchor="w", justify="left"
                            )
                            self._text_label.pack(fill="x", padx=4, pady=2)
                        self._img = None
                        self._tk.protocol("WM_DELETE_WINDOW", self._on_close)

                    def _on_close(self) -> None:
                        self._closed = True
                        try:
                            self._tk.destroy()
                        except Exception:
                            pass

                    def update(self, frame: np.ndarray, stats: Optional[Dict[str, Any]] = None) -> bool:
                        if self._closed:
                            return False
                        try:
                            image = Image.fromarray(frame)
                            if self._scale != 1.0:
                                w, h = image.size
                                image = image.resize((int(w * self._scale), int(h * self._scale)), Image.NEAREST)
                            self._img = ImageTk.PhotoImage(image)
                            self._img_label.configure(image=self._img)
                            if self._text_var is not None and stats is not None:
                                info = stats.get("info", {}) if isinstance(stats, dict) else {}
                                lines = [
                                    f"Step {stats.get('step', 0)}  Total {stats.get('cumulative', 0.0):.1f}",
                                    f"Last reward {stats.get('reward', 0.0):.2f} (env {info.get('r_env', 0.0):.2f})",
                                    f"Viruses {info.get('viruses_remaining', '?')}  Level {info.get('level', '?')}",
                                    f"Topout {info.get('topout', False)}  Cleared {info.get('cleared', False)}",
                                ]
                                term = info.get('terminal_reason')
                                if term:
                                    lines.append(f"Terminal: {term}")
                                action = stats.get('action')
                                if action is not None:
                                    lines.append(f"Action: {action}")
                                self._text_var.set("\n".join(lines))
                            self._tk.update_idletasks()
                            self._tk.update()
                            return True
                        except Exception:
                            self._on_close()
                            return False

                    def close(self) -> None:
                        self._on_close()

                viewer = _TkViewer("Dr. Mario", args.display_scale, with_stats=(args.mode == "state"))

    def current_frame(current_obs: Any, current_info: Dict[str, Any]) -> Optional[np.ndarray]:
        if args.mode == "pixel":
            return env.render()
        core = current_obs["obs"] if isinstance(current_obs, dict) else current_obs
        return _state_to_rgb(np.asarray(core), current_info)

    frame_index = 0
    cumulative_reward = 0.0

    if show_window or save_dir:
        frame = current_frame(obs, info)
        if frame is not None:
            stats = {"info": info, "step": steps, "reward": 0.0, "cumulative": cumulative_reward, "action": None}
            if viewer is not None and not viewer.update(frame, stats):
                show_window = False
            if save_dir:
                fname = save_dir / f"frame_{frame_index:05d}"
                if Image is not None:
                    Image.fromarray(frame).save(fname.with_suffix(".png"))
                else:
                    np.save(fname.with_suffix(".npy"), frame)
        frame_index += 1

    for _ in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if args.dump_state and args.mode == 'state':
            core = obs['obs'] if isinstance(obs, dict) else obs
            # print non-zero counts per channel on the most recent frame in stack
            latest = core[-1]
            nz = [int(latest[c].sum()) for c in range(latest.shape[0])]
            print('nz per channel:', nz)
        cumulative_reward += r
        if save_dir or show_window:
            frame = current_frame(obs, info)
            if frame is not None:
                stats = {
                    "info": info,
                    "step": steps + 1,
                    "reward": r,
                    "cumulative": cumulative_reward,
                    "action": int(a),
                }
                if viewer is not None and not viewer.update(frame, stats):
                    break
                if save_dir:
                    fname = save_dir / f"frame_{frame_index:05d}"
                    if Image is not None:
                        Image.fromarray(frame).save(fname.with_suffix(".png"))
                    else:
                        np.save(fname.with_suffix(".npy"), frame)
                frame_index += 1
        reward_sum += r
        steps += 1
        if term or trunc:
            break
    dt = time.time() - t0
    fps = steps / max(dt, 1e-6)
    backend_name = getattr(env.unwrapped, "backend_name", "unknown")
    print(
        f"Ran {steps} steps, reward={reward_sum:.1f}, cleared={info.get('cleared', False)}, "
        f"FPS≈{fps:.1f}, backend={backend_name}, active={info.get('backend_active', False)}"
    )
    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == '__main__':
    main()
