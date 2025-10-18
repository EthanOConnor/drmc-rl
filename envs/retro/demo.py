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

# Optional OpenCV display support
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

# Ensure repo root is on sys.path when running as a script
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.retro.register_env import register_env_id
from gymnasium import make


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
    show_window = args.show_window
    if show_window and cv2 is None:
        print("OpenCV (cv2) not installed; install opencv-python or omit --show-window.")
        show_window = False
    window_name: Optional[str] = None
    if show_window:
        window_name = "Dr. Mario"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # type: ignore[arg-type]
    for _ in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if args.dump_state and args.mode == 'state':
            core = obs['obs'] if isinstance(obs, dict) else obs
            # print non-zero counts per channel on the most recent frame in stack
            latest = core[-1]
            nz = [int(latest[c].sum()) for c in range(latest.shape[0])]
            print('nz per channel:', nz)
        if save_dir or show_window:
            frame = env.render()
            if frame is not None:
                if show_window and window_name:
                    scale = max(0.5, float(args.display_scale))
                    scaled = cv2.resize(frame, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  # type: ignore[arg-type]
                    cv2.imshow(window_name, cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))  # type: ignore[arg-type]
                    key = cv2.waitKey(1)  # type: ignore[arg-type]
                    if key & 0xFF == ord('q'):
                        break
                if save_dir:
                    fname = save_dir / f"frame_{steps:05d}"
                    if Image is not None:
                        Image.fromarray(frame).save(fname.with_suffix(".png"))
                    else:
                        np.save(fname.with_suffix(".npy"), frame)
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
    if show_window and window_name:
        cv2.destroyAllWindows()  # type: ignore[attr-defined]
    env.close()


if __name__ == '__main__':
    main()
