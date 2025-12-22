#!/usr/bin/env python3
"""Ghost parity harness: libretro (ground truth) vs C++ engine (candidate).

This tool runs the emulator and the C++ engine side-by-side, feeds them the
*exact same* controller inputs frame-by-frame, and checks for divergence in a
curated set of NES RAM addresses (including the full bottle buffer).

The intent is to make parity debugging practical:
  - deterministic resets via explicit RNG seed bytes (optional)
  - fast first-divergence detection + structured JSONL logging

Example:
  python tools/ghost_parity.py --episodes 3 --steps 5000 \\
      --core quicknes --rom-path \"legal_ROMs/Dr. Mario (Japan, USA) (rev0).nes\" \\
      --engine game_engine/drmario_engine --randomize-rng
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as a script from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.retro.register_env import register_env_id
from envs.backends.base import NES_BUTTONS
from envs.retro.action_adapters import discrete10_to_buttons


def _hex(addr: int) -> str:
    return f"0x{int(addr) & 0xFFFF:04X}"


def _button_vec_to_mask(vec: Sequence[int]) -> int:
    # Same encoding as tools/game_transcript + C++ engine:
    # 0x01 R, 0x02 L, 0x04 D, 0x08 U, 0x10 START, 0x20 SELECT, 0x40 B, 0x80 A
    mapping = {
        "RIGHT": 0x01,
        "LEFT": 0x02,
        "DOWN": 0x04,
        "UP": 0x08,
        "START": 0x10,
        "SELECT": 0x20,
        "B": 0x40,
        "A": 0x80,
    }
    mask = 0
    for name, pressed in zip(NES_BUTTONS, vec):
        if int(pressed):
            mask |= int(mapping[str(name)])
    return int(mask) & 0xFF


def _iter_compare_addrs() -> Iterable[int]:
    # Core RNG + flags.
    # Note: $0053 is `metaspriteIndex` in the disassembly (visual bookkeeping)
    # and is not modeled by the headless C++ engine, so we intentionally do not
    # compare it here.
    for a in (0x0017, 0x0018, 0x0043, 0x0046, 0x0051, 0x0055):
        yield a
    # ZP mirrors used by the planner/wrappers.
    for a in (0x0081, 0x0082, 0x0085, 0x0086, 0x008A, 0x008B, 0x0092, 0x0093, 0x0097, 0x00A5, 0x00F7):
        yield a
    # Main RAM state used by env termination + RAM→state mapper.
    for a in (
        0x0301,
        0x0302,
        0x0305,
        0x0306,
        0x0307,
        0x0309,
        0x030A,
        0x030B,
        0x0310,
        0x0311,
        0x0312,
        0x0313,
        0x0316,
        0x0317,
        0x031A,
        0x031B,
        0x0322,
        0x0323,
        0x0324,
        0x0325,
        0x0326,
        0x0327,
        0x0727,
    ):
        yield a
    # Bottle buffer.
    for a in range(0x0400, 0x0480):
        yield a


COMPARE_ADDRS: Tuple[int, ...] = tuple(_iter_compare_addrs())


@dataclass(frozen=True)
class Divergence:
    episode: int
    step: int
    action: int
    buttons_mask: int
    addr: int
    expected: int
    actual: int


def _diff_first(ram_expected: bytes, ram_actual: bytes) -> Optional[Tuple[int, int, int]]:
    """Return (addr, expected, actual) for the first mismatch in COMPARE_ADDRS."""

    for addr in COMPARE_ADDRS:
        e = ram_expected[addr]
        a = ram_actual[addr]
        if e != a:
            return int(addr), int(e), int(a)
    return None


def _make_env(
    *,
    backend: str,
    rom_path: Optional[Path],
    core: Optional[str],
    engine_path: Optional[Path],
    level: int,
    speed_setting: int,
) -> "gymnasium.Env":
    import gymnasium as gym

    kwargs: Dict[str, object] = {
        "obs_mode": "state",
        "backend": backend,
        "level": int(level),
        "speed_setting": int(speed_setting),
        # We want parity on raw game logic, not shaping.
        "use_potential_shaping": False,
        "auto_start": True,
        "rng_randomize": False,
    }
    if backend == "libretro":
        from envs.retro.core_utils import resolve_libretro_core

        if rom_path is None:
            raise ValueError("--rom-path required for libretro")
        kwargs["rom_path"] = str(rom_path)
        if core is not None:
            kwargs["core_path"] = str(resolve_libretro_core(core))
    if backend == "cpp-engine":
        if engine_path is not None:
            kwargs["backend_kwargs"] = {"engine_path": str(engine_path)}
    return gym.make("DrMarioRetroEnv-v0", **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--speed-setting", type=int, default=2, help="0=low, 1=med, 2=high")
    ap.add_argument("--core", type=str, default="quicknes")
    ap.add_argument("--rom-path", type=Path, default=None)
    ap.add_argument("--engine", type=Path, default=Path("game_engine/drmario_engine"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize-rng", action="store_true")
    ap.add_argument(
        "--start-settle-frames",
        type=int,
        default=0,
        help="Override DrMarioRetroEnv auto-start settle frames (default 0 for parity checkpoint).",
    )
    ap.add_argument(
        "--start-wait-viruses",
        type=int,
        default=0,
        help="Override DrMarioRetroEnv auto-start wait-for-viruses frames (default 0 for parity checkpoint).",
    )
    ap.add_argument(
        "--start-sync-wait-frames",
        type=int,
        default=-1,
        help=(
            "Override DrMarioRetroEnv auto-start sync waitFrames ($0051). "
            "Use -1 for 'first non-zero waitFrames' (robust across levels). "
            "Use 0..255 to sync to an exact value."
        ),
    )
    ap.add_argument(
        "--start-sync-max-frames",
        type=int,
        default=2000,
        help="Max frames to search for the start-sync waitFrames condition.",
    )
    ap.add_argument("--out", type=Path, default=Path("data/ghost_parity_divergences.jsonl.gz"))
    ap.add_argument("--stop-on-first", action="store_true", default=True)
    args = ap.parse_args()

    register_env_id()

    rng = np.random.default_rng(int(args.seed))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_emul = _make_env(
        backend="libretro",
        rom_path=args.rom_path,
        core=args.core,
        engine_path=None,
        level=int(args.level),
        speed_setting=int(args.speed_setting),
    )
    env_eng = _make_env(
        backend="cpp-engine",
        rom_path=None,
        core=None,
        engine_path=args.engine,
        level=int(args.level),
        speed_setting=int(args.speed_setting),
    )

    divergences: List[Divergence] = []
    t0 = time.perf_counter()
    try:
        for ep in range(int(args.episodes)):
            rng_seed_bytes = None
            if args.randomize_rng:
                rng_seed_bytes = [int(v) & 0xFF for v in rng.integers(0, 256, size=2, dtype=np.uint8).tolist()]

            reset_opts = {
                "start_presses": 3,
                "start_settle_frames": int(args.start_settle_frames),
                "start_wait_viruses": int(args.start_wait_viruses),
                "start_sync_wait_frames": int(args.start_sync_wait_frames),
                "start_sync_max_frames": int(args.start_sync_max_frames),
            }
            reset_opts_eng = {
                "start_presses": 0,
                "start_settle_frames": int(args.start_settle_frames),
                "start_wait_viruses": int(args.start_wait_viruses),
                "start_sync_wait_frames": int(args.start_sync_wait_frames),
                "start_sync_max_frames": int(args.start_sync_max_frames),
            }
            if rng_seed_bytes is not None:
                reset_opts["randomize_rng"] = True
                reset_opts["rng_seed_bytes"] = rng_seed_bytes
                reset_opts_eng["randomize_rng"] = True
                reset_opts_eng["rng_seed_bytes"] = rng_seed_bytes

            # Reset emulator first so we can mirror its `waitFrames` ($0051)
            # value into the engine reset. The Python auto-start boundary can
            # vary by a frame in libretro, and we want both systems to start
            # from the exact same in-level point.
            _obs_e, info_e = env_emul.reset(options=reset_opts)
            ram_e0 = info_e.get("raw_ram")
            if not isinstance(ram_e0, (bytes, bytearray)):
                raise RuntimeError("Emulator env must expose info['raw_ram'] for parity checking.")
            reset_opts_eng["intro_wait_frames"] = int(ram_e0[0x0051]) & 0xFF
            reset_opts_eng["intro_frame_counter_lo"] = int(ram_e0[0x0043]) & 0xFF

            _obs_c, info_c = env_eng.reset(options=reset_opts_eng)

            ram_e = info_e.get("raw_ram")
            ram_c = info_c.get("raw_ram")
            if not isinstance(ram_e, (bytes, bytearray)) or not isinstance(ram_c, (bytes, bytearray)):
                raise RuntimeError("Both envs must expose info['raw_ram'] for parity checking.")
            mismatch = _diff_first(bytes(ram_e), bytes(ram_c))
            if mismatch is not None:
                addr, expected, actual = mismatch
                divergences.append(
                    Divergence(
                        episode=ep,
                        step=0,
                        action=-1,
                        buttons_mask=0,
                        addr=addr,
                        expected=expected,
                        actual=actual,
                    )
                )
                if args.stop_on_first:
                    break

            held_e = {"LEFT": False, "RIGHT": False, "DOWN": False}
            held_c = {"LEFT": False, "RIGHT": False, "DOWN": False}

            for step_idx in range(1, int(args.steps) + 1):
                action = int(rng.integers(0, 10))
                vec = discrete10_to_buttons(action, held_e)
                # Keep the same hold logic for both envs (matches env internals).
                held_c.update(held_e)
                buttons_mask = _button_vec_to_mask(vec)

                _obs_e, _r_e, term_e, trunc_e, info_e = env_emul.step(action)
                _obs_c, _r_c, term_c, trunc_c, info_c = env_eng.step(action)

                ram_e = info_e.get("raw_ram")
                ram_c = info_c.get("raw_ram")
                if not isinstance(ram_e, (bytes, bytearray)) or not isinstance(ram_c, (bytes, bytearray)):
                    raise RuntimeError("Both envs must expose info['raw_ram'] for parity checking.")
                mismatch = _diff_first(bytes(ram_e), bytes(ram_c))
                if mismatch is not None:
                    addr, expected, actual = mismatch
                    divergences.append(
                        Divergence(
                            episode=ep,
                            step=step_idx,
                            action=action,
                            buttons_mask=buttons_mask,
                            addr=addr,
                            expected=expected,
                            actual=actual,
                        )
                    )
                    if args.stop_on_first:
                        break

                if (term_e or trunc_e) != (term_c or trunc_c):
                    # Termination mismatch: record a synthetic divergence at a sentinel address.
                    divergences.append(
                        Divergence(
                            episode=ep,
                            step=step_idx,
                            action=action,
                            buttons_mask=buttons_mask,
                            addr=-1,
                            expected=int(bool(term_e or trunc_e)),
                            actual=int(bool(term_c or trunc_c)),
                        )
                    )
                    if args.stop_on_first:
                        break

                if term_e or trunc_e or term_c or trunc_c:
                    break

            if args.stop_on_first and divergences:
                break

    finally:
        try:
            env_emul.close()
        except Exception:
            pass
        try:
            env_eng.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    # Write JSONL divergences (one per line).
    if out_path.suffix == ".gz":
        fp_ctx = gzip.open(out_path, "wt", encoding="utf-8", compresslevel=9)
    else:
        fp_ctx = out_path.open("w", encoding="utf-8")
    with fp_ctx as f:
        for div in divergences:
            payload = asdict(div)
            payload["addr_hex"] = "TERM" if div.addr < 0 else _hex(div.addr)
            f.write(json.dumps(payload) + "\n")

    frames = int(args.episodes) * int(args.steps)
    fps = frames / max(1e-9, elapsed)
    print(f"ghost_parity: episodes={args.episodes} steps={args.steps} fps≈{fps:,.1f}")
    print(f"divergences={len(divergences)} out={out_path}")
    if divergences:
        d0 = divergences[0]
        print(
            f"first mismatch: ep={d0.episode} step={d0.step} action={d0.action} "
            f"addr={_hex(d0.addr)} expected={d0.expected} actual={d0.actual}"
        )


if __name__ == "__main__":
    main()
