# envs/state_core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Import your existing mapper
# If your package path differs, adjust accordingly.
from envs.specs import ram_to_state as r2s

def _np_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    arr.setflags(write=False)
    return arr

def _parse_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return int(x, 0)
        except Exception:
            return None
    try:
        return int(x)
    except Exception:
        return None

def _read_offset_value(arr: np.ndarray, offsets: Dict[str, Any], group: str, addr_key: str = "addr") -> Optional[int]:
    spec = offsets.get(group)
    if not spec:
        return None
    addr = _parse_int(spec.get(addr_key))
    if addr is None:
        return None
    if 0 <= addr < arr.shape[0]:
        return int(arr[addr])
    return None

@dataclass(frozen=True)
class RamView:
    bytes: bytes
    arr: np.ndarray  # uint8 view of bytes (read-only)

    @staticmethod
    def from_bytes(b: bytes) -> RamView:
        return RamView(bytes=b, arr=_np_from_bytes(b))

@dataclass(frozen=True)
class RamVals:
    # Scalars read directly from RAM snapshot
    mode: Optional[int]
    gameplay_active: Optional[bool]
    stage_clear: Optional[bool]
    ending_active: Optional[bool]
    player_count: Optional[int]
    pill_counter: Optional[int]
    level: Optional[int]
    gravity_counter: Optional[int]
    lock_counter: Optional[int]
    speed_setting: Optional[int]
    speed_ups: Optional[int]
    hor_velocity: Optional[int]

@dataclass(frozen=True)
class Calc:
    # Derived from state tensor
    planes: np.ndarray           # (C, 16, 8) float32, read-only
    occupancy: np.ndarray        # (16, 8) bool
    static_mask: np.ndarray      # (16, 8) bool
    falling_mask: np.ndarray     # (16, 8) bool
    virus_mask: np.ndarray       # (16, 8) bool
    viruses_remaining: int
    preview: Dict[str, int]      # decode_preview_from_state()

@dataclass(frozen=True)
class EnvMeta:
    t: int
    elapsed_frames: int
    frame_skip: int
    last_terminal: Optional[str] = None

@dataclass(frozen=True)
class DrMarioState:
    ram: RamView
    ram_vals: RamVals
    calc: Calc
    env: EnvMeta
    stack4: np.ndarray            # (4, C, 16, 8), read-only for consumers

def _derive_from_planes(
    planes: np.ndarray, *, ram_bytes: bytes, ram_offsets: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Dict[str, int]]:
    # Keep derived masks independent of the configured observation channels.
    static = r2s.get_static_mask(planes).astype(bool, copy=False)
    virus = r2s.get_virus_mask(planes).astype(bool, copy=False)
    falling = r2s.decode_falling_mask_from_ram(ram_bytes, ram_offsets).astype(bool, copy=False)
    occ = np.asarray(static | virus | falling, dtype=bool)

    # Viruses remaining = count of virus cells / 1 per cell (Dr. Mario viruses are single tiles)
    v_rem = int(virus.sum())

    preview: Dict[str, int] = {}
    preview_raw = r2s.decode_preview_from_ram(ram_bytes, ram_offsets)
    if preview_raw is not None:
        try:
            first, second, rotation = preview_raw
            preview = {
                "first_color": int(first) & 0x03,
                "second_color": int(second) & 0x03,
                "rotation": int(rotation) & 0x03,
            }
        except Exception:
            preview = {}
    return occ, static, falling, virus, v_rem, preview

def _read_flags(arr: np.ndarray, offsets: Dict[str, Any]) -> Tuple[Optional[int], Optional[bool], Optional[bool], Optional[bool]]:
    mode_val = _read_offset_value(arr, offsets, "mode", "addr")
    gameplay_flag = None
    if offsets.get("mode") and mode_val is not None:
        playing_val = _parse_int(offsets["mode"].get("playing_value"))
        if playing_val is not None:
            gameplay_flag = (mode_val == playing_val)

    stage_flag = None
    if offsets.get("stage_clear"):
        val = _read_offset_value(arr, offsets, "stage_clear", "flag_addr")
        target = _parse_int(offsets["stage_clear"].get("cleared_value"))
        if val is not None and target is not None:
            stage_flag = (val == target)

    ending_flag = None
    if offsets.get("ending"):
        # `ram_offsets.json` uses `{"addr": ..., "non_ending_value": ...}`.
        val = _read_offset_value(arr, offsets, "ending", "addr")
        non_val = _parse_int(offsets["ending"].get("non_ending_value"))
        if val is not None and non_val is not None:
            ending_flag = (val != non_val)

    return mode_val, gameplay_flag, stage_flag, ending_flag

def build_state(
    *,
    ram_bytes: bytes,
    ram_offsets: Dict[str, Any],
    prev_stack4: Optional[np.ndarray],
    t: int,
    elapsed_frames: int,
    frame_skip: int,
    last_terminal: Optional[str] = None,
) -> DrMarioState:
    # Freeze RAM
    ram = RamView.from_bytes(ram_bytes)

    # Map to planes; freeze them as read-only
    planes = r2s.ram_to_state(ram_bytes, ram_offsets).astype(np.float32, copy=False)
    planes.setflags(write=False)

    # Derive masks & counts
    occupancy, static, falling, virus, v_rem, preview = _derive_from_planes(
        planes,
        ram_bytes=ram_bytes,
        ram_offsets=ram_offsets,
    )

    # Canonical scalar reads: use raw RAM bytes, not normalized planes.
    gravity = _read_offset_value(ram.arr, ram_offsets, "gravity_lock", "gravity_counter_addr")
    lock = _read_offset_value(ram.arr, ram_offsets, "gravity_lock", "lock_counter_addr")
    level = _read_offset_value(ram.arr, ram_offsets, "level", "addr")

    # Other RAM scalars
    mode_val, gameplay, stage_clear, ending = _read_flags(ram.arr, ram_offsets)
    players = _read_offset_value(ram.arr, ram_offsets, "players", "addr")
    pill_counter = _read_offset_value(ram.arr, ram_offsets, "pill_counter", "addr")

    speed_setting = _read_offset_value(ram.arr, ram_offsets, "gravity_lock", "speed_setting_addr")
    if speed_setting is None:
        speed_setting = int(ram.arr[0x008B]) if 0x008B < ram.arr.shape[0] else None

    speed_ups = _read_offset_value(ram.arr, ram_offsets, "gravity_lock", "speed_index_addr")
    if speed_ups is None:
        speed_ups = int(ram.arr[0x008A]) if 0x008A < ram.arr.shape[0] else None

    hor_velocity = int(ram.arr[0x0093]) if 0x0093 < ram.arr.shape[0] else None

    # Immutable RamVals bundle
    ram_vals = RamVals(
        mode=mode_val,
        gameplay_active=gameplay,
        stage_clear=stage_clear,
        ending_active=ending,
        player_count=players,
        pill_counter=pill_counter,
        level=level,
        gravity_counter=gravity,
        lock_counter=lock,
        speed_setting=speed_setting,
        speed_ups=speed_ups,
        hor_velocity=hor_velocity,
    )

    calc = Calc(
        planes=planes,
        occupancy=occupancy,
        static_mask=static,
        falling_mask=falling,
        virus_mask=virus,
        viruses_remaining=v_rem,
        preview=preview,
    )

    env = EnvMeta(t=t, elapsed_frames=elapsed_frames, frame_skip=frame_skip, last_terminal=last_terminal)

    # Maintain a 4-frame stack in the canonical state (so observation reads do no work)
    if prev_stack4 is None:
        stack4 = np.stack([planes] * 4, axis=0)
    else:
        stack4 = np.concatenate([prev_stack4[1:], planes[None, ...]], axis=0)
    stack4.setflags(write=False)

    return DrMarioState(ram=ram, ram_vals=ram_vals, calc=calc, env=env, stack4=stack4)
