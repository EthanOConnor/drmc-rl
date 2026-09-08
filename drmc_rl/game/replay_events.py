"""Lightweight readers for Fightcade replay event blobs (no model runtime)."""
from __future__ import annotations

import base64
import json

import numpy as np


def decode_field(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(16, 8)


def parse_quark_events(raw: bytes) -> dict[str, list]:
    """Split a JSONL event blob into init/spawn/lock/grb records in order."""
    events = {key: [] for key in ("init", "spawn", "lock", "grb")}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        for kind in ("spawn", "lock", "init"):
            if f'"t":"{kind}"' in line:
                events[kind].append(json.loads(line))
                break
        else:
            if '"grb"' in line:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "grb" in event and "f" in event:
                    events["grb"].append(event)
    return events
