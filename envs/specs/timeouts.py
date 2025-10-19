from __future__ import annotations
import os

_CMD_TIMEOUT = os.environ.get("DRMARIO_TIMEOUT")

T_MAX_DEFAULT = 8000

# Level-specific timeouts (frames). Use the closest floor key ≤ level.
LEVEL_TIMEOUTS = {
    0: 4000,
    5: 6000,
    10: 7000,
    15: 8000,
}

if _CMD_TIMEOUT is None:
    # Double the defaults if no override is provided
    T_MAX_DEFAULT *= 2
    for k in LEVEL_TIMEOUTS:
        LEVEL_TIMEOUTS[k] *= 2


def get_level_timeout(level: int) -> int:
    if _CMD_TIMEOUT is not None:
        try:
            return int(_CMD_TIMEOUT)
        except (TypeError, ValueError):
            pass  # Fall back to defaults

    keys = sorted(LEVEL_TIMEOUTS.keys())
    best = T_MAX_DEFAULT
    for k in keys:
        if level >= k:
            best = LEVEL_TIMEOUTS[k]
        else:
            break
    return best

