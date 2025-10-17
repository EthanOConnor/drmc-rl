from __future__ import annotations

T_MAX_DEFAULT = 8000

# Level-specific timeouts (frames). Use the closest floor key ≤ level.
LEVEL_TIMEOUTS = {
    0: 4000,
    5: 6000,
    10: 7000,
    15: 8000,
}


def get_level_timeout(level: int) -> int:
    keys = sorted(LEVEL_TIMEOUTS.keys())
    best = T_MAX_DEFAULT
    for k in keys:
        if level >= k:
            best = LEVEL_TIMEOUTS[k]
        else:
            break
    return best

