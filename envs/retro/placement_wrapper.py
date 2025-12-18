"""Compatibility shim for the placement macro-environment.

Historically this project exposed a large, experimental placement wrapper in
`envs.retro.placement_wrapper`. The macro-action system has since been
re-implemented as a cleaner 512-way SMDP wrapper in `envs.retro.placement_env`.

This module remains as a stable import target for older scripts/tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from envs.retro.placement_env import DrMarioPlacementEnv, make_placement_env


def render_planner_debug_view(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Legacy hook used by `training.speedrun_experiment`.

    The new placement environment does not expose the old per-frame planner
    traces via this function. Callers should rely on info dict keys emitted by
    `DrMarioPlacementEnv` instead.
    """

    return None


__all__ = ["DrMarioPlacementEnv", "make_placement_env", "render_planner_debug_view"]

