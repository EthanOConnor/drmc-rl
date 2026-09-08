import json
from types import SimpleNamespace

import pytest

from drmc_rl.execution.pace import PACES
from tools.package_human_backend import PROTOCOL_SCHEMA, verify_package


@pytest.mark.parametrize("missing", [None, "model", "scheduled_execution", "cadence", "strength", "stale_pace"])
def test_frozen_package_rejects_missing_app_capabilities(monkeypatch, tmp_path, missing):
    caps = {
        "model": {"schema": "drmc-human-afterstate-v3"},
        "scheduled_execution": {"version": 1},
        "cadence": {"unrestricted_fallback": False, "profiles": [p.to_dict() for p in PACES]},
        "strength": {"controls": ["regret", "quality"], "competitive_ceiling": {"sha256": "fixture"}},
    }
    if missing == "stale_pace":
        caps["cadence"]["profiles"][0]["motion_interval"] = 12
    elif missing:
        del caps[missing]
    monkeypatch.setattr("tools.package_human_backend.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({"schema": PROTOCOL_SCHEMA, "capabilities": caps})))
    if missing:
        with pytest.raises(ValueError, match="incompatible"):
            verify_package(tmp_path)
    else:
        assert verify_package(tmp_path) == caps
