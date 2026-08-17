from __future__ import annotations

import json
from pathlib import Path

import pytest

from drmc_rl.search.strong_league_memberwise import (
    read_davidson_calibration,
    read_mixture_members,
)


def test_member_manifest_preserves_ids_paths_and_weights(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "drmc-strong-league-continuation-mixture-v1",
                "members": [
                    {"id": "a", "checkpoint": "a.pt.gz", "sha256": "aa", "weight": 0.7},
                    {"id": "b", "checkpoint": "b.pt.gz", "sha256": "bb", "weight": 0.3},
                ],
            }
        )
    )
    members = read_mixture_members(manifest)
    assert [member.id for member in members] == ["a", "b"]
    assert members[0].checkpoint == tmp_path / "a.pt.gz"
    assert [member.weight for member in members] == pytest.approx([0.7, 0.3])


def test_grouped_calibration_schema_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema": "drmc-strong-league-wdl-calibration-v2",
                "parameters": {"slope": 1.2, "bias": -0.1, "draw_logit": -2.0},
            }
        )
    )
    calibration = read_davidson_calibration(path)
    assert calibration.slope == pytest.approx(1.2)
    assert calibration.bias == pytest.approx(-0.1)
    assert calibration.draw_logit == pytest.approx(-2.0)
    assert len(calibration.artifact_sha256) == 64
