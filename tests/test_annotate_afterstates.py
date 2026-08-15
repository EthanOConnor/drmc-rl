from __future__ import annotations

import json


def test_per_shard_reports_merge_without_losing_legacy_rows(tmp_path) -> None:
    from tools.annotate_afterstates import _collect_reports, _write_json_atomic

    legacy = {
        "schema": "old",
        "source": "old-source",
        "shards": [{"source": "2022-01.npz", "complete": True, "rows": 10}],
    }
    _write_json_atomic(tmp_path / "manifest.json", legacy)
    _write_json_atomic(
        tmp_path / "2022-02.report.json",
        {"source": "2022-02.npz", "complete": True, "rows": 20},
    )
    _write_json_atomic(
        tmp_path / "2022-01.report.json",
        {"source": "2022-01.npz", "complete": True, "rows": 11},
    )
    merged = _collect_reports(tmp_path, tmp_path / "source")
    assert [row["source"] for row in merged["shards"]] == ["2022-01.npz", "2022-02.npz"]
    assert merged["shards"][0]["rows"] == 11
    assert merged["source"] == str(tmp_path / "source")
    assert not list(tmp_path.glob(".*.tmp"))
    json.dumps(merged)
