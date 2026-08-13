import hashlib
import json
import struct

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from drmc_rl.data.human_corpus import HumanCorpus, decode_input_rle


def _corpus(tmp_path):
    release = tmp_path / "releases" / "r1"
    shard = release / "decisions" / "year=2026" / "month=08" / "part-00000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.table({"decision_id": ["a", "b"], "skill_elo": [1500.0, 1800.0]}), shard)
    ratings = release / "ratings" / "trajectories.parquet"
    ratings.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "player": ["Alice", "Alice"], "day": [100, 102],
        "skill_elo": [1500.0, 1520.0], "skill_sd": [50.0, 40.0],
    }), ratings)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    ratings_digest = hashlib.sha256(ratings.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "fcr-human-v2",
        "release_id": "r1",
        "source": {"max_day": 400},
        "stats": {"decisions": 2},
        "files": [{"path": str(shard.relative_to(release)), "kind": "decisions",
                   "rows": 2, "bytes": shard.stat().st_size, "sha256": digest},
                  {"path": str(ratings.relative_to(release)), "kind": "ratings",
                   "rows": 2, "bytes": ratings.stat().st_size, "sha256": ratings_digest}],
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "latest").symlink_to("releases/r1")
    return tmp_path


def test_manifest_verify_and_scan(tmp_path):
    corpus = HumanCorpus(_corpus(tmp_path))
    assert corpus.release_id == "r1"
    assert corpus.verify(hashes=True)["files"] == 2
    rows = sum(batch.num_rows for batch in corpus.batches(months=["2026-08"]))
    assert rows == 2
    assert corpus.rating_at("Alice", 101) == (1510.0, 45.0)
    assert corpus.rating_at("unknown", 101) == (None, None)
    assert corpus.time_split(101) == "train"
    assert corpus.time_split(180) == "validation"
    assert corpus.time_split(250) == "test"


def test_rejects_manifest_path_traversal(tmp_path):
    release = tmp_path / "latest"
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({
        "schema_version": "fcr-human-v1", "release_id": "bad",
        "files": [{"path": "../secret", "kind": "decisions", "rows": 1,
                   "bytes": 1, "sha256": "x"}],
    }))
    with pytest.raises(ValueError, match="unsafe"):
        HumanCorpus(tmp_path)


def test_decode_input_rle():
    encoded = struct.pack("<HBHB", 2, 0, 3, 4)
    assert decode_input_rle(encoded, 5) == bytes([0, 0, 4, 4, 4])
    with pytest.raises(ValueError, match="expected 4"):
        decode_input_rle(encoded, 4)
