from datetime import datetime, timedelta, timezone
import json

import pytest

from drmc_rl.arena.experiment import experiment_health, read_experiment, relative_ratings


def test_failed_training_overrides_optimistic_plan(tmp_path):
    plan = tmp_path/"experiment.json"
    plan.write_text(json.dumps({"status":"Training · faster execution", "title":"Pace study"}))
    (tmp_path/"training.json").write_text(json.dumps({
        "status":"Failed", "frames":23130802, "error":"[Errno 5] Input/output error"}))
    (tmp_path/"pipeline.json").write_text(json.dumps({"status":"Failed", "error":"training failed"}))
    data = read_experiment(plan)
    assert data["health"] == {"status":"Training stopped", "severity":"failed",
                              "message":"[Errno 5] Input/output error"}
    assert data["training"]["frames"] == 23130802


@pytest.mark.parametrize("status,age,batch,expected", [
    ("Running",30,50,"ok"),
    ("Running",300,50,"stale"),
    ("Running",300,120,"ok"),
    ("Training complete",3600,50,"ok"),
])
def test_training_freshness_distinguishes_silence_from_completion(status, age, batch, expected):
    now = datetime(2026,9,9,tzinfo=timezone.utc)
    training = {"status":status,"updated_at":(now-timedelta(seconds=age)).isoformat(),
                "batch_seconds":batch}
    health = experiment_health(training, {}, now)
    assert health["severity"] == expected
    assert health["training_age_seconds"] == age


def test_pipeline_failure_is_visible_after_training_completed():
    health = experiment_health({"status":"Training complete"},
        {"status":"Failed","error":"evaluation worker failed"}, datetime.now(timezone.utc))
    assert health["status"] == "Study stopped"
    assert health["message"] == "evaluation worker failed"


def test_relative_ratings_anchor_connected_fields_and_complete_seed_pairs():
    comparisons = {
        "ab": {"id":"ab", "a":"faster", "b":"baseline8", "level":14},
        "disconnected": {"id":"disconnected", "a":"unrelated", "b":"other", "level":14},
        "pressure": {"id":"pressure", "a":"faster", "b":"baseline8", "level":20},
    }
    rows = {}
    for comparison in comparisons:
        for i in range(80):
            rows[(comparison,i)] = {"comparison":comparison, "index":i, "seed":i//2,
                "side":i%2, "score": float(i < (60 if comparison == "ab" else 20))}
    rows[("ab",80)] = {"comparison":"ab", "index":80, "seed":40, "side":0, "score":0}
    groups = relative_ratings(comparisons, rows)
    assert len(groups) == 2
    for group in groups:
        ratings = {r["id"]:r for r in group["ratings"]}
        assert set(ratings) == {"baseline8","faster"}
        assert ratings["baseline8"]["elo"] == ratings["baseline8"]["low"] == ratings["baseline8"]["high"] == 0
        assert ratings["faster"]["games"] == 80
        assert ratings["faster"]["low"] < ratings["faster"]["elo"] < ratings["faster"]["high"]
        assert (ratings["faster"]["elo"] > 0) == (group["level"] == 14)


def test_unified_standings_connect_phases_and_keep_joint_difference_intervals():
    comparisons = {
        "pilot":{"id":"pilot","a":"old","b":"parent","level":14,"pace":"normal","rating_group":"Pilot"},
        "live":{"id":"live","a":"new","b":"old","level":14,"pace":"normal","rating_group":"Milestones"},
    }
    records = {(id,i):{"comparison":id,"index":i,"seed":i//2,"side":i%2,"score":float(i<48)}
               for id in comparisons for i in range(64)}
    groups = relative_ratings(comparisons,records,anchor="parent",unified=True)
    assert len(groups) == 1
    ratings = {r["id"]:r for r in groups[0]["ratings"]}
    assert set(ratings) == {"parent","old","new"}
    assert ratings["old"]["games"] == 128
    gap = ratings["new"]["differences"]["old"]
    reverse = ratings["old"]["differences"]["new"]
    assert gap["elo"] > 0
    assert (gap["elo"],gap["low"],gap["high"]) == (-reverse["elo"],-reverse["high"],-reverse["low"])
    assert ratings["new"]["differences"]["new"] == {"elo":0,"low":0,"high":0}
