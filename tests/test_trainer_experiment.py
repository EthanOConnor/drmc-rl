from drmc_rl.arena.experiment import relative_ratings


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
