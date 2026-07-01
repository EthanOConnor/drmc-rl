import json

from tools.export_competition_web import build_dashboard_data, write_data_js
from tools.tournament import GameResult, GameSpec, TournamentStore


def _insert(store, tid, a, b, winners):
    for k, w in enumerate(winners):
        store.insert_game(
            tid,
            a,
            b,
            GameResult(
                spec=GameSpec(game_idx=k, seed=100 + k, a_side=k % 2),
                winner=w,
                match_len_sec=30.0,
                decisions=50,
            ),
        )


def test_dashboard_export_components_and_static_method(tmp_path):
    db = tmp_path / "t.sqlite"
    store = TournamentStore(db)
    roster = {
        "entries": [
            {"name": "alpha", "checkpoint": "a.pt.gz", "mode": "plain"},
            {"name": "beta", "checkpoint": "b.pt.gz", "mode": "plain"},
            {"name": "gamma", "checkpoint": "g.pt.gz", "mode": "plain"},
            {"name": "delta", "checkpoint": "d.pt.gz", "mode": "plain"},
        ]
    }
    tid = store.get_or_create_tournament("split", roster, level=14, games_per_pair=4, seed=7)
    _insert(store, tid, "alpha", "beta", ["a", "a", "b", "a"])
    _insert(store, tid, "gamma", "delta", ["b", "b", "a", "b"])
    store.close()

    data = build_dashboard_data(db, tmp_path / "runs")
    assert data["meta"]["n_components"] == 2
    assert "no time-varying latent skill" in data["meta"]["method"]
    by_name = {a["name"]: a for a in data["agents"]}
    assert by_name["alpha"]["component"] == by_name["beta"]["component"]
    assert by_name["gamma"]["component"] == by_name["delta"]["component"]
    assert by_name["alpha"]["component"] != by_name["gamma"]["component"]
    assert by_name["alpha"]["rating"] > by_name["beta"]["rating"]
    assert by_name["delta"]["rating"] > by_name["gamma"]["rating"]


def test_write_data_js(tmp_path):
    out = tmp_path / "web" / "pool" / "data.js"
    write_data_js({"meta": {"n_games": 1}, "agents": []}, out)
    text = out.read_text()
    assert text.startswith("window.DRMC_POOL_DATA = ")
    payload = text.split(" = ", 1)[1].rstrip(";\n")
    assert json.loads(payload)["meta"]["n_games"] == 1
