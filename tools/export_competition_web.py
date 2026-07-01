"""Export internal VS tournament and pool results for the static web dashboard.

The dashboard intentionally uses a static Bradley-Terry rating fit: frozen
agents do not have a time-varying latent skill. Re-running the export can move
their estimated ratings because the observed opponent mix and game set changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from tools.tournament import DEFAULT_DB, REPO_ROOT, elo_mle, wilson_ci

DEFAULT_OUT = REPO_ROOT / "web" / "pool" / "data.js"


def _score_for_a(winner: str) -> float:
    if winner == "a":
        return 1.0
    if winner == "b":
        return 0.0
    return 0.5


def _rating_components(names: Sequence[str], games: Sequence[Tuple[int, int, float]]) -> List[int]:
    """Connected components of the rating graph."""

    adj: List[List[int]] = [[] for _ in names]
    for i, j, _s in games:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    comp = [-1] * len(names)
    cid = 0
    for start in range(len(names)):
        if comp[start] >= 0:
            continue
        q: deque[int] = deque([start])
        comp[start] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return comp


def _static_ratings(names: Sequence[str], games: Sequence[Tuple[int, int, float]]) -> Dict[str, Any]:
    """Static Bradley-Terry ratings, fit independently per connected component."""

    names = list(names)
    totals: Dict[str, List[int]] = {n: [0, 0, 0] for n in names}  # W, D, L
    for i, j, s in games:
        ni, nj = names[int(i)], names[int(j)]
        if s == 1.0:
            totals[ni][0] += 1
            totals[nj][2] += 1
        elif s == 0.0:
            totals[ni][2] += 1
            totals[nj][0] += 1
        else:
            totals[ni][1] += 1
            totals[nj][1] += 1

    components = _rating_components(names, games)
    ratings = np.zeros(len(names), dtype=np.float64)
    ses = np.full(len(names), np.inf, dtype=np.float64)
    for cid in sorted(set(components)):
        members = [i for i, c in enumerate(components) if c == cid]
        if len(members) <= 1:
            continue
        local = {old: new for new, old in enumerate(members)}
        sub_games = [
            (local[int(i)], local[int(j)], float(s))
            for i, j, s in games
            if int(i) in local and int(j) in local
        ]
        if not sub_games:
            continue
        r, se = elo_mle(len(members), sub_games)
        for old, new in local.items():
            ratings[old] = float(r[new])
            ses[old] = float(se[new])

    rows = []
    for i, name in enumerate(names):
        w, d, lo = totals[name]
        n_games = w + d + lo
        rows.append(
            {
                "name": name,
                "rating": float(ratings[i]),
                "ci95": None if not math.isfinite(float(ses[i])) else float(1.96 * ses[i]),
                "component": int(components[i]),
                "games": int(n_games),
                "wins": int(w),
                "draws": int(d),
                "losses": int(lo),
                "score_rate": (w + 0.5 * d) / n_games if n_games else None,
            }
        )
    rows.sort(key=lambda r: (r["component"], -r["rating"], -r["games"], r["name"]))
    return {"entries": rows, "components": int(max(components) + 1 if components else 0)}


def _open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _load_rows(db: Path) -> Tuple[List[sqlite3.Row], List[sqlite3.Row]]:
    con = _open_db(db)
    try:
        tournaments = con.execute("SELECT * FROM tournaments ORDER BY id").fetchall()
        games = con.execute(
            "SELECT g.*, t.name AS tournament_name, t.created AS tournament_created "
            "FROM games g JOIN tournaments t ON t.id = g.tournament_id "
            "ORDER BY t.id, g.entry_a, g.entry_b, g.game_idx"
        ).fetchall()
    finally:
        con.close()
    return tournaments, games


def _entry_meta(tournaments: Sequence[sqlite3.Row]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for t in tournaments:
        roster = json.loads(str(t["roster"]))
        for entry in roster.get("entries", []):
            name = str(entry.get("name", ""))
            if not name:
                continue
            cur = meta.setdefault(name, {"name": name, "seen_in": []})
            cur["mode"] = str(entry.get("mode", cur.get("mode", "plain")))
            if "checkpoint" in entry:
                cur["checkpoint"] = str(entry["checkpoint"])
            if entry.get("params"):
                cur["params"] = entry.get("params")
            cur["seen_in"].append(str(t["name"]))
    return meta


def _game_tuples(names: Sequence[str], rows: Iterable[sqlite3.Row]) -> List[Tuple[int, int, float]]:
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for r in rows:
        out.append((idx[str(r["entry_a"])], idx[str(r["entry_b"])], _score_for_a(str(r["winner"]))))
    return out


def _pairwise(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    rec: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
    seconds: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    decisions: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in rows:
        a, b = str(r["entry_a"]), str(r["entry_b"])
        key = (a, b) if a < b else (b, a)
        wld = rec[key]
        winner = str(r["winner"])
        if winner == "draw":
            wld[2] += 1
        else:
            winner_name = a if winner == "a" else b
            if winner_name == key[0]:
                wld[0] += 1
            else:
                wld[1] += 1
        if r["match_len_sec"] is not None:
            seconds[key].append(float(r["match_len_sec"]))
        if r["decisions"] is not None:
            decisions[key].append(int(r["decisions"]))

    out = []
    for (a, b), (wa, wb, d) in sorted(rec.items()):
        decisive = wa + wb
        lo, hi = wilson_ci(wa, decisive) if decisive else (0.0, 1.0)
        out.append(
            {
                "a": a,
                "b": b,
                "wins_a": int(wa),
                "wins_b": int(wb),
                "draws": int(d),
                "games": int(wa + wb + d),
                "win_rate_a": wa / decisive if decisive else None,
                "ci95": [float(lo), float(hi)],
                "avg_sec": float(np.mean(seconds[(a, b)])) if seconds[(a, b)] else None,
                "avg_decisions": (
                    float(np.mean(decisions[(a, b)])) if decisions[(a, b)] else None
                ),
            }
        )
    out.sort(key=lambda r: (-r["games"], r["a"], r["b"]))
    return out


def _summarize_tournament(t: sqlite3.Row, rows: Sequence[sqlite3.Row]) -> Dict[str, Any]:
    roster = json.loads(str(t["roster"]))
    names = [str(e["name"]) for e in roster.get("entries", [])]
    expected = max(0, len(names) * (len(names) - 1) // 2 * int(t["games_per_pair"]))
    games = _game_tuples(names, rows) if names else []
    ratings = _static_ratings(names, games)["entries"] if games else []
    top = [
        {k: r[k] for k in ("name", "rating", "ci95", "games", "wins", "draws", "losses")}
        for r in ratings[:5]
    ]
    return {
        "id": int(t["id"]),
        "name": str(t["name"]),
        "created": str(t["created"]),
        "level": int(t["level"]),
        "games_per_pair": int(t["games_per_pair"]),
        "seed": int(t["seed"]),
        "entries": names,
        "games_recorded": int(len(rows)),
        "games_expected": int(expected),
        "complete": bool(expected > 0 and len(rows) >= expected),
        "top": top,
    }


def _latest_jsonl(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _load_pools(runs_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for manifest_path in sorted(runs_dir.glob("*/opponent_pool/manifest.json")):
        run_dir = manifest_path.parents[1]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        latest_skill = _latest_jsonl(run_dir / "skill_history.jsonl")
        entries = []
        for e in manifest.get("entries", []):
            games = int(e.get("games", 0) or 0)
            wins = float(e.get("wins", 0.0) or 0.0)
            entries.append(
                {
                    "id": str(e.get("id", "")),
                    "file": str(e.get("file", "")),
                    "protected": bool(e.get("protected", False)),
                    "league_target": bool(e.get("league_target", False)),
                    "wins": wins,
                    "games": games,
                    "learner_score_rate": wins / games if games else None,
                }
            )
        out.append(
            {
                "run": run_dir.name,
                "path": str(run_dir.relative_to(REPO_ROOT)),
                "entries": entries,
                "latest_skill": latest_skill,
                "updated": datetime.fromtimestamp(
                    manifest_path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out


def build_dashboard_data(db: Path = DEFAULT_DB, runs_dir: Path | None = None) -> Dict[str, Any]:
    runs_dir = REPO_ROOT / "runs" if runs_dir is None else Path(runs_dir)
    tournaments, game_rows = _load_rows(Path(db))
    by_tid: Dict[int, List[sqlite3.Row]] = defaultdict(list)
    for r in game_rows:
        by_tid[int(r["tournament_id"])].append(r)

    meta = _entry_meta(tournaments)
    names = sorted({str(r["entry_a"]) for r in game_rows} | {str(r["entry_b"]) for r in game_rows})
    games = _game_tuples(names, game_rows)
    ratings = _static_ratings(names, games)
    for row in ratings["entries"]:
        row.update({k: v for k, v in meta.get(row["name"], {}).items() if k != "name"})
        seen = row.get("seen_in", [])
        row["first_seen"] = seen[0] if seen else None
        row["last_seen"] = seen[-1] if seen else None

    tournaments_out = [
        _summarize_tournament(t, by_tid[int(t["id"])]) for t in tournaments
    ]
    latest_tournament = tournaments_out[-1] if tournaments_out else None
    return {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db": str(Path(db).relative_to(REPO_ROOT)) if Path(db).is_relative_to(REPO_ROOT) else str(db),
            "n_tournaments": len(tournaments_out),
            "n_games": len(game_rows),
            "n_agents": len(names),
            "n_components": ratings["components"],
            "latest_tournament": latest_tournament["name"] if latest_tournament else None,
            "method": "static Bradley-Terry logistic MLE; no time-varying latent skill",
        },
        "agents": ratings["entries"],
        "pairwise": _pairwise(game_rows),
        "tournaments": tournaments_out,
        "pools": _load_pools(runs_dir),
    }


def write_data_js(data: Dict[str, Any], out: Path = DEFAULT_OUT) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    out.write_text(
        "window.DRMC_POOL_DATA = " + payload + ";\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    data = build_dashboard_data(Path(args.db), Path(args.runs_dir))
    write_data_js(data, Path(args.out))
    print(
        f"wrote {args.out}  tournaments={data['meta']['n_tournaments']} "
        f"games={data['meta']['n_games']} agents={data['meta']['n_agents']}"
    )


if __name__ == "__main__":
    main()
