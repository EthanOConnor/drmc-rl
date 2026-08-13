"""One-screen holistic status: processes, training runs, eval trajectories.

Usage:
  .venv/bin/python -m tools.overview

Sections: running project processes; per-run trainer status + latest WHR
grade; characterization trajectories from the tournament store (the
*_eval_step<N> tournaments the evalwatch records — treatment vs champion,
control vs champion, and the mask_opponent reliance probe). Deeper views:
`tools.vs_dashboard <run>` (live TUI), `tools.run_status <run> --history`,
`tools.tournament report --tournament <name>`.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs"
DB = RUNS / "tournaments" / "tournaments.sqlite"
FRESH_S = 12 * 3600


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def show_processes() -> None:
    section("processes")
    out = subprocess.run(["ps", "axo", "pid,etime,command"], capture_output=True, text=True).stdout
    pat = re.compile(r"training\.run|tools\.tournament +run|supervisor\.sh|evalwatch\.sh|vs_head_to_head|live_agent_server")
    seen = set()
    for line in out.splitlines():
        if not pat.search(line) or "grep" in line:
            continue
        pid, etime, cmd = line.strip().split(None, 2)
        cmd = re.sub(r"\S*/Python(\.app\S*)? -m ", "python -m ", cmd)
        key = cmd[:90]
        if key in seen:  # supervisors fork a janitor subshell; show once
            continue
        seen.add(key)
        print(f"  {pid:>6}  {etime:>10}  {cmd[:100]}")
    if not seen:
        print("  (none)")


def _read_last_jsonl(path: Path, n: int = 1):
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows[-n:]
    except Exception:
        return []


def show_runs() -> None:
    section("training runs (active <48h)")
    now = time.time()
    found = False
    for d in sorted(RUNS.iterdir()):
        metrics = next((p for p in (d / "metrics.jsonl", d / "metrics.jsonl.gz") if p.exists()), None)
        if metrics is None or now - metrics.stat().st_mtime > FRESH_S:
            continue
        found = True
        st = subprocess.run(
            [sys.executable, "-m", "tools.run_status", str(d)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout
        live = "LIVE" if now - metrics.stat().st_mtime < 300 else "stale"
        keep = [l for l in st.splitlines() if re.match(r"\s*(frames|throughput|losses|episodes)", l)]
        print(f"  {d.name} [{live}]")
        for l in keep:
            print(f"    {l.strip()}")
        skill = _read_last_jsonl(d / "skill_history.jsonl")
        if skill:
            s = skill[0]
            print(
                f"    skill: WHR {s.get('whr', 0):.0f}±{s.get('whr_std', 0):.0f} "
                f"(step {s.get('step', 0):,}, win_rate {s.get('win_rate', 0):.2f}, "
                f"n={s.get('n_games', 0)})"
            )
    if not found:
        print("  (none)")


def show_trajectories() -> None:
    section("characterization trajectories (vs champ-530m, evalwatch)")
    if not DB.exists():
        print("  (no tournament store)")
        return
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT t.name, g.entry_a, g.entry_b, g.winner FROM games g "
        "JOIN tournaments t ON t.id = g.tournament_id WHERE t.name LIKE '%_eval_step%'"
    ).fetchall()
    con.close()
    # name -> entry -> [wins, games] vs champ-530m
    by_t: dict = {}
    for name, ea, eb, winner in rows:
        for me, champ, mywin in ((ea, eb, "a"), (eb, ea, "b")):
            if champ == "champ-530m" and me != "champ-530m":
                w, g = by_t.setdefault(name, {}).setdefault(me, [0, 0])
                by_t[name][me] = [w + (winner == mywin), g + 1]
    series: dict = {}
    for name, entries in by_t.items():
        m = re.match(r"(.+)_eval_step(\d+)", name)
        if not m:
            continue
        tag, step = m.group(1), int(m.group(2))
        for entry, (w, g) in entries.items():
            kind = "masked" if entry.endswith("-noopp") else "tip"
            series.setdefault((tag, kind), []).append((step, w, g))
    for (tag, kind), pts in sorted(series.items()):
        print(f"  {tag} ({kind}):")
        for step, w, g in sorted(pts):
            wr = w / g if g else 0.0
            bar = "#" * round(wr * 20)
            print(f"    {step / 1e6:7.1f}M  {w:3d}/{g:<3d} {wr:5.1%}  {bar}")
    if not series:
        print("  (no eval tournaments yet)")


def main() -> None:
    print(f"drmc-rl overview — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    show_processes()
    show_runs()
    show_trajectories()
    print(
        "\ndeeper: tools.vs_dashboard <run> | tools.run_status <run> --history | "
        "tools.tournament report --tournament <name>"
    )


if __name__ == "__main__":
    main()
