"""Live training dashboard (rich TUI) for Dr. Mario runs.

Shows training throughput/losses, VS practice stats (win rates, garbage,
match lengths) and the WHR-scale skill estimate (tools/skill_grade.py output)
side by side, refreshing from a run directory's metrics + skill files.

    python -m tools.vs_dashboard runs/<run_dir> [--refresh 5]

Works on any run: 1P-only runs simply leave the VS panes sparse. Reads:
  - <run>/metrics.jsonl.gz   (scalar stream; tolerant of live partial gzip)
  - <run>/skill_history.jsonl (optional; rows {"step":N,"whr":x,"whr_std":s,...})
  - <run>/eval_history.jsonl  (optional; tools/eval_watch output)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

SPARK = "▁▂▃▄▅▆▇█"


def spark(values: List[float], width: int = 40) -> str:
    vals = [v for v in values if v is not None][-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in vals)


def read_scalars(path: Path) -> Dict[str, List[Any]]:
    """name -> list of (step, value), tolerant of truncated live gzip."""
    series: Dict[str, List[Any]] = {}
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    break
                if r.get("type") != "scalar":
                    continue
                series.setdefault(r["name"], []).append((r["step"], r["value"]))
    except (EOFError, OSError, FileNotFoundError):
        pass
    return series


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def last(series: Dict[str, List[Any]], name: str, fmt: str = "{:.3f}") -> str:
    pts = series.get(name)
    if not pts:
        return "-"
    v = pts[-1][1]
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def vals(series: Dict[str, List[Any]], name: str) -> List[float]:
    return [p[1] for p in series.get(name, [])]


def build_view(run_dir: Path) -> Layout:
    series = read_scalars(run_dir / "metrics.jsonl.gz")
    skill = read_jsonl(run_dir / "skill_history.jsonl")
    evals = read_jsonl(run_dir / "eval_history.jsonl")

    steps = max((pts[-1][0] for pts in series.values() if pts), default=0)

    # --- training panel ---
    t = Table.grid(padding=(0, 2))
    t.add_row("frames", f"{steps:,}")
    t.add_row("frames/s", last(series, "perf/sps", "{:.0f}"))
    t.add_row("decisions/s", last(series, "perf/dps", "{:.0f}"))
    t.add_row("update s", last(series, "perf/update_sec", "{:.2f}"))
    t.add_row("KL", last(series, "policy/kl", "{:.4f}"))
    t.add_row("entropy", last(series, "policy/entropy", "{:.2f}") + "  " + spark(vals(series, "policy/entropy")))
    t.add_row("return", last(series, "train/return_mean", "{:.2f}") + "  " + spark(vals(series, "train/return_mean")))
    training = Panel(t, title="training", border_style="cyan")

    # --- VS practice panel ---
    v = Table.grid(padding=(0, 2))
    v.add_row("win rate (p1 vs pool)", last(series, "vs/win_rate", "{:.2f}") + "  " + spark(vals(series, "vs/win_rate")))
    v.add_row("vs random/greedy", last(series, "vs/win_rate_baseline", "{:.2f}"))
    v.add_row("garbage sent /min", last(series, "vs/cpm", "{:.2f}") + "  " + spark(vals(series, "vs/cpm")))
    v.add_row("garbage recv /min", last(series, "vs/cpm_received", "{:.2f}"))
    v.add_row("match len (s, p50)", last(series, "vs/match_len_p50_sec", "{:.0f}"))
    v.add_row("matches played", last(series, "vs/matches_total", "{:.0f}"))
    v.add_row("opponent pool size", last(series, "vs/opponent_pool", "{:.0f}"))
    if vals(series, "vs/league_win_rate"):  # league modes only (docs/LEAGUE.md)
        v.add_row(
            "league wr (targets)",
            last(series, "vs/league_win_rate", "{:.2f}") + "  " + spark(vals(series, "vs/league_win_rate")),
        )
    vs_panel = Panel(v, title="vs practice", border_style="magenta")

    # --- skill panel (WHR scale) ---
    s = Table.grid(padding=(0, 2))
    if skill:
        cur = skill[-1]
        whr_hist = [r.get("whr") for r in skill if r.get("whr") is not None]
        s.add_row("est. WHR", f"[bold]{cur.get('whr', float('nan')):.0f}[/] ± {cur.get('whr_std', 0):.0f}")
        s.add_row("trend", spark(whr_hist, 48))
        for k in ("cpm", "spd", "salt", "win_rate"):
            if k in cur:
                s.add_row(f"  {k}", f"{cur[k]:.2f}")
        s.add_row("graded games", str(cur.get("n_games", "-")))
    else:
        s.add_row("est. WHR", "[dim]no skill_history.jsonl yet[/]")
    skill_panel = Panel(s, title="skill (fightcade WHR scale)", border_style="green")

    # --- eval panel ---
    e = Table.grid(padding=(0, 2))
    if evals:
        latest = evals[-1]
        for lvl, d in sorted(latest.get("levels", {}).items(), key=lambda kv: int(kv[0])):
            hist = [r["levels"].get(lvl, {}).get("clear_rate") for r in evals if lvl in r.get("levels", {})]
            e.add_row(f"L{lvl}", f"{d['clear_rate']*100:4.0f}%", spark(hist, 32))
        e.add_row("@frames", f"{latest.get('step', 0):,}")
    else:
        e.add_row("[dim]no eval_history.jsonl[/]", "", "")
    eval_panel = Panel(e, title="1P eval sweeps", border_style="yellow")

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=11),
        Layout(name="bottom"),
    )
    layout["top"].split_row(Layout(training), Layout(vs_panel))
    layout["bottom"].split_row(Layout(skill_panel), Layout(eval_panel))
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--refresh", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    console = Console()
    if args.once:
        console.print(build_view(run_dir))
        return
    with Live(build_view(run_dir), console=console, refresh_per_second=1) as live:
        while True:
            time.sleep(args.refresh)
            live.update(build_view(run_dir))


if __name__ == "__main__":
    main()
