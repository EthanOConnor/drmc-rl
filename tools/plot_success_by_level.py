#!/usr/bin/env python3
"""Plot curriculum success metrics vs steps per level from a training run.

This script is designed for the `training.run` JSONL log format:
  - run directory contains `metrics.jsonl.gz` written by `training/diagnostics/logger.py`
  - curriculum-enabled runs log:
      - `curriculum/current_level` (int-like)
      - `curriculum/rate_current` (success rate estimate for current level)
      - `curriculum/confidence_lower_bound` (Wilson LB for success rate)

By default, the script opens a file/directory picker to select a run log.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Series:
    steps: List[int]
    values: List[float]


_METRIC_SPECS = {
    "rate_current": (
        "curriculum/rate_current",
        "success rate (curriculum/rate_current)",
        "Success rate vs steps (per level)",
    ),
    "confidence_lower_bound": (
        "curriculum/confidence_lower_bound",
        "confidence lower bound (curriculum/confidence_lower_bound)",
        "Confidence lower bound vs steps (per level)",
    ),
}

_METRIC_ALIASES = {
    "rate": "rate_current",
    "rate_current": "rate_current",
    "success": "rate_current",
    "confidence": "confidence_lower_bound",
    "confidence_lower_bound": "confidence_lower_bound",
    "lb": "confidence_lower_bound",
}


def _try_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int,)):
            return int(value)
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str) and value.strip():
            return int(round(float(value)))
    except Exception:
        return None
    return None


def _try_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except Exception:
        return None
    return None


def _select_path_via_tk(*, start_dir: Path) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()

    # Prefer selecting the metrics file directly, but accept a directory too.
    file_path = filedialog.askopenfilename(
        title="Select metrics.jsonl(.gz) (or cancel to select a directory)",
        initialdir=str(start_dir),
        filetypes=[
            ("metrics.jsonl(.gz)", "*.jsonl *.jsonl.gz"),
            ("metrics.jsonl.gz", "*.jsonl.gz"),
            ("metrics.jsonl", "*.jsonl"),
            ("All files", "*"),
        ],
    )
    if file_path:
        return Path(file_path)

    dir_path = filedialog.askdirectory(
        title="Select a run directory containing metrics.jsonl",
        initialdir=str(start_dir),
    )
    if dir_path:
        selection = Path(dir_path)
    else:
        selection = None
    try:
        root.destroy()
    except Exception:
        pass
    return selection


def _resolve_metrics_path(selection: Path) -> Path:
    sel = selection.expanduser().resolve()
    if sel.is_file():
        return sel
    metrics_gz = sel / "metrics.jsonl.gz"
    if metrics_gz.is_file():
        return metrics_gz
    metrics = sel / "metrics.jsonl"
    if metrics.is_file():
        return metrics
    candidates = sorted(
        list(sel.glob("*/metrics.jsonl.gz")) + list(sel.glob("*/metrics.jsonl")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find metrics.jsonl(.gz) under: {sel}")
    if len(candidates) == 1:
        return candidates[0]

    print(f"Multiple runs found under {sel}:")
    for i, cand in enumerate(candidates[:30]):
        print(f"  [{i}] {cand.parent}")
    default = 0
    try:
        choice = input(f"Select run index (default {default}): ").strip()
    except EOFError:
        choice = ""
    idx = default
    if choice:
        try:
            idx = int(choice)
        except Exception:
            idx = default
    idx = max(0, min(idx, len(candidates) - 1))
    return candidates[idx]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".gz":
        # Be tolerant of truncated/in-progress gzip streams (common when training is still running).
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                with io.TextIOWrapper(gz, encoding="utf-8") as f:
                    while True:
                        try:
                            line = f.readline()
                        except (EOFError, OSError):
                            break
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        return

    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_metric(metric: str) -> str:
    key = str(metric or "").strip().lower()
    resolved = _METRIC_ALIASES.get(key)
    if resolved is None:
        raise ValueError(f"Unknown metric {metric!r}. Use one of: {sorted(_METRIC_SPECS)}")
    return resolved


def _extract_metric(row: Dict[str, float], metric_key: str) -> Optional[float]:
    if metric_key == "curriculum/rate_current":
        rate = row.get(metric_key)
        if rate is None:
            ws = row.get("curriculum/window_successes")
            wn = row.get("curriculum/window_n")
            if ws is not None and wn is not None and float(wn) > 0.0:
                rate = float(ws) / float(wn)
        return rate
    return row.get(metric_key)


def load_curriculum_series(
    metrics_path: Path,
    *,
    metric: str,
    min_episodes: int,
) -> Tuple[Dict[int, List[Series]], Dict[str, Any]]:
    """Return per-level segmented (steps, values) series and metadata dict.

    Segmentation rule: we only connect points for a level when that level is the
    active `curriculum/current_level` on consecutive logged steps. This avoids
    drawing sawtooth lines that jump across long stretches where that level is
    not being measured.
    """

    metadata: Dict[str, Any] = {}
    by_step: DefaultDict[int, Dict[str, float]] = defaultdict(dict)

    for entry in _iter_jsonl(metrics_path):
        if entry.get("type") == "metadata" and isinstance(entry.get("data"), dict):
            metadata.update(entry["data"])
            continue

        if entry.get("type") != "scalar":
            continue
        step_i = _try_int(entry.get("step"))
        name = entry.get("name")
        value_f = _try_float(entry.get("value"))
        if step_i is None or not isinstance(name, str) or value_f is None:
            continue
        by_step[int(step_i)][name] = float(value_f)

    metric_key = _METRIC_SPECS[_resolve_metric(metric)][0]
    min_episodes = int(max(0, int(min_episodes)))

    segments_by_level: DefaultDict[int, List[Series]] = defaultdict(list)
    steps_sorted = sorted(by_step.keys())

    active_level: Optional[int] = None
    active_steps: List[int] = []
    active_values: List[float] = []

    def _flush() -> None:
        nonlocal active_level, active_steps, active_values
        if active_level is None or not active_steps:
            active_level = None
            active_steps = []
            active_values = []
            return
        segments_by_level[int(active_level)].append(Series(steps=active_steps, values=active_values))
        active_level = None
        active_steps = []
        active_values = []

    for step in steps_sorted:
        row = by_step[step]
        level = _try_int(row.get("curriculum/current_level"))
        episodes = row.get("curriculum/episodes_current_total")
        if min_episodes > 0 and (episodes is None or float(episodes) < float(min_episodes)):
            _flush()
            continue

        value = _extract_metric(row, metric_key)

        if level is None or value is None:
            _flush()
            continue

        if active_level is None:
            active_level = int(level)
            active_steps = [int(step)]
            active_values = [float(value)]
            continue

        if int(level) != int(active_level):
            _flush()
            active_level = int(level)
            active_steps = [int(step)]
            active_values = [float(value)]
            continue

        active_steps.append(int(step))
        active_values.append(float(value))

    _flush()

    return dict(segments_by_level), metadata


def plot_series(series_by_level: Dict[int, List[Series]], *, title: str, ylabel: str) -> None:
    if not series_by_level:
        raise RuntimeError(
            "No per-level curriculum series found. "
            "Expected `curriculum/current_level` + curriculum metric in metrics.jsonl(.gz)."
        )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        _plot_series_tk(series_by_level, title=title, ylabel=ylabel)
        return

    levels = sorted(series_by_level.keys())
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(12, 6))
    for i, lvl in enumerate(levels):
        color = cmap(i % 20)
        first = True
        for seg in series_by_level[lvl]:
            if len(seg.steps) < 2:
                continue
            plt.plot(
                seg.steps,
                seg.values,
                label=(f"level {lvl}" if first else "_nolegend_"),
                linewidth=2.0,
                color=color,
            )
            first = False

    plt.ylim(0.0, 1.0)
    plt.xlabel("steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def _plot_series_tk(series_by_level: Dict[int, List[Series]], *, title: str, ylabel: str) -> None:
    """Minimal Tkinter line plot fallback (no external plotting deps)."""

    try:
        import tkinter as tk
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Tkinter is required for plotting when matplotlib is unavailable.") from e

    levels = sorted(series_by_level.keys())

    steps_all = [s for segments in series_by_level.values() for seg in segments for s in seg.steps]
    if not steps_all:
        raise RuntimeError("No data points to plot.")

    x_min = int(min(steps_all))
    x_max = int(max(steps_all))
    if x_max <= x_min:
        x_max = x_min + 1

    y_min = 0.0
    y_max = 1.0

    width = 1200
    height = 650
    margin_l = 80
    margin_r = 240
    margin_t = 50
    margin_b = 70
    plot_w = max(1, width - margin_l - margin_r)
    plot_h = max(1, height - margin_t - margin_b)

    def x_px(step: int) -> int:
        t = (float(step) - float(x_min)) / float(x_max - x_min)
        return int(margin_l + t * plot_w)

    def y_px(val: float) -> int:
        t = (float(val) - y_min) / float(y_max - y_min)
        return int(margin_t + (1.0 - t) * plot_h)

    # A small, pleasant palette (cycled).
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#b279a2",
        "#9d755d",
        "#edc949",
        "#76b7b2",
        "#59a14f",
        "#af7aa1",
    ]

    root = tk.Tk()
    root.title(title)
    canvas = tk.Canvas(root, width=width, height=height, bg="white", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # Axes.
    x0 = margin_l
    y0 = margin_t + plot_h
    x1 = margin_l + plot_w
    y1 = margin_t
    canvas.create_line(x0, y0, x1, y0, fill="#333333")
    canvas.create_line(x0, y0, x0, y1, fill="#333333")

    canvas.create_text(margin_l + plot_w // 2, height - margin_b // 2, text="steps", fill="#111111")
    canvas.create_text(margin_l // 2, margin_t + plot_h // 2, text=ylabel, fill="#111111", angle=90)

    # Y ticks (0..1).
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y_px(t)
        canvas.create_line(x0 - 5, yy, x0, yy, fill="#333333")
        canvas.create_text(x0 - 10, yy, text=f"{t:.2f}", anchor="e", fill="#111111")
        canvas.create_line(x0, yy, x1, yy, fill="#eeeeee")

    # X ticks (5 ticks).
    ticks = 5
    for i in range(ticks + 1):
        step = int(round(x_min + (x_max - x_min) * (i / ticks)))
        xx = x_px(step)
        canvas.create_line(xx, y0, xx, y0 + 5, fill="#333333")
        canvas.create_text(xx, y0 + 18, text=str(step), anchor="n", fill="#111111")

    # Plot series.
    for i, lvl in enumerate(levels):
        for seg in series_by_level[lvl]:
            if len(seg.steps) < 2:
                continue
            pts: List[int] = []
            for step, val in zip(seg.steps, seg.values, strict=True):
                pts.extend([x_px(int(step)), y_px(float(val))])
            canvas.create_line(*pts, fill=palette[i % len(palette)], width=2)

    # Legend.
    lx = margin_l + plot_w + 20
    ly = margin_t + 10
    canvas.create_text(lx, ly - 10, text="levels", anchor="nw", fill="#111111")
    for i, lvl in enumerate(levels):
        y = ly + i * 18
        if y > height - margin_b - 10:
            canvas.create_text(lx, y, text="(legend truncated)", anchor="nw", fill="#888888")
            break
        color = palette[i % len(palette)]
        canvas.create_line(lx, y + 8, lx + 18, y + 8, fill=color, width=3)
        canvas.create_text(lx + 24, y + 8, text=f"level {lvl}", anchor="w", fill="#111111")

    root.mainloop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to a run directory or a metrics.jsonl(.gz) file. If omitted, opens a picker.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="confidence_lower_bound",
        help=(
            "Curriculum metric to plot: confidence_lower_bound (default), rate_current, or both. "
            "Aliases: confidence, lb, rate, success."
        ),
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="Ignore points until curriculum/episodes_current_total reaches this value.",
    )
    args = parser.parse_args(argv)

    selection: Optional[Path]
    if args.path:
        selection = Path(args.path)
    else:
        selection = _select_path_via_tk(start_dir=(Path.cwd() / "runs"))
        if selection is None:
            print("No selection made.")
            return 2

    try:
        metrics_path = _resolve_metrics_path(selection)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    metric = str(args.metric or "").strip().lower()
    if metric == "both":
        run_id: Optional[str] = None
        for metric_name in ("confidence_lower_bound", "rate_current"):
            series_by_level, meta = load_curriculum_series(
                metrics_path,
                metric=metric_name,
                min_episodes=int(args.min_episodes),
            )
            if run_id is None:
                run_id = str(meta.get("run_id") or metrics_path.parent.name)
            _, ylabel, title_stub = _METRIC_SPECS[metric_name]
            title = f"{title_stub} — {run_id}"
            plot_series(series_by_level, title=title, ylabel=ylabel)
    else:
        metric_name = _resolve_metric(metric)
        series_by_level, meta = load_curriculum_series(
            metrics_path,
            metric=metric_name,
            min_episodes=int(args.min_episodes),
        )
        run_id = str(meta.get("run_id") or metrics_path.parent.name)
        _, ylabel, title_stub = _METRIC_SPECS[metric_name]
        title = f"{title_stub} — {run_id}"
        plot_series(series_by_level, title=title, ylabel=ylabel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
