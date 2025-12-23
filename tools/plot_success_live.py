#!/usr/bin/env python3
"""Live-updating curriculum success plot with a chart selector."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as a script from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception as exc:  # pragma: no cover
    print(f"Tkinter is required for live plotting: {exc}", file=sys.stderr)
    raise SystemExit(2)

from tools.plot_success_by_level import (
    _METRIC_SPECS,
    _resolve_metrics_path,
    load_curriculum_series,
)

_DEFAULT_METRICS = ["confidence_lower_bound", "rate_current"]

_PALETTE = [
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


def _center_window(win: tk.Misc, *, width: int, height: int) -> None:
    try:
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
    except Exception:
        return
    x = max(0, int((screen_w - width) / 2))
    y = max(0, int((screen_h - height) / 2))
    win.geometry(f"{width}x{height}+{x}+{y}")


def _select_path_with_root(root: tk.Tk, *, start_dir: Path) -> Optional[Path]:
    selection: Dict[str, Optional[Path]] = {"path": None}

    for child in list(root.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass

    root.deiconify()
    root.title("Select run data")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=(16, 14))
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Choose a metrics file or run directory.").pack(anchor="w")

    button_row = ttk.Frame(frame, padding=(0, 10))
    button_row.pack(anchor="w")

    done = tk.BooleanVar(value=False)

    def _choose_file() -> None:
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select metrics.jsonl(.gz)",
            initialdir=str(start_dir),
            filetypes=[
                ("metrics.jsonl.gz", "*.jsonl.gz"),
                ("metrics.jsonl", "*.jsonl"),
                ("gzip", "*.gz"),
                ("All files", "*"),
            ],
        )
        if file_path:
            selection["path"] = Path(file_path)
            done.set(True)

    def _choose_dir() -> None:
        dir_path = filedialog.askdirectory(
            parent=root,
            title="Select a run directory",
            initialdir=str(start_dir),
        )
        if dir_path:
            selection["path"] = Path(dir_path)
            done.set(True)

    def _cancel() -> None:
        done.set(True)

    ttk.Button(button_row, text="Choose file", command=_choose_file).pack(side="left")
    ttk.Button(button_row, text="Choose directory", command=_choose_dir).pack(
        side="left", padx=8
    )
    ttk.Button(button_row, text="Cancel", command=_cancel).pack(side="left")

    root.protocol("WM_DELETE_WINDOW", _cancel)
    root.update_idletasks()
    _center_window(root, width=360, height=120)
    try:
        root.lift()
        root.focus_force()
    except Exception:
        pass
    root.wait_variable(done)
    return selection["path"]


def _series_steps(series_by_level: Dict[int, List[object]]) -> List[int]:
    steps: List[int] = []
    for segments in series_by_level.values():
        for seg in segments:
            steps.extend(getattr(seg, "steps", []))
    return steps


def _series_values(series_by_level: Dict[int, List[object]]) -> List[float]:
    values: List[float] = []
    for segments in series_by_level.values():
        for seg in segments:
            values.extend(getattr(seg, "values", []))
    return values


class LivePlotApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        metrics_path: Path,
        default_metric: str,
        min_episodes: int,
        refresh_ms: int,
    ) -> None:
        self.root = root
        self.metrics_path = metrics_path
        self.min_episodes = int(min_episodes)
        self.refresh_ms = int(refresh_ms)
        self.metric_var = tk.StringVar(value=default_metric)

        self._last_size: Optional[int] = None
        self._last_mtime: Optional[float] = None
        self._after_id: Optional[str] = None
        self._closing = False

        root.title("Curriculum plot")

        top = ttk.Frame(root, padding=(10, 8))
        top.pack(side="top", fill="x")

        ttk.Label(top, text="chart:").pack(side="left")
        metric_menu = ttk.OptionMenu(
            top,
            self.metric_var,
            self.metric_var.get(),
            *self._metric_choices(),
        )
        metric_menu.pack(side="left", padx=6)

        ttk.Button(top, text="refresh", command=lambda: self.refresh(force=True)).pack(
            side="left", padx=6
        )

        self.status_label = ttk.Label(top, text="")
        self.status_label.pack(side="left", padx=10)

        self.canvas = tk.Canvas(root, width=1200, height=650, bg="white", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        self.metric_var.trace_add("write", lambda *args: self.refresh(force=True))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh(force=True)
        self._schedule()

    def _metric_choices(self) -> List[str]:
        return [m for m in _DEFAULT_METRICS if m in _METRIC_SPECS]

    def close(self) -> None:
        self._closing = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.root.destroy()

    def _schedule(self) -> None:
        if self._closing:
            return
        # Single scheduled loop to avoid background threads/processes.
        self._after_id = self.root.after(self.refresh_ms, self._scheduled_refresh)

    def _scheduled_refresh(self) -> None:
        if self._closing:
            return
        self.refresh(force=False)
        self._schedule()

    def refresh(self, *, force: bool) -> None:
        if self._closing:
            return
        try:
            stat = self.metrics_path.stat()
        except FileNotFoundError:
            self.status_label.configure(text=f"missing: {self.metrics_path}")
            self._draw_message("metrics file not found")
            return

        if not force:
            if self._last_size == stat.st_size and self._last_mtime == stat.st_mtime:
                return

        self._last_size = stat.st_size
        self._last_mtime = stat.st_mtime

        metric_name = self.metric_var.get()
        try:
            series_by_level, meta = load_curriculum_series(
                self.metrics_path,
                metric=metric_name,
                min_episodes=self.min_episodes,
            )
        except Exception as exc:
            self.status_label.configure(text=f"load error: {type(exc).__name__}: {exc}")
            self._draw_message("failed to parse metrics (see status)")
            return

        run_id = str(meta.get("run_id") or self.metrics_path.parent.name)
        title_stub = _METRIC_SPECS[metric_name][2]
        ylabel = _METRIC_SPECS[metric_name][1]
        title = f"{title_stub} - {run_id}"

        updated = time.strftime("%H:%M:%S")
        self.status_label.configure(text=f"{self.metrics_path.name} | {updated}")
        self.root.title(title)

        self._draw_plot(series_by_level, title=title, ylabel=ylabel)

    def _draw_message(self, text: str) -> None:
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        self.canvas.create_text(w // 2, h // 2, text=text, fill="#666666")

    def _draw_plot(self, series_by_level: Dict[int, List[object]], *, title: str, ylabel: str) -> None:
        self.canvas.delete("all")
        steps_all = _series_steps(series_by_level)
        values_all = _series_values(series_by_level)
        if not steps_all or not values_all:
            self._draw_message("no data yet")
            return

        x_min = int(min(steps_all))
        x_max = int(max(steps_all))
        if x_max <= x_min:
            x_max = x_min + 1

        y_min = 0.0
        y_max = 1.0

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
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

        x0 = margin_l
        y0 = margin_t + plot_h
        x1 = margin_l + plot_w
        y1 = margin_t
        self.canvas.create_line(x0, y0, x1, y0, fill="#333333")
        self.canvas.create_line(x0, y0, x0, y1, fill="#333333")

        self.canvas.create_text(
            margin_l + plot_w // 2,
            height - margin_b // 2,
            text="steps",
            fill="#111111",
        )
        self.canvas.create_text(
            margin_l // 2,
            margin_t + plot_h // 2,
            text=ylabel,
            fill="#111111",
            angle=90,
        )

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = y_px(t)
            self.canvas.create_line(x0 - 5, yy, x0, yy, fill="#333333")
            self.canvas.create_text(x0 - 10, yy, text=f"{t:.2f}", anchor="e", fill="#111111")
            self.canvas.create_line(x0, yy, x1, yy, fill="#eeeeee")

        ticks = 5
        for i in range(ticks + 1):
            step = int(round(x_min + (x_max - x_min) * (i / ticks)))
            xx = x_px(step)
            self.canvas.create_line(xx, y0, xx, y0 + 5, fill="#333333")
            self.canvas.create_text(xx, y0 + 18, text=str(step), anchor="n", fill="#111111")

        levels = sorted(series_by_level.keys())
        for i, lvl in enumerate(levels):
            for seg in series_by_level[lvl]:
                steps = getattr(seg, "steps", [])
                values = getattr(seg, "values", [])
                if len(steps) < 2:
                    continue
                pts: List[int] = []
                for step, val in zip(steps, values):
                    pts.extend([x_px(int(step)), y_px(float(val))])
                self.canvas.create_line(*pts, fill=_PALETTE[i % len(_PALETTE)], width=2)

        lx = margin_l + plot_w + 20
        ly = margin_t + 10
        self.canvas.create_text(lx, ly - 10, text="levels", anchor="nw", fill="#111111")
        for i, lvl in enumerate(levels):
            y = ly + i * 18
            if y > height - margin_b - 10:
                self.canvas.create_text(lx, y, text="(legend truncated)", anchor="nw", fill="#888888")
                break
            color = _PALETTE[i % len(_PALETTE)]
            self.canvas.create_line(lx, y + 8, lx + 18, y + 8, fill=color, width=3)
            self.canvas.create_text(lx + 24, y + 8, text=f"level {lvl}", anchor="w", fill="#111111")


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
        help="Default chart: confidence_lower_bound or rate_current.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=10,
        help="Ignore points until curriculum/episodes_current_total reaches this value.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
        help="Seconds between refreshes.",
    )
    args = parser.parse_args(argv)

    root = tk.Tk()

    selection: Optional[Path]
    if args.path:
        selection = Path(args.path)
    else:
        selection = _select_path_with_root(root, start_dir=(Path.cwd() / "runs"))
        if selection is None:
            print("No selection made.")
            root.destroy()
            return 2

    try:
        metrics_path = _resolve_metrics_path(selection)
    except FileNotFoundError as exc:
        print(str(exc))
        root.destroy()
        return 1

    metric = str(args.metric or "").strip().lower()
    if metric not in _DEFAULT_METRICS:
        metric = "confidence_lower_bound"

    for child in list(root.winfo_children()):
        try:
            child.destroy()
        except Exception:
            pass
    app = LivePlotApp(
        root,
        metrics_path=metrics_path,
        default_metric=metric,
        min_episodes=int(args.min_episodes),
        refresh_ms=int(max(200, float(args.refresh_seconds) * 1000)),
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
