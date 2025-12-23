#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _wilson_lower_bound(*, successes: int, n: int, sigmas: float) -> float:
    n_i = int(n)
    if n_i <= 0:
        return 0.0
    k_i = int(max(0, min(int(successes), n_i)))
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return float(k_i) / float(n_i)

    p_hat = float(k_i) / float(n_i)
    z2 = z * z
    denom = 1.0 + z2 / float(n_i)
    center = p_hat + z2 / (2.0 * float(n_i))
    adj = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * float(n_i))) / float(n_i))
    return float(min(1.0, max(0.0, (center - adj) / denom)))


def _confidence_window_size(*, target: float, sigmas: float) -> int:
    t = float(target)
    if t <= 0.0:
        return 1
    if t >= 1.0:
        return 1_000_000_000
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return 1
    denom = max(1e-12, 1.0 - t)
    n_est = (z * z) * (1.0 + t) / denom
    return int(max(1, int(math.ceil(n_est - 1e-12))))


def _perfect_streak_window_size(*, target: float, sigmas: float) -> int:
    t = float(target)
    if t <= 0.0:
        return 1
    if t >= 1.0:
        return 1_000_000_000
    z = float(max(0.0, float(sigmas)))
    if z == 0.0:
        return 1
    z2 = z * z
    n = int(math.floor((z2 * t) / max(1e-12, (1.0 - t)))) + 1
    return int(max(1, n))


def _min_successes_for_lb(*, n: int, target: float, sigmas: float) -> int:
    for k in range(int(n) + 1):
        if _wilson_lower_bound(successes=k, n=int(n), sigmas=float(sigmas)) > float(target):
            return int(k)
    return int(n) + 1


def _parse_csv_floats(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = _to_float(part)
        if val is None:
            continue
        out.append(float(val))
    return out


def _print_confidence_table(*, targets: List[float], sigmas_list: List[float]) -> None:
    if not targets or not sigmas_list:
        print("No targets/sigmas provided.")
        return
    headers = (
        "target",
        "sigmas",
        "near_n",
        "near_k_min",
        "near_min_rate",
        "perfect_n",
    )
    print(" ".join(f\"{h:>14}\" for h in headers))
    for t in targets:
        for s in sigmas_list:
            near_n = _confidence_window_size(target=float(t), sigmas=float(s))
            near_k = _min_successes_for_lb(n=near_n, target=float(t), sigmas=float(s))
            near_min_rate = float(near_k) / float(near_n) if near_n > 0 else 0.0
            perfect_n = _perfect_streak_window_size(target=float(t), sigmas=float(s))
            print(
                f\"{float(t):>14.4f}\"
                f\"{float(s):>14.2f}\"
                f\"{int(near_n):>14}\"
                f\"{int(near_k):>14}\"
                f\"{near_min_rate:>14.4f}\"
                f\"{int(perfect_n):>14}\"
            )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".gz":
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


def _load_advancement_events(metrics_path: Path) -> List[Dict[str, Any]]:
    events_by_step: Dict[int, Dict[str, Any]] = {}
    for rec in _iter_jsonl(metrics_path):
        if rec.get("type") != "scalar":
            continue
        name = rec.get("name")
        if not isinstance(name, str) or not name.startswith("curriculum/advanced"):
            continue
        step = _to_int(rec.get("step")) or 0
        events_by_step.setdefault(step, {})[name] = rec.get("value")

    events: List[Dict[str, Any]] = []
    for step in sorted(events_by_step):
        data = events_by_step[step]
        if "curriculum/advanced_to" not in data:
            continue
        level_to = _to_int(data.get("curriculum/advanced_to"))
        level_from = _to_int(data.get("curriculum/advanced_from"))
        frames_total = _to_int(data.get("curriculum/advanced_frames_total")) or int(step)
        frames_delta = _to_int(data.get("curriculum/advanced_frames_delta"))
        episodes_total = _to_int(data.get("curriculum/advanced_episodes_total"))
        episodes_delta = _to_int(data.get("curriculum/advanced_episodes_delta"))
        events.append(
            {
                "step": int(step),
                "level_from": level_from,
                "level_to": level_to,
                "frames_total": frames_total,
                "frames_delta": frames_delta,
                "episodes_total": episodes_total,
                "episodes_delta": episodes_delta,
            }
        )
    return events


def _print_table(events: List[Dict[str, Any]]) -> None:
    if not events:
        print("No curriculum advancement events found.")
        return
    headers = (
        "step",
        "from",
        "to",
        "frames_total",
        "frames_delta",
        "episodes_total",
        "episodes_delta",
    )
    print(" ".join(f"{h:>14}" for h in headers))
    for row in events:
        print(
            f"{row['step']:>14}"
            f"{row['level_from'] if row['level_from'] is not None else '-':>14}"
            f"{row['level_to'] if row['level_to'] is not None else '-':>14}"
            f"{row['frames_total'] if row['frames_total'] is not None else '-':>14}"
            f"{row['frames_delta'] if row['frames_delta'] is not None else '-':>14}"
            f"{row['episodes_total'] if row['episodes_total'] is not None else '-':>14}"
            f"{row['episodes_delta'] if row['episodes_delta'] is not None else '-':>14}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize curriculum advancement events.")
    parser.add_argument(
        "--confidence-table",
        action="store_true",
        help="Print a table of expected window sizes for confidence-based curricula.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="0.01,0.632,0.865,0.90,0.95,0.99",
        help="Comma-separated list of success thresholds to tabulate (default: common ln-hop-back values).",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="1,2,3",
        help="Comma-separated list of sigma values (default: 1,2,3).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="runs/smdp_ppo",
        help="Path to metrics.jsonl(.gz) or a run directory (default: runs/smdp_ppo)",
    )
    args = parser.parse_args()

    if bool(getattr(args, "confidence_table", False)):
        targets = _parse_csv_floats(str(getattr(args, "targets", "")))
        sigmas_list = _parse_csv_floats(str(getattr(args, "sigmas", "")))
        _print_confidence_table(targets=targets, sigmas_list=sigmas_list)
        return

    metrics_input = Path(args.metrics).expanduser()
    metrics_path = metrics_input
    if metrics_input.is_dir():
        direct_gz = metrics_input / "metrics.jsonl.gz"
        direct = metrics_input / "metrics.jsonl"
        if direct_gz.is_file():
            metrics_path = direct_gz
        elif direct.is_file():
            metrics_path = direct
        else:
            candidates = []
            for child in metrics_input.iterdir():
                if not child.is_dir():
                    continue
                candidate_gz = child / "metrics.jsonl.gz"
                candidate = child / "metrics.jsonl"
                if candidate_gz.is_file():
                    candidates.append(candidate_gz)
                elif candidate.is_file():
                    candidates.append(candidate)
            if not candidates:
                raise FileNotFoundError(f"No metrics.jsonl(.gz) found under {metrics_input}")
            metrics_path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.jsonl(.gz) not found at {metrics_path}")
    events = _load_advancement_events(metrics_path)
    _print_table(events)


if __name__ == "__main__":
    main()
