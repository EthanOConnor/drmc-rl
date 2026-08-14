"""Read consented Professor Pills replay exports and summarize input cadence."""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

NTSC_FPS = 60.0988
GAMEPLAY_BUTTONS = 0xC7  # A, B, Down, Left, Right


@dataclass(frozen=True, slots=True)
class InputCadence:
    frames: int
    press_events: int
    active_frames: int
    input_changes: int

    @property
    def actions_per_minute(self) -> float:
        minutes = self.frames / (NTSC_FPS * 60.0)
        return 0.0 if minutes <= 0 else self.press_events / minutes


@dataclass(frozen=True, slots=True)
class ProfessorRun:
    run_id: str
    handle: str
    artifact: dict[str, Any]
    tape_header: dict[str, Any]
    runs: tuple[tuple[int, int, int], ...]

    def cadence(self, side: int) -> InputCadence:
        if side not in {1, 2}:
            raise ValueError("side must be 1 or 2")
        frames = press_events = active_frames = input_changes = 0
        previous = 0
        for count, p1, p2 in self.runs:
            value = (p1 if side == 1 else p2) & GAMEPLAY_BUTTONS
            frames += count
            active_frames += count if value else 0
            press_events += (value & ~previous).bit_count()
            input_changes += int(value != previous)
            previous = value
        return InputCadence(frames, press_events, active_frames, input_changes)

    @property
    def versus_summary(self) -> dict[str, Any] | None:
        payload = self.artifact.get("payload", {})
        if payload.get("mode") != "versus":
            return None
        summary = payload.get("summary")
        return summary if isinstance(summary, dict) else None


def _decode_urlsafe(value: str) -> bytes:
    encoded = value.encode("ascii")
    return base64.urlsafe_b64decode(encoded + b"=" * (-len(encoded) % 4))


def decode_pptape(data: bytes) -> tuple[dict[str, Any], tuple[tuple[int, int, int], ...]]:
    view = memoryview(data)
    position = 0

    def take(size: int) -> memoryview:
        nonlocal position
        if size < 0 or position + size > len(view):
            raise ValueError("truncated pptape1")
        value = view[position : position + size]
        position += size
        return value

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    if bytes(take(8)) != b"PPTAPE1\0":
        raise ValueError("not a pptape1 replay")
    header = json.loads(bytes(take(u32())))
    if header.get("schema") != "pptape1":
        raise ValueError("unsupported Professor Pills tape schema")
    for _ in range(u32()):
        take(9)
    runs = tuple((u32(), int(take(1)[0]), int(take(1)[0])) for _ in range(u32()))
    if position != len(view):
        raise ValueError("pptape1 has trailing data")
    if sum(count for count, _, _ in runs) != int(header.get("frames", -1)):
        raise ValueError("pptape1 frame count does not match its runs")
    return header, runs


def read_export(path: str | Path) -> Iterator[ProfessorRun]:
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("schema") != "ppcorpus1":
                raise ValueError(f"line {line_number}: unsupported corpus schema")
            if item.get("contributed_side", 1) != 1:
                raise ValueError(f"line {line_number}: unsupported contributed side")
            artifact = item.get("artifact")
            payload = artifact.get("payload") if isinstance(artifact, dict) else None
            if not isinstance(payload, dict) or not isinstance(payload.get("tape"), str):
                raise ValueError(f"line {line_number}: replay evidence is missing")
            header, runs = decode_pptape(_decode_urlsafe(payload["tape"]))
            account = item.get("account") or {}
            yield ProfessorRun(
                run_id=str(item["run_id"]),
                handle=str(account.get("handle", "")),
                artifact=artifact,
                tape_header=header,
                runs=runs,
            )


def corpus_report(path: str | Path) -> dict[str, Any]:
    players: dict[str, dict[str, float]] = {}
    calibration: list[dict[str, Any]] = []
    count = 0
    for run in read_export(path):
        count += 1
        cadence = run.cadence(1)
        aggregate = players.setdefault(
            run.handle,
            {"runs": 0.0, "frames": 0.0, "press_events": 0.0, "active_frames": 0.0},
        )
        aggregate["runs"] += 1
        aggregate["frames"] += cadence.frames
        aggregate["press_events"] += cadence.press_events
        aggregate["active_frames"] += cadence.active_frames
        summary = run.versus_summary
        model = None if summary is None else summary.get("player_two")
        if isinstance(summary, dict) and isinstance(model, dict):
            calibration.append(
                {
                    "run_id": run.run_id,
                    "human": run.handle,
                    "human_wins": int(summary.get("p1_wins", 0)),
                    "model_wins": int(summary.get("p2_wins", 0)),
                    "winner_side": int(summary.get("winner_side", 0)),
                    "target_rating": float(model.get("target_rating", 0.0)),
                    "timing_scale": float(model.get("timing_scale_milli", 1000)) / 1000.0,
                    "human_raw_apm": cadence.actions_per_minute,
                    "model_raw_apm": run.cadence(2).actions_per_minute,
                }
            )
    player_rows = {}
    for handle, row in players.items():
        minutes = row["frames"] / (NTSC_FPS * 60.0)
        player_rows[handle] = {
            "runs": int(row["runs"]),
            "frames": int(row["frames"]),
            "raw_actions_per_minute": 0.0 if minutes <= 0 else row["press_events"] / minutes,
            "raw_active_input_fraction": 0.0
            if row["frames"] <= 0
            else row["active_frames"] / row["frames"],
        }
    return {
        "schema": "ppcorpus-report-v1",
        "runs": count,
        "players": player_rows,
        "calibration_matches": calibration,
    }


__all__ = ["InputCadence", "ProfessorRun", "corpus_report", "decode_pptape", "read_export"]
