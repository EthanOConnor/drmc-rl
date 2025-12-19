"""Interactive Rich TUI for C++ engine demo playback.

This starts the C++ engine in demo mode with manual stepping and renders:
- The bottle (board) with falling + preview pill overlays.
- A live view of key shared-memory state (counters, inputs, status flags).

Controls:
  Space     Toggle run/pause
  n         Single-step one frame
  f         Step 60 frames
  + / =     Faster (higher x)
  -         Slower (lower x)
  1/2/4     Set 1x / 2x / 4x
  0         Max speed
  b         Run benchmark suite
  c         Clear benchmark output
  h / ?     Toggle help
  r         Restart demo (relaunch engine)
  q         Quit

Notes:
  - In demo mode, the engine replays the ROM demo input stream internally,
    so we do not inject controller inputs from Python.
  - This is intended as a visualization/debugging tool, not a training UI.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime guard
    RICH_AVAILABLE = False

from game_engine.engine_shm import SHM_SIZE, open_shared_memory
from training.ui.board_viewer import BoardState, parse_board_bytes, render_board_panel


MODE_DEMO = 0x00
MODE_PLAYING = 0x04

# NES frame rates (approx). Override via `--base-fps` if needed.
NES_NTSC_FPS = 60.098813897
NES_PAL_FPS = 50.006978908

COLOR_COMBO_LEFT = [0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02]
COLOR_COMBO_RIGHT = [0x00, 0x01, 0x02, 0x00, 0x01, 0x02, 0x00, 0x01, 0x02]

COLOR_NAMES = {
    0x00: "bright_yellow",
    0x01: "bright_red",
    0x02: "bright_blue",
}

PILL_STYLES: dict[int, "Style"] = {}
if RICH_AVAILABLE:
    PILL_STYLES = {c: Style(color=name, bgcolor=name) for c, name in COLOR_NAMES.items()}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_demo_pills_table() -> list[int]:
    """Load the full 128-byte demo pill table from the disassembly.

    The disassembly file splits the table into a reachable prefix (`demo_pills`)
    and an UNUSED continuation (`demo_pills_UNUSED`), but the retail ROM table is
    still 128 bytes addressable via `index & 0x7F`.
    """
    path = _repo_root() / "dr-mario-disassembly" / "data" / "drmario_data_demo_field_pills.asm"
    if not path.exists():
        return []

    data: list[int] = []
    seen = False
    for line in path.read_text().splitlines():
        if line.startswith("demo_pills:"):
            seen = True
            continue
        if not seen:
            continue
        if line.strip() == "endif":
            break
        for match in re.findall(r"\$([0-9A-Fa-f]{2})", line):
            data.append(int(match, 16))
            if len(data) == 128:
                return data
    return data if len(data) == 128 else []


DEMO_PILLS_TABLE = _load_demo_pills_table()


def _bcd_to_int(value: int) -> int:
    return ((value >> 4) & 0x0F) * 10 + (value & 0x0F)


def _decode_pill_counter_total(bcd_total: int) -> int:
    low = bcd_total & 0xFF
    high = (bcd_total >> 8) & 0xFF
    return _bcd_to_int(high) * 100 + _bcd_to_int(low)


def _buttons_to_names(buttons: int) -> str:
    # Same encoding as the engine/transcripts:
    # 0x01 R, 0x02 L, 0x04 D, 0x08 U, 0x10 START, 0x20 SELECT, 0x40 B, 0x80 A
    parts = []
    if buttons & 0x80:
        parts.append("A")
    if buttons & 0x40:
        parts.append("B")
    if buttons & 0x20:
        parts.append("SEL")
    if buttons & 0x10:
        parts.append("START")
    if buttons & 0x08:
        parts.append("U")
    if buttons & 0x04:
        parts.append("D")
    if buttons & 0x02:
        parts.append("L")
    if buttons & 0x01:
        parts.append("R")
    return "+".join(parts) if parts else "—"


@dataclass
class PlaybackConfig:
    engine_path: Path
    max_frames: int
    base_fps: float
    start_speed_x: float
    start_max_speed: bool = False


@dataclass
class PlaybackState:
    running: bool = False
    speed_x: float = 1.0
    max_speed: bool = False
    frame_accumulator: float = 0.0
    base_fps: float = 60.0
    pending_steps: int = 0
    show_help: bool = False
    status: str = ""
    bench_results: str = ""
    last_step_time: float = 0.0
    last_render_time: float = 0.0
    last_fps_time: float = 0.0
    last_fps_frame: int = 0
    fps: float = 0.0


class _RawTerminal:
    """Put stdin into cbreak mode so we can read single-key presses."""

    def __init__(self) -> None:
        self._enabled = False
        self._fd: Optional[int] = None
        self._old: Optional[Tuple[int, ...]] = None

    def __enter__(self) -> "_RawTerminal":
        if not sys.stdin.isatty():
            return self
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._enabled = True
        return self

    def __exit__(self, *args) -> None:
        if not self._enabled or self._fd is None or self._old is None:
            return
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ch


class EngineSession:
    """Owns a running engine process + shared-memory mapping."""

    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self.mm = None
        self.state = None
        self._shm_file: Optional[Path] = None
        self._prev_shm_env: Optional[str] = None

    def start(self) -> None:
        import tempfile

        self.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            self._shm_file = Path(f.name)
            f.write(b"\x00" * SHM_SIZE)

        self._prev_shm_env = os.environ.get("DRMARIO_SHM_FILE")
        os.environ["DRMARIO_SHM_FILE"] = str(self._shm_file)
        env = os.environ.copy()

        self.proc = subprocess.Popen(
            [str(self.engine_path), "--demo", "--wait-start", "--manual-step"],
            cwd=self.engine_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        time.sleep(0.05)
        self.mm, self.state = open_shared_memory()

        # Wait until the engine has finished its initial memset/arg parsing
        # pass. If we set the start-gate bit too early, the engine can wipe it
        # during initialization and then block forever in the --wait-start
        # loop.
        deadline = time.time() + 2.0
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                out, err = self.proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited during startup.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )

            # Engine sets manual-step bit (0x02) after parsing args.
            if self.state is not None and (int(self.state.control_flags) & 0x02) != 0:
                break
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for engine initialization")
            time.sleep(0.001)

        # Release wait-start gate; engine performs its initial reset afterwards.
        self.state.control_flags |= 0x01
        # External inputs are ignored in demo mode, but keep this at 0 anyway.
        self.state.buttons = 0

        # Wait for reset to populate the demo board (viruses_remaining is BCD).
        deadline = time.time() + 2.0
        while self.state is not None and int(self.state.viruses_remaining) == 0:
            if self.proc is not None and self.proc.poll() is not None:
                out, err = self.proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited during reset.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for engine reset")
            time.sleep(0.001)

    def stop(self) -> None:
        if self.state is not None:
            try:
                self.state.control_flags |= 0x08
            except Exception:
                pass
            del self.state
            self.state = None
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:
                pass
            self.mm = None

        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

        if self._shm_file is not None:
            try:
                self._shm_file.unlink()
            except Exception:
                pass
            self._shm_file = None

        if self._prev_shm_env is None:
            os.environ.pop("DRMARIO_SHM_FILE", None)
        else:
            os.environ["DRMARIO_SHM_FILE"] = self._prev_shm_env
        self._prev_shm_env = None

    def step_once(self, timeout_s: float = 0.5) -> None:
        if self.state is None:
            return
        expected = int(self.state.frame_count) + 1
        self.state.buttons = 0
        self.state.control_flags |= 0x04
        deadline = time.time() + timeout_s
        while int(self.state.frame_count) < expected:
            if self.proc is not None and self.proc.poll() is not None:
                out, err = self.proc.communicate(timeout=1)
                raise RuntimeError(
                    "Engine exited unexpectedly.\n"
                    f"stdout:\n{out.decode(errors='replace')}\n"
                    f"stderr:\n{err.decode(errors='replace')}\n"
                )
            if time.time() > deadline:
                mode = int(getattr(self.state, "mode", 0))
                flags = int(getattr(self.state, "control_flags", 0))
                frame = int(getattr(self.state, "frame_count", 0))
                raise TimeoutError(
                    "Timed out waiting for engine step "
                    f"(frame={frame} expected={expected} mode=0x{mode:02X} control_flags=0x{flags:02X})"
                )
            time.sleep(0.0001)


def _build_board_state(state) -> BoardState:
    # Shared-memory board is already the NES 0x0400 layout (top row first).
    board = parse_board_bytes(bytes(state.board))

    raw_row = int(state.falling_pill_row)
    raw_col = int(state.falling_pill_col)
    raw_orient = int(state.falling_pill_orient) & 0x03

    # Convert NES row-from-bottom (0=bottom, 15=top) to screen row (0=top).
    screen_row = 15 - raw_row

    # Map NES orientation (0..3) to viewer orientation (0=vertical, 1=horizontal),
    # and swap colors for the "swapped" rotations (2,3) so the overlay matches
    # the actual half positions used by `confirmPlacement`.
    is_vertical = (raw_orient & 0x01) == 0x01
    viewer_orient = 0 if is_vertical else 1
    swap = raw_orient in {2, 3}
    color_l = int(state.falling_pill_color_l) & 0x03
    color_r = int(state.falling_pill_color_r) & 0x03
    if swap:
        color_l, color_r = color_r, color_l

    return BoardState(
        board=board,
        falling_row=screen_row,
        falling_col=raw_col,
        falling_orient=viewer_orient,
        falling_color_l=color_l,
        falling_color_r=color_r,
        preview_color_l=int(state.preview_pill_color_l) & 0x03,
        preview_color_r=int(state.preview_pill_color_r) & 0x03,
        viruses_remaining=_bcd_to_int(int(state.viruses_remaining)),
        frame_count=int(state.frame_count),
        pill_count=_decode_pill_counter_total(int(state.pill_counter_total)),
        level=int(state.level),
    )


def _render_upcoming_pills(state, count: int = 16) -> Text:
    text = Text()
    if int(state.mode) != MODE_DEMO or not DEMO_PILLS_TABLE:
        text.append("n/a", style="dim")
        return text

    start_idx = int(state.pill_counter) & 0x7F
    text.append(f"idx={start_idx:03d} ", style="dim")

    for i in range(count):
        pill_id = DEMO_PILLS_TABLE[(start_idx + i) & 0x7F]
        c_l = COLOR_COMBO_LEFT[pill_id] & 0x03
        c_r = COLOR_COMBO_RIGHT[pill_id] & 0x03

        style_l = PILL_STYLES.get(c_l, Style(color="white", bgcolor="white"))
        style_r = PILL_STYLES.get(c_r, Style(color="white", bgcolor="white"))
        text.append("██", style=style_l)
        text.append("██", style=style_r)
        if i != count - 1:
            text.append(" ")
    return text


def _render_stats_panel(state, ui: PlaybackState) -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim", width=18)
    table.add_column("Value", overflow="fold")

    mode = int(state.mode)
    mode_name = "DEMO" if mode == MODE_DEMO else ("PLAY" if mode == MODE_PLAYING else "UNK")

    viruses_bcd = int(state.viruses_remaining)
    viruses = _bcd_to_int(viruses_bcd)

    pill_total_raw = int(state.pill_counter_total)
    pill_total = _decode_pill_counter_total(pill_total_raw)

    table.add_row("run", "RUNNING" if ui.running else "PAUSED")
    if ui.max_speed:
        table.add_row("playback", f"MAX ({ui.fps:.1f} fps)")
    else:
        target_fps = ui.base_fps * ui.speed_x
        table.add_row("playback", f"{ui.speed_x:.2f}x (target {target_fps:.1f} fps, eff {ui.fps:.1f})")
    table.add_row("frame", str(int(state.frame_count)))
    table.add_row("mode", f"0x{mode:02X} ({mode_name})")
    table.add_row("buttons", f"0x{int(state.buttons):02X} ({_buttons_to_names(int(state.buttons))})")
    table.add_row("pressed", f"0x{int(state.buttons_pressed):02X}")
    table.add_row("held", f"0x{int(state.buttons_held):02X}")
    table.add_row("viruses", f"{viruses} (bcd=0x{viruses_bcd:02X})")
    table.add_row("pill_total", f"{pill_total} (bcd=0x{pill_total_raw:04X})")
    table.add_row("pill_idx", f"{int(state.pill_counter)}")
    table.add_row("upcoming", _render_upcoming_pills(state))
    table.add_row("game_speed", f"{int(state.speed_setting)}  speed_ups={int(state.speed_ups)}")
    table.add_row("speed_counter", str(int(state.speed_counter)))
    table.add_row("hor_velocity", str(int(state.hor_velocity)))
    table.add_row("lock_counter", str(int(state.lock_counter)))
    table.add_row("wait_frames", str(int(state.wait_frames)))
    table.add_row("stage_clear", str(int(state.stage_clear)))
    table.add_row("level_fail", str(int(state.level_fail)))
    table.add_row("fail_count", str(int(state.fail_count)))

    if ui.status:
        table.add_row("status", ui.status)

    return Panel(table, title="[bold]Engine State[/bold]", border_style="green")


def _render_help_panel() -> Panel:
    lines = [
        "Space: run/pause",
        "n: step 1",
        "f: step 60",
        "+ / =: faster (x)",
        "-: slower (x)",
        "1: 1.0x",
        "2: 2.0x",
        "4: 4.0x",
        "0: MAX speed",
        "b: benchmark",
        "c: clear benchmark",
        "r: restart",
        "h / ?: help",
        "q: quit",
    ]
    return Panel(Text("\n".join(lines), style="dim"), title="[bold]Help[/bold]", border_style="yellow")


def _render_footer(ui: PlaybackState) -> Text:
    status = "⏸ PAUSED" if not ui.running else "▶ RUNNING"
    return Text(
        f"{status} | Space Pause/Run | n Step | +/- Speed(x) | 0 MAX | b Bench | h Help | q Quit",
        style="dim",
    )


def _render_layout(board_panel: Panel, stats_panel: Panel, ui: PlaybackState) -> Layout:
    layout = Layout()
    layout.split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=1))
    layout["left"].update(board_panel)

    right = Layout()
    bench_panel = None
    if ui.bench_results:
        bench_panel = Panel(Text(ui.bench_results, style="dim"), title="[bold]Benchmark[/bold]", border_style="cyan")

    if ui.show_help:
        if bench_panel is not None:
            right.split(
                Layout(name="stats"),
                Layout(name="bench", size=7),
                Layout(name="help", size=12),
                Layout(name="footer", size=1),
            )
            right["stats"].update(stats_panel)
            right["bench"].update(bench_panel)
            right["help"].update(_render_help_panel())
            right["footer"].update(_render_footer(ui))
        else:
            right.split(Layout(name="stats"), Layout(name="help", size=12), Layout(name="footer", size=1))
            right["stats"].update(stats_panel)
            right["help"].update(_render_help_panel())
            right["footer"].update(_render_footer(ui))
    else:
        if bench_panel is not None:
            right.split(Layout(name="stats"), Layout(name="bench", size=7), Layout(name="footer", size=1))
            right["stats"].update(stats_panel)
            right["bench"].update(bench_panel)
            right["footer"].update(_render_footer(ui))
        else:
            right.split(Layout(name="stats"), Layout(name="footer", size=1))
            right["stats"].update(stats_panel)
            right["footer"].update(_render_footer(ui))

    layout["right"].update(right)
    return layout


def _bench_engine_manual_step(
    engine_path: Path, max_frames: int, render_ui: bool
) -> Tuple[float, int, int]:
    """Benchmark demo playback using manual-step mode.

    Returns:
      (seconds, frames_advanced)
    """
    session = EngineSession(engine_path)
    session.start()
    renders = 0

    try:
        start = time.perf_counter()
        last_render = start
        dummy_ui = PlaybackState(running=True, max_speed=True, base_fps=60.0)

        if render_ui and RICH_AVAILABLE:
            with open(os.devnull, "w") as devnull:
                bench_console = Console(file=devnull, force_terminal=False, width=140)
                while session.state is not None:
                    if int(session.state.mode) != MODE_DEMO:
                        break
                    if int(session.state.frame_count) >= max_frames:
                        break

                    session.step_once(timeout_s=1.0)

                    now = time.perf_counter()
                    if now - last_render >= 1.0 / 30.0:
                        board_state = _build_board_state(session.state)
                        board_panel = render_board_panel(board_state, title="C++ Engine Demo (bench)")
                        stats_panel = _render_stats_panel(session.state, dummy_ui)
                        bench_console.print(_render_layout(board_panel, stats_panel, dummy_ui))
                        renders += 1
                        last_render = now
        else:
            while session.state is not None:
                if int(session.state.mode) != MODE_DEMO:
                    break
                if int(session.state.frame_count) >= max_frames:
                    break
                session.step_once(timeout_s=1.0)

        seconds = time.perf_counter() - start
        frames = int(session.state.frame_count) if session.state is not None else 0
        return seconds, frames, renders
    finally:
        session.stop()


def _bench_tui_only_replay(max_frames: int, render_ui: bool) -> Tuple[float, int, int]:
    """Benchmark the TUI render pipeline without the engine (replay transcript)."""
    from tools.game_transcript import load_json

    transcript_path = _repo_root() / "data" / "nes_demo.json"
    transcript = load_json(transcript_path)

    board = bytearray(transcript.initial_board)
    frame_count = 0
    last_render = time.perf_counter()

    renders = 0

    start = time.perf_counter()
    dummy_ui = PlaybackState(running=True, max_speed=True, base_fps=60.0)

    if render_ui and RICH_AVAILABLE:
        with open(os.devnull, "w") as devnull:
            bench_console = Console(file=devnull, force_terminal=False, width=140)
            for fs in transcript.frames:
                frame_count = fs.frame
                if frame_count >= max_frames:
                    break
                if fs.board_changes:
                    for idx, _old, new in fs.board_changes:
                        board[idx] = new

                now = time.perf_counter()
                # Render *every frame* to measure the full UI pipeline throughput.
                board_state = BoardState(board=parse_board_bytes(memoryview(board)), frame_count=frame_count)
                board_panel = render_board_panel(board_state, title="TUI-only replay (bench)")
                stats_panel = Panel(Text(f"frame={frame_count}", style="dim"), border_style="green")
                bench_console.print(_render_layout(board_panel, stats_panel, dummy_ui))
                renders += 1
                last_render = now
    else:
        for fs in transcript.frames:
            frame_count = fs.frame
            if frame_count >= max_frames:
                break
            if fs.board_changes:
                for idx, _old, new in fs.board_changes:
                    board[idx] = new

    seconds = time.perf_counter() - start
    return seconds, frame_count, renders


def _bench_engine_freerun(engine_path: Path, frame_budget: int) -> Tuple[float, int, int]:
    """Benchmark the engine in freerun mode (`--no-sleep`, no manual stepping)."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        shm_file = Path(f.name)
        f.write(b"\x00" * SHM_SIZE)

    prev_shm_env = os.environ.get("DRMARIO_SHM_FILE")
    os.environ["DRMARIO_SHM_FILE"] = str(shm_file)
    env = os.environ.copy()

    proc = subprocess.Popen(
        [str(engine_path), "--demo", "--wait-start", "--no-sleep"],
        cwd=engine_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        time.sleep(0.05)
        mm, state = open_shared_memory()
        try:
            # Wait for reset/init (see EngineSession.start for rationale).
            deadline = time.time() + 2.0
            while int(state.viruses_remaining) == 0:
                if proc.poll() is not None:
                    out, err = proc.communicate(timeout=1)
                    raise RuntimeError(
                        "Engine exited during startup.\n"
                        f"stdout:\n{out.decode(errors='replace')}\n"
                        f"stderr:\n{err.decode(errors='replace')}\n"
                    )
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for engine initialization")
                time.sleep(0.001)

            state.frame_count = 0
            state.frame_budget = int(frame_budget)
            start = time.perf_counter()
            state.control_flags |= 0x01  # release wait-start gate
            proc.wait(timeout=5)
            seconds = time.perf_counter() - start
            frames = int(state.frame_count)
            return seconds, frames, 0
        finally:
            del state
            mm.close()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

        try:
            shm_file.unlink()
        except Exception:
            pass
        if prev_shm_env is None:
            os.environ.pop("DRMARIO_SHM_FILE", None)
        else:
            os.environ["DRMARIO_SHM_FILE"] = prev_shm_env


def run_benchmark_suite(engine_path: Path, max_frames: int = 8000) -> str:
    """Run a small benchmark suite and return a human-readable summary."""
    lines: list[str] = []

    # 1) Engine-only (freerun, true engine speed).
    try:
        sec, frames, _renders = _bench_engine_freerun(engine_path, frame_budget=min(5701, max_frames))
        fps = frames / sec if sec > 0 else 0.0
        lines.append(f"engine only (freerun): {fps:,.0f} fps")
    except Exception as e:
        lines.append(f"engine only (freerun): error: {e}")

    # 2) Engine-only (manual step, no UI).
    try:
        sec, frames, _renders = _bench_engine_manual_step(
            engine_path, max_frames=max_frames, render_ui=False
        )
        fps = frames / sec if sec > 0 else 0.0
        lines.append(f"engine only (manual):   {fps:,.0f} fps")
    except Exception as e:
        lines.append(f"engine only (manual):   error: {e}")

    # 3) Engine + TUI (manual step, render pipeline to /dev/null).
    try:
        sec, frames, renders = _bench_engine_manual_step(
            engine_path, max_frames=max_frames, render_ui=True
        )
        fps = frames / sec if sec > 0 else 0.0
        hz = renders / sec if sec > 0 else 0.0
        lines.append(f"engine+TUI (manual):    {fps:,.0f} fps (render {hz:.1f} Hz)")
    except Exception as e:
        lines.append(f"engine+TUI (manual):    error: {e}")

    # 4) TUI-only (replay, render every frame to /dev/null).
    try:
        # Rendering every frame is expensive; cap to keep the suite usable from
        # the interactive TUI.
        tui_frames = min(max_frames, 600)
        sec, frames, renders = _bench_tui_only_replay(max_frames=tui_frames, render_ui=True)
        fps = frames / sec if sec > 0 else 0.0
        hz = renders / sec if sec > 0 else 0.0
        lines.append(f"TUI only (replay {tui_frames}f): {fps:,.0f} fps (render {hz:.1f} Hz)")
    except Exception as e:
        lines.append(f"TUI only (replay):      error: {e}")

    return "\n".join(lines)


def run_tui(cfg: PlaybackConfig) -> None:
    if not RICH_AVAILABLE:
        raise SystemExit("Rich is required. Install with: pip install rich")
    console = Console()

    session = EngineSession(cfg.engine_path)
    session.start()

    ui = PlaybackState(
        running=False,
        speed_x=cfg.start_speed_x,
        max_speed=cfg.start_max_speed,
        base_fps=cfg.base_fps,
    )
    ui.last_step_time = time.perf_counter()
    ui.last_render_time = time.perf_counter()
    ui.last_fps_time = time.perf_counter()
    ui.last_fps_frame = 0

    render_interval_s = 1.0 / 30.0
    interactive = sys.stdin.isatty()
    if not interactive:
        ui.running = True
        ui.show_help = True
        ui.status = "stdin is not a TTY: running non-interactively until demo end / max_frames"

    raw = _RawTerminal()
    with raw:
        with Live(Panel("Starting…", border_style="blue"), console=console, refresh_per_second=30, screen=False) as live:
            while True:
                now = time.perf_counter()
                keypress = raw.poll_key()
                if keypress is not None:
                    if keypress in {"q", "Q"}:
                        break
                    if keypress == " ":
                        ui.running = not ui.running
                        ui.status = ""
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress in {"n", "N"}:
                        ui.pending_steps += 1
                    elif keypress in {"f", "F"}:
                        ui.pending_steps += 60
                    elif keypress in {"b", "B"}:
                        # Run benchmark suite (blocks briefly).
                        ui.running = False
                        ui.pending_steps = 0
                        ui.status = "Benchmarking…"
                        # Force a render of the status line before running.
                        if session.state is not None:
                            board_state = _build_board_state(session.state)
                            board_panel = render_board_panel(board_state, title="C++ Engine Demo")
                            stats_panel = _render_stats_panel(session.state, ui)
                            live.update(_render_layout(board_panel, stats_panel, ui))
                        session.stop()
                        ui.bench_results = run_benchmark_suite(cfg.engine_path)
                        ui.status = "Benchmark complete (press c to clear)"
                        session.start()
                    elif keypress in {"c", "C"}:
                        ui.bench_results = ""
                    elif keypress in {"+", "="}:
                        ui.max_speed = False
                        ui.speed_x = min(ui.speed_x * 1.25, 256.0)
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress == "-":
                        ui.max_speed = False
                        ui.speed_x = max(ui.speed_x / 1.25, 0.1)
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress == "0":
                        ui.max_speed = True
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress == "1":
                        ui.max_speed = False
                        ui.speed_x = 1.0
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress == "2":
                        ui.max_speed = False
                        ui.speed_x = 2.0
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress == "4":
                        ui.max_speed = False
                        ui.speed_x = 4.0
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                    elif keypress in {"h", "H", "?"}:
                        ui.show_help = not ui.show_help
                    elif keypress in {"r", "R"}:
                        ui.status = "Restarting…"
                        ui.running = False
                        ui.pending_steps = 0
                        ui.frame_accumulator = 0.0
                        ui.last_step_time = time.perf_counter()
                        session.start()

                # Step logic
                try:
                    if session.state is None:
                        ui.status = "Engine not running"
                    else:
                        # Stop if demo ended.
                        if int(session.state.mode) != MODE_DEMO:
                            ui.running = False
                            ui.status = "Demo ended (mode switched out of DEMO)"
                            if not interactive:
                                break

                        if int(session.state.frame_count) >= cfg.max_frames:
                            ui.running = False
                            ui.status = f"Reached max_frames={cfg.max_frames}"
                            if not interactive:
                                break

                        if ui.pending_steps > 0:
                            session.step_once()
                            ui.pending_steps -= 1
                        elif ui.running:
                            if ui.max_speed:
                                # Max-speed: step as many frames as we can until the
                                # next render deadline (or a safety cap), so the UI
                                # stays responsive without artificially throttling.
                                max_batch = 5000
                                deadline = ui.last_render_time + render_interval_s
                                for _ in range(max_batch):
                                    if time.perf_counter() >= deadline:
                                        break
                                    if int(session.state.mode) != MODE_DEMO:
                                        break
                                    if int(session.state.frame_count) >= cfg.max_frames:
                                        break
                                    session.step_once()
                            else:
                                target_fps = ui.base_fps * ui.speed_x
                                dt = max(0.0, now - ui.last_step_time)
                                ui.last_step_time = now
                                ui.frame_accumulator += dt * target_fps

                                steps_to_do = int(ui.frame_accumulator)
                                if steps_to_do > 0:
                                    max_batch = 2000
                                    if steps_to_do > max_batch:
                                        steps_to_do = max_batch
                                    for _ in range(steps_to_do):
                                        if int(session.state.mode) != MODE_DEMO:
                                            break
                                        if int(session.state.frame_count) >= cfg.max_frames:
                                            break
                                        session.step_once()
                                    ui.frame_accumulator = max(0.0, ui.frame_accumulator - steps_to_do)
                        else:
                            # Prevent "catch up" after a long pause.
                            ui.last_step_time = now
                            ui.frame_accumulator = 0.0
                except Exception as e:
                    ui.running = False
                    ui.status = f"Step error: {e}"

                # FPS estimate based on observed frame counter delta (works for both
                # manual-step and any future freerun modes).
                if session.state is not None and now - ui.last_fps_time >= 0.25:
                    dt = now - ui.last_fps_time
                    frame_now = int(session.state.frame_count)
                    df = frame_now - ui.last_fps_frame
                    inst = df / dt if dt > 0 else 0.0
                    ui.fps = inst if ui.fps == 0.0 else (ui.fps * 0.8 + inst * 0.2)
                    ui.last_fps_time = now
                    ui.last_fps_frame = frame_now

                # Render at a fixed interval to reduce overhead.
                if now - ui.last_render_time >= render_interval_s:
                    if session.state is None:
                        live.update(Panel("Engine not running", border_style="red"))
                    else:
                        board_state = _build_board_state(session.state)
                        board_panel = render_board_panel(board_state, title="C++ Engine Demo")
                        stats_panel = _render_stats_panel(session.state, ui)
                        live.update(_render_layout(board_panel, stats_panel, ui))
                    ui.last_render_time = now

                time.sleep(0.001)

    session.stop()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive TUI for C++ engine demo playback")
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("game_engine/drmario_engine"),
        help="Path to engine binary",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8000,
        help="Stop stepping after this many frames",
    )
    parser.add_argument(
        "--speed-x",
        type=float,
        default=1.0,
        help="Target speed multiplier vs base FPS (e.g. 2.4 = 2.4× NTSC/PAL rate; 0 = MAX)",
    )
    parser.add_argument(
        "--region",
        choices=["ntsc", "pal"],
        default="ntsc",
        help="Base framerate used for speed-x (ntsc≈60.10, pal≈50.01)",
    )
    parser.add_argument(
        "--base-fps",
        type=float,
        default=None,
        help="Override base FPS directly (advanced)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # legacy: seconds per frame; superseded by --speed-x
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite and exit",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).parent.parent
    engine_path = (repo_root / args.engine).resolve()
    if not engine_path.exists():
        print(f"Engine not found at {engine_path}. Build with: make -C game_engine")
        return 1

    if args.benchmark:
        print(run_benchmark_suite(engine_path, max_frames=int(args.max_frames)))
        return 0

    base_fps = float(args.base_fps) if args.base_fps is not None else (
        NES_NTSC_FPS if args.region == "ntsc" else NES_PAL_FPS
    )

    start_max_speed = False
    speed_x = float(args.speed_x)
    if args.delay is not None:
        # Back-compat: delay seconds per frame -> speed multiplier.
        if args.delay <= 0:
            start_max_speed = True
            speed_x = 1.0
        else:
            speed_x = (1.0 / float(args.delay)) / base_fps
    if speed_x <= 0:
        start_max_speed = True
        speed_x = 1.0

    run_tui(
        PlaybackConfig(
            engine_path=engine_path,
            max_frames=int(args.max_frames),
            base_fps=base_fps,
            start_speed_x=speed_x,
            start_max_speed=start_max_speed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
