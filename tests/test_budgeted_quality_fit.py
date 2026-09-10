import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import budgeted_quality_fit as budgeted


def config(tmp_path):
    return dict(
        output=str(tmp_path / "run"),
        scratch_root=str(tmp_path),
        allocation_seconds=0.6,
        gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        fit_config={"seed": 7},
    )


def fake_fitter(
    monkeypatch, *, fail=False, publish=True, ignore_termination=False, finish_after=10
):
    # A separate real process exercises termination and atomic-file selection;
    # no model/GPU work is claimed by this hardware-independent control test.
    code = (
        """
import json,pathlib,sys,time,os,signal
d=json.loads(pathlib.Path(sys.argv[1]).read_text());p=pathlib.Path(d['checkpoint_directory']);p.mkdir()
signal.signal(signal.SIGTERM, signal.SIG_IGN) if IGNORE else None
if PUBLISH:
 (p/'checkpoint-000001.pt').write_bytes(b'checked weights')
 now=time.monotonic_ns()
 (p/'index.json').write_text(json.dumps({'schema':'drmc-quality-checkpoints-v1','checkpoints':[{'filename':'checkpoint-000001.pt','epoch':1,'validated_monotonic_ns':now,'stored_monotonic_ns':now,'accepted_examples':8}]}))
 (p/'checkpoint-000002.pt.tmp').write_bytes(b'incomplete weights')
sys.exit(3) if FAIL else time.sleep(10)
""".replace("IGNORE", str(ignore_termination))
        .replace("PUBLISH", str(publish))
        .replace("FAIL", str(fail))
        .replace("time.sleep(10)", f"time.sleep({finish_after})")
    )
    monkeypatch.setattr(
        budgeted, "child_command", lambda path: [sys.executable, "-c", code, str(path)]
    )
    real_popen = subprocess.Popen
    state = {}

    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        state["child"] = child
        return child

    monkeypatch.setattr(budgeted.subprocess, "Popen", launch)
    monkeypatch.setattr(
        budgeted, "cuda_pids", lambda _: {state["child"].pid} if "child" in state else set()
    )
    return state


def test_deadline_exports_only_the_last_checked_model_and_stops_child(tmp_path, monkeypatch):
    state = fake_fitter(monkeypatch)
    result = budgeted.run(config(tmp_path))
    assert result["status"] == "Complete" and result["stop_reason"] == "allocation_exhausted"
    assert result["selected_checkpoint"]["epoch"] == 1
    assert result["allocation_elapsed_seconds"] >= 0.6
    assert result["observed_child_cuda_context"] and result["eligible_for_allocation_comparison"]
    assert state["child"].poll() is not None
    assert (tmp_path / "run/diagnostic.pt").read_bytes() == b"checked weights"
    assert not Path(result["checkpoint_scratch"]).exists()


def test_existing_gpu_user_is_never_terminated_or_started_over(tmp_path, monkeypatch):
    monkeypatch.setattr(budgeted, "cuda_pids", lambda _: {9087})
    monkeypatch.setattr(budgeted, "child_command", lambda _: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="already has compute processes"):
        budgeted.run(config(tmp_path))
    assert not (tmp_path / "run/diagnostic.pt").exists()


def test_natural_finish_reports_unused_time_instead_of_claiming_full_compute(tmp_path, monkeypatch):
    fake_fitter(monkeypatch, finish_after=0.1)
    result = budgeted.run(dict(config(tmp_path), allocation_seconds=3))
    assert result["status"] == "Complete" and result["stop_reason"] == "fit_exited"
    assert result["unused_allocation_seconds"] > 1 and result["cutoff_overrun_seconds"] == 0


def test_late_contention_invalidates_comparison_but_preserves_checked_weights(
    tmp_path, monkeypatch
):
    state = fake_fitter(monkeypatch)

    def pids(_):
        if "child" not in state:
            return set()
        state["checks"] = state.get("checks", 0) + 1
        return {state["child"].pid} | ({123456} if state["checks"] > 1 else set())

    monkeypatch.setattr(budgeted, "cuda_pids", pids)
    with pytest.raises(RuntimeError, match="shared with processes"):
        budgeted.run(dict(config(tmp_path), allocation_seconds=3))
    result = json.loads((tmp_path / "run/progress.json").read_text())
    assert not result["eligible_for_allocation_comparison"]
    assert state["child"].poll() is not None
    assert (tmp_path / "run/diagnostic.pt").read_bytes() == b"checked weights"
    assert Path(result["checkpoint_scratch"]).exists()


def test_fitter_failure_is_not_a_successful_budget_cutoff(tmp_path, monkeypatch):
    fake_fitter(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="fitter failed"):
        budgeted.run(config(tmp_path))
    result = json.loads((tmp_path / "run/progress.json").read_text())
    assert result["status"] == "Failed" and result["child_exit_code"] == 3
    assert (tmp_path / "run/diagnostic.pt").exists()


def test_empty_budget_does_not_export_unchecked_work(tmp_path, monkeypatch):
    fake_fitter(monkeypatch, publish=False)
    with pytest.raises(RuntimeError, match="before a fully checked checkpoint"):
        budgeted.run(config(tmp_path))
    assert not (tmp_path / "run/diagnostic.pt").exists()


def test_watchdog_kills_its_own_unresponsive_child(tmp_path, monkeypatch):
    state = fake_fitter(monkeypatch, ignore_termination=True)
    result = budgeted.run(config(tmp_path))
    assert result["status"] == "Complete" and state["child"].returncode < 0
    assert result["cutoff_overrun_seconds"] >= 2


def test_nvidia_query_filters_gpu_and_rejects_unreadable_ownership(monkeypatch):
    def response(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "GPU-ab12, 123\nGPU-ffff, 456\n", "")

    monkeypatch.setattr(budgeted.subprocess, "run", response)
    assert budgeted.cuda_pids("GPU-ab12") == {123}
    monkeypatch.setattr(
        budgeted.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "unknown", ""),
    )
    with pytest.raises(RuntimeError, match="cannot verify"):
        budgeted.cuda_pids("GPU-ab12")
