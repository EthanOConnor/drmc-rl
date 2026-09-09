import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.run_pace_study import run_study


def test_study_keeps_process_logs_off_artifact_mount_and_preserves_previous_logs(tmp_path, monkeypatch):
    output, logs = tmp_path/"overflow", tmp_path/"local-logs"
    logs.mkdir()
    (logs/"study-training.log").write_text("previous attempt\n")
    real_run = subprocess.run
    launches = []

    def launch(command, **kwargs):
        assert command[:4] == [sys.executable,"-m","tools.program","launch"]
        assert Path(kwargs["stdout"].name).parent == logs
        launches.append(command[4])
        return real_run([sys.executable,"-c","print('child output')"], **kwargs)

    monkeypatch.setattr("tools.run_pace_study.subprocess.run",launch)
    run_study({"output":str(output), "log_directory":str(logs),
               "training_config":"train.json", "evaluation_configs":["eval.json"]})
    assert launches == ["trainer-pace-strategy","trainer-planning-arena"]
    training_log = (logs/"study-training.log").read_text()
    assert training_log.startswith("previous attempt\n")
    assert "child output" in training_log
    assert "child output" in (logs/"study-evaluation-0.log").read_text()
    assert not (output/"study-training.log").exists()
    progress = json.loads((output/"pipeline.json").read_text())
    assert progress["status"].startswith("Complete")
    assert progress["evaluations_complete"] == 1
    assert progress["log_directory"] == str(logs)


def test_failed_training_never_starts_evaluation_and_publishes_diagnostics(tmp_path, monkeypatch):
    launches = []

    def launch(command, **kwargs):
        launches.append(command[4])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("tools.run_pace_study.subprocess.run",launch)
    with pytest.raises(RuntimeError,match="training failed"):
        run_study({"output":str(tmp_path), "training_config":"train.json",
                   "evaluation_configs":["eval.json"]})
    progress = json.loads((tmp_path/"pipeline.json").read_text())
    assert progress["status"] == "Failed"
    assert progress["evaluations_complete"] == 0
    assert str(tmp_path/"study-training.log") in progress["error"]
    assert "RuntimeError: training failed" in progress["traceback"]
    assert launches == ["trainer-pace-strategy"]
