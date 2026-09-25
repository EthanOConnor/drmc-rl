"""Stdlib-only client helpers: authenticated API calls, artifact upload, the training hook.

``register_snapshot`` is what a training run (or ``tools.rating_pool watch-run``)
calls when it writes a snapshot: upload the checkpoint if the coordinator
lacks it, register the entrant with its lineage, and keep the run's open
stop-rule panel job in place.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

DEFAULT_URL = os.environ.get("DRMC_POOL_URL", "http://192.168.157.190:8097")
DEFAULT_TOKEN_FILE = "~/.config/drmc-rl/study-worker.token"


def read_token(token=None, token_file=None):
    token = token or os.environ.get("DRMC_POOL_TOKEN")
    if token:
        return token.strip()
    return Path(token_file or DEFAULT_TOKEN_FILE).expanduser().read_text().strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class PoolClient:
    def __init__(self, url=None, token=None, token_file=None, timeout=300.0):
        from tools.trainer_arena_distributed import StudyClient
        self.base = (url or DEFAULT_URL).rstrip("/")
        self.study = StudyClient(self.base, read_token(token, token_file), timeout=timeout)

    def get(self, path, **query):
        suffix = ("?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})) if query else ""
        return self.study.retrying("GET", path + suffix, patience=120)

    def post(self, path, payload, *, compress=False, patience=120):
        return self.study.retrying("POST", path, payload, compress=compress, patience=patience)

    def register(self, event):
        return self.post("/api/v1/pool/registry", event)

    def has_artifact(self, digest):
        return self.get(f"/api/v1/pool/artifacts/{digest}/exists")["exists"]

    def upload(self, path, digest=None, *, mb_per_second=40.0):
        """Upload a checkpoint (throttled) unless the coordinator already has it."""
        digest = digest or sha256_file(path)
        if self.has_artifact(digest):
            return digest
        size = os.path.getsize(path)

        class Throttled:
            def __init__(self, stream):
                self.stream, self.started, self.sent = stream, time.monotonic(), 0

            def read(self, n=-1):
                block = self.stream.read(min(n if n > 0 else 1 << 20, 1 << 20))
                self.sent += len(block)
                ahead = self.sent / (mb_per_second * 2 ** 20) - (time.monotonic() - self.started)
                if ahead > 0:
                    time.sleep(ahead)
                return block
        with open(path, "rb") as stream:
            request = urllib.request.Request(f"{self.base}/api/v1/pool/artifacts/{digest}", data=Throttled(stream),
                                             method="PUT", headers={"Authorization": f"Bearer {self.study.token}",
                                                                    "Content-Length": str(size),
                                                                    "Content-Type": "application/octet-stream"})
            with urllib.request.urlopen(request, timeout=3600) as response:
                reply = json.loads(response.read())
        if reply.get("sha256") != digest:
            raise RuntimeError(f"upload of {path} failed: {reply}")
        return digest


def artifact_record(path, digest=None):
    path = Path(path).expanduser().resolve()
    return dict(sha256=digest or sha256_file(path), size=path.stat().st_size, name=path.name, paths=[str(path)])


def register_snapshot(client, path, *, run, entrant_id, era, step, parent=None, recipe=None, name=None,
                      panel_set=None, panel_games=128, panel_priority=60, anchor=None, upload=True, notes=""):
    """Register one training snapshot as an active pool entrant and keep the run's panel job open."""
    record = artifact_record(path)
    if upload:
        client.upload(path, record["sha256"])
    lineage = dict(run=run, step=int(step), parent=parent, recipe=recipe)
    client.register(dict(type="entrant", id=entrant_id, name=name or entrant_id, loader="plain", checkpoint=record,
                         era=era, lineage=lineage, status="active", tags=["snapshot", f"run:{run}"], notes=notes,
                         by=f"snapshot-hook:{run}"))
    if panel_set:
        job = dict(type="job", id=f"stop-panel-{run}", title=f"Stop-rule panel for {run}", mode="vs",
                   entrants=[f"{run}-*"], conditions=[f"set:{panel_set}"], games=int(panel_games),
                   priority=int(panel_priority), open=True, status="active", owner=f"run:{run}",
                   by=f"snapshot-hook:{run}")
        if anchor:
            job["opponents"] = [anchor]
        client.register(job)
    return record

