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
import urllib.error
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


DEFAULT_ACCESS_FILE = "~/.config/drmc-rl/pool-access.env"
PART_BYTES = 32 << 20          # chunked uploads stay under Cloudflare's 100 MB request-body cap


class AccessDenied(RuntimeError):
    """Cloudflare Access refused the request (missing or wrong service token)."""


def read_access(path=None):
    """Cloudflare Access service-token headers from a mode-600 env file, or {} when absent.

    File format (KEY=VALUE lines): CF_ACCESS_CLIENT_ID=..., CF_ACCESS_CLIENT_SECRET=...
    The environment variables of the same names take precedence.
    """
    values = {k: os.environ[k] for k in ("CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET") if os.environ.get(k)}
    file = Path(os.environ.get("DRMC_POOL_ACCESS_FILE") or path or DEFAULT_ACCESS_FILE).expanduser()
    if len(values) < 2 and file.is_file():
        if file.stat().st_mode & 0o077:
            raise PermissionError(f"{file} must be private (chmod 600): it holds a Cloudflare Access secret")
        for line in file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET") and value:
                values.setdefault(key, value.strip().strip('"'))
    if len(values) != 2:
        return {}
    return {"CF-Access-Client-Id": values["CF_ACCESS_CLIENT_ID"],
            "CF-Access-Client-Secret": values["CF_ACCESS_CLIENT_SECRET"]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None             # an Access login redirect must surface, not be followed


def _open(request, timeout):
    """urlopen that turns Cloudflare Access denials into a clear AccessDenied."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location", "") if error.headers else ""
        if error.code in (301, 302, 303, 307, 308) and "cloudflareaccess.com" in location:
            raise AccessDenied("Cloudflare Access redirected to its login page: this host needs the pool service "
                               f"token in {DEFAULT_ACCESS_FILE} (CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET)") from None
        if error.code == 403 and "json" not in (error.headers.get("Content-Type", "") if error.headers else ""):
            raise AccessDenied("Cloudflare Access refused the request (HTTP 403): the service token is missing, "
                               f"wrong or not allowed by the Access policy ({DEFAULT_ACCESS_FILE})") from None
        raise
    kind = response.headers.get("Content-Type", "")
    if "text/html" in kind and "json" not in kind:
        response.close()
        raise AccessDenied("the pool URL answered with an HTML page (an Access or proxy login?), not the pool API")
    return response


def access_study_client(base, token, *, timeout=300.0, access=None):
    """The distributed-study client with Cloudflare Access headers on every request."""
    from tools.trainer_arena_distributed import CoordinatorError, StudyClient
    import gzip
    import tempfile
    headers = read_access() if access is None else access

    class AccessStudyClient(StudyClient):
        def request(self, method, path, payload=None, *, compress=False):
            body = None
            extra = {"Authorization": f"Bearer {self.token}", "Accept": "application/json", **headers}
            if payload is not None:
                body = json.dumps(payload).encode()
                extra["Content-Type"] = "application/json"
                if compress:
                    body = gzip.compress(body, compresslevel=6)
                    extra["Content-Encoding"] = "gzip"
            request = urllib.request.Request(self.base + path, data=body, method=method, headers=extra)
            try:
                with _open(request, self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                raise CoordinatorError(error.code, error.read().decode(errors="replace")) from error

        def _download(self, digest, name, cache):
            cache = Path(cache).expanduser()
            target = cache / f"{digest}-{name}"
            if target.is_file() and sha256_file(target) == digest:
                return target
            cache.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(f"{self.base}/api/v1/checkpoints/{digest}",
                                             headers={"Authorization": f"Bearer {self.token}", **headers})
            fd, temporary = tempfile.mkstemp(dir=cache, prefix=f".{digest}.")
            try:
                sha = hashlib.sha256()
                with os.fdopen(fd, "wb") as stream, _open(request, self.timeout) as response:
                    for block in iter(lambda: response.read(1 << 20), b""):
                        sha.update(block)
                        stream.write(block)
                if sha.hexdigest() != digest:
                    raise OSError(f"artifact {name} hash mismatch (truncated or corrupted transfer)")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return target

    client = AccessStudyClient(base, token, timeout=timeout)
    client.access_headers = headers
    return client


class PoolClient:
    def __init__(self, url=None, token=None, token_file=None, timeout=300.0, access=None):
        self.base = (url or DEFAULT_URL).rstrip("/")
        self.study = access_study_client(self.base, read_token(token, token_file), timeout=timeout, access=access)

    def get(self, path, **query):
        suffix = ("?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})) if query else ""
        return self.study.retrying("GET", path + suffix, patience=120)

    def post(self, path, payload, *, compress=False, patience=120):
        return self.study.retrying("POST", path, payload, compress=compress, patience=patience)

    def register(self, event):
        return self.post("/api/v1/pool/registry", event)

    def has_artifact(self, digest):
        return self.get(f"/api/v1/pool/artifacts/{digest}/exists")["exists"]

    def _put(self, path, data, *, mb_per_second):
        started = time.monotonic()
        request = urllib.request.Request(self.base + path, data=data, method="PUT", headers={
            "Authorization": f"Bearer {self.study.token}", "Content-Length": str(len(data)),
            "Content-Type": "application/octet-stream", **self.study.access_headers})
        with _open(request, 3600) as response:
            reply = json.loads(response.read())
        ahead = len(data) / (mb_per_second * 2 ** 20) - (time.monotonic() - started)
        if ahead > 0:
            time.sleep(ahead)
        return reply

    def upload(self, path, digest=None, *, mb_per_second=40.0, part_bytes=PART_BYTES):
        """Upload a checkpoint in parts (each under Cloudflare's body cap), then have the
        coordinator join them and verify the sha256. Skipped when the coordinator has it."""
        digest = digest or sha256_file(path)
        if self.has_artifact(digest):
            return digest
        size = os.path.getsize(path)
        parts = max(1, -(-size // part_bytes))
        with open(path, "rb") as stream:
            for index in range(parts):
                data = stream.read(part_bytes)
                self._put(f"/api/v1/pool/artifacts/{digest}/parts/{index}", data, mb_per_second=mb_per_second)
        reply = self.post(f"/api/v1/pool/artifacts/{digest}/complete", dict(parts=parts, size=size), patience=600)
        if reply.get("sha256") != digest:
            raise RuntimeError(f"upload of {path} failed: {reply}")
        return digest


def artifact_record(path, digest=None):
    path = Path(path).expanduser().resolve()
    return dict(sha256=digest or sha256_file(path), size=path.stat().st_size, name=path.name, paths=[str(path)])


def register_snapshot(client, path, *, run, entrant_id, era, step, parent=None, recipe=None, name=None,
                      panel_set=None, panel_games=128, panel_priority=60, panel_step_every=None, anchor=None,
                      upload=True, notes=""):
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
        if panel_step_every:
            job["step_every"] = int(panel_step_every)
        client.register(job)
    return record

