"""Host-neutral HTTP client and wire helpers for distributed arena workers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class ArenaRemoteClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        checkpoint_cache: str | Path,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.checkpoint_cache = Path(checkpoint_cache).expanduser()
        self.timeout = float(timeout)
        if not self.token:
            raise ValueError("arena worker token must not be empty")

    def _request(
        self, method: str, path: str, payload: Any | None = None
    ) -> tuple[int, bytes, dict[str, str]]:
        body = None if payload is None else canonical_json(payload)
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            data = error.read()
            if error.code == 204:
                return error.code, data, dict(error.headers)
            detail = data.decode(errors="replace")
            raise RuntimeError(f"arena coordinator HTTP {error.code}: {detail}") from error

    def capabilities(self) -> dict[str, Any]:
        _status, body, _headers = self._request("GET", "/api/v1/capabilities")
        payload = json.loads(body)
        if int(payload["protocol_version"]) != PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported arena protocol {payload['protocol_version']} "
                f"(client={PROTOCOL_VERSION})"
            )
        return payload

    def lease(self, worker: dict[str, Any]) -> dict[str, Any] | None:
        status, body, _headers = self._request("POST", "/api/v1/leases", worker)
        if status == 204 or not body:
            return None
        payload = json.loads(body)
        if int(payload["protocol_version"]) != PROTOCOL_VERSION:
            raise RuntimeError("coordinator returned an incompatible lease")
        return payload

    def submit(self, lease_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        _status, body, _headers = self._request(
            "POST", f"/api/v1/leases/{urllib.parse.quote(lease_id)}/results", payload
        )
        return json.loads(body)

    def renew(self, lease_id: str, claim_token: str) -> dict[str, Any]:
        _status, body, _headers = self._request(
            "POST", f"/api/v1/leases/{urllib.parse.quote(lease_id)}/renew",
            {"protocol_version": PROTOCOL_VERSION, "claim_token": claim_token},
        )
        return json.loads(body)

    def materialize_checkpoint(self, agent: dict[str, Any]) -> Path:
        digest = str(agent["checkpoint_sha256"])
        suffix = "".join(Path(str(agent.get("checkpoint_name", "checkpoint.pt.gz"))).suffixes)
        target = self.checkpoint_cache / f"{digest}{suffix or '.pt.gz'}"
        if target.is_file() and _sha256_file(target) == digest:
            return target
        self.checkpoint_cache.mkdir(parents=True, exist_ok=True)
        encoded = urllib.parse.quote(str(agent["id"]), safe="")
        _status, body, _headers = self._request("GET", f"/api/v1/checkpoints/{encoded}")
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError(f"checkpoint hash mismatch for {agent['id']}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=self.checkpoint_cache
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ArenaRemoteClient",
    "PROTOCOL_VERSION",
    "canonical_json",
    "content_sha256",
]
