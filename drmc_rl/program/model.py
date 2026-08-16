"""Machine-readable authority for the unified Dr. Mario player program.

The registry deliberately separates *what the program permits* from individual
training YAML files.  A training config describes one executable experiment;
``program.yaml`` records why it exists, which gates permit it to run, which
artifacts it consumes, and which product it can support.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ALLOWED_STATUSES = frozenset({"active", "staged", "blocked", "complete", "retired"})


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading a large artifact in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray, Mapping)):
        raise ValueError(f"{name} must be a string or sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} cannot contain an empty item")
    return result


@dataclass(frozen=True, slots=True)
class GateSpec:
    id: str
    status: str
    description: str
    evidence_path: str
    criteria: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, gate_id: str, value: Mapping[str, Any]) -> "GateSpec":
        status = str(value.get("status", "blocked"))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"gate {gate_id!r} has invalid status {status!r}")
        return cls(
            id=str(gate_id),
            status=status,
            description=str(value.get("description", "")).strip(),
            evidence_path=str(value.get("evidence_path", f"runs/program/gates/{gate_id}.json")),
            criteria=_string_tuple(value.get("criteria"), name=f"gate {gate_id}.criteria"),
            depends_on=_string_tuple(value.get("depends_on"), name=f"gate {gate_id}.depends_on"),
        )


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    id: str
    status: str
    stage: int
    purpose: str
    command: tuple[str, ...]
    config: str | None = None
    requires_paths: tuple[str, ...] = ()
    requires_gates: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    resources: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, recipe_id: str, value: Mapping[str, Any]) -> "RecipeSpec":
        status = str(value.get("status", "blocked"))
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"recipe {recipe_id!r} has invalid status {status!r}")
        command = _string_tuple(value.get("command"), name=f"recipe {recipe_id}.command")
        if not command and status in {"active", "staged"}:
            raise ValueError(f"recipe {recipe_id!r} is {status} but has no command")
        config = value.get("config")
        return cls(
            id=str(recipe_id),
            status=status,
            stage=int(value.get("stage", 0)),
            purpose=str(value.get("purpose", "")).strip(),
            command=command,
            config=None if config in (None, "") else str(config),
            requires_paths=_string_tuple(
                value.get("requires_paths"), name=f"recipe {recipe_id}.requires_paths"
            ),
            requires_gates=_string_tuple(
                value.get("requires_gates"), name=f"recipe {recipe_id}.requires_gates"
            ),
            produces=_string_tuple(value.get("produces"), name=f"recipe {recipe_id}.produces"),
            resources=dict(_require_mapping(value.get("resources", {}), name="resources")),
            notes=_string_tuple(value.get("notes"), name=f"recipe {recipe_id}.notes"),
        )

    def resolved_command(
        self,
        repo_root: str | Path,
        substitutions: Mapping[str, str] | None = None,
    ) -> tuple[str, ...]:
        """Resolve explicit placeholders while rejecting unresolved shell magic.

        Commands are argv arrays rather than shell strings, so launch behavior
        is deterministic and paths with spaces are safe.
        """

        root = str(Path(repo_root).resolve())
        values = {"repo": root, **dict(substitutions or {})}
        resolved: list[str] = []
        for token in self.command:
            try:
                rendered = token.format_map(_StrictFormat(values))
            except KeyError as exc:
                raise ValueError(
                    f"recipe {self.id!r} requires substitution {exc.args[0]!r}"
                ) from exc
            if "\n" in rendered or "\x00" in rendered:
                raise ValueError(f"recipe {self.id!r} produced an unsafe command token")
            resolved.append(rendered)
        return tuple(resolved)


class _StrictFormat(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class ProductSpec:
    id: str
    description: str
    competitive_core: str
    decoder: str
    execution_profile: str | None
    requires_gates: tuple[str, ...]
    release_criteria: tuple[str, ...]

    @classmethod
    def from_mapping(cls, product_id: str, value: Mapping[str, Any]) -> "ProductSpec":
        execution = value.get("execution_profile")
        return cls(
            id=str(product_id),
            description=str(value.get("description", "")).strip(),
            competitive_core=str(value.get("competitive_core", "unified")),
            decoder=str(value.get("decoder", "quality")),
            execution_profile=None if execution in (None, "") else str(execution),
            requires_gates=_string_tuple(
                value.get("requires_gates"), name=f"product {product_id}.requires_gates"
            ),
            release_criteria=_string_tuple(
                value.get("release_criteria"), name=f"product {product_id}.release_criteria"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    version: int
    name: str
    authority: Mapping[str, str]
    principles: tuple[str, ...]
    gates: Mapping[str, GateSpec]
    recipes: Mapping[str, RecipeSpec]
    products: Mapping[str, ProductSpec]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "ProgramSpec":
        source = Path(path).resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        root = _require_mapping(payload, name="program")
        gates_raw = _require_mapping(root.get("gates", {}), name="program.gates")
        recipes_raw = _require_mapping(root.get("recipes", {}), name="program.recipes")
        products_raw = _require_mapping(root.get("products", {}), name="program.products")
        spec = cls(
            version=int(root.get("version", 1)),
            name=str(root.get("name", "drmc unified player program")),
            authority={
                str(key): str(value)
                for key, value in _require_mapping(
                    root.get("authority", {}), name="program.authority"
                ).items()
            },
            principles=_string_tuple(root.get("principles"), name="program.principles"),
            gates={
                str(key): GateSpec.from_mapping(str(key), _require_mapping(value, name=str(key)))
                for key, value in gates_raw.items()
            },
            recipes={
                str(key): RecipeSpec.from_mapping(
                    str(key), _require_mapping(value, name=str(key))
                )
                for key, value in recipes_raw.items()
            },
            products={
                str(key): ProductSpec.from_mapping(
                    str(key), _require_mapping(value, name=str(key))
                )
                for key, value in products_raw.items()
            },
            source_path=source,
        )
        spec.validate()
        return spec

    @property
    def repo_root(self) -> Path:
        # Installed package: drmc_rl/program/program.yaml -> repository root.
        return self.source_path.parents[2]

    def validate(self, *, check_paths: bool = False) -> None:
        if self.version < 1:
            raise ValueError("program version must be positive")
        for gate in self.gates.values():
            missing = sorted(set(gate.depends_on) - set(self.gates))
            if missing:
                raise ValueError(f"gate {gate.id!r} refers to unknown gates: {missing}")
        for recipe in self.recipes.values():
            missing = sorted(set(recipe.requires_gates) - set(self.gates))
            if missing:
                raise ValueError(f"recipe {recipe.id!r} refers to unknown gates: {missing}")
            if recipe.config and recipe.config not in recipe.requires_paths:
                raise ValueError(
                    f"recipe {recipe.id!r} config must also appear in requires_paths"
                )
            if check_paths and recipe.status in {"active", "staged"}:
                absent = [
                    path
                    for path in recipe.requires_paths
                    if not (self.repo_root / path).exists()
                    and not any(ch in path for ch in "{}*")
                ]
                if absent:
                    raise ValueError(f"recipe {recipe.id!r} is missing paths: {absent}")
        for product in self.products.values():
            missing = sorted(set(product.requires_gates) - set(self.gates))
            if missing:
                raise ValueError(f"product {product.id!r} refers to unknown gates: {missing}")
        self._validate_gate_dag()

    def _validate_gate_dag(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(gate_id: str) -> None:
            if gate_id in visited:
                return
            if gate_id in visiting:
                raise ValueError(f"gate dependency cycle includes {gate_id!r}")
            visiting.add(gate_id)
            for dependency in self.gates[gate_id].depends_on:
                visit(dependency)
            visiting.remove(gate_id)
            visited.add(gate_id)

        for gate_id in self.gates:
            visit(gate_id)

    def gate_evidence(self, gate_id: str) -> "GateEvidence | None":
        gate = self.gates[gate_id]
        path = self.repo_root / gate.evidence_path
        if not path.is_file():
            return None
        return GateEvidence.from_path(path)

    def gate_is_open(self, gate_id: str) -> bool:
        gate = self.gates[gate_id]
        if gate.status == "complete":
            return True
        evidence = self.gate_evidence(gate_id)
        if evidence is None or not evidence.passed:
            return False
        return all(self.gate_is_open(dep) for dep in gate.depends_on)

    def recipe_blockers(self, recipe_id: str) -> list[str]:
        recipe = self.recipes[recipe_id]
        blockers = [gate for gate in recipe.requires_gates if not self.gate_is_open(gate)]
        for path in recipe.requires_paths:
            # Runtime placeholders are satisfied by ``tools.program launch --set``
            # and are intentionally not treated as repository paths here.
            if "{" in path and path != "{repo}":
                continue
            rendered = path.replace("{repo}", str(self.repo_root))
            if "*" in rendered:
                continue
            candidate = Path(rendered)
            if not candidate.is_absolute():
                candidate = self.repo_root / candidate
            if not candidate.exists():
                blockers.append(f"missing:{path}")
        return blockers

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "authority": dict(self.authority),
            "principles": list(self.principles),
            "gates": {key: asdict(value) for key, value in self.gates.items()},
            "recipes": {key: asdict(value) for key, value in self.recipes.items()},
            "products": {key: asdict(value) for key, value in self.products.items()},
        }


@dataclass(frozen=True, slots=True)
class GateEvidence:
    gate_id: str
    passed: bool
    recorded_at: str
    commit: str | None
    metrics: Mapping[str, Any]
    artifacts: tuple[str, ...]
    notes: tuple[str, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "GateEvidence":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            gate_id=str(payload["gate_id"]),
            passed=bool(payload["passed"]),
            recorded_at=str(payload["recorded_at"]),
            commit=None if payload.get("commit") is None else str(payload["commit"]),
            metrics=dict(payload.get("metrics", {})),
            artifacts=_string_tuple(payload.get("artifacts"), name="artifacts"),
            notes=_string_tuple(payload.get("notes"), name="notes"),
        )

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema: str
    created_at: str
    artifact: str
    artifact_sha256: str
    artifact_size: int
    repository_commit: str | None
    dirty_repository: bool | None
    native_revision: str | None
    config: str | None
    config_sha256: str | None
    observation_schema: str | None
    execution_profile: str | None
    search: Mapping[str, Any]
    corpus_release: str | None
    parents: tuple[str, ...]
    metadata: Mapping[str, Any]

    @classmethod
    def build(
        cls,
        artifact: str | Path,
        *,
        repo_root: str | Path,
        config: str | Path | None = None,
        observation_schema: str | None = None,
        execution_profile: str | None = None,
        search: Mapping[str, Any] | None = None,
        corpus_release: str | None = None,
        parents: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactManifest":
        artifact_path = Path(artifact).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        root = Path(repo_root).resolve()
        commit, dirty = _git_identity(root)
        native_revision = _git_submodule_revision(root, "vendor/drmario_native")
        config_path = None if config is None else Path(config).resolve()
        if config_path is not None and not config_path.is_file():
            raise FileNotFoundError(config_path)
        return cls(
            schema="drmc-artifact-manifest-v1",
            created_at=datetime.now(UTC).isoformat(),
            artifact=str(artifact_path),
            artifact_sha256=sha256_file(artifact_path),
            artifact_size=artifact_path.stat().st_size,
            repository_commit=commit,
            dirty_repository=dirty,
            native_revision=native_revision,
            config=None if config_path is None else str(config_path),
            config_sha256=None if config_path is None else sha256_file(config_path),
            observation_schema=observation_schema,
            execution_profile=execution_profile,
            search=dict(search or {}),
            corpus_release=corpus_release,
            parents=tuple(str(item) for item in parents),
            metadata=dict(metadata or {}),
        )

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_identity(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        return commit or None, bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA"), None


def _git_submodule_revision(root: Path, path: str) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "ls-tree", "HEAD", path],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    fields = output.split()
    return fields[2] if len(fields) >= 3 else None


def format_command(command: Iterable[str]) -> str:
    return shlex.join(tuple(command))


__all__ = [
    "ALLOWED_STATUSES",
    "ArtifactManifest",
    "GateEvidence",
    "GateSpec",
    "ProductSpec",
    "ProgramSpec",
    "RecipeSpec",
    "format_command",
    "sha256_file",
]
