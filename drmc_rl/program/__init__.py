"""Authoritative program registry and launch-time guardrails."""

from .model import (
    ALLOWED_STATUSES,
    ArtifactManifest,
    GateEvidence,
    GateSpec,
    ProductSpec,
    ProgramSpec,
    RecipeSpec,
    sha256_file,
)

__all__ = [
    "ALLOWED_STATUSES",
    "ArtifactManifest",
    "GateEvidence",
    "GateSpec",
    "ProductSpec",
    "ProgramSpec",
    "RecipeSpec",
    "sha256_file",
]
