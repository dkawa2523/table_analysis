from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ReferenceInfo:
    name: str
    identifier: str | None = None
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedInputs:
    dataset_path: Path | None = None
    input_path: Path | None = None
    references: tuple[ReferenceInfo, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSettings:
    task_name: str
    stage: str
    clearml_enabled: bool
    mode: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactBundle:
    primary_path: Path | None = None
    paths: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    out: Mapping[str, Any]
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    references: tuple[ReferenceInfo, ...] = field(default_factory=tuple)
    artifacts: Mapping[str, ArtifactBundle] = field(default_factory=dict)
