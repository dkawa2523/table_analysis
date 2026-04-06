from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable as IterableABC
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .common.config_utils import cfg_value as _cfg_value
from .common.repo_utils import resolve_repo_root_fallback as _resolve_repo_root
from .common.schema_version import normalize_schema_version as _normalize_schema_version
from .platform_adapter_common import PlatformAdapterError, _RECOVERABLE_ERRORS, is_clearml_enabled

ClearMLScriptSpec = make_dataclass(
    "ClearMLScriptSpec",
    [
        ("repository", str | None),
        ("branch", str | None),
        ("entry_point", str | None),
        ("working_dir", str | None),
        ("version_policy", str),
        ("version_num", str | None),
    ],
    frozen=True,
)


def _run_git_command(cmd: IterableABC[str]) -> str | None:
    try:
        proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except _RECOVERABLE_ERRORS:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _normalize_git_remote_url(value: str) -> str:
    text = value.strip()
    if text.startswith("git@") and ":" in text:
        (host_part, path) = text.split(":", 1)
        host = host_part.split("@", 1)[-1]
        text = f"https://{host}/{path}"
    elif text.startswith("ssh://") and "@" in text:
        rest = text[len("ssh://") :]
        (host_part, _, path) = rest.partition("/")
        host = host_part.split("@", 1)[-1]
        text = f"https://{host}/{path}"
    if text.endswith(".git"):
        text = text[:-4]
    return text


def detect_git_repository_url(repo_root: Path) -> str | None:
    value = _run_git_command(["git", "-C", str(repo_root), "remote", "get-url", "origin"])
    if not value:
        return None
    return _normalize_git_remote_url(value)


def detect_git_branch(repo_root: Path) -> str | None:
    value = _run_git_command(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    if not value or value == "HEAD":
        return None
    return value


def _normalize_code_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_clearml_repository(value: Any) -> str | None:
    text = _normalize_code_ref(value)
    if not text:
        return None
    return _normalize_git_remote_url(text).rstrip("/")


def normalize_clearml_branch(value: Any) -> str | None:
    return _normalize_code_ref(value)


def _split_entry_point_command(entry_point: str) -> tuple[str, str]:
    text = str(entry_point).strip()
    if not text:
        return ("", "")
    match = _ENTRYPOINT_OVERRIDE_RE.search(text)
    if not match:
        return (text, "")
    idx = match.start()
    return (text[:idx].strip(), text[idx:].strip())


def normalize_clearml_entry_point(value: Any) -> str | None:
    text = _normalize_code_ref(value)
    if not text:
        return None
    parts = text.split()
    if parts and parts[0] in {"python", "python3"}:
        parts = parts[1:]
    normalized = " ".join(parts).strip()
    if not normalized:
        return None
    (command, _) = _split_entry_point_command(normalized)
    return command or None


def normalize_clearml_version_num(value: Any) -> str | None:
    return _normalize_code_ref(value)


def resolve_clearml_code_reference(cfg: Any) -> tuple[str | None, str | None]:
    repo_value = _normalize_code_ref(_cfg_value(cfg, "run.clearml.code_ref.repository"))
    branch_value = _normalize_code_ref(_cfg_value(cfg, "run.clearml.code_ref.branch"))
    repo_root = _resolve_repo_root()
    if repo_value and repo_value.lower() == "auto":
        repo_value = detect_git_repository_url(repo_root)
    if branch_value and branch_value.lower() == "auto":
        branch_value = detect_git_branch(repo_root)
    return (repo_value, branch_value)


def _resolve_clearml_entrypoint_override(cfg: Any) -> str | None:
    execution_value = _normalize_code_ref(_cfg_value(cfg, "run.clearml.execution"))
    if execution_value is None:
        return None
    execution = execution_value.lower()
    if execution in {"agent", "clone", "pipeline_controller"}:
        return "tools/clearml_entrypoint.py"
    return None


_ENTRYPOINT_OVERRIDE_RE = re.compile(r"\s[+~]?[^\s=]+=")


def _canonicalize_pipeline_entrypoint(
    cfg: Any,
    entry_point: str | None,
    fallback: str | None,
    task_name_override: str | None = None,
    canonicalize_pipeline: bool = True,
) -> str | None:
    if not canonicalize_pipeline:
        return entry_point
    task_name = _normalize_code_ref(task_name_override) or _normalize_code_ref(_cfg_value(cfg, "task.name"))
    if task_name != "pipeline":
        return entry_point
    return entry_point or fallback or "tools/clearml_entrypoint.py"


def _resolve_clearml_entrypoint(
    cfg: Any,
    current_entry_point: Any,
    entry_point_override: str | None,
    task_name_override: str | None = None,
    canonicalize_pipeline: bool = True,
) -> str | None:
    current_text = _normalize_code_ref(current_entry_point)
    base = current_text
    if entry_point_override is not None:
        override_text = str(entry_point_override).strip()
        base = override_text or current_text
    return _canonicalize_pipeline_entrypoint(
        cfg,
        base,
        entry_point_override or current_text,
        task_name_override=task_name_override,
        canonicalize_pipeline=canonicalize_pipeline,
    )


def _resolve_clearml_code_ref_mode(cfg: Any, *, override: str | None = None) -> str:
    text = _normalize_code_ref(override) or _normalize_code_ref(_cfg_value(cfg, "run.clearml.code_ref.mode"))
    if not text:
        return "branch_head"
    lowered = text.lower()
    if lowered in {"branch_head", "branch", "head"}:
        return "branch_head"
    if lowered in {"pin_commit", "commit", "pinned"}:
        return "pin_commit"
    if lowered in {"none", "off", "disabled"}:
        return "none"
    return "branch_head"


def _resolve_clearml_version_num(cfg: Any, *, version_mode_override: str | None = None) -> tuple[str, str | None]:
    mode = _resolve_clearml_code_ref_mode(cfg, override=version_mode_override)
    if mode == "pin_commit":
        commit_override = _normalize_code_ref(_cfg_value(cfg, "run.clearml.code_ref.commit"))
        if commit_override and commit_override.lower() != "auto":
            return (mode, commit_override)
        repo_root = _resolve_repo_root()
        commit = _run_git_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
        if not commit:
            raise PlatformAdapterError("Failed to resolve git commit for pin_commit.")
        return (mode, commit)
    if mode == "none":
        return (mode, None)
    return (mode, "")


def resolve_clearml_script_spec(
    cfg: Any,
    *,
    current_entry_point: Any | None = None,
    repo_override: str | None = None,
    branch_override: str | None = None,
    entry_point_override: str | None = None,
    working_dir_override: str | None = None,
    version_mode_override: str | None = None,
    task_name_override: str | None = None,
    canonicalize_pipeline: bool = True,
) -> ClearMLScriptSpec:
    (repo_value, branch_value) = resolve_clearml_code_reference(cfg)
    if repo_override is not None:
        repo_value = repo_override
    if branch_override is not None:
        branch_value = branch_override
    entry_override = entry_point_override if entry_point_override is not None else _resolve_clearml_entrypoint_override(cfg)
    entry_point = _resolve_clearml_entrypoint(
        cfg,
        current_entry_point,
        entry_override,
        task_name_override=task_name_override,
        canonicalize_pipeline=canonicalize_pipeline,
    )
    working_dir = _normalize_code_ref(working_dir_override) or _normalize_code_ref(_cfg_value(cfg, "run.clearml.working_dir"))
    (version_policy, version_num) = _resolve_clearml_version_num(cfg, version_mode_override=version_mode_override)
    return ClearMLScriptSpec(
        repository=repo_value,
        branch=branch_value,
        entry_point=entry_point,
        working_dir=working_dir,
        version_policy=version_policy,
        version_num=version_num,
    )


def _commit_matches(expected: str | None, actual: str | None) -> bool:
    if not expected:
        return not actual
    if not actual:
        return False
    expected_norm = expected.lower()
    actual_norm = actual.lower()
    if expected_norm == actual_norm:
        return True
    if expected_norm.startswith(actual_norm) or actual_norm.startswith(expected_norm):
        return min(len(expected_norm), len(actual_norm)) >= 7
    return False


def clearml_script_mismatches(spec: ClearMLScriptSpec, script: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_repo = normalize_clearml_repository(spec.repository)
    if expected_repo:
        actual_repo = normalize_clearml_repository(script.get("repository"))
        if actual_repo != expected_repo:
            errors.append(f"repository mismatch: {actual_repo or 'none'}")
    expected_branch = normalize_clearml_branch(spec.branch)
    if expected_branch:
        actual_branch = normalize_clearml_branch(script.get("branch"))
        if actual_branch != expected_branch:
            errors.append(f"branch mismatch: {actual_branch or 'none'}")
    expected_entry = normalize_clearml_entry_point(spec.entry_point)
    if expected_entry:
        actual_entry = normalize_clearml_entry_point(script.get("entry_point"))
        if actual_entry != expected_entry:
            errors.append(f"entry_point mismatch: {actual_entry or 'none'}")
    expected_working = _normalize_code_ref(spec.working_dir)
    if expected_working:
        actual_working = _normalize_code_ref(script.get("working_dir"))
        if actual_working != expected_working:
            errors.append(f"working_dir mismatch: {actual_working or 'none'}")
    actual_version = normalize_clearml_version_num(script.get("version_num"))
    if spec.version_policy == "branch_head":
        if actual_version:
            errors.append(f"version_num mismatch: {actual_version}")
    elif spec.version_policy == "none":
        return errors
    elif spec.version_policy == "pin_commit":
        expected_version = normalize_clearml_version_num(spec.version_num)
        if not _commit_matches(expected_version, actual_version):
            errors.append(f"version_num mismatch: {actual_version or 'none'}")
    return errors


def _normalize_clearml_user_properties(properties: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for (key, value) in properties.items():
        if isinstance(value, Mapping) and "value" in value:
            value = value.get("value")
        normalized[str(key)] = "" if value is None else str(value)
    return normalized


def _patch_platform_clearml(platform_clearml: Any) -> None:
    if platform_clearml is None:
        return
    if getattr(platform_clearml, "_ta_user_properties_patch", False):
        return

    def _safe_set_user_properties(task: Any, properties: Mapping[str, Any] | None) -> None:
        if not properties:
            return
        setter = getattr(task, "set_user_properties", None)
        if not callable(setter):
            return
        normalized = _normalize_clearml_user_properties(dict(properties))
        try:
            setter(**normalized)
            return
        except TypeError:
            pass
        try:
            setter(*normalized.items())
            return
        except TypeError:
            setter(normalized)

    platform_clearml.set_user_properties = _safe_set_user_properties
    platform_clearml._ta_user_properties_patch = True


def _load_clearml_module(clearml_enabled: bool):
    try:
        from ml_platform.integrations import clearml as platform_clearml
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError(
                "ml_platform.integrations.clearml is required for ClearML runs. Install/update ml_platform."
            ) from exc
        return None
    _patch_platform_clearml(platform_clearml)
    return platform_clearml


load_clearml_module = _load_clearml_module


def _load_clearml_dataset(clearml_enabled: bool):
    try:
        from clearml import Dataset as ClearMLDataset
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError(
                "clearml.Dataset is required for ClearML dataset registration. Install/update clearml."
            ) from exc
        return None
    return ClearMLDataset


load_clearml_dataset = _load_clearml_dataset


def _resolve_version_props(cfg: Any, *, clearml_enabled: bool) -> dict[str, str]:
    try:
        from ml_platform.versioning import get_code_version, get_platform_version, get_schema_version
    except _RECOVERABLE_ERRORS as exc:
        if clearml_enabled:
            raise PlatformAdapterError(
                "ml_platform.versioning is required for ClearML runs. Install/update ml_platform."
            ) from exc
        return {"code_version": "unknown", "platform_version": "unknown", "schema_version": "unknown"}
    schema_version = _cfg_value(cfg, "run.schema_version") or get_schema_version(cfg, default="unknown")
    return {
        "code_version": get_code_version(repo_root=Path.cwd()),
        "platform_version": get_platform_version(),
        "schema_version": _normalize_schema_version(schema_version, default="unknown"),
    }


def resolve_version_props(cfg: Any, *, clearml_enabled: Optional[bool] = None) -> dict[str, str]:
    enabled = is_clearml_enabled(cfg) if clearml_enabled is None else clearml_enabled
    return _resolve_version_props(cfg, clearml_enabled=bool(enabled))


def _load_clearml_pipeline_utils(clearml_enabled: bool):
    try:
        from ml_platform.integrations.clearml import pipeline_utils as platform_pipeline_utils
    except ImportError as exc:
        if clearml_enabled:
            raise PlatformAdapterError(
                "ml_platform.integrations.clearml.pipeline_utils not available for ClearML runs."
            ) from exc
        return None
    return platform_pipeline_utils


load_clearml_pipeline_utils = _load_clearml_pipeline_utils


__all__ = [
    "ClearMLScriptSpec",
    "clearml_script_mismatches",
    "detect_git_branch",
    "detect_git_repository_url",
    "is_clearml_enabled",
    "load_clearml_dataset",
    "load_clearml_module",
    "load_clearml_pipeline_utils",
    "normalize_clearml_branch",
    "normalize_clearml_entry_point",
    "normalize_clearml_repository",
    "normalize_clearml_version_num",
    "resolve_clearml_code_reference",
    "resolve_clearml_script_spec",
    "resolve_version_props",
]
