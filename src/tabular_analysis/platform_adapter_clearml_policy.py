from __future__ import annotations

from collections.abc import Iterable as IterableABC
import json
from typing import Any, Iterable, Mapping, Optional

from .common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value
from .common.schema_version import build_schema_tag as _build_schema_tag
from .platform_adapter_common import (
    PlatformAdapterError,
    _RECOVERABLE_ERRORS,
    _dedupe_tags,
    _normalize_requirement_lines,
    _resolve_clearml_task,
)
from .platform_adapter_clearml_env import clearml_script_mismatches, resolve_clearml_script_spec, resolve_version_props
from .platform_adapter_task_ops import _apply_task_args, _set_task_script_payload
from .platform_adapter_task_query import _task_script, _task_tags


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def _parse_json_list(text: str) -> list[Any] | None:
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        parsed = json.loads(text)
    except _RECOVERABLE_ERRORS:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _split_bracket_list(text: str) -> list[str] | None:
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    for item in inner.split(","):
        cleaned = item.strip().strip("'\"").strip()
        if cleaned:
            items.append(cleaned)
    return items


def _coerce_hydra_list_value(value: Any) -> list[str]:
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _parse_json_list(text)
        if parsed is None:
            parsed = _split_bracket_list(text)
        items = parsed if parsed is not None else [text]
    elif isinstance(value, Mapping):
        return []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, IterableABC):
        items = list(value)
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _apply_clearml_task_script_override(target: Any, cfg: Any) -> bool:
    task = _resolve_clearml_task(target)
    current = _task_script(task)
    current_repo = current.get("repository")
    current_branch = current.get("branch")
    current_entry_point = current.get("entry_point")
    spec = resolve_clearml_script_spec(cfg, current_entry_point=current_entry_point)
    if not clearml_script_mismatches(spec, current):
        return False
    setter = getattr(task, "set_script", None)
    if not callable(setter):
        raise PlatformAdapterError("ClearML Task.set_script is not available.")
    payload: dict[str, Any] = {}
    repo_to_set = spec.repository if spec.repository is not None else current_repo
    branch_to_set = spec.branch if spec.branch is not None else current_branch
    if repo_to_set is not None:
        payload["repository"] = str(repo_to_set)
    if branch_to_set is not None:
        payload["branch"] = str(branch_to_set)
    entry_point = spec.entry_point if spec.entry_point is not None else current_entry_point
    if entry_point is not None:
        payload["entry_point"] = str(entry_point)
    working_dir = spec.working_dir if spec.working_dir is not None else current.get("working_dir")
    if working_dir is not None:
        payload["working_dir"] = working_dir
    if spec.version_num is not None:
        payload["version_num"] = spec.version_num
    if not payload:
        return False
    _set_task_script_payload(task, payload)
    return True


def _apply_clearml_pipeline_args(target: Any, cfg: Any) -> bool:
    task = _resolve_clearml_task(target)
    preprocess_variants = _coerce_hydra_list_value(_cfg_value(cfg, "pipeline.grid.preprocess_variants"))
    model_variants = _coerce_hydra_list_value(_cfg_value(cfg, "pipeline.grid.model_variants"))
    if not preprocess_variants and (not model_variants):
        return False
    args: dict[str, Any] = {}
    if preprocess_variants:
        args["pipeline.grid.preprocess_variants"] = hydra_list(preprocess_variants)
    if model_variants:
        args["pipeline.grid.model_variants"] = hydra_list(model_variants)
    return _apply_task_args(task, args)


def _apply_clearml_task_requirements(task: Any, requirements: Iterable[str]) -> bool:
    normalized = _normalize_requirement_lines(requirements)
    if not normalized:
        return False
    setter = getattr(task, "set_packages", None)
    if not callable(setter):
        raise PlatformAdapterError("ClearML Task.set_packages is not available.")
    try:
        setter(normalized)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to set task requirements via ClearML: {exc}") from exc
    return True


def _resolve_clearml_pipeline_requirements(cfg: Any) -> list[str]:
    _ = cfg
    return ["clearml>=1.15.0", "uv>=0.5.0"]


def build_clearml_properties(
    cfg: Any,
    *,
    stage: str,
    task_name: str,
    extra: Optional[Mapping[str, Any]],
    clearml_enabled: bool,
) -> dict[str, Any]:
    usecase_id = _cfg_value(cfg, "run.usecase_id") or _cfg_value(cfg, "usecase_id") or "unknown"
    process = _cfg_value(cfg, "task.name") or task_name or stage or "unknown"
    versions = resolve_version_props(cfg, clearml_enabled=clearml_enabled)
    grid_run_id = _cfg_value(cfg, "run.grid_run_id")
    retrain_run_id = _cfg_value(cfg, "run.retrain_run_id")
    base: dict[str, Any] = {
        "usecase_id": usecase_id,
        "process": process,
        "schema_version": versions.get("schema_version", "unknown"),
        "code_version": versions.get("code_version", "unknown"),
        "platform_version": versions.get("platform_version", "unknown"),
        "grid_run_id": grid_run_id,
    }
    if retrain_run_id:
        base["retrain_run_id"] = retrain_run_id
    merged = dict(base)
    if extra:
        merged.update(dict(extra))
    for (key, value) in base.items():
        merged.setdefault(key, value)
    return merged


def build_clearml_tags(
    cfg: Any,
    *,
    process: str,
    schema_version: str,
    grid_run_id: Any,
    retrain_run_id: Any,
    extra_tags: Optional[Iterable[str]],
    tags: Optional[Iterable[str]],
) -> list[str]:
    usecase_id = _cfg_value(cfg, "run.usecase_id") or _cfg_value(cfg, "usecase_id") or "unknown"
    base = [f"usecase:{usecase_id}", f"process:{process}", _build_schema_tag(schema_version)]
    if grid_run_id:
        base.append(f"grid:{grid_run_id}")
    if retrain_run_id:
        base.append(f"retrain:{retrain_run_id}")
    return _dedupe_tags([*base, *(extra_tags or []), *(tags or [])])


def _ensure_clearml_names(cfg: Any, *, project_name: str, task_name: str, clearml_enabled: bool) -> None:
    if _cfg_value(cfg, "run.clearml.project_name") != project_name:
        _set_cfg_value(cfg, "run.clearml.project_name", project_name)
    if _cfg_value(cfg, "run.clearml.task_name") != task_name:
        _set_cfg_value(cfg, "run.clearml.task_name", task_name)
    if clearml_enabled:
        if _cfg_value(cfg, "run.clearml.project_name") != project_name:
            raise PlatformAdapterError("Failed to set run.clearml.project_name for ClearML init.")
        if _cfg_value(cfg, "run.clearml.task_name") != task_name:
            raise PlatformAdapterError("Failed to set run.clearml.task_name for ClearML init.")


def clearml_task_type_controller() -> str:
    return "controller"


def _apply_clearml_system_tags(task: Any, system_tags: Iterable[str] | None) -> None:
    if not system_tags:
        return
    getter = getattr(task, "get_system_tags", None)
    setter = getattr(task, "set_system_tags", None)
    if not callable(getter) or not callable(setter):
        raise PlatformAdapterError("ClearML Task.set_system_tags is not available.")
    try:
        current = getter() or []
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to read ClearML system tags: {exc}") from exc
    merged = _dedupe_tags([*current, *system_tags])
    try:
        setter(merged)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to set ClearML system tags: {exc}") from exc


def _load_clearml_project_record(project_name: str | None) -> tuple[str | None, Mapping[str, Any] | None]:
    if not project_name:
        return (None, None)
    try:
        from clearml.backend_api.session.client import APIClient
    except _RECOVERABLE_ERRORS:
        APIClient = None
    if APIClient is not None:
        try:
            client = APIClient()
            projects = client.projects.get_all(name=str(project_name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f"ClearML project lookup failed: {exc}") from exc
        for candidate in list(projects or []):
            name = getattr(candidate, "name", None)
            if str(name or "") == str(project_name):
                project_id = getattr(candidate, "id", None)
                return (str(project_id) if project_id else None, candidate)
        if projects:
            candidate = projects[0]
            project_id = getattr(candidate, "id", None)
            return (str(project_id) if project_id else None, candidate)
    try:
        from clearml.backend_api.session import Session
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"ClearML Session is not available: {exc}") from exc
    try:
        session = Session()
        response = session.send_request(
            service="projects",
            action="get_all",
            json={"name": project_name, "search_hidden": True, "size": 10},
        )
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"ClearML project lookup failed: {exc}") from exc
    if not getattr(response, "ok", False):
        raise PlatformAdapterError(f"ClearML project lookup returned non-ok response for {project_name!r}.")
    try:
        payload = response.json() or {}
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to parse ClearML project lookup payload: {exc}") from exc
    projects = []
    if isinstance(payload, Mapping) and isinstance(payload.get("projects"), list):
        projects = [proj for proj in payload.get("projects", []) if isinstance(proj, Mapping)]
    elif isinstance(payload, Mapping) and isinstance(payload.get("data"), Mapping):
        candidates = payload.get("data", {}).get("projects")
        if isinstance(candidates, list):
            projects = [proj for proj in candidates if isinstance(proj, Mapping)]
    for candidate in projects:
        name = candidate.get("name") or candidate.get("full_name") or candidate.get("path")
        if name == project_name:
            project_id = candidate.get("id") or candidate.get("project") or candidate.get("project_id")
            return (str(project_id) if project_id else None, candidate)
    if projects:
        candidate = projects[0]
        project_id = candidate.get("id") or candidate.get("project") or candidate.get("project_id")
        return (str(project_id) if project_id else None, candidate)
    return (None, None)


def _get_clearml_project_system_tags(project_name: str | None) -> list[str]:
    (_, project) = _load_clearml_project_record(project_name)
    if project is None:
        return []
    system_tags = getattr(project, "system_tags", None)
    if system_tags is None and isinstance(project, Mapping):
        system_tags = project.get("system_tags")
    if isinstance(system_tags, list):
        return _dedupe_tags(system_tags)
    try:
        return _dedupe_tags(list(system_tags or []))
    except _RECOVERABLE_ERRORS:
        return []


def _create_clearml_project_record(project_name: str) -> str | None:
    if not project_name:
        return None
    try:
        from clearml.backend_api.session import Session

        session = Session()
        response = session.send_request(service="projects", action="create", json={"name": str(project_name)})
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"ClearML project create failed: {exc}") from exc
    if not getattr(response, "ok", False):
        raise PlatformAdapterError(f"ClearML project create returned non-ok response for {project_name!r}.")
    (project_id, _) = _load_clearml_project_record(project_name)
    return project_id


def _ensure_clearml_project_system_tags(
    project_name: str | None,
    add_tags: Iterable[str] | None = None,
    *,
    remove_tags: Iterable[str] | None = None,
) -> None:
    if not project_name:
        return
    add_list = _dedupe_tags(add_tags or [])
    remove_set = {tag for tag in _dedupe_tags(remove_tags or []) if tag}
    if not add_list and (not remove_set):
        return
    (project_id, _) = _load_clearml_project_record(project_name)
    if not project_id:
        project_id = _create_clearml_project_record(str(project_name))
    if not project_id:
        raise PlatformAdapterError(f"ClearML project not found: {project_name!r}")
    system_tags = _get_clearml_project_system_tags(project_name)
    merged = _dedupe_tags([*system_tags, *add_list])
    if remove_set:
        merged = [tag for tag in merged if tag not in remove_set]
    if merged == system_tags:
        return
    try:
        from clearml.backend_api.session import Session

        session = Session()
        update = session.send_request(
            service="projects",
            action="update",
            json={"project": project_id, "system_tags": merged},
        )
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"ClearML project update failed: {exc}") from exc
    if not getattr(update, "ok", False):
        raise PlatformAdapterError(f"ClearML project update returned non-ok response for {project_name!r}.")
    if _get_clearml_project_system_tags(project_name) != merged:
        raise PlatformAdapterError(f"ClearML project system tags did not persist for {project_name!r}.")


def _apply_clearml_tags(task: Any, tags: Iterable[str] | None) -> None:
    tag_list = _dedupe_tags(tags or [])
    if not tag_list:
        return
    existing = _task_tags(task)
    merged = _dedupe_tags([*existing, *tag_list])
    setter = getattr(task, "set_tags", None)
    if callable(setter):
        try:
            setter(merged)
            return
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f"Failed to set ClearML tags: {exc}") from exc
    adder = getattr(task, "add_tags", None)
    if not callable(adder):
        raise PlatformAdapterError("ClearML Task.add_tags is not available.")
    missing = [tag for tag in tag_list if tag not in existing]
    if not missing:
        return
    try:
        adder(missing)
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to add ClearML tags: {exc}") from exc


def _apply_clearml_task_type(task: Any, task_type: str | None) -> None:
    if not task_type:
        return
    setter = getattr(task, "set_task_type", None)
    if not callable(setter):
        return
    try:
        from clearml import Task as ClearMLTask

        normalized = task_type
        if isinstance(task_type, str) and task_type.lower() == "controller":
            normalized = ClearMLTask.TaskTypes.controller
        setter(normalized)
    except _RECOVERABLE_ERRORS:
        try:
            setter(task_type)
        except _RECOVERABLE_ERRORS:
            return


apply_clearml_pipeline_args = _apply_clearml_pipeline_args
apply_clearml_system_tags = _apply_clearml_system_tags
apply_clearml_tags = _apply_clearml_tags
apply_clearml_task_requirements = _apply_clearml_task_requirements
apply_clearml_task_script_override = _apply_clearml_task_script_override
apply_clearml_task_type = _apply_clearml_task_type
ensure_clearml_names = _ensure_clearml_names
ensure_clearml_project_system_tags = _ensure_clearml_project_system_tags
get_clearml_project_system_tags = _get_clearml_project_system_tags
resolve_clearml_pipeline_requirements = _resolve_clearml_pipeline_requirements


__all__ = [
    "apply_clearml_pipeline_args",
    "apply_clearml_system_tags",
    "apply_clearml_tags",
    "apply_clearml_task_requirements",
    "apply_clearml_task_script_override",
    "apply_clearml_task_type",
    "build_clearml_properties",
    "build_clearml_tags",
    "clearml_task_type_controller",
    "ensure_clearml_names",
    "ensure_clearml_project_system_tags",
    "get_clearml_project_system_tags",
    "resolve_clearml_pipeline_requirements",
]
