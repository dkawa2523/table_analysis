from __future__ import annotations

import re
from typing import Any, Iterable

from tabular_analysis.platform_adapter_clearml_policy import ensure_clearml_project_system_tags
from tabular_analysis.platform_adapter_task_query import (
    clearml_task_id,
    clearml_task_project_name,
)
from tabular_analysis.platform_adapter_task_ops import (
    set_clearml_task_project,
    update_clearml_task_tags,
)


def _load_clearml_task(task_id: str) -> Any:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect ClearML tasks.") from exc
    try:
        return ClearMLTask.get_task(task_id=str(task_id))
    except Exception as exc:
        raise RuntimeError(f"Failed to load ClearML task {task_id!r}: {exc}") from exc


def _list_clearml_tasks(project_name: str) -> list[Any]:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect template tasks.") from exc
    try:
        tasks = ClearMLTask.get_tasks(
            project_name=str(project_name),
            allow_archived=True,
            task_filter={"order_by": ["-last_update"]},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to list ClearML tasks for project {project_name!r}: {exc}") from exc
    return list(tasks or [])


def _project_regex(prefix: str) -> str:
    return "^" + re.escape(str(prefix)).replace("/", "\\/") + ".*$"


def _task_system_tags(task: Any) -> list[str]:
    getter = getattr(task, "get_system_tags", None)
    if callable(getter):
        try:
            values = getter()
        except Exception:
            values = None
        if isinstance(values, (list, tuple, set)):
            return [str(item) for item in values if item is not None]
        if values is not None:
            return [str(values)]
    values = getattr(task, "system_tags", None)
    if isinstance(values, (list, tuple, set)):
        return [str(item) for item in values if item is not None]
    if values is not None:
        return [str(values)]
    return []


def _deprecated_pipeline_project_name(project_name: str) -> str:
    text = str(project_name).rstrip("/")
    for (source, target) in (
        ("/.pipelines/", "/_DeprecatedPipelines/"),
        ("/Pipelines/", "/_DeprecatedPipelines/"),
    ):
        if source in text:
            return text.replace(source, target, 1)
    if text.endswith("/Pipelines"):
        return text.rsplit("/Pipelines", 1)[0] + "/_DeprecatedPipelines/legacy_seed_root"
    if text.endswith("/Templates"):
        text = text.rsplit("/", 1)[0]
    return f"{text}/_Deprecated"


def _list_clearml_project_names(prefix: str) -> list[str]:
    try:
        from clearml.backend_api.session.client import APIClient
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect pipeline projects.") from exc
    client = APIClient()
    try:
        projects = client.projects.get_all(name=_project_regex(prefix))
    except Exception as exc:
        raise RuntimeError(f"Failed to list ClearML projects for prefix {prefix!r}: {exc}") from exc
    names: list[str] = []
    for project in projects or []:
        name = str(getattr(project, "name", "") or "").strip()
        if name:
            names.append(name)
    return sorted(set(names))


def _list_clearml_task_ids_by_name_prefix(prefix: str) -> list[str]:
    try:
        from clearml.backend_api.session.client import APIClient
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect ClearML tasks.") from exc
    client = APIClient()
    try:
        tasks = client.tasks.get_all(name=prefix)
    except Exception as exc:
        raise RuntimeError(f"Failed to list ClearML tasks for prefix {prefix!r}: {exc}") from exc
    task_ids: list[str] = []
    for task in tasks or []:
        name = str(getattr(task, "name", "") or "").strip()
        task_id = str(getattr(task, "id", "") or "").strip()
        if task_id and name.startswith(prefix):
            task_ids.append(task_id)
    return sorted(set(task_ids))


def _remove_task_system_tags(task_id: str, remove_tags: Iterable[str]) -> None:
    remove_set = {str(tag).strip() for tag in remove_tags if str(tag).strip()}
    if not remove_set:
        return
    task = _load_clearml_task(task_id)
    current = _task_system_tags(task)
    updated = [tag for tag in current if tag not in remove_set]
    if updated == current:
        return
    persisted = False
    setter = getattr(task, "set_system_tags", None)
    if callable(setter):
        try:
            setter(updated)
            persisted = set(_task_system_tags(_load_clearml_task(task_id))) == set(updated)
        except Exception:
            persisted = False
    if persisted:
        return
    try:
        from clearml.backend_api.session import Session
    except ImportError:
        editor = getattr(task, "_edit", None)
        if callable(editor):
            editor(system_tags=updated)
        return
    session = Session()
    response = session.send_request(service="tasks", action="edit", json={"task": str(task_id), "system_tags": updated})
    if not getattr(response, "ok", False):
        raise RuntimeError(f"Failed to remove task system tags for {task_id!r}.")


def _remove_project_pipeline_visibility(project_name: str | None) -> None:
    text = str(project_name or "").strip()
    if not text:
        return
    ensure_clearml_project_system_tags(text, remove_tags=["pipeline"])


def deprecate_pipeline_task(
    task_id: str,
    *,
    actual_project: str | None = None,
    remove_source_project_pipeline_tag: bool = False,
    fallback_target_project: str | None = None,
) -> None:
    actual_project = str(actual_project or "").strip()
    update_clearml_task_tags(task_id, add=["template:deprecated"], remove=["pipeline"])
    _remove_task_system_tags(task_id, ["pipeline"])
    if "/_DeprecatedPipelines/" in actual_project:
        target_project = actual_project
    else:
        target_project = fallback_target_project or (_deprecated_pipeline_project_name(actual_project) if actual_project else "")
    if target_project:
        set_clearml_task_project(task_id, target_project)
        _remove_project_pipeline_visibility(target_project)
    if remove_source_project_pipeline_tag and actual_project:
        _remove_project_pipeline_visibility(actual_project)


def _candidate_pipeline_cleanup_projects(active_seed_projects: set[str], *, solution_prefix: str) -> set[str]:
    return (
        active_seed_projects
        | set(_list_clearml_project_names(f"{solution_prefix}/.pipelines/"))
        | set(_list_clearml_project_names(f"{solution_prefix}/_debug_seed_probe/.pipelines/"))
        | {f"{solution_prefix}/Pipelines", f"{solution_prefix}/Pipelines/Templates"}
    )


def _is_pipeline_cleanup_project(project_name: str, *, active_seed_projects: set[str]) -> bool:
    return (
        project_name in active_seed_projects
        or "/.pipelines/" in project_name
        or project_name.endswith("/Pipelines")
        or "/Pipelines/Templates" in project_name
    )


def _should_remove_pipeline_project_visibility(project_name: str, *, active_seed_projects: set[str]) -> bool:
    return (
        project_name not in active_seed_projects
        and (
            "/_debug_seed_probe/.pipelines/" in project_name
            or project_name.endswith("/Pipelines")
            or "/Pipelines/Templates" in project_name
        )
    )


def _deprecate_pipeline_project_tasks(
    project_name: str,
    *,
    active_seed_ids: set[str],
    active_seed_projects: set[str],
) -> None:
    remove_source_project_pipeline_tag = project_name not in active_seed_projects
    for task in _list_clearml_tasks(project_name):
        task_id = clearml_task_id(task)
        if not task_id or (project_name in active_seed_projects and task_id in active_seed_ids):
            continue
        deprecate_pipeline_task(
            task_id,
            actual_project=project_name,
            remove_source_project_pipeline_tag=remove_source_project_pipeline_tag,
        )


def cleanup_stale_pipeline_tasks(
    *,
    active_seed_ids: set[str],
    active_seed_projects: set[str],
) -> None:
    if not active_seed_projects:
        return
    solution_prefix = next(iter(active_seed_projects)).split("/.pipelines/", 1)[0]
    candidate_projects = _candidate_pipeline_cleanup_projects(active_seed_projects, solution_prefix=solution_prefix)
    cleanup_projects = sorted(
        project_name
        for project_name in candidate_projects
        if _is_pipeline_cleanup_project(project_name, active_seed_projects=active_seed_projects)
    )
    for project_name in cleanup_projects:
        _deprecate_pipeline_project_tasks(
            project_name,
            active_seed_ids=active_seed_ids,
            active_seed_projects=active_seed_projects,
        )
    for task_id in _list_clearml_task_ids_by_name_prefix("seed_probe_"):
        task = _load_clearml_task(task_id)
        actual_project = str(clearml_task_project_name(task) or "").strip()
        deprecate_pipeline_task(
            task_id,
            actual_project=actual_project,
            remove_source_project_pipeline_tag=True,
            fallback_target_project=f"{solution_prefix}/_DeprecatedPipelines/debug_probe",
        )
    for project_name in sorted(
        project_name
        for project_name in candidate_projects
        if _should_remove_pipeline_project_visibility(
            project_name,
            active_seed_projects=active_seed_projects,
        )
    ):
        _remove_project_pipeline_visibility(project_name)


__all__ = [
    "cleanup_stale_pipeline_tasks",
    "deprecate_pipeline_task",
]
