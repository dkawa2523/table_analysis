from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

from tabular_analysis.clearml.live_cleanup import (
    cleanup_stale_pipeline_tasks as _cleanup_stale_pipeline_tasks_live,
)
from tabular_analysis.clearml.pipeline_ui_contract import (
    pipeline_ui_parameter_whitelist,
    pipeline_ui_profiles,
)

from tools.clearml_templates.drift_validate import (
    pipeline_duplicate_visible_param_keys,
    pipeline_noncanonical_parameter_paths,
)


@dataclass(frozen=True)
class HistoricalPipelineArchivePolicy:
    archive_deprecated_pipeline_tasks: bool = True
    archive_noncanonical_pipeline_runs: bool = True


def build_active_pipeline_seed_scope(
    *,
    templates: list[Any],
    lock_templates: Mapping[str, Any],
    is_pipeline_template: Callable[[Any], bool],
) -> tuple[set[str], set[str]]:
    pipeline_specs = [spec for spec in templates if is_pipeline_template(spec)]
    active_seed_ids = {
        str(payload.get("task_id"))
        for spec in pipeline_specs
        for payload in [lock_templates.get(spec.name)]
        if isinstance(payload, Mapping) and payload.get("task_id")
    }
    active_seed_projects = {str(spec.project_name) for spec in pipeline_specs if spec.project_name}
    return (active_seed_ids, active_seed_projects)


def _project_regex(prefix: str) -> str:
    return "^" + re.escape(str(prefix)).replace("/", "\\/") + ".*$"


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


def _task_id(task: Any) -> str:
    value = getattr(task, "id", None)
    if value:
        return str(value)
    data = getattr(task, "data", None)
    value = getattr(data, "id", None) if data is not None else None
    return str(value or "")


def _task_archived(task: Any) -> bool:
    getter = getattr(task, "get_system_tags", None)
    if callable(getter):
        try:
            tags = getter()
        except Exception:
            tags = None
        if isinstance(tags, (list, tuple, set)):
            return "archived" in {str(tag) for tag in tags if tag is not None}
        if tags is not None:
            return str(tags) == "archived"
    data = getattr(task, "data", None)
    system_tags = getattr(data, "system_tags", None) if data is not None else None
    if isinstance(system_tags, (list, tuple, set)):
        return "archived" in {str(tag) for tag in system_tags if tag is not None}
    if system_tags is not None:
        return str(system_tags) == "archived"
    value = getattr(task, "is_archived", None)
    if callable(value):
        try:
            return bool(value())
        except Exception:
            return False
    value = getattr(task, "_is_archived", None)
    if value is not None:
        return bool(value)
    return False


def _set_task_archived(task: Any) -> bool:
    if _task_archived(task):
        return False
    setter = getattr(task, "set_archived", None)
    if not callable(setter):
        return False
    for args in ((True,), tuple(),):
        try:
            setter(*args)
            return True
        except TypeError:
            continue
        except Exception:
            return False
    try:
        setter(archived=True)
        return True
    except Exception:
        return False


def _expected_visible_pipeline_param_keys() -> set[str]:
    return {
        str(key).replace(".", "/")
        for profile in pipeline_ui_profiles()
        for key in pipeline_ui_parameter_whitelist(profile)
    }


def _is_noncanonical_pipeline_run(task: Any, *, expected_param_keys: set[str]) -> bool:
    if pipeline_noncanonical_parameter_paths(task):
        return True
    if pipeline_duplicate_visible_param_keys(task, expected_param_keys=expected_param_keys):
        return True
    return False


def _solution_prefix(active_seed_projects: set[str]) -> str:
    project = next(iter(active_seed_projects), "")
    return project.split("/.pipelines/", 1)[0] if "/.pipelines/" in project else project


def archive_deprecated_pipeline_tasks(*, active_seed_projects: set[str]) -> list[str]:
    solution_prefix = _solution_prefix(active_seed_projects)
    if not solution_prefix:
        return []
    archived: list[str] = []
    for project_name in _list_clearml_project_names(f"{solution_prefix}/_DeprecatedPipelines/"):
        for task in _list_clearml_tasks(project_name):
            if _set_task_archived(task):
                task_id = _task_id(task)
                if task_id:
                    archived.append(task_id)
    return archived


def archive_noncanonical_pipeline_runs(*, active_seed_projects: set[str]) -> list[str]:
    solution_prefix = _solution_prefix(active_seed_projects)
    if not solution_prefix:
        return []
    expected_param_keys = _expected_visible_pipeline_param_keys()
    archived: list[str] = []
    for project_name in _list_clearml_project_names(f"{solution_prefix}/Pipelines/Runs/"):
        for task in _list_clearml_tasks(project_name):
            if not _is_noncanonical_pipeline_run(task, expected_param_keys=expected_param_keys):
                continue
            if _set_task_archived(task):
                task_id = _task_id(task)
                if task_id:
                    archived.append(task_id)
    return archived


def cleanup_pipeline_history(
    *,
    active_seed_ids: set[str],
    active_seed_projects: set[str],
    archive_policy: HistoricalPipelineArchivePolicy | None = None,
) -> dict[str, int]:
    if not active_seed_projects:
        return {"archived_deprecated_tasks": 0, "archived_noncanonical_runs": 0}
    _cleanup_stale_pipeline_tasks_live(
        active_seed_ids=active_seed_ids,
        active_seed_projects=active_seed_projects,
    )
    policy = archive_policy or HistoricalPipelineArchivePolicy()
    archived_deprecated = (
        archive_deprecated_pipeline_tasks(active_seed_projects=active_seed_projects)
        if policy.archive_deprecated_pipeline_tasks
        else []
    )
    archived_runs = (
        archive_noncanonical_pipeline_runs(active_seed_projects=active_seed_projects)
        if policy.archive_noncanonical_pipeline_runs
        else []
    )
    return {
        "archived_deprecated_tasks": len(archived_deprecated),
        "archived_noncanonical_runs": len(archived_runs),
    }


__all__ = [
    "HistoricalPipelineArchivePolicy",
    "archive_deprecated_pipeline_tasks",
    "archive_noncanonical_pipeline_runs",
    "build_active_pipeline_seed_scope",
    "cleanup_pipeline_history",
]
