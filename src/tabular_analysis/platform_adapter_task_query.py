from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .platform_adapter_common import (
    PlatformAdapterError,
    _CLEARML_TASK_CACHE,
    _RECOVERABLE_ERRORS,
    _dedupe_tags,
)


def _get_clearml_task(task_id: str) -> Any:
    cached = _CLEARML_TASK_CACHE.get(task_id)
    if cached is not None:
        return cached
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError("clearml is required for ClearML task retrieval.") from exc
    try:
        task = ClearMLTask.get_task(task_id=str(task_id))
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to load ClearML task: {task_id}") from exc
    _CLEARML_TASK_CACHE[task_id] = task
    return task


def clearml_task_exists(task_id: str) -> bool:
    _get_clearml_task(task_id)
    return True


def list_clearml_tasks_by_tags(
    tags: Iterable[str],
    *,
    project_name: str | None = None,
    task_name: str | None = None,
    allow_archived: bool = True,
    order_by: Iterable[str] | None = None,
) -> list[Any]:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise PlatformAdapterError("clearml is required for ClearML task queries.") from exc
    tag_list = _dedupe_tags(tags)
    if not tag_list:
        raise PlatformAdapterError("tags are required to query ClearML tasks.")
    task_filter = {"order_by": list(order_by) if order_by else ["-last_update"]}
    try:
        tasks = ClearMLTask.get_tasks(
            project_name=project_name,
            task_name=task_name,
            tags=tag_list,
            allow_archived=allow_archived,
            task_filter=task_filter,
        )
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        raise PlatformAdapterError(f"Failed to query ClearML tasks by tags: {exc}") from exc
    required = set(tag_list)
    filtered: list[Any] = []
    for task in list(tasks or []):
        task_tags = set(_task_tags(task))
        if required.issubset(task_tags):
            filtered.append(task)
    return filtered


def clearml_task_id(task: Any) -> str | None:
    task_id = getattr(task, "id", None) or getattr(task, "task_id", None)
    return str(task_id) if task_id else None


def clearml_task_tags(task: Any) -> list[str]:
    return _dedupe_tags(_task_tags(task))


def clearml_task_script(task: Any) -> dict[str, Any]:
    return _task_script(task)


def clearml_task_status_from_obj(task: Any) -> str | None:
    status = getattr(task, "status", None)
    if status:
        return str(status)
    getter = getattr(task, "get_status", None)
    if callable(getter):
        try:
            status = getter()
        except _RECOVERABLE_ERRORS:
            status = None
        if status:
            return str(status)
    return None


def clearml_task_type_from_obj(task: Any) -> str | None:
    for attr in ("task_type", "type"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    data = getattr(task, "data", None)
    if isinstance(data, Mapping):
        value = data.get("task_type") or data.get("type")
        if value:
            return str(value)
    if data is not None:
        for attr in ("task_type", "type"):
            value = getattr(data, attr, None)
            if value:
                return str(value)
    return None


def clearml_task_project_name(task: Any) -> str | None:
    getter = getattr(task, "get_project_name", None)
    if callable(getter):
        try:
            value = getter()
        except _RECOVERABLE_ERRORS:
            value = None
        if value:
            return str(value)
    for attr in ("project_name", "project"):
        value = getattr(task, attr, None)
        if value:
            text = str(value)
            if attr == "project" and "/" not in text:
                continue
            return text
    data = getattr(task, "data", None)
    if isinstance(data, Mapping):
        for key in ("project_name", "project"):
            value = data.get(key)
            if value:
                text = str(value)
                if key == "project" and "/" not in text:
                    continue
                return text
    if data is not None:
        for attr in ("project_name", "project"):
            value = getattr(data, attr, None)
            if value:
                text = str(value)
                if attr == "project" and "/" not in text:
                    continue
                return text
    return None


def find_clearml_task_id_by_tags(
    tags: Iterable[str],
    *,
    project_name: str | None = None,
    task_name: str | None = None,
    allow_archived: bool = True,
) -> str | None:
    try:
        from clearml import Task as ClearMLTask
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError("clearml is required for ClearML task queries.") from exc
    tag_list = _dedupe_tags(tags)
    if not tag_list:
        raise PlatformAdapterError("tags are required to query ClearML tasks.")
    try:
        tasks = ClearMLTask.get_tasks(
            project_name=project_name,
            task_name=task_name,
            tags=tag_list,
            allow_archived=allow_archived,
            task_filter={"order_by": ["-last_update"]},
        )
    except _RECOVERABLE_ERRORS as exc:
        raise PlatformAdapterError(f"Failed to query ClearML tasks by tags: {exc}") from exc
    if not tasks:
        return None
    return clearml_task_id(tasks[0])


def _task_tags(task: Any) -> list[str]:
    getter = getattr(task, "get_tags", None)
    if callable(getter):
        try:
            tags = getter()
        except _RECOVERABLE_ERRORS:
            tags = None
        if tags is not None:
            if isinstance(tags, (list, tuple, set)):
                return [str(tag) for tag in tags if tag is not None]
            return [str(tags)]
    tags = getattr(task, "tags", None)
    if tags is not None:
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
        return [str(tags)]
    data = getattr(task, "data", None)
    if isinstance(data, Mapping):
        tags = data.get("tags")
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
    if data is not None:
        tags = getattr(data, "tags", None)
        if isinstance(tags, (list, tuple, set)):
            return [str(tag) for tag in tags if tag is not None]
    return []


def _task_script(task: Any) -> dict[str, Any]:
    script: dict[str, Any] = {}
    getter = getattr(task, "get_script", None)
    if callable(getter):
        try:
            script_value = getter()
        except _RECOVERABLE_ERRORS:
            script_value = None
        if isinstance(script_value, Mapping):
            script.update(dict(script_value))
    data = getattr(task, "data", None)
    script_obj = getattr(data, "script", None) if data is not None else None
    if script_obj is not None:
        fallback = {
            "repository": getattr(script_obj, "repository", None),
            "branch": getattr(script_obj, "branch", None),
            "entry_point": getattr(script_obj, "entry_point", None),
            "working_dir": getattr(script_obj, "working_dir", None),
            "version_num": getattr(script_obj, "version_num", None),
            "diff": getattr(script_obj, "diff", None),
        }
        for key, value in fallback.items():
            if key not in script or script[key] is None:
                script[key] = value
    return script


def _task_parameters(task: Any) -> dict[str, Any]:
    getter = getattr(task, "get_parameters", None)
    if callable(getter):
        try:
            params = getter()
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            return dict(params)
    getter = getattr(task, "get_parameters_as_dict", None)
    if callable(getter):
        try:
            params = getter()
        except _RECOVERABLE_ERRORS:
            params = None
        if isinstance(params, Mapping):
            flat: dict[str, Any] = {}
            for section, values in params.items():
                if not isinstance(values, Mapping):
                    continue
                for key, value in values.items():
                    flat[f"{section}/{key}"] = value
            return flat
    return {}


def get_clearml_task_tags(task_id: str) -> list[str]:
    task = _get_clearml_task(task_id)
    return _dedupe_tags(_task_tags(task))


def get_clearml_task_script(task_id: str) -> dict[str, Any]:
    task = _get_clearml_task(task_id)
    script = _task_script(task)
    return {
        "repository": script.get("repository"),
        "branch": script.get("branch"),
        "entry_point": script.get("entry_point"),
        "working_dir": script.get("working_dir"),
        "version_num": script.get("version_num"),
    }


def get_clearml_task_configuration(task_id: str, *, name: str = "effective") -> Any | None:
    task = _get_clearml_task(task_id)
    getter = getattr(task, "get_configuration_object_as_dict", None)
    if callable(getter):
        try:
            return getter(str(name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f"Failed to read task configuration {name!r}: {exc}") from exc
    getter = getattr(task, "get_configuration_object", None)
    if callable(getter):
        try:
            value = getter(str(name))
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f"Failed to read task configuration {name!r}: {exc}") from exc
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    raise PlatformAdapterError("ClearML task configuration API is not available.")


def get_clearml_task_status(task_id: str) -> str | None:
    task = _get_clearml_task(task_id)
    return clearml_task_status_from_obj(task)


__all__ = [
    "_get_clearml_task",
    "_task_parameters",
    "_task_script",
    "_task_tags",
    "clearml_task_exists",
    "clearml_task_id",
    "clearml_task_project_name",
    "clearml_task_script",
    "clearml_task_status_from_obj",
    "clearml_task_tags",
    "clearml_task_type_from_obj",
    "find_clearml_task_id_by_tags",
    "get_clearml_task_configuration",
    "get_clearml_task_script",
    "get_clearml_task_status",
    "get_clearml_task_tags",
    "list_clearml_tasks_by_tags",
]
