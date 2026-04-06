from __future__ import annotations

"""Compatibility facade for the small task-oriented adapter surface used by tools/tests."""

from .platform_adapter_common import PlatformAdapterError
from .platform_adapter_artifacts import get_task_artifact_local_copy
from .platform_adapter_task_query import (
    clearml_task_id,
    clearml_task_project_name,
    clearml_task_script,
    clearml_task_status_from_obj,
    clearml_task_tags,
    list_clearml_tasks_by_tags,
)
from .platform_adapter_task_context import (
    TaskContext,
    add_task_tags,
    connect_configuration,
    connect_hyperparameters,
    init_task_context,
    report_markdown,
    save_config_resolved,
    update_task_properties,
    upload_artifact,
    write_out_json,
)

__all__ = [
    "PlatformAdapterError",
    "TaskContext",
    "add_task_tags",
    "clearml_task_id",
    "clearml_task_project_name",
    "clearml_task_script",
    "clearml_task_status_from_obj",
    "clearml_task_tags",
    "connect_configuration",
    "connect_hyperparameters",
    "get_task_artifact_local_copy",
    "init_task_context",
    "list_clearml_tasks_by_tags",
    "report_markdown",
    "save_config_resolved",
    "update_task_properties",
    "upload_artifact",
    "write_out_json",
]
