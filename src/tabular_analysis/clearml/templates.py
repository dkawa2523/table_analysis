from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from typing import Any
from ..ops.clearml_identity import build_template_project_name, build_template_tags, resolve_template_context
from ..platform_adapter_clearml_env import (
    clearml_script_mismatches,
    resolve_clearml_script_spec,
)
from ..platform_adapter_task import (
    clearml_task_id,
    clearml_task_script,
    clearml_task_status_from_obj,
    clearml_task_tags,
    list_clearml_tasks_by_tags,
)
def _template_project_name(cfg: Any, process: str) -> str | None:
    context = resolve_template_context(cfg)
    return build_template_project_name(context, process, cfg=cfg)
def resolve_template_task_id(cfg: Any, process: str) -> str:
    process_name = _normalize_str(process)
    if not process_name:
        raise ValueError("process is required for template lookup.")
    context = resolve_template_context(cfg)
    project_name = _template_project_name(cfg, process_name)
    base_tags = build_template_tags(process_name, context=context)
    expected_spec = resolve_clearml_script_spec(
        cfg,
        task_name_override=process_name,
        canonicalize_pipeline=False,
    )
    tasks = list_clearml_tasks_by_tags(base_tags, project_name=project_name)
    for task in tasks:
        task_tags = clearml_task_tags(task)
        if "template:deprecated" in task_tags:
            continue
        status = (clearml_task_status_from_obj(task) or "").lower()
        if status and status != "created":
            continue
        missing_tags = [required for required in base_tags if required not in task_tags]
        if missing_tags:
            continue
        script = clearml_task_script(task)
        if clearml_script_mismatches(expected_spec, script):
            continue
        task_id = clearml_task_id(task)
        if task_id:
            return task_id
    message = (
        f"Template task not found for process={process_name}, project={project_name}, "
        f"required_tags={base_tags}. Create/update ClearML template tasks before running pipeline steps."
    )
    raise RuntimeError(message)
