from __future__ import annotations

from typing import Any

from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..ops.clearml_identity import (
    build_pipeline_project_name,
    build_template_tags,
    resolve_template_context,
)
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

DEFAULT_PIPELINE_PROFILE = "pipeline"
PIPELINE_TEMPLATE_NAMES = {
    "pipeline",
    "train_model_full",
    "train_ensemble_full",
}


def normalize_pipeline_profile(value: Any, *, default: str = DEFAULT_PIPELINE_PROFILE) -> str:
    text = _normalize_str(value)
    return text or default


def is_pipeline_template_name(value: Any) -> bool:
    return normalize_pipeline_profile(value) in PIPELINE_TEMPLATE_NAMES


def build_pipeline_template_project_name(cfg: Any) -> str:
    context = resolve_template_context(cfg)
    return build_pipeline_project_name(context.project_root, layout=context.layout, cfg=cfg)


def build_pipeline_template_tags(
    pipeline_profile: str,
    *,
    cfg: Any | None = None,
    context: Any | None = None,
    task_kind: str = "template",
) -> list[str]:
    ctx = context or resolve_template_context(cfg)
    profile = normalize_pipeline_profile(pipeline_profile)
    base_tags = build_template_tags("pipeline", context=ctx)
    if _normalize_str(task_kind) != "template":
        base_tags = [tag for tag in base_tags if tag != "template:true"]
    return [
        *base_tags,
        f"task_kind:{task_kind}",
        f"pipeline_profile:{profile}",
    ]


def build_pipeline_template_properties(
    pipeline_profile: str,
    *,
    task_kind: str = "template",
) -> dict[str, Any]:
    return {
        "process": "pipeline",
        "task_kind": str(task_kind),
        "pipeline_profile": normalize_pipeline_profile(pipeline_profile),
    }


def resolve_pipeline_template_task_id(
    cfg: Any,
    *,
    pipeline_profile: str | None = None,
    template_task_id: str | None = None,
) -> str:
    explicit = _normalize_str(template_task_id) or _normalize_str(
        _cfg_value(cfg, "run.clearml.pipeline.template_task_id")
    )
    if explicit:
        return explicit
    profile = normalize_pipeline_profile(pipeline_profile)
    project_name = build_pipeline_template_project_name(cfg)
    required_tags = build_pipeline_template_tags(profile, cfg=cfg)
    expected_spec = resolve_clearml_script_spec(
        cfg,
        task_name_override="pipeline",
        canonicalize_pipeline=False,
    )
    tasks = list_clearml_tasks_by_tags(required_tags, project_name=project_name)
    for task in tasks:
        task_tags = clearml_task_tags(task)
        if "template:deprecated" in task_tags:
            continue
        status = (clearml_task_status_from_obj(task) or "").lower()
        if status and status not in {"created", "in_progress", "stopped"}:
            continue
        missing_tags = [required for required in required_tags if required not in task_tags]
        if missing_tags:
            continue
        script = clearml_task_script(task)
        if clearml_script_mismatches(expected_spec, script):
            continue
        task_id = clearml_task_id(task)
        if task_id:
            return task_id
    raise RuntimeError(
        "Pipeline template task not found for "
        f"project={project_name}, pipeline_profile={profile}, required_tags={required_tags}. "
        "Create/update visible pipeline templates with manage_templates.py --apply "
        "or specify run.clearml.pipeline.template_task_id."
    )


__all__ = [
    "DEFAULT_PIPELINE_PROFILE",
    "PIPELINE_TEMPLATE_NAMES",
    "build_pipeline_template_project_name",
    "build_pipeline_template_properties",
    "build_pipeline_template_tags",
    "is_pipeline_template_name",
    "normalize_pipeline_profile",
    "resolve_pipeline_template_task_id",
]
