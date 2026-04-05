from __future__ import annotations

from typing import Any

from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..ops.clearml_identity import (
    build_pipeline_seed_project_name as _build_pipeline_seed_project_name,
    build_runtime_properties,
    build_runtime_tags,
    build_template_properties,
    build_task_contract_properties,
    build_task_contract_tags,
    resolve_template_context,
)
from ..platform_adapter_clearml_env import clearml_script_mismatches, resolve_clearml_script_spec
from ..platform_adapter_task import (
    clearml_task_id,
    clearml_task_script,
    clearml_task_status_from_obj,
    clearml_task_tags,
    list_clearml_tasks_by_tags,
)
from ..processes.pipeline_support import (
    DEFAULT_PIPELINE_PROFILE,
    PIPELINE_PROFILE_SPECS,
    is_pipeline_template_name,
    normalize_pipeline_profile,
)

PIPELINE_TEMPLATE_NAMES = frozenset(PIPELINE_PROFILE_SPECS.keys())


def _pipeline_seed_script_mismatches(spec: Any, script: dict[str, Any]) -> list[str]:
    raw = clearml_script_mismatches(spec, script)
    if isinstance(raw, bool):
        return ["script mismatch"] if raw else []
    return [str(error) for error in raw]


def _task_runtime(task: Any) -> dict[str, Any]:
    data = getattr(task, "data", None)
    runtime = getattr(data, "runtime", None) if data is not None else None
    if isinstance(runtime, dict):
        return {str(key): value for key, value in runtime.items()}
    return {}


def _task_artifact_names(task: Any) -> set[str]:
    artifacts = getattr(task, "artifacts", None)
    if isinstance(artifacts, dict):
        return {str(key) for key in artifacts.keys()}
    names: set[str] = set()
    if artifacts is not None and not isinstance(artifacts, (str, bytes)):
        try:
            for item in artifacts:
                key = None
                if isinstance(item, dict):
                    key = item.get("key") or item.get("name")
                else:
                    key = getattr(item, "key", None) or getattr(item, "name", None)
                if key:
                    names.add(str(key))
        except Exception:
            return names
    return names


def build_pipeline_seed_project_name(cfg: Any, *, pipeline_profile: str | None = None) -> str:
    context = resolve_template_context(cfg)
    return _build_pipeline_seed_project_name(
        context.project_root,
        pipeline_profile=normalize_pipeline_profile(pipeline_profile),
        layout=context.layout,
        cfg=cfg,
    )


def build_pipeline_template_project_name(cfg: Any, *, pipeline_profile: str | None = None) -> str:
    return build_pipeline_seed_project_name(cfg, pipeline_profile=pipeline_profile)


def build_pipeline_seed_tags(
    pipeline_profile: str,
    *,
    cfg: Any | None = None,
    context: Any | None = None,
    task_kind: str = 'seed',
    grid_run_id: str | None = None,
    extra_tags: list[str] | None = None,
) -> list[str]:
    ctx = context or resolve_template_context(cfg)
    profile = normalize_pipeline_profile(pipeline_profile)
    if task_kind == 'run':
        return build_runtime_tags(
            process='pipeline',
            schema_version=ctx.schema_version,
            usecase_id=ctx.usecase_id,
            pipeline_profile=profile,
            grid_run_id=grid_run_id,
            extra_tags=extra_tags,
        )
    tags = build_task_contract_tags(
        process='pipeline',
        schema_version=ctx.schema_version,
        task_kind=task_kind,
        usecase_id=ctx.usecase_id,
        template_set_id=ctx.template_set_id,
        pipeline_profile=profile,
        grid_run_id=grid_run_id,
        extra_tags=extra_tags,
    )
    if task_kind == 'seed':
        template_tag = f'template_set:{ctx.template_set_id or "default"}'
        if template_tag not in tags:
            tags.append(template_tag)
    return tags


def build_pipeline_template_tags(
    pipeline_profile: str,
    *,
    cfg: Any | None = None,
    context: Any | None = None,
    task_kind: str = 'seed',
    grid_run_id: str | None = None,
    extra_tags: list[str] | None = None,
) -> list[str]:
    return build_pipeline_seed_tags(
        pipeline_profile,
        cfg=cfg,
        context=context,
        task_kind=task_kind,
        grid_run_id=grid_run_id,
        extra_tags=extra_tags,
    )


def build_pipeline_seed_properties(
    pipeline_profile: str,
    *,
    cfg: Any | None = None,
    context: Any | None = None,
    task_kind: str = 'seed',
    default_queue: str | None = None,
    heavy_queue: str | None = None,
    grid_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = context or resolve_template_context(cfg)
    base_extra = dict(extra or {})
    if task_kind == 'run':
        return build_runtime_properties(
            process='pipeline',
            schema_version=ctx.schema_version,
            usecase_id=ctx.usecase_id,
            project_root=ctx.project_root,
            template_set_id=ctx.template_set_id,
            pipeline_profile=normalize_pipeline_profile(pipeline_profile),
            default_queue=default_queue,
            heavy_queue=heavy_queue,
            grid_run_id=grid_run_id,
            extra=base_extra,
        )
    return build_template_properties(
        'pipeline',
        context=ctx,
        extra=build_task_contract_properties(
            process='pipeline',
            schema_version=ctx.schema_version,
            task_kind=task_kind,
            usecase_id=ctx.usecase_id,
            project_root=ctx.project_root,
            template_set_id=ctx.template_set_id,
            pipeline_profile=normalize_pipeline_profile(pipeline_profile),
            default_queue=default_queue,
            heavy_queue=heavy_queue,
            extra=base_extra,
        ),
    )


def build_pipeline_template_properties(
    pipeline_profile: str,
    *,
    cfg: Any | None = None,
    context: Any | None = None,
    task_kind: str = 'seed',
    default_queue: str | None = None,
    heavy_queue: str | None = None,
    grid_run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_pipeline_seed_properties(
        pipeline_profile,
        cfg=cfg,
        context=context,
        task_kind=task_kind,
        default_queue=default_queue,
        heavy_queue=heavy_queue,
        grid_run_id=grid_run_id,
        extra=extra,
    )


def resolve_pipeline_seed_task_id(
    cfg: Any,
    *,
    pipeline_profile: str | None = None,
    template_task_id: str | None = None,
) -> str:
    explicit = _normalize_str(template_task_id) or _normalize_str(
        _cfg_value(cfg, 'run.clearml.pipeline.template_task_id')
    )
    if explicit:
        return explicit
    profile = normalize_pipeline_profile(pipeline_profile)
    project_name = build_pipeline_seed_project_name(cfg, pipeline_profile=profile)
    required_tags = build_pipeline_seed_tags(profile, cfg=cfg, task_kind='seed')
    expected_spec = resolve_clearml_script_spec(
        cfg,
        task_name_override='pipeline',
        canonicalize_pipeline=False,
    )
    tasks = list_clearml_tasks_by_tags(required_tags, project_name=project_name)
    for task in tasks:
        task_tags = clearml_task_tags(task)
        if 'template:deprecated' in task_tags:
            continue
        status = (clearml_task_status_from_obj(task) or '').lower()
        if status and status != 'completed':
            continue
        missing_tags = [required for required in required_tags if required not in task_tags]
        if missing_tags:
            continue
        if _pipeline_seed_script_mismatches(expected_spec, clearml_task_script(task)):
            continue
        runtime = _task_runtime(task)
        pipeline_hash = str(runtime.get('_pipeline_hash') or '').strip()
        if not pipeline_hash or pipeline_hash.startswith('None:'):
            continue
        artifact_names = _task_artifact_names(task)
        if not {
            'pipeline_run.json',
            'run_summary.json',
            'report.json',
            'manifest.json',
            'config_resolved.yaml',
        }.issubset(artifact_names):
            continue
        task_id = clearml_task_id(task)
        if task_id:
            return task_id
    raise RuntimeError(
        'Pipeline seed task not found for '
        f'project={project_name}, pipeline_profile={profile}, required_tags={required_tags}. '
        'Create/update seed pipelines with manage_templates.py --apply '
        'or specify run.clearml.pipeline.template_task_id.'
    )


def resolve_pipeline_template_task_id(
    cfg: Any,
    *,
    pipeline_profile: str | None = None,
    template_task_id: str | None = None,
) -> str:
    return resolve_pipeline_seed_task_id(
        cfg,
        pipeline_profile=pipeline_profile,
        template_task_id=template_task_id,
    )


__all__ = [
    'DEFAULT_PIPELINE_PROFILE',
    'PIPELINE_TEMPLATE_NAMES',
    'build_pipeline_seed_project_name',
    'build_pipeline_seed_properties',
    'build_pipeline_seed_tags',
    'build_pipeline_template_project_name',
    'build_pipeline_template_properties',
    'build_pipeline_template_tags',
    'is_pipeline_template_name',
    'normalize_pipeline_profile',
    'resolve_pipeline_seed_task_id',
    'resolve_pipeline_template_task_id',
]
