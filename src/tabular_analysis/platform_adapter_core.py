from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from .platform_adapter_artifacts import hash_config, hash_recipe, hash_split, resolve_output_dir
from .platform_adapter_clearml_env import (
    ClearMLScriptSpec,
    _load_clearml_dataset,
    _load_clearml_module,
    _load_clearml_pipeline_utils,
    clearml_script_mismatches,
    detect_git_branch,
    detect_git_repository_url,
    is_clearml_enabled,
    normalize_clearml_branch,
    normalize_clearml_entry_point,
    normalize_clearml_repository,
    normalize_clearml_version_num,
    resolve_clearml_code_reference,
    resolve_clearml_script_spec,
    resolve_version_props,
)
from .platform_adapter_clearml_policy import (
    _apply_clearml_pipeline_args,
    _apply_clearml_system_tags,
    _apply_clearml_tags,
    _apply_clearml_task_requirements,
    _apply_clearml_task_script_override,
    _apply_clearml_task_type,
    _ensure_clearml_names,
    _ensure_clearml_project_system_tags,
    _get_clearml_project_system_tags,
    _resolve_clearml_pipeline_requirements,
    build_clearml_properties,
    build_clearml_tags,
    clearml_task_type_controller,
)
from .platform_adapter_common import (
    PlatformAdapterError,
    _CLEARML_TASK_CACHE,
    _RECOVERABLE_ERRORS,
    _apply_clearml_files_host_substitution,
    _dedupe_tags,
    _existing_user_properties,
    _normalize_files_host,
    _normalize_requirement_lines,
    _resolve_clearml_task,
)

if TYPE_CHECKING:
    from .platform_adapter_task_context import TaskContext

"""
Legacy compatibility surface for adapter callers that still import `platform_adapter_core`.

New code should prefer the narrower modules directly:
- platform_adapter_clearml_env
- platform_adapter_clearml_policy
- platform_adapter_task_query
- platform_adapter_task_ops
- platform_adapter_artifacts
"""


def update_registry_model_tags(
    *,
    model_id: str,
    add_tags: Iterable[str] | None = None,
    remove_prefixes: Iterable[str] | None = None,
) -> list[str]:
    from . import platform_adapter_model as _model

    return _model.update_registry_model_tags(
        model_id=model_id,
        add_tags=add_tags,
        remove_prefixes=remove_prefixes,
    )


def ensure_clearml_task_tags(task_id: str, tags: Iterable[str]) -> bool:
    from . import platform_adapter_task_ops as _task_ops

    return _task_ops.ensure_clearml_task_tags(task_id, tags)


def write_manifest(ctx: TaskContext, manifest: dict[str, Any]) -> Path:
    """Write manifest.json, using the ml_platform helper when available."""
    try:
        from ml_platform.artifacts import write_manifest as platform_write_manifest
    except ImportError as exc:
        if ctx.task is not None:
            raise PlatformAdapterError("ml_platform.artifacts.write_manifest not available.") from exc
        platform_write_manifest = None
    if platform_write_manifest is not None:
        try:
            return platform_write_manifest(
                manifest,
                output_dir=ctx.output_dir,
                task=ctx.task,
                filename="manifest.json",
            )
        except _RECOVERABLE_ERRORS as exc:
            raise PlatformAdapterError(f"Failed to write manifest via ml_platform: {exc}") from exc
    path = ctx.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def register_dataset(
    cfg: Any,
    *,
    dataset_path: Path,
    dataset_name: str,
    dataset_project: str | None = None,
    dataset_tags: Optional[Iterable[str]] = None,
    dataset_version: str | None = None,
    description: str | None = None,
    parent_dataset_ids: Optional[Iterable[str]] = None,
    task_sections: Optional[Mapping[str, Mapping[str, Any]]] = None,
    task_section_order: Optional[Iterable[str]] = None,
) -> str:
    from .platform_adapter_dataset import register_dataset as _impl

    return _impl(
        cfg,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=dataset_tags,
        dataset_version=dataset_version,
        description=description,
        parent_dataset_ids=parent_dataset_ids,
        task_sections=task_sections,
        task_section_order=task_section_order,
    )


def get_dataset_local_copy(cfg: Any, dataset_id: str) -> Path:
    from .platform_adapter_dataset import get_dataset_local_copy as _impl

    return _impl(cfg, dataset_id)


def get_dataset_info(cfg: Any, dataset_id: str) -> dict[str, Any]:
    from .platform_adapter_dataset import get_dataset_info as _impl

    return _impl(cfg, dataset_id)


def create_pipeline_controller(
    cfg: Any,
    *,
    name: str | None = None,
    tags: Iterable[str] | None = None,
    properties: Mapping[str, Any] | None = None,
    default_queue: str | None = None,
) -> Any:
    from .platform_adapter_pipeline import create_pipeline_controller as _impl

    return _impl(cfg, name=name, tags=tags, properties=properties, default_queue=default_queue)


def pipeline_require_clearml_agent(queue_name: str | None = None) -> None:
    from .platform_adapter_pipeline import pipeline_require_clearml_agent as _impl

    _impl(queue_name)


def pipeline_step_task_id_ref(step_name: str) -> str:
    from .platform_adapter_pipeline import pipeline_step_task_id_ref as _impl

    return _impl(step_name)
