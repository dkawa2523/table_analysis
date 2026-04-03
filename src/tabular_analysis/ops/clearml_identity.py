from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value, normalize_str as _normalize_str
from ..common.clearml_config import resolve_clearml_endpoint_summary
from ..common.schema_version import build_schema_tag as _build_schema_tag, normalize_schema_version as _normalize_schema_version
from ..clearml.naming import _extract_dataset_token, _sanitize_identifier
from ..common.repo_utils import resolve_repo_root_fallback
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_core import build_clearml_properties, build_clearml_tags
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_OMEGACONF_RECOVERABLE_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)
@dataclass
class ClearMLIdentity:
    project_root: str
    usecase_id: str
    tags: list[str]
    user_properties: dict[str, Any]


@dataclass(frozen=True)
class ClearMLTemplateContext:
    project_root: str
    usecase_id: str
    schema_version: str
    template_set_id: str
    layout: dict[str, Any]


@dataclass(frozen=True)
class ClearMLRuntimeContext:
    project_root: str
    usecase_id: str
    schema_version: str
    template_set_id: str
    project_name: str
    layout: dict[str, Any]
    endpoint: dict[str, str | None]
def _load_project_layout_from_file() -> dict[str, Any]:
    repo_root = resolve_repo_root_fallback(fallback=Path(__file__).resolve().parents[3])
    path = repo_root / 'conf' / 'clearml' / 'project_layout.yaml'
    if not path.exists():
        return {}
    try:
        from omegaconf import OmegaConf
    except _OPTIONAL_IMPORT_ERRORS:
        return {}
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except _OMEGACONF_RECOVERABLE_ERRORS:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}
def _resolve_project_layout(cfg: Any | None) -> dict[str, Any]:
    if cfg is not None:
        layout = _cfg_value(cfg, 'run.clearml.project_layout')
        try:
            from omegaconf import OmegaConf
        except _OPTIONAL_IMPORT_ERRORS:
            OmegaConf = None
        if OmegaConf is not None and OmegaConf.is_config(layout):
            layout = OmegaConf.to_container(layout, resolve=True)
        if isinstance(layout, Mapping):
            return dict(layout)
    return _load_project_layout_from_file()
def _ensure_project_layout(cfg: Any | None) -> dict[str, Any]:
    layout = _resolve_project_layout(cfg)
    if cfg is not None and layout:
        _set_cfg_value(cfg, 'run.clearml.project_layout', layout)
    return layout
def _infer_process_from_stage(stage: str | None) -> str | None:
    text = _normalize_str(stage)
    if not text:
        return None
    match = re.match('^\\d+_(.+)$', text)
    if match:
        return match.group(1)
    return text
def _resolve_project_root(cfg: Any) -> str:
    env_value = _normalize_str(os.getenv('TABULAR_ANALYSIS_CLEARML_PROJECT_ROOT'))
    if env_value:
        return env_value
    env_value = _normalize_str(os.getenv('CLEARML_PROJECT_ROOT'))
    if env_value:
        return env_value
    config_value = _normalize_str(_cfg_value(cfg, 'run.clearml.project_root'))
    return config_value or 'MFG'


def _resolve_schema_version(cfg: Any | None, value: Any | None = None) -> str:
    if value is not None:
        return _normalize_schema_version(value, default='v1')
    if cfg is not None:
        return _normalize_schema_version(_cfg_value(cfg, 'run.schema_version'), default='v1')
    return 'v1'


def _resolve_template_usecase_id(cfg: Any | None, value: Any | None = None) -> str:
    explicit = _normalize_str(value)
    if explicit:
        return explicit
    if cfg is not None:
        return _normalize_str(_cfg_value(cfg, 'run.clearml.template_usecase_id')) or _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'TabularAnalysis'
    return 'TabularAnalysis'


def _resolve_template_set_id(cfg: Any | None, value: Any | None = None) -> str:
    explicit = _normalize_str(value)
    if explicit:
        return explicit
    if cfg is not None:
        return _normalize_str(_cfg_value(cfg, 'run.clearml.template_set_id')) or 'default'
    return 'default'
def _resolve_policy(cfg: Any, path: str) -> Any:
    return _cfg_value(cfg, path) or {}
def _resolve_dataset_id(cfg: Any, policy: Any) -> str:
    paths = _cfg_value(policy, 'dataset_id_paths') or ['data.raw_dataset_id', 'data.processed_dataset_id', 'data.dataset_path']
    for path in list(paths) if isinstance(paths, Iterable) and (not isinstance(paths, str)) else [paths]:
        token = _extract_dataset_token(_cfg_value(cfg, str(path)))
        if token:
            return token
    fallback = _normalize_str(_cfg_value(policy, 'fallback'))
    return _sanitize_identifier(fallback or 'unknown')
def _generate_usecase_id(cfg: Any, policy: Any, now: datetime | None) -> str:
    strategy = _normalize_str(_cfg_value(policy, 'strategy')) or _normalize_str(_cfg_value(policy, 'name'))
    if not strategy:
        strategy = 'dataset_timestamp'
    if strategy == 'explicit':
        raise ValueError('run.usecase_id is required when usecase_id_policy is explicit.')
    if strategy == 'dataset_timestamp':
        prefix = _normalize_str(_cfg_value(policy, 'prefix')) or 'test'
        dataset_id = _resolve_dataset_id(cfg, policy)
        timestamp_format = _normalize_str(_cfg_value(policy, 'timestamp_format')) or '%Y%m%d_%H%M%S'
        now_value = now or datetime.now(timezone.utc)
        timestamp = now_value.strftime(timestamp_format)
        prefix = _sanitize_identifier(prefix)
        dataset_id = _sanitize_identifier(dataset_id)
        return f'{prefix}_{dataset_id}_{timestamp}'
    raise ValueError(f'Unsupported usecase_id_policy strategy: {strategy}')
def _filter_tags(values: Iterable[Any]) -> list[str]:
    filtered: list[str] = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text.lower() in ('none', 'null'):
            continue
        filtered.append(text)
    return filtered
def _filter_properties(values: Mapping[str, Any]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for (key, value) in values.items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in ('', 'none', 'null'):
            continue
        filtered[str(key)] = value
    return filtered
def build_project_name(project_root: str, usecase_id: str, stage: str, *, process: str | None=None, layout: Mapping[str, Any] | None=None, cfg: Any | None=None) -> str:
    layout_cfg = dict(layout or _resolve_project_layout(cfg))
    solution_root = _normalize_str(layout_cfg.get('solution_root')) or 'TabularAnalysis'
    separator = _normalize_str(layout_cfg.get('separator')) or '/'
    group_map = layout_cfg.get('group_map') if isinstance(layout_cfg.get('group_map'), Mapping) else {}
    misc_group = _normalize_str(layout_cfg.get('misc_group')) or '99_Misc'
    process_name = _normalize_str(process) or _infer_process_from_stage(stage)
    group = None
    if process_name and isinstance(group_map, Mapping):
        group = _normalize_str(group_map.get(process_name))
    if not group and isinstance(group_map, Mapping):
        group = _normalize_str(group_map.get(stage))
    group = group or misc_group or _normalize_str(stage) or 'unknown'
    root = _normalize_str(project_root) or 'MFG'
    usecase_value = _normalize_str(usecase_id) or 'unknown'
    root = root.rstrip(separator)
    return separator.join([root, solution_root, usecase_value, group])


def build_pipeline_project_name(
    project_root: str,
    *,
    layout: Mapping[str, Any] | None = None,
    cfg: Any | None = None,
) -> str:
    layout_cfg = dict(layout or _resolve_project_layout(cfg))
    solution_root = _normalize_str(layout_cfg.get('solution_root')) or 'TabularAnalysis'
    pipeline_root_group = _normalize_str(layout_cfg.get('pipeline_root_group')) or 'Pipelines'
    separator = _normalize_str(layout_cfg.get('separator')) or '/'
    root = (_normalize_str(project_root) or 'MFG').rstrip(separator)
    return separator.join([root, solution_root, pipeline_root_group])


def build_pipeline_child_project_name(
    project_root: str,
    usecase_id: str,
    stage: str,
    *,
    process: str | None = None,
    layout: Mapping[str, Any] | None = None,
    cfg: Any | None = None,
) -> str:
    layout_cfg = dict(layout or _resolve_project_layout(cfg))
    separator = _normalize_str(layout_cfg.get('separator')) or '/'
    pipeline_root = build_pipeline_project_name(project_root, layout=layout_cfg)
    usecase_value = _normalize_str(usecase_id) or 'unknown'
    process_name = _normalize_str(process) or _infer_process_from_stage(stage)
    group_map = layout_cfg.get('group_map') if isinstance(layout_cfg.get('group_map'), Mapping) else {}
    misc_group = _normalize_str(layout_cfg.get('misc_group')) or '99_Misc'
    group = None
    if process_name and isinstance(group_map, Mapping):
        group = _normalize_str(group_map.get(process_name))
    if not group and isinstance(group_map, Mapping):
        group = _normalize_str(group_map.get(stage))
    group = group or misc_group or _normalize_str(stage) or 'unknown'
    return separator.join([pipeline_root.rstrip(separator), usecase_value, group])


def resolve_template_context(
    cfg: Any | None=None,
    *,
    project_root: str | None=None,
    usecase_id: str | None=None,
    schema_version: str | None=None,
    template_set_id: str | None=None,
) -> ClearMLTemplateContext:
    layout = _ensure_project_layout(cfg)
    resolved_project_root = _normalize_str(project_root)
    if not resolved_project_root:
        resolved_project_root = _resolve_project_root(cfg) if cfg is not None else 'MFG'
    return ClearMLTemplateContext(
        project_root=resolved_project_root,
        usecase_id=_resolve_template_usecase_id(cfg, usecase_id),
        schema_version=_resolve_schema_version(cfg, schema_version),
        template_set_id=_resolve_template_set_id(cfg, template_set_id),
        layout=layout,
    )


def build_template_project_name(context: ClearMLTemplateContext, process: str, *, cfg: Any | None=None) -> str:
    return build_project_name(
        context.project_root,
        context.usecase_id,
        process,
        process=process,
        layout=context.layout,
        cfg=cfg,
    )


def build_template_tags(process: str, *, context: ClearMLTemplateContext) -> list[str]:
    return [
        'template:true',
        f'usecase:{context.usecase_id}',
        f'process:{process}',
        _build_schema_tag(context.schema_version, default='v1'),
        f'template_set:{context.template_set_id}',
        'solution:tabular-analysis',
    ]


def resolve_clearml_runtime_context(
    cfg: Any | None=None,
    *,
    stage: str,
    process: str | None=None,
    project_root: str | None=None,
    usecase_id: str | None=None,
    schema_version: str | None=None,
    template_set_id: str | None=None,
) -> ClearMLRuntimeContext:
    template_context = resolve_template_context(
        cfg,
        project_root=project_root,
        usecase_id=usecase_id,
        schema_version=schema_version,
        template_set_id=template_set_id,
    )
    process_name = _normalize_str(process) or _normalize_str(_cfg_value(cfg, 'task.name')) or _infer_process_from_stage(stage) or stage
    project_name = build_project_name(
        template_context.project_root,
        _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or template_context.usecase_id,
        stage,
        process=process_name,
        layout=template_context.layout,
        cfg=cfg,
    )
    return ClearMLRuntimeContext(
        project_root=template_context.project_root,
        usecase_id=_normalize_str(_cfg_value(cfg, 'run.usecase_id')) or template_context.usecase_id,
        schema_version=template_context.schema_version,
        template_set_id=template_context.template_set_id,
        project_name=project_name,
        layout=template_context.layout,
        endpoint=resolve_clearml_endpoint_summary(repo_root=resolve_repo_root_fallback(fallback=Path(__file__).resolve().parents[3])),
    )
def resolve_clearml_identity(cfg: Any, *, now: datetime | None=None) -> ClearMLIdentity:
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id'))
    policy = _resolve_policy(cfg, 'run.usecase_id_policy')
    if not usecase_id:
        usecase_id = _generate_usecase_id(cfg, policy, now)
    project_root = _resolve_project_root(cfg)
    clearml_policy = _resolve_policy(cfg, 'run.clearml.policy')
    policy_tags = _filter_tags(_cfg_value(clearml_policy, 'tags') or [])
    extra_tags = _filter_tags(_cfg_value(clearml_policy, 'extra_tags') or [])
    policy_properties = _filter_properties(_cfg_value(clearml_policy, 'properties') or {})
    extra_properties = _filter_properties(_cfg_value(clearml_policy, 'extra_properties') or {})
    merged_properties = dict(extra_properties)
    merged_properties.update(policy_properties)
    merged_tags = [*extra_tags, *policy_tags]
    return ClearMLIdentity(project_root=project_root, usecase_id=usecase_id, tags=merged_tags, user_properties=merged_properties)
def apply_clearml_identity(cfg: Any, *, stage: str, now: datetime | None=None) -> ClearMLIdentity:
    identity = resolve_clearml_identity(cfg, now=now)
    _set_cfg_value(cfg, 'run.usecase_id', identity.usecase_id)
    _set_cfg_value(cfg, 'run.clearml.project_root', identity.project_root)
    layout = _ensure_project_layout(cfg)
    process_name = _normalize_str(_cfg_value(cfg, 'task.name')) or _infer_process_from_stage(stage)
    project_name = build_project_name(identity.project_root, identity.usecase_id, stage, process=process_name, layout=layout)
    _set_cfg_value(cfg, 'run.clearml.project_name', project_name)
    _set_cfg_value(cfg, 'task.project_name', project_name)
    return identity
def resolve_clearml_metadata(cfg: Any, *, stage: str, task_name: str, identity: ClearMLIdentity | None=None, clearml_enabled: bool | None=None) -> dict[str, Any]:
    if identity is None:
        identity = resolve_clearml_identity(cfg)
    if not _normalize_str(_cfg_value(cfg, 'run.usecase_id')):
        _set_cfg_value(cfg, 'run.usecase_id', identity.usecase_id)
    if clearml_enabled is None:
        clearml_enabled = is_clearml_enabled(cfg)
    props = build_clearml_properties(cfg, stage=stage, task_name=task_name, extra=identity.user_properties, clearml_enabled=clearml_enabled)
    process = str(props.get('process') or task_name or stage)
    tags = build_clearml_tags(cfg, process=process, schema_version=str(props.get('schema_version') or 'unknown'), grid_run_id=props.get('grid_run_id'), retrain_run_id=props.get('retrain_run_id'), extra_tags=_cfg_value(cfg, 'run.clearml.extra_tags') or [], tags=identity.tags)
    layout = _ensure_project_layout(cfg)
    project_name = build_project_name(identity.project_root, identity.usecase_id, stage, process=process, layout=layout)
    return {'project_root': identity.project_root, 'project_name': project_name, 'usecase_id': identity.usecase_id, 'tags': tags, 'user_properties': props}
