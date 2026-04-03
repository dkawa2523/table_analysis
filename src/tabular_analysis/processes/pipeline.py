from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, set_cfg_value as _set_cfg_value, normalize_str as _normalize_str, to_int as _to_int
from ..common.collection_utils import to_list as _to_list, to_mapping as _to_mapping
from ..common.dataset_utils import derive_local_raw_dataset_id
from ..common.clearml_bootstrap import resolve_required_uv_extras
from ..common.json_utils import load_json as _load_json
from ..common.model_reference import build_infer_reference, resolve_preferred_infer_reference
from ..common.hydra_overrides import format_value as _format_value, overrides_to_args as _overrides_to_args, overrides_to_params as _overrides_to_params, sanitize_component as _sanitize_component
from ..common.repo_utils import resolve_repo_root_fallback as _resolve_repo_root
from ..common.cli_task_runner import run_cli_task as _run_cli_task
from dataclasses import dataclass
import inspect
from itertools import product
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Mapping
import uuid
from ..clearml.hparams import connect_pipeline
from ..clearml.pipeline_templates import (
    build_pipeline_template_tags as _build_pipeline_template_tags,
    build_pipeline_template_properties as _build_pipeline_template_properties,
    normalize_pipeline_profile,
    resolve_pipeline_template_task_id,
)
from ..clearml.templates import resolve_template_task_id
from ..clearml.ui_logger import log_scalar
from ..platform_adapter_artifacts import upload_artifact
from ..platform_adapter_clearml_env import clearml_task_type_controller, is_clearml_enabled
from ..platform_adapter_pipeline import apply_clearml_task_overrides, clone_pipeline_controller, create_pipeline_controller, enqueue_pipeline_controller, load_pipeline_controller_from_task, pipeline_require_clearml_agent, pipeline_step_task_id_ref
from ..platform_adapter_task import clearml_task_id, report_markdown, set_clearml_task_parameters, update_clearml_task_tags, ensure_clearml_task_properties, ensure_clearml_task_tags
from ..platform_adapter_task_context import TaskContext, save_config_resolved
from ..ops.clearml_identity import apply_clearml_identity, build_project_name, build_template_tags, resolve_clearml_metadata, resolve_template_context
from ..reporting.pipeline_report import build_pipeline_report_bundle
from .lifecycle import emit_outputs_and_manifest, start_runtime
_STAGE_BY_TASK = {'dataset_register': '01_dataset_register', 'preprocess': '02_preprocess', 'train_model': '03_train_model', 'train_ensemble': '04_train_ensemble', 'infer': '04_infer', 'leaderboard': '05_leaderboard'}
_PIPELINE_TEMPLATE_PROFILES = {
    'pipeline': {
        'run_dataset_register': False,
        'run_preprocess': True,
        'run_train': True,
        'run_train_ensemble': False,
        'run_leaderboard': True,
        'run_infer': False,
        'model_set': 'regression_all',
    },
    'train_model_full': {
        'run_dataset_register': False,
        'run_preprocess': True,
        'run_train': True,
        'run_train_ensemble': False,
        'run_leaderboard': False,
        'run_infer': False,
        'model_set': 'regression_all',
    },
    'train_ensemble_full': {
        'run_dataset_register': False,
        'run_preprocess': True,
        'run_train': True,
        'run_train_ensemble': True,
        'run_leaderboard': True,
        'run_infer': False,
        'model_set': 'regression_all',
    },
}
_PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS = {'run.output_dir', 'train.inputs.preprocess_run_dir'}
_PIPELINE_TEMPLATE_DYNAMIC_KEYS = {
    'run.usecase_id',
    'run.schema_version',
    'run.grid_run_id',
    'run.clearml.enabled',
    'run.clearml.execution',
    'run.clearml.project_root',
    'run.clearml.project_name',
    'run.clearml.queue_name',
    'run.clearml.pipeline_task_id',
    'run.clearml.pipeline.template_task_id',
    'run.clearml.extra_tags',
    'run.retrain_run_id',
    'ops/usecase_id_policy',
    'ops/clearml_policy',
    'data.dataset_path',
    'data.raw_dataset_id',
    'data.processed_dataset_id',
    'data.target_column',
    'data.id_columns',
    'data.drop_columns',
    'data.split.strategy',
    'data.split.test_size',
    'data.split.seed',
    'data.split.group_column',
    'data.split.time_column',
    'eval.primary_metric',
    'eval.direction',
    'eval.cv_folds',
    'eval.seed',
    'eval.task_type',
    'eval.classification.mode',
    'eval.classification.top_k',
    'eval.metrics.classification_multiclass',
    'eval.calibration.enabled',
    'eval.uncertainty.enabled',
    'eval.ci.enabled',
    'infer.model_id',
    'infer.train_task_id',
    'infer.mode',
}


@dataclass(frozen=True)
class _VisiblePipelineRunContract:
    plan: Mapping[str, Any]
    pipeline_profile: str
    metadata: Mapping[str, Any]
    queue_name: str | None


def _running_inside_clearml_task() -> bool:
    return bool(os.getenv('CLEARML_TASK_ID') or os.getenv('TRAINS_TASK_ID'))


def _create_local_task_context(cfg: Any, *, stage: str, task_name: str) -> TaskContext:
    output_dir = Path(getattr(cfg.run, 'output_dir', 'outputs')) / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    project_name = _normalize_str(_cfg_value(cfg, 'run.clearml.project_name')) or _normalize_str(_cfg_value(cfg, 'task.project_name')) or f"MFG/TabularAnalysis/{_normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'}/{stage}"
    task_name_value = _normalize_str(_cfg_value(cfg, 'run.clearml.task_name')) or task_name
    ctx = TaskContext(task=None, project_name=project_name, task_name=task_name_value, output_dir=output_dir)
    save_config_resolved(ctx, cfg)
    return ctx


def _normalize_template_arg_value(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, Path):
        return str(value)
    return value


def _template_pipeline_param_name(name: str) -> str:
    return str(name).replace('.', '/')


def _template_pipeline_params(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        _template_pipeline_param_name(key): _normalize_template_arg_value(value)
        for (key, value) in values.items()
        if value is not None
    }


def _pipeline_param_ref(name: str) -> str:
    return f'${{pipeline.{_template_pipeline_param_name(name)}}}'


def _strip_local_only_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for (key, value) in dict(overrides).items() if str(key) not in _PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS}


def _build_pipeline_template_runtime_defaults(*, cfg: Any, plan: Mapping[str, Any], grid_run_id: str, pipeline_profile: str, pipeline_task_id: str | None=None) -> dict[str, Any]:
    defaults = _merge_overrides(plan['run_overrides'], plan['data_overrides'], plan['downstream_data_overrides'], plan['eval_overrides'])
    defaults = _strip_local_only_overrides(defaults)
    defaults['task'] = 'pipeline'
    defaults['default_queue'] = _normalize_str(plan.get('queues', {}).get('default')) or _normalize_str(_cfg_value(cfg, 'run.clearml.queue_name')) or 'default'
    defaults['run.grid_run_id'] = grid_run_id
    defaults['run.clearml.execution'] = 'pipeline_controller'
    defaults['run.clearml.pipeline.template_task_id'] = _normalize_str(_cfg_value(cfg, 'run.clearml.pipeline.template_task_id'))
    defaults['run.clearml.pipeline_task_id'] = pipeline_task_id or ''
    defaults['pipeline.profile'] = normalize_pipeline_profile(pipeline_profile)
    defaults['pipeline.run_dataset_register'] = plan.get('run_dataset_register')
    defaults['pipeline.run_preprocess'] = plan.get('run_preprocess')
    defaults['pipeline.run_train'] = plan.get('run_train')
    defaults['pipeline.run_train_ensemble'] = plan.get('run_train_ensemble')
    defaults['pipeline.run_leaderboard'] = plan.get('run_leaderboard')
    defaults['pipeline.run_infer'] = plan.get('run_infer')
    defaults['pipeline.grid.preprocess_variants'] = plan.get('preprocess_variants')
    defaults['pipeline.grid.model_variants'] = plan.get('model_variants')
    defaults['pipeline.model_set'] = _normalize_str(_cfg_value(cfg, 'pipeline.model_set'))
    return {key: _normalize_template_arg_value(value) for (key, value) in defaults.items() if value is not None}


def _build_template_step_overrides(step_overrides: Mapping[str, Any], shared_defaults: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for (key, value) in _strip_local_only_overrides(step_overrides).items():
        key_text = str(key)
        if key_text in shared_defaults:
            overrides[key_text] = _pipeline_param_ref(key_text)
        else:
            overrides[key_text] = value
    return overrides


def _detect_pipeline_template_profile(plan: Mapping[str, Any]) -> str | None:
    plan_signature = {
        'run_dataset_register': bool(plan.get('run_dataset_register')),
        'run_preprocess': bool(plan.get('run_preprocess')),
        'run_train': bool(plan.get('run_train')),
        'run_train_ensemble': bool(plan.get('run_train_ensemble')),
        'run_leaderboard': bool(plan.get('run_leaderboard')),
        'run_infer': bool(plan.get('run_infer')),
        'model_set': _normalize_str(plan.get('model_set')),
    }
    for (profile, signature) in _PIPELINE_TEMPLATE_PROFILES.items():
        if all(plan_signature.get(key) == value for (key, value) in signature.items()):
            return profile
    return None
def _resolve_exec_policy_limits(cfg: Any) -> dict[str, int]:
    limits_cfg = _to_mapping(_cfg_value(cfg, 'exec_policy.limits'))
    max_jobs = _to_int(limits_cfg.get('max_jobs'), 0) if 'max_jobs' in limits_cfg else _to_int(_cfg_value(cfg, 'pipeline.grid.max_jobs'), 0)
    max_models = _to_int(limits_cfg.get('max_models'), 0) if 'max_models' in limits_cfg else _to_int(_cfg_value(cfg, 'leaderboard.top_k'), 0)
    max_hpo_trials = _to_int(limits_cfg.get('max_hpo_trials'), 0)
    if max_jobs < 0:
        max_jobs = 0
    if max_models < 0:
        max_models = 0
    if max_hpo_trials < 0:
        max_hpo_trials = 0
    return {'max_jobs': max_jobs, 'max_models': max_models, 'max_hpo_trials': max_hpo_trials}
def _resolve_exec_policy_selection(cfg: Any) -> dict[str, bool]:
    selection_cfg = _to_mapping(_cfg_value(cfg, 'exec_policy.selection'))
    return {key: bool(selection_cfg.get(key)) for key in ('calibration', 'uncertainty', 'ci') if key in selection_cfg}
def _apply_exec_policy_selection(overrides: dict[str, Any], selection: Mapping[str, bool]) -> None:
    for (key, path) in (('calibration', 'eval.calibration.enabled'), ('uncertainty', 'eval.uncertainty.enabled'), ('ci', 'eval.ci.enabled')):
        if key in selection and (not selection[key]):
            overrides[path] = False
def _resolve_exec_policy_queues(cfg: Any) -> dict[str, Any]:
    queues_cfg = _to_mapping(_cfg_value(cfg, 'exec_policy.queues'))
    def _queue(key: str) -> str | None:
        return _normalize_str(queues_cfg.get(key))
    model_variants_cfg = _to_mapping(queues_cfg.get('model_variants'))
    model_variants = {str(key): _normalize_str(value) for (key, value) in model_variants_cfg.items() if _normalize_str(value)}
    heavy_variants = {str(name) for name in _to_list(queues_cfg.get('heavy_model_variants')) if _normalize_str(name)}
    default_queue = _queue('default') or _normalize_str(_cfg_value(cfg, 'run.clearml.queue_name'))
    return {'default': default_queue, 'pipeline': _queue('pipeline'), 'dataset_register': _queue('dataset_register'), 'preprocess': _queue('preprocess'), 'train_model': _queue('train_model'), 'train_ensemble': _queue('train_ensemble'), 'train_model_heavy': _queue('train_model_heavy'), 'leaderboard': _queue('leaderboard'), 'infer': _queue('infer'), 'model_variants': model_variants, 'heavy_model_variants': heavy_variants}
def _select_queue(queues: Mapping[str, Any], process: str, *, model_variant: str | None=None) -> str | None:
    default_queue = _normalize_str(queues.get('default'))
    if process == 'train_model':
        model_variants = queues.get('model_variants') or {}
        if model_variant:
            variant_queue = _normalize_str(model_variants.get(model_variant))
            if variant_queue:
                return variant_queue
            heavy_variants = queues.get('heavy_model_variants') or set()
            if model_variant in heavy_variants:
                heavy_queue = _normalize_str(queues.get('train_model_heavy'))
                if heavy_queue:
                    return heavy_queue
        train_queue = _normalize_str(queues.get('train_model'))
        return train_queue or default_queue
    return _normalize_str(queues.get(process)) or default_queue
def _expand_param_grid(param_grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    def _to_value_list(value: Any) -> list[Any]:
        if value is None:
            return []
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_list(value):
                return [v for v in value]
        except ImportError:
            pass
        return list(value) if isinstance(value, (list, tuple, set)) else [value]
    grid_items = [(str(key), _to_value_list(raw_values)) for (key, raw_values) in param_grid.items()]
    if any((not values) for (_, values) in grid_items):
        return []
    keys = [key for (key, _) in grid_items]
    return [dict(zip(keys, values)) for values in product(*[values for (_, values) in grid_items])]
def _build_hpo_trials(model_variant: str, param_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for params in param_sets:
        if not params:
            trials.append({'params': {}, 'hpo_run_id': None, 'suffix': None})
            continue
        signature = '' if not params else '__'.join([f"{key}={(f'{params[key]:g}' if isinstance(params[key], float) else str(params[key]))}" for key in sorted(params)])
        base_id = _sanitize_component(f'{model_variant}__{signature}' if signature else str(model_variant))
        count = seen.get(base_id, 0) + 1
        seen[base_id] = count
        hpo_run_id = base_id if count == 1 else _sanitize_component(f'{base_id}__{count}')
        suffix = _sanitize_component(signature or 'trial')
        if count > 1:
            suffix = _sanitize_component(f'{suffix}__{count}')
        trials.append({'params': params, 'hpo_run_id': hpo_run_id, 'suffix': suffix})
    return trials
def _limit_hpo_trials(trials_by_model: Mapping[str, list[dict[str, Any]]], max_hpo_trials: int) -> tuple[dict[str, list[dict[str, Any]]], int]:
    if max_hpo_trials <= 0:
        return ({str(k): list(v) for (k, v) in trials_by_model.items()}, 0)
    limited: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for (model_variant, trials) in trials_by_model.items():
        if len(trials) > max_hpo_trials:
            limited[model_variant] = list(trials[:max_hpo_trials])
            skipped += len(trials) - max_hpo_trials
        else:
            limited[model_variant] = list(trials)
    return (limited, skipped)
def _build_train_plan(preprocess_variants: list[str], model_variants: list[str], trials_by_model: Mapping[str, list[dict[str, Any]]], *, max_jobs: int, max_hpo_trials: int) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[dict[str, Any]]]]:
    raw_trial_count = sum((len(trials_by_model.get(model, [])) for model in model_variants))
    raw_jobs = len(preprocess_variants) * raw_trial_count if preprocess_variants else 0
    (limited_trials, _) = _limit_hpo_trials(trials_by_model, max_hpo_trials)
    jobs: list[dict[str, Any]] = []
    for preprocess_variant in preprocess_variants:
        for model_variant in model_variants:
            for trial in limited_trials.get(model_variant, [{'params': {}, 'hpo_run_id': None, 'suffix': None}]):
                jobs.append({'preprocess_variant': preprocess_variant, 'model_variant': model_variant, 'trial': trial})
    planned_jobs = len(jobs)
    if max_jobs > 0 and planned_jobs > max_jobs:
        jobs = jobs[:max_jobs]
    planned_jobs = len(jobs)
    skipped_due_to_policy = raw_jobs - planned_jobs
    if skipped_due_to_policy < 0:
        skipped_due_to_policy = 0
    info = {'raw_jobs': raw_jobs, 'planned_jobs': planned_jobs, 'skipped_due_to_policy': skipped_due_to_policy}
    return (jobs, info, limited_trials)
def _resolve_hpo_trials(cfg: Any, model_variants: list[str]) -> tuple[bool, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    enabled = bool(_cfg_value(cfg, 'pipeline.hpo.enabled'))
    params_cfg = _to_mapping(_cfg_value(cfg, 'pipeline.hpo.params'))
    trials_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_variant in model_variants:
        model_params = _to_mapping(params_cfg.get(model_variant)) if enabled else {}
        if enabled and model_params:
            param_sets = _expand_param_grid(model_params)
            if not param_sets:
                raise ValueError(f'pipeline.hpo.params.{model_variant} is empty.')
        else:
            param_sets = [{}]
        trials_by_model[model_variant] = _build_hpo_trials(model_variant, param_sets)
    return (enabled, trials_by_model, params_cfg)
def _build_hpo_param_overrides(params: Mapping[str, Any]) -> dict[str, Any]:
    return {f'group.model.model_variant.params.{key}': value for (key, value) in params.items()}
def _should_inject_task_uv_overrides(cfg: Any) -> bool:
    if _to_list(_cfg_value(cfg, 'run.clearml.env.uv.extras')):
        return False
    return not bool(_cfg_value(cfg, 'run.clearml.env.uv.all_extras', False))
def _build_task_uv_overrides(cfg: Any, *, task_name: str, model_variant: str | None=None, infer_mode: str | None=None) -> dict[str, Any]:
    if not _should_inject_task_uv_overrides(cfg):
        return {}
    extras = resolve_required_uv_extras(task_name=task_name, model_variant_name=model_variant, infer_mode=infer_mode)
    return {'run.clearml.env.uv.extras': extras, 'run.clearml.env.uv.all_extras': False}
def _ensure_override(overrides: list[str], key: str, value: Any) -> None:
    if value is None or key in {str(item).strip().split('=', 1)[0].strip().lstrip('+~') for item in overrides}:
        return
    formatted = _format_value(value)
    if formatted is None:
        return
    overrides.append(f'{key}={formatted}')
def _hydra_task_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig
        overrides = getattr(getattr(HydraConfig.get(), 'overrides', None), 'task', None)
        return [str(item) for item in overrides if item] if overrides else []
    except (ImportError, AttributeError, TypeError, ValueError):
        return []
def _merge_overrides(*items: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for item in items for (key, value) in dict(item).items()}
def _set_first_present_override(overrides: dict[str, Any], candidates: list[tuple[str, Any]], *, error_message: str) -> dict[str, Any]:
    for (key, value) in candidates:
        if value:
            overrides[key] = value
            return overrides
    raise ValueError(error_message)
def _apply_infer_reference_overrides(overrides: dict[str, Any], *, recommendation_payload: Mapping[str, Any] | None, explicit_model_id: str | None, explicit_train_task_id: str | None, error_message: str) -> dict[str, Any]:
    preferred = resolve_preferred_infer_reference(recommendation_payload)
    explicit = build_infer_reference(model_id=explicit_model_id, train_task_id=explicit_train_task_id)
    infer_model_id = preferred.get('infer_model_id') or explicit.get('infer_model_id')
    infer_train_task_id = None if infer_model_id else (preferred.get('infer_train_task_id') or explicit.get('infer_train_task_id'))
    if infer_model_id:
        overrides['infer.model_id'] = infer_model_id
        overrides.pop('infer.train_task_id', None)
        return overrides
    if infer_train_task_id:
        overrides['infer.train_task_id'] = infer_train_task_id
        overrides.pop('infer.model_id', None)
        return overrides
    raise ValueError(error_message)
def _collect_run_overrides(cfg: Any, grid_run_id: str, *, child_execution: str | None=None) -> dict[str, Any]:
    run_cfg = getattr(cfg, 'run', None)
    overrides: dict[str, Any] = {'run.grid_run_id': grid_run_id}
    if run_cfg is None:
        return overrides
    overrides['run.usecase_id'] = getattr(run_cfg, 'usecase_id', None)
    overrides['run.schema_version'] = getattr(run_cfg, 'schema_version', None)
    overrides['run.retrain_run_id'] = getattr(run_cfg, 'retrain_run_id', None)
    policy_cfg = getattr(run_cfg, 'usecase_id_policy', None)
    if getattr(policy_cfg, 'name', None):
        overrides['ops/usecase_id_policy'] = getattr(policy_cfg, 'name')
    clearml_cfg = getattr(run_cfg, 'clearml', None)
    if clearml_cfg is not None:
        overrides['run.clearml.enabled'] = bool(getattr(clearml_cfg, 'enabled', False))
        execution = child_execution if child_execution is not None else getattr(clearml_cfg, 'execution', None)
        overrides['run.clearml.execution'] = execution
        overrides['run.clearml.project_root'] = getattr(clearml_cfg, 'project_root', None)
        overrides['run.clearml.queue_name'] = getattr(clearml_cfg, 'queue_name', None)
        overrides['run.clearml.clone_from_task_id'] = getattr(clearml_cfg, 'clone_from_task_id', None)
        env_cfg = getattr(clearml_cfg, 'env', None)
        if env_cfg is not None:
            apt_packages = _to_list(getattr(env_cfg, 'apt_packages', None))
            if apt_packages:
                overrides['run.clearml.env.apt_packages'] = apt_packages
            apt_update = getattr(env_cfg, 'apt_update', None)
            if apt_update is not None:
                overrides['run.clearml.env.apt_update'] = apt_update
            apt_allow_local = getattr(env_cfg, 'apt_allow_local', None)
            if apt_allow_local is not None:
                overrides['run.clearml.env.apt_allow_local'] = apt_allow_local
            bootstrap = getattr(env_cfg, 'bootstrap', None)
            if bootstrap and str(bootstrap).lower() not in {'none', 'false', '0', 'off'}:
                overrides['run.clearml.env.bootstrap'] = bootstrap
                uv_cfg = getattr(env_cfg, 'uv', None)
                if uv_cfg is not None:
                    overrides['run.clearml.env.uv.venv_dir'] = getattr(uv_cfg, 'venv_dir', None)
                    overrides['run.clearml.env.uv.extras'] = getattr(uv_cfg, 'extras', None)
                    overrides['run.clearml.env.uv.all_extras'] = getattr(uv_cfg, 'all_extras', None)
                    overrides['run.clearml.env.uv.frozen'] = getattr(uv_cfg, 'frozen', None)
        extra_tags = getattr(clearml_cfg, 'extra_tags', None)
        if extra_tags:
            overrides['run.clearml.extra_tags'] = list(extra_tags)
        clearml_policy = getattr(clearml_cfg, 'policy', None)
        if getattr(clearml_policy, 'name', None):
            overrides['ops/clearml_policy'] = getattr(clearml_policy, 'name')
    return overrides
def _collect_data_overrides(cfg: Any) -> dict[str, Any]:
    data_cfg = getattr(cfg, 'data', None)
    overrides: dict[str, Any] = {}
    if data_cfg is None:
        return overrides
    for key in ('dataset_path', 'raw_dataset_id', 'processed_dataset_id', 'target_column'):
        value = getattr(data_cfg, key, None)
        if value is not None:
            overrides[f'data.{key}'] = value
    id_columns = getattr(data_cfg, 'id_columns', None)
    if id_columns:
        overrides['data.id_columns'] = list(id_columns)
    drop_columns = getattr(data_cfg, 'drop_columns', None)
    if drop_columns:
        overrides['data.drop_columns'] = list(drop_columns)
    split_cfg = getattr(data_cfg, 'split', None)
    if split_cfg is not None:
        for key in ('strategy', 'test_size', 'seed', 'group_column', 'time_column'):
            value = getattr(split_cfg, key, None)
            if value is not None:
                overrides[f'data.split.{key}'] = value
    return overrides
def _build_downstream_data_overrides(data_overrides: Mapping[str, Any], *, raw_dataset_id: str | None) -> dict[str, Any]:
    overrides = dict(data_overrides)
    if not raw_dataset_id:
        return overrides
    overrides['data.raw_dataset_id'] = raw_dataset_id
    dataset_path_value = _normalize_str(overrides.get('data.dataset_path'))
    if raw_dataset_id.startswith('local:'):
        if not dataset_path_value:
            raise ValueError('data.dataset_path is required when data.raw_dataset_id is local.')
    else:
        overrides.pop('data.dataset_path', None)
    return overrides
def _collect_eval_overrides(cfg: Any) -> dict[str, Any]:
    eval_cfg = getattr(cfg, 'eval', None)
    overrides: dict[str, Any] = {}
    if eval_cfg is None:
        return overrides
    for key in ('primary_metric', 'direction', 'cv_folds', 'seed', 'task_type'):
        value = getattr(eval_cfg, key, None)
        if value is not None:
            overrides[f'eval.{key}'] = value
    classification_cfg = getattr(eval_cfg, 'classification', None)
    if classification_cfg is not None:
        for key in ('mode', 'top_k'):
            value = getattr(classification_cfg, key, None)
            if value is not None:
                overrides[f'eval.classification.{key}'] = value
    metrics_cfg = getattr(eval_cfg, 'metrics', None)
    if metrics_cfg is not None:
        value = getattr(metrics_cfg, 'classification_multiclass', None)
        if value is not None:
            overrides['eval.metrics.classification_multiclass'] = list(value)
    selection = _resolve_exec_policy_selection(cfg)
    if selection:
        _apply_exec_policy_selection(overrides, selection)
    return overrides
def _resolve_ensemble_methods(cfg: Any) -> list[str]:
    methods = [_normalize_str(item) for item in _to_list(_cfg_value(cfg, 'ensemble.methods'))]
    methods = [item for item in methods if item]
    if methods:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in methods:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
    method = _normalize_str(_cfg_value(cfg, 'ensemble.method')) or 'mean_topk'
    return [method]
def _collect_ensemble_overrides(cfg: Any) -> dict[str, Any]:
    ensemble_cfg = getattr(cfg, 'ensemble', None)
    overrides: dict[str, Any] = {}
    if ensemble_cfg is None:
        return overrides
    for key in ('top_k', 'selection_metric', 'exclude_variants'):
        value = getattr(ensemble_cfg, key, None)
        if value is not None:
            overrides[f'ensemble.{key}'] = list(value) if key == 'exclude_variants' else value
    weighted_cfg = getattr(ensemble_cfg, 'weighted', None)
    if weighted_cfg is not None:
        for key in ('search', 'n_samples', 'seed', 'top_k_max'):
            value = getattr(weighted_cfg, key, None)
            if value is not None:
                overrides[f'ensemble.weighted.{key}'] = value
    stacking_cfg = getattr(ensemble_cfg, 'stacking', None)
    if stacking_cfg is not None:
        for key in ('meta_model', 'cv_folds', 'seed', 'require_test_split'):
            value = getattr(stacking_cfg, key, None)
            if value is not None:
                overrides[f'ensemble.stacking.{key}'] = value
    return overrides
def _build_run_root(base_output_dir: Path, grid_run_id: str, name: str) -> Path:
    return base_output_dir / 'grid' / str(grid_run_id) / _sanitize_component(name)
def _stage_dir(run_root: Path, task_name: str) -> Path:
    return run_root / _STAGE_BY_TASK[task_name]
def _clearml_project(cfg: Any, stage: str) -> str:
    return build_project_name(_normalize_str(_cfg_value(cfg, 'run.clearml.project_root')) or 'MFG', _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown', stage, cfg=cfg)
def _resolve_base_task_id(cfg: Any, task_name: str, *, use_templates: bool) -> str:
    if use_templates:
        return resolve_template_task_id(cfg, task_name)
    project = _clearml_project(cfg, _STAGE_BY_TASK[task_name])
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError('clearml is required to resolve base tasks for pipeline steps.') from exc
    task = ClearMLTask.get_task(project_name=project, task_name=task_name, allow_archived=True)
    if not task or not getattr(task, 'id', None):
        raise RuntimeError(f'Base task not found for {task_name} in {project}')
    return str(task.id)
def _make_base_task_factory(base_task_id: str, *, project_name: str):
    try:
        from clearml import Task as ClearMLTask
        from clearml.backend_interface.util import get_or_create_project
    except ImportError as exc:
        raise RuntimeError('clearml is required to clone base tasks for pipeline steps.') from exc
    project_id = None
    try:
        project_id = get_or_create_project(session=ClearMLTask._get_default_session(), project_name=str(project_name))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        project_id = None
    if not project_id:
        try:
            project_id = ClearMLTask.get_project_id(str(project_name))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            project_id = None
    def _factory(node: Any):
        name = getattr(node, 'name', None) or 'pipeline_step'
        if project_id:
            task = ClearMLTask.clone(base_task_id, name=str(name), project=project_id)
        else:
            task = ClearMLTask.clone(base_task_id, name=str(name))
        try:
            task.set_project(project_name=str(project_name))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            task_id = getattr(task, 'id', None) or getattr(task, 'task_id', None)
            if task_id:
                update_clearml_task_tags(str(task_id), remove=['template:true', 'template:deprecated'])
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return task
    return _factory
def _load_model_set_payload(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError('OmegaConf is required to load model_set configs.') from exc
    try:
        cfg = OmegaConf.load(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f'Failed to load model_set config: {path}') from exc
    try:
        return OmegaConf.to_container(cfg, resolve=False)
    except (AttributeError, TypeError, ValueError):
        return cfg
def _normalize_model_set_name(value: str) -> str:
    name = value.strip()
    if name and Path(name).name != name:
        raise ValueError(f'Invalid pipeline.model_set name: {value}')
    return name
def _resolve_model_set_variants(model_set: str) -> list[str]:
    name = _normalize_model_set_name(model_set)
    if not name:
        return []
    repo_root = _resolve_repo_root()
    path = repo_root / 'conf' / 'pipeline' / 'model_sets' / f'{name}.yaml'
    if not path.exists():
        raise ValueError(f"pipeline.model_set '{name}' not found: {path}")
    payload = _load_model_set_payload(path)
    variants: list[str] = []
    if isinstance(payload, Mapping):
        variants = _to_list(payload.get('variants'))
        auto = bool(payload.get('auto'))
        task_type = _normalize_str(payload.get('task_type'))
        if auto and (not task_type):
            raise ValueError(f"model_set '{name}' requires task_type when auto=true.")
        if auto or (task_type and (not variants)):
            from ..registry.models import list_model_variants
            variants = list_model_variants(task_type=task_type)
        exclude = set(_to_list(payload.get('exclude')))
        if exclude:
            variants = [item for item in variants if item not in exclude]
    elif isinstance(payload, list):
        variants = _to_list(payload)
    else:
        raise ValueError(f'model_set config must be mapping or list: {path}')
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        name = _normalize_str(item)
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped
def _resolve_variants(cfg: Any) -> tuple[list[str], list[str]]:
    preprocess_variants = _to_list(_cfg_value(cfg, 'pipeline.preprocess_variants'))
    if not preprocess_variants:
        single = _normalize_str(_cfg_value(cfg, 'pipeline.preprocess_variant'))
        if single:
            preprocess_variants = [single]
    if not preprocess_variants:
        preprocess_variants = _to_list(_cfg_value(cfg, 'pipeline.grid.preprocess_variants'))
    if not preprocess_variants:
        fallback = _normalize_str(_cfg_value(cfg, 'preprocess_variant.name')) or _normalize_str(_cfg_value(cfg, 'group.preprocess.preprocess_variant.name'))
        if fallback:
            preprocess_variants = [fallback]
    model_set = _normalize_str(_cfg_value(cfg, 'pipeline.model_set'))
    if model_set:
        model_variants = _resolve_model_set_variants(model_set)
    else:
        model_variants = _to_list(_cfg_value(cfg, 'pipeline.model_variants'))
        if not model_variants:
            model_variants = _to_list(_cfg_value(cfg, 'pipeline.grid.model_variants'))
    if not model_variants:
        fallback = _normalize_str(_cfg_value(cfg, 'model_variant.name')) or _normalize_str(_cfg_value(cfg, 'group.model.model_variant.name'))
        if fallback:
            model_variants = [fallback]
    return (preprocess_variants, model_variants)
def _build_plan_steps(cfg: Any, *, base_output_dir: Path, grid_run_id: str, run_dataset_register: bool, run_preprocess: bool, run_train: bool, run_train_ensemble: bool, run_leaderboard: bool, run_infer: bool, preprocess_targets: list[str], train_jobs: list[dict[str, Any]], base_extra_tags: list[str], max_models: int, queues: Mapping[str, Any], ensemble_methods: list[str], ensemble_overrides: Mapping[str, Any]) -> dict[str, Any]:
    dataset_step = None
    preprocess_steps: list[dict[str, Any]] = []
    preprocess_by_variant: dict[str, dict[str, Any]] = {}
    train_steps: list[dict[str, Any]] = []
    train_ensemble_steps: list[dict[str, Any]] = []
    leaderboard_step = None
    infer_step = None
    def _step(*, step_name: str, task_name: str, run_root: Path, parents: list[str], queue: str | None, overrides: dict[str, Any], **extras: Any) -> dict[str, Any]:
        payload_overrides = dict(overrides)
        if queue:
            payload_overrides['run.clearml.queue_name'] = queue
        payload = {'step_name': step_name, 'task_name': task_name, 'run_root': run_root, 'run_dir': _stage_dir(run_root, task_name), 'parents': parents, 'queue': queue, 'overrides': payload_overrides}
        if extras:
            payload.update(extras)
        return payload
    if run_dataset_register:
        run_root = _build_run_root(base_output_dir, grid_run_id, 'dataset_register')
        step_queue = _select_queue(queues, 'dataset_register')
        overrides = {'run.output_dir': str(run_root)}
        overrides.update(_build_task_uv_overrides(cfg, task_name='dataset_register'))
        dataset_step = _step(step_name='dataset_register', task_name='dataset_register', run_root=run_root, parents=[], queue=step_queue, overrides=overrides)
    if run_preprocess:
        if not preprocess_targets:
            raise ValueError('pipeline.grid.preprocess_variants is empty.')
        for preprocess_variant in preprocess_targets:
            step_name = f'preprocess__{_sanitize_component(preprocess_variant)}'
            run_root = _build_run_root(base_output_dir, grid_run_id, f'preprocess__{preprocess_variant}')
            parents = [dataset_step['step_name']] if dataset_step else []
            step_queue = _select_queue(queues, 'preprocess')
            overrides = {'group/preprocess': preprocess_variant, 'run.output_dir': str(run_root)}
            overrides.update(_build_task_uv_overrides(cfg, task_name='preprocess'))
            step = _step(step_name=step_name, task_name='preprocess', run_root=run_root, parents=parents, queue=step_queue, overrides=overrides, preprocess_variant=preprocess_variant)
            preprocess_steps.append(step)
            preprocess_by_variant[preprocess_variant] = step
    if run_train:
        if not preprocess_by_variant:
            raise ValueError('preprocess outputs are required before train.')
        for job in train_jobs:
            preprocess_variant = str(job.get('preprocess_variant') or '')
            payload = preprocess_by_variant.get(preprocess_variant)
            if payload is None:
                raise ValueError(f'preprocess step missing for {preprocess_variant}.')
            preprocess_run_dir = payload['run_dir']
            model_variant = str(job.get('model_variant') or '')
            trial = job.get('trial') or {}
            hpo_params = trial.get('params') or {}
            hpo_run_id = trial.get('hpo_run_id')
            suffix = trial.get('suffix')
            step_name = f'train__{_sanitize_component(preprocess_variant)}__{_sanitize_component(model_variant)}'
            run_name = f'train__{preprocess_variant}__{model_variant}'
            if suffix:
                step_name = f'{step_name}__{_sanitize_component(suffix)}'
                run_name = f'{run_name}__{suffix}'
            run_root = _build_run_root(base_output_dir, grid_run_id, run_name)
            overrides = {'group/model': model_variant, '+preprocess.variant': preprocess_variant, 'train.inputs.preprocess_run_dir': str(preprocess_run_dir), 'run.output_dir': str(run_root)}
            overrides.update(_build_hpo_param_overrides(hpo_params))
            overrides.update(_build_task_uv_overrides(cfg, task_name='train_model', model_variant=model_variant))
            extra_tag = f'grid_cell:{preprocess_variant}__{model_variant}'
            extra_tags = list(base_extra_tags) if extra_tag in base_extra_tags else [*base_extra_tags, extra_tag]
            if hpo_run_id:
                hpo_tag = f'hpo:{hpo_run_id}'
                extra_tags = list(extra_tags) if hpo_tag in extra_tags else [*extra_tags, hpo_tag]
            if extra_tags:
                overrides['run.clearml.extra_tags'] = extra_tags
            step_queue = _select_queue(queues, 'train_model', model_variant=model_variant)
            train_steps.append(_step(step_name=step_name, task_name='train_model', run_root=run_root, parents=[payload['step_name']], queue=step_queue, overrides=overrides, preprocess_variant=preprocess_variant, model_variant=model_variant, hpo_run_id=hpo_run_id, hpo_params=hpo_params or None))
    if run_train_ensemble:
        if not preprocess_by_variant:
            raise ValueError('preprocess outputs are required before train_ensemble.')
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for step in train_steps:
            variant = str(step.get('preprocess_variant') or '')
            if not variant:
                continue
            by_variant.setdefault(variant, []).append(step)
        methods = ensemble_methods or ['mean_topk']
        for (preprocess_variant, parent_steps) in by_variant.items():
            for method in methods:
                method_key = _sanitize_component(method)
                step_name = f'ensemble__{_sanitize_component(preprocess_variant)}__{method_key}'
                run_root = _build_run_root(base_output_dir, grid_run_id, f'ensemble__{preprocess_variant}__{method}')
                overrides = {'run.output_dir': str(run_root), '+preprocess.variant': preprocess_variant, 'group/preprocess': preprocess_variant, 'ensemble.enabled': True, 'ensemble.method': method}
                overrides.update(ensemble_overrides)
                overrides.update(_build_task_uv_overrides(cfg, task_name='train_ensemble'))
                step_queue = _select_queue(queues, 'train_ensemble')
                train_ensemble_steps.append(_step(step_name=step_name, task_name='train_ensemble', run_root=run_root, parents=[step['step_name'] for step in parent_steps], queue=step_queue, overrides=overrides, preprocess_variant=preprocess_variant))
    if run_leaderboard:
        run_root = _build_run_root(base_output_dir, grid_run_id, 'leaderboard')
        overrides = {'run.output_dir': str(run_root)}
        if max_models > 0:
            overrides['leaderboard.top_k'] = max_models
        overrides.update(_build_task_uv_overrides(cfg, task_name='leaderboard'))
        step_queue = _select_queue(queues, 'leaderboard')
        leaderboard_step = _step(step_name='leaderboard', task_name='leaderboard', run_root=run_root, parents=[*[step['step_name'] for step in train_steps], *[step['step_name'] for step in train_ensemble_steps]], queue=step_queue, overrides=overrides)
    if run_infer:
        infer_cfg = getattr(cfg, 'infer', None)
        infer_mode = _normalize_str(getattr(infer_cfg, 'mode', None)) or 'single'
        run_root = _build_run_root(base_output_dir, grid_run_id, 'infer')
        step_queue = _select_queue(queues, 'infer')
        parents = [leaderboard_step['step_name']] if leaderboard_step else []
        overrides = {'infer.mode': infer_mode, 'run.output_dir': str(run_root)}
        overrides.update(_build_task_uv_overrides(cfg, task_name='infer', infer_mode=infer_mode))
        infer_step = _step(step_name='infer', task_name='infer', run_root=run_root, parents=parents, queue=step_queue, overrides=overrides)
    return {'dataset_register': dataset_step, 'preprocess': preprocess_steps, 'train': train_steps, 'train_ensemble': train_ensemble_steps, 'leaderboard': leaderboard_step, 'infer': infer_step}
def _build_pipeline_plan(cfg: Any, grid_run_id: str, *, child_execution: str | None=None) -> dict[str, Any]:
    base_output_dir = Path(getattr(cfg.run, 'output_dir', 'outputs')).expanduser().resolve()
    pipeline_cfg = getattr(cfg, 'pipeline', None)
    run_dataset_register = bool(getattr(pipeline_cfg, 'run_dataset_register', False))
    run_preprocess = bool(getattr(pipeline_cfg, 'run_preprocess', True))
    run_train = bool(getattr(pipeline_cfg, 'run_train', True))
    run_train_ensemble = bool(getattr(pipeline_cfg, 'run_train_ensemble', _cfg_value(cfg, 'ensemble.enabled', False)))
    run_leaderboard = bool(getattr(pipeline_cfg, 'run_leaderboard', True))
    run_infer = bool(getattr(pipeline_cfg, 'run_infer', False))
    raw_dataset_id = _normalize_str(_cfg_value(cfg, 'data.raw_dataset_id'))
    dataset_path_value = _normalize_str(_cfg_value(cfg, 'data.dataset_path'))
    if run_preprocess and (not run_dataset_register) and (not raw_dataset_id):
        if dataset_path_value:
            raw_dataset_id = derive_local_raw_dataset_id(dataset_path_value)
            _set_cfg_value(cfg, 'data.raw_dataset_id', raw_dataset_id)
        else:
            raise ValueError('data.raw_dataset_id is required when pipeline.run_dataset_register is false.')
    if run_preprocess and raw_dataset_id and raw_dataset_id.startswith('local:') and (not dataset_path_value):
        raise ValueError('data.dataset_path is required when data.raw_dataset_id is local.')
    model_set = _normalize_str(_cfg_value(cfg, 'pipeline.model_set'))
    (preprocess_variants, model_variants) = _resolve_variants(cfg)
    (hpo_enabled, hpo_trials_by_model, hpo_params_cfg) = _resolve_hpo_trials(cfg, model_variants)
    base_extra_tags = _to_list(_cfg_value(cfg, 'run.clearml.extra_tags'))
    limits = _resolve_exec_policy_limits(cfg)
    max_jobs = limits['max_jobs']
    max_models = limits['max_models']
    max_hpo_trials = limits['max_hpo_trials']
    plan_only = bool(_cfg_value(cfg, 'pipeline.plan_only')) or bool(_cfg_value(cfg, 'pipeline.dry_run')) or bool(_cfg_value(cfg, 'pipeline.plan'))
    train_jobs: list[dict[str, Any]] = []
    plan_info = {'raw_jobs': 0, 'planned_jobs': 0, 'skipped_due_to_policy': 0}
    if run_train:
        if not preprocess_variants:
            raise ValueError('pipeline.grid.preprocess_variants is empty.')
        if not model_variants:
            raise ValueError('pipeline.grid.model_variants is empty.')
        (train_jobs, plan_info, _) = _build_train_plan(preprocess_variants, model_variants, hpo_trials_by_model, max_jobs=max_jobs, max_hpo_trials=max_hpo_trials)
    if run_preprocess and (not preprocess_variants):
        raise ValueError('pipeline.grid.preprocess_variants is empty.')
    if run_train and (not run_preprocess):
        raise ValueError('pipeline.run_preprocess=false cannot be combined with run_train=true.')
    if run_train_ensemble and (not run_train):
        raise ValueError('pipeline.run_train_ensemble=true requires run_train=true.')
    run_overrides = _collect_run_overrides(cfg, grid_run_id, child_execution=child_execution)
    data_overrides = _collect_data_overrides(cfg)
    downstream_data_overrides = _build_downstream_data_overrides(data_overrides, raw_dataset_id=raw_dataset_id) if run_preprocess else dict(data_overrides)
    eval_overrides = _collect_eval_overrides(cfg)
    ensemble_methods = _resolve_ensemble_methods(cfg)
    ensemble_overrides = _collect_ensemble_overrides(cfg)
    preprocess_targets = preprocess_variants
    if run_train:
        preprocess_targets = []
        seen_preprocess: set[str] = set()
        for job in train_jobs:
            variant = str(job.get('preprocess_variant') or '')
            if not variant or variant in seen_preprocess:
                continue
            preprocess_targets.append(variant)
            seen_preprocess.add(variant)
    queues = _resolve_exec_policy_queues(cfg)
    steps = _build_plan_steps(cfg, base_output_dir=base_output_dir, grid_run_id=grid_run_id, run_dataset_register=run_dataset_register, run_preprocess=run_preprocess, run_train=run_train, run_train_ensemble=run_train_ensemble, run_leaderboard=run_leaderboard, run_infer=run_infer, preprocess_targets=preprocess_targets, train_jobs=train_jobs, base_extra_tags=base_extra_tags, max_models=max_models, queues=queues, ensemble_methods=ensemble_methods, ensemble_overrides=ensemble_overrides)
    return {'base_output_dir': base_output_dir, 'run_dataset_register': run_dataset_register, 'run_preprocess': run_preprocess, 'run_train': run_train, 'run_train_ensemble': run_train_ensemble, 'run_leaderboard': run_leaderboard, 'run_infer': run_infer, 'model_set': model_set, 'preprocess_variants': preprocess_variants, 'model_variants': model_variants, 'hpo_enabled': hpo_enabled, 'hpo_params_cfg': hpo_params_cfg, 'limits': limits, 'max_jobs': max_jobs, 'max_models': max_models, 'max_hpo_trials': max_hpo_trials, 'plan_only': plan_only, 'plan_info': plan_info, 'train_jobs': train_jobs, 'preprocess_targets': preprocess_targets, 'run_overrides': run_overrides, 'data_overrides': data_overrides, 'downstream_data_overrides': downstream_data_overrides, 'eval_overrides': eval_overrides, 'base_extra_tags': base_extra_tags, 'queues': queues, 'steps': steps}
def _collect_step_task_ids(controller: Any) -> dict[str, str]:
    getter = getattr(controller, 'get_processed_nodes', None)
    nodes = getter() if callable(getter) else {}
    payload: dict[str, str] = {}
    for (name, node) in dict(nodes).items():
        task_id = getattr(node, 'executed', None)
        if not task_id and getattr(node, 'job', None):
            job = node.job
            if hasattr(job, 'task_id'):
                task_id = job.task_id() if callable(job.task_id) else job.task_id
        if task_id:
            payload[str(name)] = str(task_id)
    return payload
def _build_ref(*, run_dir: Path | None=None, task_id: str | None=None, **extras: Any) -> dict[str, Any]:
    ref: dict[str, Any] = {}
    if task_id:
        ref['task_id'] = str(task_id)
    if run_dir is not None:
        ref['run_dir'] = str(run_dir)
    for (key, value) in extras.items():
        if value is not None:
            ref[key] = value
    return ref
def _add_pipeline_step(controller: Any, *, execution_queue: str | None=None, **kwargs: Any) -> None:
    add_step = getattr(controller, 'add_step', None)
    if not callable(add_step):
        raise AttributeError('Pipeline controller does not support add_step.')
    try:
        signature = inspect.signature(add_step)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        if execution_queue and 'execution_queue' in params:
            kwargs['execution_queue'] = execution_queue
        base_task_factory = kwargs.get('base_task_factory')
        if base_task_factory and 'base_task_factory' not in params:
            kwargs.pop('base_task_factory', None)
            try:
                node_name = kwargs.get('name') or 'pipeline_step'
                class _Node:
                    def __init__(self, name: str):
                        self.name = name
                clone_task = base_task_factory(_Node(str(node_name)))
                task_id = getattr(clone_task, 'id', None)
                if task_id and 'base_task_id' in params:
                    kwargs['base_task_id'] = str(task_id)
                if task_id and 'clone_base_task' in params:
                    kwargs['clone_base_task'] = False
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        kwargs = {key: value for (key, value) in kwargs.items() if key in params}
    elif execution_queue:
        kwargs['execution_queue'] = execution_queue
    add_step(**kwargs)
def _build_leaderboard_step_overrides(*, clearml_enabled: bool, run_overrides: Mapping[str, Any], eval_overrides: Mapping[str, Any], step_overrides: Mapping[str, Any], train_refs: list[dict[str, Any]], train_ensemble_refs: list[dict[str, Any]]) -> dict[str, Any]:
    overrides = _merge_overrides(run_overrides, eval_overrides, step_overrides)
    if clearml_enabled:
        train_task_ids = [ref.get('train_task_id') for ref in train_refs if ref.get('train_task_id')]
        train_task_ids.extend([ref.get('train_task_id') for ref in train_ensemble_refs if ref.get('train_task_id')])
        if not train_task_ids:
            raise ValueError('train_task_id is missing in train outputs (ClearML mode).')
        overrides['leaderboard.train_task_ids'] = train_task_ids
        return overrides
    train_run_dirs = [ref.get('run_dir') for ref in train_refs if ref.get('run_dir')]
    train_run_dirs.extend([ref.get('run_dir') for ref in train_ensemble_refs if ref.get('run_dir')])
    overrides['leaderboard.train_run_dirs'] = train_run_dirs
    return overrides
def _build_infer_step_overrides(*, cfg: Any, clearml_enabled: bool, run_overrides: Mapping[str, Any], step_overrides: Mapping[str, Any], leaderboard_out: Mapping[str, Any] | None) -> dict[str, Any]:
    infer_cfg = getattr(cfg, 'infer', None)
    overrides = _merge_overrides(run_overrides, step_overrides)
    explicit_model_id = _normalize_str(getattr(infer_cfg, 'model_id', None))
    explicit_train_task_id = _normalize_str(getattr(infer_cfg, 'train_task_id', None))
    return _apply_infer_reference_overrides(overrides, recommendation_payload=leaderboard_out, explicit_model_id=explicit_model_id, explicit_train_task_id=explicit_train_task_id, error_message='infer requires model_id or train_task_id.')
def _build_local_pipeline_run_summary(*, cfg: Any, plan: Mapping[str, Any], grid_run_id: str, dataset_register_ref: dict[str, Any] | None, preprocess_refs: list[dict[str, Any]], train_refs: list[dict[str, Any]], train_ensemble_refs: list[dict[str, Any]], leaderboard_ref: dict[str, Any] | None, infer_ref: dict[str, Any] | None, executed_jobs: int) -> dict[str, Any]:
    return {'grid_run_id': grid_run_id, 'plan_only': plan['plan_only'], 'planned_jobs': int(plan['plan_info'].get('planned_jobs', 0)), 'executed_jobs': int(executed_jobs), 'skipped_due_to_policy': int(plan['plan_info'].get('skipped_due_to_policy', 0)), 'dataset_register_ref': dataset_register_ref, 'preprocess_ref': preprocess_refs, 'train_refs': train_refs, 'train_ensemble_refs': train_ensemble_refs, 'leaderboard_ref': leaderboard_ref, 'infer_ref': infer_ref, 'grid': {'preprocess_variants': plan['preprocess_variants'], 'model_variants': plan['model_variants'], 'max_jobs': plan['max_jobs'], 'max_hpo_trials': plan['max_hpo_trials'], 'hpo': {'enabled': plan['hpo_enabled'], 'params': plan['hpo_params_cfg']}}, 'policy': {'limits': dict(plan['limits']), 'selection': _resolve_exec_policy_selection(cfg)}}
def _run_local_pipeline_impl(cfg: Any, grid_run_id: str, *, clearml_enabled: bool) -> dict[str, Any]:
    plan = _build_pipeline_plan(cfg, grid_run_id)
    repo_root = _resolve_repo_root()
    config_dir = repo_root / 'conf'
    run_overrides = dict(plan['run_overrides'])
    data_overrides = dict(plan['data_overrides'])
    downstream_data_overrides = dict(plan['downstream_data_overrides'])
    eval_overrides = plan['eval_overrides']
    steps = plan['steps']
    dataset_register_ref: dict[str, Any] | None = None
    preprocess_refs: list[dict[str, Any]] = []
    train_refs: list[dict[str, Any]] = []
    train_ensemble_refs: list[dict[str, Any]] = []
    leaderboard_ref: dict[str, Any] | None = None
    infer_ref: dict[str, Any] | None = None
    executed_jobs = 0
    leaderboard_out: dict[str, Any] | None = None
    if not plan['plan_only']:
        if plan['run_dataset_register']:
            step = steps['dataset_register']
            if step is None:
                raise ValueError('dataset_register step is missing.')
            overrides = _merge_overrides(run_overrides, data_overrides, step['overrides'])
            args = ['task=dataset_register', *_overrides_to_args(overrides)]
            _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
            dataset_register_ref = _build_ref(run_dir=step['run_dir'])
            dataset_out = _load_json(step['run_dir'] / 'out.json')
            raw_dataset_id = _normalize_str(dataset_out.get('raw_dataset_id'))
            if raw_dataset_id:
                data_overrides['data.raw_dataset_id'] = raw_dataset_id
                if plan['run_preprocess']:
                    downstream_data_overrides = _build_downstream_data_overrides(data_overrides, raw_dataset_id=raw_dataset_id)
        preprocess_outputs: dict[str, dict[str, Any]] = {}
        if plan['run_preprocess']:
            for step in steps['preprocess']:
                preprocess_variant = step.get('preprocess_variant')
                overrides = _merge_overrides(run_overrides, downstream_data_overrides, step['overrides'])
                args = ['task=preprocess', *_overrides_to_args(overrides)]
                _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
                out = _load_json(step['run_dir'] / 'out.json')
                preprocess_refs.append(_build_ref(run_dir=step['run_dir'], preprocess_variant=preprocess_variant, processed_dataset_id=out.get('processed_dataset_id'), split_hash=out.get('split_hash'), recipe_hash=out.get('recipe_hash')))
                preprocess_outputs[str(preprocess_variant)] = {'run_dir': step['run_dir'], 'out': out}
        if plan['run_train']:
            if not preprocess_outputs:
                raise ValueError('preprocess outputs are required before train.')
            for step in steps['train']:
                preprocess_variant = step.get('preprocess_variant')
                payload = preprocess_outputs.get(str(preprocess_variant))
                if payload is None:
                    raise ValueError(f'preprocess output missing for {preprocess_variant}.')
                preprocess_out = payload['out']
                processed_dataset_id = _normalize_str(preprocess_out.get('processed_dataset_id'))
                overrides = _merge_overrides(run_overrides, downstream_data_overrides, eval_overrides, step['overrides'])
                if processed_dataset_id:
                    overrides['data.processed_dataset_id'] = processed_dataset_id
                args = ['task=train_model', *_overrides_to_args(overrides)]
                _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
                out = _load_json(step['run_dir'] / 'out.json')
                train_refs.append(_build_ref(run_dir=step['run_dir'], preprocess_variant=preprocess_variant, model_variant=step.get('model_variant'), train_task_id=out.get('train_task_id'), model_id=out.get('model_id'), best_score=out.get('best_score'), primary_metric=out.get('primary_metric'), hpo_run_id=step.get('hpo_run_id'), hpo_params=step.get('hpo_params')))
            executed_jobs = len(train_refs)
        if plan['run_train_ensemble']:
            if not train_refs:
                raise ValueError('train outputs are required before train_ensemble.')
            for step in steps['train_ensemble']:
                overrides = _merge_overrides(run_overrides, downstream_data_overrides, eval_overrides, step['overrides'])
                args = ['task=train_ensemble', *_overrides_to_args(overrides)]
                _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
                out = _load_json(step['run_dir'] / 'out.json')
                train_ensemble_refs.append(_build_ref(run_dir=step['run_dir'], preprocess_variant=step.get('preprocess_variant'), train_task_id=out.get('train_task_id'), model_id=out.get('model_id'), best_score=out.get('best_score'), primary_metric=out.get('primary_metric'), task_type=out.get('task_type')))
        if plan['run_leaderboard']:
            if not train_refs and (not train_ensemble_refs):
                raise ValueError('train outputs are required before leaderboard.')
            step = steps['leaderboard']
            if step is None:
                raise ValueError('leaderboard step is missing.')
            overrides = _build_leaderboard_step_overrides(clearml_enabled=clearml_enabled, run_overrides=run_overrides, eval_overrides=eval_overrides, step_overrides=step['overrides'], train_refs=train_refs, train_ensemble_refs=train_ensemble_refs)
            args = ['task=leaderboard', *_overrides_to_args(overrides)]
            _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
            leaderboard_out = _load_json(step['run_dir'] / 'out.json')
            leaderboard_ref = _build_ref(run_dir=step['run_dir'])
        if plan['run_infer']:
            step = steps['infer']
            if step is None:
                raise ValueError('infer step is missing.')
            overrides = _build_infer_step_overrides(cfg=cfg, clearml_enabled=clearml_enabled, run_overrides=run_overrides, step_overrides=step['overrides'], leaderboard_out=leaderboard_out)
            args = ['task=infer', *_overrides_to_args(overrides)]
            _run_cli_task(args, cwd=repo_root, config_dir=config_dir)
            infer_ref = _build_ref(run_dir=step['run_dir'])
    return _build_local_pipeline_run_summary(cfg=cfg, plan=plan, grid_run_id=grid_run_id, dataset_register_ref=dataset_register_ref, preprocess_refs=preprocess_refs, train_refs=train_refs, train_ensemble_refs=train_ensemble_refs, leaderboard_ref=leaderboard_ref, infer_ref=infer_ref, executed_jobs=executed_jobs)
def _resolve_pipeline_queue_name(queues: Mapping[str, Any]) -> str | None:
    queue_name = next((value for value in [_select_queue(queues, 'pipeline'), _select_queue(queues, 'dataset_register'), _select_queue(queues, 'preprocess'), _select_queue(queues, 'train_model'), _select_queue(queues, 'train_ensemble'), _normalize_str(queues.get('train_model_heavy')), _select_queue(queues, 'leaderboard'), _select_queue(queues, 'infer')] if value), None)
    if queue_name:
        return queue_name
    return next((_normalize_str(value) for value in (queues.get('model_variants') or {}).values() if _normalize_str(value)), None)
def _configure_clearml_pipeline_controller(*, cfg: Any, plan: Mapping[str, Any], run_overrides: Mapping[str, Any], grid_run_id: str, queue_name: str | None, controller_execution: str) -> Any:
    pipeline_queue = _select_queue(plan['queues'], 'pipeline')
    pipeline_name = _normalize_str(_cfg_value(cfg, 'run.clearml.task_name')) or 'pipeline'
    metadata = resolve_clearml_metadata(cfg, stage=getattr(getattr(cfg, 'task', None), 'stage', '99_pipeline'), task_name='pipeline', clearml_enabled=True)
    controller = create_pipeline_controller(cfg, name=pipeline_name, tags=metadata.get('tags'), properties=metadata.get('user_properties'), default_queue=pipeline_queue)
    controller_overrides = _hydra_task_overrides()
    if controller_overrides:
        _ensure_override(controller_overrides, 'task', 'pipeline')
        _ensure_override(controller_overrides, 'run.grid_run_id', grid_run_id)
        _ensure_override(controller_overrides, 'run.output_dir', _cfg_value(cfg, 'run.output_dir'))
        _ensure_override(controller_overrides, 'run.clearml.enabled', True)
        _ensure_override(controller_overrides, 'run.clearml.execution', controller_execution or _cfg_value(cfg, 'run.clearml.execution'))
        _ensure_override(controller_overrides, 'pipeline.run_dataset_register', plan.get('run_dataset_register'))
        _ensure_override(controller_overrides, 'pipeline.run_preprocess', plan.get('run_preprocess'))
        _ensure_override(controller_overrides, 'pipeline.run_train', plan.get('run_train'))
        _ensure_override(controller_overrides, 'pipeline.run_train_ensemble', plan.get('run_train_ensemble'))
        _ensure_override(controller_overrides, 'pipeline.run_leaderboard', plan.get('run_leaderboard'))
        _ensure_override(controller_overrides, 'pipeline.run_infer', plan.get('run_infer'))
        _ensure_override(controller_overrides, 'pipeline.grid.preprocess_variants', plan.get('preprocess_variants'))
        _ensure_override(controller_overrides, 'pipeline.grid.model_variants', plan.get('model_variants'))
        _ensure_override(controller_overrides, 'data.dataset_path', _cfg_value(cfg, 'data.dataset_path'))
        _ensure_override(controller_overrides, 'data.target_column', _cfg_value(cfg, 'data.target_column'))
        _ensure_override(controller_overrides, 'data.raw_dataset_id', _cfg_value(cfg, 'data.raw_dataset_id'))
        _ensure_override(controller_overrides, 'run.usecase_id', _cfg_value(cfg, 'run.usecase_id'))
    else:
        controller_overrides_map = _merge_overrides(_collect_run_overrides(cfg, grid_run_id, child_execution=None), _collect_data_overrides(cfg), _collect_eval_overrides(cfg), {'task': 'pipeline', 'run.output_dir': _cfg_value(cfg, 'run.output_dir'), 'pipeline.run_dataset_register': plan.get('run_dataset_register'), 'pipeline.run_preprocess': plan.get('run_preprocess'), 'pipeline.run_train': plan.get('run_train'), 'pipeline.run_train_ensemble': plan.get('run_train_ensemble'), 'pipeline.run_leaderboard': plan.get('run_leaderboard'), 'pipeline.run_infer': plan.get('run_infer'), 'pipeline.grid.preprocess_variants': plan.get('preprocess_variants'), 'pipeline.grid.model_variants': plan.get('model_variants')})
        controller_overrides = _overrides_to_args(controller_overrides_map)
    if controller_overrides:
        apply_clearml_task_overrides(controller, controller_overrides)
    pipeline_require_clearml_agent(queue_name)
    return controller


def _make_pipeline_controller_base_task_resolver(*, cfg: Any, use_templates: bool):
    template_task_ids: dict[str, str] = {}

    def _base_task_kwargs(task_name: str) -> dict[str, Any]:
        task_id = template_task_ids.get(task_name)
        if not task_id:
            task_id = _resolve_base_task_id(cfg, task_name, use_templates=use_templates)
            template_task_ids[task_name] = task_id
        project_name = _clearml_project(cfg, _STAGE_BY_TASK[task_name])
        return {
            'base_task_id': task_id,
            'base_task_factory': _make_base_task_factory(task_id, project_name=project_name),
        }

    return _base_task_kwargs


def _iter_pipeline_controller_step_requests(*, cfg: Any, plan: Mapping[str, Any], steps: Mapping[str, Any]):
    run_overrides = plan['run_overrides']
    data_overrides = plan['data_overrides']
    downstream_data_overrides = plan['downstream_data_overrides']
    eval_overrides = plan['eval_overrides']
    if plan['run_dataset_register']:
        step = steps['dataset_register']
        if step is None:
            raise ValueError('dataset_register step is missing.')
        yield (step, _merge_overrides(run_overrides, data_overrides, step['overrides']))
    if plan['run_preprocess']:
        for step in steps['preprocess']:
            yield (step, _merge_overrides(run_overrides, downstream_data_overrides, step['overrides']))
    if plan['run_train']:
        if not steps['train']:
            raise ValueError('preprocess outputs are required before train.')
        for step in steps['train']:
            overrides = _merge_overrides(run_overrides, downstream_data_overrides, eval_overrides, step['overrides'])
            parents = step.get('parents') or []
            if parents:
                overrides['train.inputs.preprocess_task_id'] = pipeline_step_task_id_ref(str(parents[0]))
            yield (step, overrides)
    if plan['run_train_ensemble']:
        if not steps['train']:
            raise ValueError('train outputs are required before train_ensemble.')
        for step in steps['train_ensemble']:
            yield (step, _merge_overrides(run_overrides, downstream_data_overrides, eval_overrides, step['overrides']))
    if plan['run_leaderboard']:
        if not steps['train'] and (not steps['train_ensemble']):
            raise ValueError('train outputs are required before leaderboard.')
        step = steps['leaderboard']
        if step is None:
            raise ValueError('leaderboard step is missing.')
        train_task_refs = [pipeline_step_task_id_ref(train_step['step_name']) for train_step in steps['train']]
        train_task_refs.extend([pipeline_step_task_id_ref(train_step['step_name']) for train_step in steps['train_ensemble']])
        yield (step, _merge_overrides(run_overrides, eval_overrides, {'leaderboard.train_task_ids': train_task_refs}, step['overrides']))
    if plan['run_infer']:
        step = steps['infer']
        if step is None:
            raise ValueError('infer step is missing.')
        infer_cfg = getattr(cfg, 'infer', None)
        overrides = _merge_overrides(run_overrides, step['overrides'])
        _apply_infer_reference_overrides(
            overrides,
            recommendation_payload=None,
            explicit_model_id=_normalize_str(getattr(infer_cfg, 'model_id', None)),
            explicit_train_task_id=_normalize_str(getattr(infer_cfg, 'train_task_id', None)),
            error_message='infer requires model_id or train_task_id for remote execution.',
        )
        yield (step, overrides)


def _add_pipeline_controller_step_entries(
    *,
    cfg: Any,
    controller: Any,
    use_templates: bool,
    step_requests: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    parameter_override_builder,
) -> None:
    base_task_kwargs = _make_pipeline_controller_base_task_resolver(cfg=cfg, use_templates=use_templates)
    for (step, overrides) in step_requests:
        _add_pipeline_step(
            controller,
            name=step['step_name'],
            parents=step['parents'],
            parameter_override=parameter_override_builder(overrides),
            clone_base_task=True,
            cache_executed_step=False,
            execution_queue=step['queue'],
            **base_task_kwargs(step['task_name']),
        )


def _add_clearml_pipeline_steps(*, cfg: Any, plan: Mapping[str, Any], steps: Mapping[str, Any], controller: Any, use_templates: bool, run_overrides: Mapping[str, Any], data_overrides: Mapping[str, Any], downstream_data_overrides: Mapping[str, Any], eval_overrides: Mapping[str, Any]) -> None:
    _ = (run_overrides, data_overrides, downstream_data_overrides, eval_overrides)
    step_requests = list(_iter_pipeline_controller_step_requests(cfg=cfg, plan=plan, steps=steps))
    _add_pipeline_controller_step_entries(
        cfg=cfg,
        controller=controller,
        use_templates=use_templates,
        step_requests=step_requests,
        parameter_override_builder=lambda overrides: {f'Args/{k}': v for (k, v) in _overrides_to_params(overrides).items()},
    )
def _reset_pipeline_controller_definition(controller: Any) -> None:
    # Controllers reconstructed from Task objects only deserialize the DAG payload.
    # Re-seed the runtime defaults that ClearML's draft serialization expects.
    class_defaults = {
        '_nodes': {},
        '_running_nodes': [],
        '_start_time': None,
        '_pipeline_time_limit': None,
        '_default_execution_queue': None,
        '_always_create_from_code': True,
        '_version': getattr(controller, '_version', None) or getattr(type(controller), '_default_pipeline_version', '1.0.0'),
        '_pool_frequency': 12.0,
        '_thread': None,
        '_pipeline_args': {},
        '_pipeline_args_desc': {},
        '_pipeline_args_type': {},
        '_args_map': {},
        '_stop_event': None,
        '_experiment_created_cb': None,
        '_experiment_completed_cb': None,
        '_pre_step_callbacks': {},
        '_post_step_callbacks': {},
        '_target_project': True,
        '_add_pipeline_tags': False,
        '_reporting_lock': RLock(),
        '_pipeline_task_status_failed': None,
        '_mock_execution': False,
        '_last_progress_update_time': 0,
        '_artifact_serialization_function': None,
        '_artifact_deserialization_function': None,
        '_skip_global_imports': False,
        '_enable_local_imports': True,
        '_monitored_nodes': {},
        '_abort_running_steps_on_failure': False,
        '_def_max_retry_on_failure': 0,
        '_output_uri': None,
    }
    for (attr, default) in class_defaults.items():
        if attr == '_version' and getattr(controller, attr, None):
            continue
        setattr(controller, attr, default)
    setattr(controller, '_step_ref_pattern', re.compile(getattr(type(controller), '_step_pattern', r'\${[^}]*}')))
    setattr(controller, '_auto_connect_task', bool(getattr(controller, '_task', None)))
    setattr(
        controller,
        '_retry_on_failure_callback',
        getattr(type(controller), '_default_retry_on_failure_callback', None),
    )


def _seed_pipeline_template_parameters(controller: Any, defaults: Mapping[str, Any]) -> None:
    adder = getattr(controller, 'add_parameter', None)
    if not callable(adder):
        raise AttributeError('Pipeline controller does not support add_parameter.')
    for (key, value) in sorted(_template_pipeline_params(defaults).items()):
        adder(str(key), default=_normalize_template_arg_value(value))


def _add_clearml_pipeline_template_steps(*, cfg: Any, plan: Mapping[str, Any], steps: Mapping[str, Any], controller: Any, use_templates: bool, shared_defaults: Mapping[str, Any]) -> None:
    step_requests = list(_iter_pipeline_controller_step_requests(cfg=cfg, plan=plan, steps=steps))
    _add_pipeline_controller_step_entries(
        cfg=cfg,
        controller=controller,
        use_templates=use_templates,
        step_requests=step_requests,
        parameter_override_builder=lambda overrides: {f'Args/{k}': v for (k, v) in _overrides_to_params(_build_template_step_overrides(overrides, shared_defaults)).items()},
    )


def build_pipeline_template_draft(*, cfg: Any, controller: Any, pipeline_profile: str) -> dict[str, Any]:
    template_grid_run_id = f'template__{normalize_pipeline_profile(pipeline_profile)}'
    plan = _build_pipeline_plan(cfg, template_grid_run_id, child_execution='logging')
    shared_defaults = _build_pipeline_template_runtime_defaults(cfg=cfg, plan=plan, grid_run_id=template_grid_run_id, pipeline_profile=pipeline_profile)
    _reset_pipeline_controller_definition(controller)
    setattr(
        controller,
        '_default_execution_queue',
        _normalize_str(plan.get('queues', {}).get('default')) or _normalize_str(_cfg_value(cfg, 'run.clearml.queue_name')) or 'default',
    )
    _seed_pipeline_template_parameters(controller, shared_defaults)
    _add_clearml_pipeline_template_steps(cfg=cfg, plan=plan, steps=plan['steps'], controller=controller, use_templates=True, shared_defaults=shared_defaults)
    create_draft = getattr(controller, 'create_draft', None)
    if not callable(create_draft):
        raise AttributeError('Pipeline controller does not support create_draft.')
    create_draft()
    return {'plan': plan, 'shared_defaults': shared_defaults}
def _resolve_pipeline_template_profile(cfg: Any, plan: Mapping[str, Any]) -> str:
    explicit = _normalize_str(_cfg_value(cfg, 'pipeline.profile'))
    if explicit:
        return normalize_pipeline_profile(explicit)
    profile = _detect_pipeline_template_profile(plan)
    if profile:
        return profile
    raise ValueError(
        'No built-in visible pipeline template matches the current pipeline settings. '
        'Specify run.clearml.pipeline.template_task_id or align the config to a supported profile.'
    )


def _apply_pipeline_run_task_identity(*, task_id: str, cfg: Any, pipeline_profile: str, metadata: Mapping[str, Any]) -> None:
    template_context = resolve_template_context(cfg)
    remove_tags = [*build_template_tags('pipeline', context=template_context), 'task_kind:template']
    run_tags = _build_pipeline_template_tags(pipeline_profile, cfg=cfg, task_kind='run')
    update_clearml_task_tags(task_id, add=[*(metadata.get('tags') or []), *run_tags], remove=remove_tags)
    ensure_clearml_task_tags(task_id, ['pipeline'])
    ensure_clearml_task_properties(task_id, {**dict(metadata.get('user_properties') or {}), **_build_pipeline_template_properties(pipeline_profile, task_kind='run')})


def _build_pipeline_launch_summary(*, cfg: Any, plan: Mapping[str, Any], grid_run_id: str, controller_task_id: str, template_task_id: str, pipeline_profile: str, queue_name: str | None) -> dict[str, Any]:
    summary = _build_local_pipeline_run_summary(cfg=cfg, plan=plan, grid_run_id=grid_run_id, dataset_register_ref=None, preprocess_refs=[], train_refs=[], train_ensemble_refs=[], leaderboard_ref=None, infer_ref=None, executed_jobs=0)
    summary['status'] = 'queued'
    summary['pipeline_task_id'] = controller_task_id
    summary['template_task_id'] = template_task_id
    summary['pipeline_profile'] = pipeline_profile
    if queue_name:
        summary['controller_queue'] = queue_name
    return summary


def _resolve_visible_pipeline_run_contract(*, cfg: Any, grid_run_id: str) -> _VisiblePipelineRunContract:
    plan = _build_pipeline_plan(cfg, grid_run_id, child_execution='logging')
    pipeline_profile = _resolve_pipeline_template_profile(cfg, plan)
    metadata = resolve_clearml_metadata(
        cfg,
        stage=getattr(getattr(cfg, 'task', None), 'stage', '99_pipeline'),
        task_name='pipeline',
        clearml_enabled=True,
    )
    queue_name = _select_queue(plan['queues'], 'pipeline') or _normalize_str(_cfg_value(cfg, 'run.clearml.queue_name')) or 'services'
    return _VisiblePipelineRunContract(
        plan=plan,
        pipeline_profile=pipeline_profile,
        metadata=metadata,
        queue_name=queue_name,
    )


def _apply_visible_pipeline_run_defaults(*, target: Any, task_id: str, cfg: Any, contract: _VisiblePipelineRunContract, grid_run_id: str) -> dict[str, Any]:
    runtime_defaults = _build_pipeline_template_runtime_defaults(
        cfg=cfg,
        plan=contract.plan,
        grid_run_id=grid_run_id,
        pipeline_profile=contract.pipeline_profile,
        pipeline_task_id=task_id,
    )
    apply_clearml_task_overrides(target, _overrides_to_args(runtime_defaults))
    set_clearml_task_parameters(task_id, _template_pipeline_params(runtime_defaults), section='pipeline')
    _apply_pipeline_run_task_identity(
        task_id=task_id,
        cfg=cfg,
        pipeline_profile=contract.pipeline_profile,
        metadata=contract.metadata,
    )
    return runtime_defaults


def _enqueue_pipeline_template_run(*, cfg: Any, grid_run_id: str) -> dict[str, Any]:
    contract = _resolve_visible_pipeline_run_contract(cfg=cfg, grid_run_id=grid_run_id)
    template_task_id = resolve_pipeline_template_task_id(cfg, pipeline_profile=contract.pipeline_profile)
    controller_name = _normalize_str(_cfg_value(cfg, 'run.clearml.task_name')) or f'pipeline__{grid_run_id}'
    controller_project = contract.metadata.get('project_name') or _normalize_str(_cfg_value(cfg, 'run.clearml.project_name'))
    controller = clone_pipeline_controller(source_task_id=template_task_id, task_name=controller_name, project_name=controller_project)
    controller_task_id = clearml_task_id(controller)
    if not controller_task_id:
        raise RuntimeError('Cloned pipeline controller task id is missing.')
    _apply_visible_pipeline_run_defaults(target=controller, task_id=controller_task_id, cfg=cfg, contract=contract, grid_run_id=grid_run_id)
    enqueue_pipeline_controller(controller, contract.queue_name)
    return _build_pipeline_launch_summary(
        cfg=cfg,
        plan=contract.plan,
        grid_run_id=grid_run_id,
        controller_task_id=controller_task_id,
        template_task_id=template_task_id,
        pipeline_profile=contract.pipeline_profile,
        queue_name=contract.queue_name,
    )


def _execute_current_pipeline_controller(*, cfg: Any, ctx: TaskContext, grid_run_id: str) -> dict[str, Any]:
    if ctx.task is None:
        raise RuntimeError('Current pipeline controller task is missing.')
    contract = _resolve_visible_pipeline_run_contract(cfg=cfg, grid_run_id=grid_run_id)
    pipeline_task_id = clearml_task_id(ctx.task)
    if pipeline_task_id:
        _apply_visible_pipeline_run_defaults(target=ctx.task, task_id=pipeline_task_id, cfg=cfg, contract=contract, grid_run_id=grid_run_id)
    controller = load_pipeline_controller_from_task(source_task=ctx.task)
    if not getattr(controller, '_nodes', None):
        raise RuntimeError(
            'Current ClearML task does not contain a serialized pipeline graph. '
            'Clone a visible pipeline template task from the Pipelines tab or run manage_templates.py --apply.'
        )
    starter = getattr(controller, 'start', None)
    if not callable(starter):
        raise AttributeError('Pipeline controller does not support start().')
    starter(queue=contract.queue_name)
    step_task_ids = _collect_step_task_ids(controller)
    executed_jobs = len(contract.plan['steps']['train'])
    (dataset_register_ref, preprocess_refs, train_refs, train_ensemble_refs, leaderboard_ref, infer_ref) = _build_clearml_pipeline_refs(plan_only=contract.plan['plan_only'], steps=contract.plan['steps'], step_task_ids=step_task_ids)
    summary = _build_local_pipeline_run_summary(cfg=cfg, plan=contract.plan, grid_run_id=grid_run_id, dataset_register_ref=dataset_register_ref, preprocess_refs=preprocess_refs, train_refs=train_refs, train_ensemble_refs=train_ensemble_refs, leaderboard_ref=leaderboard_ref, infer_ref=infer_ref, executed_jobs=executed_jobs)
    summary['status'] = 'completed'
    summary['pipeline_task_id'] = pipeline_task_id
    summary['pipeline_profile'] = contract.pipeline_profile
    return summary
def _build_clearml_pipeline_refs(*, plan_only: bool, steps: Mapping[str, Any], step_task_ids: Mapping[str, str]) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
    dataset_register_ref = None
    preprocess_refs: list[dict[str, Any]] = []
    train_refs: list[dict[str, Any]] = []
    train_ensemble_refs: list[dict[str, Any]] = []
    leaderboard_ref = None
    infer_ref = None
    if plan_only:
        return (dataset_register_ref, preprocess_refs, train_refs, train_ensemble_refs, leaderboard_ref, infer_ref)
    if steps['dataset_register'] is not None:
        dataset_step = steps['dataset_register']
        dataset_register_ref = _build_ref(run_dir=dataset_step['run_dir'], task_id=step_task_ids.get(dataset_step['step_name']))
    for step in steps['preprocess']:
        preprocess_refs.append(_build_ref(run_dir=step['run_dir'], task_id=step_task_ids.get(step['step_name']), preprocess_variant=step.get('preprocess_variant')))
    for step in steps['train']:
        train_refs.append(_build_ref(run_dir=step['run_dir'], task_id=step_task_ids.get(step['step_name']), preprocess_variant=step.get('preprocess_variant'), model_variant=step.get('model_variant'), hpo_run_id=step.get('hpo_run_id'), hpo_params=step.get('hpo_params')))
    for step in steps['train_ensemble']:
        train_ensemble_refs.append(_build_ref(run_dir=step['run_dir'], task_id=step_task_ids.get(step['step_name']), preprocess_variant=step.get('preprocess_variant')))
    if steps['leaderboard'] is not None:
        leaderboard_step = steps['leaderboard']
        leaderboard_ref = _build_ref(run_dir=leaderboard_step['run_dir'], task_id=step_task_ids.get(leaderboard_step['step_name']))
    if steps['infer'] is not None:
        infer_step = steps['infer']
        infer_ref = _build_ref(run_dir=infer_step['run_dir'], task_id=step_task_ids.get(infer_step['step_name']))
    return (dataset_register_ref, preprocess_refs, train_refs, train_ensemble_refs, leaderboard_ref, infer_ref)
def _run_clearml_pipeline_impl(cfg: Any, grid_run_id: str, *, use_templates: bool, controller_execution: str | None=None, pipeline_task_id: str | None=None) -> dict[str, Any]:
    child_execution = 'logging' if use_templates else None
    controller_execution = _normalize_str(controller_execution) or ''
    run_controller_locally = controller_execution != 'pipeline_controller'
    plan = _build_pipeline_plan(cfg, grid_run_id, child_execution=child_execution)
    run_overrides = plan['run_overrides']
    if pipeline_task_id:
        run_overrides['run.clearml.pipeline_task_id'] = pipeline_task_id
    data_overrides = plan['data_overrides']
    downstream_data_overrides = plan['downstream_data_overrides']
    eval_overrides = plan['eval_overrides']
    queues = plan['queues']
    steps = plan['steps']
    step_task_ids: dict[str, str] = {}
    executed_jobs = 0
    if not plan['plan_only']:
        queue_name = _resolve_pipeline_queue_name(queues)
        controller = _configure_clearml_pipeline_controller(cfg=cfg, plan=plan, run_overrides=run_overrides, grid_run_id=grid_run_id, queue_name=queue_name, controller_execution=controller_execution)
        _add_clearml_pipeline_steps(cfg=cfg, plan=plan, steps=steps, controller=controller, use_templates=use_templates, run_overrides=run_overrides, data_overrides=data_overrides, downstream_data_overrides=downstream_data_overrides, eval_overrides=eval_overrides)
        if run_controller_locally:
            starter = getattr(controller, 'start_locally', None)
            if not callable(starter):
                raise AttributeError('Pipeline controller does not support start_locally.')
            starter(run_pipeline_steps_locally=False)
        else:
            starter = getattr(controller, 'start', None)
            if not callable(starter):
                raise AttributeError('Pipeline controller does not support start.')
            if queue_name:
                starter(queue=queue_name)
            else:
                starter()
        step_task_ids = _collect_step_task_ids(controller)
        executed_jobs = len(steps['train'])
    (dataset_register_ref, preprocess_refs, train_refs, train_ensemble_refs, leaderboard_ref, infer_ref) = _build_clearml_pipeline_refs(plan_only=plan['plan_only'], steps=steps, step_task_ids=step_task_ids)
    return _build_local_pipeline_run_summary(cfg=cfg, plan=plan, grid_run_id=grid_run_id, dataset_register_ref=dataset_register_ref, preprocess_refs=preprocess_refs, train_refs=train_refs, train_ensemble_refs=train_ensemble_refs, leaderboard_ref=leaderboard_ref, infer_ref=infer_ref, executed_jobs=executed_jobs)
def _create_pipeline_controller_runtime_context(cfg: Any) -> TaskContext:
    stage = getattr(getattr(cfg, 'task', None), 'stage', '99_pipeline')
    if not _running_inside_clearml_task():
        return _create_local_task_context(cfg, stage=stage, task_name='pipeline')
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError('clearml is required to attach to the current pipeline controller task.') from exc
    task = ClearMLTask.current_task()
    if task is None:
        return _create_local_task_context(cfg, stage=stage, task_name='pipeline')
    output_dir = Path(getattr(cfg.run, 'output_dir', 'outputs')) / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    project_name = _normalize_str(getattr(task, 'get_project_name', lambda: None)() if hasattr(task, 'get_project_name') else None) or _normalize_str(getattr(task, 'project_name', None)) or _normalize_str(_cfg_value(cfg, 'run.clearml.project_name')) or _normalize_str(_cfg_value(cfg, 'task.project_name')) or f"MFG/TabularAnalysis/{_normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'}/{stage}"
    task_name = _normalize_str(getattr(task, 'name', None)) or _normalize_str(_cfg_value(cfg, 'run.clearml.task_name')) or 'pipeline'
    ctx = TaskContext(task=task, project_name=project_name, task_name=task_name, output_dir=output_dir)
    save_config_resolved(ctx, cfg)
    return ctx
def run(cfg: Any) -> None:
    grid_run_id = _normalize_str(_cfg_value(cfg, 'run.grid_run_id'))
    if not grid_run_id:
        grid_run_id = uuid.uuid4().hex
        _set_cfg_value(cfg, 'run.grid_run_id', grid_run_id)
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    execution = _normalize_str(_cfg_value(cfg, 'run.clearml.execution')) or 'local'
    controller_execution = execution == 'pipeline_controller'
    clearml_enabled = is_clearml_enabled(cfg)
    if clearml_enabled and execution == 'pipeline_controller':
        ctx = _create_pipeline_controller_runtime_context(cfg)
    else:
        task_type = clearml_task_type_controller() if controller_execution else None
        system_tags = ['pipeline'] if controller_execution else None
        ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='pipeline', tags=identity.tags, properties=identity.user_properties, task_type=task_type, system_tags=system_tags)
    connect_pipeline(ctx, cfg, grid_run_id=grid_run_id, run_dataset_register=bool(getattr(getattr(cfg, 'pipeline', None), 'run_dataset_register', False)), run_preprocess=bool(getattr(getattr(cfg, 'pipeline', None), 'run_preprocess', True)), run_train=bool(getattr(getattr(cfg, 'pipeline', None), 'run_train', True)), run_train_ensemble=bool(getattr(getattr(cfg, 'pipeline', None), 'run_train_ensemble', _cfg_value(cfg, 'ensemble.enabled', False))), run_leaderboard=bool(getattr(getattr(cfg, 'pipeline', None), 'run_leaderboard', True)), run_infer=bool(getattr(getattr(cfg, 'pipeline', None), 'run_infer', False)), plan_only=bool(_cfg_value(cfg, 'pipeline.plan_only')) or bool(_cfg_value(cfg, 'pipeline.dry_run')) or bool(_cfg_value(cfg, 'pipeline.plan')))
    pipeline_task_id = clearml_task_id(ctx.task) if ctx.task is not None else None
    if clearml_enabled and execution == 'pipeline_controller':
        if ctx.task is None:
            pipeline_run = _enqueue_pipeline_template_run(cfg=cfg, grid_run_id=grid_run_id)
        else:
            pipeline_run = _execute_current_pipeline_controller(cfg=cfg, ctx=ctx, grid_run_id=grid_run_id)
        pipeline_task_id = _normalize_str(pipeline_run.get('pipeline_task_id')) or pipeline_task_id
    elif clearml_enabled and execution in ('agent', 'clone'):
        pipeline_run = _run_clearml_pipeline_impl(cfg, grid_run_id, use_templates=False, controller_execution=execution, pipeline_task_id=pipeline_task_id)
    else:
        pipeline_run = _run_local_pipeline_impl(cfg, grid_run_id, clearml_enabled=clearml_enabled)
    pipeline_run_path = ctx.output_dir / 'pipeline_run.json'
    pipeline_run_path.write_text(json.dumps(pipeline_run, ensure_ascii=False, indent=2), encoding='utf-8')
    if ctx.task is not None:
        upload_artifact(ctx, 'pipeline_run.json', pipeline_run_path)
        num_models = int(pipeline_run.get('planned_jobs') or 0)
        num_succeeded = int(pipeline_run.get('executed_jobs') or 0)
        num_failed = max(0, num_models - num_succeeded)
        log_scalar(ctx.task, 'pipeline', 'num_models', num_models, step=0)
        log_scalar(ctx.task, 'pipeline', 'num_succeeded', num_succeeded, step=0)
        log_scalar(ctx.task, 'pipeline', 'num_failed', num_failed, step=0)
    report_path = ctx.output_dir / 'report.md'
    limits = _resolve_exec_policy_limits(cfg)
    report_max_models = limits['max_models'] if limits['max_models'] > 0 else 5
    report_bundle = build_pipeline_report_bundle(pipeline_run, cfg=cfg, max_models=report_max_models, pipeline_run_dir=ctx.output_dir, pipeline_task_id=pipeline_task_id)
    report_path.write_text(report_bundle.markdown, encoding='utf-8')
    report_json_path = ctx.output_dir / 'report.json'
    report_json_path.write_text(json.dumps(report_bundle.payload, ensure_ascii=False, indent=2), encoding='utf-8')
    report_links_path = ctx.output_dir / 'report_links.json'
    report_links_path.write_text(json.dumps(report_bundle.links, ensure_ascii=False, indent=2), encoding='utf-8')
    if ctx.task is not None:
        upload_artifact(ctx, 'report.md', report_path)
        upload_artifact(ctx, 'report.json', report_json_path)
        upload_artifact(ctx, 'report_links.json', report_links_path)
        report_markdown(ctx, title='Pipeline Report', markdown=report_bundle.markdown)
    out = {'pipeline_run': pipeline_run}
    inputs = {'run_dataset_register': bool(getattr(getattr(cfg, 'pipeline', None), 'run_dataset_register', False)), 'run_preprocess': bool(getattr(getattr(cfg, 'pipeline', None), 'run_preprocess', True)), 'run_train': bool(getattr(getattr(cfg, 'pipeline', None), 'run_train', True)), 'run_leaderboard': bool(getattr(getattr(cfg, 'pipeline', None), 'run_leaderboard', True)), 'run_infer': bool(getattr(getattr(cfg, 'pipeline', None), 'run_infer', False)), 'plan_only': bool(_cfg_value(cfg, 'pipeline.plan_only')) or bool(_cfg_value(cfg, 'pipeline.dry_run')) or bool(_cfg_value(cfg, 'pipeline.plan'))}
    outputs = {'grid_run_id': grid_run_id, 'pipeline_run_path': str(pipeline_run_path)}
    emit_outputs_and_manifest(ctx, cfg, process='pipeline', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ('split', {}), 'recipe_hash': ('recipe', {})}, clearml_enabled=clearml_enabled)
