from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.collection_utils import to_list as _to_list


@dataclass(frozen=True)
class PipelineProfileSpec:
    name: str
    run_dataset_register: bool
    run_preprocess: bool
    run_train: bool
    run_train_ensemble: bool
    run_leaderboard: bool
    run_infer: bool
    model_set: str
    ensemble_methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineStepSpec:
    step_name: str
    task_name: str
    parents: tuple[str, ...]
    queue: str | None
    run_dir: str
    preprocess_variant: str | None = None
    model_variant: str | None = None
    ensemble_method: str | None = None
    hpo_run_id: str | None = None


@dataclass(frozen=True)
class PipelinePlan:
    profile: str
    preprocess_variants: tuple[str, ...]
    model_variants: tuple[str, ...]
    steps: tuple[PipelineStepSpec, ...]
    queues: dict[str, Any]
    run_dataset_register: bool
    run_preprocess: bool
    run_train: bool
    run_train_ensemble: bool
    run_leaderboard: bool
    run_infer: bool


DEFAULT_PIPELINE_PROFILE = 'pipeline'

PIPELINE_PROFILE_SPECS: dict[str, PipelineProfileSpec] = {
    'pipeline': PipelineProfileSpec(
        name='pipeline',
        run_dataset_register=False,
        run_preprocess=True,
        run_train=True,
        run_train_ensemble=False,
        run_leaderboard=True,
        run_infer=False,
        model_set='regression_all',
        ensemble_methods=(),
    ),
    'train_model_full': PipelineProfileSpec(
        name='train_model_full',
        run_dataset_register=False,
        run_preprocess=True,
        run_train=True,
        run_train_ensemble=False,
        run_leaderboard=False,
        run_infer=False,
        model_set='regression_all',
        ensemble_methods=(),
    ),
    'train_ensemble_full': PipelineProfileSpec(
        name='train_ensemble_full',
        run_dataset_register=False,
        run_preprocess=True,
        run_train=True,
        run_train_ensemble=True,
        run_leaderboard=True,
        run_infer=False,
        model_set='regression_all',
        ensemble_methods=('mean_topk', 'weighted', 'stacking'),
    ),
}

PIPELINE_TEMPLATE_UI_WHITELIST: dict[str, tuple[str, ...]] = {
    'pipeline': (
        'run.usecase_id',
        'data.raw_dataset_id',
        'pipeline.selection.enabled_preprocess_variants',
        'pipeline.selection.enabled_model_variants',
    ),
    'train_model_full': (
        'run.usecase_id',
        'data.raw_dataset_id',
        'pipeline.selection.enabled_preprocess_variants',
        'pipeline.selection.enabled_model_variants',
    ),
    'train_ensemble_full': (
        'run.usecase_id',
        'data.raw_dataset_id',
        'pipeline.selection.enabled_preprocess_variants',
        'pipeline.selection.enabled_model_variants',
        'ensemble.selection.enabled_methods',
        'ensemble.top_k',
    ),
}

PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS = frozenset({'run.output_dir', 'train.inputs.preprocess_run_dir'})


def normalize_pipeline_profile(value: Any, *, default: str = DEFAULT_PIPELINE_PROFILE) -> str:
    text = _normalize_str(value)
    return text or default


def get_pipeline_profile_spec(pipeline_profile: str) -> PipelineProfileSpec:
    profile = normalize_pipeline_profile(pipeline_profile)
    spec = PIPELINE_PROFILE_SPECS.get(profile)
    if spec is None:
        raise ValueError(f'Unsupported pipeline profile: {profile}')
    return spec


def is_pipeline_template_name(value: Any) -> bool:
    return normalize_pipeline_profile(value) in PIPELINE_PROFILE_SPECS


def resolve_pipeline_profile(cfg: Any | None, plan: Mapping[str, Any] | None = None) -> str:
    explicit = _normalize_str(_cfg_value(cfg, 'pipeline.profile')) if cfg is not None else ''
    if explicit:
        profile = normalize_pipeline_profile(explicit)
        if profile not in PIPELINE_PROFILE_SPECS:
            raise ValueError(f'Unsupported pipeline.profile: {profile}')
        return profile
    explicit_template_task_id = _normalize_str(_cfg_value(cfg, 'run.clearml.pipeline.template_task_id')) if cfg is not None else ''
    if plan is not None:
        signature = {
            'run_dataset_register': bool(plan.get('run_dataset_register')),
            'run_preprocess': bool(plan.get('run_preprocess')),
            'run_train': bool(plan.get('run_train')),
            'run_train_ensemble': bool(plan.get('run_train_ensemble')),
            'run_leaderboard': bool(plan.get('run_leaderboard')),
            'run_infer': bool(plan.get('run_infer')),
            'model_set': _normalize_str(plan.get('model_set')),
        }
        for profile_name, spec in PIPELINE_PROFILE_SPECS.items():
            expected = {
                'run_dataset_register': spec.run_dataset_register,
                'run_preprocess': spec.run_preprocess,
                'run_train': spec.run_train,
                'run_train_ensemble': spec.run_train_ensemble,
                'run_leaderboard': spec.run_leaderboard,
                'run_infer': spec.run_infer,
                'model_set': spec.model_set,
            }
            if signature == expected:
                return profile_name
        if explicit_template_task_id:
            return DEFAULT_PIPELINE_PROFILE
        raise ValueError(
            'No built-in visible pipeline template matches the current pipeline settings. '
            'Align the config to a supported profile or specify run.clearml.pipeline.template_task_id.'
        )
    return DEFAULT_PIPELINE_PROFILE


def build_pipeline_ui_parameter_whitelist(pipeline_profile: str) -> tuple[str, ...]:
    profile = normalize_pipeline_profile(pipeline_profile)
    if profile not in PIPELINE_TEMPLATE_UI_WHITELIST:
        raise ValueError(f'Unsupported pipeline profile for UI whitelist: {profile}')
    return PIPELINE_TEMPLATE_UI_WHITELIST[profile]


def resolve_pipeline_run_flags(cfg: Any | None) -> dict[str, bool]:
    pipeline_cfg = getattr(cfg, 'pipeline', None) if cfg is not None else None
    profile_text = _normalize_str(_cfg_value(cfg, 'pipeline.profile')) if cfg is not None else ''
    profile_spec = get_pipeline_profile_spec(profile_text) if profile_text else None
    if profile_spec is not None:
        return {
            'run_dataset_register': profile_spec.run_dataset_register,
            'run_preprocess': profile_spec.run_preprocess,
            'run_train': profile_spec.run_train,
            'run_train_ensemble': profile_spec.run_train_ensemble,
            'run_leaderboard': profile_spec.run_leaderboard,
            'run_infer': profile_spec.run_infer,
        }
    return {
        'run_dataset_register': bool(getattr(pipeline_cfg, 'run_dataset_register', False)),
        'run_preprocess': bool(getattr(pipeline_cfg, 'run_preprocess', True)),
        'run_train': bool(getattr(pipeline_cfg, 'run_train', True)),
        'run_train_ensemble': bool(
            getattr(pipeline_cfg, 'run_train_ensemble', _cfg_value(cfg, 'ensemble.enabled', False))
        ),
        'run_leaderboard': bool(getattr(pipeline_cfg, 'run_leaderboard', True)),
        'run_infer': bool(getattr(pipeline_cfg, 'run_infer', False)),
    }


def resolve_pipeline_plan_only(cfg: Any | None) -> bool:
    return bool(_cfg_value(cfg, 'pipeline.plan_only')) or bool(_cfg_value(cfg, 'pipeline.dry_run')) or bool(_cfg_value(cfg, 'pipeline.plan'))


def _dedupe_strings(values: Any) -> list[str]:
    items = [_normalize_str(item) for item in _to_list(values)]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _resolve_selection_subset(
    requested: list[str],
    enabled: Any,
    *,
    label: str,
) -> tuple[list[str], list[str]]:
    requested_values = _dedupe_strings(requested)
    enabled_values = _dedupe_strings(enabled)
    if not enabled_values:
        return (requested_values, [])
    requested_set = set(requested_values)
    invalid = [item for item in enabled_values if item not in requested_set]
    if invalid:
        raise ValueError(
            f'{label} contains values outside the fixed pipeline profile: '
            f'invalid={invalid}, allowed={requested_values}'
        )
    enabled_set = set(enabled_values)
    active = [item for item in requested_values if item in enabled_set]
    disabled = [item for item in requested_values if item not in enabled_set]
    return (active, disabled)


def resolve_pipeline_selection(
    cfg: Any | None,
    *,
    preprocess_variants: list[str],
    model_variants: list[str],
    ensemble_methods: list[str],
) -> dict[str, list[str]]:
    requested_preprocess = _dedupe_strings(preprocess_variants)
    requested_models = _dedupe_strings(model_variants)
    requested_methods = _dedupe_strings(ensemble_methods)
    (active_preprocess, disabled_preprocess) = _resolve_selection_subset(
        requested_preprocess,
        _cfg_value(cfg, 'pipeline.selection.enabled_preprocess_variants'),
        label='pipeline.selection.enabled_preprocess_variants',
    )
    (active_models, disabled_models) = _resolve_selection_subset(
        requested_models,
        _cfg_value(cfg, 'pipeline.selection.enabled_model_variants'),
        label='pipeline.selection.enabled_model_variants',
    )
    (active_methods, disabled_methods) = _resolve_selection_subset(
        requested_methods,
        _cfg_value(cfg, 'ensemble.selection.enabled_methods'),
        label='ensemble.selection.enabled_methods',
    )
    return {
        'requested_preprocess_variants': requested_preprocess,
        'active_preprocess_variants': active_preprocess,
        'disabled_preprocess_variants': disabled_preprocess,
        'requested_model_variants': requested_models,
        'active_model_variants': active_models,
        'disabled_model_variants': disabled_models,
        'requested_ensemble_methods': requested_methods,
        'active_ensemble_methods': active_methods,
        'disabled_ensemble_methods': disabled_methods,
    }


def apply_pipeline_profile_defaults(cfg: Any, pipeline_profile: str) -> Any:
    spec = get_pipeline_profile_spec(pipeline_profile)
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None

    def _set(path: str, value: Any) -> None:
        if OmegaConf is not None and OmegaConf.is_config(cfg):
            OmegaConf.update(cfg, path, value, merge=False, force_add=True)
            return
        target = cfg
        parts = path.split('.')
        for key in parts[:-1]:
            child = getattr(target, key, None)
            if child is None:
                child = {}
                if isinstance(target, Mapping):
                    target[key] = child
                else:
                    setattr(target, key, child)
            target = child
        leaf = parts[-1]
        if isinstance(target, Mapping):
            target[leaf] = value
        else:
            setattr(target, leaf, value)

    _set('pipeline.profile', spec.name)
    _set('pipeline.run_dataset_register', spec.run_dataset_register)
    _set('pipeline.run_preprocess', spec.run_preprocess)
    _set('pipeline.run_train', spec.run_train)
    _set('pipeline.run_train_ensemble', spec.run_train_ensemble)
    _set('pipeline.run_leaderboard', spec.run_leaderboard)
    _set('pipeline.run_infer', spec.run_infer)
    _set('pipeline.model_set', spec.model_set)
    _set('pipeline.model_variants', [])
    _set('pipeline.grid.model_variants', [])
    _set('pipeline.selection.enabled_preprocess_variants', [])
    _set('pipeline.selection.enabled_model_variants', [])
    _set('ensemble.enabled', spec.run_train_ensemble)
    _set('ensemble.methods', list(spec.ensemble_methods))
    if spec.ensemble_methods:
        _set('ensemble.method', spec.ensemble_methods[0])
    _set('ensemble.selection.enabled_methods', [])
    return cfg


def build_pipeline_template_defaults(
    *,
    cfg: Any,
    plan: Mapping[str, Any],
    grid_run_id: str,
    pipeline_profile: str,
    pipeline_task_id: str | None = None,
) -> dict[str, Any]:
    defaults = {
        **dict(plan.get('run_overrides') or {}),
        **dict(plan.get('data_overrides') or {}),
        **dict(plan.get('downstream_data_overrides') or {}),
        **dict(plan.get('eval_overrides') or {}),
    }
    defaults = {
        str(key): value
        for key, value in defaults.items()
        if str(key) not in PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS
    }
    defaults['task'] = 'pipeline'
    defaults['default_queue'] = (
        _normalize_str((plan.get('queues') or {}).get('default'))
        or _normalize_str(_cfg_value(cfg, 'run.clearml.queue_name'))
        or 'default'
    )
    defaults['run.grid_run_id'] = grid_run_id
    defaults['run.clearml.execution'] = 'pipeline_controller'
    defaults['run.clearml.pipeline.template_task_id'] = _normalize_str(
        _cfg_value(cfg, 'run.clearml.pipeline.template_task_id')
    )
    defaults['run.clearml.pipeline_task_id'] = pipeline_task_id or ''
    defaults['pipeline.profile'] = normalize_pipeline_profile(pipeline_profile)
    defaults['pipeline.run_dataset_register'] = plan.get('run_dataset_register')
    defaults['pipeline.run_preprocess'] = plan.get('run_preprocess')
    defaults['pipeline.run_train'] = plan.get('run_train')
    defaults['pipeline.run_train_ensemble'] = plan.get('run_train_ensemble')
    defaults['pipeline.run_leaderboard'] = plan.get('run_leaderboard')
    defaults['pipeline.run_infer'] = plan.get('run_infer')
    defaults['pipeline.plan_only'] = bool(plan.get('plan_only'))
    selection = dict(plan.get('selection') or {})
    defaults['pipeline.grid.preprocess_variants'] = selection.get('requested_preprocess_variants') or plan.get('preprocess_variants')
    defaults['pipeline.grid.model_variants'] = selection.get('requested_model_variants') or plan.get('model_variants')
    defaults['pipeline.model_set'] = _normalize_str(_cfg_value(cfg, 'pipeline.model_set'))
    defaults['pipeline.selection.enabled_preprocess_variants'] = selection.get('active_preprocess_variants') or plan.get('preprocess_variants')
    defaults['pipeline.selection.enabled_model_variants'] = selection.get('active_model_variants') or plan.get('model_variants')
    defaults['ensemble.selection.enabled_methods'] = selection.get('active_ensemble_methods') or []
    return {key: value for key, value in defaults.items() if value is not None}


def build_pipeline_run_overrides(
    *,
    cfg: Any,
    plan: Mapping[str, Any],
    grid_run_id: str,
    pipeline_profile: str,
    pipeline_task_id: str | None = None,
) -> dict[str, Any]:
    return build_pipeline_template_defaults(
        cfg=cfg,
        plan=plan,
        grid_run_id=grid_run_id,
        pipeline_profile=pipeline_profile,
        pipeline_task_id=pipeline_task_id,
    )


def extract_pipeline_editable_defaults(
    defaults: Mapping[str, Any],
    *,
    pipeline_profile: str,
) -> dict[str, Any]:
    whitelist = set(build_pipeline_ui_parameter_whitelist(pipeline_profile))
    editable = {str(key): value for key, value in dict(defaults).items() if str(key) in whitelist}
    if 'run.usecase_id' in whitelist and 'run.usecase_id' not in editable:
        editable['run.usecase_id'] = defaults.get('run.usecase_id')
    return {key: value for key, value in editable.items() if value is not None}


def build_pipeline_step_specs(plan: Mapping[str, Any]) -> tuple[PipelineStepSpec, ...]:
    step_specs: list[PipelineStepSpec] = []
    steps = plan.get('steps') or {}
    for key in ('dataset_register', 'preprocess', 'train', 'train_ensemble', 'leaderboard', 'infer'):
        payload = steps.get(key)
        if payload is None:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not item:
                continue
            step_specs.append(
                PipelineStepSpec(
                    step_name=str(item.get('step_name')),
                    task_name=str(item.get('task_name')),
                    parents=tuple(str(parent) for parent in (item.get('parents') or [])),
                    queue=_normalize_str(item.get('queue')),
                    run_dir=str(item.get('run_dir') or ''),
                    preprocess_variant=_normalize_str(item.get('preprocess_variant')),
                    model_variant=_normalize_str(item.get('model_variant')),
                    ensemble_method=_normalize_str(item.get('ensemble_method')),
                    hpo_run_id=_normalize_str(item.get('hpo_run_id')),
                )
            )
    return tuple(step_specs)


def resolve_pipeline_queues(plan: Mapping[str, Any]) -> dict[str, Any]:
    return dict(plan.get('queues') or {})


def resolve_pipeline_variants(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(str(item) for item in (plan.get('preprocess_variants') or [])),
        tuple(str(item) for item in (plan.get('model_variants') or [])),
    )


def build_pipeline_plan(plan: Mapping[str, Any], *, pipeline_profile: str) -> PipelinePlan:
    return PipelinePlan(
        profile=normalize_pipeline_profile(pipeline_profile),
        preprocess_variants=resolve_pipeline_variants(plan)[0],
        model_variants=resolve_pipeline_variants(plan)[1],
        steps=build_pipeline_step_specs(plan),
        queues=resolve_pipeline_queues(plan),
        run_dataset_register=bool(plan.get('run_dataset_register')),
        run_preprocess=bool(plan.get('run_preprocess')),
        run_train=bool(plan.get('run_train')),
        run_train_ensemble=bool(plan.get('run_train_ensemble')),
        run_leaderboard=bool(plan.get('run_leaderboard')),
        run_infer=bool(plan.get('run_infer')),
    )


def strip_local_only_pipeline_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(overrides).items()
        if str(key) not in PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS
    }


def build_pipeline_template_step_overrides(
    step_overrides: Mapping[str, Any],
    *,
    editable_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    editable_keys = set(editable_defaults.keys())
    for key, value in strip_local_only_pipeline_overrides(step_overrides).items():
        if key in editable_keys:
            overrides[key] = f"${{pipeline.{str(key).replace('.', '/')}}}"
        else:
            overrides[key] = value
    return overrides


__all__ = [
    'DEFAULT_PIPELINE_PROFILE',
    'PIPELINE_PROFILE_SPECS',
    'PIPELINE_TEMPLATE_LOCAL_ONLY_KEYS',
    'PIPELINE_TEMPLATE_UI_WHITELIST',
    'apply_pipeline_profile_defaults',
    'PipelinePlan',
    'PipelineProfileSpec',
    'PipelineStepSpec',
    'build_pipeline_plan',
    'build_pipeline_run_overrides',
    'build_pipeline_step_specs',
    'build_pipeline_template_defaults',
    'build_pipeline_template_step_overrides',
    'build_pipeline_ui_parameter_whitelist',
    'extract_pipeline_editable_defaults',
    'get_pipeline_profile_spec',
    'is_pipeline_template_name',
    'normalize_pipeline_profile',
    'resolve_pipeline_plan_only',
    'resolve_pipeline_profile',
    'resolve_pipeline_run_flags',
    'resolve_pipeline_selection',
    'resolve_pipeline_queues',
    'resolve_pipeline_variants',
    'strip_local_only_pipeline_overrides',
]
