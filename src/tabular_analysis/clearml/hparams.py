from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.repo_utils import resolve_repo_root_fallback
from pathlib import Path
from typing import Any, Iterable, Mapping
from ..platform_adapter_clearml_env import resolve_version_props
from ..platform_adapter_task_context import connect_hyperparameters
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_OMEGACONF_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_VERSION_RECOVERABLE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_DEFAULT_SECTION_ORDER = ('inputs', 'dataset', 'preprocess', 'model', 'eval', 'optimize', 'pipeline', 'clearml')
def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for (key, value) in payload.items() if value is not None}
def _merge_nested_mapping(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    for (key, value) in payload.items():
        text = str(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            child = target.get(text)
            if not isinstance(child, dict):
                child = {}
            target[text] = child
            _merge_nested_mapping(child, value)
            continue
        target[text] = _to_builtin(value)


def _build_nested_mapping(path: str, value: Any) -> dict[str, Any]:
    text = _normalize_str(path)
    if not text:
        return {}
    parts = [part for part in text.split('.') if part]
    if not parts:
        return {}
    node: Any = _to_builtin(value)
    for key in reversed(parts):
        node = {key: node}
    return node if isinstance(node, dict) else {}


def _flatten(prefix: str, params: Mapping[str, Any]) -> dict[str, Any]:
    return _build_nested_mapping(prefix, params)
def _load_sections_from_file() -> dict[str, list[str]]:
    path = resolve_repo_root_fallback(fallback=Path(__file__).resolve().parents[3]) / 'conf' / 'clearml' / 'hyperparams_sections.yaml'
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
    if not isinstance(payload, Mapping):
        return {}
    sections = payload.get('sections')
    if isinstance(sections, Mapping):
        return {str(key): list(value or []) for (key, value) in sections.items()}
    return {}
def _resolve_sections_cfg(cfg: Any) -> dict[str, list[str]]:
    sections = _cfg_value(cfg, 'run.clearml.hyperparams.sections')
    try:
        from omegaconf import OmegaConf
    except _OPTIONAL_IMPORT_ERRORS:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(sections):
        sections = OmegaConf.to_container(sections, resolve=True)
    if isinstance(sections, Mapping):
        return {str(key): list(value or []) for (key, value) in sections.items()}
    return _load_sections_from_file()
def _section_key(sections_cfg: Mapping[str, Any], canonical: str) -> str:
    if not sections_cfg:
        return canonical
    for key in sections_cfg:
        if str(key).lower() == canonical.lower():
            return str(key)
    return canonical
def _flatten_mapping(prefix: str, payload: Any, out: dict[str, Any]) -> None:
    if payload is None:
        return
    nested = _build_nested_mapping(prefix, payload)
    if nested:
        _merge_nested_mapping(out, nested)
def _flatten_cfg_rule_values(prefix: str, payload: Any, out: dict[str, Any]) -> None:
    if payload is None:
        return
    if isinstance(payload, Mapping):
        for (key, value) in payload.items():
            text = _normalize_str(key)
            if not text:
                continue
            next_prefix = f'{prefix}.{text}' if prefix else text
            _flatten_cfg_rule_values(next_prefix, value, out)
        return
    if prefix and prefix not in out:
        out[prefix] = payload


def _extract_sections(cfg: Any, sections_cfg: Mapping[str, Iterable[str]]) -> dict[str, dict[str, Any]]:
    candidate_values: dict[str, Any] = {}
    for name in _section_order(sections_cfg):
        for path in list(sections_cfg.get(name) or []):
            text = _normalize_str(path)
            if not text:
                continue
            if text.endswith('.*'):
                base = text[:-2]
                value = _cfg_value(cfg, base)
                if value is None:
                    continue
                _flatten_cfg_rule_values(base, value, candidate_values)
                continue
            if text in candidate_values:
                continue
            value = _cfg_value(cfg, text)
            if value is not None:
                candidate_values[text] = value
    (sections, _) = build_sections_from_values(candidate_values, cfg=cfg)
    return sections
def _section_order(sections_cfg: Mapping[str, Any]) -> list[str]:
    if sections_cfg:
        return [str(key) for key in sections_cfg.keys()]
    return list(_DEFAULT_SECTION_ORDER)
def _merge_section(sections: dict[str, dict[str, Any]], name: str, payload: Mapping[str, Any]) -> None:
    cleaned = _drop_none(payload)
    if not cleaned:
        return
    merged = dict(sections.get(name, {}))
    for (key, value) in cleaned.items():
        if isinstance(value, Mapping) and '.' not in str(key):
            _merge_nested_mapping(merged, {str(key): _to_builtin(value)})
            continue
        _merge_nested_mapping(merged, _build_nested_mapping(str(key), value))
    sections[name] = merged
def _execution_hparams(cfg: Any) -> dict[str, Any]:
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or _normalize_str(_cfg_value(cfg, 'usecase_id'))
    execution = _normalize_str(_cfg_value(cfg, 'run.clearml.execution'))
    try:
        versions = resolve_version_props(cfg, clearml_enabled=True)
    except _VERSION_RECOVERABLE_ERRORS:
        versions = {'schema_version': 'unknown', 'code_version': 'unknown'}
    return _drop_none({'run.usecase_id': usecase_id, 'run.schema_version': versions.get('schema_version'), 'run.code_version': versions.get('code_version'), 'run.clearml.execution': execution})
def _to_builtin(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf
    except _OPTIONAL_IMPORT_ERRORS:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _path_matches_section_rule(path: str, rule: str) -> bool:
    if rule.endswith('.*'):
        base = rule[:-2]
        return path == base or path.startswith(f'{base}.')
    return path == rule


def build_sections_from_values(
    values: Mapping[str, Any],
    *,
    cfg: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections: dict[str, dict[str, Any]] = {}
    normalized_values = {
        _normalize_str(key): value
        for (key, value) in dict(values).items()
        if _normalize_str(key) and value is not None
    }
    section_order = _section_order(sections_cfg)
    for (key, value) in normalized_values.items():
        for name in section_order:
            rules = list(sections_cfg.get(name) or [])
            if not any(_path_matches_section_rule(key, _normalize_str(rule)) for rule in rules if _normalize_str(rule)):
                continue
            payload = dict(sections.get(str(name), {}))
            _merge_nested_mapping(payload, _build_nested_mapping(key, value))
            sections[str(name)] = payload
            break
    return (sections, section_order)


def _flatten_nested_payload(payload: Any, out: dict[str, Any], *, prefix: str = '') -> None:
    if isinstance(payload, Mapping):
        for (key, value) in payload.items():
            text = _normalize_str(key)
            if not text:
                continue
            next_prefix = f'{prefix}.{text}' if prefix else text
            _flatten_nested_payload(value, out, prefix=next_prefix)
        return
    if prefix:
        out[prefix] = payload


def split_values_by_sections(
    values: Mapping[str, Any],
    *,
    cfg: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    (sections, order) = build_sections_from_values(values, cfg=cfg)
    covered: dict[str, Any] = {}
    for payload in sections.values():
        _flatten_nested_payload(payload, covered)
    normalized_values = {
        _normalize_str(key): _to_builtin(value)
        for (key, value) in dict(values).items()
        if _normalize_str(key) and value is not None
    }
    remaining = {
        key: value
        for (key, value) in normalized_values.items()
        if key not in covered
    }
    return (sections, order, remaining)
def _connect_section(ctx: Any, name: str, payload: Mapping[str, Any]) -> None:
    cleaned = _drop_none(payload)
    if not cleaned:
        return
    connect_hyperparameters(ctx, cleaned, name=name)
def _connect_sections(ctx: Any, sections: Mapping[str, Mapping[str, Any]], order: Iterable[str]) -> None:
    seen: set[str] = set()
    for name in order:
        payload = sections.get(name)
        if not payload:
            continue
        _connect_section(ctx, name, payload)
        seen.add(name)
    for (name, payload) in sections.items():
        if name in seen:
            continue
        _connect_section(ctx, name, payload)
def build_dataset_register_sections(cfg: Any, *, dataset_path: str | None, target_column: str | None, raw_dataset_id: str | None=None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    inputs_key = _section_key(sections_cfg, 'inputs')
    dataset_key = _section_key(sections_cfg, 'dataset')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, inputs_key, {'data.dataset_path': dataset_path, 'data.target_column': target_column})
    _merge_section(sections, dataset_key, {'data.raw_dataset_id': raw_dataset_id})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    return (sections, _section_order(sections_cfg))
def connect_dataset_register(ctx: Any, cfg: Any, *, dataset_path: str | None, target_column: str | None, raw_dataset_id: str | None=None) -> None:
    (sections, order) = build_dataset_register_sections(cfg, dataset_path=dataset_path, target_column=target_column, raw_dataset_id=raw_dataset_id)
    _connect_sections(ctx, sections, order)
def build_preprocess_sections(cfg: Any, *, raw_dataset_id: str | None, dataset_path: str | None, preprocess_variant: str | None, split_strategy: str | None, split_seed: int | None, store_features: bool | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    inputs_key = _section_key(sections_cfg, 'inputs')
    dataset_key = _section_key(sections_cfg, 'dataset')
    preprocess_key = _section_key(sections_cfg, 'preprocess')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, inputs_key, {'data.dataset_path': dataset_path})
    _merge_section(sections, dataset_key, {'data.raw_dataset_id': raw_dataset_id})
    _merge_section(sections, preprocess_key, {'preprocess.variant': preprocess_variant, 'data.split.strategy': split_strategy, 'data.split.seed': split_seed, 'ops.processed_dataset.store_features': store_features})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    return (sections, _section_order(sections_cfg))
def connect_preprocess(ctx: Any, cfg: Any, *, raw_dataset_id: str | None, dataset_path: str | None, preprocess_variant: str | None, split_strategy: str | None, split_seed: int | None, store_features: bool | None) -> None:
    (sections, order) = build_preprocess_sections(cfg, raw_dataset_id=raw_dataset_id, dataset_path=dataset_path, preprocess_variant=preprocess_variant, split_strategy=split_strategy, split_seed=split_seed, store_features=store_features)
    _connect_sections(ctx, sections, order)
def connect_train_model(ctx: Any, cfg: Any, *, processed_dataset_id: str | None, task_type: str | None, primary_metric: str | None, model_variant: str | None, model_params: Mapping[str, Any] | None) -> None:
    model_payload: dict[str, Any] = {'model_variant.name': model_variant, 'train.model': model_variant}
    if model_params:
        model_payload.update(_flatten('train.params', model_params))
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    dataset_key = _section_key(sections_cfg, 'dataset')
    model_key = _section_key(sections_cfg, 'model')
    eval_key = _section_key(sections_cfg, 'eval')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, dataset_key, {'data.processed_dataset_id': processed_dataset_id})
    _merge_section(sections, model_key, model_payload)
    _merge_section(sections, eval_key, {'eval.task_type': task_type, 'eval.primary_metric': primary_metric})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    _connect_sections(ctx, sections, _section_order(sections_cfg))
def connect_train_ensemble(ctx: Any, cfg: Any, *, processed_dataset_id: str | None, task_type: str | None, primary_metric: str | None, method: str | None, top_k: int | None) -> None:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    dataset_key = _section_key(sections_cfg, 'dataset')
    model_key = _section_key(sections_cfg, 'model')
    eval_key = _section_key(sections_cfg, 'eval')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, dataset_key, {'data.processed_dataset_id': processed_dataset_id})
    _merge_section(sections, model_key, {'ensemble.method': method, 'ensemble.top_k': top_k})
    _merge_section(sections, eval_key, {'eval.task_type': task_type, 'eval.primary_metric': primary_metric})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    _connect_sections(ctx, sections, _section_order(sections_cfg))
def connect_infer(ctx: Any, cfg: Any, *, model_id: str | None, model_abbr: str | None=None, infer_mode: str | None, schema_policy: str | None, input_source: str | None=None, input_path: str | None=None, input_json: str | None=None, provenance: Mapping[str, Any] | None=None, optimize_payload: Mapping[str, Any] | None=None, include_dataset: bool=True, include_execution: bool=True) -> None:
    dataset_payload: dict[str, Any] = {}
    if provenance:
        dataset_payload = {'train_task_id': provenance.get('train_task_id'), 'data.raw_dataset_id': provenance.get('raw_dataset_id'), 'data.processed_dataset_id': provenance.get('processed_dataset_id'), 'preprocess.variant': provenance.get('preprocess_variant'), 'split_hash': provenance.get('split_hash'), 'recipe_hash': provenance.get('recipe_hash')}
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    inputs_key = _section_key(sections_cfg, 'inputs')
    model_key = _section_key(sections_cfg, 'model')
    dataset_key = _section_key(sections_cfg, 'dataset')
    optimize_key = _section_key(sections_cfg, 'optimize')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, inputs_key, {'infer.mode': infer_mode, 'infer.validation.mode': schema_policy, 'infer.input_source': input_source, 'infer.input_path': input_path, 'infer.input_json': input_json})
    _merge_section(sections, model_key, {'infer.model_id': model_id, 'model_abbr': model_abbr})
    if optimize_payload:
        _merge_section(sections, optimize_key, dict(optimize_payload))
    if include_dataset:
        _merge_section(sections, dataset_key, dataset_payload)
    if include_execution:
        _merge_section(sections, clearml_key, _execution_hparams(cfg))
    _connect_sections(ctx, sections, _section_order(sections_cfg))
def connect_leaderboard(ctx: Any, cfg: Any, *, primary_metric: str | None, direction: str | None, require_comparable: bool | None, top_k: int | None, recommend_top_k: int | None) -> None:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    eval_key = _section_key(sections_cfg, 'eval')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, eval_key, {'eval.primary_metric': primary_metric, 'eval.direction': direction, 'leaderboard.require_comparable': require_comparable, 'leaderboard.top_k': top_k, 'leaderboard.recommend.top_k': recommend_top_k})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    _connect_sections(ctx, sections, _section_order(sections_cfg))
def connect_pipeline(ctx: Any, cfg: Any, *, grid_run_id: str | None, run_dataset_register: bool | None, run_preprocess: bool | None, run_train: bool | None, run_train_ensemble: bool | None, run_leaderboard: bool | None, run_infer: bool | None, plan_only: bool | None) -> None:
    sections_cfg = _resolve_sections_cfg(cfg)
    sections = _extract_sections(cfg, sections_cfg)
    inputs_key = _section_key(sections_cfg, 'inputs')
    pipeline_key = _section_key(sections_cfg, 'pipeline')
    clearml_key = _section_key(sections_cfg, 'clearml')
    _merge_section(sections, inputs_key, {'run.grid_run_id': grid_run_id})
    _merge_section(sections, pipeline_key, {'pipeline.run_dataset_register': run_dataset_register, 'pipeline.run_preprocess': run_preprocess, 'pipeline.run_train': run_train, 'pipeline.run_train_ensemble': run_train_ensemble, 'pipeline.run_leaderboard': run_leaderboard, 'pipeline.run_infer': run_infer, 'pipeline.plan_only': plan_only})
    _merge_section(sections, clearml_key, _execution_hparams(cfg))
    _connect_sections(ctx, sections, _section_order(sections_cfg))
