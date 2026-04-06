from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str, to_float as _to_float
from ..common.collection_utils import stringify_payload as _stringify_payload, to_list as _to_list, to_mapping as _to_mapping
from ..common.json_utils import load_json as _load_json
from ..common.model_reference import build_infer_reference
import csv
from dataclasses import make_dataclass
from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping
import warnings
from ..clearml.hparams import connect_leaderboard
from ..clearml.ui_logger import log_debug_table, log_plotly, log_scalar
from ..io.bundle_io import load_bundle
from ..ops.clearml_identity import apply_clearml_identity
from ..platform_adapter_artifacts import upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_task import PlatformAdapterError, clearml_task_id, clearml_task_tags, get_task_artifact_local_copy, list_clearml_tasks_by_tags, update_recommended_registry_model_tags_multi, update_task_properties
from ..viz.leaderboard_plots import build_leaderboard_table, build_pareto_scatter, build_top_k_bar, write_top_k_bar_png
from .lifecycle import emit_outputs_and_manifest, start_runtime
def _normalize_direction(value: Any) -> str | None:
    direction = _normalize_str(value)
    if direction == 'auto':
        return None
    return direction
def _normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
def _format_float(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return 'n/a'
    return f'{num:.6g}'
def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float('nan')
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)
def _normalize_metric_values(values: list[float | None], *, normalization: str) -> list[float | None]:
    valid = [value for value in values if value is not None]
    if not valid:
        return [None for _ in values]
    if normalization == 'robust':
        low = _quantile(valid, 0.1)
        high = _quantile(valid, 0.9)
    else:
        low = min(valid)
        high = max(valid)
    if not math.isfinite(low) or not math.isfinite(high) or high == low:
        return [0.0 if value is not None else None for value in values]
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        score = (value - low) / (high - low)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        normalized.append(float(score))
    return normalized
def _format_ci_interval(interval: dict[str, Any] | None) -> str | None:
    if not isinstance(interval, dict):
        return None
    low = interval.get('low')
    mid = interval.get('mid')
    high = interval.get('high')
    if low is None and mid is None and (high is None):
        return None
    return f'[{_format_float(low)}, {_format_float(mid)}, {_format_float(high)}]'
def _resolve_scoring_config(cfg: Any) -> tuple[list[str], dict[str, float], str, list[str]]:
    warnings: list[str] = []
    metrics = _to_list(_cfg_value(cfg, 'leaderboard.scoring.metrics', None))
    if not metrics:
        metrics = ['r2', 'rmse', 'mae', 'mse']
        warnings.append('leaderboard.scoring.metrics is empty; defaulting to r2/rmse/mae/mse.')
    deduped: list[str] = []
    seen: set[str] = set()
    for name in metrics:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    metrics = deduped
    weights_raw = _to_mapping(_cfg_value(cfg, 'leaderboard.scoring.weights', None))
    weights: dict[str, float] = {}
    for metric in metrics:
        weight = _to_float(weights_raw.get(metric))
        if weight is None:
            weights[metric] = 1.0
            warnings.append(f'leaderboard.scoring.weights.{metric} is missing; defaulting to 1.0.')
        else:
            weights[metric] = weight
    normalization = _normalize_str(_cfg_value(cfg, 'leaderboard.scoring.normalization', None)) or 'minmax'
    if normalization not in ('minmax', 'robust'):
        warnings.append(f"Unknown normalization '{normalization}'; falling back to minmax.")
        normalization = 'minmax'
    return (metrics, weights, normalization, warnings)
def _extract_holdout_metrics(metrics_payload: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(metrics_payload, dict):
        return {}
    holdout = metrics_payload.get('holdout')
    if not isinstance(holdout, dict):
        return {}
    metrics: dict[str, float | None] = {}
    for (key, value) in holdout.items():
        if key in ('train_rows', 'val_rows'):
            continue
        metrics[str(key)] = _to_float(value)
    return metrics
def _resolve_run_dir(ref: str) -> Path:
    path = Path(ref).expanduser()
    if path.is_file():
        return path.parent
    return path
def _dedupe_refs(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _normalize_str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
def _collect_clearml_refs(cfg: Any) -> list[str]:
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
    grid_run_id = _normalize_str(_cfg_value(cfg, 'run.grid_run_id'))
    base_tags = [f'usecase:{usecase_id}']
    if grid_run_id:
        base_tags.append(f'grid:{grid_run_id}')
    refs: list[str] = []
    for process in ('train_model', 'train_ensemble'):
        tags = [*base_tags, f'process:{process}']
        try:
            tasks = list_clearml_tasks_by_tags(tags)
        except PlatformAdapterError:
            continue
        for task in tasks:
            task_tags = clearml_task_tags(task)
            if 'template:true' in task_tags:
                continue
            task_id = clearml_task_id(task)
            if task_id:
                refs.append(task_id)
    return _dedupe_refs(refs)
def _scan_local_refs(cfg: Any) -> list[str]:
    run_root = Path(getattr(cfg.run, 'output_dir', 'outputs')).expanduser().resolve()
    search_root = run_root.parent if run_root.name.startswith('leaderboard') else run_root
    refs: list[str] = []
    for out_path in search_root.glob('**/03_train_model/out.json'):
        refs.append(str(out_path.parent))
    for out_path in search_root.glob('**/04_train_ensemble/out.json'):
        refs.append(str(out_path.parent))
    return _dedupe_refs(refs)
def _resolve_model_bundle_path(run_dir: Path, model_id: str | None) -> Path | None:
    if model_id:
        candidate = Path(model_id).expanduser()
        if candidate.exists():
            return candidate.resolve()
    candidate = run_dir / 'model_bundle.joblib'
    if candidate.exists():
        return candidate.resolve()
    return None
def _extract_variants(bundle: Any) -> tuple[str | None, str | None]:
    model_variant = None
    preprocess_variant = None
    if isinstance(bundle, dict):
        model_variant = _normalize_str(bundle.get('model_variant'))
        preprocess_bundle = bundle.get('preprocess_bundle')
        if isinstance(preprocess_bundle, dict):
            preprocess_variant = _normalize_str(preprocess_bundle.get('preprocess_variant'))
    return (model_variant, preprocess_variant)
def _extract_metric_ci(out: dict[str, Any]) -> dict[str, float | None] | None:
    if not isinstance(out, dict):
        return None
    for candidate in (out.get('primary_metric_ci'), out.get('metric_ci'), out.get('metrics_ci')):
        if isinstance(candidate, dict):
            low = _to_float(candidate.get('low'))
            mid = _to_float(candidate.get('mid'))
            high = _to_float(candidate.get('high'))
            if low is None and mid is None and (high is None):
                continue
            return {'low': low, 'mid': mid, 'high': high}
    return None
def _build_entry(*, out: dict[str, Any], manifest: dict[str, Any] | None, metrics_payload: dict[str, Any] | None, ensemble_spec_payload: dict[str, Any] | None, train_task_ref: str, model_bundle_path: Path | None, expected_primary_metric: str | None, expected_direction: str | None, expected_seed: int | None) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    processed_dataset_id = _normalize_str(out.get('processed_dataset_id'))
    split_hash = _normalize_str(out.get('split_hash'))
    recipe_hash = _normalize_str(out.get('recipe_hash'))
    model_id = _normalize_str(out.get('model_id'))
    registry_model_id = _normalize_str(out.get('registry_model_id'))
    best_score = _to_float(out.get('best_score'))
    task_type = _normalize_str(out.get('task_type'))
    primary_metric = _normalize_str(out.get('primary_metric')) or expected_primary_metric
    if primary_metric is None:
        errors.append('primary_metric is missing.')
    process_name = None
    if isinstance(manifest, dict):
        process_name = manifest.get('process')
        if isinstance(process_name, dict):
            process_name = process_name.get('name')
    process_name = _normalize_str(process_name)
    model_family = 'ensemble' if process_name == 'train_ensemble' else 'single'
    inputs = {}
    if isinstance(manifest, dict):
        inputs = manifest.get('inputs') or {}
    direction = _normalize_direction(inputs.get('direction')) or expected_direction
    task_type = task_type or _normalize_str(inputs.get('task_type'))
    if task_type is None:
        task_type = 'regression'
    seed = _normalize_int(inputs.get('seed'))
    if seed is None:
        seed = expected_seed
    if direction is None:
        errors.append('direction is missing.')
    if processed_dataset_id is None:
        errors.append('processed_dataset_id is missing.')
    if split_hash is None:
        errors.append('split_hash is missing.')
    if recipe_hash is None:
        errors.append('recipe_hash is missing.')
    if model_id is None:
        errors.append('model_id is missing.')
    if best_score is None:
        errors.append('best_score is missing or invalid.')
    model_variant = _normalize_str(inputs.get('model_variant'))
    ensemble_method = None
    n_base_models = None
    primary_metric_source = _normalize_str(out.get('primary_metric_source'))
    if not primary_metric_source and isinstance(metrics_payload, dict):
        primary_metric_source = _normalize_str(metrics_payload.get('primary_metric_source'))
    if model_family == 'ensemble':
        if isinstance(ensemble_spec_payload, dict):
            ensemble_method = _normalize_str(ensemble_spec_payload.get('method'))
            n_base_models = _normalize_int(ensemble_spec_payload.get('n_base_models'))
            if n_base_models is None:
                included = ensemble_spec_payload.get('included')
                if isinstance(included, list):
                    n_base_models = len(included)
            if not primary_metric_source:
                primary_metric_source = _normalize_str(ensemble_spec_payload.get('primary_metric_source'))
    primary_metric_source = primary_metric_source or 'valid'
    preprocess_variant = None
    if model_bundle_path is not None:
        try:
            bundle = load_bundle(model_bundle_path)
            (bundle_model_variant, bundle_preprocess_variant) = _extract_variants(bundle)
            if model_variant is None:
                model_variant = bundle_model_variant
            preprocess_variant = bundle_preprocess_variant
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            warnings.append(f'Failed to load model_bundle.joblib: {exc}')
    if model_family == 'ensemble' and ensemble_method:
        model_variant = f'ensemble_{ensemble_method}'
    if model_variant is None:
        model_variant = 'unknown'
    if preprocess_variant is None:
        preprocess_variant = 'unknown'
    metric_ci = _extract_metric_ci(out)
    if errors:
        return (None, warnings, errors)
    threshold_payload = None
    threshold_value = _to_float(out.get('best_threshold'))
    threshold_metric = _normalize_str(out.get('threshold_metric'))
    threshold_score = _to_float(out.get('threshold_score'))
    if threshold_value is not None or threshold_metric or threshold_score is not None:
        threshold_payload = {'best_threshold': threshold_value, 'metric': threshold_metric, 'score': threshold_score}
    calibration_payload = out.get('calibration') if isinstance(out.get('calibration'), dict) else None
    imbalance_payload = out.get('imbalance') if isinstance(out.get('imbalance'), dict) else None
    uncertainty_payload = out.get('uncertainty') if isinstance(out.get('uncertainty'), dict) else None
    metrics = _extract_holdout_metrics(metrics_payload)
    infer_reference = build_infer_reference(model_id=model_id, registry_model_id=registry_model_id, train_task_id=_normalize_str(out.get('train_task_id')), train_task_ref=train_task_ref)
    entry = {'train_task_ref': train_task_ref, 'train_task_id': _normalize_str(out.get('train_task_id')) or None, 'model_id': model_id, 'registry_model_id': registry_model_id, 'infer_model_id': infer_reference.get('infer_model_id'), 'infer_train_task_id': infer_reference.get('infer_train_task_id'), 'reference_kind': infer_reference.get('reference_kind'), 'best_score': best_score, 'primary_metric': primary_metric, 'primary_metric_source': primary_metric_source, 'direction': direction, 'seed': seed, 'task_type': task_type, 'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'preprocess_variant': preprocess_variant, 'model_variant': model_variant, 'model_family': model_family, 'ensemble_method': ensemble_method, 'n_base_models': n_base_models, 'primary_metric_ci_low': metric_ci.get('low') if metric_ci else None, 'primary_metric_ci_mid': metric_ci.get('mid') if metric_ci else None, 'primary_metric_ci_high': metric_ci.get('high') if metric_ci else None, 'primary_metric_ci': metric_ci, 'thresholding': threshold_payload, 'calibration': calibration_payload, 'imbalance': imbalance_payload, 'uncertainty': uncertainty_payload, 'n_classes': _normalize_int(out.get('n_classes')), 'class_labels': out.get('class_labels'), 'metrics': metrics}
    return (entry, warnings, [])
def _compare_comparability(entry: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if ref.get('processed_dataset_id') and entry.get('processed_dataset_id') != ref.get('processed_dataset_id'):
        mismatches.append('processed_dataset_id mismatch')
    if ref.get('split_hash') and entry.get('split_hash') != ref.get('split_hash'):
        mismatches.append('split_hash mismatch')
    if ref.get('primary_metric') and entry.get('primary_metric') != ref.get('primary_metric'):
        mismatches.append('primary_metric mismatch')
    if ref.get('direction') and entry.get('direction') != ref.get('direction'):
        mismatches.append('direction mismatch')
    if ref.get('task_type') and entry.get('task_type') != ref.get('task_type'):
        mismatches.append('task_type mismatch')
    if ref.get('seed') is not None:
        if entry.get('seed') is None:
            mismatches.append('seed missing')
        elif entry.get('seed') != ref.get('seed'):
            mismatches.append('seed mismatch')
    return mismatches
def _write_leaderboard_csv(path: Path, rows: Iterable[dict[str, Any]], *, metric_names: Iterable[str]=()) -> None:
    fieldnames = ['rank', 'model_family', 'ensemble_method', 'n_base_models', 'composite_score', 'best_score', 'primary_metric_ci_low', 'primary_metric_ci_mid', 'primary_metric_ci_high', 'primary_metric', 'primary_metric_source', 'task_type', *[name for name in metric_names], 'model_id', 'infer_selector', 'infer_target', 'reference_kind', 'infer_model_id', 'infer_train_task_id', 'train_task_id', 'preprocess_variant', 'model_variant', 'train_task_ref', 'processed_dataset_id', 'split_hash']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
def _resolve_infer_selector(entry: Mapping[str, Any]) -> str | None:
    if _normalize_str(entry.get('infer_model_id')):
        return 'infer.model_id'
    if _normalize_str(entry.get('infer_train_task_id')):
        return 'infer.train_task_id'
    return None
def _resolve_infer_target(entry: Mapping[str, Any]) -> str | None:
    selector = _resolve_infer_selector(entry)
    if selector == 'infer.model_id':
        return _normalize_str(entry.get('infer_model_id'))
    if selector == 'infer.train_task_id':
        return _normalize_str(entry.get('infer_train_task_id'))
    return None
def _copy_recommended_plot(cfg: Any, recommended: dict[str, Any], output_dir: Path, *, clearml_enabled: bool) -> Path | None:
    candidates = ['confusion_matrix.png', 'residuals.png', 'feature_importance.png', 'roc_curve.png']
    if clearml_enabled:
        task_id = _normalize_str(recommended.get('train_task_id') or recommended.get('train_task_ref'))
        if not task_id:
            return None
        for name in candidates:
            try:
                src = get_task_artifact_local_copy(cfg, task_id, name)
            except PlatformAdapterError:
                continue
            if not src.exists():
                continue
            dest = output_dir / 'recommended_plot.png'
            shutil.copy2(src, dest)
            return dest
    else:
        ref = _normalize_str(recommended.get('train_task_ref'))
        if not ref:
            return None
        run_dir = _resolve_run_dir(ref)
        for name in candidates:
            src = run_dir / name
            if not src.exists():
                continue
            dest = output_dir / 'recommended_plot.png'
            shutil.copy2(src, dest)
            return dest
    return None
LeaderboardRuntimeSettings = make_dataclass('LeaderboardRuntimeSettings', [('clearml_enabled', bool), ('refs', list[str]), ('require_comparable', bool), ('top_k', int), ('dry_run', bool), ('recommend_top_k', int), ('metric_source_priority', list[str]), ('allow_cross_metric_source', bool), ('allow_ensemble', bool), ('tie_breaker', str), ('expected_primary_metric', str | None), ('expected_direction', str | None), ('expected_seed', int | None), ('expected_task_type', str)], frozen=True)
def _resolve_runtime_settings(cfg: Any, *, clearml_enabled: bool) -> LeaderboardRuntimeSettings:
    lb_cfg = getattr(cfg, 'leaderboard', None)
    train_task_ids = _to_list(getattr(lb_cfg, 'train_task_ids', None))
    train_run_dirs = _to_list(getattr(lb_cfg, 'train_run_dirs', None))
    require_comparable = bool(getattr(lb_cfg, 'require_comparable', True))
    top_k = int(getattr(lb_cfg, 'top_k', 10) or 0)
    dry_run = bool(getattr(lb_cfg, 'dry_run', False))
    recommend_cfg = getattr(lb_cfg, 'recommend', None)
    recommend_top_k = int(getattr(recommend_cfg, 'top_k', 1) or 1) if recommend_cfg else 1
    if recommend_top_k < 1:
        recommend_top_k = 1
    metric_source_priority = _to_list(getattr(recommend_cfg, 'metric_source_priority', None))
    if not metric_source_priority:
        metric_source_priority = ['test', 'valid', 'meta_cv_on_valid']
    allow_cross_metric_source = bool(getattr(recommend_cfg, 'allow_cross_metric_source', False))
    allow_ensemble = bool(getattr(recommend_cfg, 'allow_ensemble', False))
    tie_breaker = _normalize_str(getattr(recommend_cfg, 'tie_breaker', None)) or 'prefer_simple'
    if clearml_enabled:
        refs = train_task_ids or _collect_clearml_refs(cfg)
        if not refs and (not dry_run):
            raise ValueError('leaderboard.train_task_ids is required when ClearML is enabled.')
    else:
        refs = train_run_dirs or train_task_ids or _scan_local_refs(cfg)
        if not refs and (not dry_run):
            raise ValueError('leaderboard.train_run_dirs (or train_task_ids) is required when ClearML is disabled.')
    expected_primary_metric = _normalize_str(getattr(getattr(cfg, 'eval', None), 'primary_metric', None))
    expected_direction = _normalize_direction(getattr(getattr(cfg, 'eval', None), 'direction', None))
    expected_seed = _normalize_int(getattr(getattr(cfg, 'eval', None), 'seed', None))
    expected_task_type = _normalize_str(getattr(getattr(cfg, 'eval', None), 'task_type', None)) or 'regression'
    return LeaderboardRuntimeSettings(clearml_enabled=clearml_enabled, refs=list(refs), require_comparable=require_comparable, top_k=top_k, dry_run=dry_run, recommend_top_k=recommend_top_k, metric_source_priority=metric_source_priority, allow_cross_metric_source=allow_cross_metric_source, allow_ensemble=allow_ensemble, tie_breaker=tie_breaker, expected_primary_metric=expected_primary_metric, expected_direction=expected_direction, expected_seed=expected_seed, expected_task_type=expected_task_type)
def _initial_ref_values(settings: LeaderboardRuntimeSettings) -> dict[str, Any]:
    return {'processed_dataset_id': None, 'split_hash': None, 'recipe_hash': None, 'primary_metric': settings.expected_primary_metric, 'direction': settings.expected_direction, 'seed': settings.expected_seed, 'task_type': settings.expected_task_type}
def _build_leaderboard_manifest_inputs(*, refs: list[str], require_comparable: bool, top_k: int, scoring_metrics: list[str], scoring_weights: dict[str, float], scoring_normalization: str, ranking_score_key: str, ranking_direction: str, ref_values: Mapping[str, Any], direction: str) -> dict[str, Any]:
    return {'train_task_refs': [str(ref) for ref in refs], 'require_comparable': require_comparable, 'top_k': top_k, 'scoring': {'metrics': scoring_metrics, 'weights': scoring_weights, 'normalization': scoring_normalization, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction}, 'primary_metric': ref_values.get('primary_metric'), 'direction': direction, 'seed': ref_values.get('seed'), 'task_type': ref_values.get('task_type'), 'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash')}
def _handle_dry_run_without_refs(ctx: Any, cfg: Any, settings: LeaderboardRuntimeSettings, ref_values: dict[str, Any]) -> bool:
    if not (settings.dry_run and (not settings.refs)):
        return False
    (scoring_metrics, scoring_weights, scoring_normalization, scoring_warnings) = _resolve_scoring_config(cfg)
    ranking_score_key = 'composite_score'
    ranking_direction = 'maximize'
    leaderboard_path = ctx.output_dir / 'leaderboard.csv'
    _write_leaderboard_csv(leaderboard_path, [], metric_names=scoring_metrics)
    recommendation = {'recommended_train_task_ref': None, 'recommended_train_task_id': None, 'recommended_model_id': None, 'recommended_registry_model_id': None, 'infer_model_id': None, 'infer_train_task_id': None, 'reference_kind': None, 'recommended_best_score': None, 'recommended_primary_metric': settings.expected_primary_metric, 'recommended_composite_score': None, 'recommended_metrics': {}, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction, 'scoring': {'metrics': scoring_metrics, 'weights': scoring_weights, 'normalization': scoring_normalization}, 'recommended_top_k': settings.recommend_top_k, 'recommendation_count': 0, 'recommended_models': []}
    recommendation_path = ctx.output_dir / 'recommendation.json'
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding='utf-8')
    summary_lines = ['# Leaderboard Summary', '', '- dry_run: true', f'- total_runs: {len(settings.refs)}', '- included: 0', '- excluded: 0', f'- require_comparable: {settings.require_comparable}', f'- ranking_score_key: {ranking_score_key}', f'- ranking_direction: {ranking_direction}', f'- scoring.normalization: {scoring_normalization}', f"- primary_metric: {settings.expected_primary_metric or 'unknown'}", f"- direction: {settings.expected_direction or 'unknown'}", f"- task_type: {settings.expected_task_type or 'unknown'}", f'- recommended_top_k: {settings.recommend_top_k}']
    if scoring_warnings:
        summary_lines.extend(['', '## Warnings'])
        summary_lines.extend([f'- {line}' for line in scoring_warnings])
    summary_path = ctx.output_dir / 'summary.md'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    decision_lines = ['# Decision Summary', '', '## Recommendation', '- recommended_model_id: n/a', '- infer_model_id: n/a', '- infer_train_task_id: n/a', '- train_task_ref: n/a', f"- primary_metric: {settings.expected_primary_metric or 'unknown'} ({settings.expected_direction or 'unknown'})", '- best_score: n/a', f'- ranking_score_key: {ranking_score_key} ({ranking_direction})', '', '## Scoring', f'- normalization: {scoring_normalization}', f"- metrics: {', '.join(scoring_metrics)}", f"- weights: {', '.join([f'{k}={scoring_weights.get(k)}' for k in scoring_metrics])}"]
    decision_summary_path = ctx.output_dir / 'decision_summary.md'
    decision_summary_path.write_text('\n'.join(decision_lines) + '\n', encoding='utf-8')
    decision_payload = {'recommended': {'model_id': None, 'registry_model_id': None, 'infer_model_id': None, 'infer_train_task_id': None, 'reference_kind': None, 'train_task_ref': None, 'train_task_id': None, 'best_score': None, 'primary_metric': settings.expected_primary_metric, 'primary_metric_ci': None, 'composite_score': None, 'metrics': {}, 'task_type': settings.expected_task_type}, 'scoring': {'metrics': scoring_metrics, 'weights': scoring_weights, 'normalization': scoring_normalization, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction}, 'comparability': {'require_comparable': settings.require_comparable, 'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash'), 'primary_metric': ref_values.get('primary_metric'), 'direction': settings.expected_direction, 'task_type': ref_values.get('task_type'), 'seed': ref_values.get('seed')}, 'leaderboard_csv': str(leaderboard_path), 'top_models': [], 'excluded_count': 0, 'warning_count': len(scoring_warnings)}
    decision_summary_json_path = ctx.output_dir / 'decision_summary.json'
    decision_summary_json_path.write_text(json.dumps(_stringify_payload(decision_payload), ensure_ascii=False, indent=2), encoding='utf-8')
    if settings.clearml_enabled:
        for (name, path) in [('leaderboard.csv', leaderboard_path), ('recommendation.json', recommendation_path), ('summary.md', summary_path), ('decision_summary.md', decision_summary_path), ('decision_summary.json', decision_summary_json_path)]:
            upload_artifact(ctx, name, path)
        selection_policy = f'composite_score:{scoring_normalization}'
        update_task_properties(ctx, {'recommended_train_task_id': None, 'recommended_model_id': None, 'recommended_registry_model_id': None, 'infer_model_id': None, 'infer_train_task_id': None, 'excluded_count': 0, 'selection_policy': selection_policy})
    out = {'leaderboard_csv': str(leaderboard_path), 'recommended_train_task_id': None, 'recommended_train_task_ref': None, 'recommended_model_id': None, 'recommended_registry_model_id': None, 'infer_model_id': None, 'infer_train_task_id': None, 'reference_kind': None, 'recommended_best_score': None, 'recommended_primary_metric': settings.expected_primary_metric, 'recommended_composite_score': None, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction, 'excluded_count': 0}
    if scoring_warnings:
        out['warnings'] = scoring_warnings
    inputs = _build_leaderboard_manifest_inputs(refs=[], require_comparable=settings.require_comparable, top_k=settings.top_k, scoring_metrics=scoring_metrics, scoring_weights=scoring_weights, scoring_normalization=scoring_normalization, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, ref_values=ref_values, direction=settings.expected_direction or 'unknown')
    outputs = {'leaderboard_csv': str(leaderboard_path), 'recommended_train_task_id': None, 'recommended_model_id': None, 'recommended_registry_model_id': None, 'infer_model_id': None, 'infer_train_task_id': None, 'recommended_composite_score': None, 'excluded_count': 0}
    split_hash = ref_values.get('split_hash') or 'unknown'
    recipe_hash = ref_values.get('recipe_hash') or 'unknown'
    emit_outputs_and_manifest(ctx, cfg, process='leaderboard', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, clearml_enabled=settings.clearml_enabled)
    return True
def _collect_entries(cfg: Any, *, refs: list[str], clearml_enabled: bool, require_comparable: bool, ref_values: dict[str, Any], expected_primary_metric: str | None, expected_direction: str | None, expected_seed: int | None) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str], list[str]]:
    entries: list[dict[str, Any]] = []
    excluded: list[str] = []
    skipped_entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    non_comparable: list[str] = []
    for ref in refs:
        ref_str = str(ref)
        entry: dict[str, Any] | None = None
        entry_warnings: list[str] = []
        entry_errors: list[str] = []
        if clearml_enabled:
            try:
                out_path = get_task_artifact_local_copy(cfg, ref, 'out.json')
                manifest_path = get_task_artifact_local_copy(cfg, ref, 'manifest.json')
            except PlatformAdapterError as exc:
                entry_errors.append(str(exc))
                out_path = None
                manifest_path = None
            if out_path is not None and manifest_path is not None:
                out = _load_json(out_path)
                manifest = _load_json(manifest_path)
                metrics_payload = None
                model_bundle_path = None
                ensemble_spec_payload = None
                try:
                    model_bundle_path = get_task_artifact_local_copy(cfg, ref, 'model_bundle.joblib')
                except PlatformAdapterError as exc:
                    entry_warnings.append(str(exc))
                try:
                    metrics_path = get_task_artifact_local_copy(cfg, ref, 'metrics.json')
                except PlatformAdapterError as exc:
                    entry_warnings.append(str(exc))
                    metrics_path = None
                if metrics_path is not None:
                    metrics_payload = _load_json(metrics_path)
                try:
                    spec_path = get_task_artifact_local_copy(cfg, ref, 'ensemble_spec.json')
                    ensemble_spec_payload = _load_json(spec_path)
                except PlatformAdapterError:
                    ensemble_spec_payload = None
                (entry, build_warnings, entry_errors) = _build_entry(out=out, manifest=manifest, metrics_payload=metrics_payload, ensemble_spec_payload=ensemble_spec_payload, train_task_ref=str(ref), model_bundle_path=model_bundle_path, expected_primary_metric=expected_primary_metric, expected_direction=expected_direction, expected_seed=expected_seed)
                entry_warnings.extend(build_warnings)
        else:
            run_dir = _resolve_run_dir(str(ref))
            if not run_dir.exists():
                entry_errors.append(f'train run dir not found: {run_dir}')
            else:
                out_path = run_dir / 'out.json'
                manifest_path = run_dir / 'manifest.json'
                if not out_path.exists() or not manifest_path.exists():
                    entry_errors.append(f'out.json/manifest.json missing under {run_dir}')
                else:
                    out = _load_json(out_path)
                    manifest = _load_json(manifest_path)
                    metrics_payload = None
                    model_bundle_path = _resolve_model_bundle_path(run_dir, _normalize_str(out.get('model_id')))
                    if model_bundle_path is None:
                        entry_warnings.append(f'model_bundle.joblib not found under {run_dir}')
                    metrics_path = run_dir / 'metrics.json'
                    if metrics_path.exists():
                        metrics_payload = _load_json(metrics_path)
                    else:
                        entry_warnings.append(f'metrics.json not found under {run_dir}')
                    ensemble_spec_payload = None
                    spec_path = run_dir / 'ensemble_spec.json'
                    if spec_path.exists():
                        try:
                            ensemble_spec_payload = _load_json(spec_path)
                        except (OSError, TypeError, ValueError) as exc:
                            entry_warnings.append(f'ensemble_spec.json invalid: {exc}')
                    (entry, build_warnings, entry_errors) = _build_entry(out=out, manifest=manifest, metrics_payload=metrics_payload, ensemble_spec_payload=ensemble_spec_payload, train_task_ref=str(run_dir), model_bundle_path=model_bundle_path, expected_primary_metric=expected_primary_metric, expected_direction=expected_direction, expected_seed=expected_seed)
                    entry_warnings.extend(build_warnings)
        for warning in entry_warnings:
            warnings.append(f'{ref_str}: {warning}')
        if entry_errors:
            excluded.append(ref_str)
            skipped_entries.append({'train_task_ref': ref_str, 'reason': 'entry_error', 'details': entry_errors})
            for error in entry_errors:
                warnings.append(f'{ref_str}: {error}')
            continue
        if entry is None:
            excluded.append(ref_str)
            skipped_entries.append({'train_task_ref': ref_str, 'reason': 'entry_missing', 'details': 'entry build failed'})
            warnings.append(f'{ref_str}: entry build failed')
            continue
        if clearml_enabled and entry.get('train_task_id') is None:
            entry['train_task_id'] = entry.get('train_task_ref')
        for key in ('processed_dataset_id', 'split_hash', 'recipe_hash', 'primary_metric', 'direction', 'seed', 'task_type'):
            if ref_values.get(key) is None and entry.get(key) is not None:
                ref_values[key] = entry.get(key)
        mismatches = _compare_comparability(entry, ref_values)
        if mismatches:
            if require_comparable:
                excluded.append(ref_str)
                skipped_entries.append({'train_task_ref': ref_str, 'reason': 'non_comparable', 'details': mismatches})
                warnings.append(f"{ref_str}: excluded ({', '.join(mismatches)})")
                continue
            non_comparable.append(ref_str)
            warnings.append(f"{ref_str}: non-comparable ({', '.join(mismatches)})")
        entries.append(entry)
    return (entries, excluded, skipped_entries, warnings, non_comparable)
def _score_and_rank_entries(cfg: Any, entries: list[dict[str, Any]], ref_values: dict[str, Any], warnings_list: list[str]) -> tuple[list[dict[str, Any]], list[str], dict[str, float], str, bool, str, str, str]:
    (scoring_metrics, scoring_weights, scoring_normalization, scoring_warnings) = _resolve_scoring_config(cfg)
    warnings_list.extend(scoring_warnings)
    metric_values: dict[str, list[float | None]] = {name: [] for name in scoring_metrics}
    for entry in entries:
        metrics_map = entry.get('metrics') or {}
        for name in scoring_metrics:
            metric_values[name].append(_to_float(metrics_map.get(name)))
    normalized_values: dict[str, list[float | None]] = {}
    for (name, values) in metric_values.items():
        if not any((value is not None for value in values)):
            warnings_list.append(f"metric '{name}' missing in all runs; skipping in composite score.")
            normalized_values[name] = [None for _ in values]
            continue
        normalized_values[name] = _normalize_metric_values(values, normalization=scoring_normalization)
    for (idx, entry) in enumerate(entries):
        for name in scoring_metrics:
            entry[name] = metric_values[name][idx]
        score_sum = 0.0
        weight_sum = 0.0
        used = False
        for name in scoring_metrics:
            weight = scoring_weights.get(name, 0.0)
            if weight == 0.0:
                continue
            normalized = normalized_values[name][idx]
            if normalized is None:
                continue
            score_sum += weight * normalized
            weight_sum += abs(weight)
            used = True
        entry['composite_score'] = score_sum / weight_sum if used and weight_sum > 0 else None
    use_composite = any((entry.get('composite_score') is not None for entry in entries))
    if not use_composite:
        warnings_list.append('Composite scoring unavailable; falling back to primary metric ranking.')
    direction = _normalize_direction(ref_values.get('direction')) or 'minimize'
    if direction not in ('minimize', 'maximize'):
        warnings_list.append(f'Invalid direction {direction}; defaulting to minimize.')
        direction = 'minimize'
    ranking_score_key = 'composite_score' if use_composite else 'best_score'
    ranking_direction = 'maximize' if use_composite else direction
    def _sort_key(item: dict[str, Any]) -> float:
        value = _to_float(item.get(ranking_score_key))
        if value is None:
            return -math.inf if ranking_direction == 'maximize' else math.inf
        return value
    entries_sorted = sorted(entries, key=_sort_key, reverse=ranking_direction == 'maximize')
    return (entries_sorted, scoring_metrics, scoring_weights, scoring_normalization, use_composite, ranking_score_key, ranking_direction, direction)
def _filter_recommend_candidates(source_entries: list[dict[str, Any]], *, metric_source_priority: list[str], allow_cross_metric_source: bool, allow_ensemble: bool) -> list[dict[str, Any]]:
    filtered = source_entries
    if not allow_cross_metric_source:
        for source in metric_source_priority:
            subset = [entry for entry in filtered if (entry.get('primary_metric_source') or 'valid') == source]
            if subset:
                filtered = subset
                break
    if not allow_ensemble:
        subset = [entry for entry in filtered if entry.get('model_family') != 'ensemble']
        if subset:
            filtered = subset
    return filtered
def _apply_recommend_tie_breaker(candidates: list[dict[str, Any]], *, tie_breaker: str, ranking_score_key: str) -> dict[str, Any]:
    if not candidates:
        raise ValueError('No candidates for tie-breaker.')
    if tie_breaker != 'prefer_simple':
        return candidates[0]
    top_score = _to_float(candidates[0].get(ranking_score_key))
    if top_score is None:
        return candidates[0]
    tie_group = []
    for entry in candidates:
        value = _to_float(entry.get(ranking_score_key))
        if value is None:
            break
        if math.isclose(value, top_score, rel_tol=1e-09, abs_tol=1e-12):
            tie_group.append(entry)
        else:
            break
    for entry in tie_group:
        if entry.get('model_family') == 'single':
            return entry
    return candidates[0]
def _build_leaderboard_rows(entries_sorted: list[dict[str, Any]], *, top_k: int, scoring_metrics: list[str]) -> tuple[list[dict[str, Any]], int]:
    effective_top_k = top_k if top_k > 0 else len(entries_sorted)
    rows = []
    for (idx, entry) in enumerate(entries_sorted[:effective_top_k], start=1):
        row = {'rank': idx, 'model_family': entry.get('model_family'), 'ensemble_method': entry.get('ensemble_method'), 'n_base_models': entry.get('n_base_models'), 'composite_score': entry.get('composite_score'), 'best_score': entry['best_score'], 'primary_metric_ci_low': entry.get('primary_metric_ci_low'), 'primary_metric_ci_mid': entry.get('primary_metric_ci_mid'), 'primary_metric_ci_high': entry.get('primary_metric_ci_high'), 'primary_metric': entry['primary_metric'], 'primary_metric_source': entry.get('primary_metric_source'), 'task_type': entry.get('task_type'), 'model_id': entry['model_id'], 'infer_selector': _resolve_infer_selector(entry), 'infer_target': _resolve_infer_target(entry), 'reference_kind': entry.get('reference_kind'), 'infer_model_id': entry.get('infer_model_id'), 'infer_train_task_id': entry.get('infer_train_task_id'), 'train_task_id': entry.get('train_task_id'), 'preprocess_variant': entry['preprocess_variant'], 'model_variant': entry['model_variant'], 'train_task_ref': entry['train_task_ref'], 'processed_dataset_id': entry['processed_dataset_id'], 'split_hash': entry['split_hash']}
        for name in scoring_metrics:
            row[name] = entry.get(name)
        rows.append(row)
    return (rows, effective_top_k)
def _build_recommendation(*, entries_sorted: list[dict[str, Any]], scoring_metrics: list[str], scoring_weights: dict[str, float], scoring_normalization: str, ranking_score_key: str, ranking_direction: str, metric_source_priority: list[str], allow_cross_metric_source: bool, allow_ensemble: bool, tie_breaker: str, recommend_top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    recommend_candidates = _filter_recommend_candidates(entries_sorted, metric_source_priority=metric_source_priority, allow_cross_metric_source=allow_cross_metric_source, allow_ensemble=allow_ensemble)
    if not recommend_candidates:
        recommend_candidates = entries_sorted
    recommended = _apply_recommend_tie_breaker(recommend_candidates, tie_breaker=tie_breaker, ranking_score_key=ranking_score_key)
    recommended_list = [recommended]
    if recommend_top_k > 1:
        for entry in recommend_candidates:
            if entry is recommended:
                continue
            recommended_list.append(entry)
            if len(recommended_list) >= recommend_top_k:
                break
    recommended_metrics = {name: recommended.get(name) for name in scoring_metrics}
    recommendation = {'recommended_train_task_ref': recommended['train_task_ref'], 'recommended_train_task_id': recommended.get('train_task_id'), 'recommended_model_id': recommended['model_id'], 'recommended_registry_model_id': recommended.get('registry_model_id'), 'infer_model_id': recommended.get('infer_model_id'), 'infer_train_task_id': recommended.get('infer_train_task_id'), 'reference_kind': recommended.get('reference_kind'), 'recommended_best_score': recommended['best_score'], 'recommended_primary_metric': recommended['primary_metric'], 'recommended_primary_metric_source': recommended.get('primary_metric_source'), 'recommended_model_family': recommended.get('model_family'), 'recommended_ensemble_method': recommended.get('ensemble_method'), 'recommended_n_base_models': recommended.get('n_base_models'), 'recommended_composite_score': recommended.get('composite_score'), 'recommended_metrics': recommended_metrics, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction, 'scoring': {'metrics': scoring_metrics, 'weights': scoring_weights, 'normalization': scoring_normalization}, 'recommended_top_k': recommend_top_k, 'recommendation_count': len(recommended_list), 'recommended_models': [{'rank': idx, 'train_task_ref': entry.get('train_task_ref'), 'train_task_id': entry.get('train_task_id'), 'model_id': entry.get('model_id'), 'registry_model_id': entry.get('registry_model_id'), 'infer_model_id': entry.get('infer_model_id'), 'infer_train_task_id': entry.get('infer_train_task_id'), 'reference_kind': entry.get('reference_kind'), 'best_score': entry.get('best_score'), 'primary_metric': entry.get('primary_metric'), 'primary_metric_source': entry.get('primary_metric_source'), 'composite_score': entry.get('composite_score'), 'processed_dataset_id': entry.get('processed_dataset_id'), 'split_hash': entry.get('split_hash'), 'recipe_hash': entry.get('recipe_hash'), 'model_family': entry.get('model_family'), 'ensemble_method': entry.get('ensemble_method')} for (idx, entry) in enumerate(recommended_list, start=1)], 'recommend_policy': {'metric_source_priority': metric_source_priority, 'allow_cross_metric_source': allow_cross_metric_source, 'allow_ensemble': allow_ensemble, 'tie_breaker': tie_breaker}}
    return (recommended, recommended_list, recommendation)
def _tag_recommended_registry_models(cfg: Any, ctx: Any, *, recommended: dict[str, Any], recommended_list: list[dict[str, Any]], recommendation: dict[str, Any], ref_values: dict[str, Any]) -> None:
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
    processed_dataset_id = _normalize_str(recommended.get('processed_dataset_id'))
    if processed_dataset_id is None:
        processed_dataset_id = _normalize_str(ref_values.get('processed_dataset_id'))
    split_hash = _normalize_str(recommended.get('split_hash')) or _normalize_str(ref_values.get('split_hash'))
    recipe_hash = _normalize_str(recommended.get('recipe_hash')) or _normalize_str(ref_values.get('recipe_hash'))
    leaderboard_task_id = clearml_task_id(ctx.task) if ctx.task is not None else None
    if processed_dataset_id and leaderboard_task_id:
        recommendations: list[tuple[str, list[str], dict[str, Any]]] = []
        for (idx, entry) in enumerate(recommended_list, start=1):
            registry_model_id = _normalize_str(entry.get('registry_model_id'))
            if not registry_model_id:
                warnings.warn(f'recommended model is missing registry_model_id (rank {idx}); skip registry tagging.')
                continue
            tags = ['leaderboard:recommended', f'task:leaderboard:{leaderboard_task_id}', f"recommend_metric:{entry.get('primary_metric')}", f'recommend_rank:{idx}']
            if recommendation.get('ranking_direction'):
                tags.append(f"recommend_direction:{recommendation.get('ranking_direction')}")
            metadata = {'recommend_rank': idx, 'recommend_score': entry.get('best_score'), 'recommend_metric': entry.get('primary_metric'), 'recommend_direction': recommendation.get('ranking_direction'), 'leaderboard_task_id': leaderboard_task_id, 'recommendation_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
            recommendations.append((registry_model_id, tags, metadata))
        if recommendations:
            try:
                update_recommended_registry_model_tags_multi(usecase_id=usecase_id, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, recommendations=recommendations, remove_prefixes=['leaderboard:recommended', 'task:leaderboard:', 'recommend_metric:', 'recommend_score:', 'recommend_rank:', 'recommend_direction:'])
            except (PlatformAdapterError, RuntimeError, TypeError, ValueError) as exc:
                warnings.warn(f'Failed to update registry recommendation tags: {exc}')
    else:
        if not processed_dataset_id:
            warnings.warn('processed_dataset_id is missing; skip registry tagging.')
        if not leaderboard_task_id:
            warnings.warn('leaderboard task id is unavailable; skip registry tagging.')
def _write_leaderboard_summary_and_skipped(ctx: Any, *, refs: list[str], entries_sorted: list[dict[str, Any]], excluded: list[dict[str, Any]], require_comparable: bool, ranking_score_key: str, ranking_direction: str, scoring_normalization: str, ref_values: dict[str, Any], direction: str, recommend_top_k: int, recommended_list: list[dict[str, Any]], rows: list[dict[str, Any]], warning_messages: list[str], skipped_entries: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary_lines = ['# Leaderboard Summary', '', f'- total_runs: {len(refs)}', f'- included: {len(entries_sorted)}', f'- excluded: {len(excluded)}', f'- require_comparable: {require_comparable}', f'- ranking_score_key: {ranking_score_key}', f'- ranking_direction: {ranking_direction}', f'- scoring.normalization: {scoring_normalization}', f"- primary_metric: {ref_values.get('primary_metric') or 'unknown'}", f'- direction: {direction}', f"- task_type: {ref_values.get('task_type') or 'unknown'}", f"- seed: {(ref_values.get('seed') if ref_values.get('seed') is not None else 'unknown')}", f"- processed_dataset_id: {ref_values.get('processed_dataset_id') or 'unknown'}", f"- split_hash: {ref_values.get('split_hash') or 'unknown'}", f'- recommended_top_k: {recommend_top_k}', f'- recommended_models: {len(recommended_list)}', '', '## Top Results']
    for row in rows:
        line = f"- rank {row['rank']}: best_score={row['best_score']} model_id={row['model_id']} train_task_ref={row['train_task_ref']}"
        if row.get('composite_score') is not None:
            line += f" composite_score={_format_float(row.get('composite_score'))}"
        if row['rank'] == 1:
            ci_low = row.get('primary_metric_ci_low')
            ci_high = row.get('primary_metric_ci_high')
            if ci_low is not None and ci_high is not None:
                line += f' ci=[{ci_low:.4g}, {ci_high:.4g}]'
        summary_lines.append(line)
    if warning_messages:
        summary_lines.extend(['', '## Warnings'])
        summary_lines.extend([f'- {line}' for line in warning_messages])
    summary_path = ctx.output_dir / 'summary.md'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    skipped_path = ctx.output_dir / 'leaderboard_skipped.json'
    skipped_path.write_text(json.dumps(skipped_entries, ensure_ascii=False, indent=2), encoding='utf-8')
    return (summary_path, skipped_path)
def _log_leaderboard_visuals(ctx: Any, *, rows: list[dict[str, Any]], scoring_metrics: list[str], ranking_score_key: str, ref_values: dict[str, Any], recommended: dict[str, Any], use_composite: bool) -> None:
    if not rows:
        return
    ranking_score_label = 'Composite Score' if use_composite else ref_values.get('primary_metric') or recommended.get('primary_metric') or 'score'
    best_primary_score = recommended.get('best_score')
    best_composite_score = recommended.get('composite_score')
    if use_composite and best_composite_score is not None:
        log_scalar(ctx.task, 'leaderboard', 'best_score', best_composite_score, step=0)
        if best_primary_score is not None:
            log_scalar(ctx.task, 'leaderboard', 'best_primary_score', best_primary_score, step=0)
    elif best_primary_score is not None:
        log_scalar(ctx.task, 'leaderboard', 'best_score', best_primary_score, step=0)
    table_fig = build_leaderboard_table(rows, metric_names=scoring_metrics, score_key=ranking_score_key, score_label=ranking_score_label, title='Leaderboard')
    if table_fig is not None:
        log_plotly(ctx.task, 'leaderboard', 'table', table_fig, step=0)
    else:
        log_debug_table(ctx.task, 'leaderboard', 'table', rows[:min(10, len(rows))], step=0)
    top_k_fig = build_top_k_bar(rows, score_key=ranking_score_key, score_label=ranking_score_label, title=f'Top-K {ranking_score_label}')
    fallback_path = None
    if top_k_fig is None:
        fallback_path = write_top_k_bar_png(rows, ctx.output_dir / 'top_k_scores.png', score_key=ranking_score_key, score_label=ranking_score_label, title=f'Top-K {ranking_score_label}')
    log_plotly(ctx.task, 'leaderboard', 'top_k_scores', top_k_fig or fallback_path, step=0)
    log_debug_table(ctx.task, 'leaderboard', 'top_k_table', rows[:min(10, len(rows))], step=0)
    pareto_fig = build_pareto_scatter(rows, x_metric='r2', y_metric='rmse')
    if pareto_fig is not None:
        log_plotly(ctx.task, 'leaderboard', 'pareto', pareto_fig, step=0)
def _write_decision_summaries(cfg: Any, ctx: Any, *, clearml_enabled: bool, rows: list[dict[str, Any]], recommended: dict[str, Any], recommended_metrics: dict[str, Any], direction: str, ranking_score_key: str, ranking_direction: str, require_comparable: bool, ref_values: dict[str, Any], excluded: list[dict[str, Any]], warning_messages: list[str], metric_source_priority: list[str], allow_cross_metric_source: bool, allow_ensemble: bool, tie_breaker: str, scoring_normalization: str, scoring_metrics: list[str], scoring_weights: dict[str, float], leaderboard_path: Path) -> tuple[Path, Path]:
    max_models = min(5, len(rows))
    decision_rows = rows[:max_models]
    recommended_ci = _format_ci_interval(recommended.get('primary_metric_ci'))
    decision_lines = ['# Decision Summary', '', '## Recommendation', f"- recommended_model_id: {recommended.get('model_id')}", f"- infer_model_id: {recommended.get('infer_model_id')}", f"- infer_train_task_id: {recommended.get('infer_train_task_id')}", f"- reference_kind: {recommended.get('reference_kind')}", f"- train_task_ref: {recommended.get('train_task_ref')}", f"- primary_metric: {recommended.get('primary_metric')} ({direction})", f"- primary_metric_source: {recommended.get('primary_metric_source')}", f"- best_score: {_format_float(recommended.get('best_score'))}", f'- ranking_score_key: {ranking_score_key} ({ranking_direction})']
    if recommended_ci:
        decision_lines.append(f'- primary_metric_ci: {recommended_ci}')
    if recommended.get('composite_score') is not None:
        decision_lines.append(f"- composite_score: {_format_float(recommended.get('composite_score'))}")
    if recommended.get('task_type'):
        decision_lines.append(f"- task_type: {recommended.get('task_type')}")
    if recommended.get('n_classes') is not None:
        decision_lines.append(f"- n_classes: {recommended.get('n_classes')}")
    if recommended.get('model_family'):
        decision_lines.append(f"- model_family: {recommended.get('model_family')}")
    if recommended.get('ensemble_method'):
        decision_lines.append(f"- ensemble_method: {recommended.get('ensemble_method')}")
    decision_lines.extend(['', '## Comparability', f'- require_comparable: {require_comparable}', f"- processed_dataset_id: {ref_values.get('processed_dataset_id') or 'unknown'}", f"- split_hash: {ref_values.get('split_hash') or 'unknown'}", f"- recipe_hash: {ref_values.get('recipe_hash') or 'unknown'}", f"- primary_metric: {ref_values.get('primary_metric') or 'unknown'}", f'- direction: {direction}', f"- task_type: {ref_values.get('task_type') or 'unknown'}", f"- seed: {(ref_values.get('seed') if ref_values.get('seed') is not None else 'unknown')}", f'- excluded_count: {len(excluded)}'])
    if warning_messages:
        decision_lines.append(f'- warning_count: {len(warning_messages)} (see summary.md)')
    decision_lines.extend(['', '## Recommend Policy', f"- metric_source_priority: {', '.join(metric_source_priority)}", f'- allow_cross_metric_source: {allow_cross_metric_source}', f'- allow_ensemble: {allow_ensemble}', f'- tie_breaker: {tie_breaker}', '', '## Scoring', f'- normalization: {scoring_normalization}', f"- metrics: {', '.join(scoring_metrics)}", f"- weights: {', '.join([f'{k}={scoring_weights.get(k)}' for k in scoring_metrics])}", '', '## Top Models', f'- source: {leaderboard_path.name}', '', '| rank | model_variant | preprocess_variant | composite_score | best_score | primary_metric | ci |', '| --- | --- | --- | --- | --- | --- | --- |'])
    for row in decision_rows:
        ci = _format_ci_interval({'low': row.get('primary_metric_ci_low'), 'mid': row.get('primary_metric_ci_mid'), 'high': row.get('primary_metric_ci_high')}) or 'n/a'
        decision_lines.append('| {rank} | {model_variant} | {preprocess_variant} | {composite} | {best_score} | {metric} | {ci} |'.format(rank=row.get('rank'), model_variant=row.get('model_variant') or 'unknown', preprocess_variant=row.get('preprocess_variant') or 'unknown', composite=_format_float(row.get('composite_score')), best_score=_format_float(row.get('best_score')), metric=row.get('primary_metric') or 'unknown', ci=ci))
    decision_lines.extend(['', '## Extra Capabilities'])
    thresholding = recommended.get('thresholding') or {}
    if thresholding.get('best_threshold') is not None:
        decision_lines.append('- thresholding: enabled metric={metric} best_threshold={thr} score={score}'.format(metric=thresholding.get('metric') or 'unknown', thr=_format_float(thresholding.get('best_threshold')), score=_format_float(thresholding.get('score'))))
    else:
        decision_lines.append('- thresholding: disabled')
    calibration = recommended.get('calibration') or {}
    if calibration.get('enabled'):
        decision_lines.append('- calibration: enabled method={method} mode={mode}'.format(method=calibration.get('method') or 'unknown', mode=calibration.get('mode') or 'unknown'))
    else:
        decision_lines.append('- calibration: disabled')
    uncertainty = recommended.get('uncertainty') or {}
    if uncertainty.get('enabled'):
        decision_lines.append('- uncertainty: enabled method={method} alpha={alpha} q={q}'.format(method=uncertainty.get('method') or 'unknown', alpha=_format_float(uncertainty.get('alpha')), q=_format_float(uncertainty.get('q'))))
    else:
        decision_lines.append('- uncertainty: disabled')
    imbalance = recommended.get('imbalance') or {}
    if imbalance.get('enabled'):
        decision_lines.append('- imbalance_handling: enabled strategy={strategy} applied={applied}'.format(strategy=imbalance.get('strategy') or 'unknown', applied=imbalance.get('applied')))
    else:
        decision_lines.append('- imbalance_handling: disabled')
    decision_lines.extend(['', '## Inference Selection', '```bash'])
    if recommended.get('infer_model_id'):
        decision_lines.append(f"python -m tabular_analysis.cli task=infer infer.model_id={recommended.get('infer_model_id')}")
    elif recommended.get('infer_train_task_id'):
        decision_lines.append(f"python -m tabular_analysis.cli task=infer infer.train_task_id={recommended.get('infer_train_task_id')}")
    decision_lines.append('```')
    decision_summary_path = ctx.output_dir / 'decision_summary.md'
    decision_summary_path.write_text('\n'.join(decision_lines) + '\n', encoding='utf-8')
    decision_payload = {'recommended': {'model_id': recommended.get('model_id'), 'registry_model_id': recommended.get('registry_model_id'), 'infer_model_id': recommended.get('infer_model_id'), 'infer_train_task_id': recommended.get('infer_train_task_id'), 'reference_kind': recommended.get('reference_kind'), 'train_task_ref': recommended.get('train_task_ref'), 'train_task_id': recommended.get('train_task_id'), 'best_score': recommended.get('best_score'), 'primary_metric': recommended.get('primary_metric'), 'primary_metric_source': recommended.get('primary_metric_source'), 'primary_metric_ci': recommended.get('primary_metric_ci'), 'composite_score': recommended.get('composite_score'), 'metrics': recommended_metrics, 'model_family': recommended.get('model_family'), 'ensemble_method': recommended.get('ensemble_method'), 'n_base_models': recommended.get('n_base_models'), 'task_type': recommended.get('task_type'), 'n_classes': recommended.get('n_classes'), 'class_labels': recommended.get('class_labels'), 'thresholding': recommended.get('thresholding'), 'calibration': recommended.get('calibration'), 'imbalance': recommended.get('imbalance'), 'uncertainty': recommended.get('uncertainty')}, 'scoring': {'metrics': scoring_metrics, 'weights': scoring_weights, 'normalization': scoring_normalization, 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction}, 'recommend_policy': {'metric_source_priority': metric_source_priority, 'allow_cross_metric_source': allow_cross_metric_source, 'allow_ensemble': allow_ensemble, 'tie_breaker': tie_breaker}, 'comparability': {'require_comparable': require_comparable, 'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash'), 'primary_metric': ref_values.get('primary_metric'), 'direction': direction, 'task_type': ref_values.get('task_type'), 'seed': ref_values.get('seed')}, 'leaderboard_csv': str(leaderboard_path), 'top_models': rows[:min(10, len(rows))], 'excluded_count': len(excluded), 'warning_count': len(warning_messages)}
    decision_summary_json_path = ctx.output_dir / 'decision_summary.json'
    decision_summary_json_path.write_text(json.dumps(_stringify_payload(decision_payload), ensure_ascii=False, indent=2), encoding='utf-8')
    if bool(_cfg_value(cfg, 'viz.enabled', True)) and (not clearml_enabled):
        _copy_recommended_plot(cfg, recommended, ctx.output_dir, clearml_enabled=False)
    return (decision_summary_path, decision_summary_json_path)
def _write_leaderboard_out_and_manifest(cfg: Any, ctx: Any, *, clearml_enabled: bool, use_composite: bool, scoring_normalization: str, ref_values: dict[str, Any], recommended: dict[str, Any], excluded: list[dict[str, Any]], non_comparable: list[dict[str, Any]], warning_messages: list[str], rows: list[dict[str, Any]], refs: list[str], require_comparable: bool, top_k: int, scoring_metrics: list[str], scoring_weights: dict[str, float], ranking_score_key: str, ranking_direction: str, direction: str, leaderboard_path: Path, recommendation_path: Path, summary_path: Path, skipped_path: Path, decision_summary_path: Path, decision_summary_json_path: Path) -> None:
    if clearml_enabled:
        for (name, path) in [('leaderboard.csv', leaderboard_path), ('recommendation.json', recommendation_path), ('summary.md', summary_path), ('leaderboard_skipped.json', skipped_path), ('decision_summary.md', decision_summary_path), ('decision_summary.json', decision_summary_json_path)]:
            upload_artifact(ctx, name, path)
        selection_policy = f'composite_score:{scoring_normalization}' if use_composite else f"primary_metric:{ref_values.get('primary_metric') or 'unknown'}"
        properties_payload = {'recommended_train_task_id': recommended.get('train_task_id') or None, 'recommended_model_id': recommended.get('model_id'), 'recommended_registry_model_id': recommended.get('registry_model_id'), 'infer_model_id': recommended.get('infer_model_id'), 'infer_train_task_id': recommended.get('infer_train_task_id'), 'recommended_primary_metric_source': recommended.get('primary_metric_source'), 'excluded_count': len(excluded), 'selection_policy': selection_policy}
        if recommended.get('composite_score') is not None:
            properties_payload['recommended_composite_score'] = recommended.get('composite_score')
        update_task_properties(ctx, properties_payload)
    out = {'leaderboard_csv': str(leaderboard_path), 'leaderboard_skipped': str(skipped_path), 'recommended_train_task_id': recommended.get('train_task_id') or None, 'recommended_train_task_ref': recommended.get('train_task_ref'), 'recommended_model_id': recommended.get('model_id'), 'recommended_registry_model_id': recommended.get('registry_model_id'), 'infer_model_id': recommended.get('infer_model_id'), 'infer_train_task_id': recommended.get('infer_train_task_id'), 'reference_kind': recommended.get('reference_kind'), 'recommended_best_score': recommended.get('best_score'), 'recommended_primary_metric': recommended.get('primary_metric'), 'recommended_primary_metric_source': recommended.get('primary_metric_source'), 'recommended_model_family': recommended.get('model_family'), 'recommended_ensemble_method': recommended.get('ensemble_method'), 'recommended_n_base_models': recommended.get('n_base_models'), 'recommended_composite_score': recommended.get('composite_score'), 'ranking_score_key': ranking_score_key, 'ranking_direction': ranking_direction, 'excluded_count': len(excluded)}
    if non_comparable:
        out['non_comparable_count'] = len(non_comparable)
    if warning_messages:
        out['warnings'] = warning_messages
    inputs = _build_leaderboard_manifest_inputs(refs=[str(ref) for ref in refs], require_comparable=require_comparable, top_k=top_k, scoring_metrics=scoring_metrics, scoring_weights=scoring_weights, scoring_normalization=scoring_normalization, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, ref_values=ref_values, direction=direction)
    outputs = {'leaderboard_csv': str(leaderboard_path), 'recommended_train_task_id': recommended.get('train_task_id') or None, 'recommended_model_id': recommended.get('model_id'), 'recommended_registry_model_id': recommended.get('registry_model_id'), 'infer_model_id': recommended.get('infer_model_id'), 'infer_train_task_id': recommended.get('infer_train_task_id'), 'recommended_primary_metric_source': recommended.get('primary_metric_source'), 'recommended_composite_score': recommended.get('composite_score'), 'excluded_count': len(excluded)}
    emit_outputs_and_manifest(ctx, cfg, process='leaderboard', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash')}, clearml_enabled=clearml_enabled)
def run(cfg: Any) -> None:
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='leaderboard', tags=identity.tags, properties=identity.user_properties)
    clearml_enabled = is_clearml_enabled(cfg)
    settings = _resolve_runtime_settings(cfg, clearml_enabled=clearml_enabled)
    refs = settings.refs
    require_comparable = settings.require_comparable
    top_k = settings.top_k
    recommend_top_k = settings.recommend_top_k
    metric_source_priority = settings.metric_source_priority
    allow_cross_metric_source = settings.allow_cross_metric_source
    allow_ensemble = settings.allow_ensemble
    tie_breaker = settings.tie_breaker
    expected_primary_metric = settings.expected_primary_metric
    expected_direction = settings.expected_direction
    expected_seed = settings.expected_seed
    connect_leaderboard(ctx, cfg, primary_metric=expected_primary_metric, direction=expected_direction, require_comparable=require_comparable, top_k=top_k, recommend_top_k=recommend_top_k)
    ref_values = _initial_ref_values(settings)
    if _handle_dry_run_without_refs(ctx, cfg, settings, ref_values):
        return
    (entries, excluded, skipped_entries, warnings, non_comparable) = _collect_entries(cfg, refs=refs, clearml_enabled=clearml_enabled, require_comparable=require_comparable, ref_values=ref_values, expected_primary_metric=expected_primary_metric, expected_direction=expected_direction, expected_seed=expected_seed)
    if not entries:
        raise ValueError('No comparable train runs found for leaderboard.')
    (entries_sorted, scoring_metrics, scoring_weights, scoring_normalization, use_composite, ranking_score_key, ranking_direction, direction) = _score_and_rank_entries(cfg, entries, ref_values, warnings)
    (rows, top_k) = _build_leaderboard_rows(entries_sorted, top_k=top_k, scoring_metrics=scoring_metrics)
    leaderboard_path = ctx.output_dir / 'leaderboard.csv'
    _write_leaderboard_csv(leaderboard_path, rows, metric_names=scoring_metrics)
    (recommended, recommended_list, recommendation) = _build_recommendation(entries_sorted=entries_sorted, scoring_metrics=scoring_metrics, scoring_weights=scoring_weights, scoring_normalization=scoring_normalization, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, metric_source_priority=metric_source_priority, allow_cross_metric_source=allow_cross_metric_source, allow_ensemble=allow_ensemble, tie_breaker=tie_breaker, recommend_top_k=recommend_top_k)
    recommended_metrics = recommendation.get('recommended_metrics', {})
    recommendation_path = ctx.output_dir / 'recommendation.json'
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding='utf-8')
    if clearml_enabled:
        _tag_recommended_registry_models(cfg, ctx, recommended=recommended, recommended_list=recommended_list, recommendation=recommendation, ref_values=ref_values)
    (summary_path, skipped_path) = _write_leaderboard_summary_and_skipped(ctx, refs=refs, entries_sorted=entries_sorted, excluded=excluded, require_comparable=require_comparable, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, scoring_normalization=scoring_normalization, ref_values=ref_values, direction=direction, recommend_top_k=recommend_top_k, recommended_list=recommended_list, rows=rows, warning_messages=warnings, skipped_entries=skipped_entries)
    if clearml_enabled:
        _log_leaderboard_visuals(ctx, rows=rows, scoring_metrics=scoring_metrics, ranking_score_key=ranking_score_key, ref_values=ref_values, recommended=recommended, use_composite=use_composite)
    (decision_summary_path, decision_summary_json_path) = _write_decision_summaries(cfg, ctx, clearml_enabled=clearml_enabled, rows=rows, recommended=recommended, recommended_metrics=recommended_metrics, direction=direction, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, require_comparable=require_comparable, ref_values=ref_values, excluded=excluded, warning_messages=warnings, metric_source_priority=metric_source_priority, allow_cross_metric_source=allow_cross_metric_source, allow_ensemble=allow_ensemble, tie_breaker=tie_breaker, scoring_normalization=scoring_normalization, scoring_metrics=scoring_metrics, scoring_weights=scoring_weights, leaderboard_path=leaderboard_path)
    _write_leaderboard_out_and_manifest(cfg, ctx, clearml_enabled=clearml_enabled, use_composite=use_composite, scoring_normalization=scoring_normalization, ref_values=ref_values, recommended=recommended, excluded=excluded, non_comparable=non_comparable, warning_messages=warnings, rows=rows, refs=refs, require_comparable=require_comparable, top_k=top_k, scoring_metrics=scoring_metrics, scoring_weights=scoring_weights, ranking_score_key=ranking_score_key, ranking_direction=ranking_direction, direction=direction, leaderboard_path=leaderboard_path, recommendation_path=recommendation_path, summary_path=summary_path, skipped_path=skipped_path, decision_summary_path=decision_summary_path, decision_summary_json_path=decision_summary_json_path)
