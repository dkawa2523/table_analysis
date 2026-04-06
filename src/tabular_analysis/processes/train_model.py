from __future__ import annotations
from ..common.collection_utils import dedupe_texts as _dedupe_tags, stringify_payload as _stringify_payload, to_container as _to_container, to_list as _to_list
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str, normalize_task_type as _normalize_task_type
from ..common.dataset_utils import load_tabular_frame as _load_dataframe, select_tabular_file as _select_tabular_file
from ..common.json_utils import load_json as _load_json
from ..common.omegaconf_utils import ensure_config_alias
from ..common.probability_utils import extract_positive_class_proba
from dataclasses import make_dataclass
import json
import math
from pathlib import Path
from typing import Any
import warnings
from ..clearml.datasets import get_processed_dataset_local_copy, get_raw_dataset_local_copy
from ..clearml.hparams import connect_train_model
from ..clearml.naming import apply_train_model_naming
from ..clearml.ui_logger import log_debug_table, log_plotly, log_scalar
from ..feature_engineering.categorical import encode_target_for_mean
from ..io.bundle_io import load_bundle, save_bundle
from ..metrics.regression import REGRESSION_METRIC_ORDER, compute_regression_metrics
from ..monitoring.drift import build_train_profile
from .drift_report import annotate_profile, resolve_drift_settings, sample_frame
from .train_shared import apply_resampling as _apply_resampling, apply_weight_strategy as _apply_weight_strategy, bootstrap_metric_ci as _bootstrap_metric_ci, build_calibration_report as _build_calibration_report, build_plotly_confusion_matrix as _build_plotly_confusion_matrix, build_plotly_roc_curve as _build_plotly_roc_curve, build_prediction_sample as _build_prediction_sample, calibrate_classifier as _calibrate_classifier, is_binary_only_metric as _is_binary_only_metric, normalize_train_key as _normalize_train_key, resolve_calibration_settings as _resolve_calibration_settings, resolve_ci_settings as _resolve_ci_settings, resolve_classification_metrics as _resolve_classification_metrics, resolve_classification_mode as _resolve_classification_mode, resolve_imbalance_settings as _resolve_imbalance_settings, resolve_regression_metrics as _resolve_regression_metrics, resolve_thresholding_settings as _resolve_thresholding_settings, resolve_uncertainty_settings as _resolve_uncertainty_settings, resolve_viz_settings as _resolve_viz_settings, select_best_threshold as _select_best_threshold
from ..ops.clearml_identity import apply_clearml_identity
from ..platform_adapter_artifacts import resolve_output_dir, upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled, resolve_version_props
from ..platform_adapter_task import get_task_artifact_local_copy, register_model_artifact, update_task_properties
from ..registry.metrics import get_metric, metric_direction, metric_requires_proba, metric_supports_thresholding
from ..registry.models import ModelWeightsUnavailableError, build_model
from ..uncertainty.conformal import compute_split_conformal_quantile
from .lifecycle import emit_outputs_and_manifest, start_runtime
from ..viz.plots import plot_confusion_matrix, plot_feature_importance, plot_interval_width_histogram, plot_reliability_curve, plot_regression_residuals, plot_roc_curve, write_confusion_matrix_csv
from ..viz.regression_plots import build_regression_metrics_table, build_residuals_plot, build_true_pred_scatter
from ..viz.render_common import plotly_go as _plotly_go
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError, json.JSONDecodeError)
def _resolve_preprocess_provenance(preprocess_run_dir: Path, assets_dir: Path | None, preprocess_out: dict[str, Any] | None, preprocess_bundle: Any) -> dict[str, Any]:
    raw_dataset_id = None
    preprocess_variant = None
    if isinstance(preprocess_out, dict):
        preprocess_variant = _normalize_str(preprocess_out.get('preprocess_variant'))
    if preprocess_variant is None and isinstance(preprocess_bundle, dict):
        preprocess_variant = _normalize_str(preprocess_bundle.get('preprocess_variant'))
    for candidate in (assets_dir, preprocess_run_dir):
        if candidate is None:
            continue
        meta_path = candidate / 'meta.json'
        if not meta_path.exists():
            continue
        meta_payload = _load_json(meta_path)
        if raw_dataset_id is None:
            raw_dataset_id = _normalize_str(meta_payload.get('raw_dataset_id'))
        if preprocess_variant is None:
            preprocess_variant = _normalize_str(meta_payload.get('preprocess_variant'))
        break
    manifest_path = preprocess_run_dir / 'manifest.json'
    if manifest_path.exists():
        manifest_payload = _load_json(manifest_path)
        inputs = manifest_payload.get('inputs') or {}
        if raw_dataset_id is None:
            raw_dataset_id = _normalize_str(inputs.get('raw_dataset_id'))
    return {'raw_dataset_id': raw_dataset_id, 'preprocess_variant': preprocess_variant}
def _format_float(value: Any) -> str:
    if value is None:
        return 'n/a'
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        return 'n/a'
    if not math.isfinite(num):
        return 'n/a'
    return f'{num:.6g}'
def _resolve_preprocess_run_dir(cfg: Any, processed_ref_path: Path | None) -> Path:
    candidate: str | None = None
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None:
        for key in ('train.inputs.preprocess_run_dir', 'train.preprocess_run_dir', 'inputs.preprocess_run_dir'):
            value = OmegaConf.select(cfg, key)
            if value:
                candidate = str(value)
                break
    if candidate:
        return Path(candidate).expanduser().resolve()
    if processed_ref_path is not None:
        if processed_ref_path.is_dir():
            return processed_ref_path
        if processed_ref_path.is_file():
            return processed_ref_path.parent
    run_cfg = getattr(cfg, 'run', None)
    base_output_dir = Path(getattr(run_cfg, 'output_dir', 'outputs'))
    candidates = [base_output_dir / '02_preprocess']
    name = base_output_dir.name
    if len(name) >= 3 and name[:2].isdigit() and (name[2] == '_'):
        parent = base_output_dir.parent
        candidates.append(parent / '02_preprocess')
        candidates.append(parent / '02_preprocess' / '02_preprocess')
    for candidate in candidates:
        if (candidate / 'out.json').exists():
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    search_root = base_output_dir.parent
    if search_root.exists():
        latest_candidates = [path.parent for path in search_root.glob('*/02_preprocess/out.json')]
        if latest_candidates:
            try:
                return max(latest_candidates, key=lambda item: item.stat().st_mtime)
            except (OSError, ValueError):
                return latest_candidates[-1]
    return resolve_output_dir(cfg, '02_preprocess')
def _rebuild_processed_from_raw(cfg: Any, *, base_dir: Path, preprocess_bundle: dict[str, Any], split_payload: dict[str, Any], task_type: str) -> tuple[Any, str]:
    meta_path = base_dir / 'meta.json'
    meta = _load_json(meta_path) if meta_path.exists() else {}
    raw_dataset_id = _normalize_str(meta.get('raw_dataset_id')) or _normalize_str(_cfg_value(cfg, 'data.raw_dataset_id'))
    dataset_path_value = _normalize_str(_cfg_value(cfg, 'data.dataset_path'))
    if raw_dataset_id and (not raw_dataset_id.startswith('local:')):
        raw_dir = get_raw_dataset_local_copy(cfg, raw_dataset_id)
        dataset_file = _select_tabular_file(raw_dir)
    elif dataset_path_value:
        dataset_file = _select_tabular_file(Path(dataset_path_value).expanduser().resolve())
    else:
        raise FileNotFoundError('raw dataset not available; set data.dataset_path or provide raw_dataset_id in meta.json.')
    raw_df = _load_dataframe(dataset_file)
    columns_info = preprocess_bundle.get('columns') or {}
    feature_columns = columns_info.get('feature_columns') or []
    target_column = _normalize_str(columns_info.get('target_column')) or _normalize_str(_cfg_value(cfg, 'data.target_column'))
    if not target_column or target_column not in raw_df.columns:
        raise ValueError('target_column not found in raw dataset for rebuild.')
    if not feature_columns:
        raise ValueError('feature_columns missing in preprocess_bundle; cannot rebuild features.')
    pipeline = preprocess_bundle.get('pipeline')
    if pipeline is None:
        raise ValueError('preprocess_bundle.pipeline is missing.')
    recipe_path = base_dir / 'recipe.json'
    recipe_payload = _load_json(recipe_path) if recipe_path.exists() else {}
    encoding_cfg = recipe_payload.get('categorical_encoding') or {}
    encoding = _normalize_str(encoding_cfg.get('encoding'))
    target_mean_cfg = encoding_cfg.get('target_mean_oof') or {}
    target_mean_folds = int(target_mean_cfg.get('folds') or 5)
    split_seed = int(split_payload.get('seed') or _cfg_value(cfg, 'data.split.seed', 42) or 42)
    train_idx = _normalize_indices(split_payload.get('train_index'), label='train_index')
    if encoding == 'target_mean_oof':
        (y_encoded, _) = encode_target_for_mean(train_values=raw_df.iloc[train_idx][target_column], all_values=raw_df[target_column], task_type=task_type)
        transformed = pipeline.transform_with_oof(raw_df[feature_columns], y_encoded, train_idx=train_idx, folds=target_mean_folds, seed=split_seed, task_type=task_type)
    else:
        transformed = pipeline.transform(raw_df[feature_columns])
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    feature_names = preprocess_bundle.get('feature_names')
    if not feature_names:
        try:
            feature_names = list(pipeline.get_feature_names_out())
        except _RECOVERABLE_ERRORS:
            feature_names = [f'f{i}' for i in range(int(getattr(transformed, 'shape', [0, 0])[1]))]
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for rebuild.') from exc
    processed_df = pd.DataFrame(transformed, columns=feature_names)
    processed_df[target_column] = raw_df[target_column].to_numpy()
    return (processed_df, target_column)
def _resolve_task_id(ctx) -> str | None:
    if ctx.task is None:
        return None
    for attr in ('id', 'task_id'):
        if value := getattr(ctx.task, attr, None):
            return str(value)
    return None
def _build_registry_tags(*, usecase_id: str, process: str, processed_dataset_id: str, split_hash: str, recipe_hash: str, preprocess_variant: str, model_variant: str, task_type: str | None, train_task_id: str | None, preprocess_task_id: str | None, pipeline_task_id: str | None) -> list[str]:
    tags = [f'usecase:{usecase_id}', f'process:{process}', f'dataset:{processed_dataset_id}', f'split:{split_hash}', f'recipe:{recipe_hash}', f'preprocess:{preprocess_variant}', f'model_variant:{model_variant}']
    if task_type:
        tags.append(f'task_type:{task_type}')
    if train_task_id:
        tags.append(f'task:train_model:{train_task_id}')
    if preprocess_task_id:
        tags.append(f'task:preprocess:{preprocess_task_id}')
    if pipeline_task_id:
        tags.append(f'task:pipeline:{pipeline_task_id}')
    return _dedupe_tags(tags)
def _normalize_indices(values: Any, *, label: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f'{label} must be a list of indices.')
    return [int(v) for v in values]
def _merge_model_variant(cfg: Any) -> dict[str, Any]:
    variant = _to_container(getattr(cfg, 'model_variant', None))
    if not variant:
        try:
            from omegaconf import OmegaConf
        except _RECOVERABLE_ERRORS:
            OmegaConf = None
        if OmegaConf is not None:
            variant = OmegaConf.select(cfg, 'group.model.model_variant')
        variant = _to_container(variant) or {}
    if not isinstance(variant, dict):
        raise TypeError('model_variant must be a dict-like object.')
    params = _to_container(variant.get('params') or {}) or {}
    train_params = _to_container(getattr(getattr(cfg, 'train', None), 'params', {}) or {}) or {}
    if not isinstance(params, dict) or not isinstance(train_params, dict):
        raise TypeError('model params must be dicts.')
    merged = {**params, **train_params}
    merged_variant = dict(variant)
    merged_variant['params'] = merged
    if not merged_variant.get('name'):
        merged_variant['name'] = _normalize_str(getattr(getattr(cfg, 'train', None), 'model', None))
    return merged_variant
def _extract_feature_importance(model: Any, feature_names: list[str] | None) -> tuple[list[str], list[float]] | None:
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS:
        return None
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = getattr(model, 'feature_importances_', None)
    elif hasattr(model, 'coef_'):
        coef = getattr(model, 'coef_', None)
        if coef is not None:
            coef_arr = np.asarray(coef)
            if coef_arr.ndim > 1:
                coef_arr = np.mean(np.abs(coef_arr), axis=0)
            importance = np.abs(coef_arr)
    if importance is None:
        return None
    importance_arr = np.asarray(importance).reshape(-1)
    names = list(feature_names or [])
    if len(names) != len(importance_arr):
        names = [f'feature_{idx}' for idx in range(len(importance_arr))]
    return (names, [float(v) for v in importance_arr.tolist()])
def _write_feature_importance_csv(names: list[str], scores: list[float], output_dir: Path) -> Path:
    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    lines = ['feature,importance']
    for (name, score) in pairs:
        lines.append(f'{name},{float(score)}')
    path = output_dir / 'feature_importance.csv'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path
def _build_plotly_feature_importance(names: list[str], scores: list[float], *, top_n: int) -> Any | None:
    go = _plotly_go()
    if go is None:
        return None
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS:
        return None
    values = np.asarray(scores, dtype=float).reshape(-1)
    order = np.argsort(values)[::-1]
    if top_n <= 0 or top_n > len(order):
        top_n = len(order)
    order = order[:top_n]
    labels = [str(names[idx]) for idx in order]
    values = values[order]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color='#4C78A8'))
    fig.update_layout(title='Feature Importance', xaxis_title='importance', yaxis_title='feature', yaxis=dict(autorange='reversed'), margin=dict(l=40, r=20, t=40, b=40))
    return fig
def _write_preds_valid(*, output_dir: Path, y_true: Any, y_pred: Any, y_proba: Any | None, task_type: str, class_labels: list[str] | None) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    try:
        import numpy as np
        import pandas as pd
    except _RECOVERABLE_ERRORS:
        return (None, None, None)
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    n = min(y_true_arr.shape[0], y_pred_arr.shape[0])
    if n <= 0:
        return (None, None, None)
    payload: dict[str, Any] = {'y_true': y_true_arr[:n], 'y_pred': y_pred_arr[:n]}
    preds_schema: dict[str, Any] = {'task_type': task_type, 'columns': ['y_true', 'y_pred']}
    classes_path: Path | None = None
    if task_type == 'classification':
        labels = class_labels or []
        if y_proba is not None:
            proba_arr = np.asarray(y_proba)
            if proba_arr.ndim == 1:
                label0 = labels[0] if len(labels) > 0 else '0'
                label1 = labels[1] if len(labels) > 1 else '1'
                payload[f'proba__{label0}'] = 1.0 - proba_arr[:n]
                payload[f'proba__{label1}'] = proba_arr[:n]
                preds_schema['proba_columns'] = [f'proba__{label0}', f'proba__{label1}']
            elif proba_arr.ndim == 2:
                n_classes = int(proba_arr.shape[1])
                if not labels:
                    labels = [str(idx) for idx in range(n_classes)]
                if len(labels) != n_classes:
                    labels = labels[:n_classes] + [str(idx) for idx in range(len(labels), n_classes)]
                proba_columns: list[str] = []
                for (idx, label) in enumerate(labels[:n_classes]):
                    col = f'proba__{label}'
                    payload[col] = proba_arr[:n, idx]
                    proba_columns.append(col)
                preds_schema['proba_columns'] = proba_columns
            preds_schema['class_labels'] = labels
        if labels:
            classes_path = output_dir / 'classes.json'
            classes_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    preds_path = output_dir / 'preds_valid.parquet'
    pd.DataFrame(payload).to_parquet(preds_path, index=False)
    return (preds_path, classes_path, preds_schema)
def _record_model_failure(*, ctx: Any, cfg: Any, processed_dataset_id: str, split_hash: str, recipe_hash: str, model_variant_name: str, primary_metric: str, direction: str, cv_folds: int, cv_seed: int, task_type: str, n_classes: int | None, error: Exception) -> None:
    train_task_id = _resolve_task_id(ctx)
    error_payload = {'type': error.__class__.__name__, 'message': str(error)}
    out = {'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'train_task_id': train_task_id, 'model_id': None, 'best_score': None, 'primary_metric': primary_metric, 'task_type': task_type, 'status': 'failed', 'error': error_payload, 'model_variant': model_variant_name}
    if n_classes is not None:
        out['n_classes'] = n_classes
    clearml_enabled = is_clearml_enabled(cfg)
    inputs = {'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'model_variant': model_variant_name, 'primary_metric': primary_metric, 'direction': direction, 'cv_folds': cv_folds, 'seed': cv_seed, 'task_type': task_type}
    outputs = {'model_id': None, 'best_score': None, 'primary_metric': primary_metric, 'task_type': task_type, 'status': 'failed'}
    if n_classes is not None:
        outputs['n_classes'] = n_classes
    emit_outputs_and_manifest(ctx, cfg, process='train_model', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, manifest_extra={'error': error_payload}, clearml_enabled=clearml_enabled)
TrainMetricArtifacts = make_dataclass('TrainMetricArtifacts', [('metrics_payload', dict[str, Any]), ('metrics_path', Path), ('metrics_ci_path', Path | None), ('postprocess_payload', dict[str, Any]), ('postprocess_path', Path | None)], frozen=True)
def _build_train_metric_artifacts(*, output_dir: Path, primary_metric: str, direction: str, task_type: str, metrics_holdout: dict[str, float], train_rows: int, val_rows: int, n_classes: int | None, cv_summary: dict[str, Any] | None, threshold_payload: dict[str, Any] | None, calibration_payload: dict[str, Any] | None, uncertainty_payload: dict[str, Any], ci_payload: dict[str, Any] | None) -> TrainMetricArtifacts:
    metrics_payload: dict[str, Any] = {'primary_metric': primary_metric, 'direction': direction, 'task_type': task_type, 'holdout': {**metrics_holdout, 'train_rows': int(train_rows), 'val_rows': int(val_rows)}}
    if n_classes is not None:
        metrics_payload['n_classes'] = n_classes
    if cv_summary is not None:
        metrics_payload['cv'] = cv_summary
    if threshold_payload is not None:
        metrics_payload['thresholding'] = threshold_payload
    if calibration_payload is not None:
        metrics_payload['calibration'] = calibration_payload
    metrics_payload['uncertainty'] = uncertainty_payload
    if ci_payload is not None:
        metrics_payload['ci'] = ci_payload
    metrics_path = output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics_ci_path: Path | None = None
    if ci_payload is not None:
        metrics_ci_path = output_dir / 'metrics_ci.json'
        metrics_ci_path.write_text(json.dumps(ci_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    postprocess_payload: dict[str, Any] = {}
    if threshold_payload is not None:
        postprocess_payload.update({'threshold': threshold_payload.get('best_threshold'), 'metric': threshold_payload.get('metric'), 'score': threshold_payload.get('best_score'), 'direction': threshold_payload.get('direction')})
    if calibration_payload is not None:
        postprocess_payload['calibration'] = calibration_payload
    postprocess_path: Path | None = None
    if postprocess_payload:
        postprocess_path = output_dir / 'postprocess.json'
        postprocess_path.write_text(json.dumps(postprocess_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return TrainMetricArtifacts(metrics_payload=metrics_payload, metrics_path=metrics_path, metrics_ci_path=metrics_ci_path, postprocess_payload=postprocess_payload, postprocess_path=postprocess_path)
def _write_model_card(*, output_dir: Path, processed_dataset_id: str, split_hash: str, schema_version: str, preprocess_variant: str, encoding_note: str, recipe_hash: str, model_variant_name: str, model_class_path: str | None, model_params: dict[str, Any], cv_seed: int, task_type: str, n_classes: int | None, primary_metric: str, direction: str, best_score: float, ci_payload: dict[str, Any] | None, train_rows: int, val_rows: int, metrics_holdout: dict[str, float], threshold_payload: dict[str, Any] | None, calibration_payload: dict[str, Any] | None, uncertainty_payload: dict[str, Any], imbalance_report: dict[str, Any]) -> Path:
    model_card_lines = ['# Model Card', '', '## Dataset', f'- processed_dataset_id: {processed_dataset_id}', f'- split_hash: {split_hash}', f'- schema_version: {schema_version}', '', '## Preprocess', f'- preprocess_variant: {preprocess_variant}', f'- categorical_encoding: {encoding_note}', f'- recipe_hash: {recipe_hash}', '', '## Model', f'- model_variant: {model_variant_name}']
    if model_class_path:
        model_card_lines.append(f'- model_class: {model_class_path}')
    model_card_lines.extend([f'- hyperparams: {json.dumps(model_params, ensure_ascii=True, sort_keys=True)}', f'- seed: {cv_seed}', f'- task_type: {task_type}'])
    if n_classes is not None:
        model_card_lines.append(f'- n_classes: {n_classes}')
    model_card_lines.extend(['', '## Metrics', f'- primary_metric: {primary_metric} ({direction})', f'- best_score: {_format_float(best_score)}'])
    if isinstance(ci_payload, dict):
        interval = ci_payload.get('primary_metric')
        if isinstance(interval, dict):
            low = interval.get('low')
            mid = interval.get('mid')
            high = interval.get('high')
            if low is not None or mid is not None or high is not None:
                model_card_lines.append(f"- primary_metric_ci: [{_format_float(low)}, {_format_float(mid)}, {_format_float(high)}]")
    model_card_lines.extend([f'- train_rows: {train_rows}', f'- val_rows: {val_rows}'])
    metric_items = sorted([(k, v) for (k, v) in metrics_holdout.items() if k != primary_metric], key=lambda item: item[0]) if metrics_holdout else []
    if metric_items:
        trimmed = metric_items[:4]
        extra_metrics = ', '.join((f'{name}={_format_float(value)}' for (name, value) in trimmed))
        if len(metric_items) > 4:
            extra_metrics += f' (+{len(metric_items) - 4} more)'
        model_card_lines.append(f'- other_metrics: {extra_metrics}')
    model_card_lines.extend(['', '## Calibration / Thresholding / Uncertainty'])
    if threshold_payload is not None:
        model_card_lines.append(f"- thresholding: enabled metric={threshold_payload.get('metric')} best_threshold={_format_float(threshold_payload.get('best_threshold'))} score={_format_float(threshold_payload.get('best_score'))}")
    else:
        model_card_lines.append('- thresholding: disabled')
    if calibration_payload is not None:
        model_card_lines.append(f"- calibration: enabled method={calibration_payload.get('method')} mode={calibration_payload.get('mode')}")
    else:
        model_card_lines.append('- calibration: disabled')
    if uncertainty_payload.get('enabled'):
        model_card_lines.append(f"- uncertainty: enabled method={uncertainty_payload.get('method')} alpha={_format_float(uncertainty_payload.get('alpha'))} q={_format_float(uncertainty_payload.get('q'))}")
    else:
        model_card_lines.append('- uncertainty: disabled')
    imbalance_strategy = imbalance_report.get('strategy')
    imbalance_applied = imbalance_report.get('applied')
    if imbalance_report.get('enabled'):
        model_card_lines.append(f'- imbalance_handling: enabled strategy={imbalance_strategy} applied={imbalance_applied}')
    else:
        model_card_lines.append('- imbalance_handling: disabled')
    model_card_lines.extend(['', '## Limitations', '- Evaluated on a single split; performance may vary on new data.', '- Confirm leakage checks and target stability before promotion.', '- Not validated for out-of-scope inputs or populations.'])
    model_card_path = output_dir / 'model_card.md'
    model_card_path.write_text('\n'.join(model_card_lines) + '\n', encoding='utf-8')
    return model_card_path
TrainInputResolution = make_dataclass('TrainInputResolution', [('processed_ref', str | None), ('processed_ref_path', Path | None), ('preprocess_task_id', str | None), ('pipeline_task_id', str | None), ('preprocess_run_dir', Path), ('preprocess_out', dict[str, Any] | None), ('assets_dir', Path), ('processed_dataset_id', str), ('split_hash', str), ('recipe_hash', str)], frozen=True)
TrainFrameResolution = make_dataclass('TrainFrameResolution', [('split_payload', dict[str, Any]), ('train_idx', list[int]), ('val_idx', list[int]), ('preprocess_bundle', Any), ('task_type', str), ('target_column', str), ('feature_names', list[str]), ('X', Any), ('y', Any), ('label_encoder', Any | None), ('n_classes', int | None), ('classification_mode', str | None), ('class_labels', list[str] | None), ('class_labels_raw', list[Any] | None)], frozen=True)
def _resolve_train_inputs(cfg: Any, *, clearml_enabled: bool) -> TrainInputResolution:
    processed_ref = _normalize_str(getattr(getattr(cfg, 'data', None), 'processed_dataset_id', None))
    processed_ref_path: Path | None = None
    if processed_ref:
        candidate = Path(processed_ref).expanduser()
        if candidate.exists():
            processed_ref_path = candidate.resolve()
    preprocess_task_id = _normalize_str(_cfg_value(cfg, 'train.inputs.preprocess_task_id') or _cfg_value(cfg, 'train.preprocess_task_id') or _cfg_value(cfg, 'inputs.preprocess_task_id'))
    pipeline_task_id = _normalize_str(_cfg_value(cfg, 'run.clearml.pipeline_task_id'))
    preprocess_run_dir = _resolve_preprocess_run_dir(cfg, processed_ref_path)
    preprocess_out_path = preprocess_run_dir / 'out.json'
    preprocess_out: dict[str, Any] | None = None
    assets_dir: Path | None = None
    if preprocess_out_path.exists():
        preprocess_out = _load_json(preprocess_out_path)
        processed_dataset_id = _normalize_str(preprocess_out.get('processed_dataset_id'))
        split_hash = _normalize_str(preprocess_out.get('split_hash'))
        recipe_hash = _normalize_str(preprocess_out.get('recipe_hash'))
        if not processed_dataset_id or not split_hash or (not recipe_hash):
            raise ValueError('preprocess out.json is missing required keys.')
        if processed_ref and processed_ref_path is None and (processed_ref != processed_dataset_id):
            raise ValueError('data.processed_dataset_id does not match preprocess out.json. Set train.inputs.preprocess_run_dir to the matching preprocess output.')
        assets_dir = preprocess_run_dir
    else:
        if clearml_enabled and preprocess_task_id and (not processed_ref):
            try:
                task_out = get_task_artifact_local_copy(cfg, preprocess_task_id, 'out.json')
                task_payload = _load_json(task_out)
                processed_ref = _normalize_str(task_payload.get('processed_dataset_id')) or processed_ref
            except _RECOVERABLE_ERRORS as exc:
                warnings.warn(f'Failed to fetch preprocess out.json from task {preprocess_task_id}: {exc}')
        if processed_ref_path is not None:
            assets_dir = processed_ref_path if processed_ref_path.is_dir() else processed_ref_path.parent
        elif processed_ref and clearml_enabled and (not processed_ref.startswith('local:')):
            assets_dir = get_processed_dataset_local_copy(cfg, processed_ref)
        else:
            raise FileNotFoundError('preprocess out.json not found; specify train.inputs.preprocess_run_dir or data.processed_dataset_id.')
        meta_path = assets_dir / 'meta.json'
        meta_payload = _load_json(meta_path) if meta_path.exists() else {}
        processed_dataset_id = _normalize_str(processed_ref) or _normalize_str(meta_payload.get('processed_dataset_id'))
        split_hash = _normalize_str(meta_payload.get('split_hash'))
        recipe_hash = _normalize_str(meta_payload.get('recipe_hash'))
        if not processed_dataset_id or not split_hash or (not recipe_hash):
            raise ValueError('processed dataset meta.json is missing required keys.')
    if assets_dir is None or not assets_dir.exists():
        raise FileNotFoundError('processed dataset assets directory not found.')
    return TrainInputResolution(processed_ref=processed_ref, processed_ref_path=processed_ref_path, preprocess_task_id=preprocess_task_id, pipeline_task_id=pipeline_task_id, preprocess_run_dir=preprocess_run_dir, preprocess_out=preprocess_out, assets_dir=assets_dir, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash)
def _resolve_train_frame(cfg: Any, *, assets_dir: Path, processed_ref_path: Path | None, preprocess_out: dict[str, Any] | None) -> TrainFrameResolution:
    split_path = assets_dir / 'split.json'
    if not split_path.exists():
        raise FileNotFoundError(f'split.json not found under: {assets_dir}')
    split_payload = _load_json(split_path)
    train_idx = _normalize_indices(split_payload.get('train_index'), label='train_index')
    val_idx = _normalize_indices(split_payload.get('val_index'), label='val_index')
    bundle_path = assets_dir / 'preprocess_bundle.joblib'
    if not bundle_path.exists():
        raise FileNotFoundError(f'preprocess_bundle.joblib not found: {bundle_path}')
    preprocess_bundle = load_bundle(bundle_path)
    task_type = _normalize_task_type(getattr(getattr(cfg, 'eval', None), 'task_type', None))
    processed_dataset_path: Path | None = None
    if processed_ref_path is not None:
        if processed_ref_path.is_file():
            processed_dataset_path = processed_ref_path
        elif processed_ref_path.is_dir():
            candidate = processed_ref_path / 'processed_dataset.parquet'
            if candidate.exists():
                processed_dataset_path = candidate
    if processed_dataset_path is None and preprocess_out is not None:
        out_path_value = _normalize_str(preprocess_out.get('processed_dataset_path'))
        if out_path_value:
            candidate = Path(out_path_value).expanduser()
            if candidate.exists():
                processed_dataset_path = candidate.resolve()
    if processed_dataset_path is None:
        candidate = assets_dir / 'processed_dataset.parquet'
        if candidate.exists():
            processed_dataset_path = candidate
    df = None
    dataset_target_column: str | None = None
    if processed_dataset_path is not None:
        df = _load_dataframe(processed_dataset_path)
    else:
        x_path = assets_dir / 'X.parquet'
        y_path = assets_dir / 'y.parquet'
        if x_path.exists() and y_path.exists():
            df_x = _load_dataframe(x_path)
            df_y = _load_dataframe(y_path)
            if getattr(df_y, 'shape', (0, 0))[1] != 1:
                raise ValueError('y.parquet must contain exactly one column.')
            dataset_target_column = str(df_y.columns[0])
            df = df_x.copy()
            df[dataset_target_column] = df_y.iloc[:, 0].to_numpy()
        if df is None:
            (df, dataset_target_column) = _rebuild_processed_from_raw(cfg, base_dir=assets_dir, preprocess_bundle=preprocess_bundle if isinstance(preprocess_bundle, dict) else {}, split_payload=split_payload, task_type=task_type)
    if df is None:
        raise FileNotFoundError('processed dataset not found; specify preprocess_run_dir or dataset features.')
    bundle_columns = {}
    if isinstance(preprocess_bundle, dict):
        bundle_columns = preprocess_bundle.get('columns', {}) or {}
    target_column = _normalize_str(bundle_columns.get('target_column')) or dataset_target_column or _normalize_str(getattr(getattr(cfg, 'data', None), 'target_column', None))
    if not target_column or target_column not in df.columns:
        raise ValueError(f'target_column not found in processed dataset: {target_column}')
    feature_names = None
    if isinstance(preprocess_bundle, dict):
        feature_names = preprocess_bundle.get('feature_names')
    if not feature_names:
        names_path = assets_dir / 'feature_names.json'
        if names_path.exists():
            names_payload = json.loads(names_path.read_text(encoding='utf-8'))
            feature_names = [str(value) for value in names_payload] if isinstance(names_payload, list) else None
    if feature_names and all((name in df.columns for name in feature_names)):
        X = df[feature_names]
    else:
        X = df.drop(columns=[target_column])
        feature_names = list(X.columns)
    y = df[target_column].to_numpy()
    label_encoder = None
    n_classes: int | None = None
    classification_mode: str | None = None
    class_labels: list[str] | None = None
    class_labels_raw: list[Any] | None = None
    if task_type == 'classification':
        try:
            from sklearn.preprocessing import LabelEncoder
        except _RECOVERABLE_ERRORS as exc:
            raise RuntimeError('scikit-learn is required for classification.') from exc
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        n_classes = int(len(label_encoder.classes_))
        if n_classes < 2:
            raise ValueError('classification requires at least 2 classes in target.')
        classification_mode = _resolve_classification_mode(cfg, n_classes=n_classes)
        class_labels_raw = list(label_encoder.classes_)
        class_labels = [str(label) for label in class_labels_raw]
    return TrainFrameResolution(split_payload=split_payload, train_idx=train_idx, val_idx=val_idx, preprocess_bundle=preprocess_bundle, task_type=task_type, target_column=target_column, feature_names=list(feature_names), X=X, y=y, label_encoder=label_encoder, n_classes=n_classes, classification_mode=classification_mode, class_labels=class_labels, class_labels_raw=class_labels_raw)
def _finalize_train_outputs(*, cfg: Any, ctx: Any, clearml_enabled: bool, preprocess_run_dir: Path, preprocess_bundle: Any, model_variant: dict[str, Any], model_variant_fit: dict[str, Any], model_variant_name: str, processed_dataset_id: str, split_hash: str, recipe_hash: str, train_idx: list[int], val_idx: list[int], cv_seed: int, task_type: str, n_classes: int | None, primary_metric: str, direction: str, best_score: float, ci_payload: dict[str, Any] | None, metrics_holdout: dict[str, float], threshold_payload: dict[str, Any] | None, calibration_payload: dict[str, Any] | None, uncertainty_payload: dict[str, Any], imbalance_report: dict[str, Any], class_labels: list[str] | None, preds_schema: dict[str, Any] | None, train_task_id: str | None, preprocess_task_id: str | None, pipeline_task_id: str | None, model_bundle_path: Path, metrics_path: Path, metrics_ci_path: Path | None, preds_valid_path: Path | None, classes_path: Path | None, postprocess_path: Path | None, calibration_report_path: Path | None, feature_importance_path: Path | None, confusion_csv_path: Path | None) -> None:
    model_id = str(model_bundle_path)
    versions = resolve_version_props(cfg, clearml_enabled=clearml_enabled)
    recipe_payload: dict[str, Any] = {}
    recipe_path = preprocess_run_dir / 'recipe.json'
    if recipe_path.exists():
        recipe_payload = _load_json(recipe_path)
    preprocess_variant = _normalize_str(recipe_payload.get('variant')) if recipe_payload else None
    if preprocess_variant is None and isinstance(preprocess_bundle, dict):
        preprocess_variant = _normalize_str(preprocess_bundle.get('preprocess_variant'))
    if preprocess_variant is None:
        preprocess_variant = 'unknown'
    enc = recipe_payload.get('categorical_encoding') if isinstance(recipe_payload, dict) else None
    encoding = _normalize_str(enc.get('encoding')) if isinstance(enc, dict) else None
    if not encoding:
        encoding_note = 'unknown'
    elif encoding == 'auto':
        hash_cfg = enc.get('hashing') or {}
        encoding_note = f"auto(onehot_max={enc.get('auto_onehot_max_categories')}, hash_n_features={(hash_cfg.get('n_features') if isinstance(hash_cfg, dict) else None)})"
    elif encoding == 'hashing':
        hash_cfg = enc.get('hashing') or {}
        encoding_note = f"hashing(n_features={(hash_cfg.get('n_features') if isinstance(hash_cfg, dict) else None)})"
    elif encoding == 'target_mean_oof':
        oof_cfg = enc.get('target_mean_oof') or {}
        folds = oof_cfg.get('folds') if isinstance(oof_cfg, dict) else None
        smoothing = oof_cfg.get('smoothing') if isinstance(oof_cfg, dict) else None
        encoding_note = f'target_mean_oof(folds={folds}, smoothing={smoothing})'
    else:
        encoding_note = encoding
    model_class_path = _normalize_str(model_variant.get('class_path'))
    if model_class_path is None:
        model_class_path = _normalize_str(model_variant_fit.get('class_path'))
    model_params = _stringify_payload(model_variant_fit.get('params') or {})
    model_card_path = _write_model_card(output_dir=ctx.output_dir, processed_dataset_id=processed_dataset_id, split_hash=split_hash, schema_version=str(versions.get('schema_version', 'unknown')), preprocess_variant=preprocess_variant, encoding_note=encoding_note, recipe_hash=recipe_hash, model_variant_name=model_variant_name, model_class_path=model_class_path, model_params=model_params, cv_seed=cv_seed, task_type=task_type, n_classes=n_classes, primary_metric=primary_metric, direction=direction, best_score=best_score, ci_payload=ci_payload, train_rows=len(train_idx), val_rows=len(val_idx), metrics_holdout=metrics_holdout, threshold_payload=threshold_payload, calibration_payload=calibration_payload, uncertainty_payload=uncertainty_payload, imbalance_report=imbalance_report)
    registry_model_id: str | None = None
    registry_status: str | None = None
    registry_error: dict[str, Any] | None = None
    if clearml_enabled:
        usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
        model_name = f'{usecase_id}:{model_variant_name}:{processed_dataset_id}'
        tags = _build_registry_tags(usecase_id=usecase_id, process='train_model', processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, preprocess_variant=preprocess_variant, model_variant=model_variant_name, task_type=task_type, train_task_id=train_task_id, preprocess_task_id=preprocess_task_id, pipeline_task_id=pipeline_task_id)
        try:
            registry_model_id = register_model_artifact(ctx, model_path=model_bundle_path, model_name=model_name, tags=tags)
            registry_status = 'registered'
        except _RECOVERABLE_ERRORS as exc:
            registry_status = 'failed'
            registry_error = {'type': exc.__class__.__name__, 'message': str(exc)}
            warnings.warn(f'Failed to register model in ClearML registry: {exc}')
    if clearml_enabled:
        upload_artifact(ctx, 'metrics.json', metrics_path)
        if metrics_ci_path is not None:
            upload_artifact(ctx, metrics_ci_path.name, metrics_ci_path)
        if preds_valid_path is not None:
            upload_artifact(ctx, preds_valid_path.name, preds_valid_path)
        if classes_path is not None:
            upload_artifact(ctx, classes_path.name, classes_path)
        upload_artifact(ctx, 'model_bundle.joblib', model_bundle_path)
        upload_artifact(ctx, 'model_card.md', model_card_path)
        if postprocess_path is not None:
            upload_artifact(ctx, postprocess_path.name, postprocess_path)
        if calibration_report_path is not None:
            upload_artifact(ctx, calibration_report_path.name, calibration_report_path)
        if feature_importance_path is not None:
            upload_artifact(ctx, feature_importance_path.name, feature_importance_path)
        if confusion_csv_path is not None:
            upload_artifact(ctx, confusion_csv_path.name, confusion_csv_path)
        extra_props = {'task_type': task_type}
        if n_classes is not None:
            extra_props['n_classes'] = n_classes
        if threshold_payload is not None:
            extra_props['best_threshold'] = threshold_payload.get('best_threshold')
        extra_props['imbalance_enabled'] = bool(imbalance_report.get('enabled'))
        extra_props['imbalance_strategy'] = imbalance_report.get('strategy')
        extra_props['imbalance_applied'] = bool(imbalance_report.get('applied'))
        if registry_model_id:
            extra_props['registry_model_id'] = registry_model_id
        if registry_status:
            extra_props['registry_status'] = registry_status
        update_task_properties(ctx, {'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'model_id': model_id, 'primary_metric': primary_metric, 'best_score': best_score, **extra_props})
    out = {'model_id': model_id, 'train_task_id': train_task_id, 'preprocess_task_id': preprocess_task_id, 'pipeline_task_id': pipeline_task_id, 'best_score': best_score, 'primary_metric': primary_metric, 'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'task_type': task_type}
    if registry_model_id:
        out['registry_model_id'] = registry_model_id
    if registry_status:
        out['registry_status'] = registry_status
    if registry_error:
        out['registry_error'] = registry_error
    if n_classes is not None:
        out['n_classes'] = n_classes
    if class_labels is not None:
        out['class_labels'] = class_labels
    if threshold_payload is not None:
        out['best_threshold'] = threshold_payload.get('best_threshold')
        out['threshold_metric'] = threshold_payload.get('metric')
        out['threshold_score'] = threshold_payload.get('best_score')
    if calibration_payload is not None:
        out['calibration'] = calibration_payload
    if ci_payload is not None:
        out['primary_metric_ci'] = ci_payload.get('primary_metric')
    out['imbalance'] = imbalance_report
    out['uncertainty'] = uncertainty_payload
    if preds_valid_path is not None:
        out['preds_valid_path'] = str(preds_valid_path)
    if classes_path is not None:
        out['classes_path'] = str(classes_path)
    if preds_schema is not None:
        out['preds_schema'] = preds_schema
    inputs = {'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'model_variant': model_variant_name, 'primary_metric': primary_metric, 'direction': direction, 'cv_folds': int(getattr(getattr(cfg, 'eval', None), 'cv_folds', 0) or 0), 'seed': cv_seed, 'task_type': task_type}
    outputs = {'model_id': model_id, 'best_score': best_score, 'primary_metric': primary_metric, 'task_type': task_type}
    if n_classes is not None:
        outputs['n_classes'] = n_classes
    if threshold_payload is not None:
        outputs['best_threshold'] = threshold_payload.get('best_threshold')
        outputs['threshold_metric'] = threshold_payload.get('metric')
        outputs['threshold_score'] = threshold_payload.get('best_score')
    emit_outputs_and_manifest(ctx, cfg, process='train_model', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, clearml_enabled=clearml_enabled)
def _emit_train_viz_and_logs(*, cfg: Any, ctx: Any, clearml_enabled: bool, task_type: str, model: Any, feature_names: list[str], y_val: Any, y_val_pred: Any, y_val_proba: Any | None, n_classes: int | None, class_labels: list[str] | None, primary_metric: str, metrics_holdout: dict[str, float], regression_metrics: dict[str, float] | None, uncertainty_payload: dict[str, Any]) -> tuple[Path | None, Path | None]:
    viz_settings = _resolve_viz_settings(cfg)
    feature_importance_path: Path | None = None
    feature_importance_plot_path: Path | None = None
    residuals_plot_path: Path | None = None
    confusion_csv_path: Path | None = None
    confusion_plot_path: Path | None = None
    roc_plot_path: Path | None = None
    importance_payload = _extract_feature_importance(model, feature_names)
    importance_names: list[str] | None = None
    importance_scores: list[float] | None = None
    if importance_payload is not None:
        (names, scores) = importance_payload
        importance_names = names
        importance_scores = scores
        feature_importance_path = _write_feature_importance_csv(names, scores, ctx.output_dir)
        if viz_settings['enabled']:
            feature_importance_plot_path = plot_feature_importance(names, scores, ctx.output_dir / 'feature_importance.png', top_n=viz_settings['max_features'])
    class_names = class_labels
    if task_type == 'regression':
        if viz_settings['enabled']:
            residuals_plot_path = plot_regression_residuals(y_val, y_val_pred, ctx.output_dir / 'residuals.png', max_points=viz_settings['max_points'])
        if viz_settings['enabled'] and uncertainty_payload.get('enabled') and (uncertainty_payload.get('q') is not None):
            width = float(uncertainty_payload['q']) * 2.0
            interval_widths = [width for _ in range(int(len(y_val_pred) or 1))]
            plot_interval_width_histogram(interval_widths, ctx.output_dir / 'interval_widths.png', title='Prediction Interval Widths')
    else:
        confusion_csv_path = write_confusion_matrix_csv(y_val, y_val_pred, ctx.output_dir / 'confusion_matrix.csv', class_names=class_names)
        if viz_settings['enabled']:
            confusion_plot_path = plot_confusion_matrix(y_val, y_val_pred, ctx.output_dir / 'confusion_matrix.png', class_names=class_names, normalize=viz_settings['confusion_normalize'])
            if viz_settings['roc_curve'] and n_classes == 2 and (y_val_proba is not None):
                roc_plot_path = plot_roc_curve(y_val, y_val_proba[:, 1], ctx.output_dir / 'roc_curve.png')
    if clearml_enabled:
        if task_type == 'regression':
            for name in REGRESSION_METRIC_ORDER:
                if name in metrics_holdout:
                    log_scalar(ctx.task, 'metrics', name, metrics_holdout[name], step=0)
        if metrics_holdout.get(primary_metric) is not None and (task_type != 'regression' or primary_metric not in REGRESSION_METRIC_ORDER):
            log_scalar(ctx.task, 'metrics', primary_metric, metrics_holdout[primary_metric], step=0)
        if viz_settings['enabled']:
            if importance_names and importance_scores:
                fig = _build_plotly_feature_importance(importance_names, importance_scores, top_n=viz_settings['max_features'])
                log_plotly(ctx.task, 'train_model', 'feature_importance', fig or feature_importance_plot_path, step=0)
            if task_type == 'regression':
                metrics_table = build_regression_metrics_table(regression_metrics or metrics_holdout)
                log_plotly(ctx.task, 'train_model', 'metrics_table', metrics_table, step=0)
                scatter = build_true_pred_scatter(y_val, y_val_pred, r2=metrics_holdout.get('r2'), max_points=viz_settings['max_points'])
                log_plotly(ctx.task, 'train_model', 'true_vs_pred', scatter, step=0)
                fig = build_residuals_plot(y_val, y_val_pred, max_points=viz_settings['max_points'])
                log_plotly(ctx.task, 'train_model', 'residuals', fig or residuals_plot_path, step=0)
            else:
                fig = _build_plotly_confusion_matrix(y_val, y_val_pred, class_names=class_names, normalize=viz_settings['confusion_normalize'])
                log_plotly(ctx.task, 'train_model', 'confusion_matrix', fig or confusion_plot_path, step=0)
                if y_val_proba is not None and n_classes == 2:
                    fig = _build_plotly_roc_curve(y_val, y_val_proba[:, 1])
                    log_plotly(ctx.task, 'train_model', 'roc_curve', fig or roc_plot_path, step=0)
    return (feature_importance_path, confusion_csv_path)
TrainRuntimeContext = make_dataclass('TrainRuntimeContext', [('resolved_inputs', TrainInputResolution), ('frame', TrainFrameResolution), ('thresholding_cfg', dict[str, Any]), ('calibration_cfg', dict[str, Any]), ('imbalance_cfg', dict[str, Any]), ('uncertainty_cfg', dict[str, Any]), ('ci_cfg', dict[str, Any]), ('fbeta_beta', float), ('uncertainty_payload', dict[str, Any]), ('X_train', Any), ('y_train', Any), ('X_train_fit', Any), ('y_train_fit', Any), ('X_val', Any), ('y_val', Any), ('train_profile', dict[str, Any] | None), ('model_variant', dict[str, Any]), ('model_variant_name', str), ('model_variant_fit', dict[str, Any]), ('imbalance_report', dict[str, Any]), ('resample_strategy', str | None), ('primary_metric', str), ('direction', str), ('cv_folds', int), ('cv_seed', int)])
TrainExecutionResult = make_dataclass('TrainExecutionResult', [('model', Any), ('calibrated_model', Any | None), ('metrics_holdout', dict[str, float]), ('regression_metrics', dict[str, float] | None), ('y_val_pred', Any), ('y_val_proba', Any | None), ('threshold_payload', dict[str, Any] | None), ('calibration_payload', dict[str, Any] | None), ('calibration_report_path', Path | None), ('best_score', float), ('debug_sample', dict[str, list[Any]] | None), ('ci_payload', dict[str, Any] | None), ('preds_valid_path', Path | None), ('classes_path', Path | None), ('preds_schema', dict[str, Any] | None), ('cv_summary', dict[str, Any] | None)])
def _prepare_train_runtime(cfg: Any, *, ctx: Any, clearml_enabled: bool) -> TrainRuntimeContext:
    resolved_inputs = _resolve_train_inputs(cfg, clearml_enabled=clearml_enabled)
    frame = _resolve_train_frame(cfg, assets_dir=resolved_inputs.assets_dir, processed_ref_path=resolved_inputs.processed_ref_path, preprocess_out=resolved_inputs.preprocess_out)
    thresholding_cfg = _resolve_thresholding_settings(cfg)
    calibration_cfg = _resolve_calibration_settings(cfg)
    imbalance_cfg = _resolve_imbalance_settings(cfg)
    uncertainty_cfg = _resolve_uncertainty_settings(cfg)
    ci_cfg = _resolve_ci_settings(cfg)
    task_type = frame.task_type
    n_classes = frame.n_classes
    if uncertainty_cfg['enabled'] and task_type != 'regression':
        raise ValueError('eval.uncertainty.enabled is supported for regression only.')
    if thresholding_cfg['enabled']:
        if task_type != 'classification':
            raise ValueError('threshold optimization is supported for classification only.')
        if n_classes != 2:
            raise ValueError('threshold optimization supports binary classification only.')
    if calibration_cfg['enabled'] and task_type != 'classification':
        raise ValueError('probability calibration is supported for classification only.')
    fbeta_beta = _cfg_value(cfg, 'eval.metrics.fbeta_beta', 1.0)
    try:
        fbeta_beta = float(fbeta_beta)
    except _RECOVERABLE_ERRORS:
        fbeta_beta = 1.0
    uncertainty_payload: dict[str, Any] = {'enabled': bool(uncertainty_cfg.get('enabled')), 'method': uncertainty_cfg.get('method'), 'alpha': uncertainty_cfg.get('alpha'), 'q': None}
    if not frame.train_idx or not frame.val_idx:
        raise ValueError('split.json must include non-empty train_index and val_index.')
    X_train = frame.X.iloc[frame.train_idx]
    y_train = frame.y[frame.train_idx]
    X_val = frame.X.iloc[frame.val_idx]
    y_val = frame.y[frame.val_idx]
    drift_settings = resolve_drift_settings(cfg)
    train_profile: dict[str, Any] | None = None
    if drift_settings['enabled']:
        (drift_sample, sample_info) = sample_frame(X_train, sample_n=drift_settings['sample_n'], seed=drift_settings['sample_seed'])
        train_profile = build_train_profile(drift_sample, feature_columns=list(X_train.columns))
        train_profile = annotate_profile(train_profile, role='train', sample_info=sample_info, metrics=drift_settings['metrics'])
        settings = train_profile.get('settings')
        if isinstance(settings, dict):
            alert_thresholds = drift_settings.get('alert_thresholds')
            if alert_thresholds:
                settings['alert_thresholds'] = dict(alert_thresholds)
        train_profile_path = ctx.output_dir / 'train_profile.json'
        train_profile_path.write_text(json.dumps(train_profile, ensure_ascii=False, indent=2), encoding='utf-8')
        if clearml_enabled:
            upload_artifact(ctx, train_profile_path.name, train_profile_path)
    model_variant = _merge_model_variant(cfg)
    model_variant_name = _normalize_str(model_variant.get('name')) or 'unknown'
    model_variant_fit = dict(model_variant)
    model_variant_fit['params'] = dict(model_variant.get('params') or {})
    imbalance_report: dict[str, Any] = {'enabled': bool(imbalance_cfg.get('enabled')), 'strategy': imbalance_cfg.get('strategy'), 'applied': False}
    X_train_fit = X_train
    y_train_fit = y_train
    resample_strategy: str | None = None
    if imbalance_cfg.get('enabled'):
        if task_type != 'classification':
            imbalance_report['reason'] = 'task_type_not_classification'
        elif not imbalance_cfg.get('strategy'):
            imbalance_report['reason'] = 'strategy_missing'
        elif imbalance_cfg.get('strategy') in ('oversample', 'undersample'):
            imbalance_seed = int(_cfg_value(cfg, 'eval.seed', 42) or 42)
            (X_train_fit, y_train_fit, resample_info, resample_reason) = _apply_resampling(imbalance_cfg['strategy'], X_train, y_train, seed=imbalance_seed)
            if resample_reason:
                imbalance_report['reason'] = resample_reason
            else:
                imbalance_report['applied'] = True
                imbalance_report['detail'] = resample_info
                resample_strategy = imbalance_cfg['strategy']
        else:
            (model_variant_fit, weight_report) = _apply_weight_strategy(model_variant=model_variant_fit, task_type=task_type, y_train=y_train, n_classes=n_classes, imbalance_cfg=imbalance_cfg)
            imbalance_report.update(weight_report)
        if not imbalance_report.get('applied'):
            warnings.warn(f"Imbalance strategy '{imbalance_cfg.get('strategy')}' was skipped: {imbalance_report.get('reason')}")
    primary_metric = (_normalize_str(getattr(getattr(cfg, 'eval', None), 'primary_metric', None)) or 'rmse').lower()
    direction = _normalize_str(getattr(getattr(cfg, 'eval', None), 'direction', None))
    if not direction or direction == 'auto':
        direction = metric_direction(primary_metric, task_type)
    direction = direction.lower()
    connect_train_model(ctx, cfg, processed_dataset_id=resolved_inputs.processed_dataset_id, task_type=task_type, primary_metric=primary_metric, model_variant=model_variant_name, model_params=model_variant_fit.get('params') if isinstance(model_variant_fit, dict) else None)
    cv_folds = int(getattr(getattr(cfg, 'eval', None), 'cv_folds', 0) or 0)
    cv_seed = int(getattr(getattr(cfg, 'eval', None), 'seed', 42) or 42)
    return TrainRuntimeContext(resolved_inputs=resolved_inputs, frame=frame, thresholding_cfg=thresholding_cfg, calibration_cfg=calibration_cfg, imbalance_cfg=imbalance_cfg, uncertainty_cfg=uncertainty_cfg, ci_cfg=ci_cfg, fbeta_beta=fbeta_beta, uncertainty_payload=uncertainty_payload, X_train=X_train, y_train=y_train, X_train_fit=X_train_fit, y_train_fit=y_train_fit, X_val=X_val, y_val=y_val, train_profile=train_profile, model_variant=model_variant, model_variant_name=model_variant_name, model_variant_fit=model_variant_fit, imbalance_report=imbalance_report, resample_strategy=resample_strategy, primary_metric=primary_metric, direction=direction, cv_folds=cv_folds, cv_seed=cv_seed)
def _fit_predict_and_score(cfg: Any, *, ctx: Any, runtime: TrainRuntimeContext) -> TrainExecutionResult:
    calibration_report_path: Path | None = None
    frame = runtime.frame
    task_type = frame.task_type
    n_classes = frame.n_classes
    try:
        model = build_model(runtime.model_variant_fit, task_type=task_type)
        model.fit(runtime.X_train_fit, runtime.y_train_fit)
        predictor = model
        calibrated_model = None
        calibration_payload: dict[str, Any] | None = None
        if task_type == 'classification' and runtime.calibration_cfg['enabled']:
            if not hasattr(model, 'predict_proba'):
                raise ValueError('calibration requires predict_proba on the model.')
            calibrated_model = _calibrate_classifier(model, runtime.X_val, runtime.y_val, method=runtime.calibration_cfg['method'], mode=runtime.calibration_cfg['mode'])
            predictor = calibrated_model
            calibration_payload = {'enabled': True, 'method': runtime.calibration_cfg['method'], 'mode': runtime.calibration_cfg['mode']}
        metrics_holdout: dict[str, float] = {}
        regression_metrics: dict[str, float] | None = None
        y_val_pred = predictor.predict(runtime.X_val)
        y_val_proba = None
        if task_type == 'classification' and hasattr(predictor, 'predict_proba'):
            y_val_proba = predictor.predict_proba(runtime.X_val)
        threshold_payload: dict[str, Any] | None = None
        if runtime.thresholding_cfg['enabled']:
            if y_val_proba is None:
                raise ValueError('threshold optimization requires predict_proba on the model.')
            (best_threshold, threshold_score, threshold_direction) = _select_best_threshold(runtime.y_val, y_val_proba, metric_name=runtime.thresholding_cfg['metric'], grid=runtime.thresholding_cfg['grid'], task_type=task_type, n_classes=n_classes, beta=runtime.fbeta_beta)
            positive_proba = extract_positive_class_proba(y_val_proba)
            y_val_pred = (positive_proba >= best_threshold).astype(int)
            threshold_payload = {'enabled': True, 'metric': runtime.thresholding_cfg['metric'], 'best_threshold': best_threshold, 'best_score': threshold_score, 'direction': threshold_direction}
        if calibration_payload is not None and y_val_proba is not None:
            (report, curve_conf, curve_acc) = _build_calibration_report(runtime.y_val, y_val_proba, n_classes=n_classes)
            calibration_report_path = ctx.output_dir / 'calibration_report.json'
            calibration_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            plot_reliability_curve(curve_conf, curve_acc, ctx.output_dir / 'calibration_reliability.png', title='Reliability Diagram')
        if task_type == 'classification':
            if y_val_proba is not None and n_classes and (n_classes > 2):
                import numpy as np
                y_val_pred = np.asarray(y_val_proba).argmax(axis=1)
            mode = frame.classification_mode or 'binary'
            metric_names = _resolve_classification_metrics(cfg, classification_mode=mode, n_classes=n_classes, imbalance_enabled=bool(runtime.imbalance_cfg.get('enabled')))
            for name in metric_names:
                if n_classes is not None and n_classes != 2 and _is_binary_only_metric(name):
                    warnings.warn(f"metric '{name}' is binary-only; skipping for multiclass.")
                    continue
                if metric_requires_proba(name, task_type) and y_val_proba is None:
                    continue
                metric_fn = get_metric(name, task_type, n_classes=n_classes, beta=runtime.fbeta_beta)
                metrics_holdout[name] = float(metric_fn(runtime.y_val, y_val_pred, y_val_proba))
            if runtime.primary_metric not in metrics_holdout:
                if metric_requires_proba(runtime.primary_metric, task_type) and y_val_proba is None:
                    raise ValueError(f"primary_metric '{runtime.primary_metric}' requires predict_proba but model does not support it.")
                metric_fn = get_metric(runtime.primary_metric, task_type, n_classes=n_classes, beta=runtime.fbeta_beta)
                metrics_holdout[runtime.primary_metric] = float(metric_fn(runtime.y_val, y_val_pred, y_val_proba))
        else:
            metric_names = _resolve_regression_metrics(cfg)
            regression_metrics = compute_regression_metrics(runtime.y_val, y_val_pred, metrics=metric_names)
            metrics_holdout.update(regression_metrics)
            if runtime.primary_metric not in metrics_holdout:
                metrics_holdout[runtime.primary_metric] = float(get_metric(runtime.primary_metric, task_type)(runtime.y_val, y_val_pred))
        if runtime.uncertainty_cfg.get('enabled'):
            q = compute_split_conformal_quantile(runtime.y_val, y_val_pred, alpha=float(runtime.uncertainty_cfg.get('alpha') or 0.1), use_abs_residual=bool(runtime.uncertainty_cfg.get('use_abs_residual', True)))
            runtime.uncertainty_payload['q'] = q
        best_score = metrics_holdout[runtime.primary_metric]
        debug_sample = None
        if is_clearml_enabled(cfg):
            debug_sample = _build_prediction_sample(runtime.y_val, y_val_pred, y_val_proba)
        ci_payload: dict[str, Any] | None = None
        if runtime.ci_cfg.get('enabled'):
            (ci_interval, ci_info) = _bootstrap_metric_ci(runtime.y_val, y_val_pred, y_val_proba, metric_name=runtime.primary_metric, task_type=task_type, n_classes=n_classes, n_boot=int(runtime.ci_cfg.get('n_boot') or 0), alpha=float(runtime.ci_cfg.get('alpha') or 0.05), seed=int(runtime.ci_cfg.get('seed') or 0), beta=runtime.fbeta_beta, point_estimate=best_score)
            if ci_interval is None:
                warnings.warn('Bootstrap CI skipped: no valid samples.')
                ci_interval = {'low': None, 'mid': float(best_score), 'high': None}
            if ci_info['n_boot_effective'] < ci_info['n_boot']:
                warnings.warn(f"Bootstrap CI used fewer samples than requested ({ci_info['n_boot_effective']}/{ci_info['n_boot']}).")
            ci_payload = {'primary_metric': ci_interval, 'n_boot': ci_info['n_boot'], 'n_boot_effective': ci_info['n_boot_effective'], 'alpha': ci_info['alpha'], 'seed': ci_info['seed']}
        preds_valid_path = None
        classes_path = None
        preds_schema = None
        try:
            (preds_valid_path, classes_path, preds_schema) = _write_preds_valid(output_dir=ctx.output_dir, y_true=runtime.y_val, y_pred=y_val_pred, y_proba=y_val_proba, task_type=task_type, class_labels=frame.class_labels)
        except _RECOVERABLE_ERRORS as exc:
            warnings.warn(f'Failed to write preds_valid.parquet: {exc}')
        cv_summary: dict[str, Any] | None = None
        if runtime.cv_folds and runtime.cv_folds > 1:
            if len(frame.train_idx) < runtime.cv_folds:
                warnings.warn('eval.cv_folds is larger than the training split size; skipping CV.')
            else:
                try:
                    import numpy as np
                    from sklearn.model_selection import KFold
                except _RECOVERABLE_ERRORS as exc:
                    raise RuntimeError('scikit-learn is required for cross-validation.') from exc
                kf = KFold(n_splits=runtime.cv_folds, shuffle=True, random_state=runtime.cv_seed)
                scores: list[float] = []
                metric_fn = get_metric(runtime.primary_metric, task_type, n_classes=n_classes, beta=runtime.fbeta_beta)
                X_train_full = frame.X.iloc[frame.train_idx]
                y_train_full = frame.y[frame.train_idx]
                needs_proba = metric_requires_proba(runtime.primary_metric, task_type)
                for (fold_train_idx, fold_val_idx) in kf.split(X_train_full):
                    fold_X_train = X_train_full.iloc[fold_train_idx]
                    fold_y_train = y_train_full[fold_train_idx]
                    if runtime.resample_strategy:
                        (fold_X_train, fold_y_train, _, _) = _apply_resampling(runtime.resample_strategy, fold_X_train, fold_y_train, seed=runtime.cv_seed)
                    fold_model = build_model(runtime.model_variant_fit, task_type=task_type)
                    fold_model.fit(fold_X_train, fold_y_train)
                    fold_pred = fold_model.predict(X_train_full.iloc[fold_val_idx])
                    fold_proba = None
                    if needs_proba:
                        if not hasattr(fold_model, 'predict_proba'):
                            raise ValueError(f"primary_metric '{runtime.primary_metric}' requires predict_proba but model does not support it.")
                        fold_proba = fold_model.predict_proba(X_train_full.iloc[fold_val_idx])
                    scores.append(float(metric_fn(y_train_full[fold_val_idx], fold_pred, fold_proba)))
                cv_summary = {'folds': runtime.cv_folds, 'seed': runtime.cv_seed, 'scores': scores, 'mean': float(np.mean(scores)) if scores else None, 'std': float(np.std(scores)) if scores else None}
    except ModelWeightsUnavailableError as exc:
        _record_model_failure(ctx=ctx, cfg=cfg, processed_dataset_id=runtime.resolved_inputs.processed_dataset_id, split_hash=runtime.resolved_inputs.split_hash, recipe_hash=runtime.resolved_inputs.recipe_hash, model_variant_name=runtime.model_variant_name, primary_metric=runtime.primary_metric, direction=runtime.direction, cv_folds=runtime.cv_folds, cv_seed=runtime.cv_seed, task_type=task_type, n_classes=n_classes, error=exc)
        raise
    return TrainExecutionResult(model=model, calibrated_model=calibrated_model, metrics_holdout=metrics_holdout, regression_metrics=regression_metrics, y_val_pred=y_val_pred, y_val_proba=y_val_proba, threshold_payload=threshold_payload, calibration_payload=calibration_payload, calibration_report_path=calibration_report_path, best_score=best_score, debug_sample=debug_sample, ci_payload=ci_payload, preds_valid_path=preds_valid_path, classes_path=classes_path, preds_schema=preds_schema, cv_summary=cv_summary)
def _build_and_save_train_bundle(*, ctx: Any, clearml_enabled: bool, runtime: TrainRuntimeContext, result: TrainExecutionResult, metric_artifacts: TrainMetricArtifacts) -> tuple[Path, str | None]:
    resolved = runtime.resolved_inputs
    frame = runtime.frame
    train_task_id = _resolve_task_id(ctx)
    preprocess_provenance = _resolve_preprocess_provenance(resolved.preprocess_run_dir, resolved.assets_dir, resolved.preprocess_out, frame.preprocess_bundle)
    provenance_payload = {'train_task_id': train_task_id, 'raw_dataset_id': preprocess_provenance.get('raw_dataset_id'), 'processed_dataset_id': resolved.processed_dataset_id, 'preprocess_variant': preprocess_provenance.get('preprocess_variant'), 'split_hash': resolved.split_hash, 'recipe_hash': resolved.recipe_hash}
    model_bundle = {'model': result.model, 'calibrated_model': result.calibrated_model, 'model_variant': runtime.model_variant_name, 'primary_metric': runtime.primary_metric, 'best_score': result.best_score, 'metrics': metric_artifacts.metrics_payload, 'preprocess_bundle': frame.preprocess_bundle, 'feature_names': frame.feature_names, 'target_column': frame.target_column, 'processed_dataset_id': resolved.processed_dataset_id, 'split_hash': resolved.split_hash, 'recipe_hash': resolved.recipe_hash, 'task_type': frame.task_type, 'label_encoder': frame.label_encoder, 'class_labels': frame.class_labels_raw, 'n_classes': frame.n_classes, 'postprocess': metric_artifacts.postprocess_payload or None, 'uncertainty': {**runtime.uncertainty_payload, 'use_abs_residual': runtime.uncertainty_cfg.get('use_abs_residual')}, 'provenance': {key: value for (key, value) in provenance_payload.items() if value is not None}}
    if runtime.train_profile is not None:
        model_bundle['train_profile'] = runtime.train_profile
    if result.calibration_payload is not None:
        model_bundle['calibration'] = result.calibration_payload
    model_bundle_path = ctx.output_dir / 'model_bundle.joblib'
    save_bundle(model_bundle_path, model_bundle)
    if clearml_enabled and result.debug_sample is not None:
        log_debug_table(ctx.task, 'train_model', 'prediction_sample', result.debug_sample, step=0)
    return (model_bundle_path, train_task_id)
def _run_train_model_flow(cfg: Any, *, ctx: Any, clearml_enabled: bool) -> None:
    runtime = _prepare_train_runtime(cfg, ctx=ctx, clearml_enabled=clearml_enabled)
    result = _fit_predict_and_score(cfg, ctx=ctx, runtime=runtime)
    frame = runtime.frame
    metric_artifacts = _build_train_metric_artifacts(output_dir=ctx.output_dir, primary_metric=runtime.primary_metric, direction=runtime.direction, task_type=frame.task_type, metrics_holdout=result.metrics_holdout, train_rows=len(frame.train_idx), val_rows=len(frame.val_idx), n_classes=frame.n_classes, cv_summary=result.cv_summary, threshold_payload=result.threshold_payload, calibration_payload=result.calibration_payload, uncertainty_payload=runtime.uncertainty_payload, ci_payload=result.ci_payload)
    (model_bundle_path, train_task_id) = _build_and_save_train_bundle(ctx=ctx, clearml_enabled=clearml_enabled, runtime=runtime, result=result, metric_artifacts=metric_artifacts)
    resolved = runtime.resolved_inputs
    (feature_importance_path, confusion_csv_path) = _emit_train_viz_and_logs(cfg=cfg, ctx=ctx, clearml_enabled=clearml_enabled, task_type=frame.task_type, model=result.model, feature_names=frame.feature_names, y_val=runtime.y_val, y_val_pred=result.y_val_pred, y_val_proba=result.y_val_proba, n_classes=frame.n_classes, class_labels=frame.class_labels, primary_metric=runtime.primary_metric, metrics_holdout=result.metrics_holdout, regression_metrics=result.regression_metrics, uncertainty_payload=runtime.uncertainty_payload)
    _finalize_train_outputs(cfg=cfg, ctx=ctx, clearml_enabled=clearml_enabled, preprocess_run_dir=resolved.preprocess_run_dir, preprocess_bundle=frame.preprocess_bundle, model_variant=runtime.model_variant, model_variant_fit=runtime.model_variant_fit, model_variant_name=runtime.model_variant_name, processed_dataset_id=resolved.processed_dataset_id, split_hash=resolved.split_hash, recipe_hash=resolved.recipe_hash, train_idx=frame.train_idx, val_idx=frame.val_idx, cv_seed=runtime.cv_seed, task_type=frame.task_type, n_classes=frame.n_classes, primary_metric=runtime.primary_metric, direction=runtime.direction, best_score=result.best_score, ci_payload=result.ci_payload, metrics_holdout=result.metrics_holdout, threshold_payload=result.threshold_payload, calibration_payload=result.calibration_payload, uncertainty_payload=runtime.uncertainty_payload, imbalance_report=runtime.imbalance_report, class_labels=frame.class_labels, preds_schema=result.preds_schema, train_task_id=train_task_id, preprocess_task_id=resolved.preprocess_task_id, pipeline_task_id=resolved.pipeline_task_id, model_bundle_path=model_bundle_path, metrics_path=metric_artifacts.metrics_path, metrics_ci_path=metric_artifacts.metrics_ci_path, preds_valid_path=result.preds_valid_path, classes_path=result.classes_path, postprocess_path=metric_artifacts.postprocess_path, calibration_report_path=result.calibration_report_path, feature_importance_path=feature_importance_path, confusion_csv_path=confusion_csv_path)
def run(cfg: Any) -> None:
    ensure_config_alias(cfg, 'group.model.model_variant', 'model_variant')
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    apply_train_model_naming(cfg)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='train_model', tags=identity.tags, properties=identity.user_properties)
    clearml_enabled = is_clearml_enabled(cfg)
    _run_train_model_flow(cfg, ctx=ctx, clearml_enabled=clearml_enabled)
