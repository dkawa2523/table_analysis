from __future__ import annotations
from ..common.collection_utils import to_container as _to_container
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str, normalize_task_type as _normalize_task_type
from ..common.dataset_utils import load_tabular_frame as _load_dataframe, select_tabular_file as _select_tabular_file
from ..common.json_utils import load_json as _load_json
from ..common.model_reference import build_infer_reference
from ..common.probability_utils import extract_positive_class_proba
import csv
from dataclasses import make_dataclass
import json
import math
import numbers
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings
from ..clearml.datasets import get_processed_dataset_local_copy
from ..clearml.hparams import connect_infer
from ..clearml.ui_logger import log_debug_table, log_debug_text, log_plotly, log_scalar, report_input_output_table
from ..io.bundle_io import load_bundle
from ..io.schema import extract_schema_dtypes
from ..monitoring.drift import build_drift_report, build_train_profile, render_drift_markdown
from ..ops.alerting import emit_alert
from ..ops.clearml_identity import apply_clearml_identity, build_project_name
from ..ops.data_quality import raise_on_quality_fail, run_data_quality_gate
from .drift_report import append_drift_summary, annotate_profile, resolve_drift_settings, sample_frame
from ..platform_adapter_artifacts import get_task_artifact_local_copy, hash_recipe, resolve_clearml_task_url, upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_common import PlatformAdapterError
from ..platform_adapter_model import resolve_model_reference
from ..platform_adapter_task_context import update_task_properties
from ..platform_adapter_task_ops import (
    clone_clearml_task,
    enqueue_clearml_task,
    get_clearml_task_status,
    reset_clearml_task_args,
    set_clearml_task_entry_point,
    set_clearml_task_parameters,
    update_clearml_task_tags,
)
from ..uncertainty.conformal import apply_split_conformal_interval
from .infer_support import build_model_reference_payload, build_optimize_hparams as _build_optimize_hparams, ensure_drift_frame as _ensure_drift_frame, frame_from_payload as _frame_from_payload, handle_infer_dry_run as _handle_infer_dry_run, iter_tabular_chunks as _iter_tabular_chunks, load_batch_inputs as _load_batch_inputs, load_train_profile as _load_train_profile, parse_bool as _parse_bool, resolve_batch_children_settings as _resolve_batch_children_settings, resolve_batch_execution_mode, resolve_batch_settings as _resolve_batch_settings, resolve_calibration_info as _resolve_calibration_info, resolve_class_labels as _resolve_class_labels, resolve_optimize_settings as _resolve_optimize_settings, resolve_preprocess_columns as _resolve_preprocess_columns, resolve_threshold_used as _resolve_threshold_used, to_int_or_none as _to_int_or_none
from .lifecycle import emit_outputs_and_manifest, start_runtime
from .pipeline_support import DEFAULT_PIPELINE_CHILD_QUEUE as _DEFAULT_PIPELINE_CHILD_QUEUE
from ..viz.infer_plots import build_input_output_table, build_label_distribution, build_prediction_histogram
from ..viz.optuna_plots import build_contour, build_optimization_history, build_parallel_coordinate, build_param_importance
from ..viz.plots import plot_interval_width_histogram
_COERCE_FAILURE_SAMPLE_LIMIT = 200
_INTERVAL_WIDTH_SAMPLE_LIMIT = 10000
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError, json.JSONDecodeError)
def _get_child_reported_single_value(task_id: str, name: str) -> float | None:
    if not task_id:
        return None
    try:
        from clearml import Task as ClearMLTask
    except ImportError:
        return None
    try:
        task = ClearMLTask.get_task(task_id=task_id)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    getter = getattr(task, 'get_reported_single_value', None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
def _has_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return len(value) > 0
    return True
def _validate_infer_input_sources(*, mode: str, has_input_json: bool, has_input_path: bool, has_batch_inputs_json: bool, has_batch_inputs_path: bool, has_search_space: bool) -> None:
    errors: list[str] = []
    if mode == 'single':
        if has_search_space:
            errors.append('infer.optimize.search_space is not supported in infer.mode=single.')
        if has_batch_inputs_json or has_batch_inputs_path:
            errors.append('infer.batch inputs are not supported in infer.mode=single.')
        if has_input_json and has_input_path:
            errors.append('Specify only one of infer.input_json or infer.input_path for infer.mode=single.')
    elif mode == 'batch':
        if has_search_space:
            errors.append('infer.optimize.search_space is not supported in infer.mode=batch.')
        if has_batch_inputs_json and has_input_json:
            errors.append('Specify only one of infer.batch.inputs_json or infer.input_json for infer.mode=batch.')
        if has_batch_inputs_path and has_input_path:
            errors.append('Specify only one of infer.batch.inputs_path or infer.input_path for infer.mode=batch.')
        if (has_batch_inputs_json or has_input_json) and (has_batch_inputs_path or has_input_path):
            errors.append('Specify either inputs_json or inputs_path for infer.mode=batch, not both.')
    elif mode == 'optimize':
        if has_input_json or has_input_path or has_batch_inputs_json or has_batch_inputs_path:
            errors.append('infer.mode=optimize only supports infer.optimize.search_space.')
        if not has_search_space:
            errors.append('infer.optimize.search_space is required for infer.mode=optimize.')
    if errors:
        raise ValueError(' '.join(errors))
def _verify_processed_dataset(cfg: Any, *, processed_dataset_id: str | None, recipe_hash: str | None, validation_mode: str) -> None:
    if not processed_dataset_id or processed_dataset_id.startswith('local:'):
        return
    if not is_clearml_enabled(cfg):
        return
    try:
        dataset_dir = get_processed_dataset_local_copy(cfg, processed_dataset_id)
    except (PlatformAdapterError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if validation_mode == 'strict':
            raise
        warnings.warn(f'Failed to fetch processed dataset {processed_dataset_id}: {exc}')
        return
    dataset_recipe_hash: str | None = None
    meta_path = dataset_dir / 'meta.json'
    if meta_path.exists():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            meta_payload = {}
        if isinstance(meta_payload, dict):
            dataset_recipe_hash = meta_payload.get('recipe_hash')
    if dataset_recipe_hash is None:
        recipe_path = dataset_dir / 'recipe.json'
        if recipe_path.exists():
            try:
                recipe_payload = json.loads(recipe_path.read_text(encoding='utf-8'))
                if isinstance(recipe_payload, dict):
                    dataset_recipe_hash = hash_recipe(recipe_payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                dataset_recipe_hash = None
    if recipe_hash and dataset_recipe_hash and (recipe_hash != dataset_recipe_hash):
        message = f'processed_dataset recipe_hash mismatch: bundle={recipe_hash} dataset={dataset_recipe_hash}'
        if validation_mode == 'strict':
            raise ValueError(message)
        warnings.warn(message)
def _emit_drift_alert(cfg: Any, ctx: Any, summary: Mapping[str, Any], drift_settings: Mapping[str, Any], *, warn_count: int, fail_count: int, sample_rows: int | None=None) -> None:
    if warn_count <= 0 and fail_count <= 0:
        return
    severity = 'error' if fail_count > 0 else 'warning'
    psi_max = summary.get('psi_max')
    psi_mean = summary.get('psi_mean')
    title = 'Drift threshold exceeded' if fail_count > 0 else 'Drift warning threshold exceeded'
    message = f'warn_count={warn_count}, fail_count={fail_count}, psi_max={psi_max}, psi_mean={psi_mean}'
    context = {'_cfg': cfg, '_ctx': ctx, 'warn_count': warn_count, 'fail_count': fail_count, 'psi_warn_threshold': drift_settings.get('psi_warn_threshold'), 'psi_fail_threshold': drift_settings.get('psi_fail_threshold'), 'metrics': drift_settings.get('metrics'), 'psi_max': psi_max, 'psi_mean': psi_mean}
    if sample_rows is not None:
        context['sample_rows'] = sample_rows
    emit_alert('drift', severity, title, message, context)
def _build_optuna_sampler(name: str, seed: int | None) -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Optuna is required for infer.mode=optimize. Install with: uv sync --extra optuna (or pip install optuna)') from exc
    key = (name or '').strip().lower()
    if key in ('tpe', 'tp'):
        return optuna.samplers.TPESampler(seed=seed)
    if key in ('random', 'rand'):
        return optuna.samplers.RandomSampler(seed=seed)
    if key in ('cmaes', 'cma'):
        try:
            return optuna.samplers.CmaEsSampler(seed=seed)
        except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError('infer.optimize.sampler=cmaes requires cmaes dependency.') from exc
    raise ValueError('infer.optimize.sampler must be tpe, random, or cmaes.')
def _suggest_optuna_params(trial: Any, search_space: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for entry in search_space:
        name = entry.get('name')
        if not name:
            continue
        type_name = entry.get('type')
        if not type_name:
            if 'choices' in entry or 'values' in entry:
                type_name = 'categorical'
            else:
                type_name = 'float'
        type_name = _normalize_optimize_type(type_name)
        if type_name == 'categorical':
            choices = entry.get('choices') or entry.get('values')
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                raise ValueError(f'infer.optimize.search_space {name} missing choices.')
            params[str(name)] = trial.suggest_categorical(str(name), list(choices))
            continue
        low = entry.get('low')
        high = entry.get('high')
        if low is None or high is None:
            raise ValueError(f'infer.optimize.search_space {name} requires low/high.')
        step = entry.get('step')
        log_scale = _parse_bool(entry.get('log')) if 'log' in entry else False
        if step is not None and log_scale:
            raise ValueError(f'infer.optimize.search_space {name} cannot set log and step together.')
        if type_name == 'int':
            try:
                low_int = int(low)
                high_int = int(high)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'infer.optimize.search_space {name} requires int low/high.') from exc
            step_int = None
            if step is not None:
                try:
                    step_int = int(step)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f'infer.optimize.search_space {name} step must be int.') from exc
            params[str(name)] = trial.suggest_int(str(name), low_int, high_int, step=step_int, log=log_scale)
            continue
        try:
            low_float = float(low)
            high_float = float(high)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'infer.optimize.search_space {name} requires float low/high.') from exc
        step_float = None
        if step is not None:
            try:
                step_float = float(step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'infer.optimize.search_space {name} step must be float.') from exc
        params[str(name)] = trial.suggest_float(str(name), low_float, high_float, step=step_float, log=log_scale)
    return params
def _coerce_objective_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            num = float(text)
        except (TypeError, ValueError, OverflowError):
            return None
        return num if math.isfinite(num) else None
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        if len(value) == 1:
            return _coerce_objective_value(value[0])
    return None
def _resolve_objective_value(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in payload:
            numeric = _coerce_objective_value(payload.get(key))
            if numeric is not None:
                return numeric
    return None
def _resolve_child_queue(cfg: Any) -> str | None:
    queue = _normalize_str(_cfg_value(cfg, 'exec_policy.queues.infer'))
    if not queue:
        queue = _normalize_str(_cfg_value(cfg, 'exec_policy.queues.default'))
    if not queue:
        queue = _DEFAULT_PIPELINE_CHILD_QUEUE
    return queue
def _wait_for_child_tasks(task_ids: Sequence[str], *, timeout_sec: float, poll_interval_sec: float) -> dict[str, str]:
    statuses: dict[str, str] = {}
    remaining = set(task_ids)
    deadline = time.monotonic() + timeout_sec
    terminal = {'completed', 'failed', 'stopped', 'closed', 'aborted'}
    while remaining and time.monotonic() < deadline:
        for task_id in list(remaining):
            status = get_clearml_task_status(task_id)
            if status:
                status_value = str(status).lower()
                if status_value in terminal:
                    statuses[task_id] = status_value
                    remaining.remove(task_id)
        if remaining:
            time.sleep(poll_interval_sec)
    for task_id in remaining:
        statuses[task_id] = 'timeout'
    return statuses
def _resolve_child_task_context(cfg: Any, ctx: Any, *, infer_cfg: Any, meta: Mapping[str, Any], context_name: str) -> dict[str, str | None]:
    queue_name = _resolve_child_queue(cfg)
    if not queue_name:
        raise ValueError(
            f'exec_policy.queues.infer or exec_policy.queues.default is required to enqueue {context_name} child tasks.'
        )
    child_project_name = None
    project_root = _normalize_str(_cfg_value(cfg, 'run.clearml.project_root'))
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id'))
    if project_root and usecase_id:
        child_project_name = build_project_name(project_root, usecase_id, stage='infer', process='infer_child', cfg=cfg)
    source_task_id = _normalize_str(_cfg_value(cfg, 'run.clearml.clone_from_task_id')) or str(getattr(ctx.task, 'id', ''))
    if not source_task_id:
        raise PlatformAdapterError(f'ClearML task id is required to clone {context_name} child tasks.')
    explicit_model_id = _normalize_str(getattr(infer_cfg, 'model_id', None))
    child_reference = build_infer_reference(
        model_id=explicit_model_id or _normalize_str(meta.get('model_id')),
        registry_model_id=None if explicit_model_id else _normalize_str(meta.get('registry_model_id')),
        train_task_id=_normalize_str(getattr(infer_cfg, 'train_task_id', None)) or _normalize_str(meta.get('train_task_id')),
    )
    child_model_id = child_reference.get('infer_model_id')
    child_train_task_id = child_reference.get('infer_train_task_id')
    if not child_model_id and (not child_train_task_id):
        raise ValueError(f'infer.model_id or infer.train_task_id is required for {context_name} child tasks.')
    return {'queue_name': queue_name, 'child_project_name': child_project_name, 'source_task_id': source_task_id, 'child_model_id': child_model_id, 'child_train_task_id': child_train_task_id}
def _clone_and_enqueue_infer_child(*, source_task_id: str, child_name: str, queue_name: str, overrides: Mapping[str, Any], tags: Sequence[str] | None=None, reset_args_first: bool=False) -> str:
    child_task_id = clone_clearml_task(source_task_id=source_task_id, task_name=child_name)
    try:
        set_clearml_task_entry_point(child_task_id, 'tools/clearml_entrypoint.py')
    except _RECOVERABLE_ERRORS:
        pass
    if reset_args_first:
        reset_clearml_task_args(child_task_id, [])
    if tags:
        update_clearml_task_tags(child_task_id, add=[str(tag) for tag in tags if str(tag).strip()])
    set_clearml_task_parameters(child_task_id, dict(overrides))
    enqueue_clearml_task(child_task_id, queue_name)
    return child_task_id
def _load_child_prediction_payload(cfg: Any, task_id: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        pred_path = get_task_artifact_local_copy(cfg, task_id, 'prediction.json')
        return (_load_prediction_payload(pred_path), None)
    except _RECOVERABLE_ERRORS as exc:
        scalar_value = _get_child_reported_single_value(task_id, 'infer/prediction')
        if scalar_value is not None:
            return ({'prediction': scalar_value}, None)
        return (None, str(exc))
def _resolve_uncertainty_settings(cfg: Any, bundle: Mapping[str, Any], *, task_type: str) -> dict[str, Any]:
    cfg_enabled = bool(_cfg_value(cfg, 'eval.uncertainty.enabled', False))
    payload = bundle.get('uncertainty')
    if not isinstance(payload, Mapping):
        payload = {}
    method = _normalize_str(payload.get('method')) or 'conformal_split'
    alpha = payload.get('alpha')
    try:
        alpha = float(alpha) if alpha is not None else None
    except (TypeError, ValueError, OverflowError):
        alpha = None
    q = payload.get('q')
    try:
        q = float(q) if q is not None else None
    except (TypeError, ValueError, OverflowError):
        q = None
    info = {'enabled': bool(payload.get('enabled')), 'method': method, 'alpha': alpha, 'q': q, 'use_abs_residual': payload.get('use_abs_residual')}
    enabled = bool(info.get('enabled'))
    if cfg_enabled and (not enabled):
        raise ValueError('eval.uncertainty.enabled is true but model_bundle is missing uncertainty data; retrain with eval.uncertainty.enabled=true.')
    if enabled and task_type != 'regression':
        raise ValueError('uncertainty is supported for regression only.')
    if enabled:
        method = _normalize_str(info.get('method')) or 'conformal_split'
        if method != 'conformal_split':
            raise ValueError('Only conformal_split uncertainty is supported.')
        q = info.get('q')
        if q is None or not math.isfinite(float(q)):
            raise ValueError('uncertainty.q is missing or invalid in model_bundle.')
    return info
def _dtype_kind(expected: str) -> str:
    try:
        from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype, pandas_dtype
    except ImportError:
        return 'object'
    try:
        dtype = pandas_dtype(expected)
    except (TypeError, ValueError):
        return 'object'
    if is_bool_dtype(dtype):
        return 'bool'
    if is_numeric_dtype(dtype):
        return 'numeric'
    if is_datetime64_any_dtype(dtype):
        return 'datetime'
    if is_categorical_dtype(dtype):
        return 'category'
    if is_string_dtype(dtype):
        return 'string'
    return 'object'
def _collect_coerce_failures(series, coerced, *, column: str, reason: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    try:
        missing_before = series.isna()
        missing_after = coerced.isna()
    except (AttributeError, TypeError, ValueError):
        return failures
    failed = ~missing_before & missing_after
    if hasattr(series, 'index'):
        for idx in series.index[failed]:
            failures.append({'row_index': (None if idx is None else int(idx) if isinstance(idx, numbers.Integral) else str(idx)), 'column': column, 'reason': reason})
    return failures
def _coerce_bool_series(series):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    def _coerce(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, numbers.Integral):
            if int(value) == 1:
                return True
            if int(value) == 0:
                return False
            return None
        text = str(value).strip().lower()
        if text in _BOOL_TRUTHY:
            return True
        if text in _BOOL_FALSY:
            return False
        return None
    return series.map(_coerce)
def _parse_json_payload(value: Any, *, key_name: str) -> Any | None:
    payload = _to_container(value)
    if payload is None:
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{key_name} must be valid JSON.') from exc
    return payload
def _frame_to_records(df: Any) -> list[dict[str, Any]]:
    try:
        records = df.to_dict(orient='records')
    except (AttributeError, TypeError, ValueError):
        return []
    sanitized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        sanitized.append({str(k): _sanitize_json_value(v) for (k, v) in record.items()})
    return sanitized
def _sanitize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        num = float(value)
        if not math.isfinite(num):
            return None
        return num
    if isinstance(value, str):
        return value
    return str(value)
def _flatten_prediction_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    flattened: dict[str, Any] = {}
    for (key, value) in payload.items():
        if key == 'predicted_proba':
            if isinstance(value, Mapping):
                labels = list(value.keys())
                safe_labels = _build_proba_column_labels(labels)
                for (label, safe) in zip(labels, safe_labels):
                    flattened[f'pred_proba_{safe}'] = _sanitize_json_value(value.get(label))
                continue
            if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                for (idx, item) in enumerate(value):
                    flattened[f'pred_proba_{idx}'] = _sanitize_json_value(item)
                continue
        if key == 'top_k' and isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
            for (idx, entry) in enumerate(value, start=1):
                if isinstance(entry, Mapping):
                    flattened[f'top{idx}_label'] = _sanitize_json_value(entry.get('label'))
                    flattened[f'top{idx}_proba'] = _sanitize_json_value(entry.get('proba'))
                else:
                    flattened[f'top{idx}'] = _sanitize_json_value(entry)
            continue
        if key == 'prediction' and isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
            if len(value) == 1:
                flattened['prediction'] = _sanitize_json_value(value[0])
            else:
                for (idx, item) in enumerate(value):
                    flattened[f'prediction_{idx}'] = _sanitize_json_value(item)
            continue
        flattened[str(key)] = _sanitize_json_value(value)
    return flattened
def _load_prediction_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == '.json':
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except _RECOVERABLE_ERRORS:
            return None
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, Mapping):
                    return dict(entry)
        return None
    if suffix == '.csv':
        try:
            with path.open('r', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row:
                        return dict(row)
        except _RECOVERABLE_ERRORS:
            return None
    return None
def _select_distribution_samples(output_df: Any) -> tuple[list[float] | None, list[Any] | None]:
    if output_df is None:
        return (None, None)
    try:
        from pandas.api.types import is_numeric_dtype
    except _RECOVERABLE_ERRORS:
        return (None, None)
    if getattr(output_df, 'empty', True):
        return (None, None)
    if 'prediction' in output_df.columns and is_numeric_dtype(output_df['prediction']):
        values = [float(value) for value in output_df['prediction'].dropna().tolist() if isinstance(value, numbers.Real) and math.isfinite(float(value))]
        return (values, None)
    proba_cols = [col for col in output_df.columns if isinstance(col, str) and (col.startswith('pred_proba_') or col.startswith('proba_'))]
    if proba_cols:
        try:
            series = output_df[proba_cols].max(axis=1, skipna=True)
            values = [float(value) for value in series.dropna().tolist() if isinstance(value, numbers.Real) and math.isfinite(float(value))]
            return (values, None)
        except _RECOVERABLE_ERRORS:
            pass
    if 'pred_label' in output_df.columns:
        labels = output_df['pred_label'].tolist()
        return (None, labels)
    return (None, None)
def _resolve_top_k(cfg: Any, *, n_classes: int | None) -> int | None:
    value = _cfg_value(cfg, 'eval.classification.top_k', None)
    if value is None:
        return None
    try:
        top_k = int(value)
    except _RECOVERABLE_ERRORS:
        return None
    if top_k <= 1:
        return None
    if n_classes is not None and top_k > n_classes:
        top_k = n_classes
    return top_k if top_k > 1 else None
def _maybe_attach_feature_names(data: Any, feature_names: Sequence[str] | None):
    if not feature_names:
        return data
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS:
        return data
    try:
        n_cols = int(getattr(data, 'shape', [0, 0])[1])
    except _RECOVERABLE_ERRORS:
        return data
    if len(feature_names) != n_cols:
        return data
    return pd.DataFrame(data, columns=list(feature_names))
def _resolve_model_bundle_path(cfg: Any, *, clearml_enabled: bool) -> tuple[Path, dict[str, Any]]:
    infer_cfg = getattr(cfg, 'infer', None)
    model_id = _normalize_str(getattr(infer_cfg, 'model_id', None))
    train_task_id = _normalize_str(getattr(infer_cfg, 'train_task_id', None))
    try:
        resolved = resolve_model_reference(cfg=cfg if clearml_enabled else None, model_id=model_id, train_task_id=train_task_id)
    except PlatformAdapterError as exc:
        raise RuntimeError(str(exc)) from exc
    meta = {'model_id': resolved.model_id, 'train_task_id': resolved.train_task_id, 'registry_model_id': resolved.registry_model_id, 'model_source': resolved.source, 'reference_kind': resolved.kind, **dict(resolved.metadata)}
    return (resolved.model_bundle_path, meta)
def _build_dummy_input(preprocess_bundle: dict[str, Any]):
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    (feature_columns, numeric_features, categorical_features, _) = _resolve_preprocess_columns(preprocess_bundle)
    numeric_set = set(numeric_features)
    categorical_set = set(categorical_features)
    if not feature_columns:
        raise ValueError('preprocess bundle does not define feature_columns.')
    row: dict[str, Any] = {}
    for col in feature_columns:
        if col in categorical_set:
            row[col] = 'unknown'
        elif col in numeric_set:
            row[col] = 0.0
        else:
            row[col] = None
    return pd.DataFrame([row])
def _align_input_frame(df, preprocess_bundle: Mapping[str, Any], *, allow_missing: bool):
    (feature_columns, numeric_features, categorical_features, target_column) = _resolve_preprocess_columns(preprocess_bundle)
    if feature_columns:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing and (not allow_missing):
            raise ValueError(f'Input is missing required columns: {missing}')
        if missing:
            df = df.copy()
            numeric_set = set(numeric_features)
            categorical_set = set(categorical_features)
            for col in missing:
                if col in numeric_set:
                    df[col] = float('nan')
                elif col in categorical_set:
                    df[col] = None
                else:
                    df[col] = None
        df = df[feature_columns]
    else:
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])
        feature_columns = list(df.columns)
    return (df, feature_columns)
def _collect_schema_issues(df, preprocess_bundle: Mapping[str, Any]) -> dict[str, Any]:
    (feature_columns, _numeric_features, _categorical_features, target_column) = _resolve_preprocess_columns(preprocess_bundle)
    expected_dtypes = extract_schema_dtypes(preprocess_bundle.get('schema') or {})
    if feature_columns:
        missing_columns = [col for col in feature_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in feature_columns]
        if target_column and target_column in extra_columns:
            extra_columns.remove(target_column)
    else:
        missing_columns = []
        extra_columns = []
    dtype_mismatch: list[dict[str, Any]] = []
    for (col, expected) in expected_dtypes.items():
        if feature_columns and col not in feature_columns:
            continue
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if str(expected) != actual:
            dtype_mismatch.append({'column': col, 'expected': str(expected), 'actual': actual})
    return {'feature_columns': feature_columns, 'expected_dtypes': expected_dtypes, 'missing_columns': missing_columns, 'extra_columns': extra_columns, 'dtype_mismatch': dtype_mismatch}
def _coerce_frame_to_schema(df, *, expected_dtypes: Mapping[str, str], feature_columns: Sequence[str]) -> tuple[Any, list[dict[str, Any]]]:
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    coerce_failures: list[dict[str, Any]] = []
    for (col, expected) in expected_dtypes.items():
        if feature_columns and col not in feature_columns:
            continue
        if col not in df.columns:
            continue
        kind = _dtype_kind(expected)
        if kind == 'numeric':
            coerced = pd.to_numeric(df[col], errors='coerce')
            coerce_failures.extend(_collect_coerce_failures(df[col], coerced, column=col, reason='numeric_coerce_failed'))
            df[col] = coerced
        elif kind == 'datetime':
            coerced = pd.to_datetime(df[col], errors='coerce')
            coerce_failures.extend(_collect_coerce_failures(df[col], coerced, column=col, reason='datetime_coerce_failed'))
            df[col] = coerced
        elif kind == 'bool':
            coerced = _coerce_bool_series(df[col])
            coerce_failures.extend(_collect_coerce_failures(df[col], coerced, column=col, reason='bool_coerce_failed'))
            df[col] = coerced
    return (df, coerce_failures)
def _count_schema_issues(issues: Mapping[str, Any]) -> int:
    total = 0
    for key in ('missing_columns', 'extra_columns', 'dtype_mismatch', 'coerce_failures'):
        value = issues.get(key) or []
        try:
            total += len(value)
        except _RECOVERABLE_ERRORS:
            continue
    return total
def _init_validation_accumulator() -> dict[str, Any]:
    return {'issues': {'missing_columns': set(), 'extra_columns': set(), 'dtype_mismatch': {}, 'coerce_failures': []}, 'warnings_count': 0, 'errors_count': 0, 'ok': True, 'total_issues': 0}
def _update_validation_accumulator(acc: dict[str, Any], issues: Mapping[str, Any], *, validation_mode: str) -> None:
    issue_count = _count_schema_issues(issues)
    acc['total_issues'] += issue_count
    if issue_count > 0:
        acc['ok'] = False
    if validation_mode == 'strict':
        acc['errors_count'] += issue_count
    else:
        acc['warnings_count'] += issue_count
    missing = issues.get('missing_columns') or []
    extra = issues.get('extra_columns') or []
    dtype_mismatch = issues.get('dtype_mismatch') or []
    coerce_failures = issues.get('coerce_failures') or []
    acc['issues']['missing_columns'].update(missing)
    acc['issues']['extra_columns'].update(extra)
    for entry in dtype_mismatch:
        key = (entry.get('column'), entry.get('expected'), entry.get('actual'))
        acc['issues']['dtype_mismatch'][key] = dict(entry)
    if coerce_failures and len(acc['issues']['coerce_failures']) < _COERCE_FAILURE_SAMPLE_LIMIT:
        remaining = _COERCE_FAILURE_SAMPLE_LIMIT - len(acc['issues']['coerce_failures'])
        acc['issues']['coerce_failures'].extend(list(coerce_failures)[:remaining])
def _finalize_validation_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    issues = acc['issues']
    return {'issues': {'missing_columns': sorted(issues.get('missing_columns', [])), 'extra_columns': sorted(issues.get('extra_columns', [])), 'dtype_mismatch': list(issues.get('dtype_mismatch', {}).values()), 'coerce_failures': list(issues.get('coerce_failures', []))}, 'warnings_count': int(acc.get('warnings_count') or 0), 'errors_count': int(acc.get('errors_count') or 0), 'ok': bool(acc.get('ok'))}
def _format_list_preview(values: Sequence[str], *, limit: int=10) -> str:
    if not values:
        return ''
    preview = ', '.join(values[:limit])
    return f'{preview}, ...(+{len(values) - limit})' if len(values) > limit else preview
def _write_schema_errors(output_dir: Path, issues: Mapping[str, Any]) -> tuple[Path, Path]:
    errors_json_path = output_dir / 'errors.json'
    errors_csv_path = output_dir / 'errors.csv'
    payload = {'missing_columns': list(issues.get('missing_columns') or []), 'extra_columns': list(issues.get('extra_columns') or []), 'dtype_mismatch': list(issues.get('dtype_mismatch') or []), 'coerce_failures': list(issues.get('coerce_failures') or [])}
    errors_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    fieldnames = ['issue_type', 'column', 'row_index', 'expected_dtype', 'actual_dtype', 'reason']
    with errors_csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for col in payload['missing_columns']:
            writer.writerow({'issue_type': 'missing_columns', 'column': col})
        for col in payload['extra_columns']:
            writer.writerow({'issue_type': 'extra_columns', 'column': col})
        for entry in payload['dtype_mismatch']:
            writer.writerow({'issue_type': 'dtype_mismatch', 'column': entry.get('column'), 'expected_dtype': entry.get('expected'), 'actual_dtype': entry.get('actual')})
        for entry in payload['coerce_failures']:
            writer.writerow({'issue_type': 'coerce_failures', 'column': entry.get('column'), 'row_index': entry.get('row_index'), 'reason': entry.get('reason')})
    return (errors_json_path, errors_csv_path)
def _write_infer_summary(output_dir: Path, *, mode: str, validation_mode: str, validation: Mapping[str, Any], errors_path: Path | None, uncertainty: Mapping[str, Any] | None=None) -> Path:
    issues = validation.get('issues') or {}
    missing_columns = list(issues.get('missing_columns') or [])
    extra_columns = list(issues.get('extra_columns') or [])
    dtype_mismatch = list(issues.get('dtype_mismatch') or [])
    coerce_failures = list(issues.get('coerce_failures') or [])
    lines = ['# Infer Summary', '', f'- mode: {mode}', f'- schema_validation_mode: {validation_mode}', f"- schema_validation_ok: {bool(validation.get('ok'))}", f"- warnings_count: {validation.get('warnings_count')}", f"- errors_count: {validation.get('errors_count')}"]
    if errors_path is not None:
        lines.append(f'- errors_path: {errors_path}')
    if uncertainty and uncertainty.get('enabled'):
        alpha = uncertainty.get('alpha')
        method = uncertainty.get('method')
        q_value = uncertainty.get('q')
        lines.extend(['', '## Prediction Interval'])
        lines.append(f'- method: {method}')
        if alpha is not None:
            coverage = (1.0 - float(alpha)) * 100.0
            lines.append(f'- alpha: {alpha} ({coverage:.1f}% interval)')
        else:
            lines.append('- alpha: unknown')
        if q_value is not None:
            lines.append(f'- interval: pred ± {q_value}')
    if _count_schema_issues(issues) > 0:
        lines.extend(['', '## Schema Validation Issues'])
        if missing_columns:
            lines.append(f'- missing_columns: {_format_list_preview(missing_columns)}')
        if extra_columns:
            lines.append(f'- extra_columns: {_format_list_preview(extra_columns)}')
        if dtype_mismatch:
            mismatch_cols = [entry.get('column', '') for entry in dtype_mismatch if entry.get('column')]
            lines.append(f'- dtype_mismatch: {_format_list_preview(mismatch_cols)}')
        if coerce_failures:
            lines.append(f'- coerce_failures: {len(coerce_failures)} rows')
    summary_path = output_dir / 'summary.md'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary_path
def _upload_summary_artifacts(ctx: Any, *, clearml_enabled: bool, summary_path: Path, errors_json_path: Path | None=None, errors_csv_path: Path | None=None) -> None:
    if not clearml_enabled:
        return
    upload_artifact(ctx, summary_path.name, summary_path)
    if errors_json_path is not None:
        upload_artifact(ctx, errors_json_path.name, errors_json_path)
    if errors_csv_path is not None:
        upload_artifact(ctx, errors_csv_path.name, errors_csv_path)
def _write_input_preview(df, mode: str, output_dir: Path) -> Path:
    if mode == 'single':
        path = output_dir / 'input_preview.json'
        payload: dict[str, Any] = {}
        if not df.empty:
            row = df.iloc[0].to_dict()
            payload = {str(k): _sanitize_json_value(v) for (k, v) in row.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return path
    path = output_dir / 'input_preview.csv'
    df.head(5).to_csv(path, index=False)
    return path
def _load_preview_sample(path: Path | None, *, max_rows: int=5) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix == '.json':
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except _RECOVERABLE_ERRORS:
            return None
        if isinstance(payload, list):
            return [dict(row) for row in payload[:max_rows] if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            return [dict(payload)]
        return None
    if suffix == '.csv':
        rows: list[dict[str, Any]] = []
        try:
            with path.open('r', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                for (idx, row) in enumerate(reader):
                    if idx >= max_rows:
                        break
                    rows.append(dict(row))
        except _RECOVERABLE_ERRORS:
            return None
        return rows
    return None
def _log_debug_samples(ctx: Any, *, input_sample: Any | None, output_sample: Any | None, input_preview_path: Path | None, predictions_path: Path | None) -> None:
    if ctx is None or getattr(ctx, 'task', None) is None:
        return
    if input_sample is None:
        input_sample = _load_preview_sample(input_preview_path)
    if output_sample is None:
        output_sample = _load_preview_sample(predictions_path)
    if input_sample is not None:
        log_debug_table(ctx.task, 'infer', 'input_sample', input_sample, step=0)
    elif input_preview_path is not None:
        log_debug_text(ctx.task, 'infer', 'input_sample', f'input preview: {input_preview_path}', step=0)
    if output_sample is not None:
        log_debug_table(ctx.task, 'infer', 'output_sample', output_sample, step=0)
    elif predictions_path is not None:
        log_debug_text(ctx.task, 'infer', 'output_sample', f'predictions path: {predictions_path}', step=0)
def _resolve_table_samples(input_sample: Any | None, output_sample: Any | None, *, input_preview_path: Path | None, predictions_path: Path | None) -> tuple[Any | None, Any | None]:
    if input_sample is None:
        input_sample = _load_preview_sample(input_preview_path)
    if output_sample is None:
        output_sample = _load_preview_sample(predictions_path)
    return (input_sample, output_sample)
def _resolve_input_output_table_settings(cfg: Any) -> dict[str, int]:
    def _as_int(value: Any, default: int) -> int:
        try:
            num = int(value)
        except _RECOVERABLE_ERRORS:
            return default
        return num if num > 0 else default
    return {'max_rows': _as_int(_cfg_value(cfg, 'infer.plots.max_rows', 5), 5), 'max_input_columns': _as_int(_cfg_value(cfg, 'infer.plots.max_input_columns', 20), 20), 'max_output_columns': _as_int(_cfg_value(cfg, 'infer.plots.max_output_columns', 12), 12)}
def _resolve_model_abbr(bundle: Mapping[str, Any], meta: Mapping[str, Any]) -> str | None:
    model_variant = _normalize_str(bundle.get('model_variant'))
    if model_variant:
        return model_variant
    model_id = _normalize_str(meta.get('model_id'))
    if model_id:
        try:
            return Path(model_id).stem or model_id
        except _RECOVERABLE_ERRORS:
            return model_id
    return None
def _resolve_model_provenance(bundle: Mapping[str, Any], *, preprocess_bundle: Mapping[str, Any], meta: Mapping[str, Any], model_bundle_path: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    raw = bundle.get('provenance')
    if isinstance(raw, Mapping):
        provenance.update(dict(raw))
    if provenance.get('train_task_id') is None:
        provenance['train_task_id'] = _normalize_str(meta.get('train_task_id'))
    if provenance.get('processed_dataset_id') is None:
        provenance['processed_dataset_id'] = _normalize_str(bundle.get('processed_dataset_id'))
    if provenance.get('split_hash') is None:
        provenance['split_hash'] = _normalize_str(bundle.get('split_hash'))
    if provenance.get('recipe_hash') is None:
        provenance['recipe_hash'] = _normalize_str(bundle.get('recipe_hash'))
    if provenance.get('preprocess_variant') is None:
        provenance['preprocess_variant'] = _normalize_str(preprocess_bundle.get('preprocess_variant'))
    if provenance.get('raw_dataset_id') is None:
        provenance['raw_dataset_id'] = _normalize_str(bundle.get('raw_dataset_id'))
    if provenance.get('raw_dataset_id') is None:
        manifest_path = model_bundle_path.parent / 'manifest.json'
        if manifest_path.exists():
            try:
                manifest_payload = _load_json(manifest_path)
            except _RECOVERABLE_ERRORS:
                manifest_payload = {}
            inputs = manifest_payload.get('inputs') or {}
            provenance['raw_dataset_id'] = _normalize_str(inputs.get('raw_dataset_id'))
    return {key: value for (key, value) in provenance.items() if value is not None}
def _preds_to_rows(preds: Any) -> list[list[Any]]:
    if preds is None:
        return []
    if hasattr(preds, 'tolist'):
        data = preds.tolist()
    else:
        try:
            data = list(preds)
        except _RECOVERABLE_ERRORS:
            data = preds
    if isinstance(data, list):
        if not data:
            return []
        first = data[0]
        if isinstance(first, (list, tuple)):
            return [list(row) for row in data]
        return [[item] for item in data]
    return [[data]]
def _build_classification_predictions_frame(transformed: Any, *, predictor: Any, model: Any, threshold_used: float | None, class_labels: list[Any] | None, label_encoder: Any, top_k: int | None, proba_prefix: str) -> tuple[Any, list[Any]]:
    if not hasattr(predictor, 'predict_proba'):
        raise ValueError('classification infer requires predict_proba on the model.')
    proba = predictor.predict_proba(transformed)
    if threshold_used is not None:
        positive_proba = extract_positive_class_proba(proba)
        preds = (positive_proba >= threshold_used).astype(int)
    else:
        preds = predictor.predict(transformed)
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        pred_labels = label_encoder.inverse_transform(preds)
    else:
        pred_labels = preds
        if class_labels:
            try:
                pred_labels = [class_labels[int(idx)] for idx in preds]
            except _RECOVERABLE_ERRORS:
                pred_labels = preds
    if hasattr(pred_labels, 'tolist'):
        pred_labels = pred_labels.tolist()
    if not isinstance(pred_labels, list):
        pred_labels = list(pred_labels)
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('numpy is required for classification probabilities.') from exc
    proba_arr = np.asarray(proba)
    if proba_arr.ndim == 1:
        proba_arr = np.stack([1.0 - proba_arr, proba_arr], axis=1)
    if class_labels is None or len(class_labels) != int(proba_arr.shape[1]):
        class_labels = [str(i) for i in range(int(proba_arr.shape[1]))]
    top_indices = None
    if top_k is not None:
        order = np.argsort(proba_arr, axis=1)[:, ::-1]
        top_indices = order[:, :top_k]
    data: dict[str, list[Any]] = {'pred_label': [_sanitize_json_value(label) for label in pred_labels]}
    if threshold_used is not None:
        data['threshold_used'] = [_sanitize_json_value(threshold_used)] * len(pred_labels)
    proba_columns = _format_proba_columns(class_labels, prefix=proba_prefix)
    for (idx, col_name) in enumerate(proba_columns):
        data[col_name] = [_sanitize_json_value(v) for v in proba_arr[:, idx]]
    if top_indices is not None:
        for rank in range(1, top_k + 1):
            labels: list[Any] = []
            probs: list[Any] = []
            for (row_idx, col_idx) in enumerate(top_indices[:, rank - 1]):
                label = class_labels[col_idx] if class_labels else str(col_idx)
                labels.append(_sanitize_json_value(label))
                probs.append(_sanitize_json_value(proba_arr[row_idx][col_idx]))
            data[f'top{rank}_label'] = labels
            data[f'top{rank}_proba'] = probs
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    return (pd.DataFrame(data), class_labels)
def _build_regression_predictions_frame(transformed: Any, *, model: Any, uncertainty_enabled: bool, uncertainty_q: float | None) -> tuple[Any, list[float] | None]:
    preds = model.predict(transformed)
    rows = _preds_to_rows(preds)
    lower = None
    upper = None
    if uncertainty_enabled and uncertainty_q is not None:
        try:
            (lower, upper) = apply_split_conformal_interval(preds, float(uncertainty_q))
        except ValueError as exc:
            raise ValueError('uncertainty intervals require 1D regression predictions.') from exc
    if rows:
        n_cols = len(rows[0])
    else:
        n_cols = 1
    data: dict[str, list[Any]] = {}
    if n_cols == 1:
        data['prediction'] = [_sanitize_json_value(row[0]) for row in rows] if rows else [None]
        if lower is not None and upper is not None:
            data['pred_lower'] = [_sanitize_json_value(v) for v in lower]
            data['pred_upper'] = [_sanitize_json_value(v) for v in upper]
    else:
        for idx in range(n_cols):
            data[f'prediction_{idx}'] = [_sanitize_json_value(row[idx]) if idx < len(row) else None for row in rows]
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    interval_widths = None
    if lower is not None and upper is not None and (n_cols == 1):
        interval_widths = [float(upper[idx] - lower[idx]) for idx in range(len(lower))]
    return (pd.DataFrame(data), interval_widths)
def _write_predictions_csv_chunk(path: Path, df: Any, *, write_mode: str, header_written: bool) -> bool:
    if write_mode == 'append':
        mode = 'a'
        header = not path.exists() and (not header_written)
    else:
        mode = 'a' if header_written else 'w'
        header = not header_written
    df.to_csv(path, mode=mode, header=header, index=False)
    return True
def _write_predictions_parquet_chunk(path: Path, df: Any, writer: Any | None):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pyarrow is required for parquet output.') from exc
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer
def _update_drift_sample(sample_df: Any, new_df: Any, *, sample_n: int | None, rng: Any):
    if sample_n is None:
        return new_df if sample_df is None else sample_df
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS:
        return new_df if sample_df is None else sample_df
    if sample_df is None:
        if len(new_df) <= sample_n:
            return new_df
        if rng is not None:
            return new_df.sample(n=sample_n, random_state=int(rng.integers(0, 1000000000)))
        return new_df.head(sample_n)
    combined = pd.concat([sample_df, new_df], ignore_index=True)
    if len(combined) <= sample_n:
        return combined
    if rng is not None:
        return combined.sample(n=sample_n, random_state=int(rng.integers(0, 1000000000)))
    return combined.head(sample_n)
def _update_interval_width_sample(sample: list[float], widths: list[float], *, rng: Any) -> list[float]:
    if not widths:
        return sample
    sample.extend([float(value) for value in widths])
    if len(sample) <= _INTERVAL_WIDTH_SAMPLE_LIMIT:
        return sample
    if rng is not None:
        try:
            import numpy as np
        except _RECOVERABLE_ERRORS:
            return sample[:_INTERVAL_WIDTH_SAMPLE_LIMIT]
        sample = list(np.asarray(sample)[rng.choice(len(sample), size=_INTERVAL_WIDTH_SAMPLE_LIMIT, replace=False)])
        return sample
    return sample[:_INTERVAL_WIDTH_SAMPLE_LIMIT]
def _build_proba_column_labels(labels: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    safe_labels: list[str] = []
    for (idx, label) in enumerate(labels):
        safe = re.sub('[^0-9A-Za-z_]+', '_', str(label)).strip('_')
        if not safe:
            safe = str(idx)
        if safe in seen:
            safe = f'{safe}_{idx}'
        seen.add(safe)
        safe_labels.append(safe)
    return safe_labels
def _format_proba_columns(labels: Sequence[Any], *, prefix: str='proba_') -> list[str]:
    return [f'{prefix}{safe}' for safe in _build_proba_column_labels(labels)]
def _build_proba_payload(values: Sequence[Any], labels: Sequence[Any] | None) -> Any:
    if labels:
        return {str(label): _sanitize_json_value(value) for (label, value) in zip(labels, values)}
    return [_sanitize_json_value(value) for value in values]
def _prepare_inputs(cfg: Any, preprocess_bundle: dict[str, Any], mode: str):
    infer_cfg = getattr(cfg, 'infer', None)
    dataset_path = _normalize_str(getattr(infer_cfg, 'input_path', None))
    if not dataset_path:
        dataset_path = _normalize_str(getattr(getattr(cfg, 'data', None), 'dataset_path', None))
    input_payload = _parse_json_payload(getattr(infer_cfg, 'input_json', None), key_name='infer.input_json')
    df = None
    resolved_path = None
    if input_payload is not None:
        df = _frame_from_payload(input_payload)
    elif dataset_path:
        path = Path(dataset_path).expanduser().resolve()
        if path.suffix.lower() == '.json':
            input_payload = _load_json(path)
            df = _frame_from_payload(input_payload)
        else:
            data_path = _select_tabular_file(path)
            df = _load_dataframe(data_path)
            resolved_path = str(data_path)
    else:
        df = _build_dummy_input(preprocess_bundle)
    if df is None:
        raise RuntimeError('infer failed to resolve input data.')
    if mode == 'single':
        return (df.head(1), resolved_path or dataset_path)
    if mode == 'batch':
        if input_payload is not None and resolved_path is None and (not dataset_path):
            return (df, None)
        if not (resolved_path or dataset_path):
            raise ValueError('infer.input_path or data.dataset_path is required for batch mode.')
    return (df, resolved_path or dataset_path)
def _validate_inputs(df, preprocess_bundle: Mapping[str, Any], *, validation_mode: str) -> tuple[Any, dict[str, Any]]:
    issue_info = _collect_schema_issues(df, preprocess_bundle)
    feature_columns = issue_info.get('feature_columns') or []
    expected_dtypes = issue_info.get('expected_dtypes') or {}
    dtype_mismatch = list(issue_info.get('dtype_mismatch') or [])
    coerce_failures: list[dict[str, Any]] = []
    if validation_mode == 'coerce':
        df = df.copy()
        (df, coerce_failures) = _coerce_frame_to_schema(df, expected_dtypes=expected_dtypes, feature_columns=feature_columns)
        for entry in dtype_mismatch:
            col = entry.get('column')
            if col in df.columns:
                entry['actual_after'] = str(df[col].dtype)
                entry['coerced'] = True
    issues = {'missing_columns': list(issue_info.get('missing_columns') or []), 'extra_columns': list(issue_info.get('extra_columns') or []), 'dtype_mismatch': dtype_mismatch, 'coerce_failures': coerce_failures}
    total_issues = _count_schema_issues(issues)
    ok = total_issues == 0
    if validation_mode == 'strict':
        warnings_count = 0
        errors_count = total_issues
    else:
        warnings_count = total_issues
        errors_count = 0
    if not (validation_mode == 'strict' and total_issues > 0):
        (df, _) = _align_input_frame(df, preprocess_bundle, allow_missing=validation_mode != 'strict')
    return (df, {'issues': issues, 'warnings_count': warnings_count, 'errors_count': errors_count, 'ok': ok})
def _validate_inputs_with_summary(*, ctx: Any, clearml_enabled: bool, mode: str, validation_mode: str, preprocess_bundle: Mapping[str, Any], inputs_df: Any, uncertainty_info: Mapping[str, Any]) -> tuple[Any, dict[str, Any], Path, Path | None, Path | None]:
    (validated_df, validation) = _validate_inputs(inputs_df, preprocess_bundle, validation_mode=validation_mode)
    issues = validation.get('issues') or {}
    errors_json_path: Path | None = None
    errors_csv_path: Path | None = None
    if _count_schema_issues(issues) > 0:
        (errors_json_path, errors_csv_path) = _write_schema_errors(ctx.output_dir, issues)
    summary_path = _write_infer_summary(ctx.output_dir, mode=mode, validation_mode=validation_mode, validation=validation, errors_path=errors_json_path, uncertainty=uncertainty_info)
    _upload_summary_artifacts(ctx, clearml_enabled=clearml_enabled, summary_path=summary_path, errors_json_path=errors_json_path, errors_csv_path=errors_csv_path)
    if validation_mode == 'strict' and (not validation.get('ok')):
        raise ValueError('Input schema validation failed; see errors.json for details.')
    return (validated_df, validation, summary_path, errors_json_path, errors_csv_path)
InferRuntimeSettings = make_dataclass('InferRuntimeSettings', [('mode', str), ('validation_mode', str), ('optimize_settings', dict[str, Any] | None), ('infer_cfg', Any), ('batch_cfg', Any), ('model_id_value', str | None), ('train_task_id_value', str | None), ('input_payload', list[dict[str, Any]] | dict[str, Any] | None), ('batch_inputs_payload', list[dict[str, Any]] | None), ('batch_inputs_path_value', str | None), ('input_path_value', str | None), ('batch_execution', str), ('use_batch_children', bool), ('use_optimize_children', bool), ('is_child_task', bool), ('input_source', str), ('input_json_label', str | None), ('table_settings', dict[str, int]), ('include_dataset', bool), ('include_execution', bool), ('optimize_hparams', dict[str, Any] | None), ('has_search_space', bool), ('dry_run', bool)], frozen=True)
InferModelContext = make_dataclass('InferModelContext', [('model_bundle_path', Path), ('meta', dict[str, Any]), ('bundle', dict[str, Any]), ('model', Any), ('calibrated_model', Any), ('preprocess_bundle', dict[str, Any]), ('processed_dataset_id', str | None), ('split_hash', str), ('recipe_hash', str), ('task_type', str), ('n_classes', int | None), ('model_abbr', str | None), ('provenance', dict[str, Any])], frozen=True)
def _resolve_infer_model_context(cfg: Any, *, clearml_enabled: bool, validation_mode: str) -> InferModelContext:
    (model_bundle_path, meta) = _resolve_model_bundle_path(cfg, clearml_enabled=clearml_enabled)
    bundle = load_bundle(model_bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError('model_bundle.joblib is invalid.')
    model = bundle.get('model')
    calibrated_model = bundle.get('calibrated_model')
    preprocess_bundle = bundle.get('preprocess_bundle') or {}
    processed_dataset_id = _normalize_str(bundle.get('processed_dataset_id'))
    split_hash = _normalize_str(bundle.get('split_hash'))
    recipe_hash = _normalize_str(bundle.get('recipe_hash'))
    task_type = _normalize_task_type(bundle.get('task_type'))
    n_classes = bundle.get('n_classes')
    try:
        n_classes = int(n_classes) if n_classes is not None else None
    except _RECOVERABLE_ERRORS:
        n_classes = None
    if model is None or not isinstance(preprocess_bundle, dict):
        raise ValueError('model_bundle is missing model or preprocess_bundle.')
    if not split_hash or not recipe_hash:
        raise ValueError('model_bundle is missing split_hash or recipe_hash.')
    config_processed_dataset_id = _normalize_str(_cfg_value(cfg, 'data.processed_dataset_id'))
    if config_processed_dataset_id and processed_dataset_id:
        if config_processed_dataset_id != processed_dataset_id:
            message = f'data.processed_dataset_id does not match model_bundle processed_dataset_id ({config_processed_dataset_id} != {processed_dataset_id}).'
            if validation_mode == 'strict':
                raise ValueError(message)
            warnings.warn(message)
    dataset_id_for_check = config_processed_dataset_id or processed_dataset_id
    _verify_processed_dataset(cfg, processed_dataset_id=dataset_id_for_check, recipe_hash=recipe_hash, validation_mode=validation_mode)
    model_abbr = _resolve_model_abbr(bundle, meta)
    provenance = _resolve_model_provenance(bundle, preprocess_bundle=preprocess_bundle, meta=meta, model_bundle_path=model_bundle_path)
    return InferModelContext(model_bundle_path=model_bundle_path, meta=dict(meta), bundle=bundle, model=model, calibrated_model=calibrated_model, preprocess_bundle=preprocess_bundle, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, task_type=task_type, n_classes=n_classes, model_abbr=model_abbr, provenance=provenance)
def _resolve_infer_runtime_settings(cfg: Any, *, clearml_enabled: bool) -> InferRuntimeSettings:
    mode = _normalize_str(getattr(getattr(cfg, 'infer', None), 'mode', None))
    if not mode:
        group_mode = getattr(getattr(cfg, 'group', None), 'infer_mode', None)
        mode = _normalize_str(getattr(getattr(group_mode, 'infer_mode', None), 'name', None))
    mode = (mode or 'single').lower()
    if mode not in ('single', 'batch', 'optimize'):
        raise ValueError(f'Unsupported infer mode: {mode}')
    validation_mode = (_normalize_str(getattr(getattr(getattr(cfg, 'infer', None), 'validation', None), 'mode', None)) or 'warn').lower()
    if validation_mode not in ('warn', 'strict', 'coerce'):
        raise ValueError(f'Unsupported infer.validation.mode: {validation_mode}')
    optimize_settings = _resolve_optimize_settings(cfg) if mode == 'optimize' else None
    infer_cfg = getattr(cfg, 'infer', None)
    batch_cfg = getattr(infer_cfg, 'batch', None)
    optimize_cfg = getattr(infer_cfg, 'optimize', None)
    has_input_json = _has_payload(getattr(infer_cfg, 'input_json', None))
    has_input_path = _normalize_str(getattr(infer_cfg, 'input_path', None)) is not None
    has_batch_inputs_json = _has_payload(getattr(batch_cfg, 'inputs_json', None))
    has_batch_inputs_path = _normalize_str(getattr(batch_cfg, 'inputs_path', None)) is not None
    has_search_space = _has_payload(getattr(optimize_cfg, 'search_space', None))
    _validate_infer_input_sources(mode=mode, has_input_json=has_input_json, has_input_path=has_input_path, has_batch_inputs_json=has_batch_inputs_json, has_batch_inputs_path=has_batch_inputs_path, has_search_space=has_search_space)
    model_id_value = _normalize_str(getattr(infer_cfg, 'model_id', None))
    train_task_id_value = _normalize_str(getattr(infer_cfg, 'train_task_id', None))
    input_payload = _parse_json_payload(getattr(infer_cfg, 'input_json', None), key_name='infer.input_json')
    batch_inputs_payload = _parse_json_payload(getattr(batch_cfg, 'inputs_json', None), key_name='infer.batch.inputs_json')
    if mode == 'batch' and batch_inputs_payload is None:
        batch_inputs_payload = input_payload
    batch_inputs_path_value = _normalize_str(getattr(batch_cfg, 'inputs_path', None))
    input_path_value = batch_inputs_path_value if mode == 'batch' else None
    if not input_path_value:
        input_path_value = _normalize_str(getattr(infer_cfg, 'input_path', None))
    if not input_path_value:
        input_path_value = _normalize_str(getattr(getattr(cfg, 'data', None), 'dataset_path', None))
    batch_execution = resolve_batch_execution_mode(cfg, clearml_enabled=clearml_enabled) if mode == 'batch' else 'inline'
    has_batch_input_source = any((batch_inputs_payload is not None, batch_inputs_path_value, input_payload is not None, input_path_value))
    use_batch_children = mode == 'batch' and batch_execution == 'clearml_children' and has_batch_input_source
    use_optimize_children = mode == 'optimize' and clearml_enabled
    is_child_task = _parse_bool(_cfg_value(cfg, 'infer.batch.child_task')) or _parse_bool(_cfg_value(cfg, 'infer.optimize.child_task'))
    connect_payload = batch_inputs_payload if mode == 'batch' and batch_inputs_payload is not None else input_payload
    input_source = 'generated'
    input_json_label = None
    if connect_payload is not None:
        input_source = 'json'
        input_json_label = 'inline'
    elif input_path_value:
        input_source = 'path'
    if mode == 'optimize':
        input_source = 'search_space'
    table_settings = _resolve_input_output_table_settings(cfg)
    include_dataset = not (mode == 'batch' and use_batch_children and (not is_child_task) or (mode == 'optimize' and use_optimize_children and (not is_child_task)))
    include_execution = not is_child_task
    optimize_hparams = None
    if mode == 'optimize' and (not is_child_task):
        optimize_hparams = _build_optimize_hparams(optimize_settings)
    dry_run = bool(getattr(infer_cfg, 'dry_run', False))
    if not dry_run and (not (model_id_value or train_task_id_value)):
        raise ValueError('infer.model_id or infer.train_task_id is required.')
    return InferRuntimeSettings(mode=mode, validation_mode=validation_mode, optimize_settings=optimize_settings, infer_cfg=infer_cfg, batch_cfg=batch_cfg, model_id_value=model_id_value, train_task_id_value=train_task_id_value, input_payload=input_payload, batch_inputs_payload=batch_inputs_payload, batch_inputs_path_value=batch_inputs_path_value, input_path_value=input_path_value, batch_execution=batch_execution, use_batch_children=use_batch_children, use_optimize_children=use_optimize_children, is_child_task=is_child_task, input_source=input_source, input_json_label=input_json_label, table_settings=table_settings, include_dataset=include_dataset, include_execution=include_execution, optimize_hparams=optimize_hparams, has_search_space=has_search_space, dry_run=dry_run)
def _run_infer_drift_analysis(cfg: Any, ctx: Any, *, clearml_enabled: bool, drift_settings: dict[str, Any], validation_mode: str, bundle: dict[str, Any], model_bundle_path: Path, meta: dict[str, Any], summary_path: Path, drift_source: Any | None, feature_names: list[str] | None, sample_rows_override: int | None=None, sample_n_override: int | None=None) -> tuple[Path | None, bool | None]:
    if not drift_settings['enabled']:
        return (None, None)
    train_task_id = _normalize_str(meta.get('train_task_id'))
    (train_profile, train_profile_path) = _load_train_profile(cfg, bundle, model_bundle_path, train_task_id=train_task_id, clearml_enabled=clearml_enabled)
    if train_profile is None:
        raise ValueError('train_profile.json not found; run train_model with monitor.drift.enabled=true.')
    if drift_source is None:
        try:
            import pandas as pd
            drift_source = pd.DataFrame(columns=list(feature_names) if feature_names else [])
        except _RECOVERABLE_ERRORS:
            drift_source = _ensure_drift_frame([], feature_names)
    (drift_sample, sample_info) = sample_frame(drift_source, sample_n=sample_n_override if sample_n_override is not None else drift_settings['sample_n'], seed=drift_settings['sample_seed'])
    if sample_rows_override is not None:
        sample_info['rows'] = sample_rows_override
    drift_report = build_drift_report(train_profile, drift_sample, psi_warn_threshold=drift_settings['psi_warn_threshold'], psi_fail_threshold=drift_settings['psi_fail_threshold'], strict=validation_mode == 'strict', metrics=drift_settings['metrics'], train_profile_path=str(train_profile_path) if train_profile_path else None)
    train_settings = train_profile.get('settings') or {}
    max_bins = _to_int_or_none(train_settings.get('max_bins')) or 10
    max_categories = _to_int_or_none(train_settings.get('max_categories')) or 10
    feature_types = train_profile.get('feature_types') or {}
    infer_profile = build_train_profile(drift_sample, feature_columns=list(getattr(drift_source, 'columns', [])), numeric_features=feature_types.get('numeric'), categorical_features=feature_types.get('categorical'), max_bins=max_bins, quantiles=train_settings.get('quantiles'), max_categories=max_categories)
    drift_report['infer_profile'] = annotate_profile(infer_profile, role='infer', sample_info=dict(sample_info), metrics=drift_settings['metrics'])
    drift_report['sampling'] = dict(sample_info)
    drift_report['train_profile_summary'] = {'rows': train_profile.get('rows'), 'sampling': train_profile.get('sampling'), 'settings': train_profile.get('settings')}
    drift_summary = drift_report.get('summary', {})
    (warn_count, fail_count) = (int(drift_summary.get('warn_count', 0) or 0), int(drift_summary.get('fail_count', 0) or 0))
    drift_alert = bool(warn_count > 0)
    _emit_drift_alert(cfg, ctx, drift_summary, drift_settings, warn_count=warn_count, fail_count=fail_count, sample_rows=sample_info.get('rows'))
    drift_report_path = ctx.output_dir / 'drift_report.json'
    drift_report_path.write_text(json.dumps(drift_report, ensure_ascii=False, indent=2), encoding='utf-8')
    drift_report_md_path = ctx.output_dir / 'drift_report.md'
    drift_report_md_path.write_text(render_drift_markdown(drift_report), encoding='utf-8')
    append_drift_summary(summary_path, drift_report, drift_alert=drift_alert)
    if clearml_enabled:
        upload_artifact(ctx, summary_path.name, summary_path)
        upload_artifact(ctx, drift_report_path.name, drift_report_path)
        upload_artifact(ctx, drift_report_md_path.name, drift_report_md_path)
    if validation_mode == 'strict' and fail_count > 0:
        raise ValueError('Drift PSI exceeded fail threshold; see drift_report.json for details.')
    return (drift_report_path, drift_alert)
def _predict_nonchunked_classification(*, ctx: Any, cfg: Any, predictor: Any, model: Any, bundle: dict[str, Any], n_classes: int | None, transformed: Any, mode: str, output_format: str, clearml_enabled: bool, debug_output_sample: Any | None) -> tuple[Path, Any | None]:
    if not hasattr(predictor, 'predict_proba'):
        raise ValueError('classification infer requires predict_proba on the model.')
    proba = predictor.predict_proba(transformed)
    threshold_used = _resolve_threshold_used(bundle, n_classes=n_classes)
    if threshold_used is not None:
        positive_proba = extract_positive_class_proba(proba)
        preds = (positive_proba >= threshold_used).astype(int)
    else:
        preds = predictor.predict(transformed)
    class_labels = _resolve_class_labels(bundle, model)
    label_encoder = bundle.get('label_encoder')
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        pred_labels = label_encoder.inverse_transform(preds)
    else:
        pred_labels = preds
        if class_labels:
            try:
                pred_labels = [class_labels[int(idx)] for idx in preds]
            except _RECOVERABLE_ERRORS:
                pred_labels = preds
    if hasattr(pred_labels, 'tolist'):
        pred_labels = pred_labels.tolist()
    if not isinstance(pred_labels, list):
        pred_labels = list(pred_labels)
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('numpy is required for classification probabilities.') from exc
    proba_arr = np.asarray(proba)
    if proba_arr.ndim == 1:
        proba_arr = np.stack([1.0 - proba_arr, proba_arr], axis=1)
    if class_labels is None or len(class_labels) != int(proba_arr.shape[1]):
        class_labels = [str(i) for i in range(int(proba_arr.shape[1]))]
    is_multiclass = n_classes is not None and n_classes > 2
    proba_prefix = 'proba_' if is_multiclass else 'pred_proba_'
    top_k = _resolve_top_k(cfg, n_classes=n_classes)
    top_indices = None
    if top_k is not None:
        order = np.argsort(proba_arr, axis=1)[:, ::-1]
        top_indices = order[:, :top_k]
    if mode == 'single':
        predictions_path = ctx.output_dir / 'prediction.json'
        payload: dict[str, Any] = {'predicted_label': None, 'pred_label': None}
        if pred_labels:
            label_value = _sanitize_json_value(pred_labels[0])
            payload['predicted_label'] = label_value
            payload['pred_label'] = label_value
            payload['prediction'] = label_value
        if threshold_used is not None:
            payload['threshold_used'] = _sanitize_json_value(threshold_used)
        if hasattr(proba_arr, '__len__') and len(proba_arr) > 0:
            payload['predicted_proba'] = _build_proba_payload(proba_arr[0], class_labels)
            if class_labels and is_multiclass:
                safe_labels = _build_proba_column_labels(class_labels)
                for (safe, value) in zip(safe_labels, proba_arr[0]):
                    payload[f'proba_{safe}'] = _sanitize_json_value(value)
        if top_indices is not None:
            top_payload = []
            for idx in top_indices[0]:
                label = class_labels[idx] if class_labels else str(idx)
                top_payload.append({'label': _sanitize_json_value(label), 'proba': _sanitize_json_value(proba_arr[0][idx])})
            payload['top_k'] = top_payload
        predictions_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        if debug_output_sample is None:
            debug_output_sample = payload
        if clearml_enabled:
            flat = _flatten_prediction_payload(payload)
            pred_value = flat.get('prediction')
            if isinstance(pred_value, numbers.Real):
                log_scalar(ctx.task, 'infer', 'prediction', pred_value, step=0)
                logger = ctx.task.get_logger()
                reporter = getattr(logger, 'report_single_value', None)
                if callable(reporter):
                    reporter('infer/prediction', float(pred_value))
    else:
        if proba is None:
            raise ValueError('classification infer requires predicted probabilities.')
        if not class_labels:
            class_labels = [str(i) for i in range(int(proba_arr.shape[1]))]
        if output_format == 'parquet':
            predictions_path = ctx.output_dir / 'predictions.parquet'
            data: dict[str, list[Any]] = {'pred_label': [_sanitize_json_value(label) for label in pred_labels]}
            if threshold_used is not None:
                data['threshold_used'] = [_sanitize_json_value(threshold_used)] * len(pred_labels)
            proba_columns = _format_proba_columns(class_labels, prefix=proba_prefix)
            for (idx, col_name) in enumerate(proba_columns):
                data[col_name] = [_sanitize_json_value(v) for v in proba_arr[:, idx]]
            if top_indices is not None:
                for rank in range(1, top_k + 1):
                    labels: list[Any] = []
                    probs: list[Any] = []
                    for (row_idx, col_idx) in enumerate(top_indices[:, rank - 1]):
                        label = class_labels[col_idx] if class_labels else str(col_idx)
                        labels.append(_sanitize_json_value(label))
                        probs.append(_sanitize_json_value(proba_arr[row_idx][col_idx]))
                    data[f'top{rank}_label'] = labels
                    data[f'top{rank}_proba'] = probs
            try:
                import pandas as pd
            except _RECOVERABLE_ERRORS as exc:
                raise RuntimeError('pandas is required for infer.') from exc
            pd.DataFrame(data).to_parquet(predictions_path, index=False)
        else:
            predictions_path = ctx.output_dir / 'predictions.csv'
            columns = ['pred_label']
            if threshold_used is not None:
                columns.append('threshold_used')
            columns.extend(_format_proba_columns(class_labels, prefix=proba_prefix))
            if top_k is not None:
                for rank in range(1, top_k + 1):
                    columns.append(f'top{rank}_label')
                    columns.append(f'top{rank}_proba')
            with predictions_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                for (idx, label) in enumerate(pred_labels):
                    row = [_sanitize_json_value(label)]
                    if threshold_used is not None:
                        row.append(_sanitize_json_value(threshold_used))
                    row.extend([_sanitize_json_value(v) for v in proba_arr[idx]])
                    if top_indices is not None:
                        for rank_idx in top_indices[idx]:
                            top_label = class_labels[rank_idx] if class_labels else str(rank_idx)
                            row.append(_sanitize_json_value(top_label))
                            row.append(_sanitize_json_value(proba_arr[idx][rank_idx]))
                    writer.writerow(row)
        if debug_output_sample is None and pred_labels:
            sample_rows: list[dict[str, Any]] = []
            max_rows = min(5, len(pred_labels))
            proba_columns = _format_proba_columns(class_labels, prefix=proba_prefix)
            for idx in range(max_rows):
                row: dict[str, Any] = {'pred_label': _sanitize_json_value(pred_labels[idx])}
                if threshold_used is not None:
                    row['threshold_used'] = _sanitize_json_value(threshold_used)
                for (col_idx, col_name) in enumerate(proba_columns):
                    if col_idx >= proba_arr.shape[1]:
                        break
                    row[col_name] = _sanitize_json_value(proba_arr[idx][col_idx])
                if top_indices is not None:
                    for (rank, class_idx) in enumerate(top_indices[idx], start=1):
                        label = class_labels[class_idx] if class_labels else str(class_idx)
                        row[f'top{rank}_label'] = _sanitize_json_value(label)
                        row[f'top{rank}_proba'] = _sanitize_json_value(proba_arr[idx][class_idx])
                sample_rows.append(row)
            debug_output_sample = sample_rows
    return (predictions_path, debug_output_sample)
def _predict_nonchunked_regression(*, ctx: Any, model: Any, transformed: Any, mode: str, output_format: str, clearml_enabled: bool, uncertainty_enabled: bool, uncertainty_q: float | None, viz_enabled: bool, debug_output_sample: Any | None) -> tuple[Path, Path | None, Any | None]:
    preds = model.predict(transformed)
    rows = _preds_to_rows(preds)
    lower = None
    upper = None
    if uncertainty_enabled and uncertainty_q is not None:
        try:
            (lower, upper) = apply_split_conformal_interval(preds, float(uncertainty_q))
        except ValueError as exc:
            raise ValueError('uncertainty intervals require 1D regression predictions.') from exc
    if mode == 'single':
        predictions_path = ctx.output_dir / 'prediction.json'
        if rows:
            row = rows[0]
            if len(row) == 1:
                payload = {'prediction': _sanitize_json_value(row[0])}
                if lower is not None and upper is not None:
                    payload['pred_lower'] = _sanitize_json_value(lower[0])
                    payload['pred_upper'] = _sanitize_json_value(upper[0])
            else:
                payload = {'prediction': [_sanitize_json_value(v) for v in row]}
        else:
            payload = {'prediction': None}
        predictions_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        if debug_output_sample is None:
            debug_output_sample = payload
        if clearml_enabled:
            flat = _flatten_prediction_payload(payload)
            pred_value = flat.get('prediction')
            if isinstance(pred_value, numbers.Real):
                log_scalar(ctx.task, 'infer', 'prediction', pred_value, step=0)
                logger = ctx.task.get_logger()
                reporter = getattr(logger, 'report_single_value', None)
                if callable(reporter):
                    reporter('infer/prediction', float(pred_value))
    else:
        n_cols = len(rows[0]) if rows else 1
        if output_format == 'parquet':
            predictions_path = ctx.output_dir / 'predictions.parquet'
            data: dict[str, list[Any]] = {}
            if n_cols == 1:
                data['prediction'] = [_sanitize_json_value(row[0]) for row in rows] if rows else [None]
                if lower is not None and upper is not None:
                    data['pred_lower'] = [_sanitize_json_value(v) for v in lower]
                    data['pred_upper'] = [_sanitize_json_value(v) for v in upper]
            else:
                for idx in range(n_cols):
                    data[f'prediction_{idx}'] = [_sanitize_json_value(row[idx]) if idx < len(row) else None for row in rows]
            try:
                import pandas as pd
            except _RECOVERABLE_ERRORS as exc:
                raise RuntimeError('pandas is required for infer.') from exc
            pd.DataFrame(data).to_parquet(predictions_path, index=False)
        else:
            predictions_path = ctx.output_dir / 'predictions.csv'
            with predictions_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                if n_cols == 1:
                    header = ['prediction']
                    if lower is not None and upper is not None:
                        header.extend(['pred_lower', 'pred_upper'])
                else:
                    header = [f'prediction_{idx}' for idx in range(n_cols)]
                writer.writerow(header)
                for (idx, row) in enumerate(rows):
                    values = list(row)
                    if len(values) < n_cols:
                        values.extend([None] * (n_cols - len(values)))
                    row_values = [_sanitize_json_value(v) for v in values]
                    if lower is not None and upper is not None and (n_cols == 1):
                        row_values.append(_sanitize_json_value(lower[idx]))
                        row_values.append(_sanitize_json_value(upper[idx]))
                    writer.writerow(row_values)
        if debug_output_sample is None:
            sample_rows: list[dict[str, Any]] = []
            max_rows = min(5, len(rows))
            for idx in range(max_rows):
                row_values = list(rows[idx]) if rows else []
                if n_cols == 1:
                    payload = {'prediction': _sanitize_json_value(row_values[0]) if row_values else None}
                    if lower is not None and upper is not None and (idx < len(lower)):
                        payload['pred_lower'] = _sanitize_json_value(lower[idx])
                        payload['pred_upper'] = _sanitize_json_value(upper[idx])
                    sample_rows.append(payload)
                else:
                    payload = {}
                    for col_idx in range(n_cols):
                        key = f'prediction_{col_idx}'
                        value = row_values[col_idx] if col_idx < len(row_values) else None
                        payload[key] = _sanitize_json_value(value)
                    sample_rows.append(payload)
            if sample_rows:
                debug_output_sample = sample_rows
    interval_plot_path: Path | None = None
    if lower is not None and upper is not None and viz_enabled:
        try:
            widths = (upper - lower).tolist()
        except _RECOVERABLE_ERRORS:
            widths = [float(upper[idx] - lower[idx]) for idx in range(len(lower))]
        interval_plot_path = plot_interval_width_histogram(widths, ctx.output_dir / 'interval_widths.png', title='Prediction Interval Widths')
    return (predictions_path, interval_plot_path, debug_output_sample)
ChunkPredictState = make_dataclass('ChunkPredictState', [('input_preview_path', Path | None), ('debug_input_sample', Any | None), ('drift_sample', Any), ('class_labels', list[Any] | None), ('interval_widths_sample', list[float]), ('debug_output_sample', Any | None), ('parquet_writer', Any | None), ('header_written', bool)], frozen=True)
def _process_chunk_for_prediction(*, chunk_df: Any, mode: str, output_dir: Path, pipeline: Any, feature_names: list[str] | None, drift_enabled: bool, drift_sample_n: int | None, task_type: str, predictor: Any, model: Any, threshold_used: float | None, label_encoder: Any, top_k: int | None, proba_prefix: str, uncertainty_enabled: bool, uncertainty_q: float | None, rng: Any | None, output_format: str, predictions_path: Path, write_mode: str, state: ChunkPredictState) -> ChunkPredictState:
    input_preview_path = state.input_preview_path
    debug_input_sample = state.debug_input_sample
    drift_sample = state.drift_sample
    class_labels = state.class_labels
    interval_widths_sample = list(state.interval_widths_sample)
    debug_output_sample = state.debug_output_sample
    parquet_writer = state.parquet_writer
    header_written = state.header_written
    if input_preview_path is None:
        input_preview_path = _write_input_preview(chunk_df, mode, output_dir)
        if debug_input_sample is None:
            debug_input_sample = chunk_df.head(5).copy()
    transformed = pipeline.transform(chunk_df)
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    transformed = _maybe_attach_feature_names(transformed, feature_names)
    if drift_enabled:
        drift_chunk = _ensure_drift_frame(transformed, feature_names)
        drift_sample = _update_drift_sample(drift_sample, drift_chunk, sample_n=drift_sample_n, rng=rng)
    if task_type == 'classification':
        (pred_df, class_labels) = _build_classification_predictions_frame(transformed, predictor=predictor, model=model, threshold_used=threshold_used, class_labels=class_labels, label_encoder=label_encoder, top_k=top_k, proba_prefix=proba_prefix)
    else:
        (pred_df, interval_widths) = _build_regression_predictions_frame(transformed, model=model, uncertainty_enabled=uncertainty_enabled, uncertainty_q=uncertainty_q)
        if interval_widths:
            interval_widths_sample = _update_interval_width_sample(interval_widths_sample, interval_widths, rng=rng)
    if debug_output_sample is None:
        debug_output_sample = pred_df.head(5).copy()
    if output_format == 'parquet':
        parquet_writer = _write_predictions_parquet_chunk(predictions_path, pred_df, parquet_writer)
    else:
        header_written = _write_predictions_csv_chunk(predictions_path, pred_df, write_mode=write_mode, header_written=header_written)
    return ChunkPredictState(input_preview_path=input_preview_path, debug_input_sample=debug_input_sample, drift_sample=drift_sample, class_labels=class_labels, interval_widths_sample=interval_widths_sample, debug_output_sample=debug_output_sample, parquet_writer=parquet_writer, header_written=header_written)
def _log_infer_artifacts(ctx: Any, *, clearml_enabled: bool, interval_plot_path: Path | None, debug_input_sample: Any | None, debug_output_sample: Any | None, input_preview_path: Path, predictions_path: Path, table_settings: dict[str, int]) -> None:
    if not clearml_enabled:
        return
    upload_artifact(ctx, predictions_path.name, predictions_path)
    upload_artifact(ctx, input_preview_path.name, input_preview_path)
    if interval_plot_path is not None:
        log_plotly(ctx.task, 'infer', 'interval_widths', interval_plot_path, step=0)
    (table_input, table_output) = _resolve_table_samples(debug_input_sample, debug_output_sample, input_preview_path=input_preview_path, predictions_path=predictions_path)
    report_input_output_table(ctx.task, 'infer', 'input_output_table', table_input, table_output, max_rows=table_settings['max_rows'], max_input_columns=table_settings['max_input_columns'], max_output_columns=table_settings['max_output_columns'], output_path=ctx.output_dir / 'input_output_table.png', step=0)
    _log_debug_samples(ctx, input_sample=debug_input_sample, output_sample=debug_output_sample, input_preview_path=input_preview_path, predictions_path=predictions_path)
def _build_infer_manifest_payloads(*, mode: str, meta: dict[str, Any], model_bundle_path: Path, task_type: str, calibration_info: dict[str, Any] | None, calibrated_model: Any, validation_mode: str, validation: dict[str, Any], predictions_path: Path, input_preview_path: Path, errors_path: Path | None, drift_report_path: Path | None, drift_alert: bool | None, input_path: str | None, processed_dataset_id: str | None, chunked: dict[str, Any] | None=None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference_payload = build_model_reference_payload(meta, model_bundle_path=model_bundle_path)
    model_id = reference_payload.get('model_id')
    train_task_id = reference_payload.get('train_task_id')
    out = {'predictions_path': str(predictions_path), 'input_preview_path': str(input_preview_path), 'mode': mode, 'model_id': model_id, 'train_task_id': train_task_id, 'infer_model_id': reference_payload.get('infer_model_id'), 'infer_train_task_id': reference_payload.get('infer_train_task_id'), 'registry_model_id': reference_payload.get('registry_model_id'), 'reference_kind': reference_payload.get('reference_kind'), 'schema_validation': {'mode': validation_mode, 'ok': bool(validation.get('ok')), 'warnings_count': int(validation.get('warnings_count') or 0), 'errors_count': int(validation.get('errors_count') or 0)}}
    if task_type == 'classification':
        if calibration_info is None and calibrated_model is not None:
            calibration_info = {'enabled': True}
        if calibration_info is not None:
            out['calibration'] = {**dict(calibration_info), 'calibrated_proba': calibrated_model is not None}
    if errors_path is not None:
        out['errors_path'] = str(errors_path)
    if drift_report_path is not None:
        out['drift_report_path'] = str(drift_report_path)
    if drift_alert is not None:
        out['drift_alert'] = bool(drift_alert)
    if chunked is not None:
        out['chunked'] = dict(chunked)
    inputs = {'model_id': model_id, 'train_task_id': train_task_id, 'infer_model_id': reference_payload.get('infer_model_id'), 'infer_train_task_id': reference_payload.get('infer_train_task_id'), 'mode': mode, 'input_path': input_path, 'processed_dataset_id': processed_dataset_id}
    outputs = {'predictions_path': str(predictions_path), 'input_preview_path': str(input_preview_path)}
    if errors_path is not None:
        outputs['errors_path'] = str(errors_path)
    if drift_report_path is not None:
        outputs['drift_report_path'] = str(drift_report_path)
    if drift_alert is not None:
        outputs['drift_alert'] = bool(drift_alert)
    return (out, inputs, outputs)
def _emit_infer_outputs(cfg: Any, ctx: Any, *, clearml_enabled: bool, drift_alert: bool | None, mode: str, meta: dict[str, Any], model_bundle_path: Path, task_type: str, calibration_info: dict[str, Any] | None, calibrated_model: Any, validation_mode: str, validation: dict[str, Any], predictions_path: Path, input_preview_path: Path, errors_path: Path | None, drift_report_path: Path | None, input_path: str | None, processed_dataset_id: str | None, split_hash: str, recipe_hash: str, chunked: dict[str, Any] | None=None) -> None:
    if clearml_enabled and drift_alert is not None:
        update_task_properties(ctx, {'drift_alert': bool(drift_alert)})
    (out, inputs, outputs) = _build_infer_manifest_payloads(mode=mode, meta=meta, model_bundle_path=model_bundle_path, task_type=task_type, calibration_info=calibration_info, calibrated_model=calibrated_model, validation_mode=validation_mode, validation=validation, predictions_path=predictions_path, input_preview_path=input_preview_path, errors_path=errors_path, drift_report_path=drift_report_path, drift_alert=drift_alert, input_path=input_path, processed_dataset_id=processed_dataset_id, chunked=chunked)
    inputs.update({'split_hash': split_hash, 'recipe_hash': recipe_hash})
    emit_outputs_and_manifest(ctx, cfg, process='infer', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, clearml_enabled=clearml_enabled)
def _run_optimize_mode_impl(cfg: Any, ctx: Any, *, clearml_enabled: bool, optimize_settings: dict[str, Any] | None, preprocess_bundle: dict[str, Any], validation_mode: str, uncertainty_info: dict[str, Any], infer_cfg: Any, meta: dict[str, Any], model_bundle_path: Path, model_abbr: str | None, table_settings: dict[str, int], processed_dataset_id: str | None, split_hash: str, recipe_hash: str) -> None:
    mode = 'optimize'
    if not clearml_enabled:
        raise ValueError('infer.mode=optimize requires ClearML; set run.clearml.enabled=true.')
    if not optimize_settings:
        raise ValueError('infer.optimize settings are required for optimize mode.')
    search_space = optimize_settings.get('search_space') or []
    if not search_space:
        raise ValueError('infer.optimize.search_space is required for optimize mode.')
    (base_df, base_input_path) = _prepare_inputs(cfg, preprocess_bundle, 'single')
    (base_df, validation, summary_path, errors_json_path, _) = _validate_inputs_with_summary(ctx=ctx, clearml_enabled=clearml_enabled, mode=mode, validation_mode=validation_mode, preprocess_bundle=preprocess_bundle, inputs_df=base_df, uncertainty_info=uncertainty_info)
    input_preview_path = _write_input_preview(base_df, 'single', ctx.output_dir)
    if clearml_enabled:
        upload_artifact(ctx, input_preview_path.name, input_preview_path)
    base_records = _frame_to_records(base_df)
    base_record = base_records[0] if base_records else {}
    (feature_columns, _, _, _) = _resolve_preprocess_columns(preprocess_bundle)
    if feature_columns:
        search_names = [entry.get('name') for entry in search_space if entry.get('name')]
        missing = [name for name in search_names if name not in feature_columns]
        if missing:
            log_debug_text(ctx.task, 'infer', 'optimize_search_space_warning', f'search_space columns not in feature_columns: {missing}', step=0)
    child_context = _resolve_child_task_context(cfg, ctx, infer_cfg=infer_cfg, meta=meta, context_name='optimize')
    queue_name = str(child_context['queue_name'])
    child_project_name = child_context['child_project_name']
    source_task_id = str(child_context['source_task_id'])
    child_model_id = child_context['child_model_id']
    child_train_task_id = child_context['child_train_task_id']
    try:
        import optuna
        from optuna.trial import TrialState
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('Optuna is required for infer.mode=optimize. Install with: uv sync --extra optuna (or pip install optuna)') from exc
    sampler = _build_optuna_sampler(optimize_settings.get('sampler_name'), optimize_settings.get('sampler_seed'))
    study = optuna.create_study(direction=optimize_settings.get('direction'), sampler=sampler)
    completed = {'completed', 'closed'}
    trial_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    objective_keys = optimize_settings.get('objective_keys') or ['prediction']
    for _ in range(int(optimize_settings.get('n_trials') or 0)):
        trial = study.ask()
        params = _suggest_optuna_params(trial, search_space)
        trial_number = int(trial.number) + 1
        record = dict(base_record)
        for (key, value) in params.items():
            record[str(key)] = _sanitize_json_value(value)
        payload = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
        child_name = f'infer__single__trial={trial_number}'
        if model_abbr:
            child_name = f'infer__single__model={model_abbr}__trial={trial_number}'
        overrides: dict[str, Any] = {'task': 'infer', 'infer.mode': 'single', 'infer.input_json': payload, 'infer.dry_run': False, 'infer.validation.mode': validation_mode, 'infer.optimize.child_task': True, 'infer.optimize.search_space': '', 'infer.optimize.sampler': '', 'infer.optimize.n_trials': '', 'infer.optimize.direction': '', 'infer.optimize.objective.key': '', 'infer.optimize.top_k': '', 'run.clearml.execution': 'logging', 'run.clearml.queue_name': queue_name, 'run.clearml.enabled': True, 'run.clearml.task_name': child_name, 'run.clearml.env.bootstrap': 'auto'}
        if child_project_name:
            overrides['task.project_name'] = child_project_name
        if child_train_task_id:
            overrides['infer.train_task_id'] = child_train_task_id
        else:
            overrides['infer.model_id'] = child_model_id
        child_task_id = _clone_and_enqueue_infer_child(source_task_id=source_task_id, child_name=child_name, queue_name=queue_name, overrides=overrides, tags=[f'parent:{source_task_id}', 'trial:optimize'], reset_args_first=True)
        statuses = _wait_for_child_tasks([child_task_id], timeout_sec=optimize_settings.get('wait_timeout_sec'), poll_interval_sec=optimize_settings.get('poll_interval_sec'))
        status = statuses.get(child_task_id, 'unknown')
        child_url = resolve_clearml_task_url(cfg, child_task_id)
        child_rows.append({'trial_number': trial_number, 'task_id': child_task_id, 'status': status, 'url': child_url})
        payload_out = None
        error = None
        if status in completed:
            (payload_out, error) = _load_child_prediction_payload(cfg, child_task_id)
        else:
            error = f'child_status={status}'
        output_flat = _flatten_prediction_payload(payload_out or {})
        objective_value = _resolve_objective_value(output_flat, objective_keys)
        if objective_value is None:
            study.tell(trial, state=TrialState.FAIL)
        else:
            study.tell(trial, objective_value)
        row: dict[str, Any] = {'trial_number': trial_number, 'objective': objective_value, 'state': 'complete' if objective_value is not None else 'fail', 'child_task_id': child_task_id, 'child_status': status, 'child_url': child_url}
        if error:
            row['error'] = error
        elif objective_value is None:
            row['error'] = 'objective_value_missing'
        for (key, value) in params.items():
            row[f'in.{key}'] = _sanitize_json_value(value)
        for (key, value) in output_flat.items():
            row[f'out.{key}'] = value
        trial_rows.append(row)
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    trials_path = ctx.output_dir / 'optimize_trials.csv'
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
    if clearml_enabled:
        upload_artifact(ctx, trials_path.name, trials_path)
    completed_trials = [row for row in trial_rows if isinstance(row.get('objective'), numbers.Real)]
    direction = optimize_settings.get('direction') or 'maximize'
    reverse = direction == 'maximize'
    ranked = sorted(completed_trials, key=lambda r: float(r.get('objective', 0.0)), reverse=reverse)
    top_k = int(optimize_settings.get('top_k') or 0)
    top_trials = ranked[:top_k] if top_k > 0 else ranked
    top_inputs: list[dict[str, Any]] = []
    top_outputs: list[dict[str, Any]] = []
    for row in top_trials:
        input_row = {key[3:]: row[key] for key in row if isinstance(key, str) and key.startswith('in.')}
        output_row = {key[4:]: row[key] for key in row if isinstance(key, str) and key.startswith('out.')}
        top_inputs.append(input_row)
        top_outputs.append(output_row)
    best_value = None
    best_params = None
    if ranked:
        best_value = ranked[0].get('objective')
        best_params = {key[3:]: ranked[0][key] for key in ranked[0] if isinstance(key, str) and key.startswith('in.')}
    if clearml_enabled:
        history_fig = build_optimization_history(study, log_scale=bool(optimize_settings.get('history_log_scale')), output_path=ctx.output_dir / 'optuna_history.png')
        log_plotly(ctx.task, 'infer', 'optuna_history', history_fig, step=0)
        parallel_fig = build_parallel_coordinate(study, output_path=ctx.output_dir / 'optuna_parallel.png')
        log_plotly(ctx.task, 'infer', 'optuna_parallel', parallel_fig, step=0)
        importance_fig = build_param_importance(study, output_path=ctx.output_dir / 'optuna_importance.png')
        log_plotly(ctx.task, 'infer', 'optuna_importance', importance_fig, step=0)
        contour_params = optimize_settings.get('contour_params')
        if not contour_params:
            contour_params = [entry.get('name') for entry in search_space if entry.get('name')][:2]
        if contour_params and len(contour_params) >= 2:
            contour_fig = build_contour(study, params=contour_params, output_path=ctx.output_dir / 'optuna_contour.png')
            log_plotly(ctx.task, 'infer', 'optuna_contour', contour_fig, step=0)
        report_input_output_table(ctx.task, 'infer', 'input_output_table', top_inputs, top_outputs, max_rows=table_settings['max_rows'], max_input_columns=table_settings['max_input_columns'], max_output_columns=table_settings['max_output_columns'], output_path=ctx.output_dir / 'optimize_input_output_table.png', step=0)
        log_debug_table(ctx.task, 'infer', 'optimize_trials', trial_rows, step=0)
        log_debug_table(ctx.task, 'infer', 'optimize_child_tasks', child_rows, step=0)
        failed_children = [row.get('task_id') for row in child_rows if row.get('status') not in completed]
        log_debug_text(ctx.task, 'infer', 'optimize_child_summary', json.dumps({'total': len(child_rows), 'completed': len([row for row in child_rows if row.get('status') in completed]), 'failed': len(failed_children), 'failed_task_ids': failed_children}, ensure_ascii=False), step=0)
    (out, _, outputs) = _build_infer_manifest_payloads(mode=mode, meta=meta, model_bundle_path=model_bundle_path, task_type='regression', calibration_info=None, calibrated_model=None, validation_mode=validation_mode, validation=validation, predictions_path=trials_path, input_preview_path=input_preview_path, errors_path=errors_json_path, drift_report_path=None, drift_alert=None, input_path=base_input_path, processed_dataset_id=processed_dataset_id, chunked=None)
    out.pop('predictions_path', None)
    outputs.pop('predictions_path', None)
    out['optimize'] = {'n_trials': optimize_settings.get('n_trials'), 'direction': optimize_settings.get('direction'), 'objective_keys': objective_keys, 'completed': len(completed_trials), 'failed': len(trial_rows) - len(completed_trials), 'best_value': best_value, 'best_params': best_params, 'top_k': top_k}
    out['child_tasks'] = child_rows
    out['optimize_trials_path'] = str(trials_path)
    outputs['optimize_trials_path'] = str(trials_path)
    outputs['summary_path'] = str(summary_path)
    optimize_reference = build_model_reference_payload(meta, model_bundle_path=model_bundle_path)
    emit_outputs_and_manifest(ctx, cfg, process='infer', out=out, inputs={'model_id': optimize_reference.get('model_id'), 'train_task_id': optimize_reference.get('train_task_id'), 'infer_model_id': optimize_reference.get('infer_model_id'), 'infer_train_task_id': optimize_reference.get('infer_train_task_id'), 'mode': mode, 'input_path': base_input_path, 'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash}, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, clearml_enabled=clearml_enabled)
def _run_batch_children_mode_impl(cfg: Any, ctx: Any, *, clearml_enabled: bool, preprocess_bundle: dict[str, Any], validation_mode: str, uncertainty_info: dict[str, Any], mode: str, batch_inputs_df: Any, batch_inputs_path: str | None, batch_children_settings: dict[str, Any], infer_cfg: Any, meta: dict[str, Any], model_bundle_path: Path, model_abbr: str | None, table_settings: dict[str, int], processed_dataset_id: str | None, split_hash: str, recipe_hash: str) -> None:
    (validation_df, validation, summary_path, errors_json_path, _) = _validate_inputs_with_summary(ctx=ctx, clearml_enabled=clearml_enabled, mode=mode, validation_mode=validation_mode, preprocess_bundle=preprocess_bundle, inputs_df=batch_inputs_df.copy(), uncertainty_info=uncertainty_info)
    input_preview_path = _write_input_preview(validation_df, mode, ctx.output_dir)
    input_records = _frame_to_records(validation_df)
    condition_ids = list(range(1, len(input_records) + 1))
    child_task_ids: list[str] = []
    child_rows: list[dict[str, Any]] = []
    completed = {'completed', 'closed'}
    statuses: dict[str, str] = {}
    if input_records:
        child_context = _resolve_child_task_context(cfg, ctx, infer_cfg=infer_cfg, meta=meta, context_name='batch')
        queue_name = str(child_context['queue_name'])
        child_project_name = child_context['child_project_name']
        source_task_id = str(child_context['source_task_id'])
        child_model_id = child_context['child_model_id']
        child_train_task_id = child_context['child_train_task_id']
        for (idx, record) in enumerate(input_records, start=1):
            payload = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
            child_name = f'infer__single__case={idx}'
            if model_abbr:
                child_name = f'infer__single__model={model_abbr}__case={idx}'
            overrides: dict[str, Any] = {'task': 'infer', 'infer.mode': 'single', 'infer.input_json': payload, 'infer.dry_run': False, 'infer.validation.mode': validation_mode, 'infer.batch.child_task': True, 'run.clearml.execution': 'logging', 'run.clearml.queue_name': queue_name, 'run.clearml.enabled': True, 'run.clearml.task_name': child_name, 'run.clearml.env.bootstrap': 'auto'}
            if child_project_name:
                overrides['task.project_name'] = child_project_name
            if child_train_task_id:
                overrides['infer.train_task_id'] = child_train_task_id
            else:
                overrides['infer.model_id'] = child_model_id
            child_task_id = _clone_and_enqueue_infer_child(source_task_id=source_task_id, child_name=child_name, queue_name=queue_name, overrides=overrides)
            child_task_ids.append(child_task_id)
            child_rows.append({'condition_id': idx, 'task_id': child_task_id, 'status': 'queued', 'url': resolve_clearml_task_url(cfg, child_task_id)})
        statuses = _wait_for_child_tasks(child_task_ids, timeout_sec=batch_children_settings['wait_timeout_sec'], poll_interval_sec=batch_children_settings['poll_interval_sec'])
    output_rows: list[dict[str, Any]] = []
    for row in child_rows:
        task_id = row['task_id']
        status = statuses.get(task_id, 'unknown')
        row['status'] = status
        payload = None
        error = None
        if status in completed:
            (payload, error) = _load_child_prediction_payload(cfg, task_id)
        output_row = {'condition_id': row['condition_id'], 'child_task_id': task_id, 'child_status': status}
        if payload is not None:
            output_row.update(_flatten_prediction_payload(payload))
        if error:
            output_row['child_error'] = error
        output_rows.append(output_row)
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas is required for infer.') from exc
    input_df = pd.DataFrame(input_records)
    if condition_ids:
        input_df.insert(0, 'condition_id', condition_ids)
    output_df = pd.DataFrame(output_rows)
    predictions_path = ctx.output_dir / 'predictions.csv'
    output_df.to_csv(predictions_path, index=False)
    if clearml_enabled:
        upload_artifact(ctx, predictions_path.name, predictions_path)
        upload_artifact(ctx, input_preview_path.name, input_preview_path)
        table_rows = batch_children_settings['max_children'] or len(input_df)
        table_fig = build_input_output_table(input_df, output_df, max_rows=table_rows, max_input_columns=table_settings['max_input_columns'], max_output_columns=table_settings['max_output_columns'], title='Batch Inputs -> Predictions', output_path=ctx.output_dir / 'batch_input_output_table.png')
        log_plotly(ctx.task, 'infer', 'batch_input_output_table', table_fig, step=0)
        (dist_values, dist_labels) = _select_distribution_samples(output_df)
        if dist_values:
            dist_fig = build_prediction_histogram(dist_values, title='Prediction Distribution', output_path=ctx.output_dir / 'prediction_distribution.png')
            log_plotly(ctx.task, 'infer', 'prediction_distribution', dist_fig, step=0)
        elif dist_labels:
            label_fig = build_label_distribution(dist_labels, title='Prediction Labels', output_path=ctx.output_dir / 'prediction_labels.png')
            log_plotly(ctx.task, 'infer', 'prediction_labels', label_fig, step=0)
        child_table_fig = build_input_output_table(None, child_rows, max_rows=table_rows, max_input_columns=table_settings['max_input_columns'], max_output_columns=table_settings['max_output_columns'], title='Batch Child Tasks', output_path=ctx.output_dir / 'batch_child_tasks_table.png')
        if child_table_fig is not None:
            log_plotly(ctx.task, 'infer', 'batch_child_tasks_table', child_table_fig, step=0)
        log_debug_table(ctx.task, 'infer', 'batch_child_tasks', child_rows, step=0)
    (out, _, outputs) = _build_infer_manifest_payloads(mode=mode, meta=meta, model_bundle_path=model_bundle_path, task_type='regression', calibration_info=None, calibrated_model=None, validation_mode=validation_mode, validation=validation, predictions_path=predictions_path, input_preview_path=input_preview_path, errors_path=errors_json_path, drift_report_path=None, drift_alert=None, input_path=batch_inputs_path, processed_dataset_id=processed_dataset_id, chunked=None)
    out['batch_children'] = {'inputs_path': batch_inputs_path, 'inputs_count': len(input_records), 'child_tasks': len(child_task_ids), 'completed': sum((1 for status in statuses.values() if status in completed)), 'failed': sum((1 for status in statuses.values() if status == 'failed')), 'timeouts': sum((1 for status in statuses.values() if status == 'timeout')), 'max_children': batch_children_settings['max_children']}
    out['child_tasks'] = child_rows
    batch_reference = build_model_reference_payload(meta, model_bundle_path=model_bundle_path)
    emit_outputs_and_manifest(ctx, cfg, process='infer', out=out, inputs={'model_id': batch_reference.get('model_id'), 'train_task_id': batch_reference.get('train_task_id'), 'infer_model_id': batch_reference.get('infer_model_id'), 'infer_train_task_id': batch_reference.get('infer_train_task_id'), 'mode': mode, 'input_path': batch_inputs_path, 'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash}, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, clearml_enabled=clearml_enabled)
def _run_chunked_batch_mode_impl(cfg: Any, ctx: Any, *, mode: str, input_payload: Any, input_path_value: str | None, chunk_size: int | None, output_format: str, write_mode: str, max_rows: int | None, preprocess_bundle: dict[str, Any], uncertainty_info: dict[str, Any], drift_settings: dict[str, Any], task_type: str, n_classes: int | None, bundle: dict[str, Any], model: Any, predictor: Any, quality_target: str | None, quality_id_columns: Any, validation_mode: str, table_settings: dict[str, int], clearml_enabled: bool, meta: dict[str, Any], model_bundle_path: Path, calibration_info: dict[str, Any] | None, calibrated_model: Any, processed_dataset_id: str | None, split_hash: str, recipe_hash: str, debug_input_sample: Any | None, debug_output_sample: Any | None) -> bool:
    dataset_path = input_path_value
    chunked_input_path: Path | None = None
    if mode == 'batch' and chunk_size is not None and (input_payload is None) and dataset_path:
        candidate = Path(dataset_path).expanduser().resolve()
        if candidate.suffix.lower() != '.json':
            chunked_input_path = _select_tabular_file(candidate)
    if not (mode == 'batch' and chunk_size is not None and (input_payload is None) and (chunked_input_path is not None)):
        return False
    if output_format == 'parquet' and write_mode == 'append':
        raise ValueError('infer.batch.write_mode=append is not supported for parquet output.')
    predictions_path = ctx.output_dir / 'predictions.parquet' if output_format == 'parquet' else ctx.output_dir / 'predictions.csv'
    if write_mode == 'overwrite' and predictions_path.exists():
        predictions_path.unlink()
    pipeline = preprocess_bundle.get('pipeline')
    if pipeline is None:
        raise ValueError('preprocess_bundle.pipeline is missing.')
    feature_names = preprocess_bundle.get('feature_names')
    uncertainty_enabled = bool(uncertainty_info.get('enabled'))
    uncertainty_q = uncertainty_info.get('q') if uncertainty_enabled else None
    viz_enabled = bool(_cfg_value(cfg, 'viz.enabled', True))
    validation_acc = _init_validation_accumulator()
    errors_log_path: Path | None = None
    errors_log_handle = None
    drift_sample = None
    drift_sample_n = drift_settings['sample_n']
    if drift_settings['enabled'] and drift_sample_n is None:
        drift_sample_n = max_rows or chunk_size
    rng = None
    if drift_settings['enabled'] or (uncertainty_enabled and viz_enabled):
        try:
            import numpy as np
            rng = np.random.default_rng(drift_settings['sample_seed'])
        except _RECOVERABLE_ERRORS:
            rng = None
    threshold_used = None
    class_labels = None
    label_encoder = None
    proba_prefix = 'pred_proba_'
    top_k = None
    if task_type == 'classification':
        threshold_used = _resolve_threshold_used(bundle, n_classes=n_classes)
        class_labels = _resolve_class_labels(bundle, model)
        label_encoder = bundle.get('label_encoder')
        is_multiclass = n_classes is not None and n_classes > 2
        proba_prefix = 'proba_' if is_multiclass else 'pred_proba_'
        top_k = _resolve_top_k(cfg, n_classes=n_classes)
    input_preview_path = None
    total_rows = 0
    total_chunks = 0
    chunk_state = ChunkPredictState(input_preview_path=None, debug_input_sample=debug_input_sample, drift_sample=drift_sample, class_labels=class_labels, interval_widths_sample=[], debug_output_sample=debug_output_sample, parquet_writer=None, header_written=False)
    quality_checked = False
    def _log_chunk_issues(*, chunk_index: int, row_offset: int, rows: int, issues_payload: Mapping[str, Any]) -> None:
        nonlocal errors_log_handle, errors_log_path
        if errors_log_handle is None:
            errors_log_path = ctx.output_dir / 'errors.jsonl'
            errors_log_handle = errors_log_path.open('w', encoding='utf-8')
        payload = {'chunk_index': chunk_index, 'row_start': row_offset, 'row_end': row_offset + rows - 1, 'rows': rows, 'issues_count': _count_schema_issues(issues_payload), 'issues': issues_payload}
        errors_log_handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
    def _finalize_preview() -> Path:
        nonlocal input_preview_path
        if input_preview_path is not None:
            return input_preview_path
        try:
            import pandas as pd
            empty_df = pd.DataFrame()
        except _RECOVERABLE_ERRORS as exc:
            raise RuntimeError('pandas is required for infer.') from exc
        input_preview_path = _write_input_preview(empty_df, mode, ctx.output_dir)
        return input_preview_path
    def _run_quality_gate(chunk: Any) -> None:
        nonlocal quality_checked
        if quality_checked:
            return
        quality_result = run_data_quality_gate(cfg=cfg, ctx=ctx, df=chunk, target_column=quality_target, task_type=task_type, id_columns=quality_id_columns, output_dir=ctx.output_dir)
        raise_on_quality_fail(cfg=cfg, ctx=ctx, gate=quality_result['gate'], payload=quality_result['payload'], json_path=quality_result['paths']['json'])
        quality_checked = True
    def _predict_chunk(chunk_df: Any) -> None:
        nonlocal chunk_state
        chunk_state = _process_chunk_for_prediction(chunk_df=chunk_df, mode=mode, output_dir=ctx.output_dir, pipeline=pipeline, feature_names=feature_names, drift_enabled=bool(drift_settings['enabled']), drift_sample_n=drift_sample_n, task_type=task_type, predictor=predictor, model=model, threshold_used=threshold_used, label_encoder=label_encoder, top_k=top_k, proba_prefix=proba_prefix, uncertainty_enabled=uncertainty_enabled, uncertainty_q=uncertainty_q, rng=rng, output_format=output_format, predictions_path=predictions_path, write_mode=write_mode, state=chunk_state)
    predict_during_validation = validation_mode != 'strict'
    for (chunk_index, (chunk, row_offset)) in enumerate(_iter_tabular_chunks(chunked_input_path, chunk_size=chunk_size, max_rows=max_rows), start=1):
        total_rows += len(chunk)
        total_chunks += 1
        _run_quality_gate(chunk)
        (chunk_df, chunk_validation) = _validate_inputs(chunk, preprocess_bundle, validation_mode=validation_mode)
        issues = chunk_validation.get('issues') or {}
        if _count_schema_issues(issues) > 0:
            _log_chunk_issues(chunk_index=chunk_index, row_offset=row_offset, rows=len(chunk), issues_payload=issues)
        _update_validation_accumulator(validation_acc, issues, validation_mode=validation_mode)
        if predict_during_validation:
            _predict_chunk(chunk_df)
    if errors_log_handle is not None:
        errors_log_handle.close()
    validation = _finalize_validation_accumulator(validation_acc)
    summary_path = _write_infer_summary(ctx.output_dir, mode=mode, validation_mode=validation_mode, validation=validation, errors_path=errors_log_path, uncertainty=uncertainty_info)
    _upload_summary_artifacts(ctx, clearml_enabled=clearml_enabled, summary_path=summary_path, errors_json_path=errors_log_path)
    if validation_mode == 'strict' and (not validation.get('ok')):
        raise ValueError('Input schema validation failed; see errors.jsonl for details.')
    if validation_mode == 'strict':
        total_rows = 0
        total_chunks = 0
        for (chunk_index, (chunk, row_offset)) in enumerate(_iter_tabular_chunks(chunked_input_path, chunk_size=chunk_size, max_rows=max_rows), start=1):
            total_rows += len(chunk)
            total_chunks += 1
            (chunk_df, _) = _validate_inputs(chunk, preprocess_bundle, validation_mode=validation_mode)
            _predict_chunk(chunk_df)
    if chunk_state.parquet_writer is not None:
        chunk_state.parquet_writer.close()
    input_preview_path = chunk_state.input_preview_path
    debug_input_sample = chunk_state.debug_input_sample
    drift_sample = chunk_state.drift_sample
    interval_widths_sample = chunk_state.interval_widths_sample
    debug_output_sample = chunk_state.debug_output_sample
    input_preview_path = _finalize_preview()
    (drift_report_path, drift_alert) = _run_infer_drift_analysis(cfg, ctx, clearml_enabled=clearml_enabled, drift_settings=drift_settings, validation_mode=validation_mode, bundle=bundle, model_bundle_path=model_bundle_path, meta=meta, summary_path=summary_path, drift_source=drift_sample, feature_names=feature_names, sample_rows_override=total_rows, sample_n_override=drift_sample_n)
    interval_plot_path: Path | None = None
    if uncertainty_enabled and interval_widths_sample and viz_enabled:
        interval_plot_path = plot_interval_width_histogram(interval_widths_sample, ctx.output_dir / 'interval_widths.png', title='Prediction Interval Widths')
    _log_infer_artifacts(ctx, clearml_enabled=clearml_enabled, interval_plot_path=interval_plot_path, debug_input_sample=debug_input_sample, debug_output_sample=debug_output_sample, input_preview_path=input_preview_path, predictions_path=predictions_path, table_settings=table_settings)
    _emit_infer_outputs(cfg, ctx, clearml_enabled=clearml_enabled, drift_alert=drift_alert, mode=mode, meta=meta, model_bundle_path=model_bundle_path, task_type=task_type, calibration_info=calibration_info, calibrated_model=calibrated_model, validation_mode=validation_mode, validation=validation, predictions_path=predictions_path, input_preview_path=input_preview_path, errors_path=errors_log_path, drift_report_path=drift_report_path, input_path=str(chunked_input_path), processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, chunked={'chunk_size': chunk_size, 'rows': total_rows, 'chunks': total_chunks, 'errors_count': int(validation_acc.get('total_issues') or 0), 'output_format': output_format, 'write_mode': write_mode, 'max_rows': max_rows})
    return True
def _run_standard_infer_mode(*, cfg: Any, ctx: Any, mode: str, clearml_enabled: bool, preprocess_bundle: dict[str, Any], validation_mode: str, task_type: str, quality_target: str | None, quality_id_columns: Any, uncertainty_info: dict[str, Any], model_bundle_path: Path, meta: dict[str, Any], bundle: dict[str, Any], model: Any, predictor: Any, n_classes: int | None, table_settings: dict[str, Any], drift_settings: dict[str, Any], output_format: str, calibration_info: dict[str, Any], calibrated_model: Any | None, processed_dataset_id: str | None, split_hash: str | None, recipe_hash: str | None, debug_input_sample: Any | None, debug_output_sample: Any | None) -> None:
    (inputs_df, input_path) = _prepare_inputs(cfg, preprocess_bundle, mode)
    quality_result = run_data_quality_gate(cfg=cfg, ctx=ctx, df=inputs_df, target_column=quality_target, task_type=task_type, id_columns=quality_id_columns, output_dir=ctx.output_dir)
    raise_on_quality_fail(cfg=cfg, ctx=ctx, gate=quality_result['gate'], payload=quality_result['payload'], json_path=quality_result['paths']['json'])
    (inputs_df, validation, summary_path, errors_json_path, _) = _validate_inputs_with_summary(ctx=ctx, clearml_enabled=clearml_enabled, mode=mode, validation_mode=validation_mode, preprocess_bundle=preprocess_bundle, inputs_df=inputs_df, uncertainty_info=uncertainty_info)
    input_preview_path = _write_input_preview(inputs_df, mode, ctx.output_dir)
    if debug_input_sample is None:
        try:
            debug_input_sample = inputs_df.head(5).copy()
        except _RECOVERABLE_ERRORS:
            debug_input_sample = None
    pipeline = preprocess_bundle.get('pipeline')
    if pipeline is None:
        raise ValueError('preprocess_bundle.pipeline is missing.')
    transformed = pipeline.transform(inputs_df)
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    feature_names = preprocess_bundle.get('feature_names')
    if feature_names:
        try:
            import pandas as pd
            if hasattr(transformed, 'shape') and len(feature_names) == int(transformed.shape[1]):
                transformed = pd.DataFrame(transformed, columns=list(feature_names))
        except _RECOVERABLE_ERRORS:
            pass
    (drift_report_path, drift_alert) = _run_infer_drift_analysis(cfg, ctx, clearml_enabled=clearml_enabled, drift_settings=drift_settings, validation_mode=validation_mode, bundle=bundle, model_bundle_path=model_bundle_path, meta=meta, summary_path=summary_path, drift_source=_ensure_drift_frame(transformed, feature_names), feature_names=feature_names)
    uncertainty_enabled = bool(uncertainty_info.get('enabled'))
    uncertainty_q = uncertainty_info.get('q') if uncertainty_enabled else None
    viz_enabled = bool(_cfg_value(cfg, 'viz.enabled', True))
    if task_type == 'classification':
        (predictions_path, debug_output_sample) = _predict_nonchunked_classification(ctx=ctx, cfg=cfg, predictor=predictor, model=model, bundle=bundle, n_classes=n_classes, transformed=transformed, mode=mode, output_format=output_format, clearml_enabled=clearml_enabled, debug_output_sample=debug_output_sample)
        interval_plot_path = None
    else:
        (predictions_path, interval_plot_path, debug_output_sample) = _predict_nonchunked_regression(ctx=ctx, model=model, transformed=transformed, mode=mode, output_format=output_format, clearml_enabled=clearml_enabled, uncertainty_enabled=uncertainty_enabled, uncertainty_q=uncertainty_q, viz_enabled=viz_enabled, debug_output_sample=debug_output_sample)
    _log_infer_artifacts(ctx, clearml_enabled=clearml_enabled, interval_plot_path=interval_plot_path, debug_input_sample=debug_input_sample, debug_output_sample=debug_output_sample, input_preview_path=input_preview_path, predictions_path=predictions_path, table_settings=table_settings)
    _emit_infer_outputs(cfg, ctx, clearml_enabled=clearml_enabled, drift_alert=drift_alert, mode=mode, meta=meta, model_bundle_path=model_bundle_path, task_type=task_type, calibration_info=calibration_info, calibrated_model=calibrated_model, validation_mode=validation_mode, validation=validation, predictions_path=predictions_path, input_preview_path=input_preview_path, errors_path=errors_json_path, drift_report_path=drift_report_path, input_path=input_path, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, chunked=None)
def run(cfg: Any) -> None:
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='infer', tags=identity.tags, properties=identity.user_properties)
    clearml_enabled = is_clearml_enabled(cfg)
    settings = _resolve_infer_runtime_settings(cfg, clearml_enabled=clearml_enabled)
    mode = settings.mode
    validation_mode = settings.validation_mode
    drift_settings = resolve_drift_settings(cfg)
    debug_input_sample: Any | None = None
    debug_output_sample: Any | None = None
    if clearml_enabled:
        summary_lines = [f'mode: {mode}', f'input_source: {settings.input_source}', f'batch_execution: {settings.batch_execution}', f"model_id: {settings.model_id_value or 'n/a'}", f"train_task_id: {settings.train_task_id_value or 'n/a'}", f"search_space: {('set' if settings.has_search_space else 'none')}"]
        log_debug_text(ctx.task, 'infer', 'settings', '\n'.join(summary_lines), step=0)
    if _handle_infer_dry_run(ctx, cfg, clearml_enabled=clearml_enabled, infer_cfg=settings.infer_cfg, mode=mode, validation_mode=validation_mode, input_source=settings.input_source, input_path_value=settings.input_path_value, input_json_label=settings.input_json_label, include_dataset=settings.include_dataset, include_execution=settings.include_execution, optimize_settings=settings.optimize_settings, optimize_hparams=settings.optimize_hparams):
        return
    model_ctx = _resolve_infer_model_context(cfg, clearml_enabled=clearml_enabled, validation_mode=validation_mode)
    connect_reference = build_model_reference_payload(model_ctx.meta, model_bundle_path=model_ctx.model_bundle_path)
    connect_infer(ctx, cfg, model_id=connect_reference.get('infer_model_id') or connect_reference.get('infer_train_task_id') or str(model_ctx.model_bundle_path), model_abbr=model_ctx.model_abbr, infer_mode=mode, schema_policy=validation_mode, input_source=settings.input_source, input_path=settings.input_path_value, input_json=settings.input_json_label, provenance=model_ctx.provenance, optimize_payload=settings.optimize_hparams, include_dataset=settings.include_dataset, include_execution=settings.include_execution)
    columns_info = model_ctx.preprocess_bundle.get('columns') or {}
    quality_target = _normalize_str(columns_info.get('target_column') or _cfg_value(cfg, 'data.target_column'))
    quality_id_columns = columns_info.get('id_columns') or _cfg_value(cfg, 'data.id_columns') or []
    calibration_info = _resolve_calibration_info(model_ctx.bundle)
    predictor = model_ctx.calibrated_model if model_ctx.calibrated_model is not None else model_ctx.model
    uncertainty_info = _resolve_uncertainty_settings(cfg, model_ctx.bundle, task_type=model_ctx.task_type)
    batch_settings = _resolve_batch_settings(cfg)
    batch_children_settings = _resolve_batch_children_settings(cfg)
    batch_inputs_df = None
    batch_inputs_path = None
    if mode == 'batch' and (settings.batch_inputs_payload is not None or settings.batch_inputs_path_value):
        (batch_inputs_df, batch_inputs_path) = _load_batch_inputs(settings.batch_inputs_payload, settings.batch_inputs_path_value or settings.input_path_value, max_rows=batch_children_settings['max_children'])
    if mode == 'optimize':
        _run_optimize_mode_impl(cfg, ctx, clearml_enabled=clearml_enabled, optimize_settings=settings.optimize_settings, preprocess_bundle=model_ctx.preprocess_bundle, validation_mode=validation_mode, uncertainty_info=uncertainty_info, infer_cfg=settings.infer_cfg, meta=model_ctx.meta, model_bundle_path=model_ctx.model_bundle_path, model_abbr=model_ctx.model_abbr, table_settings=settings.table_settings, processed_dataset_id=model_ctx.processed_dataset_id, split_hash=model_ctx.split_hash, recipe_hash=model_ctx.recipe_hash)
        return
    if settings.use_batch_children and batch_inputs_df is not None:
        _run_batch_children_mode_impl(cfg, ctx, clearml_enabled=clearml_enabled, preprocess_bundle=model_ctx.preprocess_bundle, validation_mode=validation_mode, uncertainty_info=uncertainty_info, mode=mode, batch_inputs_df=batch_inputs_df, batch_inputs_path=batch_inputs_path, batch_children_settings=batch_children_settings, infer_cfg=settings.infer_cfg, meta=model_ctx.meta, model_bundle_path=model_ctx.model_bundle_path, model_abbr=model_ctx.model_abbr, table_settings=settings.table_settings, processed_dataset_id=model_ctx.processed_dataset_id, split_hash=model_ctx.split_hash, recipe_hash=model_ctx.recipe_hash)
        return
    if _run_chunked_batch_mode_impl(cfg, ctx, mode=mode, input_payload=settings.input_payload, input_path_value=settings.input_path_value, chunk_size=batch_settings['chunk_size'], output_format=batch_settings['output_format'], write_mode=batch_settings['write_mode'], max_rows=batch_settings['max_rows'], preprocess_bundle=model_ctx.preprocess_bundle, uncertainty_info=uncertainty_info, drift_settings=drift_settings, task_type=model_ctx.task_type, n_classes=model_ctx.n_classes, bundle=model_ctx.bundle, model=model_ctx.model, predictor=predictor, quality_target=quality_target, quality_id_columns=quality_id_columns, validation_mode=validation_mode, table_settings=settings.table_settings, clearml_enabled=clearml_enabled, meta=model_ctx.meta, model_bundle_path=model_ctx.model_bundle_path, calibration_info=calibration_info, calibrated_model=model_ctx.calibrated_model, processed_dataset_id=model_ctx.processed_dataset_id, split_hash=model_ctx.split_hash, recipe_hash=model_ctx.recipe_hash, debug_input_sample=debug_input_sample, debug_output_sample=debug_output_sample):
        return
    _run_standard_infer_mode(cfg=cfg, ctx=ctx, mode=mode, clearml_enabled=clearml_enabled, preprocess_bundle=model_ctx.preprocess_bundle, validation_mode=validation_mode, task_type=model_ctx.task_type, quality_target=quality_target, quality_id_columns=quality_id_columns, uncertainty_info=uncertainty_info, model_bundle_path=model_ctx.model_bundle_path, meta=model_ctx.meta, bundle=model_ctx.bundle, model=model_ctx.model, predictor=predictor, n_classes=model_ctx.n_classes, table_settings=settings.table_settings, drift_settings=drift_settings, output_format=batch_settings['output_format'], calibration_info=calibration_info, calibrated_model=model_ctx.calibrated_model, processed_dataset_id=model_ctx.processed_dataset_id, split_hash=model_ctx.split_hash, recipe_hash=model_ctx.recipe_hash, debug_input_sample=debug_input_sample, debug_output_sample=debug_output_sample)
