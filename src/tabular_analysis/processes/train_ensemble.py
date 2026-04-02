from __future__ import annotations
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str, normalize_task_type as _normalize_task_type, to_float as _to_float, to_int as _to_int
from ..common.collection_utils import dedupe_texts as _dedupe_tags, to_list as _to_list
from ..common.json_utils import load_json as _load_json
from dataclasses import make_dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import warnings
from ..clearml.hparams import connect_train_ensemble
from ..clearml.naming import apply_train_ensemble_naming
from ..clearml.ui_logger import log_debug_table, log_plotly, log_scalar
from ..io.bundle_io import load_bundle, save_bundle
from ..metrics.regression import REGRESSION_METRIC_ORDER, compute_regression_metrics
from ..ops.clearml_identity import apply_clearml_identity
from .train_shared import build_plotly_confusion_matrix as _build_plotly_confusion_matrix, build_plotly_roc_curve as _build_plotly_roc_curve, build_prediction_sample as _build_prediction_sample
from ..platform_adapter_artifacts import upload_artifact
from ..platform_adapter_clearml_env import is_clearml_enabled, resolve_version_props
from ..platform_adapter_task import PlatformAdapterError, clearml_task_id, clearml_task_status_from_obj, clearml_task_tags, get_task_artifact_local_copy, list_clearml_tasks_by_tags, register_model_artifact, update_task_properties
from ..registry.metrics import get_metric, metric_direction
from .lifecycle import emit_outputs_and_manifest, start_runtime
from ..viz.plots import plot_confusion_matrix, plot_regression_residuals, plot_roc_curve
from ..viz.regression_plots import build_regression_metrics_table, build_residuals_plot, build_true_pred_scatter
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError, json.JSONDecodeError)
PredsPayload = make_dataclass('PredsPayload', [('y_true', Any), ('y_pred', Any), ('y_proba', Any | None), ('class_labels', list[str] | None), ('source', str)])
Candidate = make_dataclass('Candidate', [('train_task_ref', str), ('train_task_id', str | None), ('preprocess_task_id', str | None), ('model_id', str | None), ('model_variant', str), ('task_type', str), ('n_classes', int | None), ('processed_dataset_id', str | None), ('split_hash', str | None), ('recipe_hash', str | None), ('metric_value', float), ('preds', PredsPayload), ('model_bundle_path', Path)])
class EnsemblePredictor:
    def __init__(self, *, task_type: str, method: str, base_models: list[Any], weights: list[float] | None=None, meta_model: Any | None=None, n_classes: int | None=None) -> None:
        self.task_type = task_type
        self.method = method
        self.base_models = list(base_models)
        self.weights = list(weights) if weights is not None else None
        self.meta_model = meta_model
        self.n_classes = n_classes
    def _predict_base(self, X, *, proba: bool) -> list[Any]:
        preds: list[Any] = []
        for model in self.base_models:
            if proba:
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            preds.append(pred)
        return preds
    def predict(self, X):
        import numpy as np
        if self.task_type != 'classification':
            if self.method == 'stacking':
                if self.meta_model is None:
                    raise ValueError('meta_model is required for stacking.')
                return self.meta_model.predict(self._stack_meta_features(X))
            preds = self._predict_base(X, proba=False)
            arr = np.column_stack([np.asarray(p).reshape(-1) for p in preds])
            weights = self.weights or [1.0 / arr.shape[1]] * arr.shape[1]
            return np.average(arr, axis=1, weights=weights)
        proba = self.predict_proba(X)
        return np.asarray(proba).argmax(axis=1)
    def predict_proba(self, X):
        import numpy as np
        if self.task_type != 'classification':
            raise ValueError('predict_proba is only available for classification.')
        if self.method == 'stacking' and self.meta_model is not None:
            meta_features = self._stack_meta_features(X)
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(meta_features)
            raise ValueError('meta_model lacks predict_proba.')
        proba_list = self._predict_base(X, proba=True)
        proba_arr = np.stack([np.asarray(p) for p in proba_list], axis=1)
        weights = self.weights or [1.0 / proba_arr.shape[1]] * proba_arr.shape[1]
        return np.average(proba_arr, axis=1, weights=weights)
    def _stack_meta_features(self, X):
        import numpy as np
        if self.task_type == 'classification':
            proba_list = self._predict_base(X, proba=True)
            arrays = [np.asarray(p) for p in proba_list]
            return np.hstack(arrays)
        preds = self._predict_base(X, proba=False)
        return np.column_stack([np.asarray(p).reshape(-1) for p in preds])
def _build_registry_tags(*, usecase_id: str, process: str, processed_dataset_id: str, split_hash: str, recipe_hash: str, preprocess_variant: str, model_variant: str, task_type: str | None, train_ensemble_task_id: str | None, train_task_ids: Iterable[str], preprocess_task_ids: Iterable[str], pipeline_task_id: str | None) -> list[str]:
    tags = [f'usecase:{usecase_id}', f'process:{process}', f'dataset:{processed_dataset_id}', f'split:{split_hash}', f'recipe:{recipe_hash}', f'preprocess:{preprocess_variant}', f'model_variant:{model_variant}']
    if task_type:
        tags.append(f'task_type:{task_type}')
    if train_ensemble_task_id:
        tags.append(f'task:train_ensemble:{train_ensemble_task_id}')
    for task_id in train_task_ids:
        if task_id:
            tags.append(f'task:train_model:{task_id}')
    for task_id in preprocess_task_ids:
        if task_id:
            tags.append(f'task:preprocess:{task_id}')
    if pipeline_task_id:
        tags.append(f'task:pipeline:{pipeline_task_id}')
    return _dedupe_tags(tags)
def _resolve_preprocess_variant(cfg: Any) -> str:
    for path in ('preprocess.variant', 'preprocess_variant.name', 'group.preprocess.preprocess_variant.name'):
        if value := _normalize_str(_cfg_value(cfg, path)):
            return value
    return 'unknown'
def _resolve_primary_metric(cfg: Any) -> str:
    return _normalize_str(_cfg_value(cfg, 'eval.primary_metric')) or 'rmse'
def _resolve_selection_metric(cfg: Any) -> str:
    return _normalize_str(_cfg_value(cfg, 'ensemble.selection_metric')) or _resolve_primary_metric(cfg)
def _resolve_top_k(cfg: Any) -> int:
    return _to_int(_cfg_value(cfg, 'ensemble.top_k'))
def _load_preds_valid(preds_path: Path, *, task_type: str, classes_path: Path | None) -> PredsPayload:
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('pandas/numpy are required to load preds_valid.') from exc
    df = pd.read_parquet(preds_path)
    if 'y_true' not in df.columns:
        raise ValueError('preds_valid.parquet missing y_true.')
    y_true = df['y_true'].to_numpy()
    y_pred = df['y_pred'].to_numpy() if 'y_pred' in df.columns else None
    if task_type == 'regression':
        if y_pred is None:
            raise ValueError('preds_valid.parquet missing y_pred for regression.')
        return PredsPayload(y_true=y_true, y_pred=y_pred, y_proba=None, class_labels=None, source='preds_valid')
    class_labels = None
    if classes_path is not None and classes_path.exists():
        payload = _load_json(classes_path)
        class_labels = [str(item) for item in payload] if isinstance(payload, list) else None
    proba_cols = [c for c in df.columns if c.startswith('proba__')]
    if len(proba_cols) < 2:
        raise ValueError('preds_valid.parquet missing proba__* columns for classification.')
    if class_labels:
        ordered_cols = []
        for label in class_labels:
            col = f'proba__{label}'
            if col in df.columns:
                ordered_cols.append(col)
        if len(ordered_cols) == len(class_labels):
            proba_cols = ordered_cols
    else:
        class_labels = [c.split('proba__', 1)[-1] for c in proba_cols]
    proba = df[proba_cols].to_numpy()
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    return PredsPayload(y_true=y_true, y_pred=y_pred, y_proba=proba, class_labels=class_labels, source='preds_valid')
def _extract_metric_value(metrics_payload: dict[str, Any] | None, metric_name: str) -> float | None:
    if not isinstance(metrics_payload, dict):
        return None
    holdout = metrics_payload.get('holdout')
    if isinstance(holdout, dict) and metric_name in holdout:
        return _to_float(holdout.get(metric_name))
    if isinstance(holdout, dict):
        key = _normalize_str(metric_name)
        if key:
            key = key.lower().replace('-', '_')
            for candidate_key in holdout:
                cand_norm = _normalize_str(candidate_key)
                if cand_norm and cand_norm.lower().replace('-', '_') == key:
                    return _to_float(holdout.get(candidate_key))
    return None
def _extract_model_variant(manifest: dict[str, Any] | None) -> str:
    if isinstance(manifest, dict):
        inputs = manifest.get('inputs')
        if isinstance(inputs, dict):
            if variant := _normalize_str(inputs.get('model_variant')):
                return variant
    return 'unknown'
def _status_completed(status: str | None) -> bool:
    return (not status) or status.strip().lower() in ('completed', 'closed', 'finished', 'success')
def _append_skipped(skipped: list[dict[str, Any]], *, train_task_id: str, model_variant: str, reason: str, details: Any=None) -> None:
    payload = {'train_task_id': train_task_id, 'model_variant': model_variant, 'reason': reason}
    if details is not None:
        payload['details'] = details
    skipped.append(payload)
def _candidate_matches_reference(*, ref_values: Mapping[str, Any], skipped: list[dict[str, Any]], train_task_id: str, model_variant: str, processed_dataset_id: str | None, split_hash: str | None, task_type: str, n_classes: int | None, preds: PredsPayload) -> bool:
    if ref_values.get('processed_dataset_id') and processed_dataset_id != ref_values.get('processed_dataset_id'):
        _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='preprocess_mismatch', details=processed_dataset_id or 'unknown')
        return False
    if ref_values.get('split_hash') and split_hash != ref_values.get('split_hash'):
        _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='split_mismatch', details=split_hash or 'unknown')
        return False
    if task_type != ref_values.get('task_type'):
        _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='task_type_mismatch', details=task_type)
        return False
    if n_classes is not None and ref_values.get('n_classes') is not None and n_classes != ref_values.get('n_classes'):
        _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='classes_mismatch', details=n_classes)
        return False
    if preds.class_labels and ref_values.get('class_labels') is not None and preds.class_labels != ref_values.get('class_labels'):
        _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='classes_mismatch', details='label_order')
        return False
    try:
        import numpy as np
        if not np.array_equal(preds.y_true, ref_values.get('y_true')):
            _append_skipped(skipped, train_task_id=train_task_id, model_variant=model_variant, reason='preds_mismatch', details='y_true')
            return False
    except _RECOVERABLE_ERRORS:
        pass
    return True
def _sync_reference_values(*, ref_values: dict[str, Any], skipped: list[dict[str, Any]], train_task_id: str, model_variant: str, processed_dataset_id: str | None, split_hash: str | None, recipe_hash: str | None, task_type: str, n_classes: int | None, preds: PredsPayload) -> bool:
    if not ref_values:
        ref_values.update({'processed_dataset_id': processed_dataset_id, 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'task_type': task_type, 'n_classes': n_classes, 'class_labels': preds.class_labels, 'y_true': preds.y_true})
        return True
    return _candidate_matches_reference(ref_values=ref_values, skipped=skipped, train_task_id=train_task_id, model_variant=model_variant, processed_dataset_id=processed_dataset_id, split_hash=split_hash, task_type=task_type, n_classes=n_classes, preds=preds)
def _collect_candidates_clearml(cfg: Any, *, preprocess_variant: str, selection_metric: str, exclude_variants: set[str]) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any]]:
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
    grid_run_id = _normalize_str(_cfg_value(cfg, 'run.grid_run_id'))
    tags = [f'usecase:{usecase_id}', 'process:train_model', f'preprocess:{preprocess_variant}']
    if grid_run_id:
        tags.append(f'grid:{grid_run_id}')
    skipped: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    ref_values: dict[str, Any] = {}
    tasks = list_clearml_tasks_by_tags(tags)
    for task in tasks:
        task_id = clearml_task_id(task)
        if not task_id:
            continue
        task_tags = clearml_task_tags(task)
        if 'template:true' in task_tags:
            continue
        status = clearml_task_status_from_obj(task)
        if status and (not _status_completed(status)):
            _append_skipped(skipped, train_task_id=task_id, model_variant='unknown', reason='status_incomplete', details=status)
            continue
        try:
            out_path = get_task_artifact_local_copy(cfg, task_id, 'out.json')
        except PlatformAdapterError as exc:
            _append_skipped(skipped, train_task_id=task_id, model_variant='unknown', reason='missing_out_json', details=str(exc))
            continue
        out = _load_json(out_path)
        out_status = _normalize_str(out.get('status'))
        if out_status and (not _status_completed(out_status)):
            _append_skipped(skipped, train_task_id=task_id, model_variant='unknown', reason='status_incomplete', details=out_status)
            continue
        processed_dataset_id = _normalize_str(out.get('processed_dataset_id'))
        split_hash = _normalize_str(out.get('split_hash'))
        recipe_hash = _normalize_str(out.get('recipe_hash'))
        preprocess_task_id = _normalize_str(out.get('preprocess_task_id'))
        task_type = _normalize_task_type(out.get('task_type'))
        n_classes = _to_int(out.get('n_classes'))
        model_id = _normalize_str(out.get('model_id'))
        manifest = None
        try:
            manifest_path = get_task_artifact_local_copy(cfg, task_id, 'manifest.json')
            manifest = _load_json(manifest_path)
        except PlatformAdapterError:
            manifest = None
        model_variant = _extract_model_variant(manifest)
        if model_variant in exclude_variants:
            _append_skipped(skipped, train_task_id=task_id, model_variant=model_variant, reason='excluded_variant')
            continue
        try:
            metrics_path = get_task_artifact_local_copy(cfg, task_id, 'metrics.json')
            metrics_payload = _load_json(metrics_path)
        except PlatformAdapterError as exc:
            _append_skipped(skipped, train_task_id=task_id, model_variant=model_variant, reason='missing_metrics', details=str(exc))
            continue
        metric_value = _extract_metric_value(metrics_payload, selection_metric)
        if metric_value is None:
            _append_skipped(skipped, train_task_id=task_id, model_variant=model_variant, reason='metric_missing', details=selection_metric)
            continue
        try:
            preds_path = get_task_artifact_local_copy(cfg, task_id, 'preds_valid.parquet')
            classes_path = None
            try:
                classes_path = get_task_artifact_local_copy(cfg, task_id, 'classes.json')
            except PlatformAdapterError:
                classes_path = None
            preds = _load_preds_valid(preds_path, task_type=task_type, classes_path=classes_path)
        except _RECOVERABLE_ERRORS as exc:
            _append_skipped(skipped, train_task_id=task_id, model_variant=model_variant, reason='preds_invalid', details=str(exc))
            continue
        try:
            model_bundle_path = get_task_artifact_local_copy(cfg, task_id, 'model_bundle.joblib')
        except PlatformAdapterError as exc:
            _append_skipped(skipped, train_task_id=task_id, model_variant=model_variant, reason='missing_model_bundle', details=str(exc))
            continue
        if not _sync_reference_values(ref_values=ref_values, skipped=skipped, train_task_id=task_id, model_variant=model_variant, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, task_type=task_type, n_classes=n_classes, preds=preds):
            continue
        candidates.append(Candidate(train_task_ref=task_id, train_task_id=task_id, preprocess_task_id=preprocess_task_id, model_id=model_id, model_variant=model_variant, task_type=task_type, n_classes=n_classes, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, metric_value=float(metric_value), preds=preds, model_bundle_path=model_bundle_path))
    return (candidates, skipped, ref_values)
def _parse_preprocess_variant_from_run_root(run_root: Path) -> str | None:
    name = run_root.name
    if not name.startswith('train__'):
        return None
    parts = name.split('__')
    return parts[1] or None if len(parts) >= 2 else None
def _collect_candidates_local(cfg: Any, *, preprocess_variant: str, selection_metric: str, exclude_variants: set[str]) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    ref_values: dict[str, Any] = {}
    run_root = Path(_cfg_value(cfg, 'run.output_dir') or '.').expanduser()
    search_root = run_root.parent if run_root.name.startswith('ensemble__') else run_root
    for out_path in search_root.glob('train__*/03_train_model/out.json'):
        run_dir = out_path.parent
        train_root = run_dir.parent
        parsed_variant = _parse_preprocess_variant_from_run_root(train_root)
        if parsed_variant and parsed_variant != preprocess_variant:
            continue
        out = _load_json(out_path)
        out_status = _normalize_str(out.get('status'))
        if out_status and (not _status_completed(out_status)):
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant='unknown', reason='status_incomplete', details=out_status)
            continue
        processed_dataset_id = _normalize_str(out.get('processed_dataset_id'))
        split_hash = _normalize_str(out.get('split_hash'))
        recipe_hash = _normalize_str(out.get('recipe_hash'))
        preprocess_task_id = _normalize_str(out.get('preprocess_task_id'))
        task_type = _normalize_task_type(out.get('task_type'))
        n_classes = _to_int(out.get('n_classes'))
        model_id = _normalize_str(out.get('model_id'))
        manifest = None
        manifest_path = run_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                manifest = _load_json(manifest_path)
            except _RECOVERABLE_ERRORS:
                manifest = None
        model_variant = _extract_model_variant(manifest)
        if model_variant in exclude_variants:
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant=model_variant, reason='excluded_variant')
            continue
        metrics_path = run_dir / 'metrics.json'
        if not metrics_path.exists():
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant=model_variant, reason='missing_metrics')
            continue
        metrics_payload = _load_json(metrics_path)
        metric_value = _extract_metric_value(metrics_payload, selection_metric)
        if metric_value is None:
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant=model_variant, reason='metric_missing', details=selection_metric)
            continue
        preds_path = run_dir / 'preds_valid.parquet'
        classes_path = run_dir / 'classes.json'
        try:
            preds = _load_preds_valid(preds_path, task_type=task_type, classes_path=classes_path)
        except _RECOVERABLE_ERRORS as exc:
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant=model_variant, reason='preds_invalid', details=str(exc))
            continue
        model_bundle_path = run_dir / 'model_bundle.joblib'
        if not model_bundle_path.exists():
            _append_skipped(skipped, train_task_id=str(run_dir), model_variant=model_variant, reason='missing_model_bundle')
            continue
        if not _sync_reference_values(ref_values=ref_values, skipped=skipped, train_task_id=str(run_dir), model_variant=model_variant, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, task_type=task_type, n_classes=n_classes, preds=preds):
            continue
        candidates.append(Candidate(train_task_ref=str(run_dir), train_task_id=_normalize_str(out.get('train_task_id')), preprocess_task_id=preprocess_task_id, model_id=model_id, model_variant=model_variant, task_type=task_type, n_classes=n_classes, processed_dataset_id=processed_dataset_id, split_hash=split_hash, recipe_hash=recipe_hash, metric_value=float(metric_value), preds=preds, model_bundle_path=model_bundle_path))
    return (candidates, skipped, ref_values)
def _load_base_models(candidates: list[Candidate], *, skipped: list[dict[str, Any]]) -> tuple[list[Any], list[Candidate], dict[str, Any] | None]:
    base_models: list[Any] = []
    loaded_candidates: list[Candidate] = []
    preprocess_bundle: dict[str, Any] | None = None
    for cand in candidates:
        try:
            bundle = load_bundle(cand.model_bundle_path)
        except _RECOVERABLE_ERRORS as exc:
            _append_skipped(skipped, train_task_id=cand.train_task_ref, model_variant=cand.model_variant, reason='missing_model_bundle', details=str(exc))
            continue
        if not isinstance(bundle, dict):
            _append_skipped(skipped, train_task_id=cand.train_task_ref, model_variant=cand.model_variant, reason='missing_model_bundle', details='bundle_not_dict')
            continue
        predictor = bundle.get('calibrated_model') or bundle.get('model')
        if predictor is None:
            _append_skipped(skipped, train_task_id=cand.train_task_ref, model_variant=cand.model_variant, reason='missing_model_bundle', details='model_missing')
            continue
        base_models.append(predictor)
        loaded_candidates.append(cand)
        if preprocess_bundle is None:
            preprocess = bundle.get('preprocess_bundle')
            if isinstance(preprocess, dict):
                preprocess_bundle = preprocess
    return (base_models, loaded_candidates, preprocess_bundle)
def _record_failure(*, ctx: Any, cfg: Any, reason: str, ref_values: Mapping[str, Any], skipped: list[dict[str, Any]]) -> None:
    task_id = None
    if getattr(ctx, 'task', None) is not None:
        task_id = getattr(ctx.task, 'id', None)
        if task_id:
            task_id = str(task_id)
    out = {'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash'), 'train_task_id': task_id, 'model_id': None, 'best_score': None, 'primary_metric': _resolve_primary_metric(cfg), 'task_type': ref_values.get('task_type'), 'status': 'failed', 'reason': reason}
    if ref_values.get('n_classes') is not None:
        out['n_classes'] = ref_values.get('n_classes')
    clearml_enabled = is_clearml_enabled(cfg)
    versions = resolve_version_props(cfg, clearml_enabled=clearml_enabled)
    emit_outputs_and_manifest(ctx, cfg, process='train_ensemble', out=out, inputs={'preprocess_variant': _resolve_preprocess_variant(cfg), 'method': _normalize_str(_cfg_value(cfg, 'ensemble.method'))}, outputs={'model_id': None, 'best_score': None, 'primary_metric': _resolve_primary_metric(cfg), 'task_type': ref_values.get('task_type')}, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash')}, manifest_extra={'error': {'type': 'no_valid_base_models', 'message': reason}}, clearml_enabled=clearml_enabled)
    if skipped:
        spec_path = ctx.output_dir / 'ensemble_spec.json'
        spec = {'method': _normalize_str(_cfg_value(cfg, 'ensemble.method')) or 'unknown', 'selection_metric': _resolve_selection_metric(cfg), 'direction': metric_direction(_resolve_selection_metric(cfg), ref_values.get('task_type') or 'regression'), 'top_k': _resolve_top_k(cfg), 'preprocess_variant': _resolve_preprocess_variant(cfg), 'included': [], 'skipped': skipped, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'code_version': versions.get('code_version', 'unknown'), 'schema_version': versions.get('schema_version', 'unknown')}
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        if is_clearml_enabled(cfg):
            upload_artifact(ctx, 'ensemble_spec.json', spec_path)
EnsembleFitResult = make_dataclass('EnsembleFitResult', [('y_true', Any), ('y_pred', Any), ('y_proba', Any | None), ('ensemble_model', EnsemblePredictor), ('metrics_holdout', dict[str, Any]), ('best_score', float), ('primary_metric_source', str), ('method_used', str), ('weights', list[float] | None), ('ensemble_meta', dict[str, Any]), ('primary_direction', str)], frozen=True)
def _fit_ensemble_predictor(*, cfg: Any, selected: list[Candidate], base_models: list[Any], method: str, primary_metric: str, task_type: str, n_classes: int | None) -> EnsembleFitResult:
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS as exc:
        raise RuntimeError('numpy is required for ensemble training.') from exc
    if not selected:
        raise ValueError('No selected candidates for ensemble fit.')
    y_true = np.asarray(selected[0].preds.y_true)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    primary_direction = metric_direction(primary_metric, task_type)
    method_used = method if method in {'mean_topk', 'weighted', 'stacking'} else 'mean_topk'
    seed = _to_int(_cfg_value(cfg, 'ensemble.stacking.seed')) or 42
    ensemble_meta: dict[str, Any] = {'selection_metric_direction': primary_direction}
    weights: list[float] | None = None
    meta_model: Any | None = None
    def _normalize_weights(raw: list[float]) -> list[float] | None:
        if not raw:
            return None
        clipped = [max(float(v), 0.0) for v in raw]
        total = sum(clipped)
        if total <= 0.0 or not math.isfinite(total):
            return None
        return [v / total for v in clipped]
    if method_used == 'weighted':
        score_values = [float(c.metric_value) for c in selected]
        if primary_direction == 'minimize':
            score_values = [1.0 / max(abs(v), 1e-12) for v in score_values]
        weights = _normalize_weights(score_values)
        if weights is None:
            method_used = 'mean_topk'
            ensemble_meta['weights_fallback'] = 'uniform'
    y_pred: Any
    y_proba: Any | None
    if task_type == 'classification':
        inferred_classes = int(n_classes or 0)
        if inferred_classes < 2:
            try:
                inferred_classes = int(max(y_true)) + 1
            except _RECOVERABLE_ERRORS:
                inferred_classes = 2
        inferred_classes = max(inferred_classes, 2)
        proba_list: list[Any] = []
        for cand in selected:
            cand_proba = cand.preds.y_proba
            if cand_proba is not None:
                arr = np.asarray(cand_proba)
                if arr.ndim == 1:
                    arr = np.stack([1.0 - arr, arr], axis=1)
                proba_list.append(arr)
                continue
            pred = np.asarray(cand.preds.y_pred).reshape(-1).astype(int)
            arr = np.zeros((pred.shape[0], inferred_classes), dtype=float)
            arr[np.arange(pred.shape[0]), np.clip(pred, 0, inferred_classes - 1)] = 1.0
            proba_list.append(arr)
        if method_used == 'stacking':
            try:
                from sklearn.linear_model import LogisticRegression
            except _RECOVERABLE_ERRORS:
                method_used = 'mean_topk'
                ensemble_meta['stacking_fallback'] = 'missing_sklearn'
            else:
                X_meta = np.hstack([np.asarray(p) for p in proba_list])
                meta_model = LogisticRegression(max_iter=1000, random_state=seed)
                meta_model.fit(X_meta, y_true)
                if hasattr(meta_model, 'predict_proba'):
                    y_proba = meta_model.predict_proba(X_meta)
                else:
                    y_proba = np.asarray(meta_model.predict(X_meta))
                    if y_proba.ndim == 1:
                        y_proba = np.stack([1.0 - y_proba, y_proba], axis=1)
                y_pred = np.asarray(y_proba).argmax(axis=1)
        if method_used != 'stacking':
            proba_stack = np.stack([np.asarray(p) for p in proba_list], axis=1)
            if weights is None:
                weights = [1.0 / proba_stack.shape[1]] * proba_stack.shape[1]
            y_proba = np.average(proba_stack, axis=1, weights=weights)
            y_pred = np.asarray(y_proba).argmax(axis=1)
        metrics_holdout: dict[str, Any] = {}
        try:
            primary_fn = get_metric(primary_metric, task_type, n_classes=inferred_classes, beta=1.0)
            metrics_holdout[primary_metric] = float(primary_fn(y_true, y_pred, y_proba))
        except _RECOVERABLE_ERRORS:
            pass
        try:
            accuracy_fn = get_metric('accuracy', task_type, n_classes=inferred_classes, beta=1.0)
            metrics_holdout.setdefault('accuracy', float(accuracy_fn(y_true, y_pred, y_proba)))
        except _RECOVERABLE_ERRORS:
            pass
        if primary_metric not in metrics_holdout:
            raise ValueError(f"primary_metric '{primary_metric}' missing from ensemble metrics.")
    else:
        pred_matrix = np.column_stack([np.asarray(c.preds.y_pred).reshape(-1) for c in selected])
        if method_used == 'stacking':
            try:
                from sklearn.linear_model import Ridge
            except _RECOVERABLE_ERRORS:
                method_used = 'mean_topk'
                ensemble_meta['stacking_fallback'] = 'missing_sklearn'
            else:
                meta_model = Ridge(random_state=seed)
                meta_model.fit(pred_matrix, y_true)
                y_pred = meta_model.predict(pred_matrix)
        if method_used != 'stacking':
            if weights is None:
                weights = [1.0 / pred_matrix.shape[1]] * pred_matrix.shape[1]
            y_pred = np.average(pred_matrix, axis=1, weights=weights)
        y_proba = None
        metrics_holdout = compute_regression_metrics(y_true, y_pred, metrics=list(REGRESSION_METRIC_ORDER))
        if primary_metric not in metrics_holdout:
            primary_fn = get_metric(primary_metric, task_type, n_classes=n_classes, beta=1.0)
            metrics_holdout[primary_metric] = float(primary_fn(y_true, y_pred, None))
    metrics_holdout['val_rows'] = int(len(y_true))
    best_score = float(metrics_holdout[primary_metric])
    ensemble_model = EnsemblePredictor(task_type=task_type, method=method_used, base_models=base_models, weights=weights, meta_model=meta_model, n_classes=n_classes)
    primary_metric_source = 'holdout'
    ensemble_meta['method'] = method_used
    ensemble_meta['weights'] = weights
    ensemble_meta['n_selected'] = len(selected)
    return EnsembleFitResult(y_true=y_true, y_pred=y_pred, y_proba=y_proba, ensemble_model=ensemble_model, metrics_holdout=metrics_holdout, best_score=best_score, primary_metric_source=primary_metric_source, method_used=method_used, weights=weights, ensemble_meta=ensemble_meta, primary_direction=primary_direction)
def _prepare_visual_artifacts(*, cfg: Any, ctx: Any, clearml_enabled: bool, task_type: str, n_classes: int | None, y_true: Any, y_pred: Any, y_proba: Any | None, class_names: list[str] | None) -> tuple[dict[str, Any], Any | None, Path | None, Path | None, Path | None]:
    try:
        max_points = int(_cfg_value(cfg, 'viz.max_points', 1000))
    except _RECOVERABLE_ERRORS:
        max_points = 1000
    viz_settings = {'enabled': bool(_cfg_value(cfg, 'viz.enabled', True)), 'max_points': max_points, 'confusion_normalize': bool(_cfg_value(cfg, 'viz.confusion_normalize', True)), 'roc_curve': bool(_cfg_value(cfg, 'viz.roc_curve', True))}
    debug_sample = None
    residuals_plot_path: Path | None = None
    confusion_plot_path: Path | None = None
    roc_plot_path: Path | None = None
    if clearml_enabled:
        debug_sample = _build_prediction_sample(y_true, y_pred, y_proba)
        if viz_settings['enabled']:
            if task_type == 'regression':
                residuals_plot_path = plot_regression_residuals(y_true, y_pred, ctx.output_dir / 'residuals.png', max_points=viz_settings['max_points'])
            else:
                confusion_plot_path = plot_confusion_matrix(y_true, y_pred, ctx.output_dir / 'confusion_matrix.png', class_names=class_names, normalize=viz_settings['confusion_normalize'])
                if viz_settings['roc_curve'] and n_classes == 2 and (y_proba is not None):
                    roc_plot_path = plot_roc_curve(y_true, y_proba[:, 1], ctx.output_dir / 'roc_curve.png')
    return (viz_settings, debug_sample, residuals_plot_path, confusion_plot_path, roc_plot_path)
def _write_out_and_manifest(*, ctx: Any, cfg: Any, versions: Mapping[str, Any], ref_values: Mapping[str, Any], preprocess_variant: str, method: str, top_k: int, selection_metric: str, primary_metric: str, primary_direction: str, task_type: str, primary_metric_source: str, model_id: str, best_score: float, task_id: str | None, pipeline_task_id: str | None, preprocess_task_ids: list[str], train_task_ids: list[str], n_classes: int | None, registry_model_id: str | None, registry_status: str | None, registry_error: dict[str, Any] | None) -> None:
    out = {'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash'), 'train_task_id': task_id, 'pipeline_task_id': pipeline_task_id, 'preprocess_task_ids': preprocess_task_ids, 'base_train_task_ids': train_task_ids, 'model_id': model_id, 'best_score': best_score, 'primary_metric': primary_metric, 'task_type': task_type, 'primary_metric_source': primary_metric_source}
    if registry_model_id:
        out['registry_model_id'] = registry_model_id
    if registry_status:
        out['registry_status'] = registry_status
    if registry_error:
        out['registry_error'] = registry_error
    if n_classes is not None:
        out['n_classes'] = n_classes
    inputs = {'preprocess_variant': preprocess_variant, 'method': method, 'top_k': top_k, 'selection_metric': selection_metric, 'primary_metric': primary_metric, 'direction': primary_direction}
    outputs = {'model_id': model_id, 'best_score': best_score, 'primary_metric': primary_metric, 'task_type': task_type, 'primary_metric_source': primary_metric_source}
    emit_outputs_and_manifest(ctx, cfg, process='train_ensemble', out=out, inputs=inputs, outputs=outputs, hash_payloads={'config_hash': ('config', cfg), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash')}, clearml_enabled=is_clearml_enabled(cfg))
def _persist_ensemble_artifacts(*, ctx: Any, cfg: Any, versions: Mapping[str, Any], selected: list[Candidate], skipped: list[dict[str, Any]], ref_values: Mapping[str, Any], preprocess_variant: str, method: str, method_used: str, selection_metric: str, selection_direction: str, top_k: int, primary_metric: str, primary_metric_source: str, primary_direction: str, task_type: str, n_classes: int | None, weights: list[float] | None, ensemble_meta: Mapping[str, Any], ensemble_model: EnsemblePredictor, preprocess_bundle: Mapping[str, Any], metrics_holdout: Mapping[str, Any], best_score: float, task_id: str | None) -> dict[str, Any]:
    included = [{'train_task_id': c.train_task_id or c.train_task_ref, 'model_variant': c.model_variant, 'metric_value': c.metric_value, 'preds_ref': c.preds.source} for c in selected]
    ensemble_spec = {'method': method_used, 'selection_metric': selection_metric, 'direction': selection_direction, 'top_k': top_k, 'preprocess_variant': preprocess_variant, 'primary_metric': primary_metric, 'primary_metric_source': primary_metric_source, 'n_base_models': len(included), 'included': included, 'skipped': skipped, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'code_version': versions.get('code_version', 'unknown'), 'schema_version': versions.get('schema_version', 'unknown')}
    if method_used != method:
        ensemble_spec['configured_method'] = method
    if weights is not None:
        ensemble_spec['weights'] = {c.train_task_id or c.train_task_ref: float(weights[i]) for (i, c) in enumerate(selected)}
    ensemble_spec.update(dict(ensemble_meta))
    spec_path = ctx.output_dir / 'ensemble_spec.json'
    spec_path.write_text(json.dumps(ensemble_spec, ensure_ascii=False, indent=2), encoding='utf-8')
    metrics_payload = {'primary_metric': primary_metric, 'direction': primary_direction, 'task_type': task_type, 'primary_metric_source': primary_metric_source, 'holdout': dict(metrics_holdout)}
    metrics_path = ctx.output_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    ensemble_payload = {'method': method_used, 'base_train_task_ids': [c.train_task_id or c.train_task_ref for c in selected], 'weights': weights, 'primary_metric_source': primary_metric_source}
    if method_used != method:
        ensemble_payload['configured_method'] = method
    model_bundle = {'model': ensemble_model, 'model_variant': f'ensemble_{method_used}', 'primary_metric': primary_metric, 'best_score': best_score, 'metrics': metrics_payload, 'preprocess_bundle': dict(preprocess_bundle), 'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'recipe_hash': ref_values.get('recipe_hash'), 'task_type': task_type, 'n_classes': n_classes, 'ensemble': ensemble_payload, 'provenance': {'train_task_id': task_id, 'processed_dataset_id': ref_values.get('processed_dataset_id'), 'preprocess_variant': preprocess_variant}}
    model_bundle_path = ctx.output_dir / 'model_bundle.joblib'
    save_bundle(model_bundle_path, model_bundle)
    model_id = str(model_bundle_path)
    train_task_ids = [c.train_task_id or c.train_task_ref for c in selected if c.train_task_id or c.train_task_ref]
    preprocess_task_ids = sorted({c.preprocess_task_id for c in selected if c.preprocess_task_id is not None and str(c.preprocess_task_id).strip()})
    return {'included': included, 'spec_path': spec_path, 'metrics_path': metrics_path, 'model_bundle_path': model_bundle_path, 'model_id': model_id, 'train_task_ids': train_task_ids, 'preprocess_task_ids': preprocess_task_ids}
def _register_ensemble_model_if_enabled(*, ctx: Any, cfg: Any, clearml_enabled: bool, ref_values: Mapping[str, Any], preprocess_variant: str, task_type: str, method_used: str, task_id: str | None, pipeline_task_id: str | None, train_task_ids: list[str], preprocess_task_ids: list[str], model_bundle_path: Path) -> tuple[str | None, str | None, dict[str, Any] | None]:
    registry_model_id: str | None = None
    registry_status: str | None = None
    registry_error: dict[str, Any] | None = None
    if not clearml_enabled:
        return (registry_model_id, registry_status, registry_error)
    train_task_tag_limit = _to_int(_cfg_value(cfg, 'train_ensemble.registry.tag_limits.train_task_ids'))
    train_task_ids_tagged = train_task_ids
    train_task_truncated = False
    if train_task_tag_limit and train_task_tag_limit > 0 and (len(train_task_ids) > train_task_tag_limit):
        train_task_ids_tagged = train_task_ids[:train_task_tag_limit]
        train_task_truncated = True
    raw_full_ids = _cfg_value(cfg, 'train_ensemble.registry.metadata.full_train_task_ids')
    if raw_full_ids is None:
        full_ids_to_metadata = True
    elif isinstance(raw_full_ids, bool):
        full_ids_to_metadata = raw_full_ids
    else:
        full_ids_to_metadata = str(raw_full_ids).strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    ensemble_variant = f'ensemble_{method_used}'
    usecase_id = _normalize_str(_cfg_value(cfg, 'run.usecase_id')) or 'unknown'
    model_name = f"{usecase_id}:{ensemble_variant}:{ref_values.get('processed_dataset_id')}"
    metadata: dict[str, Any] | None = None
    if full_ids_to_metadata:
        metadata = {'train_task_ids_full': train_task_ids, 'preprocess_task_ids_full': preprocess_task_ids, 'pipeline_task_id': pipeline_task_id, 'train_task_ids_tag_limit': train_task_tag_limit, 'train_task_ids_truncated': train_task_truncated}
    tags = _build_registry_tags(usecase_id=usecase_id, process='train_ensemble', processed_dataset_id=str(ref_values.get('processed_dataset_id')), split_hash=str(ref_values.get('split_hash')), recipe_hash=str(ref_values.get('recipe_hash')), preprocess_variant=preprocess_variant, model_variant=ensemble_variant, task_type=task_type, train_ensemble_task_id=task_id, train_task_ids=train_task_ids_tagged, preprocess_task_ids=preprocess_task_ids, pipeline_task_id=pipeline_task_id)
    if train_task_ids:
        tags.append(f'task:train_model_count:{len(train_task_ids)}')
    if train_task_truncated:
        tags.append('task:train_model_truncated:true')
    tags = _dedupe_tags(tags)
    try:
        registry_model_id = register_model_artifact(ctx, model_path=model_bundle_path, model_name=model_name, tags=tags, metadata=metadata)
        registry_status = 'registered'
    except _RECOVERABLE_ERRORS as exc:
        registry_status = 'failed'
        registry_error = {'type': exc.__class__.__name__, 'message': str(exc)}
        warnings.warn(f'Failed to register ensemble model in ClearML registry: {exc}')
    return (registry_model_id, registry_status, registry_error)
def _log_ensemble_to_clearml(*, ctx: Any, clearml_enabled: bool, spec_path: Path, metrics_path: Path, model_bundle_path: Path, ref_values: Mapping[str, Any], model_id: str, primary_metric: str, best_score: float, task_type: str, n_classes: int | None, registry_model_id: str | None, registry_status: str | None, metrics_holdout: Mapping[str, Any], included: list[dict[str, Any]], skipped: list[dict[str, Any]], weights: list[float] | None, selected: list[Candidate], debug_sample: Any | None, viz_settings: Mapping[str, Any], y_true: Any, y_pred: Any, y_proba: Any | None, class_names: list[str] | None, residuals_plot_path: Path | None, confusion_plot_path: Path | None, roc_plot_path: Path | None) -> None:
    if not clearml_enabled:
        return
    upload_artifact(ctx, 'ensemble_spec.json', spec_path)
    upload_artifact(ctx, 'metrics.json', metrics_path)
    upload_artifact(ctx, 'model_bundle.joblib', model_bundle_path)
    props = {'processed_dataset_id': ref_values.get('processed_dataset_id'), 'split_hash': ref_values.get('split_hash'), 'model_id': model_id, 'primary_metric': primary_metric, 'best_score': best_score, 'task_type': task_type, 'n_classes': n_classes}
    if registry_model_id:
        props['registry_model_id'] = registry_model_id
    if registry_status:
        props['registry_status'] = registry_status
    update_task_properties(ctx, props)
    if task_type == 'regression':
        for name in REGRESSION_METRIC_ORDER:
            if name in metrics_holdout:
                log_scalar(ctx.task, 'metrics', name, metrics_holdout[name], step=0)
    log_scalar(ctx.task, 'ensemble', 'best_score', best_score, step=0)
    log_scalar(ctx.task, 'ensemble', 'n_included', len(included), step=0)
    log_scalar(ctx.task, 'ensemble', 'n_skipped', len(skipped), step=0)
    log_scalar(ctx.task, 'metrics', primary_metric, best_score, step=0)
    log_debug_table(ctx.task, 'ensemble', 'included', included, step=0)
    if skipped:
        log_debug_table(ctx.task, 'ensemble', 'skipped', skipped, step=0)
    if weights is not None:
        weights_table = [{'train_task_id': c.train_task_id or c.train_task_ref, 'model_variant': c.model_variant, 'weight': float(weights[i])} for (i, c) in enumerate(selected)]
        log_debug_table(ctx.task, 'ensemble', 'weights', weights_table, step=0)
    if debug_sample is not None:
        log_debug_table(ctx.task, 'train_ensemble', 'prediction_sample', debug_sample, step=0)
    if viz_settings['enabled']:
        if task_type == 'regression':
            numeric_metrics = {key: float(value) for (key, value) in metrics_holdout.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}
            metrics_table = build_regression_metrics_table(numeric_metrics)
            log_plotly(ctx.task, 'train_ensemble', 'metrics_table', metrics_table, step=0)
            scatter = build_true_pred_scatter(y_true, y_pred, r2=metrics_holdout.get('r2'), max_points=viz_settings['max_points'])
            log_plotly(ctx.task, 'train_ensemble', 'true_vs_pred', scatter, step=0)
            fig = build_residuals_plot(y_true, y_pred, max_points=viz_settings['max_points'])
            log_plotly(ctx.task, 'train_ensemble', 'residuals', fig or residuals_plot_path, step=0)
        else:
            fig = _build_plotly_confusion_matrix(y_true, y_pred, class_names=class_names, normalize=viz_settings['confusion_normalize'])
            log_plotly(ctx.task, 'train_ensemble', 'confusion_matrix', fig or confusion_plot_path, step=0)
            if y_proba is not None and n_classes == 2:
                fig = _build_plotly_roc_curve(y_true, y_proba[:, 1])
                log_plotly(ctx.task, 'train_ensemble', 'roc_curve', fig or roc_plot_path, step=0)
def run(cfg: Any) -> None:
    identity = apply_clearml_identity(cfg, stage=cfg.task.stage)
    apply_train_ensemble_naming(cfg)
    ctx = start_runtime(cfg, stage=cfg.task.stage, task_name='train_ensemble', tags=identity.tags, properties=identity.user_properties)
    clearml_enabled = is_clearml_enabled(cfg)
    preprocess_variant = _resolve_preprocess_variant(cfg)
    selection_metric = _resolve_selection_metric(cfg)
    primary_metric = _resolve_primary_metric(cfg)
    method = _normalize_str(_cfg_value(cfg, 'ensemble.method')) or 'mean_topk'
    top_k = _resolve_top_k(cfg)
    exclude_variants = set(_to_list(_cfg_value(cfg, 'ensemble.exclude_variants')))
    if clearml_enabled:
        (candidates, skipped, ref_values) = _collect_candidates_clearml(cfg, preprocess_variant=preprocess_variant, selection_metric=selection_metric, exclude_variants=exclude_variants)
    else:
        (candidates, skipped, ref_values) = _collect_candidates_local(cfg, preprocess_variant=preprocess_variant, selection_metric=selection_metric, exclude_variants=exclude_variants)
    if not candidates:
        _record_failure(ctx=ctx, cfg=cfg, reason='no_valid_base_models', ref_values=ref_values, skipped=skipped)
        return
    task_type = candidates[0].task_type
    n_classes = candidates[0].n_classes
    selection_direction = metric_direction(selection_metric, task_type)
    sorted_candidates = sorted(candidates, key=lambda c: c.metric_value, reverse=selection_direction == 'maximize')
    if top_k <= 0:
        top_k = len(sorted_candidates)
    if method == 'weighted':
        top_k_max = _to_int(_cfg_value(cfg, 'ensemble.weighted.top_k_max')) or top_k
        top_k = min(top_k, top_k_max)
    selected = sorted_candidates[:max(top_k, 1)]
    (base_models, loaded_candidates, preprocess_bundle) = _load_base_models(selected, skipped=skipped)
    if not base_models:
        _record_failure(ctx=ctx, cfg=cfg, reason='no_valid_base_models', ref_values=ref_values, skipped=skipped)
        return
    if preprocess_bundle is None:
        raise ValueError('preprocess_bundle is missing from base model bundle.')
    selected = loaded_candidates
    if not selected:
        _record_failure(ctx=ctx, cfg=cfg, reason='no_valid_base_models', ref_values=ref_values, skipped=skipped)
        return
    fit_result = _fit_ensemble_predictor(cfg=cfg, selected=selected, base_models=base_models, method=method, primary_metric=primary_metric, task_type=task_type, n_classes=n_classes)
    y_true = fit_result.y_true
    y_pred = fit_result.y_pred
    y_proba = fit_result.y_proba
    ensemble_model = fit_result.ensemble_model
    metrics_holdout = fit_result.metrics_holdout
    best_score = fit_result.best_score
    primary_metric_source = fit_result.primary_metric_source
    method_used = fit_result.method_used
    weights = fit_result.weights
    ensemble_meta = fit_result.ensemble_meta
    primary_direction = fit_result.primary_direction
    class_names = selected[0].preds.class_labels if task_type == 'classification' else None
    (viz_settings, debug_sample, residuals_plot_path, confusion_plot_path, roc_plot_path) = _prepare_visual_artifacts(cfg=cfg, ctx=ctx, clearml_enabled=clearml_enabled, task_type=task_type, n_classes=n_classes, y_true=y_true, y_pred=y_pred, y_proba=y_proba, class_names=class_names)
    task_id = None
    if getattr(ctx, 'task', None) is not None:
        task_id = getattr(ctx.task, 'id', None)
        if task_id:
            task_id = str(task_id)
    versions = resolve_version_props(cfg, clearml_enabled=clearml_enabled)
    artifact_payload = _persist_ensemble_artifacts(ctx=ctx, cfg=cfg, versions=versions, selected=selected, skipped=skipped, ref_values=ref_values, preprocess_variant=preprocess_variant, method=method, method_used=method_used, selection_metric=selection_metric, selection_direction=selection_direction, top_k=top_k, primary_metric=primary_metric, primary_metric_source=primary_metric_source, primary_direction=primary_direction, task_type=task_type, n_classes=n_classes, weights=weights, ensemble_meta=ensemble_meta, ensemble_model=ensemble_model, preprocess_bundle=preprocess_bundle, metrics_holdout=metrics_holdout, best_score=best_score, task_id=task_id)
    included = artifact_payload['included']
    spec_path = artifact_payload['spec_path']
    metrics_path = artifact_payload['metrics_path']
    model_bundle_path = artifact_payload['model_bundle_path']
    model_id = artifact_payload['model_id']
    train_task_ids = artifact_payload['train_task_ids']
    preprocess_task_ids = artifact_payload['preprocess_task_ids']
    pipeline_task_id = _normalize_str(_cfg_value(cfg, 'run.clearml.pipeline_task_id'))
    (registry_model_id, registry_status, registry_error) = _register_ensemble_model_if_enabled(ctx=ctx, cfg=cfg, clearml_enabled=clearml_enabled, ref_values=ref_values, preprocess_variant=preprocess_variant, task_type=task_type, method_used=method_used, task_id=task_id, pipeline_task_id=pipeline_task_id, train_task_ids=train_task_ids, preprocess_task_ids=preprocess_task_ids, model_bundle_path=model_bundle_path)
    connect_train_ensemble(ctx, cfg, processed_dataset_id=ref_values.get('processed_dataset_id'), task_type=task_type, primary_metric=primary_metric, method=method, top_k=top_k)
    _log_ensemble_to_clearml(ctx=ctx, clearml_enabled=clearml_enabled, spec_path=spec_path, metrics_path=metrics_path, model_bundle_path=model_bundle_path, ref_values=ref_values, model_id=model_id, primary_metric=primary_metric, best_score=best_score, task_type=task_type, n_classes=n_classes, registry_model_id=registry_model_id, registry_status=registry_status, metrics_holdout=metrics_holdout, included=included, skipped=skipped, weights=weights, selected=selected, debug_sample=debug_sample, viz_settings=viz_settings, y_true=y_true, y_pred=y_pred, y_proba=y_proba, class_names=class_names, residuals_plot_path=residuals_plot_path, confusion_plot_path=confusion_plot_path, roc_plot_path=roc_plot_path)
    _write_out_and_manifest(ctx=ctx, cfg=cfg, versions=versions, ref_values=ref_values, preprocess_variant=preprocess_variant, method=method, top_k=top_k, selection_metric=selection_metric, primary_metric=primary_metric, primary_direction=primary_direction, task_type=task_type, primary_metric_source=primary_metric_source, model_id=model_id, best_score=best_score, task_id=task_id, pipeline_task_id=pipeline_task_id, preprocess_task_ids=preprocess_task_ids, train_task_ids=train_task_ids, n_classes=n_classes, registry_model_id=registry_model_id, registry_status=registry_status, registry_error=registry_error)
