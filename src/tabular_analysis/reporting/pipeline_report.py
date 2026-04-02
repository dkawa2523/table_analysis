from __future__ import annotations
from ..common.config_utils import normalize_str as _normalize_str, to_float as _to_float
from ..common.model_reference import resolve_preferred_infer_reference
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from ..platform_adapter_clearml_env import is_clearml_enabled
from ..platform_adapter_task import get_task_artifact_local_copy, resolve_clearml_task_url
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError, json.JSONDecodeError)
@dataclass
class PipelineReportBundle:
    markdown: str
    payload: dict[str, Any]
    links: dict[str, Any]
def _format_value(value: Any, *, fallback: str='n/a') -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return f'{value:.6g}'
    text = str(value).strip()
    return text if text else fallback
def _format_rate(value: Any, *, fallback: str='n/a') -> str:
    try:
        return f'{float(value):.1%}'
    except (TypeError, ValueError, OverflowError):
        return fallback
def _shorten_path(value: str, *, keep: int=2) -> str:
    if '/' not in value and '\\' not in value:
        return value
    path = Path(value)
    parts = path.parts
    if len(parts) <= keep:
        return value
    return '.../' + '/'.join(parts[-keep:])
def _md_escape(text: str) -> str:
    return text.replace('|', '\\|').replace('\n', ' ')
def _safe_load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None
def _safe_load_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                rows.append(dict(row))
    except (OSError, TypeError, ValueError, csv.Error):
        return []
    return rows
def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
class _ArtifactResolver:
    def __init__(self, cfg: Any | None):
        self._cfg = cfg
        self._clearml_enabled = bool(cfg) and is_clearml_enabled(cfg)
    def resolve(self, ref: Mapping[str, Any] | None, name: str) -> Path | None:
        if not ref:
            return None
        run_dir = _normalize_str(ref.get('run_dir'))
        if run_dir:
            path = Path(run_dir) / name
            if path.exists():
                return path
        if self._clearml_enabled:
            task_id = _normalize_str(ref.get('task_id') or ref.get('train_task_id'))
            if task_id:
                try:
                    return get_task_artifact_local_copy(self._cfg, task_id, name)
                except _RECOVERABLE_ERRORS:
                    return None
        return None
def _build_model_rows_from_train_refs(train_refs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (idx, ref) in enumerate(train_refs, start=1):
        rows.append({'rank': idx, 'model_variant': ref.get('model_variant'), 'preprocess_variant': ref.get('preprocess_variant'), 'primary_metric': ref.get('primary_metric'), 'best_score': ref.get('best_score'), 'model_id': ref.get('model_id'), 'train_task_ref': ref.get('train_task_id') or ref.get('task_id')})
    return rows
def _append_models_table(lines: list[str], rows: list[Mapping[str, Any]], max_models: int) -> None:
    if not rows:
        lines.append('- No leaderboard or train results available.')
        return
    lines.extend(['', '| Rank | Model | Preprocess | Metric | Score | Model ID |', '| --- | --- | --- | --- | --- | --- |'])
    for row in rows[:max_models]:
        model_id = _normalize_str(row.get('model_id'))
        model_id_display = _shorten_path(model_id) if model_id else 'n/a'
        items = [_format_value(row.get('rank')), _format_value(row.get('model_variant')), _format_value(row.get('preprocess_variant')), _format_value(row.get('primary_metric')), _format_value(row.get('best_score')), model_id_display]
        lines.append('| ' + ' | '.join((_md_escape(str(item)) for item in items)) + ' |')
def _normalize_top_model_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized['rank'] = _coerce_int(row.get('rank'))
    normalized['best_score'] = _to_float(row.get('best_score'))
    normalized['primary_metric_ci_low'] = _to_float(row.get('primary_metric_ci_low'))
    normalized['primary_metric_ci_mid'] = _to_float(row.get('primary_metric_ci_mid'))
    normalized['primary_metric_ci_high'] = _to_float(row.get('primary_metric_ci_high'))
    for key in ('primary_metric', 'task_type', 'model_id', 'preprocess_variant', 'model_variant', 'train_task_ref', 'processed_dataset_id', 'split_hash'):
        if key in normalized:
            normalized[key] = _normalize_str(row.get(key))
    return normalized
def _extract_notes(recommended: Mapping[str, Any]) -> dict[str, Any]:
    notes: dict[str, Any] = {}
    for key in ('thresholding', 'calibration', 'imbalance', 'uncertainty'):
        value = recommended.get(key)
        if isinstance(value, Mapping):
            notes[key] = dict(value)
        elif value is not None:
            notes[key] = value
        else:
            notes[key] = None
    return notes
def _summarize_data_quality(data_quality: Mapping[str, Any]) -> dict[str, Any]:
    return {'rows_total': data_quality.get('rows_total'), 'rows_scanned': data_quality.get('rows_scanned'), 'scan_sampled': data_quality.get('scan_sampled'), 'duplicates_count': data_quality.get('duplicates_count'), 'duplicates_rate': data_quality.get('duplicates_rate'), 'missing_top': data_quality.get('missing_top') or [], 'leak_suspects': data_quality.get('leak_suspects') or []}
def _summarize_data_quality_from_out(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {'rows_scanned': summary.get('rows'), 'duplicates_count': summary.get('duplicates_count'), 'missing_columns': summary.get('missing_columns'), 'leak_suspect_count': summary.get('leak_suspect_count') or summary.get('leak_suspects')}
def _collect_report_payload(pipeline_run: Mapping[str, Any], *, cfg: Any | None, max_models: int) -> dict[str, Any]:
    resolver = _ArtifactResolver(cfg)
    grid_run_id = pipeline_run.get('grid_run_id')
    planned_jobs = pipeline_run.get('planned_jobs')
    executed_jobs = pipeline_run.get('executed_jobs')
    skipped_jobs = pipeline_run.get('skipped_due_to_policy')
    plan_only = pipeline_run.get('plan_only')
    grid = pipeline_run.get('grid', {}) if isinstance(pipeline_run, Mapping) else {}
    preprocess_variants = list(grid.get('preprocess_variants') or [])
    model_variants = list(grid.get('model_variants') or [])
    hpo_cfg = grid.get('hpo', {}) if isinstance(grid, Mapping) else {}
    hpo_enabled = bool(hpo_cfg.get('enabled'))
    preprocess_refs = list(pipeline_run.get('preprocess_ref') or [])
    preprocess_ref = preprocess_refs[0] if preprocess_refs else None
    preprocess_out = _safe_load_json(resolver.resolve(preprocess_ref, 'out.json')) or {}
    preprocess_schema = _safe_load_json(resolver.resolve(preprocess_ref, 'schema.json')) or {}
    split_payload = _safe_load_json(resolver.resolve(preprocess_ref, 'split.json')) or {}
    recipe_payload = _safe_load_json(resolver.resolve(preprocess_ref, 'recipe.json')) or {}
    dataset_register_ref = pipeline_run.get('dataset_register_ref')
    dataset_out = _safe_load_json(resolver.resolve(dataset_register_ref, 'out.json')) or {}
    data_quality_payload = _safe_load_json(resolver.resolve(dataset_register_ref, 'data_quality.json')) or {}
    rows = preprocess_schema.get('rows') or dataset_out.get('raw_schema', {}).get('rows')
    feature_cols = preprocess_schema.get('columns') or dataset_out.get('raw_schema', {}).get('columns')
    target_column = preprocess_schema.get('target_column') or dataset_out.get('raw_schema', {}).get('target_column')
    id_columns = preprocess_schema.get('id_columns') or []
    drop_columns = preprocess_schema.get('drop_columns') or []
    preprocess_variant = (_normalize_str(preprocess_ref.get('preprocess_variant')) if preprocess_ref else None) or _normalize_str(recipe_payload.get('variant', {}).get('name'))
    split_hash = _normalize_str(preprocess_out.get('split_hash')) or (_normalize_str(preprocess_ref.get('split_hash')) if preprocess_ref else None)
    recipe_hash = _normalize_str(preprocess_out.get('recipe_hash')) or (_normalize_str(preprocess_ref.get('recipe_hash')) if preprocess_ref else None)
    leaderboard_ref = pipeline_run.get('leaderboard_ref')
    leaderboard_csv = resolver.resolve(leaderboard_ref, 'leaderboard.csv')
    leaderboard_rows = _safe_load_csv(leaderboard_csv)
    leaderboard_out = _safe_load_json(resolver.resolve(leaderboard_ref, 'out.json')) or {}
    recommendation = _safe_load_json(resolver.resolve(leaderboard_ref, 'recommendation.json')) or {}
    decision_summary = _safe_load_json(resolver.resolve(leaderboard_ref, 'decision_summary.json')) or {}
    recommended_detail: Mapping[str, Any] = {}
    if isinstance(decision_summary.get('recommended'), Mapping):
        recommended_detail = decision_summary['recommended']
    preferred_reference = resolve_preferred_infer_reference(recommendation, recommended_detail, leaderboard_out)
    recommended_model_id = _normalize_str(preferred_reference.get('infer_model_id') or recommendation.get('recommended_model_id') or leaderboard_out.get('recommended_model_id'))
    recommended_registry_model_id = _normalize_str(recommended_detail.get('registry_model_id') or recommendation.get('recommended_registry_model_id') or leaderboard_out.get('recommended_registry_model_id'))
    recommended_metric = _normalize_str(recommended_detail.get('primary_metric') or recommendation.get('recommended_primary_metric') or leaderboard_out.get('recommended_primary_metric'))
    recommended_score = recommended_detail.get('best_score')
    if recommended_score is None:
        recommended_score = recommendation.get('recommended_best_score')
    if recommended_score is None:
        recommended_score = leaderboard_out.get('recommended_best_score')
    recommended_train_ref = _normalize_str(recommended_detail.get('train_task_ref') or recommendation.get('recommended_train_task_ref') or leaderboard_out.get('recommended_train_task_ref') or preferred_reference.get('infer_train_task_id') or leaderboard_out.get('recommended_train_task_id'))
    thresholding = recommended_detail.get('thresholding') if isinstance(recommended_detail.get('thresholding'), Mapping) else None
    recommended_threshold = thresholding.get('best_threshold') if isinstance(thresholding, Mapping) else None
    if recommended_threshold is None and recommended_train_ref:
        ref: dict[str, Any]
        if Path(recommended_train_ref).exists():
            ref = {'run_dir': recommended_train_ref}
        else:
            ref = {'task_id': recommended_train_ref}
        train_out = _safe_load_json(resolver.resolve(ref, 'out.json')) or {}
        recommended_threshold = train_out.get('best_threshold')
    comparability: dict[str, Any] = {}
    if isinstance(decision_summary.get('comparability'), Mapping):
        comparability = dict(decision_summary.get('comparability') or {})
    if decision_summary.get('excluded_count') is not None:
        comparability['excluded_count'] = decision_summary.get('excluded_count')
    if decision_summary.get('warning_count') is not None:
        comparability['warning_count'] = decision_summary.get('warning_count')
    if not comparability:
        comparability = {'processed_dataset_id': _normalize_str(preprocess_ref.get('processed_dataset_id') if preprocess_ref else None), 'split_hash': split_hash, 'recipe_hash': recipe_hash, 'primary_metric': recommended_metric}
    direction = _normalize_str(comparability.get('direction'))
    task_type = _normalize_str(recommended_detail.get('task_type') or comparability.get('task_type'))
    top_models: list[dict[str, Any]] = []
    if isinstance(decision_summary.get('top_models'), list):
        top_models = [_normalize_top_model_row(item) for item in decision_summary.get('top_models') or [] if isinstance(item, Mapping)]
    if not top_models and leaderboard_rows:
        top_models = [_normalize_top_model_row(item) for item in leaderboard_rows]
    train_refs = list(pipeline_run.get('train_refs') or [])
    if not top_models and train_refs:
        top_models = [_normalize_top_model_row(item) for item in _build_model_rows_from_train_refs(train_refs)]
    models_tried = len(leaderboard_rows) if leaderboard_rows else len(train_refs) if train_refs else len(top_models)
    has_infer_target = bool(preferred_reference.get('infer_model_id') or preferred_reference.get('infer_train_task_id'))
    status = 'ready' if has_infer_target else 'incomplete'
    if data_quality_payload:
        data_quality = _summarize_data_quality(data_quality_payload)
    else:
        summary = dataset_out.get('data_quality_summary') or {}
        data_quality = _summarize_data_quality_from_out(summary) if summary else {}
    notes = _extract_notes(recommended_detail)
    rationale = None
    if has_infer_target:
        if recommended_metric and direction:
            rationale = f'Top-ranked by {recommended_metric} ({direction}).'
        elif recommended_metric:
            rationale = f'Top-ranked by {recommended_metric}.'
        else:
            rationale = 'Top-ranked model from leaderboard.'
        if comparability.get('require_comparable') is not None:
            rationale += f" require_comparable={comparability.get('require_comparable')}."
        if comparability.get('excluded_count'):
            rationale += f" excluded_count={comparability.get('excluded_count')}."
    actions: list[str] = []
    if not has_infer_target:
        actions.append('Wait for training/leaderboard to finish, then regenerate the report.')
    if not leaderboard_rows and train_refs:
        actions.append('Run leaderboard to select the best model explicitly.')
    if len(model_variants) <= 1:
        actions.append('Try additional model variants for a stronger baseline.')
    if len(preprocess_variants) <= 1:
        actions.append('Try alternative preprocessing variants for robustness.')
    if not hpo_enabled and model_variants:
        actions.append('Enable pipeline.hpo to explore parameter grids on promising models.')
    if not actions:
        actions.append('Validate the recommended model on a fresh holdout set.')
    actions = actions[:3]
    return {'report_version': 1, 'grid_run_id': grid_run_id, 'status': status, 'summary': {'recommended_model_id': recommended_model_id, 'recommended_registry_model_id': recommended_registry_model_id, 'infer_model_id': preferred_reference.get('infer_model_id'), 'infer_train_task_id': preferred_reference.get('infer_train_task_id'), 'reference_kind': preferred_reference.get('reference_kind'), 'primary_metric': recommended_metric, 'best_score': recommended_score, 'models_tried': models_tried, 'train_task_ref': recommended_train_ref, 'planned_jobs': planned_jobs, 'executed_jobs': executed_jobs, 'skipped_due_to_policy': skipped_jobs, 'plan_only': plan_only, 'recommendation_rationale': rationale}, 'dataset': {'raw_dataset_id': dataset_out.get('raw_dataset_id'), 'processed_dataset_id': _normalize_str(preprocess_ref.get('processed_dataset_id') if preprocess_ref else None) or comparability.get('processed_dataset_id'), 'rows': rows, 'feature_columns': feature_cols, 'target_column': target_column, 'id_columns': id_columns, 'drop_columns': drop_columns}, 'data_quality': data_quality, 'split': {'preprocess_variant': preprocess_variant, 'strategy': split_payload.get('strategy'), 'test_size': split_payload.get('test_size'), 'seed': split_payload.get('seed'), 'group_column': split_payload.get('group_column'), 'time_column': split_payload.get('time_column'), 'split_hash': split_hash, 'recipe_hash': recipe_hash}, 'comparability': comparability, 'top_models': top_models[:max_models], 'recommendation': {'model_id': recommended_model_id, 'registry_model_id': recommended_registry_model_id, 'infer_model_id': preferred_reference.get('infer_model_id'), 'infer_train_task_id': preferred_reference.get('infer_train_task_id'), 'reference_kind': preferred_reference.get('reference_kind'), 'train_task_ref': recommended_train_ref, 'primary_metric': recommended_metric, 'best_score': recommended_score, 'direction': direction, 'primary_metric_ci': recommended_detail.get('primary_metric_ci'), 'task_type': task_type, 'n_classes': recommended_detail.get('n_classes'), 'threshold': recommended_threshold, 'thresholding': notes.get('thresholding'), 'calibration': notes.get('calibration'), 'imbalance': notes.get('imbalance'), 'uncertainty': notes.get('uncertainty')}, 'notes': notes, 'next_actions': actions}
def _render_report_markdown(payload: Mapping[str, Any], *, max_models: int) -> str:
    summary = payload.get('summary') or {}
    dataset = payload.get('dataset') or {}
    split = payload.get('split') or {}
    comparability = payload.get('comparability') or {}
    recommendation = payload.get('recommendation') or {}
    notes = payload.get('notes') or {}
    data_quality = payload.get('data_quality') or {}
    lines: list[str] = ['# Pipeline Summary', '', '## Conclusion', f"- grid_run_id: {_format_value(payload.get('grid_run_id'))}", f"- recommended_model_id: {_format_value(summary.get('recommended_model_id'))}", f"- infer_model_id: {_format_value(summary.get('infer_model_id'))}", f"- infer_train_task_id: {_format_value(summary.get('infer_train_task_id'))}", f"- primary_metric: {_format_value(summary.get('primary_metric'))}", f"- best_score: {_format_value(summary.get('best_score'))}", f"- status: {_format_value(payload.get('status'))}", f"- models_tried: {_format_value(summary.get('models_tried'))}"]
    if summary.get('pipeline_url'):
        lines.append(f"- pipeline_url: {_format_value(summary.get('pipeline_url'))}")
    if summary.get('planned_jobs') is not None:
        lines.append(f"- planned_jobs: {_format_value(summary.get('planned_jobs'))}")
    if summary.get('executed_jobs') is not None:
        lines.append(f"- executed_jobs: {_format_value(summary.get('executed_jobs'))}")
    if summary.get('skipped_due_to_policy') is not None:
        lines.append(f"- skipped_due_to_policy: {_format_value(summary.get('skipped_due_to_policy'))}")
    if summary.get('plan_only'):
        lines.append('- plan_only: true')
    if summary.get('train_task_ref'):
        lines.append(f"- train_task_ref: {_format_value(summary.get('train_task_ref'))}")
    lines.extend(['', '## Data Overview', f"- raw_dataset_id: {_format_value(dataset.get('raw_dataset_id'))}", f"- processed_dataset_id: {_format_value(dataset.get('processed_dataset_id'))}", f"- rows: {_format_value(dataset.get('rows'))}", f"- feature_columns: {_format_value(dataset.get('feature_columns'))}", f"- target_column: {_format_value(dataset.get('target_column'))}"])
    if dataset.get('id_columns'):
        lines.append(f"- id_columns: {_format_value(dataset.get('id_columns'))}")
    if dataset.get('drop_columns'):
        lines.append(f"- drop_columns: {_format_value(dataset.get('drop_columns'))}")
    lines.extend(['', '## Data Quality'])
    if data_quality:
        rows_scanned = data_quality.get('rows_scanned')
        rows_total = data_quality.get('rows_total')
        sampled = data_quality.get('scan_sampled')
        if sampled and rows_total:
            lines.append(f'- sampled: true ({_format_value(rows_scanned)}/{_format_value(rows_total)})')
        else:
            lines.append(f'- rows_scanned: {_format_value(rows_scanned)}')
        duplicates_count = data_quality.get('duplicates_count')
        duplicates_rate = data_quality.get('duplicates_rate')
        if duplicates_count is not None or duplicates_rate is not None:
            lines.append(f'- duplicates: {_format_value(duplicates_count)} ({_format_rate(duplicates_rate)})')
        missing_top = data_quality.get('missing_top') or []
        if missing_top:
            items = []
            for item in missing_top[:5]:
                if not isinstance(item, Mapping):
                    continue
                col = _normalize_str(item.get('column'))
                rate = item.get('missing_rate')
                if not col:
                    continue
                items.append(f'{_md_escape(col)}({_format_rate(rate)})')
            lines.append(f"- missing_top: {(', '.join(items) if items else 'n/a')}")
        elif data_quality.get('missing_columns') is not None:
            lines.append(f"- missing_columns: {_format_value(data_quality.get('missing_columns'))}")
        else:
            lines.append('- missing_top: n/a')
        leak_suspects = data_quality.get('leak_suspects') or []
        if leak_suspects:
            items = []
            for suspect in leak_suspects[:5]:
                if not isinstance(suspect, Mapping):
                    continue
                col = _normalize_str(suspect.get('column'))
                reason = _normalize_str(suspect.get('reason'))
                severity = _normalize_str(suspect.get('severity'))
                if not col:
                    continue
                label = f"{_md_escape(col)}:{reason or 'suspect'}"
                if severity:
                    label += f'({severity})'
                items.append(label)
            lines.append(f"- leak_suspects: {(', '.join(items) if items else 'n/a')}")
        elif data_quality.get('leak_suspect_count') is not None:
            lines.append(f"- leak_suspects: {_format_value(data_quality.get('leak_suspect_count'))}")
        else:
            lines.append('- leak_suspects: n/a')
    else:
        lines.append('- data_quality: n/a')
    lines.extend(['', '## Comparability', f"- require_comparable: {_format_value(comparability.get('require_comparable'))}", f"- processed_dataset_id: {_format_value(comparability.get('processed_dataset_id'))}", f"- split_hash: {_format_value(comparability.get('split_hash'))}", f"- recipe_hash: {_format_value(comparability.get('recipe_hash'))}", f"- primary_metric: {_format_value(comparability.get('primary_metric'))}", f"- direction: {_format_value(comparability.get('direction'))}", f"- task_type: {_format_value(comparability.get('task_type'))}", f"- seed: {_format_value(comparability.get('seed'))}"])
    if comparability.get('excluded_count') is not None:
        lines.append(f"- excluded_count: {_format_value(comparability.get('excluded_count'))}")
    if comparability.get('warning_count') is not None:
        lines.append(f"- warning_count: {_format_value(comparability.get('warning_count'))}")
    lines.extend(['', '## Split / Recipe / Hashes', f"- preprocess_variant: {_format_value(split.get('preprocess_variant'))}", f"- split.strategy: {_format_value(split.get('strategy'))}", f"- split.test_size: {_format_value(split.get('test_size'))}", f"- split.seed: {_format_value(split.get('seed'))}", f"- split_hash: {_format_value(split.get('split_hash'))}", f"- recipe_hash: {_format_value(split.get('recipe_hash'))}"])
    if split.get('group_column'):
        lines.append(f"- split.group_column: {_format_value(split.get('group_column'))}")
    if split.get('time_column'):
        lines.append(f"- split.time_column: {_format_value(split.get('time_column'))}")
    lines.extend(['', f'## Models Tried (Top {max_models})'])
    _append_models_table(lines, payload.get('top_models') or [], max_models)
    lines.extend(['', '## Recommendation', f"- model_id: {_format_value(recommendation.get('model_id'))}", f"- registry_model_id: {_format_value(recommendation.get('registry_model_id'))}", f"- infer_model_id: {_format_value(recommendation.get('infer_model_id'))}", f"- infer_train_task_id: {_format_value(recommendation.get('infer_train_task_id'))}", f"- primary_metric: {_format_value(recommendation.get('primary_metric'))}", f"- best_score: {_format_value(recommendation.get('best_score'))}"])
    if recommendation.get('direction'):
        lines.append(f"- direction: {_format_value(recommendation.get('direction'))}")
    if recommendation.get('primary_metric_ci') is not None:
        lines.append(f"- primary_metric_ci: {_format_value(recommendation.get('primary_metric_ci'))}")
    if recommendation.get('threshold') is not None:
        lines.append(f"- threshold: {_format_value(recommendation.get('threshold'))}")
    if recommendation.get('train_task_ref'):
        lines.append(f"- train_task_ref: {_format_value(recommendation.get('train_task_ref'))}")
    rationale = summary.get('recommendation_rationale')
    if rationale:
        lines.append(f'- rationale: {_format_value(rationale)}')
    lines.extend(['', '## Notes'])
    thresholding = notes.get('thresholding') or {}
    if isinstance(thresholding, Mapping) and thresholding.get('best_threshold') is not None:
        lines.append('- thresholding: enabled metric={metric} best_threshold={thr} score={score}'.format(metric=thresholding.get('metric') or 'unknown', thr=_format_value(thresholding.get('best_threshold')), score=_format_value(thresholding.get('score'))))
    else:
        lines.append('- thresholding: disabled')
    calibration = notes.get('calibration') or {}
    if isinstance(calibration, Mapping) and calibration.get('enabled'):
        lines.append('- calibration: enabled method={method} mode={mode}'.format(method=calibration.get('method') or 'unknown', mode=calibration.get('mode') or 'unknown'))
    else:
        lines.append('- calibration: disabled')
    uncertainty = notes.get('uncertainty') or {}
    if isinstance(uncertainty, Mapping) and uncertainty.get('enabled'):
        lines.append('- uncertainty: enabled method={method} alpha={alpha} q={q}'.format(method=uncertainty.get('method') or 'unknown', alpha=_format_value(uncertainty.get('alpha')), q=_format_value(uncertainty.get('q'))))
    else:
        lines.append('- uncertainty: disabled')
    imbalance = notes.get('imbalance') or {}
    if isinstance(imbalance, Mapping) and imbalance.get('enabled'):
        lines.append('- imbalance_handling: enabled strategy={strategy} applied={applied}'.format(strategy=imbalance.get('strategy') or 'unknown', applied=imbalance.get('applied')))
    else:
        lines.append('- imbalance_handling: disabled')
    lines.extend(['', '## Next Actions'])
    for action in payload.get('next_actions') or []:
        lines.append(f'- {action}')
    return '\n'.join(lines) + '\n'
def _build_link_entry(cfg: Any | None, run_dir: str | None, task_id: str | None, *, extra: Mapping[str, Any] | None=None) -> dict[str, Any] | None:
    entry: dict[str, Any] = {}
    if run_dir:
        entry['run_dir'] = str(run_dir)
    if task_id:
        entry['task_id'] = str(task_id)
        clearml_url = resolve_clearml_task_url(cfg, str(task_id)) if cfg is not None else None
        if clearml_url:
            entry['clearml_url'] = clearml_url
    if extra:
        entry.update(dict(extra))
    return entry or None
def build_pipeline_report_links(pipeline_run: Mapping[str, Any], *, cfg: Any | None=None, pipeline_run_dir: Path | None=None, pipeline_task_id: str | None=None) -> dict[str, Any]:
    grid_run_id = pipeline_run.get('grid_run_id')
    links: dict[str, Any] = {'grid_run_id': grid_run_id}
    pipeline_entry = _build_link_entry(cfg, str(pipeline_run_dir) if pipeline_run_dir else None, pipeline_task_id)
    links['pipeline'] = pipeline_entry
    dataset_ref = pipeline_run.get('dataset_register_ref') or {}
    dataset_entry = None
    if isinstance(dataset_ref, Mapping):
        dataset_entry = _build_link_entry(cfg, _normalize_str(dataset_ref.get('run_dir')), _normalize_str(dataset_ref.get('task_id')))
    links['dataset_register'] = dataset_entry
    preprocess_entries: list[dict[str, Any]] = []
    for ref in pipeline_run.get('preprocess_ref') or []:
        if not isinstance(ref, Mapping):
            continue
        entry = _build_link_entry(cfg, _normalize_str(ref.get('run_dir')), _normalize_str(ref.get('task_id')), extra={'preprocess_variant': _normalize_str(ref.get('preprocess_variant'))})
        if entry:
            preprocess_entries.append(entry)
    links['preprocess'] = preprocess_entries
    train_entries: list[dict[str, Any]] = []
    for ref in pipeline_run.get('train_refs') or []:
        if not isinstance(ref, Mapping):
            continue
        entry = _build_link_entry(cfg, _normalize_str(ref.get('run_dir')), _normalize_str(ref.get('task_id') or ref.get('train_task_id')), extra={'preprocess_variant': _normalize_str(ref.get('preprocess_variant')), 'model_variant': _normalize_str(ref.get('model_variant'))})
        if entry:
            train_entries.append(entry)
    links['train'] = train_entries
    leaderboard_ref = pipeline_run.get('leaderboard_ref') or {}
    leaderboard_entry = None
    if isinstance(leaderboard_ref, Mapping):
        leaderboard_entry = _build_link_entry(cfg, _normalize_str(leaderboard_ref.get('run_dir')), _normalize_str(leaderboard_ref.get('task_id')))
    links['leaderboard'] = leaderboard_entry
    infer_ref = pipeline_run.get('infer_ref') or {}
    infer_entry = None
    if isinstance(infer_ref, Mapping):
        infer_entry = _build_link_entry(cfg, _normalize_str(infer_ref.get('run_dir')), _normalize_str(infer_ref.get('task_id')))
    links['infer'] = infer_entry
    return links
def build_pipeline_report_bundle(pipeline_run: Mapping[str, Any], *, cfg: Any | None=None, max_models: int=5, pipeline_run_dir: Path | None=None, pipeline_task_id: str | None=None) -> PipelineReportBundle:
    links = build_pipeline_report_links(pipeline_run, cfg=cfg, pipeline_run_dir=pipeline_run_dir, pipeline_task_id=pipeline_task_id)
    payload = dict(_collect_report_payload(pipeline_run, cfg=cfg, max_models=max_models))
    payload['links'] = links
    pipeline_entry = links.get('pipeline') if isinstance(links, Mapping) else None
    if isinstance(pipeline_entry, Mapping):
        pipeline_url = pipeline_entry.get('clearml_url')
        if pipeline_url:
            summary = dict(payload.get('summary') or {})
            summary['pipeline_url'] = pipeline_url
            payload['summary'] = summary
    markdown = _render_report_markdown(payload, max_models=max_models)
    return PipelineReportBundle(markdown=markdown, payload=payload, links=links)
