from __future__ import annotations
from ..common.collection_utils import to_container as _to_container
from ..common.config_utils import cfg_value as _cfg_value
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from ..ops.alerting import emit_alert
from ..platform_adapter_artifacts import upload_artifact
from ..platform_adapter_task import TaskContext, update_task_properties
from ..quality.data_quality import compute_data_quality, render_data_quality_markdown, summarize_data_quality
_DEFAULT_THRESHOLDS = {'missing_rate_warn': 0.2, 'missing_rate_fail': 0.5, 'duplicates_rate_warn': 0.01, 'duplicates_rate_fail': 0.05, 'constant_columns_warn': 1, 'constant_columns_fail': 5, 'mixed_type_columns_warn': 1, 'mixed_type_columns_fail': 5, 'high_cardinality_ratio_warn': 0.8, 'high_cardinality_ratio_fail': 1.01, 'high_cardinality_min_unique': 20, 'high_cardinality_count_fail': 9999, 'id_like_ratio_warn': 0.98, 'id_like_ratio_fail': 1.01, 'id_like_min_unique': 20, 'leak_near_match_warn': 0.98, 'leak_correlation_warn': 0.98, 'leak_mapping_accuracy_warn': 0.98}
_DEFAULT_NAME_PATTERNS = {'leak_suspect': ['target', 'label', 'outcome', 'response'], 'id_suspect': ['id', 'uuid', 'guid']}
_NUMERIC_COERCE_ERRORS = (TypeError, ValueError)
def _normalize_mode(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in ('off', 'disable', 'disabled', 'none', 'no'):
        return 'off'
    if key in ('fail', 'error', 'strict'):
        return 'fail'
    if key in ('warn', 'warning'):
        return 'warn'
    return None
def _merge_thresholds(raw: Any) -> dict[str, Any]:
    merged = dict(_DEFAULT_THRESHOLDS)
    payload = _to_container(raw)
    if isinstance(payload, Mapping):
        for (key, value) in payload.items():
            if value is None:
                continue
            merged[str(key)] = value
    return merged
def _merge_patterns(raw: Any) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: list(values) for (key, values) in _DEFAULT_NAME_PATTERNS.items()}
    payload = _to_container(raw)
    if isinstance(payload, Mapping):
        for (key, values) in payload.items():
            if values is None:
                continue
            if isinstance(values, (str, bytes)):
                seq: Iterable[Any] = [values]
            else:
                seq = values
            items = [str(item).strip().lower() for item in seq if str(item).strip()]
            if not items:
                continue
            existing = merged.get(str(key), [])
            for item in items:
                if item not in existing:
                    existing.append(item)
            merged[str(key)] = existing
    return merged
def resolve_quality_settings(cfg: Any) -> dict[str, Any]:
    raw_mode = _normalize_mode(_cfg_value(cfg, 'data.quality.mode', None))
    if raw_mode is None:
        mode = 'warn'
    else:
        mode = raw_mode
    enabled = _cfg_value(cfg, 'data.quality.enabled', None)
    enabled = bool(enabled) if enabled is not None else True
    if mode == 'off':
        enabled = False
    max_rows_scan = _cfg_value(cfg, 'data.quality.max_rows_scan', 50000)
    try:
        max_rows_scan = int(max_rows_scan)
    except _NUMERIC_COERCE_ERRORS:
        max_rows_scan = 50000
    thresholds = _merge_thresholds(_cfg_value(cfg, 'data.quality.thresholds', None))
    name_patterns = _merge_patterns(_cfg_value(cfg, 'data.quality.name_patterns', None))
    return {'mode': mode, 'enabled': enabled, 'max_rows_scan': max_rows_scan, 'thresholds': thresholds, 'name_patterns': name_patterns}
def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except _NUMERIC_COERCE_ERRORS:
        return None
def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except _NUMERIC_COERCE_ERRORS:
        return None
def _add_issue(issues: list[dict[str, Any]], *, issue_type: str, severity: str, count: int | None=None, detail: Mapping[str, Any] | None=None) -> None:
    if count is not None and count <= 0:
        return
    payload: dict[str, Any] = {'type': issue_type, 'severity': severity}
    if count is not None:
        payload['count'] = int(count)
    if detail:
        payload['detail'] = dict(detail)
    issues.append(payload)
def _limit_list(values: Iterable[Any], limit: int=10) -> list[Any]:
    items = list(values)
    if limit <= 0:
        return []
    return items[:limit]
def evaluate_quality(payload: dict[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    missing_rate = _safe_float(payload.get('missing_rate_total'))
    missing_warn = _safe_float(thresholds.get('missing_rate_warn'))
    missing_fail = _safe_float(thresholds.get('missing_rate_fail'))
    if missing_rate is not None:
        if missing_fail is not None and missing_rate >= missing_fail:
            _add_issue(issues, issue_type='missing_rate', severity='fail', count=1, detail={'missing_rate_total': missing_rate, 'fail_threshold': missing_fail})
        elif missing_warn is not None and missing_rate >= missing_warn:
            _add_issue(issues, issue_type='missing_rate', severity='warn', count=1, detail={'missing_rate_total': missing_rate, 'warn_threshold': missing_warn})
    duplicates_rate = _safe_float(payload.get('duplicates_rate'))
    duplicates_count = _safe_int(payload.get('duplicates_count'))
    rows_scanned = _safe_int(payload.get('rows_scanned')) or 0
    if duplicates_rate is None and duplicates_count is not None and rows_scanned:
        duplicates_rate = duplicates_count / rows_scanned
    dup_warn = _safe_float(thresholds.get('duplicates_rate_warn'))
    dup_fail = _safe_float(thresholds.get('duplicates_rate_fail'))
    if duplicates_rate is not None:
        if dup_fail is not None and duplicates_rate >= dup_fail:
            _add_issue(issues, issue_type='duplicates', severity='fail', count=duplicates_count or 1, detail={'duplicates_rate': duplicates_rate, 'fail_threshold': dup_fail})
        elif dup_warn is not None and duplicates_rate >= dup_warn:
            _add_issue(issues, issue_type='duplicates', severity='warn', count=duplicates_count or 1, detail={'duplicates_rate': duplicates_rate, 'warn_threshold': dup_warn})
    constant_cols = payload.get('constant_columns') or []
    const_count = len(constant_cols)
    const_warn = _safe_int(thresholds.get('constant_columns_warn'))
    const_fail = _safe_int(thresholds.get('constant_columns_fail'))
    if const_count > 0:
        severity = 'warn'
        if const_fail is not None and const_count >= const_fail:
            severity = 'fail'
        elif const_warn is not None and const_count < const_warn:
            severity = 'warn'
        _add_issue(issues, issue_type='constant_columns', severity=severity, count=const_count, detail={'columns': _limit_list(constant_cols)})
    mixed_type_cols = payload.get('mixed_type_columns') or []
    mixed_type_details = payload.get('mixed_type_details') or mixed_type_cols
    mixed_count = len(mixed_type_cols)
    mixed_warn = _safe_int(thresholds.get('mixed_type_columns_warn'))
    mixed_fail = _safe_int(thresholds.get('mixed_type_columns_fail'))
    if mixed_count > 0:
        severity = 'warn'
        if mixed_fail is not None and mixed_count >= mixed_fail:
            severity = 'fail'
        elif mixed_warn is not None and mixed_count < mixed_warn:
            severity = 'warn'
        _add_issue(issues, issue_type='mixed_type_columns', severity=severity, count=mixed_count, detail={'columns': _limit_list(mixed_type_details)})
    high_card_details = payload.get('high_cardinality_details') or []
    high_card_count = len(high_card_details)
    high_card_ratio_fail = _safe_float(thresholds.get('high_cardinality_ratio_fail'))
    high_card_count_fail = _safe_int(thresholds.get('high_cardinality_count_fail'))
    fail_count = 0
    if high_card_ratio_fail is not None:
        for item in high_card_details:
            ratio = _safe_float(item.get('unique_ratio'))
            if ratio is not None and ratio >= high_card_ratio_fail:
                fail_count += 1
    if high_card_count > 0:
        severity = 'warn'
        if fail_count > 0 or (high_card_count_fail and high_card_count >= high_card_count_fail):
            severity = 'fail'
        _add_issue(issues, issue_type='high_cardinality', severity=severity, count=high_card_count, detail={'columns': _limit_list(high_card_details)})
    id_like_cols = payload.get('id_like_columns') or []
    id_like_count = len(id_like_cols)
    id_like_ratio_fail = _safe_float(thresholds.get('id_like_ratio_fail'))
    id_fail_count = 0
    if id_like_ratio_fail is not None:
        for item in id_like_cols:
            ratio = _safe_float(item.get('unique_ratio'))
            if ratio is not None and ratio >= id_like_ratio_fail:
                id_fail_count += 1
    if id_like_count > 0:
        severity = 'warn'
        if id_fail_count > 0:
            severity = 'fail'
        _add_issue(issues, issue_type='id_like', severity=severity, count=id_like_count, detail={'columns': _limit_list(id_like_cols)})
    name_suspects = payload.get('name_suspects') or []
    name_count = len(name_suspects)
    if name_count > 0:
        _add_issue(issues, issue_type='name_suspect', severity='warn', count=name_count, detail={'columns': _limit_list(name_suspects)})
    leak_suspects = payload.get('leak_suspects') or []
    leak_errors = [s for s in leak_suspects if (s or {}).get('severity') == 'error']
    leak_warnings = [s for s in leak_suspects if (s or {}).get('severity') != 'error']
    if leak_errors:
        _add_issue(issues, issue_type='leak', severity='fail', count=len(leak_errors), detail={'columns': _limit_list(leak_errors)})
    if leak_warnings:
        _add_issue(issues, issue_type='leak', severity='warn', count=len(leak_warnings), detail={'columns': _limit_list(leak_warnings)})
    issue_count = 0
    status = 'pass'
    for issue in issues:
        if issue.get('severity') == 'fail':
            status = 'fail'
        elif status != 'fail':
            status = 'warn'
        count = _safe_int(issue.get('count'))
        if count is None:
            issue_count += 1
        else:
            issue_count += count
    payload['quality_issues'] = issues
    payload['quality_status'] = status
    payload['quality_issue_count'] = issue_count
    return payload
def _format_fail_reasons(issues: Iterable[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for issue in issues:
        if issue.get('severity') != 'fail':
            continue
        issue_type = str(issue.get('type') or 'unknown')
        count = _safe_int(issue.get('count'))
        if count is None or count <= 1:
            reasons.append(issue_type)
        else:
            reasons.append(f'{issue_type}({count})')
    return reasons
def _build_disabled_payload(*, schema: Mapping[str, Any] | None, target_column: str | None, task_type: str | None, id_columns: Iterable[Any] | None, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    rows = None
    columns = None
    if isinstance(schema, Mapping):
        rows = schema.get('rows')
        columns = schema.get('columns')
    payload: dict[str, Any] = {'rows_total': rows, 'rows_scanned': rows, 'columns': columns, 'columns_scanned': columns, 'scan_sampled': False, 'target_column': target_column, 'task_type': task_type, 'id_columns': [str(c) for c in id_columns or []], 'dtype_counts': {}, 'missing_total': None, 'missing_rate_total': None, 'missing_columns': None, 'missing_top': [], 'duplicates_count': None, 'duplicates_rate': None, 'constant_columns': [], 'mixed_type_columns': [], 'mixed_type_details': [], 'high_cardinality_columns': [], 'high_cardinality_details': [], 'high_cardinality_threshold': thresholds.get('high_cardinality_ratio_warn'), 'id_like_columns': [], 'name_suspects': [], 'quality_thresholds': dict(thresholds), 'leak_suspects': [], 'leak_summary': {'available': False, 'note': 'data_quality_disabled_or_unavailable'}}
    payload['quality_issues'] = []
    payload['quality_status'] = 'pass'
    payload['quality_issue_count'] = 0
    return payload
def _write_quality_artifacts(payload: Mapping[str, Any], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f'{prefix}.json'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    md_path = output_dir / f'{prefix}.md'
    md_path.write_text(render_data_quality_markdown(payload), encoding='utf-8')
    return (json_path, md_path)
def run_data_quality_gate(*, cfg: Any, ctx: TaskContext, df: Any | None, target_column: str | None, task_type: str | None, id_columns: Iterable[Any] | None, output_dir: Path, schema: Mapping[str, Any] | None=None, artifact_prefix: str='data_quality') -> dict[str, Any]:
    settings = resolve_quality_settings(cfg)
    thresholds = settings['thresholds']
    payload: dict[str, Any]
    if df is None or not settings['enabled']:
        payload = _build_disabled_payload(schema=schema, target_column=target_column, task_type=task_type, id_columns=id_columns, thresholds=thresholds)
    else:
        payload = compute_data_quality(df, target_column=target_column, task_type=task_type, id_columns=id_columns, max_rows_scan=settings['max_rows_scan'], thresholds=thresholds, name_patterns=settings['name_patterns'])
        payload = evaluate_quality(payload, thresholds)
    (json_path, md_path) = _write_quality_artifacts(payload, output_dir, artifact_prefix)
    upload_artifact(ctx, json_path.name, json_path)
    upload_artifact(ctx, md_path.name, md_path)
    status = payload.get('quality_status') or 'pass'
    issue_count = _safe_int(payload.get('quality_issue_count')) or 0
    update_task_properties(ctx, {'quality_status': str(status), 'quality_issue_count': str(issue_count)})
    issues = payload.get('quality_issues') or []
    gate = {'mode': settings['mode'], 'status': status, 'should_fail': settings['mode'] == 'fail' and status == 'fail', 'issue_count': issue_count, 'fail_reasons': _format_fail_reasons(issues)}
    return {'payload': payload, 'summary': summarize_data_quality(payload), 'gate': gate, 'paths': {'json': json_path, 'md': md_path}}
def raise_on_quality_fail(*, cfg: Any, ctx: TaskContext, gate: Mapping[str, Any], payload: Mapping[str, Any], json_path: Path) -> None:
    if not gate.get('should_fail'):
        return
    reasons = gate.get('fail_reasons') or []
    reason_text = ', '.join((str(item) for item in reasons)) if reasons else 'quality_failed'
    message = f'data_quality gate failed: {reason_text}'
    emit_alert('data_quality', 'error', 'Data quality gate failed', message, {'_cfg': cfg, '_ctx': ctx, 'quality_status': payload.get('quality_status'), 'quality_issue_count': payload.get('quality_issue_count'), 'fail_reasons': reasons, 'data_quality_path': str(json_path)})
    raise ValueError(message)
