from __future__ import annotations
from ..common.config_utils import normalize_str as _normalize_str, normalize_task_type as _normalize_task_type
import math
import numbers
from datetime import date, datetime
from typing import Any, Iterable, Mapping
_HIGH_CARDINALITY_THRESHOLD = 0.8
_CORRELATION_THRESHOLD = 0.98
_NEAR_MATCH_THRESHOLD = 0.98
_MAPPING_ACCURACY_THRESHOLD = 0.98
_MAX_MISSING_TOP = 10
_MAX_CORR_TOP = 5
_ID_LIKE_THRESHOLD = 0.98
_HIGH_CARDINALITY_MIN_UNIQUE = 20
_ID_LIKE_MIN_UNIQUE = 20
_MAX_TYPE_SAMPLE = 200
_RECOVERABLE_ERRORS = (AttributeError, ImportError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, FloatingPointError)
_DEFAULT_NAME_PATTERNS = {'leak_suspect': ('target', 'label', 'outcome', 'response'), 'id_suspect': ('id', 'uuid', 'guid')}
def _as_str_list(values: Iterable[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(v) for v in values if v is not None]
def _threshold_float(thresholds: Mapping[str, Any] | None, key: str, default: float) -> float:
    if not thresholds:
        return float(default)
    value = thresholds.get(key, default)
    try:
        return float(value)
    except _RECOVERABLE_ERRORS:
        return float(default)
def _threshold_int(thresholds: Mapping[str, Any] | None, key: str, default: int) -> int:
    if not thresholds:
        return int(default)
    value = thresholds.get(key, default)
    try:
        return int(value)
    except _RECOVERABLE_ERRORS:
        return int(default)
def _normalize_patterns(name_patterns: Mapping[str, Any] | None, target_column: str | None) -> dict[str, list[str]]:
    patterns: dict[str, list[str]] = {key: [str(item).lower() for item in values if str(item).strip()] for (key, values) in _DEFAULT_NAME_PATTERNS.items()}
    if isinstance(name_patterns, Mapping):
        for (key, values) in name_patterns.items():
            if values is None:
                continue
            if isinstance(values, (str, bytes)):
                seq = [values]
            else:
                seq = values
            items = [str(item).lower() for item in seq if str(item).strip()]
            if not items:
                continue
            existing = patterns.get(str(key), [])
            for item in items:
                if item not in existing:
                    existing.append(item)
            patterns[str(key)] = existing
    if target_column:
        target_key = target_column.strip().lower()
        if target_key:
            existing = patterns.get('leak_suspect', [])
            if target_key not in existing:
                existing.append(target_key)
            patterns['leak_suspect'] = existing
    return patterns
def _pattern_matches(name: str, patterns: Iterable[str]) -> list[str]:
    hits: list[str] = []
    lowered = name.lower()
    for pattern in patterns:
        pattern_key = str(pattern).strip().lower()
        if not pattern_key:
            continue
        if pattern_key in lowered:
            hits.append(pattern_key)
    return hits
def _value_type_bucket(value: Any, *, np: Any | None, pd: Any | None) -> str | None:
    if value is None:
        return None
    if pd is not None:
        try:
            if bool(pd.isna(value)):
                return None
        except _RECOVERABLE_ERRORS:
            pass
    if isinstance(value, (str, bytes)):
        return 'string'
    if isinstance(value, bool):
        return 'bool'
    if np is not None:
        try:
            if isinstance(value, np.bool_):
                return 'bool'
        except _RECOVERABLE_ERRORS:
            pass
    if isinstance(value, numbers.Number):
        return 'number'
    if np is not None:
        try:
            if isinstance(value, np.number):
                return 'number'
        except _RECOVERABLE_ERRORS:
            pass
    if isinstance(value, (datetime, date)):
        return 'datetime'
    if np is not None:
        try:
            if isinstance(value, np.datetime64):
                return 'datetime'
        except _RECOVERABLE_ERRORS:
            pass
    return type(value).__name__
def _sample_type_buckets(series, *, max_samples: int, np: Any | None, pd: Any | None) -> dict[str, int]:
    try:
        values = series.dropna()
        if max_samples and len(values) > max_samples:
            values = values.head(max_samples)
        iterable = values.values if hasattr(values, 'values') else values
    except _RECOVERABLE_ERRORS:
        return {}
    buckets: dict[str, int] = {}
    for value in iterable:
        bucket = _value_type_bucket(value, np=np, pd=pd)
        if bucket is None:
            continue
        buckets[bucket] = buckets.get(bucket, 0) + 1
    return buckets
def _is_numeric_series(series) -> bool:
    try:
        from pandas.api.types import is_numeric_dtype
    except _RECOVERABLE_ERRORS:
        dtype_text = str(getattr(series, 'dtype', '')).lower()
        return dtype_text.startswith(('int', 'uint', 'float', 'double'))
    try:
        return bool(is_numeric_dtype(series))
    except _RECOVERABLE_ERRORS:
        return False
def _is_integer_like(series) -> bool:
    try:
        from pandas.api.types import is_bool_dtype, is_integer_dtype
    except _RECOVERABLE_ERRORS:
        dtype_text = str(getattr(series, 'dtype', '')).lower()
        return dtype_text.startswith(('int', 'uint', 'bool'))
    try:
        return bool(is_integer_dtype(series) or is_bool_dtype(series))
    except _RECOVERABLE_ERRORS:
        return False
def _series_equals(left, right) -> bool:
    try:
        return bool(left.equals(right))
    except _RECOVERABLE_ERRORS:
        return False
def _match_rate(left, right) -> float:
    if left is None or right is None:
        return 0.0
    try:
        eq = left.eq(right)
        try:
            both_na = left.isna() & right.isna()
            eq = eq | both_na
        except _RECOVERABLE_ERRORS:
            pass
        if hasattr(eq, 'mean'):
            return float(eq.mean())
    except _RECOVERABLE_ERRORS:
        return 0.0
    return 0.0
def compute_data_quality(df, target_column: str | None, task_type: str | None, id_columns: Iterable[Any] | None=None, *, max_rows_scan: int=50000, thresholds: Mapping[str, Any] | None=None, name_patterns: Mapping[str, Any] | None=None) -> dict[str, Any]:
    """Compute basic data quality metrics.

    Returns a JSON-serializable dict. Keys duplicates_count and leak_suspects are top-level.
    """
    rows_total = int(getattr(df, 'shape', [0, 0])[0] or 0)
    cols_total = int(getattr(df, 'shape', [0, 0])[1] or 0)
    sampled = False
    if max_rows_scan and rows_total > max_rows_scan:
        df_scan = df.head(int(max_rows_scan))
        sampled = True
    else:
        df_scan = df
    rows_scanned = int(getattr(df_scan, 'shape', [0, 0])[0] or 0)
    cols_scanned = int(getattr(df_scan, 'shape', [0, 0])[1] or 0)
    target_column = _normalize_str(target_column)
    task_type = _normalize_task_type(task_type)
    id_columns_list = _as_str_list(id_columns)
    id_columns_set = set(id_columns_list)
    try:
        import numpy as np
    except _RECOVERABLE_ERRORS:
        np = None
    try:
        import pandas as pd
    except _RECOVERABLE_ERRORS:
        pd = None
    thresholds = thresholds or {}
    patterns = _normalize_patterns(name_patterns, target_column)
    leak_name_patterns = patterns.get('leak_suspect', [])
    id_name_patterns = patterns.get('id_suspect', [])
    high_card_ratio = _threshold_float(thresholds, 'high_cardinality_ratio_warn', _HIGH_CARDINALITY_THRESHOLD)
    high_card_min_unique = _threshold_int(thresholds, 'high_cardinality_min_unique', _HIGH_CARDINALITY_MIN_UNIQUE)
    id_like_ratio = _threshold_float(thresholds, 'id_like_ratio_warn', _ID_LIKE_THRESHOLD)
    id_like_min_unique = _threshold_int(thresholds, 'id_like_min_unique', _ID_LIKE_MIN_UNIQUE)
    near_match_threshold = _threshold_float(thresholds, 'leak_near_match_warn', _NEAR_MATCH_THRESHOLD)
    corr_threshold = _threshold_float(thresholds, 'leak_correlation_warn', _CORRELATION_THRESHOLD)
    mapping_threshold = _threshold_float(thresholds, 'leak_mapping_accuracy_warn', _MAPPING_ACCURACY_THRESHOLD)
    dtype_counts: dict[str, int] = {}
    try:
        dtype_series = df_scan.dtypes.astype(str)
        dtype_counts = {str(k): int(v) for (k, v) in dtype_series.value_counts().to_dict().items()}
    except _RECOVERABLE_ERRORS:
        dtype_counts = {}
    missing_total = 0
    missing_columns = 0
    missing_rate_total = 0.0
    missing_top: list[dict[str, Any]] = []
    null_counts = None
    try:
        null_counts = df_scan.isna().sum()
        missing_total = int(null_counts.sum())
        missing_columns = int((null_counts > 0).sum())
        denom = rows_scanned * cols_scanned
        missing_rate_total = float(missing_total / denom) if denom else 0.0
        missing_records: list[dict[str, Any]] = []
        for col in df_scan.columns:
            count = int(null_counts[col])
            if count <= 0:
                continue
            rate = float(count / rows_scanned) if rows_scanned else 0.0
            missing_records.append({'column': str(col), 'missing_count': count, 'missing_rate': rate})
        missing_records.sort(key=lambda item: item['missing_rate'], reverse=True)
        missing_top = missing_records[:_MAX_MISSING_TOP]
    except _RECOVERABLE_ERRORS:
        missing_top = []
    duplicates_count = 0
    duplicates_rate = 0.0
    try:
        duplicates_count = int(df_scan.duplicated().sum()) if rows_scanned else 0
        duplicates_rate = float(duplicates_count / rows_scanned) if rows_scanned else 0.0
    except _RECOVERABLE_ERRORS:
        duplicates_count = 0
        duplicates_rate = 0.0
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    high_cardinality_details: list[dict[str, Any]] = []
    id_like_columns: list[dict[str, Any]] = []
    mixed_type_columns: list[str] = []
    mixed_type_details: list[dict[str, Any]] = []
    name_suspects: list[dict[str, Any]] = []
    try:
        for col in df_scan.columns:
            col_name = str(col)
            series = df_scan[col]
            try:
                unique_count = int(series.nunique(dropna=False))
            except _RECOVERABLE_ERRORS:
                unique_count = 0
            unique_ratio = float(unique_count / rows_scanned) if rows_scanned else 0.0
            if unique_count <= 1:
                constant_columns.append(col_name)
            missing_count = None
            missing_rate = 0.0
            if null_counts is not None:
                try:
                    missing_count = int(null_counts[col])
                    missing_rate = float(missing_count / rows_scanned) if rows_scanned else 0.0
                except _RECOVERABLE_ERRORS:
                    missing_count = None
                    missing_rate = 0.0
            if target_column and col_name == target_column:
                continue
            if rows_scanned > 0:
                if unique_count >= id_like_min_unique and unique_ratio >= id_like_ratio:
                    id_like_columns.append({'column': col_name, 'unique_count': unique_count, 'unique_ratio': unique_ratio, 'missing_rate': missing_rate, 'declared_id': col_name in id_columns_set})
                if unique_count >= high_card_min_unique and unique_ratio >= high_card_ratio:
                    if not _is_numeric_series(series) or _is_integer_like(series):
                        high_cardinality_columns.append(col_name)
                        high_cardinality_details.append({'column': col_name, 'unique_count': unique_count, 'unique_ratio': unique_ratio, 'missing_rate': missing_rate, 'declared_id': col_name in id_columns_set})
            dtype_text = str(getattr(series, 'dtype', '')).lower()
            if rows_scanned > 0 and 'object' in dtype_text:
                type_buckets = _sample_type_buckets(series, max_samples=_MAX_TYPE_SAMPLE, np=np, pd=pd)
                if len(type_buckets) > 1:
                    mixed_type_columns.append(col_name)
                    mixed_type_details.append({'column': col_name, 'types': dict(type_buckets)})
            if leak_name_patterns:
                matches = _pattern_matches(col_name, leak_name_patterns)
                if matches:
                    name_suspects.append({'column': col_name, 'kind': 'leak_suspect', 'patterns': matches})
            if id_name_patterns:
                matches = _pattern_matches(col_name, id_name_patterns)
                if matches:
                    name_suspects.append({'column': col_name, 'kind': 'id_suspect', 'patterns': matches, 'declared_id': col_name in id_columns_set})
    except _RECOVERABLE_ERRORS:
        constant_columns = []
        high_cardinality_columns = []
        high_cardinality_details = []
        id_like_columns = []
        mixed_type_columns = []
        mixed_type_details = []
        name_suspects = []
    leak_suspects: list[dict[str, Any]] = []
    leak_summary: dict[str, Any] = {'available': False, 'exact_match': [], 'near_match': [], 'correlation_top': [], 'mapping_suspects': [], 'thresholds': {'near_match': near_match_threshold, 'correlation': corr_threshold, 'mapping_accuracy': mapping_threshold}}
    leak_seen: set[tuple[str, str]] = set()
    def _add_suspect(*, column: str, reason: str, severity: str, score: float | None=None, detail: Mapping[str, Any] | None=None) -> None:
        key = (column, reason)
        if key in leak_seen:
            return
        leak_seen.add(key)
        payload = {'column': column, 'reason': reason, 'severity': severity}
        if score is not None:
            payload['score'] = float(score)
        if detail:
            payload['detail'] = dict(detail)
        leak_suspects.append(payload)
    if target_column and target_column in getattr(df_scan, 'columns', []):
        leak_summary['available'] = True
        target_series = df_scan[target_column]
        feature_columns = [col for col in df_scan.columns if str(col) != target_column and str(col) not in id_columns_list]
        for col in feature_columns:
            series = df_scan[col]
            if _series_equals(series, target_series):
                leak_summary['exact_match'].append({'column': str(col)})
                _add_suspect(column=str(col), reason='exact_match', severity='error', score=1.0)
                continue
            match_rate = _match_rate(series, target_series)
            if match_rate >= near_match_threshold:
                leak_summary['near_match'].append({'column': str(col), 'match_rate': float(match_rate)})
                _add_suspect(column=str(col), reason='near_match', severity='warning', score=match_rate)
        if task_type == 'regression':
            try:
                import pandas as pd
            except _RECOVERABLE_ERRORS:
                pd = None
            corr_candidates: list[dict[str, Any]] = []
            if pd is not None:
                target_numeric = pd.to_numeric(target_series, errors='coerce')
                numeric_columns = df_scan[feature_columns].select_dtypes(include=['number']).columns.tolist()
                for col in numeric_columns:
                    series = pd.to_numeric(df_scan[col], errors='coerce')
                    mask = series.notna() & target_numeric.notna()
                    if int(mask.sum()) < 2:
                        continue
                    corr = float(series[mask].corr(target_numeric[mask]))
                    if math.isnan(corr):
                        continue
                    corr_candidates.append({'column': str(col), 'corr': corr})
                corr_candidates.sort(key=lambda item: abs(item['corr']), reverse=True)
                leak_summary['correlation_top'] = corr_candidates[:_MAX_CORR_TOP]
                for item in corr_candidates:
                    if abs(item['corr']) >= corr_threshold:
                        _add_suspect(column=str(item['column']), reason='high_correlation', severity='warning', score=float(item['corr']))
        else:
            try:
                import pandas as pd
            except _RECOVERABLE_ERRORS:
                pd = None
            if pd is not None and rows_scanned:
                for col in feature_columns:
                    col_values = df_scan[col].fillna('__MISSING__')
                    target_values = target_series.fillna('__MISSING__')
                    counts = pd.DataFrame({'col': col_values, 'target': target_values}).groupby(['col', 'target'], dropna=False).size()
                    if counts.empty:
                        continue
                    max_by_col = counts.groupby(level=0).max()
                    accuracy = float(max_by_col.sum() / rows_scanned)
                    target_per_col = counts.groupby(level=0).size()
                    deterministic = bool(target_per_col.max() <= 1)
                    if deterministic or accuracy >= mapping_threshold:
                        leak_summary['mapping_suspects'].append({'column': str(col), 'accuracy': accuracy, 'unique_values': int(target_per_col.shape[0]), 'deterministic': deterministic})
                        _add_suspect(column=str(col), reason='deterministic_mapping' if deterministic else 'high_accuracy_mapping', severity='warning', score=accuracy)
    payload = {'rows_total': rows_total, 'rows_scanned': rows_scanned, 'columns': cols_total, 'columns_scanned': cols_scanned, 'scan_sampled': sampled, 'target_column': target_column, 'task_type': task_type, 'id_columns': id_columns_list, 'dtype_counts': dtype_counts, 'missing_total': missing_total, 'missing_rate_total': missing_rate_total, 'missing_columns': missing_columns, 'missing_top': missing_top, 'duplicates_count': duplicates_count, 'duplicates_rate': duplicates_rate, 'constant_columns': constant_columns, 'mixed_type_columns': mixed_type_columns, 'mixed_type_details': mixed_type_details, 'high_cardinality_columns': high_cardinality_columns, 'high_cardinality_details': high_cardinality_details, 'high_cardinality_threshold': high_card_ratio, 'id_like_columns': id_like_columns, 'name_suspects': name_suspects, 'quality_thresholds': {'high_cardinality_ratio_warn': high_card_ratio, 'high_cardinality_min_unique': high_card_min_unique, 'id_like_ratio_warn': id_like_ratio, 'id_like_min_unique': id_like_min_unique, 'leak_near_match_warn': near_match_threshold, 'leak_correlation_warn': corr_threshold, 'leak_mapping_accuracy_warn': mapping_threshold, 'mixed_type_columns_warn': thresholds.get('mixed_type_columns_warn'), 'mixed_type_columns_fail': thresholds.get('mixed_type_columns_fail')}, 'leak_suspects': leak_suspects, 'leak_summary': leak_summary}
    return payload
def summarize_data_quality(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing_top = payload.get('missing_top') or []
    missing_rate_max = None
    if missing_top:
        try:
            missing_rate_max = float(missing_top[0].get('missing_rate'))
        except _RECOVERABLE_ERRORS:
            missing_rate_max = None
    leak_suspects = payload.get('leak_suspects') or []
    leak_columns = [str(item.get('column')) for item in leak_suspects if item and item.get('column') is not None]
    leak_error = any(((item or {}).get('severity') == 'error' for item in leak_suspects if item is not None))
    summary = {'rows': payload.get('rows_total'), 'columns': payload.get('columns'), 'missing_columns': payload.get('missing_columns'), 'missing_rate_max': missing_rate_max, 'duplicates_count': payload.get('duplicates_count'), 'high_cardinality_count': len(payload.get('high_cardinality_columns') or []), 'id_like_count': len(payload.get('id_like_columns') or []), 'mixed_type_count': len(payload.get('mixed_type_columns') or []), 'name_suspect_count': len(payload.get('name_suspects') or []), 'leak_suspect_count': len(leak_suspects), 'leak_suspects': leak_columns, 'leak_error': leak_error, 'quality_status': payload.get('quality_status'), 'quality_issue_count': payload.get('quality_issue_count')}
    return summary
def render_data_quality_markdown(payload: Mapping[str, Any]) -> str:
    rows_total = payload.get('rows_total')
    rows_scanned = payload.get('rows_scanned')
    sampled = payload.get('scan_sampled')
    lines = ['# Data Quality Summary', '']
    quality_status = payload.get('quality_status')
    if quality_status is not None:
        lines.append(f'- quality_status: {quality_status}')
    issue_count = payload.get('quality_issue_count')
    if issue_count is not None:
        lines.append(f'- quality_issue_count: {issue_count}')
    lines.extend([f'- rows_total: {rows_total}', f'- rows_scanned: {rows_scanned}', f"- columns: {payload.get('columns')}", f'- sampled: {bool(sampled)}', f"- missing_columns: {payload.get('missing_columns')}", f"- missing_rate_total: {payload.get('missing_rate_total'):.6f}" if isinstance(payload.get('missing_rate_total'), float) else f"- missing_rate_total: {payload.get('missing_rate_total')}", f"- duplicates_count: {payload.get('duplicates_count')}"])
    missing_top = payload.get('missing_top') or []
    if missing_top:
        items = []
        for item in missing_top:
            col = item.get('column')
            rate = item.get('missing_rate')
            if col is None:
                continue
            try:
                rate_text = f'{float(rate):.1%}'
            except _RECOVERABLE_ERRORS:
                rate_text = str(rate)
            items.append(f'{col}({rate_text})')
        lines.append(f"- missing_top: {', '.join(items)}")
    else:
        lines.append('- missing_top: n/a')
    constant_cols = payload.get('constant_columns') or []
    lines.append('- constant_columns: ' + (', '.join(constant_cols) if constant_cols else 'n/a'))
    mixed_type_details = payload.get('mixed_type_details') or []
    if mixed_type_details:
        items = []
        for item in mixed_type_details[:10]:
            col = item.get('column')
            types = item.get('types') or {}
            if col is None:
                continue
            type_text = ','.join(sorted((str(key) for key in types.keys()))) if types else 'n/a'
            items.append(f'{col}({type_text})')
        lines.append(f"- mixed_type_columns: {', '.join(items)}")
    else:
        lines.append('- mixed_type_columns: n/a')
    high_card_cols = payload.get('high_cardinality_columns') or []
    lines.append('- high_cardinality_columns: ' + (', '.join(high_card_cols) if high_card_cols else 'n/a'))
    id_like_cols = payload.get('id_like_columns') or []
    if id_like_cols:
        items = []
        for item in id_like_cols[:10]:
            col = item.get('column')
            ratio = item.get('unique_ratio')
            if col is None:
                continue
            try:
                ratio_text = f'{float(ratio):.1%}'
            except _RECOVERABLE_ERRORS:
                ratio_text = str(ratio)
            items.append(f'{col}({ratio_text})')
        lines.append(f"- id_like_columns: {', '.join(items)}")
    else:
        lines.append('- id_like_columns: n/a')
    name_suspects = payload.get('name_suspects') or []
    if name_suspects:
        items = []
        for item in name_suspects[:10]:
            col = item.get('column')
            kind = item.get('kind')
            patterns = item.get('patterns') or []
            if col is None:
                continue
            pattern_text = ','.join((str(p) for p in patterns)) if patterns else 'n/a'
            items.append(f'{col}:{kind}({pattern_text})')
        lines.append(f"- name_suspects: {', '.join(items)}")
    else:
        lines.append('- name_suspects: none')
    quality_issues = payload.get('quality_issues') or []
    if quality_issues:
        items = []
        for issue in quality_issues[:10]:
            issue_type = issue.get('type')
            severity = issue.get('severity')
            count = issue.get('count')
            if issue_type is None:
                continue
            if count is None:
                items.append(f'{issue_type}({severity})')
            else:
                items.append(f'{issue_type}({severity},{count})')
        lines.append(f"- quality_issues: {', '.join(items)}")
    else:
        lines.append('- quality_issues: none')
    leak_suspects = payload.get('leak_suspects') or []
    if leak_suspects:
        items = []
        for suspect in leak_suspects[:10]:
            col = suspect.get('column')
            reason = suspect.get('reason')
            severity = suspect.get('severity')
            score = suspect.get('score')
            if col is None:
                continue
            if score is None:
                items.append(f'{col}:{reason}({severity})')
            else:
                try:
                    score_text = f'{float(score):.3f}'
                except _RECOVERABLE_ERRORS:
                    score_text = str(score)
                items.append(f'{col}:{reason}({severity},{score_text})')
        lines.append(f"- leak_suspects: {', '.join(items)}")
    else:
        lines.append('- leak_suspects: none')
    return '\n'.join(lines) + '\n'
