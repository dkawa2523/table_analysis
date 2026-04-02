from __future__ import annotations
from ..common.drift_utils import collect_top_drift_features, normalize_drift_metric_names
from ..common.feature_types import infer_tabular_feature_types
from ..common.config_utils import to_float as _to_float, to_int as _to_int
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping, Sequence
_DEFAULT_BINS = 10
_DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
_DEFAULT_MAX_CATEGORIES = 10
_PSI_EPSILON = 1e-06
_COERCE_ERRORS = (TypeError, ValueError)
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_QUANTILE_RECOVERABLE_ERRORS = (AttributeError, TypeError, ValueError)
def _quantile_key(value: float) -> str:
    try:
        pct = int(round(float(value) * 100))
    except _COERCE_ERRORS:
        pct = 0
    return f'p{pct:02d}'
def _build_hist_edges(values, *, bins: int) -> list[float]:
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError('numpy is required for drift profiling.') from exc
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 1.0]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0, 1.0]
    quantiles = np.linspace(0.0, 1.0, int(max(bins, 1)) + 1)
    edges = np.quantile(arr, quantiles)
    edges = np.unique(edges.astype(float))
    if edges.size == 1:
        center = float(edges[0])
        if math.isfinite(center):
            eps = abs(center) * 1e-06 or 1e-06
            edges = np.array([center - eps, center + eps], dtype=float)
        else:
            edges = np.array([0.0, 1.0], dtype=float)
    edges = np.sort(edges)
    if edges.size < 2:
        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
        if min_val == max_val:
            eps = abs(min_val) * 1e-06 or 1e-06
            edges = np.array([min_val - eps, max_val + eps], dtype=float)
        else:
            edges = np.array([min_val, max_val], dtype=float)
    return [float(v) for v in edges.tolist()]
def _hist_counts(values, edges: Sequence[float]) -> list[int]:
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError('numpy is required for drift profiling.') from exc
    if not edges or len(edges) < 2:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0 for _ in range(len(edges) - 1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0 for _ in range(len(edges) - 1)]
    (counts, _) = np.histogram(arr, bins=np.asarray(edges, dtype=float))
    return [int(v) for v in counts.tolist()]
def _psi(expected: Sequence[float], actual: Sequence[float]) -> float | None:
    if not expected or not actual or len(expected) != len(actual):
        return None
    total_expected = sum(expected)
    total_actual = sum(actual)
    if total_expected <= 0.0 or total_actual <= 0.0:
        return None
    psi_value = 0.0
    for (exp, act) in zip(expected, actual):
        exp = max(float(exp) / total_expected, _PSI_EPSILON)
        act = max(float(act) / total_actual, _PSI_EPSILON)
        psi_value += (act - exp) * math.log(act / exp)
    return float(psi_value)
def _ks_from_hist(expected: Sequence[float], actual: Sequence[float]) -> float | None:
    if not expected or not actual or len(expected) != len(actual):
        return None
    total_expected = sum(expected)
    total_actual = sum(actual)
    if total_expected <= 0.0 or total_actual <= 0.0:
        return None
    cum_expected = 0.0
    cum_actual = 0.0
    max_diff = 0.0
    for (exp, act) in zip(expected, actual):
        cum_expected += float(exp) / total_expected
        cum_actual += float(act) / total_actual
        diff = abs(cum_actual - cum_expected)
        if diff > max_diff:
            max_diff = diff
    return float(max_diff)
def build_train_profile(df, feature_columns: Iterable[str] | None=None, *, numeric_features: Iterable[str] | None=None, categorical_features: Iterable[str] | None=None, max_bins: int=_DEFAULT_BINS, quantiles: Sequence[float] | None=None, max_categories: int=_DEFAULT_MAX_CATEGORIES) -> dict[str, Any]:
    try:
        import pandas as pd
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError('pandas is required for drift profiling.') from exc
    if feature_columns is None:
        feature_columns = list(getattr(df, 'columns', []))
    feature_columns = list(feature_columns)
    if numeric_features is None or categorical_features is None:
        (numeric_features, categorical_features) = infer_tabular_feature_types(df, feature_columns)
    numeric_set = {str(col) for col in numeric_features or []}
    categorical_set = {str(col) for col in categorical_features or []}
    numeric_list: list[str] = []
    categorical_list: list[str] = []
    for col in feature_columns:
        if col in numeric_set:
            numeric_list.append(str(col))
        elif col in categorical_set:
            categorical_list.append(str(col))
        else:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_list.append(str(col))
            else:
                categorical_list.append(str(col))
    quantiles = list(quantiles) if quantiles is not None else list(_DEFAULT_QUANTILES)
    rows = int(getattr(df, 'shape', [0, 0])[0] or 0)
    profile: dict[str, Any] = {'profile_version': 1, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'rows': rows, 'settings': {'max_bins': int(max_bins), 'quantiles': [float(q) for q in quantiles], 'max_categories': int(max_categories)}, 'feature_types': {'numeric': numeric_list, 'categorical': categorical_list}, 'numeric': {}, 'categorical': {}}
    numeric_profile: dict[str, Any] = {}
    for col in numeric_list:
        series = pd.to_numeric(df[col], errors='coerce')
        missing_count = int(series.isna().sum())
        non_missing = series.dropna()
        count = int(non_missing.shape[0])
        mean = _to_float(non_missing.mean()) if count else None
        std = _to_float(non_missing.std(ddof=0)) if count else None
        min_val = _to_float(non_missing.min()) if count else None
        max_val = _to_float(non_missing.max()) if count else None
        quantile_payload: dict[str, Any] = {}
        if count and quantiles:
            try:
                q_values = non_missing.quantile(quantiles)
                if hasattr(q_values, 'items'):
                    for (q, value) in q_values.items():
                        quantile_payload[_quantile_key(float(q))] = _to_float(value)
                else:
                    quantile_payload[_quantile_key(float(quantiles[0]))] = _to_float(q_values)
            except _QUANTILE_RECOVERABLE_ERRORS:
                quantile_payload = {}
        edges = _build_hist_edges(non_missing.to_numpy(), bins=int(max_bins))
        counts = _hist_counts(non_missing.to_numpy(), edges)
        numeric_profile[str(col)] = {'count': count, 'missing_count': missing_count, 'mean': mean, 'std': std, 'min': min_val, 'max': max_val, 'quantiles': quantile_payload, 'hist': {'bin_edges': edges, 'counts': counts}}
    categorical_profile: dict[str, Any] = {}
    for col in categorical_list:
        series = df[col]
        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        non_missing = series[~missing_mask]
        total = int(non_missing.shape[0])
        top_entries: list[dict[str, Any]] = []
        other_count = 0
        if total > 0:
            value_counts = non_missing.astype(str).value_counts()
            top = value_counts.head(int(max_categories))
            top_total = 0
            for (value, count) in top.items():
                count_int = _to_int(count, default=0)
                top_total += count_int
                rate = float(count_int / total) if total else 0.0
                top_entries.append({'value': str(value), 'count': count_int, 'rate': rate})
            other_count = max(total - top_total, 0)
        other_rate = float(other_count / total) if total else 0.0
        categorical_profile[str(col)] = {'count': total, 'missing_count': missing_count, 'top': top_entries, 'other': {'count': other_count, 'rate': other_rate}}
    profile['numeric'] = numeric_profile
    profile['categorical'] = categorical_profile
    return profile
def _missing_rate(count: int, missing_count: int) -> float:
    denom = count + missing_count
    if denom <= 0:
        return 0.0
    return float(missing_count / denom)
def build_drift_report(train_profile: Mapping[str, Any], infer_df, *, psi_warn_threshold: float=0.2, psi_fail_threshold: float | None=0.4, strict: bool=False, metrics: Sequence[str] | None=None, train_profile_path: str | None=None) -> dict[str, Any]:
    try:
        import pandas as pd
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError('pandas is required for drift reporting.') from exc
    warn_threshold = _to_float(psi_warn_threshold)
    if warn_threshold is None:
        warn_threshold = 0.0
    fail_threshold = _to_float(psi_fail_threshold) if psi_fail_threshold is not None else None
    train_numeric = train_profile.get('numeric') or {}
    train_categorical = train_profile.get('categorical') or {}
    metrics_list = normalize_drift_metric_names(metrics, default_metrics=('psi',))
    compute_psi = 'psi' in metrics_list
    compute_ks = 'ks' in metrics_list
    rows = int(getattr(infer_df, 'shape', [0, 0])[0] or 0)
    report: dict[str, Any] = {'report_version': 1, 'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'rows': rows, 'metrics': metrics_list, 'thresholds': {'warn': float(warn_threshold), 'fail': float(fail_threshold) if strict and fail_threshold is not None else None}, 'numeric': {}, 'categorical': {}}
    if train_profile_path:
        report['train_profile_path'] = str(train_profile_path)
    numeric_report: dict[str, Any] = {}
    for (col, info) in train_numeric.items():
        if col not in infer_df.columns:
            numeric_report[str(col)] = {'psi': None, 'status': 'missing', 'reason': 'column_missing'}
            continue
        hist = (info or {}).get('hist') or {}
        edges = hist.get('bin_edges') or []
        train_counts = [int(v) for v in hist.get('counts') or []]
        if not edges or len(edges) < 2 or len(train_counts) != len(edges) - 1:
            numeric_report[str(col)] = {'psi': None, 'status': 'n/a', 'reason': 'invalid_histogram'}
            continue
        train_count = _to_int((info or {}).get('count'), default=0)
        train_missing = _to_int((info or {}).get('missing_count'), default=0)
        series = pd.to_numeric(infer_df[col], errors='coerce')
        missing_count = int(series.isna().sum())
        non_missing = series.dropna()
        actual_counts = _hist_counts(non_missing.to_numpy(), edges)
        expected = [float(v) for v in train_counts]
        actual = [float(v) for v in actual_counts]
        psi_value = _psi(expected, actual) if compute_psi else None
        ks_value = _ks_from_hist(expected, actual) if compute_ks else None
        if compute_psi:
            status = 'ok'
            if psi_value is None:
                status = 'n/a'
            elif psi_value >= warn_threshold:
                status = 'warn'
            if strict and fail_threshold is not None and (psi_value is not None) and (psi_value >= fail_threshold):
                status = 'fail'
        else:
            status = 'n/a'
        entry = {'psi': psi_value, 'status': status, 'missing_rate_train': _missing_rate(train_count, train_missing), 'missing_rate_infer': _missing_rate(int(non_missing.shape[0]), missing_count), 'expected': {'bin_edges': [float(v) for v in edges], 'counts': [int(v) for v in train_counts]}, 'actual': {'counts': [int(v) for v in actual_counts]}}
        if compute_ks:
            entry['ks'] = ks_value
        numeric_report[str(col)] = entry
    categorical_report: dict[str, Any] = {}
    for (col, info) in train_categorical.items():
        if col not in infer_df.columns:
            categorical_report[str(col)] = {'psi': None, 'status': 'missing', 'reason': 'column_missing'}
            continue
        train_count = _to_int((info or {}).get('count'), default=0)
        train_missing = _to_int((info or {}).get('missing_count'), default=0)
        top_entries = list((info or {}).get('top') or [])
        top_values = [str(entry.get('value')) for entry in top_entries if entry.get('value') is not None]
        train_top_counts = {str(entry.get('value')): _to_int(entry.get('count'), default=0) for entry in top_entries}
        train_other = _to_int((info or {}).get('other', {}).get('count'), default=0)
        series = infer_df[col]
        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())
        non_missing = series[~missing_mask]
        total = int(non_missing.shape[0])
        value_counts = non_missing.astype(str).value_counts() if total else {}
        actual_top_counts = [int(value_counts.get(val, 0)) for val in top_values]
        actual_other = max(total - sum(actual_top_counts), 0)
        expected = [float(train_top_counts.get(val, 0)) for val in top_values] + [float(train_other)]
        actual = [float(v) for v in actual_top_counts] + [float(actual_other)]
        psi_value = _psi(expected, actual) if compute_psi else None
        if compute_psi:
            status = 'ok'
            if psi_value is None:
                status = 'n/a'
            elif psi_value >= warn_threshold:
                status = 'warn'
            if strict and fail_threshold is not None and (psi_value is not None) and (psi_value >= fail_threshold):
                status = 'fail'
        else:
            status = 'n/a'
        categorical_report[str(col)] = {'psi': psi_value, 'status': status, 'missing_rate_train': _missing_rate(train_count, train_missing), 'missing_rate_infer': _missing_rate(total, missing_count), 'top_values': top_values, 'expected': {'counts': [int(v) for v in expected], 'labels': [*top_values, 'other']}, 'actual': {'counts': [int(v) for v in actual], 'labels': [*top_values, 'other']}}
    report['numeric'] = numeric_report
    report['categorical'] = categorical_report
    psi_values: list[float] = []
    ks_values: list[float] = []
    warned_features: list[str] = []
    failed_features: list[str] = []
    missing_features: list[str] = []
    for section in (numeric_report, categorical_report):
        for (name, entry) in section.items():
            status = entry.get('status')
            if status == 'missing':
                missing_features.append(name)
                continue
            psi_value = entry.get('psi')
            if psi_value is not None:
                psi_values.append(float(psi_value))
                if psi_value >= warn_threshold:
                    warned_features.append(name)
                if strict and fail_threshold is not None and (psi_value >= fail_threshold):
                    failed_features.append(name)
            if compute_ks and 'ks' in entry and (entry.get('ks') is not None):
                ks_values.append(float(entry.get('ks')))
    psi_max = max(psi_values) if psi_values else None
    psi_mean = float(sum(psi_values) / len(psi_values)) if psi_values else None
    ks_max = max(ks_values) if ks_values else None
    ks_mean = float(sum(ks_values) / len(ks_values)) if ks_values else None
    report['summary'] = {'features_total': len(numeric_report) + len(categorical_report), 'psi_max': psi_max, 'psi_mean': psi_mean, 'ks_max': ks_max, 'ks_mean': ks_mean, 'warn_threshold': float(warn_threshold), 'fail_threshold': float(fail_threshold) if strict and fail_threshold is not None else None, 'warn_count': len(warned_features), 'fail_count': len(failed_features), 'warned_features': warned_features, 'failed_features': failed_features, 'missing_features': missing_features}
    return report
def render_drift_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get('summary') or {}
    lines = ['# Drift Report', '', f"- rows: {report.get('rows')}", f"- psi_warn_threshold: {summary.get('warn_threshold')}", f"- psi_fail_threshold: {summary.get('fail_threshold')}", f"- psi_max: {summary.get('psi_max')}", f"- psi_mean: {summary.get('psi_mean')}", f"- ks_max: {summary.get('ks_max')}", f"- ks_mean: {summary.get('ks_mean')}", f"- warn_count: {summary.get('warn_count')}", f"- fail_count: {summary.get('fail_count')}"]
    missing_features = summary.get('missing_features') or []
    if missing_features:
        lines.append(f"- missing_features: {', '.join(map(str, missing_features[:10]))}")
    top_features = collect_top_drift_features(report, metric='psi', limit=10, include_metrics=('ks',))
    if top_features:
        lines.extend(['', '## Top Drift Features'])
        for item in top_features:
            ks_value = item.get('ks')
            ks_text = f' ks={ks_value:.4f}' if ks_value is not None else ''
            lines.append(f"- {item['feature']} ({item['kind']}): psi={item['psi']:.4f}{ks_text} [{item.get('status')}]")
    return '\n'.join(lines) + '\n'
