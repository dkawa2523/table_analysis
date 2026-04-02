from __future__ import annotations
from ..common.drift_utils import collect_top_drift_features, normalize_drift_metric_names, select_primary_drift_metric
from ..common.config_utils import cfg_value as _cfg_value, to_float as _to_float, to_int as _to_int
from pathlib import Path
from typing import Any, Mapping, Sequence
_SAMPLING_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def resolve_drift_settings(cfg: Any) -> dict[str, Any]:
    monitor_enabled = bool(_cfg_value(cfg, 'monitor.drift.enabled', False))
    infer_enabled = bool(_cfg_value(cfg, 'infer.drift.enabled', False))
    enabled = monitor_enabled or infer_enabled
    sample_n = _to_int(_cfg_value(cfg, 'monitor.drift.sample_n', 5000))
    if sample_n is not None and sample_n <= 0:
        sample_n = None
    metrics = normalize_drift_metric_names(_cfg_value(cfg, 'monitor.drift.metrics', None), default_metrics=('psi', 'ks'))
    psi_alert = _to_float(_cfg_value(cfg, 'monitor.drift.alert_thresholds.psi', None))
    warn_threshold = psi_alert
    if warn_threshold is None:
        warn_threshold = _to_float(_cfg_value(cfg, 'infer.drift.psi_warn_threshold', None))
    if warn_threshold is None:
        warn_threshold = 0.2
    fail_threshold = _to_float(_cfg_value(cfg, 'infer.drift.psi_fail_threshold', None))
    sample_seed = _to_int(_cfg_value(cfg, 'eval.seed', 42))
    if sample_seed is None:
        sample_seed = 42
    return {'enabled': enabled, 'sample_n': sample_n, 'sample_seed': sample_seed, 'metrics': metrics, 'psi_warn_threshold': warn_threshold, 'psi_fail_threshold': fail_threshold, 'alert_thresholds': {'psi': warn_threshold}}
def sample_frame(df: Any, *, sample_n: int | None, seed: int | None=None) -> tuple[Any, dict[str, Any]]:
    rows = int(getattr(df, 'shape', [0, 0])[0] or 0)
    info = {'rows': rows, 'sample_n': sample_n, 'sampled': False, 'sampled_rows': rows, 'seed': seed}
    if sample_n is None or sample_n <= 0 or rows <= sample_n:
        return (df, info)
    if hasattr(df, 'sample'):
        try:
            sampled = df.sample(n=int(sample_n), random_state=seed)
            info['sampled'] = True
            info['sampled_rows'] = int(sample_n)
            return (sampled, info)
        except _SAMPLING_ERRORS:
            pass
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS:
        return (df, info)
    rng = np.random.default_rng(seed)
    indices = rng.choice(rows, size=int(sample_n), replace=False)
    try:
        sampled = df.iloc[indices]
    except _SAMPLING_ERRORS:
        sampled = df[indices]
    info['sampled'] = True
    info['sampled_rows'] = int(sample_n)
    return (sampled, info)
def annotate_profile(profile: dict[str, Any], *, role: str | None=None, sample_info: Mapping[str, Any] | None=None, metrics: Sequence[str] | None=None) -> dict[str, Any]:
    if role:
        profile['role'] = str(role)
    if sample_info:
        profile['sampling'] = dict(sample_info)
    if metrics:
        settings = profile.get('settings')
        if isinstance(settings, dict):
            settings['metrics'] = [str(item) for item in metrics]
    return profile
def build_drift_summary_lines(report: Mapping[str, Any], *, limit: int=5, drift_alert: bool | None=None) -> list[str]:
    if not report:
        return []
    summary = report.get('summary') or {}
    lines = ['## Drift Summary']
    lines.append(f"- rows: {report.get('rows')}")
    metrics = report.get('metrics') or []
    if metrics:
        lines.append(f"- metrics: {', '.join([str(item) for item in metrics])}")
    if 'psi_max' in summary:
        lines.append(f"- psi_max: {summary.get('psi_max')}")
    if 'psi_mean' in summary:
        lines.append(f"- psi_mean: {summary.get('psi_mean')}")
    if 'ks_max' in summary:
        lines.append(f"- ks_max: {summary.get('ks_max')}")
    if 'ks_mean' in summary:
        lines.append(f"- ks_mean: {summary.get('ks_mean')}")
    if 'warn_count' in summary:
        lines.append(f"- warn_count: {summary.get('warn_count')}")
    if 'fail_count' in summary:
        lines.append(f"- fail_count: {summary.get('fail_count')}")
    if drift_alert is not None:
        lines.append(f'- drift_alert: {bool(drift_alert)}')
    metric = select_primary_drift_metric(report)
    top_features = collect_top_drift_features(report, metric=metric, limit=limit)
    if top_features:
        lines.append('')
        lines.append('### Top Drift Features')
        for item in top_features:
            value = item['value']
            status = item.get('status')
            lines.append(f"- {item['feature']} ({item['kind']}): {metric}={value:.4f} [{status}]")
    return lines
def append_drift_summary(summary_path: Path, report: Mapping[str, Any], *, limit: int=5, drift_alert: bool | None=None) -> None:
    lines = build_drift_summary_lines(report, limit=limit, drift_alert=drift_alert)
    if not lines:
        return
    content = summary_path.read_text(encoding='utf-8') if summary_path.exists() else ''
    if content and (not content.endswith('\n')):
        content += '\n'
    if content:
        content += '\n'
    content += '\n'.join(lines) + '\n'
    summary_path.write_text(content, encoding='utf-8')
