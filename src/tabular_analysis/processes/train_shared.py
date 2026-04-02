from __future__ import annotations
from ..common.collection_utils import to_list as _to_list
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..viz.render_common import plotly_go as _plotly_go
from typing import Any
_RECOVERABLE_ERRORS = (AttributeError, ImportError, ModuleNotFoundError, RuntimeError, TypeError, ValueError, OverflowError)
def resolve_classification_mode(cfg: Any, *, n_classes: int) -> str:
    mode = _normalize_str(_cfg_value(cfg, 'eval.classification.mode', 'auto')) or 'auto'
    mode = mode.lower()
    if mode not in ('auto', 'binary', 'multiclass'):
        raise ValueError('eval.classification.mode must be auto, binary, or multiclass.')
    if mode == 'auto':
        return 'binary' if n_classes == 2 else 'multiclass'
    if mode == 'binary' and n_classes != 2:
        raise ValueError('eval.classification.mode=binary requires exactly 2 classes.')
    if mode == 'multiclass' and n_classes < 3:
        raise ValueError('eval.classification.mode=multiclass requires at least 3 classes.')
    return mode
def resolve_classification_metrics(cfg: Any, *, classification_mode: str, n_classes: int | None, imbalance_enabled: bool) -> list[str]:
    metrics: list[str] = []
    if classification_mode == 'multiclass':
        metrics = _to_list(_cfg_value(cfg, 'eval.metrics.classification_multiclass', None))
        if not metrics:
            metrics = ['accuracy', 'f1_macro', 'logloss']
    else:
        metrics = _to_list(_cfg_value(cfg, 'eval.metrics.classification_binary', None))
        if not metrics:
            metrics = ['accuracy', 'f1', 'log_loss']
            if n_classes == 2:
                metrics.append('roc_auc')
    if imbalance_enabled:
        extra = _to_list(_cfg_value(cfg, 'eval.metrics.classification_imbalance', None))
        if extra:
            metrics.extend(extra)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in metrics:
        key = _normalize_str(name)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
def build_prediction_sample(y_true: Any, y_pred: Any, y_proba: Any | None, *, max_rows: int=5) -> Any | None:
    try:
        import numpy as np
        import pandas as pd
    except _RECOVERABLE_ERRORS:
        return None
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = min(len(y_true_arr), len(y_pred_arr))
    if n <= 0:
        return None
    payload: dict[str, Any] = {'y_true': y_true_arr[:n], 'y_pred': y_pred_arr[:n]}
    if y_proba is not None:
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim == 2:
            col = 1 if proba_arr.shape[1] > 1 else 0
            payload['pred_proba'] = proba_arr[:n, col]
        else:
            payload['pred_proba'] = proba_arr.reshape(-1)[:n]
    return pd.DataFrame(payload).head(max_rows)
def build_plotly_confusion_matrix(y_true: Any, y_pred: Any, *, class_names: list[str] | None, normalize: bool) -> Any | None:
    go = _plotly_go()
    if go is None:
        return None
    try:
        import numpy as np
        from sklearn.metrics import confusion_matrix
    except _RECOVERABLE_ERRORS:
        return None
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape[0] == 0 or y_pred_arr.shape[0] == 0:
        return None
    labels = list(range(len(class_names))) if class_names is not None else None
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            display_cm = np.divide(display_cm, row_sums, out=np.zeros_like(display_cm), where=row_sums != 0)
    if class_names is None:
        values = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        class_names = [str(value) for value in values]
    fig = go.Figure(go.Heatmap(z=display_cm, x=class_names, y=class_names, colorscale='Blues', showscale=True))
    fig.update_layout(title='Confusion Matrix (normalized)' if normalize else 'Confusion Matrix', xaxis_title='predicted', yaxis_title='true', margin=dict(l=40, r=20, t=40, b=40))
    return fig
def build_plotly_roc_curve(y_true: Any, y_score: Any) -> Any | None:
    go = _plotly_go()
    if go is None:
        return None
    try:
        import numpy as np
        from sklearn.metrics import auc, roc_curve
    except _RECOVERABLE_ERRORS:
        return None
    y_true_arr = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim > 1:
        scores = scores[:, -1]
    if y_true_arr.shape[0] == 0:
        return None
    (fpr, tpr, _) = roc_curve(y_true_arr, scores)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.3f}'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='chance'))
    fig.update_layout(title='ROC Curve', xaxis_title='false positive rate', yaxis_title='true positive rate', margin=dict(l=40, r=20, t=40, b=40))
    return fig
