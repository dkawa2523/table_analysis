from __future__ import annotations
from ..common.config_utils import normalize_task_type as _normalize_task_type
from ..common.probability_utils import extract_positive_class_proba
from typing import Any, Callable
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def _normalize_key(value: Any) -> str:
    text = str(value or '').strip().lower()
    return text.replace('-', '_')
def metric_requires_proba(name: str, task_type: str) -> bool:
    key = _normalize_key(name)
    task = _normalize_task_type(task_type)
    if task != 'classification':
        return False
    return key in ('roc_auc', 'auc', 'log_loss', 'logloss', 'cross_entropy', 'pr_auc', 'average_precision', 'average_precision_score')
def metric_supports_thresholding(name: str, task_type: str) -> bool:
    """Return True if the metric can be optimized via a classification threshold."""
    key = _normalize_key(name)
    task = _normalize_task_type(task_type)
    if task != 'classification':
        return False
    if metric_requires_proba(key, task):
        return False
    return key in ('accuracy', 'acc', 'f1', 'f1_score', 'balanced_accuracy', 'fbeta', 'f_beta')
def metric_direction(name: str, task_type: str) -> str:
    key = _normalize_key(name)
    task = _normalize_task_type(task_type)
    if task == 'classification':
        if key in ('log_loss', 'logloss', 'cross_entropy'):
            return 'minimize'
        return 'maximize'
    if key in ('r2', 'r2_score'):
        return 'maximize'
    return 'minimize'
def _ensure_2d_proba(y_proba: Any, *, n_classes: int | None) -> Any:
    import numpy as np
    arr = np.asarray(y_proba)
    if arr.ndim == 1:
        if n_classes is not None and n_classes != 2:
            raise ValueError('probability array must be 2D for multi-class log_loss.')
        arr = np.stack([1.0 - arr, arr], axis=1)
    if arr.ndim != 2:
        raise ValueError('probability array must be 1D or 2D.')
    if n_classes is not None and arr.shape[1] != n_classes:
        raise ValueError('probability columns do not match n_classes.')
    return arr
def get_metric(name: str, task_type: str='regression', **kwargs: Any) -> Callable:
    key = _normalize_key(name)
    if not key:
        raise ValueError('metric name is required.')
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, fbeta_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError('scikit-learn is required for metrics.') from exc
    task = _normalize_task_type(task_type)
    if task == 'classification':
        n_classes = kwargs.get('n_classes')
        average = kwargs.get('average')
        if key in ('accuracy', 'acc'):
            return lambda y_true, y_pred, y_proba=None: float(accuracy_score(y_true, y_pred))
        if key in ('f1', 'f1_score', 'f1_macro', 'f1_micro', 'f1_weighted'):
            avg = average
            if key.endswith('_macro'):
                avg = 'macro'
            elif key.endswith('_micro'):
                avg = 'micro'
            elif key.endswith('_weighted'):
                avg = 'weighted'
            if avg is None:
                if isinstance(n_classes, int) and n_classes > 2:
                    avg = 'macro'
                else:
                    avg = 'binary'
            return lambda y_true, y_pred, y_proba=None: float(f1_score(y_true, y_pred, average=avg))
        if key in ('fbeta', 'f_beta', 'fbeta_score'):
            avg = average
            if avg is None:
                if isinstance(n_classes, int) and n_classes > 2:
                    avg = 'macro'
                else:
                    avg = 'binary'
            beta = kwargs.get('beta', 1.0)
            try:
                beta = float(beta)
            except (TypeError, ValueError):
                beta = 1.0
            return lambda y_true, y_pred, y_proba=None: float(fbeta_score(y_true, y_pred, beta=beta, average=avg))
        if key in ('balanced_accuracy', 'balanced_acc'):
            return lambda y_true, y_pred, y_proba=None: float(balanced_accuracy_score(y_true, y_pred))
        if key in ('roc_auc', 'auc'):
            def _roc_auc(y_true, y_pred=None, y_proba=None):
                if n_classes is not None and n_classes != 2:
                    raise ValueError('roc_auc is supported for binary classification only.')
                if y_proba is None:
                    raise ValueError('roc_auc requires predicted probabilities.')
                return float(roc_auc_score(y_true, extract_positive_class_proba(y_proba, error_message='probability array must have at least 2 columns for roc_auc.')))
            return _roc_auc
        if key in ('pr_auc', 'average_precision', 'average_precision_score'):
            def _pr_auc(y_true, y_pred=None, y_proba=None):
                if n_classes is not None and n_classes != 2:
                    raise ValueError('pr_auc is supported for binary classification only.')
                if y_proba is None:
                    raise ValueError('pr_auc requires predicted probabilities.')
                return float(average_precision_score(y_true, extract_positive_class_proba(y_proba, error_message='probability array must have at least 2 columns for roc_auc.')))
            return _pr_auc
        if key in ('log_loss', 'logloss', 'cross_entropy'):
            def _log_loss(y_true, y_pred=None, y_proba=None):
                if y_proba is None:
                    raise ValueError('log_loss requires predicted probabilities.')
                proba = _ensure_2d_proba(y_proba, n_classes=n_classes)
                labels = list(range(int(n_classes))) if n_classes else None
                return float(log_loss(y_true, proba, labels=labels))
            return _log_loss
        raise ValueError(f'Unsupported metric: {name}')
    if key in ('rmse', 'root_mean_squared_error'):
        return lambda y_true, y_pred, y_proba=None: float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if key in ('mse', 'mean_squared_error'):
        return lambda y_true, y_pred, y_proba=None: float(mean_squared_error(y_true, y_pred))
    if key in ('mae', 'mean_absolute_error'):
        return lambda y_true, y_pred, y_proba=None: float(mean_absolute_error(y_true, y_pred))
    if key in ('r2', 'r2_score'):
        return lambda y_true, y_pred, y_proba=None: float(r2_score(y_true, y_pred))
    raise ValueError(f'Unsupported metric: {name}')
