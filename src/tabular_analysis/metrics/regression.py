from __future__ import annotations
from typing import Any, Iterable
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
REGRESSION_METRIC_ORDER = ("r2", "mse", "rmse", "mae")
def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")
def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    try:
        import numpy as np  # type: ignore
        from sklearn.metrics import (  # type: ignore
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("scikit-learn is required for regression metrics.") from exc
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    values = {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }
    if metrics is None:
        return values
    selected: dict[str, float] = {}
    for name in metrics:
        key = _normalize_key(name)
        if not key:
            continue
        if key not in values:
            raise ValueError(f"Unsupported regression metric: {name}")
        selected[key] = values[key]
    return selected
