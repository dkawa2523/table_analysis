from __future__ import annotations
import math
from typing import Any, Sequence
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_NUMERIC_COERCE_ERRORS = (TypeError, ValueError)
def _as_1d_array(values: Any, *, label: str):
    try:
        import numpy as np  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("numpy is required for conformal prediction intervals.") from exc
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        if arr.shape[1] == 1:
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"{label} must be 1D for conformal intervals.")
    return arr.reshape(-1)
def compute_split_conformal_quantile(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    alpha: float = 0.1,
    use_abs_residual: bool = True,
) -> float:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    true_arr = _as_1d_array(y_true, label="y_true")
    pred_arr = _as_1d_array(y_pred, label="y_pred")
    if true_arr.shape[0] != pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length for conformal quantile.")
    residuals = true_arr - pred_arr
    if use_abs_residual:
        try:
            import numpy as np  # type: ignore
        except _OPTIONAL_IMPORT_ERRORS as exc:
            raise RuntimeError("numpy is required for conformal prediction intervals.") from exc
        residuals = np.abs(residuals)
    try:
        import numpy as np  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("numpy is required for conformal prediction intervals.") from exc
    q = float(np.quantile(residuals, 1.0 - float(alpha)))
    if not math.isfinite(q):
        raise ValueError("Computed conformal quantile is not finite.")
    if q < 0:
        q = abs(q)
    return float(q)
def apply_split_conformal_interval(
    preds: Sequence[float],
    q: float,
) -> tuple[Any, Any]:
    pred_arr = _as_1d_array(preds, label="preds")
    try:
        q_value = float(q)
    except _NUMERIC_COERCE_ERRORS as exc:
        raise ValueError("q must be a numeric value.") from exc
    if not math.isfinite(q_value):
        raise ValueError("q must be finite.")
    if q_value < 0:
        raise ValueError("q must be non-negative.")
    lower = pred_arr - q_value
    upper = pred_arr + q_value
    return lower, upper
