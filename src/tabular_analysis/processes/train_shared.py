from __future__ import annotations

import math
from typing import Any

from ..common.collection_utils import to_container as _to_container, to_list as _to_list
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..metrics.regression import REGRESSION_METRIC_ORDER
from ..viz.render_common import plotly_go as _plotly_go

_DEFAULT_THRESHOLD_GRID = [i / 100 for i in range(5, 100, 5)]
_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
    TypeError,
    ValueError,
    OverflowError,
)


def _normalize_key(value: Any) -> str | None:
    text = _normalize_str(value)
    return None if text is None else text.lower().replace("-", "_")


def _coerce_default(value: Any, caster: Any, default: Any) -> Any:
    try:
        return caster(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _normalize_threshold_grid(values: Any) -> list[float]:
    container = _to_container(values)
    if container is None:
        return []
    raw_values = list(container) if isinstance(container, (list, tuple, set)) else [container]
    grid: list[float] = []
    for value in raw_values:
        try:
            num = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(num) or num < 0.0 or num > 1.0:
            continue
        grid.append(float(num))
    return sorted(set(grid)) if grid else []


def _resolve_enabled_text(
    cfg: Any,
    *,
    enabled_path: str,
    value_path: str,
    default: str,
) -> tuple[bool, str]:
    enabled = bool(_cfg_value(cfg, enabled_path, False))
    value = (_normalize_str(_cfg_value(cfg, value_path, default)) or default).lower()
    return (enabled, value)


def _validate_enabled_choice(
    *,
    enabled: bool,
    value: str,
    allowed: tuple[str, ...],
    message: str,
) -> None:
    if enabled and value not in allowed:
        raise ValueError(message)


def _normalize_class_weight(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("", "none", "null"):
            return None
        if key == "balanced":
            return "balanced"
    return value


def resolve_regression_metrics(cfg: Any) -> list[str]:
    metrics = list(REGRESSION_METRIC_ORDER)
    extras = _to_list(_cfg_value(cfg, "eval.metrics.regression", None))
    if extras:
        metrics.extend(extras)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in metrics:
        key = _normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def resolve_viz_settings(cfg: Any) -> dict[str, Any]:
    return {
        "enabled": bool(_cfg_value(cfg, "viz.enabled", True)),
        "heavy_enabled": bool(_cfg_value(cfg, "viz.heavy.enabled", False)),
        "max_features": _coerce_default(_cfg_value(cfg, "viz.max_features", 20), int, 20),
        "max_points": _coerce_default(_cfg_value(cfg, "viz.max_points", 1000), int, 1000),
        "confusion_normalize": bool(_cfg_value(cfg, "viz.confusion_normalize", True)),
        "roc_curve": bool(_cfg_value(cfg, "viz.roc_curve", True)),
    }


def resolve_thresholding_settings(cfg: Any) -> dict[str, Any]:
    enabled = bool(_cfg_value(cfg, "eval.thresholding.enabled", False))
    metric = _normalize_str(_cfg_value(cfg, "eval.thresholding.metric", "f1")) or "f1"
    grid: list[float] = []
    if enabled:
        raw_grid = _cfg_value(cfg, "eval.thresholding.grid", None)
        grid = _normalize_threshold_grid(raw_grid)
        if not grid and raw_grid is None:
            grid = list(_DEFAULT_THRESHOLD_GRID)
        if not grid:
            raise ValueError("eval.thresholding.grid must include values between 0 and 1.")
    return {"enabled": enabled, "metric": metric, "grid": grid}


def resolve_calibration_settings(cfg: Any) -> dict[str, Any]:
    (enabled, method) = _resolve_enabled_text(
        cfg,
        enabled_path="eval.calibration.enabled",
        value_path="eval.calibration.method",
        default="sigmoid",
    )
    (_, mode) = _resolve_enabled_text(
        cfg,
        enabled_path="eval.calibration.enabled",
        value_path="eval.calibration.mode",
        default="prefit",
    )
    _validate_enabled_choice(
        enabled=enabled,
        value=method,
        allowed=("sigmoid", "isotonic"),
        message="eval.calibration.method must be 'sigmoid' or 'isotonic'.",
    )
    _validate_enabled_choice(
        enabled=enabled,
        value=mode,
        allowed=("prefit",),
        message="eval.calibration.mode must be 'prefit'.",
    )
    return {"enabled": enabled, "method": method, "mode": mode}


def resolve_uncertainty_settings(cfg: Any) -> dict[str, Any]:
    (enabled, method) = _resolve_enabled_text(
        cfg,
        enabled_path="eval.uncertainty.enabled",
        value_path="eval.uncertainty.method",
        default="conformal_split",
    )
    alpha = _coerce_default(_cfg_value(cfg, "eval.uncertainty.alpha", 0.1), float, 0.1)
    use_abs_residual = bool(_cfg_value(cfg, "eval.uncertainty.use_abs_residual", True))
    _validate_enabled_choice(
        enabled=enabled,
        value=method,
        allowed=("conformal_split",),
        message="eval.uncertainty.method must be 'conformal_split'.",
    )
    if enabled and not 0.0 < alpha < 1.0:
        raise ValueError("eval.uncertainty.alpha must be between 0 and 1.")
    return {
        "enabled": enabled,
        "method": method,
        "alpha": alpha,
        "use_abs_residual": use_abs_residual,
    }


def resolve_ci_settings(cfg: Any) -> dict[str, Any]:
    enabled = bool(_cfg_value(cfg, "eval.ci.enabled", False))
    n_boot = _coerce_default(_cfg_value(cfg, "eval.ci.n_boot", 200), int, 200)
    alpha = _coerce_default(_cfg_value(cfg, "eval.ci.alpha", 0.05), float, 0.05)
    seed = _coerce_default(_cfg_value(cfg, "eval.ci.seed", 0), int, 0)
    if enabled and n_boot <= 0:
        raise ValueError("eval.ci.n_boot must be > 0.")
    if enabled and not 0.0 < alpha < 1.0:
        raise ValueError("eval.ci.alpha must be between 0 and 1.")
    return {"enabled": enabled, "n_boot": n_boot, "alpha": alpha, "seed": seed}


def resolve_imbalance_settings(cfg: Any) -> dict[str, Any]:
    enabled = bool(_cfg_value(cfg, "eval.imbalance.enabled", False))
    strategy = _normalize_key(_cfg_value(cfg, "eval.imbalance.strategy", "class_weight"))
    class_weight = _normalize_class_weight(_cfg_value(cfg, "eval.imbalance.class_weight", None))
    pos_weight = _cfg_value(cfg, "eval.imbalance.pos_weight", None)
    if enabled and strategy not in ("class_weight", "pos_weight", "oversample", "undersample"):
        raise ValueError(
            "eval.imbalance.strategy must be class_weight, pos_weight, oversample, or undersample."
        )
    return {
        "enabled": enabled,
        "strategy": strategy,
        "class_weight": class_weight,
        "pos_weight": pos_weight,
    }


def resolve_classification_mode(cfg: Any, *, n_classes: int) -> str:
    mode = (_normalize_str(_cfg_value(cfg, "eval.classification.mode", "auto")) or "auto").lower()
    if mode not in ("auto", "binary", "multiclass"):
        raise ValueError("eval.classification.mode must be auto, binary, or multiclass.")
    if mode == "auto":
        return "binary" if n_classes == 2 else "multiclass"
    if mode == "binary" and n_classes != 2:
        raise ValueError("eval.classification.mode=binary requires exactly 2 classes.")
    if mode == "multiclass" and n_classes < 3:
        raise ValueError("eval.classification.mode=multiclass requires at least 3 classes.")
    return mode


def resolve_classification_metrics(
    cfg: Any,
    *,
    classification_mode: str,
    n_classes: int | None,
    imbalance_enabled: bool,
) -> list[str]:
    if classification_mode == "multiclass":
        metrics = _to_list(_cfg_value(cfg, "eval.metrics.classification_multiclass", None))
        if not metrics:
            metrics = ["accuracy", "f1_macro", "logloss"]
    else:
        metrics = _to_list(_cfg_value(cfg, "eval.metrics.classification_binary", None))
        if not metrics:
            metrics = ["accuracy", "f1", "log_loss"]
            if n_classes == 2:
                metrics.append("roc_auc")
    if imbalance_enabled:
        extra = _to_list(_cfg_value(cfg, "eval.metrics.classification_imbalance", None))
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


def build_prediction_sample(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None,
    *,
    max_rows: int = 5,
) -> Any | None:
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
    payload: dict[str, Any] = {"y_true": y_true_arr[:n], "y_pred": y_pred_arr[:n]}
    if y_proba is not None:
        proba_arr = np.asarray(y_proba)
        if proba_arr.ndim == 2:
            col = 1 if proba_arr.shape[1] > 1 else 0
            payload["pred_proba"] = proba_arr[:n, col]
        else:
            payload["pred_proba"] = proba_arr.reshape(-1)[:n]
    return pd.DataFrame(payload).head(max_rows)


def build_plotly_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    *,
    class_names: list[str] | None,
    normalize: bool,
) -> Any | None:
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
        with np.errstate(divide="ignore", invalid="ignore"):
            display_cm = np.divide(
                display_cm,
                row_sums,
                out=np.zeros_like(display_cm),
                where=row_sums != 0,
            )
    if class_names is None:
        values = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        class_names = [str(value) for value in values]
    fig = go.Figure(
        go.Heatmap(
            z=display_cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix (normalized)" if normalize else "Confusion Matrix",
        xaxis_title="predicted",
        yaxis_title="true",
        margin=dict(l=40, r=20, t=40, b=40),
    )
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
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="chance",
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="false positive rate",
        yaxis_title="true positive rate",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


__all__ = [
    "build_plotly_confusion_matrix",
    "build_plotly_roc_curve",
    "build_prediction_sample",
    "resolve_calibration_settings",
    "resolve_ci_settings",
    "resolve_classification_metrics",
    "resolve_classification_mode",
    "resolve_imbalance_settings",
    "resolve_regression_metrics",
    "resolve_thresholding_settings",
    "resolve_uncertainty_settings",
    "resolve_viz_settings",
]
