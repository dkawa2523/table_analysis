from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

from ..common.collection_utils import to_container as _to_container, to_list as _to_list
from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.probability_utils import extract_positive_class_proba
from ..metrics.regression import REGRESSION_METRIC_ORDER
from ..registry.metrics import get_metric, metric_direction, metric_requires_proba, metric_supports_thresholding
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


def normalize_train_key(value: Any) -> str | None:
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


def _supports_param(class_path: str, param: str) -> tuple[bool, str | None]:
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        parameters = inspect.signature(cls.__init__).parameters.values()
    except (ImportError, AttributeError, TypeError, ValueError):
        return (False, "dependency_unavailable")
    if any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters):
        return (True, None)
    return ((param in inspect.signature(cls.__init__).parameters), None if param in inspect.signature(cls.__init__).parameters else "param_not_supported")


def resolve_regression_metrics(cfg: Any) -> list[str]:
    metrics = list(REGRESSION_METRIC_ORDER)
    extras = _to_list(_cfg_value(cfg, "eval.metrics.regression", None))
    if extras:
        metrics.extend(extras)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in metrics:
        key = normalize_train_key(name)
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
    strategy = normalize_train_key(_cfg_value(cfg, "eval.imbalance.strategy", "class_weight"))
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


def apply_resampling(strategy: str, X_train: Any, y_train: Any, *, seed: int) -> tuple[Any, Any, dict[str, Any] | None, str | None]:
    try:
        sampler_cls = (
            __import__("imblearn.over_sampling", fromlist=["RandomOverSampler"]).RandomOverSampler
            if strategy == "oversample"
            else __import__("imblearn.under_sampling", fromlist=["RandomUnderSampler"]).RandomUnderSampler
        )
    except ImportError:
        return (X_train, y_train, None, "optional_dependency_missing")
    try:
        sampler = sampler_cls(random_state=seed)
    except (TypeError, ValueError):
        sampler = sampler_cls()
    (X_resampled, y_resampled) = sampler.fit_resample(X_train, y_train)
    info = {"train_rows_before": int(len(y_train)), "train_rows_after": int(len(y_resampled))}
    return (X_resampled, y_resampled, info, None)


def apply_weight_strategy(*, model_variant: dict[str, Any], task_type: str, y_train: Any, n_classes: int | None, imbalance_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {"applied": False}
    class_path = model_variant.get("class_path")
    if isinstance(class_path, dict) and task_type:
        class_path = class_path.get(task_type)
    if not isinstance(class_path, str):
        report["reason"] = "missing_class_path"
        return (model_variant, report)
    strategy = imbalance_cfg.get("strategy")
    if strategy == "class_weight":
        class_weight = imbalance_cfg.get("class_weight")
        if class_weight is None:
            report["reason"] = "class_weight_null"
            return (model_variant, report)
        for param_name in ("class_weight", "class_weights"):
            (supports, reason) = _supports_param(class_path, param_name)
            if not supports:
                if param_name == "class_weight" and reason == "dependency_unavailable":
                    report["reason"] = "optional_dependency_missing"
                    return (model_variant, report)
                continue
            resolved = class_weight
            if param_name == "class_weights":
                if isinstance(class_weight, str) and class_weight.strip().lower() == "balanced":
                    try:
                        import numpy as np
                        from sklearn.utils.class_weight import compute_class_weight
                    except ImportError as exc:
                        raise RuntimeError("scikit-learn is required for class_weight balancing.") from exc
                    resolved = [float(weight) for weight in compute_class_weight("balanced", np.arange(int(n_classes or 0)), y_train).tolist()]
                elif isinstance(class_weight, (list, tuple)) and n_classes is not None and len(class_weight) == n_classes:
                    resolved = [float(item) for item in class_weight]
                elif isinstance(class_weight, dict) and n_classes is not None:
                    resolved = [1.0] * n_classes
                    for (key, weight) in class_weight.items():
                        try:
                            idx = int(key)
                        except (TypeError, ValueError, OverflowError):
                            continue
                        if 0 <= idx < n_classes:
                            resolved[idx] = float(weight)
                else:
                    resolved = None
            if param_name == "class_weights" and resolved is None:
                report["reason"] = "class_weights_unresolved" if n_classes is not None else "n_classes_unknown"
                return (model_variant, report)
            model_variant["params"][param_name] = resolved
            report.update({"applied": True, "detail": {"class_weight": class_weight}})
            return (model_variant, report)
        report["reason"] = "class_weight_not_supported"
        return (model_variant, report)
    if strategy == "pos_weight":
        if n_classes is not None and n_classes != 2:
            report["reason"] = "pos_weight_binary_only"
            return (model_variant, report)
        raw_pos_weight = imbalance_cfg.get("pos_weight")
        if raw_pos_weight is not None:
            try:
                pos_weight = float(raw_pos_weight)
            except (TypeError, ValueError, OverflowError):
                pos_weight = None
            else:
                pos_weight = pos_weight if pos_weight > 0 else None
        else:
            try:
                import numpy as np
            except ImportError:
                pos_weight = None
            else:
                counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
                neg = int(counts[0]) if counts.size > 0 else 0
                pos = int(counts[1]) if counts.size > 1 else 0
                pos_weight = float(neg / pos) if neg > 0 and pos > 0 else None
        if pos_weight is None:
            report["reason"] = "pos_weight_unavailable"
            return (model_variant, report)
        (supports, reason) = _supports_param(class_path, "scale_pos_weight")
        if not supports:
            report["reason"] = "optional_dependency_missing" if reason == "dependency_unavailable" else "pos_weight_not_supported"
            return (model_variant, report)
        model_variant["params"]["scale_pos_weight"] = pos_weight
        report.update({"applied": True, "detail": {"pos_weight": pos_weight}})
        return (model_variant, report)
    report["reason"] = "unknown_strategy"
    return (model_variant, report)


def is_binary_only_metric(name: str) -> bool:
    return (normalize_train_key(name) or "") in {"roc_auc", "auc", "pr_auc", "average_precision", "average_precision_score"}


def select_best_threshold(y_true: Any, y_proba: Any, *, metric_name: str, task_type: str, n_classes: int | None, grid: list[float], beta: float | None = None) -> tuple[float, float, str]:
    if not metric_supports_thresholding(metric_name, task_type):
        raise ValueError(f"thresholding.metric '{metric_name}' is not supported for threshold optimization.")
    metric_fn = get_metric(metric_name, task_type, n_classes=n_classes, beta=beta)
    direction = metric_direction(metric_name, task_type)
    positive_proba = extract_positive_class_proba(y_proba)
    best: tuple[float, float] | None = None
    for threshold in grid:
        score = float(metric_fn(y_true, (positive_proba >= threshold).astype(int), y_proba))
        is_better = best is None or (direction == "maximize" and score > best[1]) or (direction == "minimize" and score < best[1])
        if is_better:
            best = (float(threshold), score)
    if best is None:
        raise ValueError("Failed to select a valid threshold from eval.thresholding.grid.")
    return (best[0], best[1], direction)


def calibrate_classifier(model: Any, X_val: Any, y_val: Any, *, method: str, mode: str) -> Any:
    try:
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for calibration.") from exc
    if mode != "prefit":
        raise ValueError(f"Unsupported calibration mode: {mode}")
    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator


def build_calibration_report(y_true: Any, y_proba: Any, *, n_classes: int | None, n_bins: int = 10) -> tuple[dict[str, Any], list[float], list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for calibration reports.") from exc
    y_true_arr = np.asarray(y_true).reshape(-1)
    proba_arr = np.asarray(y_proba)
    if proba_arr.ndim == 1:
        confidence = proba_arr.astype(float)
        correct = (y_true_arr == 1).astype(float)
        mode = "binary_positive"
    elif proba_arr.ndim != 2:
        raise ValueError("predict_proba output must be a 1D or 2D array.")
    elif int(n_classes or proba_arr.shape[1]) <= 2:
        positive = proba_arr[:, 1] if proba_arr.shape[1] >= 2 else proba_arr[:, -1]
        confidence = positive.astype(float)
        correct = (y_true_arr == 1).astype(float)
        mode = "binary_positive"
    else:
        top_idx = np.argmax(proba_arr, axis=1)
        confidence = proba_arr[np.arange(proba_arr.shape[0]), top_idx].astype(float)
        correct = (top_idx == y_true_arr).astype(float)
        mode = "multiclass_top1"
    confidence = np.clip(np.asarray(confidence, dtype=float).reshape(-1), 0.0, 1.0)
    correct = np.clip(np.asarray(correct, dtype=float).reshape(-1), 0.0, 1.0)
    n_bins = max(int(n_bins or 10), 1)
    n_samples = int(confidence.shape[0])
    if n_samples == 0:
        return ({"n_samples": 0, "n_bins": n_bins, "mode": mode, "ece": None, "curve": []}, [], [])
    bin_indices = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bin_indices, minlength=n_bins)
    sum_conf = np.bincount(bin_indices, weights=confidence, minlength=n_bins)
    sum_acc = np.bincount(bin_indices, weights=correct, minlength=n_bins)
    mean_conf = np.divide(sum_conf, counts, out=np.zeros_like(sum_conf, dtype=float), where=counts > 0)
    mean_acc = np.divide(sum_acc, counts, out=np.zeros_like(sum_acc, dtype=float), where=counts > 0)
    curve = [
        {"bin": int(idx), "count": int(counts[idx]), "mean_confidence": float(mean_conf[idx]), "mean_accuracy": float(mean_acc[idx])}
        for idx in range(n_bins)
        if counts[idx] > 0
    ]
    ece = float(np.sum((counts / max(n_samples, 1)) * np.abs(mean_acc - mean_conf)))
    return (
        {"n_samples": n_samples, "n_bins": n_bins, "mode": mode, "ece": ece, "curve": curve},
        [float(item["mean_confidence"]) for item in curve],
        [float(item["mean_accuracy"]) for item in curve],
    )


def bootstrap_metric_ci(y_true: Any, y_pred: Any, y_proba: Any | None, *, metric_name: str, task_type: str, n_classes: int | None, n_boot: int, alpha: float, seed: int, beta: float | None, point_estimate: float | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for bootstrap confidence intervals.") from exc
    if metric_requires_proba(metric_name, task_type) and y_proba is None:
        raise ValueError("primary_metric requires predicted probabilities for CI.")
    metric_fn = get_metric(metric_name, task_type, n_classes=n_classes, beta=beta)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None
    n_samples = int(y_true_arr.shape[0])
    if n_samples <= 1:
        raise ValueError("bootstrap CI requires at least 2 validation samples.")
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    attempts = 0
    max_attempts = max(n_boot * 10, n_boot + 50)
    while len(scores) < n_boot and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        try:
            y_proba_sample = y_proba_arr[sample_idx] if y_proba_arr is not None else None
            score = float(metric_fn(y_true_arr[sample_idx], y_pred_arr[sample_idx], y_proba_sample))
        except (RuntimeError, TypeError, ValueError, FloatingPointError):
            continue
        if math.isfinite(score):
            scores.append(score)
    info = {"n_boot": int(n_boot), "n_boot_effective": int(len(scores)), "alpha": float(alpha), "seed": int(seed), "attempts": int(attempts)}
    if not scores:
        return (None, info)
    values = np.asarray(scores, dtype=float)
    mid = float(point_estimate) if point_estimate is not None else float(np.quantile(values, 0.5))
    return (
        {
            "low": float(np.quantile(values, alpha / 2.0)),
            "mid": mid,
            "high": float(np.quantile(values, 1.0 - alpha / 2.0)),
        },
        info,
    )


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
    "apply_resampling",
    "apply_weight_strategy",
    "build_plotly_confusion_matrix",
    "build_calibration_report",
    "build_plotly_roc_curve",
    "build_prediction_sample",
    "bootstrap_metric_ci",
    "calibrate_classifier",
    "is_binary_only_metric",
    "normalize_train_key",
    "resolve_calibration_settings",
    "resolve_ci_settings",
    "resolve_classification_metrics",
    "resolve_classification_mode",
    "resolve_imbalance_settings",
    "resolve_regression_metrics",
    "resolve_thresholding_settings",
    "resolve_uncertainty_settings",
    "resolve_viz_settings",
    "select_best_threshold",
]
