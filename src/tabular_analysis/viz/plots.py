from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence
from .render_common import render_placeholder
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    return plt
def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)
def _normalize_top_n(top_n: int, total: int) -> int:
    if top_n <= 0 or top_n > total:
        return total
    return top_n
def plot_feature_importance(feature_names: Sequence[str], importances: Sequence[float], output_path: str | Path, *, top_n: int=20, title: str | None=None) -> Path:
    import numpy as np
    names = [str(name) for name in feature_names]
    values = np.asarray(importances, dtype=float).reshape(-1)
    if len(names) != len(values):
        names = [f'feature_{idx}' for idx in range(len(values))]
    order = np.argsort(values)[::-1]
    top_n = _normalize_top_n(int(top_n), len(order))
    order = order[:top_n]
    names = [names[idx] for idx in order]
    values = values[order]
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or 'Feature Importance')
        return path
    (fig, ax) = plt.subplots(figsize=(8, max(2.5, 0.35 * len(names))))
    ax.barh(range(len(values)), values, color='#4C78A8')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('importance')
    ax.set_title(title or 'Feature Importance')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
def plot_regression_residuals(y_true: Sequence[float], y_pred: Sequence[float], output_path: str | Path, *, max_points: int=1000, title: str | None=None) -> Path:
    import numpy as np
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    n = y_true_arr.shape[0]
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        y_true_arr = y_true_arr[idx]
        y_pred_arr = y_pred_arr[idx]
    residuals = y_true_arr - y_pred_arr
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or 'Residuals vs Predicted')
        return path
    (fig, ax) = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(y_pred_arr, residuals, alpha=0.6, s=16, color='#F58518')
    ax.axhline(0.0, color='#333333', linewidth=1.0, linestyle='--')
    ax.set_xlabel('predicted')
    ax.set_ylabel('residual (true - pred)')
    ax.set_title(title or 'Residuals vs Predicted')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
def plot_interval_width_histogram(widths: Sequence[float], output_path: str | Path, *, bins: int=20, title: str | None=None) -> Path:
    import numpy as np
    values = np.asarray(widths, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([0.0], dtype=float)
    if bins <= 0:
        bins = 20
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or 'Interval Widths')
        return path
    (fig, ax) = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=bins, color='#54A24B', edgecolor='#FFFFFF')
    ax.set_xlabel('interval_width')
    ax.set_ylabel('count')
    ax.set_title(title or 'Interval Widths')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], output_path: str | Path, *, class_names: Sequence[str] | None=None, normalize: bool=True, title: str | None=None) -> Path:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    labels = None
    if class_names is not None:
        labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            display_cm = np.divide(display_cm, row_sums, out=np.zeros_like(display_cm), where=row_sums != 0)
    if class_names is None:
        if labels is not None:
            class_names = [str(label) for label in labels]
        else:
            values = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
            class_names = [str(value) for value in values]
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or ('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix'))
        return path
    (fig, ax) = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(display_cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title or ('Confusion Matrix (normalized)' if normalize else 'Confusion Matrix'))
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    annotate_values = display_cm if normalize else cm
    fmt = '.2f' if normalize else 'd'
    for i in range(annotate_values.shape[0]):
        for j in range(annotate_values.shape[1]):
            value = annotate_values[i, j]
            text_color = 'white' if display_cm[i, j] > display_cm.max() * 0.6 else 'black'
            ax.text(j, i, format(value, fmt), ha='center', va='center', color=text_color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
def plot_reliability_curve(confidence: Sequence[float], accuracy: Sequence[float], output_path: str | Path, *, title: str | None=None) -> Path:
    import numpy as np
    conf_arr = np.asarray(confidence, dtype=float).reshape(-1)
    acc_arr = np.asarray(accuracy, dtype=float).reshape(-1)
    if conf_arr.shape[0] != acc_arr.shape[0]:
        raise ValueError('confidence and accuracy must have the same length.')
    if conf_arr.size == 0:
        conf_arr = np.asarray([0.0], dtype=float)
        acc_arr = np.asarray([0.0], dtype=float)
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or 'Reliability Diagram')
        return path
    (fig, ax) = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], color='#999999', linestyle='--', label='perfect')
    ax.plot(conf_arr, acc_arr, color='#4C78A8', marker='o', label='observed')
    ax.set_xlabel('mean predicted probability')
    ax.set_ylabel('empirical accuracy')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or 'Reliability Diagram')
    ax.legend(loc='lower right')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
def write_confusion_matrix_csv(y_true: Sequence[int], y_pred: Sequence[int], output_path: str | Path, *, class_names: Sequence[str] | None=None) -> Path:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    labels = None
    if class_names is not None:
        labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    if class_names is None:
        values = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        class_names = [str(value) for value in values]
    path = _as_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true\\pred', *class_names])
        for (idx, row) in enumerate(cm):
            label = class_names[idx] if idx < len(class_names) else str(idx)
            writer.writerow([label, *[int(value) for value in row]])
    return path
def plot_roc_curve(y_true: Sequence[int], y_score: Sequence[float], output_path: str | Path, *, title: str | None=None) -> Path:
    import numpy as np
    from sklearn.metrics import auc, roc_curve
    y_true_arr = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim > 1:
        scores = scores[:, -1]
    (fpr, tpr, _) = roc_curve(y_true_arr, scores)
    roc_auc = auc(fpr, tpr)
    plt = _import_matplotlib()
    path = _as_path(output_path)
    if plt is None:
        render_placeholder(path, title or 'ROC Curve')
        return path
    (fig, ax) = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color='#54A24B', label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='#999999', linestyle='--')
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title(title or 'ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
