from __future__ import annotations
from typing import Any, Mapping, Sequence
from .render_common import plotly_go
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def _sample_points(y_true: Sequence[float], y_pred: Sequence[float], max_points: int) -> tuple[Any, Any]:
    import numpy as np
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    if max_points > 0 and y_true_arr.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(y_true_arr.shape[0], size=max_points, replace=False)
        y_true_arr = y_true_arr[idx]
        y_pred_arr = y_pred_arr[idx]
    return (y_true_arr, y_pred_arr)
def build_regression_metrics_table(metrics: Mapping[str, float], *, metric_order: Sequence[str]=('r2', 'mse', 'rmse', 'mae')) -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    ordered: list[tuple[str, float]] = []
    seen = set()
    for key in metric_order:
        if key in metrics:
            ordered.append((key, float(metrics[key])))
            seen.add(key)
    for (key, value) in metrics.items():
        if key not in seen:
            ordered.append((str(key), float(value)))
    labels = [name for (name, _) in ordered]
    values = [f'{value:.6g}' for (_, value) in ordered]
    fig = go.Figure(data=[go.Table(header=dict(values=['metric', 'value'], fill_color='#F2F2F2', align='left'), cells=dict(values=[labels, values], align='left'))])
    fig.update_layout(title='Regression Metrics', margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_true_pred_scatter(y_true: Sequence[float], y_pred: Sequence[float], *, r2: float | None=None, max_points: int=1000, title: str='True vs Predicted') -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    (y_true_arr, y_pred_arr) = _sample_points(y_true, y_pred, max_points)
    if y_true_arr.size == 0:
        return None
    min_val = float(np.nanmin([np.nanmin(y_true_arr), np.nanmin(y_pred_arr)]))
    max_val = float(np.nanmax([np.nanmax(y_true_arr), np.nanmax(y_pred_arr)]))
    title_text = title if r2 is None else f'{title} (R2={r2:.3f})'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true_arr, y=y_pred_arr, mode='markers', marker=dict(size=6, color='#4C78A8', opacity=0.7), name='predictions'))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='#333333', dash='dash'), name='y=x'))
    fig.update_layout(title=title_text, xaxis_title='true', yaxis_title='predicted', margin=dict(l=40, r=20, t=40, b=40))
    return fig
def build_residuals_plot(y_true: Sequence[float], y_pred: Sequence[float], *, max_points: int=1000, title: str='Residuals vs Predicted') -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    (y_true_arr, y_pred_arr) = _sample_points(y_true, y_pred, max_points)
    if y_true_arr.size == 0:
        return None
    residuals = y_true_arr - y_pred_arr
    fig = go.Figure(go.Scatter(x=y_pred_arr, y=residuals, mode='markers', marker=dict(size=6, color='#F58518', opacity=0.7)))
    fig.add_trace(go.Scatter(x=[float(np.nanmin(y_pred_arr)), float(np.nanmax(y_pred_arr))], y=[0.0, 0.0], mode='lines', line=dict(color='#333333', dash='dash'), name='zero'))
    fig.update_layout(title=title, xaxis_title='predicted', yaxis_title='residual (true - pred)', margin=dict(l=40, r=20, t=40, b=40))
    return fig
