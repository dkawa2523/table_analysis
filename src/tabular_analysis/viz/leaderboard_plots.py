from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Sequence
from .render_common import plotly_go
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_NUMERIC_RECOVERABLE_ERRORS = (TypeError, ValueError)
def _format_value(value: Any) -> str:
    if value is None:
        return 'n/a'
    try:
        num = float(value)
    except _NUMERIC_RECOVERABLE_ERRORS:
        return str(value)
    if not math.isfinite(num):
        return 'n/a'
    return f'{num:.6g}'
def _label_from_row(row: dict[str, Any]) -> str:
    label = row.get('model_variant') or row.get('model_id') or row.get('train_task_ref')
    text = str(label) if label is not None else 'unknown'
    if len(text) > 30:
        text = text[:27] + '...'
    return f"{row.get('rank') or '?'}:{text}"
def build_leaderboard_table(rows: Sequence[dict[str, Any]], *, metric_names: Sequence[str], score_key: str='composite_score', score_label: str | None=None, max_rows: int=20, title: str='Leaderboard') -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    if max_rows <= 0:
        max_rows = len(rows)
    view_rows = list(rows[:max_rows])
    columns = [('rank', 'rank'), ('model_variant', 'model'), ('preprocess_variant', 'preprocess')]
    for name in metric_names:
        columns.append((name, name))
    columns.append((score_key, score_label or score_key))
    header_values = [label for (_, label) in columns]
    cell_values: list[list[Any]] = []
    for (key, _) in columns:
        col_items = []
        for row in view_rows:
            value = row.get(key)
            if key in ('rank', 'model_variant', 'preprocess_variant'):
                col_items.append(str(value) if value is not None else 'n/a')
            else:
                col_items.append(_format_value(value))
        cell_values.append(col_items)
    row_colors = ['#FFF4CC' if row.get('rank') == 1 else '#FFFFFF' for row in view_rows]
    fill_colors = [row_colors for _ in columns]
    fig = go.Figure(data=[go.Table(header=dict(values=header_values, fill_color='#F2F2F2', align='left'), cells=dict(values=cell_values, fill_color=fill_colors, align='left'))])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_top_k_bar(rows: Sequence[dict[str, Any]], *, score_key: str, score_label: str, title: str | None=None) -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    labels: list[str] = []
    scores: list[float] = []
    colors: list[str] = []
    for row in rows:
        labels.append(_label_from_row(row))
        value = row.get(score_key)
        try:
            score = float(value)
        except _NUMERIC_RECOVERABLE_ERRORS:
            score = 0.0
        scores.append(score)
        colors.append('#F28E2B' if row.get('rank') == 1 else '#4C78A8')
    fig = go.Figure(go.Bar(x=labels, y=scores, marker_color=colors))
    fig.update_layout(title=title or f'Top-K {score_label}', xaxis_title='rank/model', yaxis_title=score_label, margin=dict(l=40, r=20, t=40, b=80))
    return fig
def build_pareto_scatter(rows: Sequence[dict[str, Any]], *, x_metric: str='r2', y_metric: str='rmse', title: str | None=None) -> Any | None:
    go = plotly_go()
    if go is None:
        return None
    x_values: list[float] = []
    y_values: list[float] = []
    texts: list[str] = []
    colors: list[str] = []
    for row in rows:
        x = row.get(x_metric)
        y = row.get(y_metric)
        try:
            x_val = float(x)
            y_val = float(y)
        except _NUMERIC_RECOVERABLE_ERRORS:
            continue
        if not (math.isfinite(x_val) and math.isfinite(y_val)):
            continue
        x_values.append(x_val)
        y_values.append(y_val)
        texts.append(_label_from_row(row))
        colors.append('#F28E2B' if row.get('rank') == 1 else '#4C78A8')
    if not x_values:
        return None
    fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='markers+text', text=texts, textposition='top center', marker=dict(size=8, color=colors, opacity=0.8))])
    fig.update_layout(title=title or f'Pareto: {x_metric} vs {y_metric}', xaxis_title=x_metric, yaxis_title=y_metric, margin=dict(l=40, r=20, t=40, b=40))
    return fig
def write_top_k_bar_png(rows: Sequence[dict[str, Any]], output_path: Path, *, score_key: str, score_label: str, title: str | None=None) -> Path | None:
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    labels: list[str] = []
    scores: list[float] = []
    colors: list[str] = []
    for row in rows:
        labels.append(_label_from_row(row))
        value = row.get(score_key)
        try:
            score = float(value)
        except _NUMERIC_RECOVERABLE_ERRORS:
            score = 0.0
        scores.append(score)
        colors.append('#F28E2B' if row.get('rank') == 1 else '#4C78A8')
    (fig, ax) = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(scores)), scores, color=colors)
    ax.set_ylabel(score_label)
    ax.set_title(title or f'Top-K {score_label}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
