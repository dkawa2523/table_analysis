from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from .render_common import fallback_image, plotly_go
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_FRAME_RECOVERABLE_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)
_NUMERIC_RECOVERABLE_ERRORS = (TypeError, ValueError)
def _limit_columns(columns: Sequence[str], max_columns: int) -> list[str]:
    if max_columns <= 0 or max_columns >= len(columns):
        return list(columns)
    return list(columns)[:max_columns]
def _sample_dataframe(df: Any, max_rows: int):
    if max_rows <= 0:
        return df
    try:
        n_rows = int(df.shape[0])
    except _FRAME_RECOVERABLE_ERRORS:
        return df
    if n_rows > max_rows:
        try:
            return df.sample(n=max_rows, random_state=0)
        except _FRAME_RECOVERABLE_ERRORS:
            return df.head(max_rows)
    return df
def _slugify(value: Any, max_len: int=40) -> str:
    text = str(value)
    cleaned = []
    for ch in text:
        if '0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
            cleaned.append(ch)
        elif ch in ('-', '_'):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append('_')
    slug = ''.join(cleaned).strip('_')
    if not slug:
        slug = 'col'
    return slug[:max_len]
def _fallback_image(output_dir: Path | None, filename: str, title: str) -> Path | None:
    return fallback_image(output_dir / filename if output_dir is not None else None, title)
def summarize_dataframe(df: Any, *, columns: Sequence[str] | None=None) -> dict[str, Any]:
    try:
        rows = int(df.shape[0])
    except _FRAME_RECOVERABLE_ERRORS:
        rows = 0
    if columns is None:
        columns = list(getattr(df, 'columns', []))
    col_list = list(columns)
    col_count = len(col_list)
    missing_total = 0
    missing_rate = 0.0
    missing_columns = 0
    if rows > 0 and col_count > 0:
        try:
            missing_counts = df[col_list].isna().sum()
            missing_total = int(missing_counts.sum())
            missing_columns = int((missing_counts > 0).sum())
            denom = rows * col_count
            missing_rate = float(missing_total / denom) if denom else 0.0
        except _FRAME_RECOVERABLE_ERRORS:
            pass
    return {'rows': rows, 'columns': col_count, 'missing_total': missing_total, 'missing_rate': missing_rate, 'missing_columns': missing_columns}
def build_profile_summary(df: Any, *, feature_columns: Sequence[str], numeric_features: Sequence[str] | None=None, categorical_features: Sequence[str] | None=None) -> dict[str, Any]:
    summary = summarize_dataframe(df, columns=feature_columns)
    summary['features'] = summary.pop('columns')
    summary['numeric_features'] = len(list(numeric_features or []))
    summary['categorical_features'] = len(list(categorical_features or []))
    return summary
def build_head_table(df: Any, *, max_rows: int=5, max_columns: int=20, title: str='Head Sample', output_dir: Path | None=None) -> Any | Path | None:
    go = plotly_go()
    if go is None:
        return _fallback_image(output_dir, 'head_table.png', title)
    try:
        head = df.head(max_rows)
    except _FRAME_RECOVERABLE_ERRORS:
        return _fallback_image(output_dir, 'head_table.png', title)
    try:
        if max_columns > 0 and head.shape[1] > max_columns:
            head = head.iloc[:, :max_columns]
    except _FRAME_RECOVERABLE_ERRORS:
        pass
    columns = list(getattr(head, 'columns', []))
    values = []
    for col in columns:
        try:
            col_values = head[col].tolist()
        except _FRAME_RECOVERABLE_ERRORS:
            col_values = []
        values.append(['' if value is None else str(value) for value in col_values])
    fig = go.Figure(data=[go.Table(header=dict(values=[str(col) for col in columns], fill_color='#E8EEF7', align='left'), cells=dict(values=values, fill_color='#F9FBFD', align='left'))])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_missingness_bar(df: Any, *, columns: Sequence[str] | None=None, max_columns: int=30, title: str='Missing Rate by Column', output_dir: Path | None=None) -> Any | Path | None:
    go = plotly_go()
    if go is None:
        return _fallback_image(output_dir, 'missingness_bar.png', title)
    if columns is None:
        columns = list(getattr(df, 'columns', []))
    col_list = list(columns)
    if not col_list:
        return None
    try:
        missing_rate = df[col_list].isna().mean().sort_values(ascending=False)
    except _FRAME_RECOVERABLE_ERRORS:
        return _fallback_image(output_dir, 'missingness_bar.png', title)
    if max_columns > 0 and len(missing_rate) > max_columns:
        missing_rate = missing_rate.iloc[:max_columns]
    fig = go.Figure(go.Bar(x=missing_rate.values, y=[str(col) for col in missing_rate.index], orientation='h', marker_color='#4C78A8'))
    fig.update_layout(title=title, xaxis_title='missing_rate', yaxis_title='column', yaxis=dict(autorange='reversed'), margin=dict(l=40, r=20, t=40, b=40))
    return fig
def build_numeric_histograms(df: Any, columns: Sequence[str], *, max_columns: int=6, bins: int=30, sample_rows: int=5000, title_prefix: str='Numeric Histogram', output_dir: Path | None=None) -> list[tuple[str, Any | Path | None]]:
    column_list = _limit_columns(list(columns), max_columns)
    if not column_list:
        return []
    go = plotly_go()
    if go is None:
        return [(str(col), _fallback_image(output_dir, f'numeric_hist_{_slugify(col)}.png', f'{title_prefix}: {col}')) for col in column_list]
    try:
        import pandas as pd
    except _OPTIONAL_IMPORT_ERRORS:
        return []
    sampled = _sample_dataframe(df, sample_rows)
    results: list[tuple[str, Any | Path | None]] = []
    for col in column_list:
        try:
            series = pd.to_numeric(sampled[col], errors='coerce').dropna()
        except _FRAME_RECOVERABLE_ERRORS:
            series = None
        if series is None or series.empty:
            continue
        fig = go.Figure(go.Histogram(x=series, nbinsx=bins, marker_color='#4C78A8'))
        fig.update_layout(title=f'{title_prefix}: {col}', xaxis_title=str(col), yaxis_title='count', margin=dict(l=40, r=20, t=40, b=40))
        results.append((str(col), fig))
    return results
def build_categorical_topk_bars(df: Any, columns: Sequence[str], *, max_columns: int=6, top_k: int=10, sample_rows: int=5000, title_prefix: str='Top Categories', output_dir: Path | None=None) -> list[tuple[str, Any | Path | None]]:
    column_list = _limit_columns(list(columns), max_columns)
    if not column_list:
        return []
    go = plotly_go()
    if go is None:
        return [(str(col), _fallback_image(output_dir, f'categorical_topk_{_slugify(col)}.png', f'{title_prefix}: {col}')) for col in column_list]
    sampled = _sample_dataframe(df, sample_rows)
    results: list[tuple[str, Any | Path | None]] = []
    for col in column_list:
        try:
            counts = sampled[col].value_counts(dropna=True).head(top_k)
        except _FRAME_RECOVERABLE_ERRORS:
            counts = None
        if counts is None or counts.empty:
            continue
        labels = [str(idx) for idx in counts.index]
        fig = go.Figure(go.Bar(x=counts.values, y=labels, orientation='h', marker_color='#F58518'))
        fig.update_layout(title=f'{title_prefix}: {col}', xaxis_title='count', yaxis_title='category', yaxis=dict(autorange='reversed'), margin=dict(l=40, r=20, t=40, b=40))
        results.append((str(col), fig))
    return results
def build_target_distribution(df: Any, target_column: str, *, bins: int=30, top_k: int=10, sample_rows: int=5000, title: str='Target Distribution', output_dir: Path | None=None) -> Any | Path | None:
    if not target_column or target_column not in getattr(df, 'columns', []):
        return None
    try:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    series = df[target_column]
    if is_numeric_dtype(series) and (not is_bool_dtype(series)):
        results = build_numeric_histograms(df, [target_column], max_columns=1, bins=bins, sample_rows=sample_rows, title_prefix=title, output_dir=output_dir)
        return results[0][1] if results else _fallback_image(output_dir, 'target_distribution.png', title)
    results = build_categorical_topk_bars(df, [target_column], max_columns=1, top_k=top_k, sample_rows=sample_rows, title_prefix=title, output_dir=output_dir)
    return results[0][1] if results else _fallback_image(output_dir, 'target_distribution.png', title)
def build_profile_comparison_table(raw_summary: dict[str, Any], processed_summary: dict[str, Any], *, title: str='Raw vs Processed Summary', output_dir: Path | None=None) -> Any | Path | None:
    go = plotly_go()
    if go is None:
        return _fallback_image(output_dir, 'preprocess_summary.png', title)
    metrics = [('rows', 'rows'), ('features', 'features'), ('numeric_features', 'numeric_features'), ('categorical_features', 'categorical_features'), ('missing_rate', 'missing_rate'), ('missing_columns', 'missing_columns')]
    def _format_value(key: str, value: Any) -> str:
        if value is None:
            return '-'
        if key == 'missing_rate':
            try:
                return f'{float(value):.4f}'
            except _NUMERIC_RECOVERABLE_ERRORS:
                return str(value)
        try:
            return str(int(value))
        except _NUMERIC_RECOVERABLE_ERRORS:
            return str(value)
    metric_labels = [label for (_, label) in metrics]
    raw_values = [_format_value(key, raw_summary.get(key)) for (key, _) in metrics]
    processed_values = [_format_value(key, processed_summary.get(key)) for (key, _) in metrics]
    fig = go.Figure(data=[go.Table(header=dict(values=['metric', 'raw', 'processed'], fill_color='#E8EEF7', align='left'), cells=dict(values=[metric_labels, raw_values, processed_values], fill_color='#F9FBFD', align='left'))])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_missing_rate_comparison_bar(raw_rate: float, processed_rate: float, *, title: str='Missing Rate (raw vs processed)', output_dir: Path | None=None) -> Any | Path | None:
    go = plotly_go()
    if go is None:
        return _fallback_image(output_dir, 'missing_rate_comparison.png', title)
    fig = go.Figure(go.Bar(x=['raw', 'processed'], y=[raw_rate, processed_rate], marker_color=['#4C78A8', '#F58518']))
    fig.update_layout(title=title, xaxis_title='dataset', yaxis_title='missing_rate', margin=dict(l=40, r=20, t=40, b=40))
    return fig
