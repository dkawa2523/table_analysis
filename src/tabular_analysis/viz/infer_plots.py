from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from .render_common import fallback_image, plotly_go
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
_FRAME_RECOVERABLE_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)
_NUMERIC_RECOVERABLE_ERRORS = (TypeError, ValueError)
def _to_dataframe(value: Any, *, max_rows: int | None=None) -> Any | None:
    if value is None:
        return None
    try:
        import pandas as pd
    except _OPTIONAL_IMPORT_ERRORS:
        return None
    if isinstance(value, pd.DataFrame):
        df = value.copy()
    elif isinstance(value, Mapping):
        df = pd.DataFrame([dict(value)])
    elif isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        if not value:
            df = pd.DataFrame()
        elif all((isinstance(item, Mapping) for item in value)):
            df = pd.DataFrame([dict(item) for item in value])
        else:
            df = pd.DataFrame(value)
    else:
        return None
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    return df
def _limit_columns(df: Any, max_columns: int | None) -> Any:
    if df is None:
        return None
    if max_columns is None or max_columns <= 0:
        return df
    try:
        n_cols = int(df.shape[1])
    except _FRAME_RECOVERABLE_ERRORS:
        return df
    if n_cols <= max_columns:
        return df
    return df.iloc[:, :max_columns]
def _stringify(value: Any) -> str:
    if value is None:
        return ''
    try:
        import pandas as pd
        if pd.isna(value):
            return ''
    except (_OPTIONAL_IMPORT_ERRORS, TypeError, ValueError):
        pass
    text = str(value)
    if len(text) > 80:
        return text[:77] + '...'
    return text
def build_input_output_table(input_sample: Any, output_sample: Any, *, max_rows: int=5, max_input_columns: int=20, max_output_columns: int=12, title: str='Input -> Output', output_path: Path | None=None) -> Any | Path | None:
    input_df = _to_dataframe(input_sample, max_rows=max_rows)
    output_df = _to_dataframe(output_sample, max_rows=max_rows)
    if input_df is None and output_df is None:
        return None
    if input_df is not None and output_df is not None:
        try:
            limit = min(len(input_df), len(output_df))
        except _FRAME_RECOVERABLE_ERRORS:
            limit = None
        if limit is not None and limit >= 0:
            input_df = input_df.head(limit)
            output_df = output_df.head(limit)
    input_df = _limit_columns(input_df, max_input_columns)
    output_df = _limit_columns(output_df, max_output_columns)
    if input_df is not None:
        input_df = input_df.copy()
        input_df.columns = [f'in.{col}' for col in input_df.columns]
    if output_df is not None:
        output_df = output_df.copy()
        output_df.columns = [f'out.{col}' for col in output_df.columns]
    if input_df is None:
        combined = output_df
    elif output_df is None:
        combined = input_df
    else:
        combined = input_df.reset_index(drop=True)
        combined = combined.join(output_df.reset_index(drop=True), how='outer')
    if combined is None:
        return None
    go = plotly_go()
    if go is None:
        return fallback_image(output_path, title)
    columns = list(getattr(combined, 'columns', []))
    values = []
    for col in columns:
        try:
            col_values = combined[col].tolist()
        except _FRAME_RECOVERABLE_ERRORS:
            col_values = []
        values.append([_stringify(value) for value in col_values])
    fig = go.Figure(data=[go.Table(header=dict(values=[str(col) for col in columns], fill_color='#E8EEF7', align='left'), cells=dict(values=values, fill_color='#F9FBFD', align='left'))])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_prediction_histogram(values: Sequence[Any], *, title: str='Prediction Distribution', bins: int=20, output_path: Path | None=None) -> Any | Path | None:
    numeric: list[float] = []
    for value in values:
        try:
            num = float(value)
        except _NUMERIC_RECOVERABLE_ERRORS:
            continue
        if math.isfinite(num):
            numeric.append(num)
    if not numeric:
        return fallback_image(output_path, title)
    go = plotly_go()
    if go is None:
        return fallback_image(output_path, title)
    if bins <= 0:
        bins = 20
    fig = go.Figure(data=[go.Histogram(x=numeric, nbinsx=bins, marker_color='#4C78A8')])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
def build_label_distribution(labels: Sequence[Any], *, title: str='Prediction Labels', output_path: Path | None=None) -> Any | Path | None:
    counts: dict[str, int] = {}
    for value in labels:
        key = _stringify(value) or 'unknown'
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return fallback_image(output_path, title)
    go = plotly_go()
    if go is None:
        return fallback_image(output_path, title)
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    names = [item[0] for item in items]
    values = [item[1] for item in items]
    fig = go.Figure(data=[go.Bar(x=names, y=values, marker_color='#F58518')])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=40, b=20))
    return fig
