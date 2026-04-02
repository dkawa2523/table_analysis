from __future__ import annotations

from typing import Iterable

_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)


def infer_tabular_feature_types(df, feature_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    try:
        from pandas.api.types import is_bool_dtype, is_numeric_dtype
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("pandas is required for feature type inference.") from exc

    numeric: list[str] = []
    categorical: list[str] = []
    for col in feature_columns:
        col_name = str(col)
        series = df[col]
        if is_bool_dtype(series):
            categorical.append(col_name)
        elif is_numeric_dtype(series):
            numeric.append(col_name)
        else:
            categorical.append(col_name)
    return (numeric, categorical)

