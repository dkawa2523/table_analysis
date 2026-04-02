from __future__ import annotations
from typing import Any, Dict, Mapping
def infer_schema(df) -> Dict[str, Any]:
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    null_count = df.isna().sum()
    fields: Dict[str, Any] = {}
    for col in df.columns:
        key = str(col)
        count = int(null_count[col])
        rate = float(count / rows) if rows else 0.0
        fields[key] = {
            "dtype": str(df[col].dtype),
            "null_count": count,
            "null_rate": rate,
        }
    return {"rows": rows, "columns": cols, "fields": fields}
def extract_schema_dtypes(schema: Mapping[str, Any]) -> Dict[str, str]:
    fields = schema.get("fields")
    if not isinstance(fields, Mapping):
        return {}
    dtypes: Dict[str, str] = {}
    for name, info in fields.items():
        dtype = None
        if isinstance(info, Mapping):
            dtype = info.get("dtype")
        elif info is not None:
            dtype = info
        if dtype is None:
            continue
        dtypes[str(name)] = str(dtype)
    return dtypes
