from __future__ import annotations

from typing import Any

_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)


def extract_positive_class_proba(
    y_proba: Any,
    *,
    error_message: str = "predict_proba output must include positive-class probabilities.",
) -> Any:
    try:
        import numpy as np
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("numpy is required for positive-class probability extraction.") from exc

    arr = np.asarray(y_proba)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    raise ValueError(error_message)

