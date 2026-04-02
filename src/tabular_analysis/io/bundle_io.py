from __future__ import annotations
from pathlib import Path
from typing import Any
_OPTIONAL_IMPORT_ERRORS = (ImportError, ModuleNotFoundError)
def save_bundle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("joblib is required for bundle serialization.") from exc
    joblib.dump(obj, path)
def load_bundle(path: Path) -> Any:
    try:
        import joblib  # type: ignore
    except _OPTIONAL_IMPORT_ERRORS as exc:
        raise RuntimeError("joblib is required for bundle serialization.") from exc
    return joblib.load(path)
