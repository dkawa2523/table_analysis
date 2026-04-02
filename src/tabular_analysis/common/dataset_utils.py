from __future__ import annotations
import hashlib
from pathlib import Path
TABULAR_SUFFIXES = (".csv", ".parquet", ".pq")
def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
def select_tabular_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted([p for p in path.rglob("*") if p.suffix.lower() in TABULAR_SUFFIXES])
        if candidates:
            return candidates[0]
        raise ValueError(f"No CSV/Parquet files found under: {path}")
    raise FileNotFoundError(str(path))
def load_tabular_frame(path: Path):
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pandas is required for tabular dataset loading.") from exc
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")
def derive_local_raw_dataset_id(dataset_path: str | Path) -> str:
    path = Path(dataset_path).expanduser().resolve()
    dataset_file = select_tabular_file(path)
    digest = hash_file(dataset_file)
    return f"local:{digest}"
