from __future__ import annotations
from pathlib import Path
from .config_utils import resolve_repo_root
def resolve_repo_root_fallback(*, marker_dir: str = "conf", fallback: Path | None = None) -> Path:
    """Resolve repository root and keep a fallback for CLI/tools scripts."""
    return resolve_repo_root(marker_dir=marker_dir, fallback=fallback)
