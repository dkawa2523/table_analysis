from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
def resolve_config_dir(explicit: str | None, anchor_file: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--config-dir does not exist: {path}")
        return path
    env = Path.cwd() / "conf"
    if env.exists():
        return env
    fallback = Path(anchor_file).resolve().parents[2] / "conf"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("conf/ directory was not found. Run from repository root or set --config-dir.")
def _resolve_config_dir(explicit: str | None, anchor_file: str) -> Path:
    return resolve_config_dir(explicit, anchor_file)
def compose_config(config_dir: Path, config_name: str, overrides: Iterable[str]) -> Any:
    from hydra import compose, initialize_config_dir  # type: ignore
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))
__all__ = [
    "resolve_config_dir",
    "_resolve_config_dir",
    "compose_config",
]
