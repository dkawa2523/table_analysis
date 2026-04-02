from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import math
def normalize_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
def normalize_task_type(value: Any, *, default: str = "regression") -> str:
    key = normalize_str(value)
    if key is None:
        return default
    normalized = key.lower()
    if normalized in ("classification", "classifier", "class"):
        return "classification"
    if normalized in ("regression", "regressor", "reg"):
        return "regression"
    return default
def _normalize_task_type(value: Any, *, default: str = "regression") -> str:
    return normalize_task_type(value, default=default)
def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number
def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default
def cfg_value(cfg: Any, dotted_path: str, default: Any | None = None) -> Any:
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None:
        try:
            value = OmegaConf.select(cfg, dotted_path)
        except (AttributeError, KeyError, TypeError, ValueError):
            value = None
        if value is not None:
            return value
    current = cfg
    for key in dotted_path.split("."):
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return default if current is None else current
def set_cfg_value(cfg: Any, dotted_path: str, value: Any) -> bool:
    if cfg is None:
        return False
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        try:
            was_struct = OmegaConf.is_struct(cfg)
        except (AttributeError, TypeError, ValueError):
            was_struct = False
        if was_struct:
            try:
                OmegaConf.set_struct(cfg, False)
            except (AttributeError, TypeError, ValueError):
                pass
        try:
            OmegaConf.update(cfg, dotted_path, value, merge=False)
            return True
        except (AttributeError, TypeError, ValueError):
            return False
        finally:
            if was_struct:
                try:
                    OmegaConf.set_struct(cfg, True)
                except (AttributeError, TypeError, ValueError):
                    pass
    current = cfg
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        if isinstance(current, Mapping):
            if key not in current or not isinstance(current[key], Mapping):
                current[key] = {}
            current = current[key]
            continue
        if not hasattr(current, key) or getattr(current, key) is None:
            setattr(current, key, type("CfgNode", (), {})())
        current = getattr(current, key)
    last = keys[-1]
    if isinstance(current, Mapping):
        current[last] = value
        return True
    try:
        setattr(current, last, value)
        return True
    except (AttributeError, TypeError, ValueError):
        return False
def resolve_repo_root(*, marker_dir: str = "conf", fallback: Path | None = None) -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for base in candidates:
        for parent in [base, *base.parents]:
            if (parent / marker_dir).exists():
                return parent
    if fallback is not None:
        return fallback
    return Path.cwd()
