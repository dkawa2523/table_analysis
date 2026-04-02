from __future__ import annotations
from collections.abc import Mapping
from typing import Any
def to_list(values: Any) -> list[str]:
    if values is None:
        return []
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_list(values):
        return [str(v) for v in values if v is not None]
    if isinstance(values, (list, tuple, set)):
        return [str(v) for v in values if v is not None]
    return [str(values)]
def to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        try:
            container = OmegaConf.to_container(value, resolve=True)
        except (AttributeError, TypeError, ValueError):
            container = None
        if isinstance(container, Mapping):
            return dict(container)
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}
def to_container(value: Any, *, resolve: bool = True) -> Any:
    if value is None:
        return None
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        try:
            return OmegaConf.to_container(value, resolve=resolve)
        except (AttributeError, TypeError, ValueError):
            return value
    return value
def dedupe_texts(values: Any) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in values or []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
def stringify_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): stringify_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [stringify_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
