from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..registry.models import resolve_model_variant_optional_extra


_TASKS_WITHOUT_EXTRAS = frozenset({"dataset_register", "preprocess", "leaderboard", "pipeline"})


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = _normalize_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def resolve_model_variant_name_from_overrides(overrides: Mapping[str, Any]) -> str | None:
    candidates = (
        "group/model",
        "group.model.model_variant.name",
        "model_variant.name",
        "infer.model_variant",
    )
    for key in candidates:
        value = _normalize_name(overrides.get(key))
        if value:
            return value
    return None


def resolve_required_uv_extras(
    *,
    task_name: str | None,
    model_variant_name: str | None = None,
    infer_mode: str | None = None,
    explicit_extras: Iterable[str] | None = None,
    explicit_extras_provided: bool = False,
    train_ensemble_requires_model_extra: bool = False,
    infer_fallback_extra: str | None = "models",
) -> list[str]:
    if explicit_extras_provided:
        return _dedupe(explicit_extras or [])

    normalized_task = _normalize_name(task_name)
    extras: list[str] = []
    model_extra = resolve_model_variant_optional_extra(model_variant_name)

    if normalized_task in _TASKS_WITHOUT_EXTRAS:
        extras = []
    elif normalized_task == "train_model":
        extras = [model_extra] if model_extra else []
    elif normalized_task == "train_ensemble":
        extras = [model_extra] if train_ensemble_requires_model_extra and model_extra else []
    elif normalized_task == "infer":
        extras = [model_extra] if model_extra else []
        if not extras and infer_fallback_extra:
            extras = [infer_fallback_extra]

    if _normalize_name(infer_mode) == "optimize":
        extras.append("optuna")
    return _dedupe(extras)


__all__ = [
    "resolve_model_variant_name_from_overrides",
    "resolve_required_uv_extras",
]
