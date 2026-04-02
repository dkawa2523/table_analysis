from __future__ import annotations

from typing import Any, Mapping


def normalize_reference_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_infer_reference(
    *,
    model_id: Any = None,
    registry_model_id: Any = None,
    train_task_id: Any = None,
    train_task_ref: Any = None,
) -> dict[str, str | None]:
    normalized_model_id = normalize_reference_value(model_id)
    normalized_registry_model_id = normalize_reference_value(registry_model_id)
    normalized_train_task_id = normalize_reference_value(train_task_id)
    normalized_train_task_ref = normalize_reference_value(train_task_ref)
    infer_model_id = normalized_registry_model_id or normalized_model_id
    infer_train_task_id = None
    if infer_model_id is None:
        infer_train_task_id = normalized_train_task_id or normalized_train_task_ref
    reference_kind = None
    if normalized_registry_model_id:
        reference_kind = "registry_model"
    elif normalized_model_id:
        reference_kind = "local_bundle"
    elif infer_train_task_id:
        reference_kind = "train_task_artifact"
    return {
        "model_id": normalized_model_id,
        "registry_model_id": normalized_registry_model_id,
        "train_task_id": normalized_train_task_id,
        "train_task_ref": normalized_train_task_ref,
        "infer_model_id": infer_model_id,
        "infer_train_task_id": infer_train_task_id,
        "reference_kind": reference_kind,
    }


def resolve_preferred_infer_reference(*payloads: Mapping[str, Any] | None) -> dict[str, str | None]:
    priority = {"registry_model": 3, "local_bundle": 2, "train_task_artifact": 1, None: 0}
    best = build_infer_reference()
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        reference = build_infer_reference(
            model_id=payload.get("infer_model_id") or payload.get("model_id") or payload.get("recommended_model_id"),
            registry_model_id=payload.get("registry_model_id") or payload.get("recommended_registry_model_id"),
            train_task_id=payload.get("infer_train_task_id") or payload.get("train_task_id") or payload.get("recommended_train_task_id"),
            train_task_ref=payload.get("train_task_ref") or payload.get("recommended_train_task_ref"),
        )
        if priority.get(reference.get("reference_kind"), 0) > priority.get(best.get("reference_kind"), 0):
            best = reference
    return best


__all__ = [
    "build_infer_reference",
    "normalize_reference_value",
    "resolve_preferred_infer_reference",
]
