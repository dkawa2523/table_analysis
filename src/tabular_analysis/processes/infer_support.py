from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..common.config_utils import cfg_value as _cfg_value, normalize_str as _normalize_str
from ..common.model_reference import build_infer_reference


def resolve_batch_execution_mode(cfg: Any, *, clearml_enabled: bool) -> str:
    execution = _normalize_str(_cfg_value(cfg, "infer.batch.execution")) or "inline"
    execution = execution.lower()
    if execution not in {"inline", "clearml_children"}:
        raise ValueError("infer.batch.execution must be inline or clearml_children.")
    if execution == "clearml_children" and not clearml_enabled:
        raise ValueError("infer.batch.execution=clearml_children requires run.clearml.enabled=true.")
    return execution


def build_model_reference_payload(meta: Mapping[str, Any], *, model_bundle_path: Path) -> dict[str, str | None]:
    legacy_model_id = _normalize_str(meta.get("model_id")) or str(model_bundle_path)
    reference_kind = _normalize_str(meta.get("reference_kind"))
    explicit_model_id = _normalize_str(meta.get("model_id"))
    if reference_kind == "train_task_artifact" and _normalize_str(meta.get("train_task_id")):
        reference = build_infer_reference(
            model_id=explicit_model_id,
            registry_model_id=meta.get("registry_model_id"),
            train_task_id=meta.get("train_task_id"),
        )
        if reference.get("registry_model_id") is None:
            reference = build_infer_reference(train_task_id=meta.get("train_task_id"))
    else:
        reference = build_infer_reference(
            model_id=explicit_model_id,
            registry_model_id=meta.get("registry_model_id"),
            train_task_id=meta.get("train_task_id"),
        )
        if reference.get("infer_model_id") is None and reference.get("infer_train_task_id") is None:
            reference = build_infer_reference(model_id=legacy_model_id)
    return {
        "model_id": legacy_model_id,
        "train_task_id": reference.get("train_task_id"),
        "registry_model_id": reference.get("registry_model_id"),
        "infer_model_id": reference.get("infer_model_id"),
        "infer_train_task_id": reference.get("infer_train_task_id"),
        "reference_kind": reference.get("reference_kind"),
    }


__all__ = [
    "build_model_reference_payload",
    "resolve_batch_execution_mode",
]
