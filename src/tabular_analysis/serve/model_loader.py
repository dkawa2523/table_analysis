from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from ..platform_adapter_model import resolve_model_reference
from .settings import ServingSettings


@dataclass(frozen=True)
class ModelResolution:
    model_bundle_path: Path
    model_ref: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

def _load_registry_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"model registry state not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model_registry_state.json must contain a JSON object.")
    return payload


def _resolve_stage_ref(settings: ServingSettings) -> tuple[str, dict[str, Any]]:
    payload = _load_registry_state(settings.model_registry_state_path)
    usecases = payload.get("usecases") or {}
    if not isinstance(usecases, Mapping) or not usecases:
        raise ValueError("model registry state does not contain usecases.")

    usecase_id = settings.usecase_id
    if usecase_id is None:
        usecase_id = next(iter(usecases.keys()))

    usecase_payload = usecases.get(usecase_id) or {}
    stages = usecase_payload.get("stages") or {}
    stage_name = settings.model_stage or "production"
    stage_payload = stages.get(stage_name) or {}
    current = stage_payload.get("current") or {}
    model_ref = current.get("model_id")
    if not model_ref:
        raise ValueError(f"model registry state does not define current model for stage={stage_name} usecase={usecase_id}.")
    return (str(model_ref), {"usecase_id": str(usecase_id), "stage": str(stage_name), "registry_state_path": str(settings.model_registry_state_path)})


def resolve_model_bundle(
    model_bundle_path: str | Path | None = None,
    *,
    settings: ServingSettings | None = None,
) -> ModelResolution:
    settings = settings or ServingSettings.from_env()

    if model_bundle_path is not None:
        resolved = resolve_model_reference(model_id=str(model_bundle_path))
        return ModelResolution(model_bundle_path=resolved.model_bundle_path, model_ref=resolved.model_id or str(resolved.model_bundle_path), source=resolved.source, metadata=dict(resolved.metadata))

    if settings.model_ref:
        resolved = resolve_model_reference(model_id=settings.model_ref)
        return ModelResolution(model_bundle_path=resolved.model_bundle_path, model_ref=resolved.model_id or str(resolved.model_bundle_path), source=resolved.source, metadata=dict(resolved.metadata))

    if settings.model_stage:
        (model_ref, metadata) = _resolve_stage_ref(settings)
        resolved = resolve_model_reference(model_id=model_ref)
        return ModelResolution(model_bundle_path=resolved.model_bundle_path, model_ref=resolved.model_id or str(resolved.model_bundle_path), source="stage_registry", metadata={**metadata, **dict(resolved.metadata)})

    raise FileNotFoundError("MODEL_REF or MODEL_STAGE is required.")
