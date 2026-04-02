from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_api_keys(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [item.strip() for item in str(value).split(",")]
    return tuple(item for item in values if item)


@dataclass(frozen=True)
class ServingSettings:
    model_ref: str | None
    model_stage: str | None
    schema_mode: str
    api_keys: tuple[str, ...]
    audit_log_path: Path | None
    usecase_id: str | None
    model_registry_state_path: Path

    @property
    def strict_schema(self) -> bool:
        return self.schema_mode == "strict"

    @classmethod
    def from_env(cls) -> "ServingSettings":
        schema_mode = (_normalize_text(os.getenv("SCHEMA_MODE")) or "warn").lower()
        if schema_mode not in ("warn", "strict", "coerce"):
            schema_mode = "warn"

        registry_state = _normalize_text(os.getenv("MODEL_REGISTRY_STATE_PATH"))
        registry_path = Path(registry_state).expanduser() if registry_state else Path.cwd() / "model_registry_state.json"
        audit_log = _normalize_text(os.getenv("AUDIT_LOG_PATH"))

        model_ref = (
            _normalize_text(os.getenv("MODEL_REF"))
            or _normalize_text(os.getenv("TABULAR_MODEL_BUNDLE"))
            or _normalize_text(os.getenv("TABULAR_MODEL_BUNDLE_PATH"))
            or _normalize_text(os.getenv("MODEL_BUNDLE_PATH"))
        )
        model_stage = _normalize_text(os.getenv("MODEL_STAGE"))
        usecase_id = _normalize_text(os.getenv("USECASE_ID"))
        api_keys = _parse_api_keys(os.getenv("API_KEY"))
        return cls(
            model_ref=model_ref,
            model_stage=model_stage,
            schema_mode=schema_mode,
            api_keys=api_keys,
            audit_log_path=Path(audit_log).expanduser() if audit_log else None,
            usecase_id=usecase_id,
            model_registry_state_path=registry_path,
        )

