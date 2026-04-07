from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_PIPELINE_PROFILE = "pipeline"


@dataclass(frozen=True)
class PipelineUiField:
    concept_key: str

    @property
    def hyperparameters_key(self) -> str:
        return self.concept_key

    @property
    def operator_inputs_key(self) -> str:
        return self.concept_key


_COMMON_PIPELINE_UI_FIELDS = (
    PipelineUiField("run.usecase_id"),
    PipelineUiField("data.raw_dataset_id"),
    PipelineUiField("data.target_column"),
    PipelineUiField("data.split.strategy"),
    PipelineUiField("data.split.test_size"),
    PipelineUiField("data.split.seed"),
    PipelineUiField("eval.primary_metric"),
    PipelineUiField("eval.direction"),
    PipelineUiField("eval.task_type"),
    PipelineUiField("eval.cv_folds"),
    PipelineUiField("eval.seed"),
    PipelineUiField("eval.ci.enabled"),
    PipelineUiField("eval.calibration.enabled"),
    PipelineUiField("eval.classification.mode"),
    PipelineUiField("eval.classification.top_k"),
    PipelineUiField("eval.metrics.classification_multiclass"),
    PipelineUiField("pipeline.profile"),
    PipelineUiField("pipeline.model_set"),
    PipelineUiField("pipeline.grid.preprocess_variants"),
    PipelineUiField("pipeline.grid.model_variants"),
    PipelineUiField("pipeline.selection.enabled_preprocess_variants"),
    PipelineUiField("pipeline.selection.enabled_model_variants"),
)


_PIPELINE_UI_FIELDS_BY_PROFILE: dict[str, tuple[PipelineUiField, ...]] = {
    "pipeline": _COMMON_PIPELINE_UI_FIELDS,
    "train_model_full": _COMMON_PIPELINE_UI_FIELDS,
    "train_ensemble_full": (
        *_COMMON_PIPELINE_UI_FIELDS,
        PipelineUiField("ensemble.selection.enabled_methods"),
        PipelineUiField("ensemble.top_k"),
    ),
}


PIPELINE_UI_BOOTSTRAP_EXACT_KEYS = frozenset(
    {
        "task",
        "run.clearml.enabled",
        "run.clearml.execution",
        "run.clearml.project_root",
        "run.schema_version",
    }
)

PIPELINE_UI_BOOTSTRAP_PREFIXES = (
    "run.clearml.env.",
)


def _normalize_pipeline_profile(pipeline_profile: str) -> str:
    profile = str(pipeline_profile or "").strip() or DEFAULT_PIPELINE_PROFILE
    if profile not in _PIPELINE_UI_FIELDS_BY_PROFILE:
        raise ValueError(f"Unsupported pipeline profile for UI contract: {profile}")
    return profile


def _format_override_literal(value: Any) -> str:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def _merge_path(
    target: dict[str, Any],
    *,
    path: str,
    value: Any,
) -> None:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=False)
    except ImportError:
        pass
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        return
    cursor = target
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = value


def pipeline_ui_profiles() -> tuple[str, ...]:
    return tuple(_PIPELINE_UI_FIELDS_BY_PROFILE.keys())


def pipeline_ui_fields(pipeline_profile: str) -> tuple[PipelineUiField, ...]:
    return _PIPELINE_UI_FIELDS_BY_PROFILE[_normalize_pipeline_profile(pipeline_profile)]


def pipeline_ui_parameter_whitelist(pipeline_profile: str) -> tuple[str, ...]:
    return tuple(field.concept_key for field in pipeline_ui_fields(pipeline_profile))


def pipeline_ui_hyperparameter_keys(pipeline_profile: str) -> tuple[str, ...]:
    return tuple(field.hyperparameters_key for field in pipeline_ui_fields(pipeline_profile))


def pipeline_ui_hyperparameter_key_for(
    pipeline_profile: str,
    concept_key: str,
) -> str | None:
    concept = str(concept_key or "").strip()
    if not concept:
        return None
    for field in pipeline_ui_fields(pipeline_profile):
        if field.concept_key == concept:
            return field.hyperparameters_key
    return None


def pipeline_ui_bootstrap_keys() -> tuple[str, ...]:
    return tuple(sorted(PIPELINE_UI_BOOTSTRAP_EXACT_KEYS))


def pipeline_ui_bootstrap_key_allowed(key: str) -> bool:
    text = str(key or "").strip()
    return bool(text) and (
        text in PIPELINE_UI_BOOTSTRAP_EXACT_KEYS
        or any(text.startswith(prefix) for prefix in PIPELINE_UI_BOOTSTRAP_PREFIXES)
    )


def build_pipeline_operator_inputs_payload(
    values: Mapping[str, Any],
    *,
    pipeline_profile: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    normalized_values = {
        str(key): value
        for (key, value) in dict(values).items()
        if value is not None
    }
    for field in pipeline_ui_fields(pipeline_profile):
        if field.concept_key not in normalized_values:
            continue
        _merge_path(
            payload,
            path=field.operator_inputs_key,
            value=normalized_values[field.concept_key],
        )
    return payload


def build_pipeline_visible_hyperparameter_args(
    values: Mapping[str, Any],
    *,
    pipeline_profile: str,
    include_bootstrap: bool = False,
) -> list[str]:
    normalized_values = {
        str(key): value
        for (key, value) in dict(values).items()
        if value is not None
    }
    args: list[str] = []
    seen: set[str] = set()

    if include_bootstrap:
        bootstrap_keys = [
            str(key)
            for key in normalized_values.keys()
            if pipeline_ui_bootstrap_key_allowed(str(key))
        ]
        for key in sorted(bootstrap_keys):
            if key in seen:
                continue
            seen.add(key)
            args.append(f"{key}={_format_override_literal(normalized_values[key])}")

    task_value = normalized_values.get("task")
    if task_value is not None and "task" not in seen:
        seen.add("task")
        args.insert(0, f"task={_format_override_literal(task_value)}")

    for field in pipeline_ui_fields(pipeline_profile):
        key = field.hyperparameters_key
        if key in seen or key not in normalized_values:
            continue
        seen.add(key)
        args.append(f"{key}={_format_override_literal(normalized_values[key])}")
    return args


__all__ = [
    "DEFAULT_PIPELINE_PROFILE",
    "PIPELINE_UI_BOOTSTRAP_EXACT_KEYS",
    "PIPELINE_UI_BOOTSTRAP_PREFIXES",
    "PipelineUiField",
    "build_pipeline_operator_inputs_payload",
    "build_pipeline_visible_hyperparameter_args",
    "pipeline_ui_bootstrap_key_allowed",
    "pipeline_ui_bootstrap_keys",
    "pipeline_ui_fields",
    "pipeline_ui_hyperparameter_key_for",
    "pipeline_ui_hyperparameter_keys",
    "pipeline_ui_parameter_whitelist",
    "pipeline_ui_profiles",
]
