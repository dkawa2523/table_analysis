from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping

from omegaconf import OmegaConf

from tabular_analysis.clearml.pipeline_ui_contract import (
    build_pipeline_visible_hyperparameter_args,
)
from tabular_analysis.common.config_utils import set_cfg_value as _set_cfg_value
from tabular_analysis.processes.pipeline import _build_pipeline_plan
from tabular_analysis.processes.pipeline_support import (
    DEFAULT_PIPELINE_CONTROLLER_QUEUE,
    apply_pipeline_profile_defaults,
    build_pipeline_template_defaults,
    normalize_pipeline_profile,
)


def seed_runtime_defaults(seed_definition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for (key, value) in dict(seed_definition.get("shared_defaults") or {}).items()
        if value is not None
    }


def expected_pipeline_seed_defaults(resolved: Any) -> dict[str, Any]:
    if resolved.cfg is None:
        raise RuntimeError(f"Resolved config is missing for pipeline seed {resolved.spec.name}.")
    cfg = resolved.cfg
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    else:
        cfg = deepcopy(cfg)
    apply_pipeline_profile_defaults(cfg, resolved.spec.name)
    _set_cfg_value(cfg, "pipeline.plan_only", True)
    seed_grid_run_id = f"seed__{normalize_pipeline_profile(resolved.spec.name)}"
    plan = _build_pipeline_plan(
        cfg,
        seed_grid_run_id,
        child_execution="logging",
    )
    return build_pipeline_template_defaults(
        cfg=cfg,
        plan=plan,
        grid_run_id=seed_grid_run_id,
        pipeline_profile=resolved.spec.name,
    )


def pipeline_seed_sections_from_defaults(
    defaults: Mapping[str, Any],
    *,
    resolved: Any,
) -> dict[str, dict[str, Any]]:
    del defaults, resolved
    return {}


def published_pipeline_seed_sections(
    *,
    resolved: Any,
    seed_definition: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return pipeline_seed_sections_from_defaults(
        seed_runtime_defaults(seed_definition),
        resolved=resolved,
    )
def published_pipeline_seed_args(
    resolved: Any,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> list[str]:
    seed_defaults = dict(defaults or expected_pipeline_seed_defaults(resolved))
    return build_pipeline_visible_hyperparameter_args(
        seed_defaults,
        pipeline_profile=resolved.spec.name,
        include_bootstrap=True,
    )


def flatten_nested_override_paths(payload: Mapping[str, Any], *, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for (key, value) in payload.items():
        text = str(key)
        next_prefix = f"{prefix}.{text}" if prefix else text
        if isinstance(value, Mapping):
            paths.update(flatten_nested_override_paths(value, prefix=next_prefix))
            continue
        paths.add(next_prefix.replace(".", "/"))
    return paths


def expected_published_pipeline_seed_param_keys(
    resolved: Any,
) -> set[str]:
    defaults = expected_pipeline_seed_defaults(resolved)
    sections = pipeline_seed_sections_from_defaults(defaults, resolved=resolved)
    keys: set[str] = set()
    for item in published_pipeline_seed_args(
        resolved,
        defaults=defaults,
    ):
        key, _ = item.split("=", 1)
        keys.add(str(key).replace(".", "/"))
    for (section_name, payload) in sections.items():
        if str(section_name) == "properties":
            continue
        keys.update(flatten_nested_override_paths(payload))
    return keys


def canonical_seed_pipeline_hash(task: Any) -> str | None:
    data = getattr(task, "data", None)
    configuration = getattr(data, "configuration", None) if data is not None else None
    pipeline_item = configuration.get("Pipeline") if isinstance(configuration, dict) else None
    description = str(getattr(pipeline_item, "description", "") or "").strip()
    if description.lower().startswith("pipeline state:"):
        value = description.split(":", 1)[1].strip()
        if value:
            return f"{value}:1.0.0"
    config_value = str(getattr(pipeline_item, "value", "") or "").strip()
    if config_value:
        return f"{hashlib.md5(config_value.encode('utf-8')).hexdigest()}:1.0.0"
    return None


def seed_controller_queue_name(
    seed_definition: Mapping[str, Any],
    *,
    resolve_pipeline_controller_queue_name: Callable[[Mapping[str, Any]], str | None],
) -> str:
    plan = seed_definition.get("plan")
    if isinstance(plan, Mapping):
        queue_name = resolve_pipeline_controller_queue_name(dict(plan.get("queues") or {}))
        if str(queue_name or "").strip():
            return str(queue_name).strip()
    return DEFAULT_PIPELINE_CONTROLLER_QUEUE


__all__ = [
    "canonical_seed_pipeline_hash",
    "expected_pipeline_seed_defaults",
    "expected_published_pipeline_seed_param_keys",
    "flatten_nested_override_paths",
    "pipeline_seed_sections_from_defaults",
    "published_pipeline_seed_args",
    "published_pipeline_seed_sections",
    "seed_controller_queue_name",
    "seed_runtime_defaults",
]
