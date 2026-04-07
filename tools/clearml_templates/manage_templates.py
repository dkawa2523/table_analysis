#!/usr/bin/env python3
"""ClearML template task management (plan/apply/validate)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PLATFORM_SRC = _REPO.parent / "ml_platform_v1-master" / "src"
for _path in (_REPO, _SRC, _PLATFORM_SRC):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tabular_analysis.ops.clearml_identity import resolve_template_context
from tabular_analysis.clearml.pipeline_ui_contract import (
    pipeline_ui_bootstrap_key_allowed,
)
from tabular_analysis.clearml.pipeline_templates import (
    build_pipeline_template_properties,
    build_pipeline_template_tags,
    is_pipeline_template_name,
)
from tabular_analysis.clearml.live_cleanup import (
    deprecate_pipeline_task as _deprecate_pipeline_task,
)
from tabular_analysis.common.hydra_config import compose_config
from tabular_analysis.common.config_utils import set_cfg_value as _set_cfg_value
from tabular_analysis.processes.pipeline_support import (
    PIPELINE_RAW_DATASET_ID_SENTINEL,
    apply_pipeline_profile_defaults,
    build_pipeline_step_specs,
    build_pipeline_ui_parameter_whitelist,
    build_pipeline_visible_hyperparameter_args,
    normalize_pipeline_profile,
    resolve_pipeline_controller_queue_name,
)
from tabular_analysis.platform_adapter_clearml_env import (
    clearml_script_mismatches,
    resolve_clearml_script_spec,
    resolve_version_props,
)
from tabular_analysis.platform_adapter_clearml_policy import (
    ensure_clearml_project_system_tags as _ensure_clearml_project_system_tags,
    get_clearml_project_system_tags as _get_clearml_project_system_tags,
)
from tabular_analysis.platform_adapter_pipeline import (
    create_pipeline_seed_controller,
    load_pipeline_controller_from_task,
)
from tabular_analysis.platform_adapter_task_query import (
    clearml_task_exists,
    clearml_task_id,
    clearml_task_project_name,
    clearml_task_script,
    clearml_task_tags,
    clearml_task_type_from_obj,
    get_clearml_task_configuration,
)
from tabular_analysis.platform_adapter_task_ops import (
    clone_clearml_task,
    create_clearml_task,
    enqueue_clearml_task,
    ensure_clearml_task_requirements,
    ensure_clearml_task_properties,
    ensure_clearml_task_script,
    get_task_artifact_local_copy,
    ensure_clearml_task_tags,
    replace_clearml_task_hyperparameters,
    replace_clearml_task_tags,
    reset_clearml_task,
    set_clearml_task_project,
    set_clearml_task_configuration,
    set_clearml_task_runtime_properties,
    upload_clearml_task_artifact,
)
from tools.clearml_templates.drift_validate import (
    configuration_paths as _configuration_paths,
    iter_parameter_paths as _iter_parameter_paths,
    normalize_task_status as _normalize_task_status,
    pipeline_duplicate_visible_param_keys as _pipeline_duplicate_visible_param_keys,
    pipeline_disallowed_arg_keys as _pipeline_disallowed_arg_keys,
    pipeline_noncanonical_parameter_paths as _pipeline_noncanonical_parameter_paths,
    pipeline_parameter_keys as _pipeline_parameter_keys,
    task_artifact_names as _task_artifact_names,
    task_name as _task_name,
    task_parameters as _task_parameters,
    task_runtime as _task_runtime,
    task_user_properties as _task_user_properties,
)
from tools.clearml_templates.seed_publish import (
    canonical_seed_pipeline_hash as _canonical_seed_pipeline_hash,
    expected_pipeline_seed_defaults as _expected_pipeline_seed_defaults,
    pipeline_seed_sections_from_defaults as _pipeline_seed_sections_from_defaults,
    published_pipeline_seed_sections as _published_pipeline_seed_sections,
    seed_controller_queue_name as _seed_publish_seed_controller_queue_name,
    seed_runtime_defaults as _seed_runtime_defaults,
)
from tools.clearml_templates.seed_publish import (
    expected_published_pipeline_seed_param_keys as _seed_publish_expected_published_pipeline_seed_param_keys,
    published_pipeline_seed_args as _seed_publish_published_pipeline_seed_args,
)
from tools.clearml_templates.stale_cleanup import (
    HistoricalPipelineArchivePolicy as _HistoricalPipelineArchivePolicy,
    build_active_pipeline_seed_scope as _build_active_pipeline_seed_scope,
    cleanup_pipeline_history as _cleanup_pipeline_history,
)
from tabular_analysis.processes.pipeline import _build_pipeline_plan, build_pipeline_seed_controller


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    project_name: str
    task_name_template: str
    entrypoint: str
    default_overrides: list[str]
    requirements: list[str]
    tags: list[str]
    properties_minimal: dict[str, Any]


@dataclass(frozen=True)
class PlanContext:
    project_root: str
    usecase_id: str
    schema_version: str
    template_set_id: str
    solution_root: str
    pipeline_seed_namespace: str
    pipeline_root_group: str
    pipeline_runs_group: str
    templates_root_group: str
    step_templates_group: str
    runs_root_group: str
    group_map: dict[str, str]


@dataclass(frozen=True)
class ResolvedTemplateSpec:
    spec: TemplateSpec
    module: str | None
    script: str | None
    entry_args: list[str]
    overrides: list[str]
    expected_requirements: list[str]
    expected_tags: list[str]
    expected_properties: dict[str, Any]
    script_spec: Any
    cfg: Any | None = None




def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(data)


def _load_run_defaults(repo_root: Path) -> PlanContext:
    run_cfg_path = repo_root / "conf" / "run" / "base.yaml"
    layout_cfg_path = repo_root / "conf" / "clearml" / "project_layout.yaml"
    solution_root = "TabularAnalysis"
    pipeline_seed_namespace = ".pipelines"
    pipeline_root_group = "Pipelines"
    pipeline_runs_group = "Runs"
    templates_root_group = "Templates"
    step_templates_group = "Steps"
    runs_root_group = "Runs"
    group_map: dict[str, str] = {}
    if layout_cfg_path.exists():
        layout_cfg = OmegaConf.load(layout_cfg_path)
        solution_root = str(getattr(layout_cfg, "solution_root", None) or solution_root)
        pipeline_seed_namespace = str(getattr(layout_cfg, "pipeline_seed_namespace", None) or pipeline_seed_namespace)
        pipeline_root_group = str(getattr(layout_cfg, "pipeline_root_group", None) or pipeline_root_group)
        pipeline_runs_group = str(getattr(layout_cfg, "pipeline_runs_group", None) or pipeline_runs_group)
        templates_root_group = str(getattr(layout_cfg, "templates_root_group", None) or templates_root_group)
        step_templates_group = str(getattr(layout_cfg, "step_templates_group", None) or step_templates_group)
        runs_root_group = str(getattr(layout_cfg, "runs_root_group", None) or runs_root_group)
        raw_group_map = getattr(layout_cfg, "group_map", None)
        if raw_group_map is not None:
            rendered_group_map = OmegaConf.to_container(raw_group_map, resolve=False)
            if isinstance(rendered_group_map, dict):
                group_map = {str(key): str(value) for key, value in rendered_group_map.items()}
    if not run_cfg_path.exists():
        return PlanContext(
            project_root="MFG",
            usecase_id="TabularAnalysis",
            schema_version="v1",
            template_set_id="default",
            solution_root=solution_root,
            pipeline_seed_namespace=pipeline_seed_namespace,
            pipeline_root_group=pipeline_root_group,
            pipeline_runs_group=pipeline_runs_group,
            templates_root_group=templates_root_group,
            step_templates_group=step_templates_group,
            runs_root_group=runs_root_group,
            group_map=group_map,
        )
    cfg = OmegaConf.load(run_cfg_path)
    template_ctx = resolve_template_context(cfg)
    return PlanContext(
        project_root=str(template_ctx.project_root),
        usecase_id=str(template_ctx.usecase_id),
        schema_version=str(template_ctx.schema_version),
        template_set_id=str(template_ctx.template_set_id),
        solution_root=solution_root,
        pipeline_seed_namespace=pipeline_seed_namespace,
        pipeline_root_group=pipeline_root_group,
        pipeline_runs_group=pipeline_runs_group,
        templates_root_group=templates_root_group,
        step_templates_group=step_templates_group,
        runs_root_group=runs_root_group,
        group_map=group_map,
    )


def _load_code_ref_mode(repo_root: Path, override: str | None) -> str:
    if override:
        return str(override)
    run_cfg_path = repo_root / "conf" / "run" / "base.yaml"
    if not run_cfg_path.exists():
        return "branch"
    cfg = OmegaConf.load(run_cfg_path)
    clearml_cfg = getattr(cfg, "clearml", None)
    code_ref = getattr(clearml_cfg, "code_ref", None)
    value = getattr(code_ref, "mode", None) if code_ref is not None else None
    return str(value) if value else "branch"


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_value(value: Any, ctx: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormatDict(ctx))
    if isinstance(value, list):
        return [_format_value(item, ctx) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(val, ctx) for key, val in value.items()}
    return value


def _load_templates(spec_path: Path, ctx: PlanContext) -> list[TemplateSpec]:
    raw = _load_yaml(spec_path)
    templates = raw.get("templates")
    if not isinstance(templates, dict):
        raise ValueError("templates must be a mapping")

    context = {
        "project_root": ctx.project_root,
        "usecase_id": ctx.usecase_id,
        "schema_version": ctx.schema_version,
        "template_set_id": ctx.template_set_id,
        "solution_root": ctx.solution_root,
        "pipeline_seed_namespace": ctx.pipeline_seed_namespace,
        "pipeline_root_group": ctx.pipeline_root_group,
        "pipeline_runs_group": ctx.pipeline_runs_group,
        "templates_root_group": ctx.templates_root_group,
        "step_templates_group": ctx.step_templates_group,
        "runs_root_group": ctx.runs_root_group,
        "group_map": ctx.group_map,
    }
    specs: list[TemplateSpec] = []
    for name, payload in templates.items():
        if not isinstance(payload, dict):
            raise ValueError(f"template {name} must be a mapping")
        rendered = _format_value(payload, context)
        specs.append(
            TemplateSpec(
                name=str(name),
                project_name=str(rendered.get("project_name", "")),
                task_name_template=str(rendered.get("task_name_template", "")),
                entrypoint=str(rendered.get("entrypoint", "")),
                default_overrides=[str(item) for item in (rendered.get("default_overrides") or [])],
                requirements=[str(item) for item in (rendered.get("requirements") or [])],
                tags=[str(item) for item in (rendered.get("tags") or [])],
                properties_minimal=dict(rendered.get("properties_minimal") or {}),
            )
        )
    return specs


def _parse_entrypoint(entrypoint: str) -> tuple[str | None, str | None, list[str]]:
    parts = shlex.split(entrypoint)
    if not parts:
        raise ValueError("entrypoint is empty")
    if parts[0] in {"python", "python3"}:
        parts = parts[1:]
    if not parts:
        raise ValueError("entrypoint missing command")
    if parts[0] == "-m":
        if len(parts) < 2:
            raise ValueError("entrypoint module is missing")
        module = parts[1]
        args = parts[2:]
        return (module, None, args)
    script = parts[0]
    args = parts[1:]
    return (None, script, args)


def _detect_repo_url(repo_root: Path) -> str | None:
    candidates = [
        ["git", "config", "--get", "remote.origin.url"],
        ["git", "remote", "get-url", "origin"],
    ]
    for cmd in candidates:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            value = proc.stdout.strip()
            if value:
                return value
    return None


def _detect_repo_branch(repo_root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    if not value or value.upper() == "HEAD":
        return None
    return value


def _clearml_config_present(repo_root: Path) -> bool:
    env_path = os.getenv("CLEARML_CONFIG_FILE")
    if env_path:
        if Path(env_path).expanduser().exists():
            return True
    for key in ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY", "CLEARML_API_HOST", "CLEARML_WEB_HOST"):
        if os.getenv(key):
            return True
    for candidate in (
        repo_root / "clearml.conf",
        Path.home() / "clearml.conf",
        Path.home() / ".clearml.conf",
        Path.home() / ".config" / "clearml.conf",
    ):
        if candidate.exists():
            return True
    return False


def _plan_output(spec_path: Path, ctx: PlanContext, templates: list[TemplateSpec], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_path": str(spec_path),
        "context": {
            "project_root": ctx.project_root,
            "usecase_id": ctx.usecase_id,
            "schema_version": ctx.schema_version,
            "template_set_id": ctx.template_set_id,
            "solution_root": ctx.solution_root,
            "pipeline_seed_namespace": ctx.pipeline_seed_namespace,
            "pipeline_root_group": ctx.pipeline_root_group,
            "pipeline_runs_group": ctx.pipeline_runs_group,
            "templates_root_group": ctx.templates_root_group,
            "step_templates_group": ctx.step_templates_group,
            "runs_root_group": ctx.runs_root_group,
            "group_map": ctx.group_map,
        },
        "templates": [
            {
                "name": spec.name,
                "project_name": spec.project_name,
                "task_name": spec.task_name_template,
                "entrypoint": spec.entrypoint,
                "default_overrides": spec.default_overrides,
                "tags": spec.tags,
                "properties_minimal": spec.properties_minimal,
            }
            for spec in templates
        ],
        "outputs": {
            "lock_file": str(_repo_root() / "conf" / "clearml" / "templates.lock.yaml"),
            "plan_json": str(output_dir / "template_plan.json"),
            "plan_md": str(output_dir / "template_plan.md"),
        },
    }
    (output_dir / "template_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ClearML Template Task Plan",
        "",
        f"Generated: {payload['generated_at']}",
        f"Spec: `{payload['spec_path']}`",
        "",
        "## Context",
        f"- project_root: `{ctx.project_root}`",
        f"- usecase_id: `{ctx.usecase_id}`",
        f"- schema_version: `{ctx.schema_version}`",
        f"- template_set_id: `{ctx.template_set_id}`",
        f"- solution_root: `{ctx.solution_root}`",
        f"- pipeline_seed_namespace: `{ctx.pipeline_seed_namespace}`",
        f"- pipeline_root_group: `{ctx.pipeline_root_group}`",
        f"- pipeline_runs_group: `{ctx.pipeline_runs_group}`",
        f"- templates_root_group: `{ctx.templates_root_group}`",
        f"- step_templates_group: `{ctx.step_templates_group}`",
        f"- runs_root_group: `{ctx.runs_root_group}`",
        "",
        "## Templates",
    ]
    for spec in templates:
        lines.extend(
            [
                f"### {spec.name}",
                f"- project_name: `{spec.project_name}`",
                f"- task_name: `{spec.task_name_template}`",
                f"- entrypoint: `{spec.entrypoint}`",
                f"- default_overrides: `{', '.join(spec.default_overrides)}`",
                f"- tags: `{', '.join(spec.tags)}`",
                f"- properties_minimal: `{json.dumps(spec.properties_minimal, ensure_ascii=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Apply",
            "```bash",
            "python tools/clearml_templates/manage_templates.py --apply",
            "```",
            f"Lock file: `{payload['outputs']['lock_file']}`",
            "",
            "## Validate",
            "```bash",
            "python tools/clearml_templates/manage_templates.py --validate",
            "```",
        ]
    )
    (output_dir / "template_plan.md").write_text("\n".join(lines), encoding="utf-8")


def _load_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 2, "context": {}, "templates": {}}
    data = _load_yaml(path)
    if "templates" not in data or not isinstance(data.get("templates"), dict):
        data["templates"] = {}
    if "context" not in data or not isinstance(data.get("context"), dict):
        data["context"] = {}
    if "version" not in data:
        data["version"] = 2
    return data


def _save_lock(path: Path, payload: Mapping[str, Any]) -> None:
    content = OmegaConf.to_yaml(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_lock_template_entry(
    existing: Mapping[str, Any] | None,
    *,
    task_id: str,
    project_name: str,
    task_name: str,
    entrypoint: str,
    now: str,
    kind: str = "template",
) -> dict[str, Any]:
    payload = {
        "task_id": task_id,
        "project_name": project_name,
        "task_name": task_name,
        "entrypoint": entrypoint,
        "kind": kind,
        "updated_at": now,
    }
    existing_payload = dict(existing or {})
    stable_keys = ("task_id", "project_name", "task_name", "entrypoint", "kind")
    if all(str(existing_payload.get(key, "")) == str(payload[key]) for key in stable_keys):
        previous_updated_at = str(existing_payload.get("updated_at", "")).strip()
        if previous_updated_at:
            payload["updated_at"] = previous_updated_at
    return payload


def _lock_context_payload(ctx: PlanContext) -> dict[str, Any]:
    return {
        "project_root": ctx.project_root,
        "usecase_id": ctx.usecase_id,
        "schema_version": ctx.schema_version,
        "template_set_id": ctx.template_set_id,
        "solution_root": ctx.solution_root,
        "pipeline_seed_namespace": ctx.pipeline_seed_namespace,
        "pipeline_root_group": ctx.pipeline_root_group,
        "pipeline_runs_group": ctx.pipeline_runs_group,
        "templates_root_group": ctx.templates_root_group,
        "step_templates_group": ctx.step_templates_group,
        "runs_root_group": ctx.runs_root_group,
        "group_map": dict(ctx.group_map),
    }


def _lock_context(path: Path) -> dict[str, Any]:
    payload = _load_lock(path)
    context = payload.get("context") or {}
    return dict(context) if isinstance(context, Mapping) else {}


def _resolve_live_plan_context(
    *,
    defaults: PlanContext,
    lock_path: Path,
    project_root: str | None,
    usecase_id: str | None,
    schema_version: str | None,
    template_set_id: str | None,
) -> PlanContext:
    lock_context = _lock_context(lock_path)

    def _explicit_or_locked(name: str, explicit: str | None, fallback: str) -> str:
        explicit_value = str(explicit).strip() if explicit is not None else ""
        locked_value = str(lock_context.get(name, "")).strip()
        if explicit_value and locked_value and explicit_value != locked_value:
            raise ValueError(
                f"Lock context mismatch for {name}: explicit={explicit_value!r}, lock={locked_value!r}. "
                "Use the locked context or update the lock intentionally."
            )
        if explicit_value:
            return explicit_value
        if locked_value:
            return locked_value
        return str(fallback)

    resolved_project_root = _explicit_or_locked("project_root", project_root, "")
    if not resolved_project_root:
        raise ValueError(
            "project_root is required for --apply/--validate. "
            "Pass --project-root or create/update templates.lock.yaml with a context.project_root."
        )
    layout_keys = (
        "solution_root",
        "pipeline_seed_namespace",
        "pipeline_root_group",
        "pipeline_runs_group",
        "templates_root_group",
        "step_templates_group",
        "runs_root_group",
    )
    layout_drift = {
        key: {"lock": str(lock_context.get(key, "")), "current": str(getattr(defaults, key))}
        for key in layout_keys
        if lock_context.get(key) is not None and str(lock_context.get(key, "")) != str(getattr(defaults, key))
    }
    locked_group_map = lock_context.get("group_map")
    if locked_group_map is not None and dict(locked_group_map) != dict(defaults.group_map):
        layout_drift["group_map"] = {
            "lock": dict(locked_group_map),
            "current": dict(defaults.group_map),
        }
    if layout_drift:
        raise ValueError(
            "Lock layout drift detected. Update templates.lock.yaml intentionally after changing "
            f"the ClearML project layout: {layout_drift}"
        )
    return PlanContext(
        project_root=resolved_project_root,
        usecase_id=_explicit_or_locked("usecase_id", usecase_id, defaults.usecase_id),
        schema_version=_explicit_or_locked("schema_version", schema_version, defaults.schema_version),
        template_set_id=_explicit_or_locked("template_set_id", template_set_id, defaults.template_set_id),
        solution_root=str(defaults.solution_root),
        pipeline_seed_namespace=str(defaults.pipeline_seed_namespace),
        pipeline_root_group=str(defaults.pipeline_root_group),
        pipeline_runs_group=str(defaults.pipeline_runs_group),
        templates_root_group=str(defaults.templates_root_group),
        step_templates_group=str(defaults.step_templates_group),
        runs_root_group=str(defaults.runs_root_group),
        group_map=dict(defaults.group_map),
    )


def _normalized_args(args: Iterable[str]) -> list[str]:
    normalized = []
    for item in args:
        item = str(item).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        normalized.append(item)
    return normalized


def _arg_pairs(args: Iterable[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in _normalized_args(args):
        key, value = item.split("=", 1)
        pairs.append((str(key), str(value)))
    return pairs


def _pipeline_seed_bootstrap_overrides(args: Iterable[str]) -> list[str]:
    bootstrap_args: list[str] = []
    for item in _normalized_args(args):
        key, _ = item.split("=", 1)
        if not pipeline_ui_bootstrap_key_allowed(str(key)):
            continue
        bootstrap_args.append(item)
    return bootstrap_args


def _pipeline_seed_script_mismatches(spec: Any, script: Mapping[str, Any]) -> list[str]:
    raw = clearml_script_mismatches(spec, script)
    if isinstance(raw, bool):
        return ["script mismatch"] if raw else []
    return [
        error
        for error in raw
        if not str(error).startswith("version_num mismatch:")
    ]


def _published_pipeline_seed_args(
    resolved: ResolvedTemplateSpec,
    *,
    defaults: Mapping[str, Any] | None = None,
) -> list[str]:
    return _seed_publish_published_pipeline_seed_args(
        resolved,
        defaults=defaults,
    )


def _expected_published_pipeline_seed_param_keys(resolved: ResolvedTemplateSpec) -> set[str]:
    return _seed_publish_expected_published_pipeline_seed_param_keys(
        resolved,
    )


def _materialized_pipeline_seed_args(
    *,
    resolved: ResolvedTemplateSpec,
    seed_definition: Mapping[str, Any],
) -> list[str]:
    return build_pipeline_visible_hyperparameter_args(
        _seed_runtime_defaults(seed_definition),
        pipeline_profile=resolved.spec.name,
        include_bootstrap=True,
    )


def _seed_controller_queue_name(seed_definition: Mapping[str, Any]) -> str:
    return _seed_publish_seed_controller_queue_name(
        seed_definition,
        resolve_pipeline_controller_queue_name=resolve_pipeline_controller_queue_name,
    )


def _task_status_text(task_id: str) -> str:
    task = _load_clearml_task(task_id)
    status = getattr(task, "status", None)
    if status is None:
        data = getattr(task, "data", None)
        status = getattr(data, "status", None) if data is not None else None
    return _normalize_task_status(status)


def _wait_for_terminal_task_status(
    task_id: str,
    *,
    timeout_seconds: float=180.0,
    poll_seconds: float=1.0,
) -> str:
    deadline = time.time() + max(float(timeout_seconds), 0.0)
    last_status = ""
    while True:
        last_status = _task_status_text(task_id)
        if last_status in {"completed", "failed", "stopped"}:
            return last_status
        if time.time() >= deadline:
            return last_status
        time.sleep(max(float(poll_seconds), 0.1))


def _normalize_internal_compose_overrides(args: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for item in _normalized_args(args):
        text = str(item)
        if text.startswith("++"):
            normalized.append(text[1:])
            continue
        if text.startswith("+"):
            normalized.append(text[1:])
            continue
        normalized.append(text)
    return normalized


def _is_pipeline_template(spec: TemplateSpec) -> bool:
    return is_pipeline_template_name(spec.name)


def _compose_pipeline_template_cfg(repo_root: Path, spec: TemplateSpec, ctx: PlanContext, *, entry_args: list[str]) -> Any:
    config_dir = repo_root / "conf"
    overrides = [
        *_normalize_internal_compose_overrides(entry_args),
        *_normalize_internal_compose_overrides(spec.default_overrides),
        f"run.clearml.project_root={ctx.project_root}",
        f"run.clearml.template_set_id={ctx.template_set_id}",
        f"run.schema_version={ctx.schema_version}",
        f"run.usecase_id={ctx.usecase_id}",
        f"run.clearml.pipeline.project_name={spec.project_name}",
        f"task.project_name={spec.project_name}",
        f"run.clearml.task_name={spec.task_name_template}",
        f"data.raw_dataset_id={PIPELINE_RAW_DATASET_ID_SENTINEL}",
    ]
    cfg = compose_config(config_dir, "config", overrides)
    return apply_pipeline_profile_defaults(cfg, spec.name)


def _load_clearml_task(task_id: str) -> Any:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect template tasks.") from exc
    return ClearMLTask.get_task(task_id=str(task_id))


def _list_clearml_tasks(project_name: str) -> list[Any]:
    try:
        from clearml import Task as ClearMLTask
    except ImportError as exc:
        raise RuntimeError("clearml is required to inspect template tasks.") from exc
    try:
        tasks = ClearMLTask.get_tasks(
            project_name=str(project_name),
            allow_archived=True,
            task_filter={"order_by": ["-last_update"]},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to list ClearML tasks for project {project_name!r}: {exc}") from exc
    return list(tasks or [])


def _resolve_template_spec(
    spec: TemplateSpec,
    *,
    ctx: PlanContext,
    repo_root: Path,
    repo: str | None,
    branch: str | None,
    version_mode: str | None,
    cfg: Any | None = None,
) -> ResolvedTemplateSpec:
    module, script, entry_args = _parse_entrypoint(spec.entrypoint)
    entry_point = f"-m {module}" if module else script
    resolved_cfg = cfg
    if resolved_cfg is None and _is_pipeline_template(spec):
        resolved_cfg = _compose_pipeline_template_cfg(repo_root, spec, ctx, entry_args=entry_args)
    spec_cfg = resolved_cfg if resolved_cfg is not None else {"run": {"clearml": {"code_ref": {"mode": version_mode}}}}
    script_spec = resolve_clearml_script_spec(
        spec_cfg,
        entry_point_override=entry_point,
        repo_override=repo,
        branch_override=branch,
        version_mode_override=version_mode,
        task_name_override="pipeline" if _is_pipeline_template(spec) else spec.name,
        canonicalize_pipeline=False,
    )
    expected_tags = build_pipeline_template_tags(spec.name, cfg=resolved_cfg) if _is_pipeline_template(spec) else list(spec.tags)
    expected_properties = dict(spec.properties_minimal)
    if _is_pipeline_template(spec):
        expected_properties.update(build_pipeline_template_properties(spec.name, cfg=resolved_cfg))
        expected_properties["code_version"] = resolve_version_props(
            resolved_cfg,
            clearml_enabled=True,
        ).get("code_version", "unknown")
    overrides = (
        _normalize_internal_compose_overrides([*entry_args, *spec.default_overrides])
        if _is_pipeline_template(spec)
        else _normalized_args([*entry_args, *spec.default_overrides])
    )
    return ResolvedTemplateSpec(
        spec=spec,
        module=module,
        script=script,
        entry_args=entry_args,
        overrides=overrides,
        expected_requirements=list(spec.requirements),
        expected_tags=expected_tags,
        expected_properties=expected_properties,
        script_spec=script_spec,
        cfg=resolved_cfg,
    )


def _sync_pipeline_seed_identity(task_id: str, *, resolved: ResolvedTemplateSpec) -> None:
    replace_clearml_task_tags(task_id, resolved.expected_tags)
    ensure_clearml_task_tags(task_id, ["pipeline"])
    ensure_clearml_task_properties(task_id, resolved.expected_properties)
    _ensure_pipeline_seed_project(task_id, resolved.spec.project_name)


def _ensure_pipeline_seed_project(task_id: str, project_name: str) -> None:
    set_clearml_task_project(task_id, project_name)
    refreshed = _load_clearml_task(task_id)
    actual_project = clearml_task_project_name(refreshed)
    if str(actual_project or "") != str(project_name):
        raise RuntimeError(
            f"Pipeline seed project drifted after materialization: {actual_project!r} != {project_name!r}"
        )
    _ensure_clearml_project_system_tags(project_name, ["pipeline"], remove_tags=["hidden"])


def _pipeline_controller_step_names(controller: Any) -> tuple[str, ...]:
    nodes = getattr(controller, "_nodes", None) or {}
    return tuple(sorted(str(name) for name in dict(nodes).keys()))


def _expected_pipeline_template_step_names(cfg: Any, spec_name: str) -> tuple[str, ...]:
    plan = _build_pipeline_plan(
        cfg,
        f"seed__{normalize_pipeline_profile(spec_name)}",
        child_execution="logging",
    )
    return tuple(sorted(spec.step_name for spec in build_pipeline_step_specs(plan)))


def _pipeline_seed_drift_reasons(
    task: Any,
    *,
    task_id: str,
    resolved: ResolvedTemplateSpec,
) -> list[str]:
    spec = resolved.spec
    reasons: list[str] = []
    task_type = (clearml_task_type_from_obj(task) or "").lower()
    if "controller" not in task_type:
        reasons.append(f"task type {task_type or 'unknown'}")
    actual_project = clearml_task_project_name(task)
    if str(actual_project or "") != spec.project_name:
        reasons.append(f"project {actual_project!r}")
    status = _task_status_text(task_id)
    if status != "completed":
        reasons.append(f"status {status or 'unknown'}")
    try:
        actual_step_names = _pipeline_controller_step_names(
            load_pipeline_controller_from_task(source_task_id=task_id)
        )
    except Exception as exc:
        reasons.append(f"graph inspect failed: {exc}")
    else:
        expected_step_names = _expected_pipeline_template_step_names(resolved.cfg, spec.name)
        if actual_step_names != expected_step_names:
            reasons.append(
                "graph drift "
                f"{list(actual_step_names)} -> {list(expected_step_names)}"
            )
    actual_tags = set(clearml_task_tags(task))
    missing_tags = [tag for tag in resolved.expected_tags if tag not in actual_tags]
    if missing_tags:
        reasons.append(f"missing tags {missing_tags}")
    actual_properties = _task_user_properties(task)
    missing_properties = {
        key: value
        for key, value in resolved.expected_properties.items()
        if str(actual_properties.get(str(key), "")) != str(value)
    }
    if missing_properties:
        reasons.append(f"property drift {missing_properties}")
    if _pipeline_seed_script_mismatches(resolved.script_spec, clearml_task_script(task)):
        reasons.append("script metadata drift")
    actual_param_keys = _pipeline_parameter_keys(task)
    expected_param_keys = {
        str(key).replace(".", "/")
        for key in build_pipeline_ui_parameter_whitelist(spec.name)
    }
    missing_param_keys = sorted(expected_param_keys - actual_param_keys)
    if missing_param_keys:
        reasons.append(f"missing editable params {missing_param_keys}")
    disallowed_arg_keys = sorted(_pipeline_disallowed_arg_keys(task))
    if disallowed_arg_keys:
        reasons.append(f"stale policy args {disallowed_arg_keys}")
    noncanonical_param_paths = sorted(_pipeline_noncanonical_parameter_paths(task))
    if noncanonical_param_paths:
        reasons.append(f"non-canonical hyperparameters {noncanonical_param_paths}")
    duplicate_visible_params = sorted(_pipeline_duplicate_visible_param_keys(task, expected_param_keys=expected_param_keys))
    if duplicate_visible_params:
        reasons.append(f"duplicate visible params {duplicate_visible_params}")
    expected_runtime_param_keys = _expected_published_pipeline_seed_param_keys(resolved)
    missing_runtime_param_keys = sorted(expected_runtime_param_keys - actual_param_keys)
    if missing_runtime_param_keys:
        reasons.append(f"missing seed runtime params {missing_runtime_param_keys}")
    runtime = _task_runtime(task)
    pipeline_hash = str(runtime.get("_pipeline_hash") or "").strip()
    if not pipeline_hash:
        reasons.append("missing runtime _pipeline_hash")
    elif pipeline_hash.startswith("None:"):
        reasons.append(f"non-canonical runtime _pipeline_hash {pipeline_hash!r}")
    data = getattr(task, "data", None)
    configuration = getattr(data, "configuration", None) if data is not None else None
    pipeline_item = configuration.get("Pipeline") if isinstance(configuration, dict) else None
    if pipeline_item is None:
        reasons.append("missing Configuration/Pipeline")
    else:
        pipeline_description = str(getattr(pipeline_item, "description", "") or "").strip()
        if not pipeline_description.lower().startswith("pipeline state:"):
            reasons.append(f"Pipeline description {pipeline_description!r}")
    artifact_names = _task_artifact_names(task)
    required_artifacts = {
        "pipeline_run.json",
        "run_summary.json",
        "report.json",
        "manifest.json",
        "config_resolved.yaml",
    }
    missing_artifacts = sorted(required_artifacts - artifact_names)
    if missing_artifacts:
        reasons.append(f"missing artifacts {missing_artifacts}")
    try:
        operator_inputs = get_clearml_task_configuration(task_id, name="OperatorInputs")
    except Exception as exc:
        reasons.append(f"OperatorInputs inspect failed: {exc}")
    else:
        if operator_inputs is None:
            reasons.append("missing Configuration/OperatorInputs")
        else:
            actual_operator_keys = set(_configuration_paths(operator_inputs))
            expected_operator_keys = set(build_pipeline_ui_parameter_whitelist(spec.name))
            missing_operator_keys = sorted(expected_operator_keys - actual_operator_keys)
            if missing_operator_keys:
                reasons.append(f"missing OperatorInputs keys {missing_operator_keys}")
    return reasons


def _pipeline_seed_requires_recreate(reasons: Iterable[str]) -> bool:
    recreate_markers = (
        "stale policy args",
        "non-canonical hyperparameters",
        "duplicate visible params",
    )
    return any(any(marker in str(reason) for marker in recreate_markers) for reason in reasons)


def _restore_pipeline_seed_task(
    task_id: str,
    *,
    resolved: ResolvedTemplateSpec,
    seed_definition: Mapping[str, Any],
    args: Iterable[str] | None = None,
    sections: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if args is None:
        seed_defaults = _seed_runtime_defaults(seed_definition)
        args = build_pipeline_visible_hyperparameter_args(
            seed_defaults,
            pipeline_profile=resolved.spec.name,
            include_bootstrap=True,
        )
    if sections is None:
        sections = {}
    replace_clearml_task_hyperparameters(
        task_id,
        args=list(args or []),
        sections=sections,
    )
    replace_clearml_task_tags(task_id, resolved.expected_tags)
    ensure_clearml_task_tags(task_id, ["pipeline"])
    ensure_clearml_task_properties(task_id, resolved.expected_properties)
    ensure_clearml_task_requirements(task_id, resolved.expected_requirements)
    ensure_clearml_task_script(
        task_id,
        repo=resolved.script_spec.repository,
        branch=resolved.script_spec.branch,
        entry_point=resolved.script_spec.entry_point,
        working_dir=resolved.script_spec.working_dir,
        version_num=resolved.script_spec.version_num,
        diff="",
    )
    _sync_pipeline_seed_identity(task_id, resolved=resolved)


def _validate_materialized_seed(
    task_id: str,
    *,
    resolved: ResolvedTemplateSpec,
) -> None:
    refreshed = _load_clearml_task(task_id)
    reasons = _pipeline_seed_drift_reasons(
        refreshed,
        task_id=task_id,
        resolved=resolved,
    )
    if reasons:
        raise RuntimeError(
            f"Pipeline seed {resolved.spec.name} ({task_id}) drifted after materialization: {reasons}"
        )


def _copy_materialized_seed_artifacts(
    source_task_id: str,
    target_task_id: str,
    *,
    cfg: Any,
) -> None:
    source_task = _load_clearml_task(source_task_id)
    for artifact_name in sorted(_task_artifact_names(source_task)):
        local_path = get_task_artifact_local_copy(cfg, source_task_id, artifact_name)
        upload_clearml_task_artifact(target_task_id, artifact_name, local_path)


def _publish_seed_clone(task_id: str) -> None:
    task = _load_clearml_task(task_id)
    for method_name in ("mark_started", "started", "completed", "publish"):
        method = getattr(task, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            continue


def _promote_materialized_pipeline_seed(
    source_task_id: str,
    *,
    resolved: ResolvedTemplateSpec,
    seed_definition: Mapping[str, Any],
) -> str:
    promoted_task_id = clone_clearml_task(
        source_task_id=source_task_id,
        task_name=resolved.spec.task_name_template,
    )
    _restore_pipeline_seed_task(
        promoted_task_id,
        resolved=resolved,
        seed_definition=seed_definition,
        args=_published_pipeline_seed_args(
            resolved,
            defaults=_seed_runtime_defaults(seed_definition),
        ),
        sections=_published_pipeline_seed_sections(
            resolved=resolved,
            seed_definition=seed_definition,
        ),
    )
    set_clearml_task_configuration(
        promoted_task_id,
        dict(seed_definition.get("operator_inputs") or {}),
        name="OperatorInputs",
        description="Editable operator-facing pipeline inputs.",
    )
    _copy_materialized_seed_artifacts(
        source_task_id,
        promoted_task_id,
        cfg=resolved.cfg,
    )
    source_task = _load_clearml_task(source_task_id)
    pipeline_hash = _canonical_seed_pipeline_hash(source_task)
    if pipeline_hash:
        set_clearml_task_runtime_properties(promoted_task_id, {"_pipeline_hash": pipeline_hash})
    _sync_pipeline_seed_identity(promoted_task_id, resolved=resolved)
    _publish_seed_clone(promoted_task_id)
    _validate_materialized_seed(promoted_task_id, resolved=resolved)
    _deprecate_pipeline_task(
        source_task_id,
        actual_project=str(clearml_task_project_name(source_task) or ""),
    )
    return promoted_task_id


def _materialize_pipeline_seed(
    task_id: str,
    *,
    resolved: ResolvedTemplateSpec,
    seed_definition: Mapping[str, Any],
) -> str:
    reset_clearml_task(task_id, force=True)
    _restore_pipeline_seed_task(
        task_id,
        resolved=resolved,
        seed_definition=seed_definition,
        args=_materialized_pipeline_seed_args(
            resolved=resolved,
            seed_definition=seed_definition,
        ),
    )
    set_clearml_task_configuration(
        task_id,
        dict(seed_definition.get("operator_inputs") or {}),
        name="OperatorInputs",
        description="Editable operator-facing pipeline inputs.",
    )
    queue_name = _seed_controller_queue_name(seed_definition)
    print(
        f"Materialize pipeline seed {resolved.spec.name}: "
        f"task_id={task_id} queue={queue_name}"
    )
    enqueue_clearml_task(task_id, queue_name, force=True)
    final_status = _wait_for_terminal_task_status(task_id, timeout_seconds=300.0, poll_seconds=2.0)
    if final_status != "completed":
        raise RuntimeError(
            f"Pipeline seed {resolved.spec.name} ({task_id}) did not complete successfully "
            f"(last_status={final_status!r})."
        )
    refreshed = _load_clearml_task(task_id)
    pipeline_hash = _canonical_seed_pipeline_hash(refreshed)
    if pipeline_hash:
        set_clearml_task_runtime_properties(task_id, {"_pipeline_hash": pipeline_hash})
    _sync_pipeline_seed_identity(task_id, resolved=resolved)
    return _promote_materialized_pipeline_seed(
        task_id,
        resolved=resolved,
        seed_definition=seed_definition,
    )


def _upsert_standard_template(
    resolved: ResolvedTemplateSpec,
    *,
    lock_templates: dict[str, Any],
    now: str,
) -> None:
    spec = resolved.spec
    existing = lock_templates.get(spec.name) if isinstance(lock_templates, dict) else None
    existing_id = existing.get("task_id") if isinstance(existing, dict) else None
    recreate = False
    task_id: str | None = None
    if existing_id:
        try:
            clearml_task_exists(str(existing_id))
            task = _load_clearml_task(str(existing_id))
            actual_project = clearml_task_project_name(task)
            if str(actual_project or "") != spec.project_name:
                recreate = True
                _deprecate_pipeline_task(str(existing_id), actual_project=str(actual_project or ""))
                print(f"Recreate template {spec.name}: project drift {actual_project!r} -> {spec.project_name!r}")
            else:
                task_id = str(existing_id)
        except Exception:
            print(f"Existing task id not found, recreating: {spec.name}")
    if task_id:
        try:
            changes: list[str] = []
            if reset_clearml_task_args(task_id, resolved.overrides):
                changes.append("args")
            if ensure_clearml_task_script(
                task_id,
                repo=resolved.script_spec.repository,
                branch=resolved.script_spec.branch,
                entry_point=resolved.script_spec.entry_point,
                working_dir=resolved.script_spec.working_dir,
                version_num=resolved.script_spec.version_num,
                diff="",
            ):
                changes.append("script")
            if ensure_clearml_task_requirements(task_id, resolved.expected_requirements):
                changes.append("requirements")
            if ensure_clearml_task_tags(task_id, resolved.expected_tags):
                changes.append("tags")
            if ensure_clearml_task_properties(task_id, resolved.expected_properties):
                changes.append("properties")
            if changes:
                print(f"Update template {spec.name}: {', '.join(changes)}")
            else:
                print(f"Reuse template {spec.name}: {task_id}")
        except Exception:
            recreate = True
    if not task_id or recreate:
        task_id = create_clearml_task(
            project_name=spec.project_name,
            task_name=spec.task_name_template,
            module=resolved.module,
            script=resolved.script,
            args=resolved.overrides,
            repo=resolved.script_spec.repository,
            branch=resolved.script_spec.branch,
            tags=resolved.expected_tags,
            properties=resolved.expected_properties,
        )
        ensure_clearml_task_script(
            task_id,
            repo=resolved.script_spec.repository,
            branch=resolved.script_spec.branch,
            entry_point=resolved.script_spec.entry_point,
            working_dir=resolved.script_spec.working_dir,
            version_num=resolved.script_spec.version_num,
            diff="",
        )
        ensure_clearml_task_requirements(task_id, resolved.expected_requirements)
        print(f"Created template {spec.name}: {task_id}")
    lock_templates[spec.name] = _build_lock_template_entry(
        existing=existing if isinstance(existing, dict) else None,
        task_id=task_id,
        project_name=spec.project_name,
        task_name=spec.task_name_template,
        entrypoint=spec.entrypoint,
        now=now,
    )


def _upsert_pipeline_seed(
    resolved: ResolvedTemplateSpec,
    *,
    lock_templates: dict[str, Any],
    now: str,
) -> None:
    spec = resolved.spec
    cfg = resolved.cfg
    if cfg is None:
        raise RuntimeError(f"Resolved config is missing for pipeline seed {spec.name}.")
    arg_pairs = _arg_pairs(
        [
            *resolved.entry_args,
            *_pipeline_seed_bootstrap_overrides(resolved.overrides),
        ]
    )
    existing = lock_templates.get(spec.name) if isinstance(lock_templates, dict) else None
    existing_id = existing.get("task_id") if isinstance(existing, dict) else None
    task_id: str | None = None
    controller = None
    seed_definition: Mapping[str, Any] | None = None
    if existing_id:
        try:
            task = _load_clearml_task(str(existing_id))
            reasons = _pipeline_seed_drift_reasons(task, task_id=str(existing_id), resolved=resolved)
            if not reasons:
                print(f"Reuse pipeline seed {spec.name}: {existing_id}")
                lock_templates[spec.name] = _build_lock_template_entry(
                    existing=existing if isinstance(existing, dict) else None,
                    task_id=str(existing_id),
                    project_name=spec.project_name,
                    task_name=spec.task_name_template,
                    entrypoint=spec.entrypoint,
                    now=now,
                    kind="seed",
                )
                return
            if _pipeline_seed_requires_recreate(reasons):
                actual_project = clearml_task_project_name(task)
                _deprecate_pipeline_task(str(existing_id), actual_project=str(actual_project or ""))
                print(f"Recreate pipeline seed {spec.name}: {', '.join(reasons)}")
            else:
                reset_clearml_task(str(existing_id), force=True)
                controller = load_pipeline_controller_from_task(source_task_id=str(existing_id))
                task_id = str(existing_id)
        except Exception:
            try:
                _deprecate_pipeline_task(str(existing_id))
            except Exception:
                pass
    if controller is None or not task_id:
        controller = create_pipeline_seed_controller(
            project_name=spec.project_name,
            task_name=spec.task_name_template,
            module=resolved.module,
            script=resolved.script,
            args=arg_pairs,
            repo=resolved.script_spec.repository,
            branch=resolved.script_spec.branch,
            commit=resolved.script_spec.version_num,
            working_dir=resolved.script_spec.working_dir,
        )
        task_id = clearml_task_id(controller)
        if not task_id:
            raise RuntimeError(f"Failed to create pipeline seed controller for {spec.name}.")
        print(f"Created pipeline seed {spec.name}: {task_id}")
    else:
        print(f"Sync pipeline seed {spec.name}: {task_id}")
    if seed_definition is None:
        seed_definition = build_pipeline_seed_controller(cfg=cfg, controller=controller, pipeline_profile=spec.name)
    task_id = _materialize_pipeline_seed(task_id, resolved=resolved, seed_definition=seed_definition)
    lock_templates[spec.name] = _build_lock_template_entry(
        existing=existing if isinstance(existing, dict) else None,
        task_id=task_id,
        project_name=spec.project_name,
        task_name=spec.task_name_template,
        entrypoint=spec.entrypoint,
        now=now,
        kind="seed",
    )


def _cleanup_stale_pipeline_tasks(
    *,
    templates: list[TemplateSpec],
    lock_templates: Mapping[str, Any],
) -> None:
    (active_seed_ids, active_seed_projects) = _build_active_pipeline_seed_scope(
        templates=templates,
        lock_templates=lock_templates,
        is_pipeline_template=_is_pipeline_template,
    )
    if not active_seed_projects:
        return
    cleanup_summary = _cleanup_pipeline_history(
        active_seed_ids=active_seed_ids,
        active_seed_projects=active_seed_projects,
        archive_policy=_HistoricalPipelineArchivePolicy(
            archive_deprecated_pipeline_tasks=True,
            archive_noncanonical_pipeline_runs=True,
        ),
    )
    archived_deprecated = int(cleanup_summary.get("archived_deprecated_tasks") or 0)
    archived_runs = int(cleanup_summary.get("archived_noncanonical_runs") or 0)
    if archived_deprecated or archived_runs:
        print(
            "Pipeline history cleanup: "
            f"archived_deprecated_tasks={archived_deprecated} "
            f"archived_noncanonical_runs={archived_runs}"
        )


def _apply_templates(
    templates: list[TemplateSpec],
    *,
    ctx: PlanContext,
    repo_root: Path,
    lock_path: Path,
    repo: str | None,
    branch: str | None,
    version_mode: str | None,
) -> None:
    lock = _load_lock(lock_path)
    lock_templates = lock.get("templates", {})
    lock["version"] = 2
    lock["context"] = _lock_context_payload(ctx)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ordered_templates = [
        *[spec for spec in templates if not _is_pipeline_template(spec)],
        *[spec for spec in templates if _is_pipeline_template(spec)],
    ]

    for spec in ordered_templates:
        resolved = _resolve_template_spec(
            spec,
            ctx=ctx,
            repo_root=repo_root,
            repo=repo,
            branch=branch,
            version_mode=version_mode,
        )
        if _is_pipeline_template(spec):
            _upsert_pipeline_seed(
                resolved,
                lock_templates=lock_templates,
                now=now,
            )
        else:
            _upsert_standard_template(
                resolved,
                lock_templates=lock_templates,
                now=now,
            )
        lock["templates"] = lock_templates
        _save_lock(lock_path, lock)

    lock["templates"] = lock_templates
    _save_lock(lock_path, lock)
    _cleanup_stale_pipeline_tasks(templates=templates, lock_templates=lock_templates)


def _validate_templates(
    templates: list[TemplateSpec],
    *,
    ctx: PlanContext,
    repo_root: Path,
    lock_path: Path,
    repo: str | None,
    branch: str | None,
    version_mode: str | None,
) -> bool:
    if not lock_path.exists():
        print(f"Lock file not found: {lock_path}")
        return False
    lock = _load_lock(lock_path)
    expected_context = _lock_context_payload(ctx)
    actual_context = lock.get("context") or {}
    context_drift = {
        key: expected_context[key]
        for key in expected_context
        if str(actual_context.get(key, "")) != str(expected_context[key])
    }
    if context_drift:
        print(f"Lock context drift: {context_drift}")
        return False
    locked = lock.get("templates", {})
    if not isinstance(locked, dict) or not locked:
        print("No templates in lock file.")
        return False
    errors: list[str] = []
    for spec in templates:
        resolved = _resolve_template_spec(
            spec,
            ctx=ctx,
            repo_root=repo_root,
            repo=repo,
            branch=branch,
            version_mode=version_mode,
        )
        payload = locked.get(spec.name)
        if not isinstance(payload, dict) or not payload.get("task_id"):
            errors.append(f"{spec.name}: missing lock entry")
            continue
        expected_kind = "seed" if _is_pipeline_template(spec) else "template"
        if str(payload.get("kind", expected_kind)) != expected_kind:
            errors.append(f"{spec.name}: lock kind must be {expected_kind!r}")
        task_id = str(payload["task_id"])
        try:
            task = _load_clearml_task(task_id)
        except Exception as exc:
            errors.append(f"{spec.name}: failed to load task {task_id}: {exc}")
            continue
        actual_project = clearml_task_project_name(task)
        if str(actual_project or "") != spec.project_name:
            errors.append(f"{spec.name}: unexpected project {actual_project!r} != {spec.project_name!r}")
        if _task_name(task) and str(_task_name(task)) != spec.task_name_template:
            errors.append(f"{spec.name}: unexpected task name {_task_name(task)!r} != {spec.task_name_template!r}")
        if _is_pipeline_template(spec):
            task_type = (clearml_task_type_from_obj(task) or "").lower()
            if "controller" not in task_type:
                errors.append(f"{spec.name}: task type must be controller, got {task_type or 'unknown'}")
            status = _task_status_text(task_id)
            if status != "completed":
                errors.append(f"{spec.name}: seed pipeline status must be completed, got {status or 'unknown'}")
            project_system_tags = set(_get_clearml_project_system_tags(spec.project_name))
            if "pipeline" not in project_system_tags:
                errors.append(f"{spec.name}: project system tags missing 'pipeline' for {spec.project_name!r}")
            try:
                actual_step_names = _pipeline_controller_step_names(
                    load_pipeline_controller_from_task(source_task_id=task_id)
                )
                expected_step_names = _expected_pipeline_template_step_names(resolved.cfg, spec.name)
                if actual_step_names != expected_step_names:
                    errors.append(
                        f"{spec.name}: pipeline graph drift "
                        f"{list(actual_step_names)} != {list(expected_step_names)}"
                    )
            except Exception as exc:
                errors.append(f"{spec.name}: failed to inspect pipeline graph: {exc}")
            actual_param_keys = _pipeline_parameter_keys(task)
            expected_param_keys = {
                str(key).replace(".", "/")
                for key in build_pipeline_ui_parameter_whitelist(spec.name)
            }
            missing_param_keys = sorted(expected_param_keys - actual_param_keys)
            if missing_param_keys:
                errors.append(
                    f"{spec.name}: missing editable pipeline parameters "
                    f"{missing_param_keys}"
                )
            runtime = _task_runtime(task)
            pipeline_hash = str(runtime.get("_pipeline_hash") or "").strip()
            if not pipeline_hash:
                errors.append(f"{spec.name}: missing runtime _pipeline_hash")
            elif pipeline_hash.startswith("None:"):
                errors.append(
                    f"{spec.name}: seed must be a canonical completed run, got _pipeline_hash={pipeline_hash!r}"
                )
            artifact_names = _task_artifact_names(task)
            required_artifacts = {
                "pipeline_run.json",
                "run_summary.json",
                "report.json",
                "manifest.json",
                "config_resolved.yaml",
            }
            missing_artifacts = sorted(required_artifacts - artifact_names)
            if missing_artifacts:
                errors.append(f"{spec.name}: missing seed artifacts {missing_artifacts}")
            try:
                operator_inputs = get_clearml_task_configuration(task_id, name="OperatorInputs")
            except Exception as exc:
                errors.append(f"{spec.name}: failed to inspect Configuration/OperatorInputs: {exc}")
            else:
                if operator_inputs is None:
                    errors.append(f"{spec.name}: missing Configuration/OperatorInputs")
                else:
                    actual_operator_keys = set(_configuration_paths(operator_inputs))
                    expected_operator_keys = set(build_pipeline_ui_parameter_whitelist(spec.name))
                    missing_operator_keys = sorted(expected_operator_keys - actual_operator_keys)
                    if missing_operator_keys:
                        errors.append(
                            f"{spec.name}: missing OperatorInputs keys {missing_operator_keys}"
                        )
        actual_tags = set(clearml_task_tags(task))
        missing_tags = [tag for tag in resolved.expected_tags if tag not in actual_tags]
        if missing_tags:
            errors.append(f"{spec.name}: missing tags {missing_tags}")
        actual_properties = _task_user_properties(task)
        missing_properties = {
            key: value
            for key, value in resolved.expected_properties.items()
            if str(actual_properties.get(str(key), "")) != str(value)
        }
        if missing_properties:
            errors.append(f"{spec.name}: property drift {missing_properties}")
        if _pipeline_seed_script_mismatches(resolved.script_spec, clearml_task_script(task)):
            errors.append(f"{spec.name}: script metadata drift")
    if errors:
        print("Template validation failed:")
        for item in errors:
            print(f"- {item}")
        return False
    print("OK: all template tasks validated.")
    return True


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(repo_root / "conf" / "clearml" / "templates.yaml"))
    parser.add_argument("--lock", default=str(repo_root / "conf" / "clearml" / "templates.lock.yaml"))
    parser.add_argument("--output-dir", default=str(repo_root / "artifacts"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--usecase-id", default=None)
    parser.add_argument("--schema-version", default=None)
    parser.add_argument("--template-set-id", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--code-ref-mode", default=None)
    args = parser.parse_args()

    if not args.plan and not args.apply and not args.validate:
        parser.error("Select one of --plan, --apply, --validate")

    defaults = _load_run_defaults(repo_root)
    lock_path = Path(args.lock)
    if args.apply or args.validate:
        try:
            ctx = _resolve_live_plan_context(
                defaults=defaults,
                lock_path=lock_path,
                project_root=args.project_root,
                usecase_id=args.usecase_id,
                schema_version=args.schema_version,
                template_set_id=args.template_set_id,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        ctx = PlanContext(
            project_root=str(args.project_root or defaults.project_root),
            usecase_id=str(args.usecase_id or defaults.usecase_id),
            schema_version=str(args.schema_version or defaults.schema_version),
            template_set_id=str(args.template_set_id or defaults.template_set_id),
            solution_root=str(defaults.solution_root),
            pipeline_seed_namespace=str(defaults.pipeline_seed_namespace),
            pipeline_root_group=str(defaults.pipeline_root_group),
            pipeline_runs_group=str(defaults.pipeline_runs_group),
            templates_root_group=str(defaults.templates_root_group),
            step_templates_group=str(defaults.step_templates_group),
            runs_root_group=str(defaults.runs_root_group),
            group_map=dict(defaults.group_map),
        )

    spec_path = Path(args.spec)
    templates = _load_templates(spec_path, ctx)

    if args.plan:
        _plan_output(spec_path, ctx, templates, Path(args.output_dir))
        print("OK: plan written")
        return 0

    if not _clearml_config_present(repo_root):
        print("ClearML config not detected; apply/validate require a live ClearML connection.", file=sys.stderr)
        return 1

    repo_url = args.repo or _detect_repo_url(repo_root)
    branch_name = args.branch or _detect_repo_branch(repo_root)
    version_mode = _load_code_ref_mode(repo_root, args.code_ref_mode)
    print(f"code_ref_mode: {version_mode}")
    if branch_name:
        print(f"code_ref_branch: {branch_name}")
    if args.apply:
        _apply_templates(
            templates,
            ctx=ctx,
            repo_root=repo_root,
            lock_path=lock_path,
            repo=repo_url,
            branch=branch_name,
            version_mode=version_mode,
        )
        return 0

    if args.validate:
        ok = _validate_templates(
            templates,
            ctx=ctx,
            repo_root=repo_root,
            lock_path=lock_path,
            repo=repo_url,
            branch=branch_name,
            version_mode=version_mode,
        )
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
