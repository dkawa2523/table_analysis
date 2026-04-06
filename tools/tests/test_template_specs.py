#!/usr/bin/env python3
"""Spec validation for ClearML template tasks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REQUIRED_TEMPLATES = {
    "dataset_register",
    "preprocess",
    "train_model",
    "train_model_full",
    "train_ensemble_full",
    "infer",
    "leaderboard",
    "pipeline",
}

PIPELINE_TEMPLATE_NAMES = {
    "pipeline",
    "train_model_full",
    "train_ensemble_full",
}

REQUIRED_FIELDS = {
    "project_name",
    "task_name_template",
    "entrypoint",
    "default_overrides",
    "tags",
    "properties_minimal",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return dict(data)


def _load_stage_map(repo: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in (repo / "conf" / "task").glob("*.yaml"):
        payload = _load_yaml(path)
        task_cfg = payload.get("task") if isinstance(payload.get("task"), dict) else None
        task_name = None
        stage = None
        if task_cfg:
            task_name = task_cfg.get("name")
            stage = task_cfg.get("stage")
        if not task_name:
            task_name = path.stem
        if not stage:
            continue
        mapping[str(task_name)] = str(stage)
    return mapping
def _load_group_map(repo: Path) -> dict[str, str]:
    layout_path = repo / "conf" / "clearml" / "project_layout.yaml"
    if not layout_path.exists():
        return {}
    payload = _load_yaml(layout_path)
    group_map = payload.get("group_map")
    if not isinstance(group_map, dict):
        return {}
    return {str(key): str(value) for key, value in group_map.items()}


def _load_layout_tokens(repo: Path) -> dict[str, str]:
    layout_path = repo / "conf" / "clearml" / "project_layout.yaml"
    if not layout_path.exists():
        return {
            "pipeline_seed_namespace": ".pipelines",
            "pipeline_root_group": "Pipelines",
            "templates_root_group": "Templates",
            "step_templates_group": "Steps",
        }
    payload = _load_yaml(layout_path)
    return {
        "pipeline_seed_namespace": str(payload.get("pipeline_seed_namespace") or ".pipelines"),
        "pipeline_root_group": str(payload.get("pipeline_root_group") or "Pipelines"),
        "templates_root_group": str(payload.get("templates_root_group") or "Templates"),
        "step_templates_group": str(payload.get("step_templates_group") or "Steps"),
    }


def _assert_contains(items: list[str], prefix: str) -> None:
    for item in items:
        if item.startswith(prefix):
            return
    raise AssertionError(f"Missing tag prefix: {prefix}")


def _assert_overrides(overrides: list[str]) -> None:
    if "run.clearml.enabled=true" not in overrides:
        raise AssertionError("default_overrides must include run.clearml.enabled=true")
    if not any(item.startswith("run.clearml.execution=") for item in overrides):
        raise AssertionError("default_overrides must include run.clearml.execution=...")
    if "run.clearml.env.uv.all_extras=true" in overrides:
        raise AssertionError("default_overrides must not force run.clearml.env.uv.all_extras=true")


def _validate_spec(repo: Path) -> None:
    spec_path = repo / "conf" / "clearml" / "templates.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file missing: {spec_path}")
    payload = _load_yaml(spec_path)
    templates = payload.get("templates")
    if not isinstance(templates, dict):
        raise AssertionError("templates must be a mapping")

    template_names = set(str(name) for name in templates.keys())
    missing = REQUIRED_TEMPLATES - template_names
    if missing:
        raise AssertionError(f"Missing templates: {sorted(missing)}")

    stage_map = _load_stage_map(repo)
    group_map = _load_group_map(repo)
    layout_tokens = _load_layout_tokens(repo)
    for name, spec in templates.items():
        if not isinstance(spec, dict):
            raise AssertionError(f"template {name} must be a mapping")
        for field in REQUIRED_FIELDS:
            if field not in spec:
                raise AssertionError(f"template {name} missing field: {field}")
        project_name = str(spec.get("project_name"))
        task_name = str(spec.get("task_name_template"))
        entrypoint = str(spec.get("entrypoint"))
        overrides = list(spec.get("default_overrides") or [])
        tags = [str(item) for item in (spec.get("tags") or [])]
        props = spec.get("properties_minimal")

        if not project_name:
            raise AssertionError(f"template {name} project_name empty")
        if not task_name:
            raise AssertionError(f"template {name} task_name_template empty")
        if not entrypoint:
            raise AssertionError(f"template {name} entrypoint empty")
        if not isinstance(props, dict):
            raise AssertionError(f"template {name} properties_minimal must be a mapping")

        if "clearml_entrypoint.py" not in entrypoint or "task=" not in entrypoint:
            raise AssertionError(f"template {name} entrypoint must include clearml_entrypoint.py and task=...")

        _assert_overrides([str(item) for item in overrides])
        _assert_contains(tags, "template_set:")
        _assert_contains(tags, "usecase:")
        expected_process = "pipeline" if name in PIPELINE_TEMPLATE_NAMES else str(name)
        _assert_contains(tags, f"process:{expected_process}")
        _assert_contains(tags, "schema:")
        expected_task_kind = "seed" if name in PIPELINE_TEMPLATE_NAMES else "template"
        _assert_contains(tags, f"task_kind:{expected_task_kind}")
        if name in PIPELINE_TEMPLATE_NAMES:
            _assert_contains(tags, f"pipeline_profile:{name}")
            if "run.clearml.execution=pipeline_controller" not in [str(item) for item in overrides]:
                raise AssertionError(f"pipeline seed {name} must use run.clearml.execution=pipeline_controller")
            if "pipeline.run_dataset_register=false" not in [str(item) for item in overrides]:
                raise AssertionError(f"pipeline seed {name} must pin pipeline.run_dataset_register=false")
            if any(str(item).startswith("+pipeline.model_set=") for item in overrides):
                raise AssertionError(f"pipeline seed {name} must not carry stale +pipeline.model_set overrides")
            if any(
                str(item).startswith(prefix)
                for prefix in ("pipeline.model_variants=", "pipeline.grid.model_variants=")
                for item in overrides
            ):
                raise AssertionError(
                    f"pipeline seed {name} must not expose graph-shaping model variant overrides in spec defaults"
                )

        for key in ("usecase_id", "process", "schema_version", "project_root", "template_set_id", "task_kind"):
            if key not in props:
                raise AssertionError(f"template {name} missing properties_minimal.{key}")
        if name in PIPELINE_TEMPLATE_NAMES:
            if props.get("process") != "pipeline":
                raise AssertionError(f"pipeline seed {name} properties_minimal.process must be pipeline")
            if props.get("task_kind") != "seed":
                raise AssertionError(f"pipeline seed {name} missing properties_minimal.task_kind=seed")
            if props.get("pipeline_profile") != name:
                raise AssertionError(f"pipeline seed {name} missing properties_minimal.pipeline_profile={name}")

        project_key = "pipeline" if name in PIPELINE_TEMPLATE_NAMES else name
        if name in PIPELINE_TEMPLATE_NAMES:
            expected_tokens = [
                "{pipeline_seed_namespace}",
                f"/{layout_tokens['pipeline_seed_namespace'].strip('/')}/",
                f"/{layout_tokens['pipeline_seed_namespace'].strip('/')}/{name}",
            ]
        else:
            stage = stage_map.get(project_key)
            project_group = group_map.get(project_key)
            expected_tokens = [
                token
                for token in (
                    layout_tokens["templates_root_group"],
                    layout_tokens["step_templates_group"],
                    project_group,
                    stage,
                    f"{{group_map[{project_key}]}}",
                    "{templates_root_group}",
                    "{step_templates_group}",
                )
                if token
            ]
        if expected_tokens and not any(token in project_name for token in expected_tokens) and "{group_map[" not in project_name:
            raise AssertionError(f"template {name} project_name should include one of {expected_tokens}")


def _run_plan(repo: Path) -> None:
    cmd = [sys.executable, str(repo / "tools" / "clearml_templates" / "manage_templates.py"), "--plan"]
    proc = subprocess.run(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"plan failed (exit={proc.returncode})\n{proc.stdout}")

    plan_json = repo / "artifacts" / "template_plan.json"
    plan_md = repo / "artifacts" / "template_plan.md"
    if not plan_json.exists():
        raise AssertionError("template_plan.json not created")
    if not plan_md.exists():
        raise AssertionError("template_plan.md not created")

    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    context = payload.get("context") or {}
    if not context.get("template_set_id"):
        raise AssertionError("plan context missing template_set_id")
    template_names = {item.get("name") for item in payload.get("templates", [])}
    missing = REQUIRED_TEMPLATES - template_names
    if missing:
        raise AssertionError(f"plan missing templates: {sorted(missing)}")

    md_text = plan_md.read_text(encoding="utf-8")
    for name in REQUIRED_TEMPLATES:
        if name not in md_text:
            raise AssertionError(f"plan markdown missing {name}")


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    _validate_spec(repo)
    _run_plan(repo)
    print("OK: template specs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
