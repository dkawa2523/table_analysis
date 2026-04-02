#!/usr/bin/env python3
"""ClearML template task management (plan/apply/validate)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tabular_analysis.ops.clearml_identity import resolve_template_context
from tabular_analysis.platform_adapter_clearml_env import resolve_clearml_script_spec
from tabular_analysis.platform_adapter_task import (
    clearml_task_exists,
    create_clearml_task,
    ensure_clearml_task_properties,
    ensure_clearml_task_script,
    ensure_clearml_task_tags,
)


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    project_name: str
    task_name_template: str
    entrypoint: str
    default_overrides: list[str]
    tags: list[str]
    properties_minimal: dict[str, Any]


@dataclass(frozen=True)
class PlanContext:
    project_root: str
    usecase_id: str
    schema_version: str
    template_set_id: str
    solution_root: str
    group_map: dict[str, str]


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
    group_map: dict[str, str] = {}
    if layout_cfg_path.exists():
        layout_cfg = OmegaConf.load(layout_cfg_path)
        solution_root = str(getattr(layout_cfg, "solution_root", None) or solution_root)
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
        return {"version": 1, "templates": {}}
    data = _load_yaml(path)
    if "templates" not in data or not isinstance(data.get("templates"), dict):
        data["templates"] = {}
    if "version" not in data:
        data["version"] = 1
    return data


def _save_lock(path: Path, payload: Mapping[str, Any]) -> None:
    content = OmegaConf.to_yaml(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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


def _apply_templates(
    templates: list[TemplateSpec],
    *,
    lock_path: Path,
    repo: str | None,
    branch: str | None,
    version_mode: str | None,
) -> None:
    lock = _load_lock(lock_path)
    lock_templates = lock.get("templates", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for spec in templates:
        module, script, entry_args = _parse_entrypoint(spec.entrypoint)
        entry_point = f"-m {module}" if module else script
        spec_cfg = {"run": {"clearml": {"code_ref": {"mode": version_mode}}}}
        script_spec = resolve_clearml_script_spec(
            spec_cfg,
            entry_point_override=entry_point,
            repo_override=repo,
            branch_override=branch,
            version_mode_override=version_mode,
            task_name_override=spec.name,
            canonicalize_pipeline=False,
        )

        existing = lock_templates.get(spec.name) if isinstance(lock_templates, dict) else None
        existing_id = None
        if isinstance(existing, dict):
            existing_id = existing.get("task_id")
        if existing_id:
            try:
                clearml_task_exists(str(existing_id))
                changes: list[str] = []
                if ensure_clearml_task_script(
                    str(existing_id),
                    repo=script_spec.repository,
                    branch=script_spec.branch,
                    entry_point=script_spec.entry_point,
                    working_dir=script_spec.working_dir,
                    version_num=script_spec.version_num,
                    diff="",
                ):
                    changes.append("script")
                if ensure_clearml_task_tags(str(existing_id), spec.tags):
                    changes.append("tags")
                if ensure_clearml_task_properties(str(existing_id), spec.properties_minimal):
                    changes.append("properties")
                if changes:
                    print(f"Update template {spec.name}: {', '.join(changes)}")
                else:
                    print(f"Reuse template {spec.name}: {existing_id}")
                continue
            except Exception:
                print(f"Existing task id not found, recreating: {spec.name}")

        overrides = _normalized_args([*entry_args, *spec.default_overrides])
        task_id = create_clearml_task(
            project_name=spec.project_name,
            task_name=spec.task_name_template,
            module=module,
            script=script,
            args=overrides,
            repo=script_spec.repository,
            branch=script_spec.branch,
            tags=spec.tags,
            properties=spec.properties_minimal,
        )
        ensure_clearml_task_script(
            task_id,
            repo=script_spec.repository,
            branch=script_spec.branch,
            entry_point=script_spec.entry_point,
            working_dir=script_spec.working_dir,
            version_num=script_spec.version_num,
            diff="",
        )
        lock_templates[spec.name] = {
            "task_id": task_id,
            "project_name": spec.project_name,
            "task_name": spec.task_name_template,
            "entrypoint": spec.entrypoint,
            "updated_at": now,
        }
        print(f"Created template {spec.name}: {task_id}")

    lock["templates"] = lock_templates
    _save_lock(lock_path, lock)


def _validate_templates(lock_path: Path) -> bool:
    if not lock_path.exists():
        print(f"Lock file not found: {lock_path}")
        return True
    lock = _load_lock(lock_path)
    templates = lock.get("templates", {})
    if not isinstance(templates, dict) or not templates:
        print("No templates in lock file.")
        return True
    missing: list[str] = []
    for name, payload in templates.items():
        if not isinstance(payload, dict):
            missing.append(str(name))
            continue
        task_id = payload.get("task_id")
        if not task_id:
            missing.append(str(name))
            continue
        try:
            clearml_task_exists(str(task_id))
        except Exception:
            missing.append(str(name))
    if missing:
        print("Missing template tasks: " + ", ".join(sorted(missing)))
        return False
    print("OK: all template tasks exist.")
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
    ctx = PlanContext(
        project_root=str(args.project_root or defaults.project_root),
        usecase_id=str(args.usecase_id or defaults.usecase_id),
        schema_version=str(args.schema_version or defaults.schema_version),
        template_set_id=str(args.template_set_id or defaults.template_set_id),
        solution_root=str(defaults.solution_root),
        group_map=dict(defaults.group_map),
    )

    spec_path = Path(args.spec)
    templates = _load_templates(spec_path, ctx)

    if args.plan:
        _plan_output(spec_path, ctx, templates, Path(args.output_dir))
        print("OK: plan written")
        return 0

    if not _clearml_config_present(repo_root):
        print("ClearML config not detected; skipping apply/validate.")
        return 0

    repo_url = args.repo or _detect_repo_url(repo_root)
    version_mode = _load_code_ref_mode(repo_root, args.code_ref_mode)
    print(f"code_ref_mode: {version_mode}")
    if args.apply:
        _apply_templates(
            templates,
            lock_path=Path(args.lock),
            repo=repo_url,
            branch=args.branch,
            version_mode=version_mode,
        )
        return 0

    if args.validate:
        ok = _validate_templates(Path(args.lock))
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
